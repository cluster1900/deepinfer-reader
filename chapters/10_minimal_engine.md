# 第 10 章　最小引擎：玩具模型、扁平变长批次与 `step` 循环

万丈高楼平地起。要验证一个推理引擎的 KV Cache 和调度器写得对不对，我们首先得有一个能算注意力、能吐出 logits 的语言模型，以及一个最精简的 `step()` 主循环。

这一章我们一起走读 `minidi/model.py` 和 `minidi/engine.py`，完成三件事：
1. 写一个只有 40 行代码的单层因果注意力模型 `TinyLM`，作为全书的“黄金标尺”；
2. 把传统的三维补齐批次（Padded Batch）改造成生产引擎使用的**扁平变长批次（Flat Varlen Batch）**；
3. 搭起 `Engine.step()` 的五步骨架，并实现**与批次无关的确定性采样**。

---

## 10.1 黄金标尺：为什么 `TinyLM` 只需要 40 行代码？

很多同学会问：既然是测推理引擎，为什么不用随机数直接假装成模型输出的 logits，而要真写一个带 Q、K、V 投影和因果掩码（Causal Mask）的注意力模型？

因为**假 logits 抓不到 KV Cache 寻址的 Bug**！如果模型输出不依赖历史上下文，哪怕你把请求 A 的 KV 块指到了请求 B 的头上，或者把第 5 个位置写到了第 6 个槽位，假模型照样能跑通。而在真正的因果注意力模型里，**当前位置的每一个浮点数都严格依赖从位置 $0$ 到当前位置的全部历史 K 和 V**——只要块表寻址或缓存计数错了一格，算出来的 logits 就会立刻失真，被单元测试当场逮住。

打开 `minidi/model.py`（L14–L46），我们的 `TinyLM` 麻雀虽小、五脏俱全：

```python
class TinyLM:
    def __init__(self, vocab_size=64, dim=16, max_len=256, seed=0, noise=0.0):
        rng = np.random.default_rng(seed)
        self.vocab_size, self.dim = vocab_size, dim
        # 词嵌入表 + 正弦位置编码 + 单头注意力权重 (Wq, Wk, Wv, Wo)
        self.emb = rng.normal(0, 0.5, (vocab_size, dim))
        pos = np.arange(max_len)[:, None] / (10000 ** (np.arange(dim)[None, :] / dim))
        self.pos = np.where(np.arange(dim)[None, :] % 2 == 0, np.sin(pos), np.cos(pos))
        self.wq, self.wk, self.wv, self.wo = [rng.normal(0, 0.3, (dim, dim)) for _ in range(4)]

    def reference_logits(self, tokens: list[int]) -> np.ndarray:
        """黄金参考实现：不使用任何 KV Cache，每次从头计算整段序列 [0..n-1] 的因果注意力。"""
        n = len(tokens)
        h = self.emb[tokens] + self.pos[:n]                  # [n, dim]
        q, k, v = h @ self.wq, h @ self.wk, h @ self.wv      # [n, dim]
        att = (q @ k.T) / np.sqrt(self.dim)                  # [n, n]
        mask = np.tril(np.ones((n, n), dtype=bool))          # 下三角因果掩码：位置 i 只能看 0..i
        att = np.where(mask, att, -np.inf)
        att = np.exp(att - att.max(axis=-1, keepdims=True))
        att /= att.sum(axis=-1, keepdims=True)
        return (h + att @ v) @ self.wo                       # [n, vocab_size]
```

看 `reference_logits(tokens)`：它没有任何状态，只要喂给它 `[t_0, t_1, ..., t_{n-1}]`，它就老老实实算出 `n` 行 logits，最后一行 `logits[-1]` 就是预测下一个 token 的概率分布。这就是我们检验一切复杂优化的**绝对真理**。

---

## 10.2 告别 Padding：生产引擎是怎么用“扁平变长批次”算注意力的？

在入门教程里，同时跑多条请求通常会把短句子补零（Padding）到和长句子一样长，拼成一个 `[batch_size, max_seq_len, dim]` 的三维张量。但在真实推理引擎里，请求 A 可能在算长度为 15 的 prefill chunk，请求 B 在算长度为 1 的 decode token，请求 C 在验证长度为 4 的投机草稿——如果补零，90% 的算力都会浪费在算 Padding 上！

因此，无论是 DeepInfer（`InputMetadata`）还是 FlashMLA，都采用**扁平变长批次（Flat Varlen Batch）**：把所有请求在本步需要新算的 query token 首尾相接，拍平成一个二维张量 `[total_q, dim]`，同时为每条请求附带三样寻址信息：

1. `query_tokens`（长度 `q = seqlen_q`）：本步新送进模型的 token 列表；
2. `start_pos`：这批新 token 的第 0 个位置在原序列里对应的绝对下标（即此前已写入 KV Cache 的长度 `num_cached`）；
3. `block_table`：该请求占用的物理块号列表。

在 `TinyLM.paged_forward(kv, batch)`（`minidi/model.py` L69–L93）中，无论是 Prefill、分块 Prefill、普通 Decode 还是投机验证，全部走同一套五步流程：

```python
def paged_forward(self, kv: PagedKV, batch: list[tuple[list[int], int, list[int]]]) -> np.ndarray:
    out = []
    for query_tokens, start, block_table in batch:
        n = len(query_tokens)                                 # 本步新增长度 q = seqlen_q
        positions = np.arange(start, start + n)               # 绝对位置 start .. start+n-1
        h = self.emb[query_tokens] + self.pos[positions]      # 1. 查词表并加上对应位置编码
        q, k_new, v_new = h @ self.wq, h @ self.wk, h @ self.wv
        kv.write(block_table, start, k_new, v_new)            # 2. 先把本步新算的 K/V 写进物理块！
        k_all, v_all = kv.read(block_table, start + n)        # 3. 读出 0 .. start+n-1 的全部历史 K/V
        att = (q @ k_all.T) / np.sqrt(self.dim)               # 4. 形状 [q, seqlen_k] 的矩形注意力
        # 5. 矩形因果掩码：第 i 个 query（绝对位置 start+i）只能看 0 .. start+i 的 key
        mask = np.arange(start + n)[None, :] <= (start + np.arange(n))[:, None]
        att = np.where(mask, att, -np.inf)
        ...
        out.append((h + att @ v_all) @ self.wo)
    return np.concatenate(out, axis=0)                        # 返回扁平拼接的 [total_q, vocab_size]
```

请特别留意第 5 步那个矩形因果掩码 `np.arange(start + n)[None, :] <= (start + np.arange(n))[:, None]`：
- 当跑 **普通 Decode**（`n = 1`）时，唯一的那 1 个 query 在绝对位置 `start`，它能看到 `0 .. start` 的全部历史 K/V；
- 当跑 **分块 Prefill**（比如第 2 个 chunk，`start = 16, n = 14`）或 **投机解码验证**（`n = 1 + nextn = 4`）时，第 $i$ 个 query（绝对位置 `start + i`）既能看到前面所有的历史缓存 `0 .. start - 1`，又能看到本步排在它前面的同伴 `start .. start + i`，但看不到排在它后面的同伴 `start + i + 1 .. start + n - 1`！

这就是为什么同一个 `paged_forward` 函数能通吃全书所有阶段。

---

## 10.3 `Engine.step()` 的五步主循环

有了 `TinyLM.paged_forward`，我们就可以在 `minidi/engine.py`（L53–L92）里写出与第 2 章 `LLMEngine.step()` 完全同构的五步循环：

```{.mermaid caption="图 10-1　minidi 同步引擎 Engine.step() 的五步闭环（minidi/engine.py L53–L92）"}
flowchart TB
  subgraph Row1["第一阶段：调度拍板、组装扁平批次与分页前向"]
    direction LR
    S1["1. scheduler.schedule()<br/>拍板 Prefill 或 Decode<br/>选出请求与 seqlens_q"] --> S2["2. 组装扁平批次 batch<br/>(query_tokens, start, block_table)"] --> F1["3. model.paged_forward()<br/>先写新 K/V 入块，再读历史算注意力"]
  end

  subgraph Row2["第二阶段：按位置确定性采样与两阶段账本结算"]
    direction LR
    F2["4. 按 (seed, pos) 采样：<br/>Prefill 末块采最后一行<br/>Decode 逐行验证草稿"] --> P1["5. process_outputs()：<br/>乐观推进 → 追加词判停 → 回填 num_cached = len - 1"] --> P2["流转去向：<br/>未完回 waiting / 生成留 running / 结束释放块"]
  end

  Row1 --> Row2
```

看着图 10-1 的第 2 步和第 4 步，把两个最容易写错的边界记牢：

- **组装批次时，`start_pos` 永远等于 `seq.num_cached`**：
  在 Prefill 步里，因为 `seq.tokens` 装的是完整的 prompt，所以我们切出 `seq.tokens[seq.num_cached : seq.num_cached + q]` 送进模型；而在 Decode 步里，因为第 2 章讲过的不变量 `seq.num_cached == seq.get_len() - 1`，所以 `seq.tokens[-1]` 正好就是上一步刚采出来、位于下标 `num_cached` 处、还没进过模型的那个新 token！
- **分块 Prefill 的中间 chunk 绝不采样**：
  如果一个长度为 30 的 prompt 被切成了 `16 + 14` 两个 chunk，跑完第一个 chunk（`q = 16`）时，`seq.num_cached + q`（16）还小于 `seq.get_len()`（30）。此时模型算出的 16 行 logits 全部丢弃，给 `process_outputs` 传一个空列表 `[]`，只为了把前 16 个位置的 K/V 写进物理块；直到第二个 chunk 把剩下的 14 个位置跑完（`16 + 14 >= 30`），才取最后一行 logits 采样出第 1 个输出 token！

---

## 10.4 怎么用 5 行代码实现“批不变确定性采样”？

在第 8 章我们看到，DeepInfer 为了让随机采样结果不受同批其他请求干扰，专门给每条请求配了 `seq_seed` 并按采样位置派生随机数。在 `minidi/engine.py`（L18–L26）中，我们只用 5 行 NumPy 代码就实现了完全相同的数学性质：

```python
def sample(logits_row: np.ndarray, seq: Sequence, position: int) -> int:
    if seq.temperature <= 0:
        return int(np.argmax(logits_row))
    # 核心：随机数发生器仅由 (请求的独立 seed, 当前生成位置的绝对下标 position) 唯一决定！
    rng = np.random.default_rng([seq.seed, position])
    probs = np.exp((logits_row - logits_row.max()) / seq.temperature)
    probs /= probs.sum()
    return int(rng.choice(len(probs), p=probs))
```

为什么传入 `[seq.seed, position]` 就能保证批不变？
因为对于请求 $A$ 的第 $p$ 个位置来说，不管这一步批次里只有它自己，还是有另外 10 条请求排在它前面，也不管它中间有没有被抢占重算过，当它生成位置 $p$ 的 token 时，`[seq.seed, position]` 这对数字永远一模一样，抽出的随机数也永远一模一样！

在 `tests/test_engine.py::test_seeded_sampling_is_batch_invariant` 中，我们让 6 条开启随机采样（`temperature = 0.8`）的请求先后在两种极端配置下运行：
- 第一次：大块池、大批次（`max_num_seqs = 8, max_num_batched_tokens = 64`），正序入队；
- 第二次：小块池、极小批次（`max_num_seqs = 2, max_num_batched_tokens = 8`），逆序入队，中途频繁分块和抢占。

测试断言两组运行还原回原顺序后的输出**逐 token 100% 相等**。搭好了这个坚实的最小闭环，下一章我们就来看看 `PagedKV` 和 `BlockManager` 是怎么把离散的物理块变成连续的逻辑序列的。
