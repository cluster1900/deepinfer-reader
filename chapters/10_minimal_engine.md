# 第 10 章　最小引擎：step 循环

## 10.1 玩具模型

`TinyLM`（`minidi/model.py`）是单层单头注意力：embedding + 正弦位置编码，Q/K/V 投影，因果注意力，残差，输出投影。权重固定种子随机初始化。它不需要“会说话”，只需要满足一个性质：**下一个 token 依赖完整的前缀**。这样 KV 缓存哪怕错一个位置，输出都会变，测试才能抓到错误。

参考实现每次从头算：

```python
def reference_logits(self, tokens):
    n = len(tokens)
    h = self.emb[tokens] + self.pos[:n]
    q, k, v = h @ self.wq, h @ self.wk, h @ self.wv
    att = q @ k.T / np.sqrt(self.dim)
    att = np.where(np.tril(np.ones((n, n), bool)), att, -np.inf)
    ...
    return (h + att @ v) @ self.wo
```

## 10.2 扁平变长批次

真实引擎不会把请求补齐到相同长度再算，而是把所有请求本步要算的 token 拼成一个长序列，再用每个请求的 `seqlen_q`（本步算几个）、`seqlen_k`（能看到多长）和 `block_table`（KV 在哪些块）描述边界。DeepInfer 的 `InputMetadata` 就是这样组织的（第 2 章）。

`paged_forward(kv, batch)` 的 `batch` 每项是 `(query_tokens, start_pos, block_table)`：

1. 计算这几个 token 的 Q/K/V；
2. **先写 KV**：把 K/V 写入 `block_table` 指向的块（对应 DeepInfer 的 `store_kv_cache`）；
3. **再读 KV**：按 `block_table` 读回 `[0, start+n)` 的全部 K/V，做因果注意力。

返回所有请求拼在一起的 logits。prefill 一个 chunk 和 decode 一个 token 用的是同一个函数，区别只是 `query_tokens` 的长度。

## 10.3 step 循环

`Engine.step`（`minidi/engine.py`）是 DeepInfer `LLMEngine.step` 的同步版本：

```python
def step(self):
    out = self.scheduler.schedule()                  # 1. 调度：本步算谁、各算几个 token
    batch = [...]                                    # 2. 组扁平批次
    logits = self.model.paged_forward(self.kv, batch)  # 3. forward
    new_tokens = [...]                               # 4. 采样（prefill 未完成的 chunk 不采样）
    self.scheduler.process_outputs(out, new_tokens)  # 5. 落账：推进长度、判停、入 running
```

`generate` 不断调 `step` 直到没有未完成请求，最后释放所有块。

```{.mermaid caption="图 10-1　minidi 同步引擎的一步（engine.py Engine.step）"}
flowchart LR
  W["waiting 队列"] --> S["Scheduler.schedule<br/>prefill 步或 decode 步"]
  R["running 列表"] --> S
  S --> B["组扁平批次<br/>(tokens, start, block_table)"]
  B --> F["TinyLM.paged_forward<br/>先写 KV 再读 KV"]
  F --> P["sample<br/>(seed, position) 定随机数"]
  P --> O["process_outputs<br/>推进长度、判停"]
  O -->|"prefill 完成"| R
  O -->|"chunk 未完"| W
  O -->|"结束"| X["释放块"]
```

注意第 4 步：prefill 只有**最后一个 chunk** 才采样。一个 30 token 的 prompt 被切成 16 + 14 两个 chunk 时，第一个 chunk 的 logits 直接丢弃。

## 10.4 采样与批不变

```python
def sample(logits_row, seq, position):
    if seq.temperature <= 0:
        return int(np.argmax(logits_row))
    rng = np.random.default_rng([seq.seed, position])
    ...
```

随机采样的随机数由 `(seed, position)` 决定，而不是一个全局随机数生成器。如果用全局生成器，请求 A 拿到第几个随机数取决于本步批里排在它前面的有几个请求，输出就会随组批方式变化。DeepInfer 用每请求的 `seq_seed` 解决同一个问题（第 8 章）。

`test_seeded_sampling_is_batch_invariant` 让 6 个请求用两种完全不同的组批方式各跑一次（一次大批、token 预算 64；一次每批最多 2 个、预算 8，而且请求顺序倒过来），断言每个请求的输出一致。CPU 上 NumPy 的矩阵乘对单个请求是确定的，所以这里只需要处理随机数；GPU 上还要处理 kernel 的归约顺序，那是 `DEEPINFER_BATCH_INVARIANT` 做的事。

## 10.5 第一个测试

```python
@pytest.mark.parametrize("seed", range(12))
def test_engine_matches_reference_with_preemption(seed):
    ...
    assert Engine(M, **kw).generate(ps, max_tokens=n) == ref(ps, n)
```

12 组随机的块数（14–39）、EOS 和 prompt，都要求输出与参考逐 token 相等，并且结束后除块 0 外所有块都回到空闲链表。块数偏小的配置会触发抢占，`test_preemption_actually_happens` 确认这 12 组里确实发生过抢占，避免“测试通过只是因为没走到那条路径”。这条测试是后面所有改动的安全网。
