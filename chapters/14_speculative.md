# 第 14 章　手写投机解码与贪心变长验证（`nextn` & `varlen.py`）

这一章我们把第 6 章拆解的两大投机解码核心机制——**投机草稿生成与单步并行验证回退**（`minidi/engine.py`）以及 **基于累乘置信度与阶梯耗时的贪心变长验证算法**（`minidi/varlen.py`）——在 CPU 上亲手实现出来，并用数学证明和暴力枚举测试回答一个关键问题：**为什么按累乘接受率全局排序贪心取前缀，一定能找到全局最优解？**

---

## 14.1 25 行实现投机草稿生成与单步验证

为了在 `minidi` 中模拟不同质量的草稿模型（类似 DeepInfer 的 DFlash 模块），我们在 `TinyLM` 中加入了一个可控噪声参数 `noise`。当 `noise = 0.0` 时，草稿模型与主模型完全一致（草稿命中率 100%）；当 `noise > 0` 时，草稿模型的权重叠加了高斯扰动，噪声越大，猜错的概率越高。

在 Decode 步开始前，`Engine._propose(seq)`（`minidi/engine.py` L48–L51）让草稿模型往后快速猜出 `nextn` 个候选草稿 token 存入 `seq.draft_tokens`。随后在 `Engine.step()` 中，主模型把 **`[seq.tokens[-1]] + seq.draft_tokens`**（共 `q = 1 + nextn` 个 token）拼在一起，通过一次 `paged_forward` 并行算出 `q` 行 logits！

拿到这 `q` 行 logits 后，主模型怎么决定接受几个草稿？看 `minidi/engine.py` L70–L80 的验证循环：

```python
# rows 的形状是 [q, vocab_size]，其中 q = 1 + len(seq.draft_tokens)
accepted = []
for j in range(q):
    # 第 j 行 logits 是在看完了主输入 token 以及前 j 个草稿 token 后，主模型对位置 start+j+1 的真实预测！
    t = sample(rows[j], seq, start + j + 1)
    accepted.append(t)
    # 如果当前是前 q-1 行（正在核对第 j 个草稿），且主模型采出的 t 与草稿猜的 draft_tokens[j] 不一致：
    if j < q - 1 and t != seq.draft_tokens[j]:
        break                                   # 当场拒绝后续所有草稿！但保留主模型刚刚采出的修正词 t！
```

我们用 $\text{nextn} = 3$、第 2 个草稿猜错的具体场景，把上面这段 8 行代码的输入输出画出来：

```{.mermaid caption="图 14-1　单步投机验证与自动赠送修正 Token 的过程（以 nextn = 3、第 2 个草稿被拒为例）"}
flowchart TB
  subgraph Input["1. 单步并行送入主模型的 4 个位置（start .. start+3）"]
    direction LR
    I0["位置 start：主输入 t<br/>（上一步已确认的末尾词）"] --> I1["位置 start+1：草稿 d₁"] --> I2["位置 start+2：草稿 d₂"] --> I3["位置 start+3：草稿 d₃"]
  end

  subgraph Verify["2. 主模型一次 paged_forward 产出 4 行 Logits 并逐行核对"]
    direction LR
    R0["第 0 行 Logits（看完 t）<br/>主模型采出 d₁ == 草稿 d₁ ✔<br/>（第 1 个草稿通过！）"] --> R1["第 1 行 Logits（看完 t, d₁）<br/>主模型采出 x ≠ 草稿 d₂ ✘<br/>（第 2 个草稿被拒，直接采纳修正词 x 并 break！）"] --> R2["第 2、3 行 Logits（看完 d₂, d₃）<br/>因前提 d₂ 已错，全部丢弃！"]
  end

  subgraph Commit["3. process_outputs 落账与 KV 自动回退"]
    direction LR
    C1["本步最终追加 2 个合法 Token：[d₁, x]<br/>（哪怕 0 个草稿通过，也保底产出 1 个主采样词）"] --> C2["回填 num_cached = get_len() - 1：<br/>d₂、d₃ 写过的 KV 槽位直接标记作废，下一步自动覆盖！"]
  end

  Input --> Verify --> Commit
```

看着图 14-1，你会发现投机解码两个最迷人的性质：

1. **进可攻、退可守（保底 1 个，至多 $1 + \text{nextn}$ 个）**：
   - 就算第 1 个草稿 $d_1$ 就猜错了，第 0 行 logits 本来就是主模型看完真实前缀 $t$ 后算出来的分布，从第 0 行采出的修正词 $x$ 就是 100% 合法的下一个 token！
   - 当第 1 个草稿 $d_1$ 猜对了，第 1 行 logits 就变成了主模型看完真实前缀 $(t, d_1)$ 后算出的分布！此时哪怕第 2 个草稿 $d_2$ 猜错了，从第 1 行采出的修正词 $x$ 依然是白赚的第 2 个合法 token！
   - 如果全部 3 个草稿 $d_1, d_2, d_3$ 全猜对了，第 3 行 logits（看完 $t, d_1, d_2, d_3$）还能再额外送你 1 个新 token，一步总共吐出 $3 + 1 = 4$ 个 token！
2. **分页 KV 下的回退是“零成本”的**：
   在刚才的 `paged_forward` 里，$t, d_1, d_2, d_3$ 一共 4 个位置的 K 和 V 都已经写进了物理块。现在 $d_2, d_3$ 被拒绝了，需要去物理块里把它们擦掉吗？**完全不需要！** 因为 `process_outputs` 最后那行 `seq.num_cached = seq.get_len() - 1` 把已缓存指针精准地拨回到了有效位置（只前进了 2 格而不是 4 格），下一步 Decode 自然会从新的 `num_cached` 开始把废数据直接覆盖掉；如果 $d_2, d_3$ 当初跨块多占了一个新物理块，下一步 `_schedule_decode` 的 `elif need < 0:` 分支还会把那个多余的块自动退还给块管理器！

在 `tests/test_engine.py::test_speculative_decoding_is_lossless` 中，我们分别用 `noise = 0.0`（全中）、`0.2`（高命中）和 `0.6`（高频拒绝）跑完整生成，断言所有输出与不用投机解码的 `reference_generate` 100% 逐字相等。

---

## 14.2 35 行实现 `MTPVarlen` 贪心变长验证，并证明其全局最优性

接下来我们看 `minidi/varlen.py`（共 56 行）。它逐行复刻了第 6 章讲过的 C++ 函数 `MTPVarlen::schedule`（`csrc/engine/mtp_varlen.hpp` L150–L267）：

```python
def cumulative(conf: list[float]) -> list[float]:
    """对应 scheduler.hpp L1242-L1250：将每个草稿位置的独立置信度做前缀累乘。"""
    out, p = [], 1.0
    for c in conf:
        p *= c
        out.append(p)
    return out

def schedule_varlen(acc_rates: list[list[float]], curve: list[tuple[int, float]],
                    slope_ms: float = 0.0, sampler_ms: float = 0.0):
    """对应 mtp_varlen.hpp L150-L267：贪心寻找令 期望产出/步耗时 最大的总预算 best_bsz。"""
    n = len(acc_rates)
    # 将所有请求的所有候选草稿按 (累乘接受率降序, 请求下标升序) 排成一队
    queue = sorted(((r, i) for i, rs in enumerate(acc_rates) for r in rs), key=lambda x: (-x[0], x[1]))
    best_tp, best_bsz = 0.0, 0
    expected, bsz = float(n), n               # 起点：每条请求必验 1 个主输入 token

    def update():
        nonlocal best_tp, best_bsz
        t = step_time(bsz, curve, slope_ms, sampler_ms)
        if t is not None and expected / t > best_tp:
            best_tp, best_bsz = expected / t, bsz

    update()
    for r, _ in queue:                        # 每次往锅里加 1 个边际收益最高的草稿
        expected += r
        bsz += 1
        update()

    seqlens_q = [1] * n
    for _, i in queue[: best_bsz - n]:
        seqlens_q[i] += 1
    return seqlens_q, best_bsz, best_tp
```

### 为什么这个 $O(K \log K)$ 的简单贪心算法一定能找到全局最优解？

假设批次里有 $B$ 条请求，每条请求最多选前 $k_i \in \{0, 1, \dots, 5\}$ 个草稿，组合总数高达 $6^B$ 种（当 $B=64$ 时是 $6^{64}$ 种天文数字）。为什么 `schedule_varlen` 不需要做复杂的动态规划，只需排个序扫一遍就能拿到全局最优？

因为这个问题同时满足两条漂亮的数学性质：
1. **分母只依赖总个数**：单步耗时 `step_time(bsz)` 只取决于本步验证的 token 总数 $\text{bsz} = B + \sum_{i=1}^B k_i$，与这些 token 分给了哪条请求无关。因此，只要对于每一个固定的总个数 $\text{bsz}$，我们能挑出使分子（期望总产出）最大的那组 $\sum k_i = \text{bsz} - B$ 个草稿，然后再对所有可能的 $\text{bsz}$ 取最大值，就一定能找到全局最优！
2. **分子单调性保证“全局最大子集”必然是“合法的各请求前缀”**：因为每条请求内部的累乘接受率满足 $p_{i, 1} \ge p_{i, 2} \ge \dots \ge p_{i, 5} \ge 0$（连乘的因子都在 $[0, 1]$ 之间），当我们从全局所有候选草稿里挑出数值最大的前 $M = \text{bsz} - B$ 个元素时，如果请求 $i$ 的第 $j$ 个草稿入选了，那么排在它前面、数值不小于它的第 $1 \dots j-1$ 个草稿**必定也已经入选**！

为了用代码铁证这一数学结论，我们在 `tests/test_algorithms.py::test_varlen_greedy_is_optimal_over_prefixes` 中专门写了一个暴力枚举器（遍历 4 条请求全部 $5^4 = 625$ 种合法前缀组合），验证 `schedule_varlen` 算出的 `best_tp` 与暴力穷举的全局最大值分毫不差！
