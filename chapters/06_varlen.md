# 第 6 章　投机解码与置信度驱动的变长验证

## 6.1 投机解码一句话

decode 每步只产出 1 个 token，GPU 算力大量闲置。投机解码让一个便宜的“草稿”模块先猜 k 个 token，主模型一次 forward 验证 1+k 个位置，按接受/拒绝规则保留一个前缀。这个规则保证输出分布与不用投机时相同（`minidi` 在第 14 章实现并用 `test_speculative_decoding_is_lossless` 验证）。

DeepInfer 的 sampler docstring（`model_executor/layers/sampler.py` L60 起）描述了这个流程：验证上一轮的 draft，“逐步比较 draft token 在新旧分布下的 logprob，按 speculative decoding 的接受/拒绝规则决定是否接受”；全部接受时在最后一个 draft 之后正常采样；再用 nextn 模块预测下一轮的 draft。draft 长度由 `DEEPINFER_USE_NEXTN_DECODE` 控制，默认 5（`environ.py` L129）。

## 6.2 固定长度的浪费

固定验证 5 个 draft 时，每个请求每步占 6 个位置。但第 5 个 draft 被接受的前提是前 4 个都被接受，概率往往很低，大部分位置是白算。Q&A 里的说法：DSpark 通过 Confidence Head 预测每个 draft token 的接受率，“调度时把验证算力分配给边际收益最高的 token”。

两个关键量：

- **累乘接受率**。模型给出的是每个位置独立的 confidence，第 j 个 draft 被接受需要前 j 个都接受，所以要累乘。`process_MTPInfo` 里的注释写的就是“modeling返回的confidence rate是每个token独立的，使用前要做一个累乘”（`scheduler.hpp` L1242–L1250）。
- **步耗时**。一步的耗时不是 token 数的线性函数，而是由 CUDA Graph 档位决定的阶梯函数（第 5 章）。

## 6.3 `MTPVarlen::schedule`：贪心找吞吐最大点

decode 调度选完请求后，调 `_mtp_varlen.schedule(scheduled, seqlens_q)` 决定每个请求验证几个 token（`scheduler.hpp` L652）。算法在 `csrc/engine/mtp_varlen.hpp` L150–L267：

1. 没有性能曲线（没开 `DEEPINFER_ENABLE_MTP_VARLEN`）时，每个请求固定 `1 + draft 数`（L153–L158）。
2. 起点：每个请求必验 1 个 token（上一步的输出），`cur_bsz = num_seqs`，期望 token 数 `cur_expected = num_seqs`。
3. 把所有请求所有 draft 的累乘接受率放进一个队列，按接受率降序排序（L188–L214）。
4. 逐个加入：`cur_bsz += 1`，`cur_expected += 接受率`，估计这一步的耗时，吞吐 = `cur_expected / time_ms`，记下吞吐最大的 `best_bsz`（L235–L259）。
5. 把队列前 `best_bsz - num_seqs` 个 token 按所属请求累加到 `seqlens_q`（L263–L266）。

因为累乘接受率在一个请求内单调不增，按降序取前缀时，同一请求的 draft 一定从前往后取，结果天然是合法的前缀。

耗时估计是 `baseline_ms + slope_ms * graph_size + sampler_ms`（L243）：

- `baseline_ms` 来自启动时测的**性能曲线**。`ModelRunner.profile_decode_perf`（`model_runner.py` L562 起）对每个 CUDA Graph 档位 replay 1 次热身、5 次计时，去掉最高最低各 1 个取平均，再在 EP 组里取平均，结果缓存到磁盘。只有开了 `DEEPINFER_ENABLE_MTP_VARLEN` 才测（`worker/base.py` L280–L281）。
- `slope_ms` 修正上下文长度的影响：曲线是在固定输入下测的，真实运行时上下文更长、更慢。`ForwardRuntimeEstimator`（L44–L131）用 `(实测 forward_ms - baseline_ms) / graph_size` 作样本，先按 graph size 分组取中位数，再对各组中位数取中位数（各档等权，每档至少 10 个样本，窗口 5000 步），并截到不小于 0（L226）。
- `sampler_ms`：`SamplerRuntimeEstimator`（L20–L42）用最近 9 步的中位数，至少 5 个样本；优先用同一 `num_seqs` 的历史，不够时退回全局历史（L216–L223）。

实测耗时由 `llm_engine.py` 的 post_process 回调 `update_mtp_runtime` 喂回来（L997 附近）。

![图 6-1　变长验证的贪心过程（合成的 8 个请求、阶梯形性能曲线，示意用）](images/plots/ch06_varlen.png)

图 6-1 用 `minidi/varlen.py`（逐行对照上面的 C++）画出：横轴是本步验证 token 总数，纵轴是期望吞吐。曲线在 CUDA Graph 档位的边界处跳变，最大点往往落在某个档位的右边界上。示例数据和曲线都是合成的，不代表真实硬件。

Q&A 给出的线上效果：每步验证的 draft token 数“砍掉约 30%（5.00 → 3.2–4.4）”，平均接受长度“基本不变（3.29 vs 3.20）”，在每 rank 约 60 并发的生产运行点“折合 +4~7% 吞吐”。这些数字来自官方文档，本书无法复现。

## 6.4 两轮接受率：配合异步调度

开了 ZOS 后，调度第 N+1 步时第 N 步的 confidence 还在 GPU 上。`schedule` 开头的长注释（L160–L169）说明了处理办法：

- 用 **N-2 轮**的接受率算总预算 `best_bsz`（CPU 上一定有）；
- 用 **N-1 轮**的接受率分配每个请求的 `seqlens_q`：不开 ZOS 时 CPU 已经有了；开 ZOS 时 CPU 只确定总预算，逐请求的分配在 GPU 上重算。

每个请求的 `mtp.schedule_cnt` 和 `mtp.process_cnt`（`sequence.hpp` L129–L131）区分这两种情况：相等时 `accept_rates` 是 N-1 轮、`prev_accept_rates` 是 N-2 轮；`schedule_cnt == process_cnt + 1` 时两者各往前推一轮。L195–L196 断言只会是这两种情况之一，这就是第 5 章“在途抢占时 `process_MTPInfo` 也要跳过”的原因。

GPU 侧的重算是 tilelang 算子 `compute_seqlens_q_by_confidence_rates`（`tilelang_ops/schedule_mtp_varlen.py`），在 `_correct_seqlens_qk` 里调用（`worker.py` L218–L219）。它的 docstring 说明融合了 remap scatter：新请求的接受率填 0，命中的请求取上一步的 confidence。Q&A 的描述是“累乘、全局排序取 top-k，并融合 remap scatter，结果回写 pinned buffer”。

```{.mermaid caption="图 6-2　开 ZOS 时变长验证的分工"}
flowchart LR
  A["CPU schedule(N+1)"] -->|"N-2 轮接受率"| B["MTPVarlen::schedule<br/>算 best_bsz（总预算）"]
  B --> C["选 CUDA Graph 档位<br/>补 padding"]
  D["GPU 上 sample(N) 的 confidence"] --> E["compute_seqlens_q_by_confidence_rates<br/>累乘 + 全局 top-k + remap"]
  C --> E
  E --> F["forward(N+1) 的 seqlens_q"]
  E -->|"D2H 拷回"| G["CPU 元数据修正<br/>（execute_model_step 第 3 阶段）"]
```

## 6.5 代价：可复现性

变长验证让每个请求本步验证几个 token 依赖于同批其他请求的接受率。README 写明，要获得确定性采样，必须同时设 `DEEPINFER_BATCH_INVARIANT=1 DEEPINFER_ENABLE_MTP_VARLEN=0` 并给 `SamplingParams` 传固定的 `seq_seed`（仅 NVIDIA）。这一点在第 8 章展开。

## 6.6 草稿模块

DeepInfer 的 draft 来自模型自带的 nextn（MTP）模块，代码里还有 DFlash 草稿模型（`model_executor/models/mtp/dflash.py`，有独立的 CUDA Graph runner）。sampler docstring 提到“DFlash 的 main KV cache 已由主模型 forward 更新，无额外 draft prefill”。本书没有深入草稿模型的结构，第 7 章只讲与系统相关的部分。
