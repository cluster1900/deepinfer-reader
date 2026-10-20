# 第 6 章　投机解码 DFlash 与置信度驱动的变长验证

在 Decode 阶段，如果每跑一步沉重的主模型前向只生成 1 个 token，GPU 的算术逻辑单元（ALU）绝大部分时间都在等显存带宽把庞大的模型权重搬进寄存器，算力利用率极低。

**投机解码（Speculative Decoding）** 的破局思路众所周知：先用一个轻量级的“草稿模块（Draft Model）”一口气猜出后面 $\text{nextn} = 5$ 个候选草稿 token，然后让主模型在一步前向里同时验证 `1 + 5 = 6` 个位置。只要验证通过几个，一步就能吐出好几个 token，而且通过标准的接受-拒绝采样（Rejection Sampling），最终生成的文本分布与逐个解码**在数学上 100% 无损等价**。

但在超大规模生产落地中，传统投机解码有两个痛点：

1. **草稿模型自己慢怎么办？** 如果草稿模型要串行自回归跑 5 步才能猜出 5 个草稿 token，草稿本身的耗时就会吃掉一大半收益。
2. **固定验证 5 个草稿太浪费怎么办？** 第 5 个草稿被接受的前提是前 4 个草稿全部猜对。如果某条请求在第 2 个草稿就已经拿不准了（置信度极低），硬逼着主模型为它验证满 5 个草稿，只会白白撑大本步的总 token 数 `total_q`，把 CUDA Graph 推向更慢的档位！

DeepInfer 用 **DFlash（官方 Q&A 亦称 DSpark）并行草稿架构** 解决了第一个痛点，又用 **基于置信度的贪心变长验证（`MTPVarlen`）** 解决了第二个痛点。本章我们将这两大机制拆开细看。

---

## 6.1 DFlash 草稿模型：一次前向并行算特征，马尔可夫头逐步采样

打开 `deepinfer/model_executor/models/mtp/dflash.py`（L90–L346）和 `layers/sampler.py`，你会发现 DeepInfer 里的 `nextn`（MTP）投机模块不是传统的每步串行跑一层 Transformer，而是专门设计的 **`DFlashDraftModel`**（代码 L98–L104 明确断言只支持 `DFlashNextPredictHeadHC`）。

它的工作流程分为极为精巧的三步，全部在一次 Decode 步中一气呵成：

```{.mermaid caption="图 6-1　DFlash 投机草稿模块的“并行提特征 + 串行马尔可夫采样”三步流水线（dflash.py L140–L345）"}
flowchart TB
  subgraph Step12["第一阶段：主模型顺带更新缓存 + 5 个草稿位一次并行提特征"]
    direction LR
    M1["1. 主模型 forward() 提取目标层特征<br/>经 main_kv_proj 写入草稿 SWA 环"] --> D1["2. 构造 [golden, noise × 4] 共 5 个输入<br/>送入 DFlash 层（窗口 133）单次并行算出隐藏特征"]
  end

  subgraph Step3["第二阶段：轻量级马尔可夫头 + 置信度头逐步采样（sample_drafts, step_i = 0..4）"]
    direction LR
    S1["3. step 0 直接采样 draft[0]；<br/>step ≥ 1 将上一草稿词向量经 markov_w1/w2 注入当前特征"] --> S2["4. 经共享 lm_head 采样 draft[i]，同时过 ConfidenceHead<br/>输出每步独立置信度 confidence_rates[i]"]
  end

  Step12 --> Step3
```

对照 `dflash.py` 的源码，这三步把“快”和“准”结合到了极致：

1. **零开销共享主模型特征（`update_main_cache`，`deepseek.py` L223 & `dflash.py` L140–L175）**：
   DFlash 不需要自己单独做 prefill。主模型在跑前向时，会把若干目标层（`dspark_target_layer_ids`）的隐藏状态拼接起来，通过 `main_kv_proj` 投影成 K 和 V，直接写进 DFlash 自己的 SWA 环形缓冲区里。
2. **用噪声占位，一次并行跑完所有草稿层（`_embed_draft_init_tokens` & `forward_drafts`，L230–L277）**：
   进入采样阶段后，主模型先对上一轮验证通过的最后一个位置采样出 1 个必定接受的 **`golden_token`**。接着，DFlash 为每条请求构造长度为 `dflash_block_size = nextn = 5` 的初始序列：第 0 个位置放刚刚采出的 `golden_token`，第 $1 \dots 4$ 个位置全部填固定的占位符 `dflash_noise_token_id`！然后把这 5 个位置**一次性并行送入 DFlash 的 Transformer 层**（`forward_drafts`），在 `128 + 5 = 133` 的滑动窗口上做一次前向（有独立的 `DFlashCudaGraphRunner` 加速），拿到 5 个位置的骨干特征 `hidden_states`。
3. **马尔可夫头（Markov Head）注入自回归依赖 + 置信度头（ConfidenceHead）打分（`sample_drafts`，L279–L345）**：
   既然第 $1 \dots 4$ 个位置输入的是盲猜的噪声 token，怎么让第 $i$ 个草稿知道第 $i-1$ 个草稿到底采出了哪个词呢？
   `sample_drafts` 用一个极轻量的循环 `for step_i in range(5):` 解决这个问题：
   - 当 `step_i == 0` 时，直接把第 0 个位置的 `hidden_states` 送进共享词表头 `lm_head` 采样出第一个草稿 `draft[0]`；
   - 当 `step_i >= 1` 时，取出上一步刚采出的草稿词 `draft[step_i - 1]` 的词向量，经过两层轻量投影 `markov_w2(act(markov_w1(norm(prev_emb))))` 得到 `markov_out`，加到当前位置的骨干特征上（`combined_states = hidden_states_i + markov_out`），再送入 `lm_head` 采样出 `draft[step_i]`！
   - 与此同时，把 `combined_states` 送入 **`ConfidenceHead`**（L312–L315），立刻预测出当前草稿 token 被主模型接受的独立概率 `confidence_rates[:, step_i]`！

这样一来，沉重的 Transformer 注意力层只跑了 **1 次并行前向**，而串行自回归只发生在极轻量的两层小矩阵（Markov Head）上，既保住了多步草稿之间的条件依赖，又把草稿生成耗时压到了最低。

---

## 6.2 为什么需要“变长验证”？从独立置信度到累乘接受率

现在，每条请求手里都有了 5 个草稿 token，以及 `ConfidenceHead` 给出的 5 个独立置信度 $c_1, c_2, c_3, c_4, c_5 \in [0, 1]$。

请注意一个关键的数学事实（见 `scheduler.hpp` L1242–L1250 `process_MTPInfo` 的中文注释）：

> “modeling 返回的 confidence rate 是每个 token 独立的，使用前要做一个**累乘**。”

为什么必须累乘？因为在投机解码的验证规则里，只有当第 $1, 2, \dots, j-1$ 个草稿**全部被接受**时，第 $j$ 个草稿才有机会被采纳！因此，第 $j$ 个草稿最终能为我们贡献 1 个有效输出 token 的**边际期望收益 $p_j$**，等于前 $j$ 个独立置信度的乘积：

$$p_j = \prod_{m=1}^{j} c_m \quad (p_1 \ge p_2 \ge p_3 \ge p_4 \ge p_5)$$

举个具体数字例子：假设某条请求的 5 个独立置信度是 $[0.9, 0.8, 0.5, 0.5, 0.4]$，那么累乘后的期望接受率就是：

$$[0.90,\; 0.72,\; 0.36,\; 0.18,\; 0.072]$$

你看，验证第 1 个草稿有 $0.90$ 的把握多拿 1 个 token，而验证第 5 个草稿只有 $0.072$ 的期望收益！如果此时批次里有 8 条请求，固定把所有草稿全验一遍需要算 $8 \times (1 + 5) = 48$ 个 token；而如果我们砍掉那些期望收益只有 $0.07$、$0.18$ 的鸡肋草稿，把本步总 token 数 `total_q` 控制在 **32** 个以内，就能直接用上 `graph_size = 32` 的小号 CUDA Graph，步耗时大幅下降，单位时间产出的有效 token 反而更多！

---

## 6.3 `MTPVarlen::schedule`：贪心寻找吞吐量最高点

这套“砍鸡肋草稿、卡最佳档位”的算法实现在 `csrc/engine/mtp_varlen.hpp`（L150–L267）中。它的逻辑非常直观，我们可以用四步把它算明白：

1. **保底必算项**：假设本步有 $B$ 条活跃请求。每条请求至少要验证 1 个主输入 token（上一步刚采出的那个 token），它 100% 会产出 1 个新 token。所以初始总长度 $\text{cur\_bsz} = B$，初始期望产出 $\text{cur\_expected} = B$。
2. **所有候选草稿按累乘接受率从高到低排队**：把 $B$ 条请求各自手头的 5 个草稿的累乘接受率 $p_{i, j}$ 全部丢进同一个全局数组 `queue` 里，按 $p_{i, j}$ 从大到小降序排列（L188–L214）。
   - *思考题：全局按数值排序后，会不会出现“某条请求的第 3 个草稿入选了，但它的第 2 个草稿没入选”的非法情况？*
   - *答案：绝不会！* 因为同一条请求内部的累乘接受率满足单调不增（$p_{i, 1} \ge p_{i, 2} \ge \dots \ge p_{i, 5}$），按从大到小取前缀时，排在前面的草稿必定先于排在后面的草稿入选，天然构成合法的前缀！
3. **逐个往锅里加草稿，算期望吞吐量**：从高到低遍历 `queue`，每加入第 $k$ 个高价值草稿，总 token 数变成 $\text{cur\_bsz} = B + k$，期望产出变成 $\text{cur\_expected} += p_{(k)}$。接着查询 $\text{cur\_bsz}$ 对应的 CUDA Graph 档位估计耗时 $\text{time\_ms}(\text{cur\_bsz})$，计算期望吞吐量：
   $$\text{Throughput}(\text{cur\_bsz}) = \frac{\text{cur\_expected}}{\text{time\_ms}(\text{cur\_bsz})}$$
   记录下令 $\text{Throughput}$ 达到最大值的最佳总预算 **`best_bsz`**（L235–L259）。
4. **按 `best_bsz` 分发名额**：把 `queue` 前 `best_bsz - B` 个入选草稿分回各自所属的请求，得到每条请求本步真正要验证的长度 `seqlens_q[i]`（L263–L266）。

其中，单步耗时估计公式为（L243）：

$$\text{time\_ms} = \text{baseline\_ms}(\text{graph\_size}) + \text{slope\_ms} \times \text{graph\_size} + \text{sampler\_ms}$$

- `baseline_ms`：引擎启动 `warm_up_model` 时，对每个 CUDA Graph 档位实测 5 次去掉极值求平均得到的基准阶梯耗时（`model_runner.py` L562 `profile_decode_perf`）；
- `slope_ms`：`ForwardRuntimeEstimator`（L44–L131）在运行过程中根据最近 5,000 步的 `(实测 forward_ms - baseline_ms) / graph_size` 动态统计的中位数斜率，用来补偿上下文变长带来的注意力耗时增长；
- `sampler_ms`：`SamplerRuntimeEstimator`（L20–L42）统计的最近 9 步采样器耗时中位数。

我们用 `minidi/varlen.py`（与 C++ 逐行同构）跑一个 8 请求、每请求 5 草稿的例子，就能亲眼看到这条带有阶梯跳变的吞吐曲线：

![图 6-2　变长验证的贪心寻优过程：吞吐率在 CUDA Graph 档位跳变点出现锯齿，算法自动锁定最高峰 total_q=24（由 minidi/varlen.py 生成）](images/plots/ch06_varlen.png)

看图 6-2：每当 `total_q` 跨过 16、24、32、48 这些 CUDA Graph 档位边界时，分母耗时跳升一级，曲线就会向下挫一截；而贪心算法精准地把总预算卡在了 **`total_q = 24`（`graph_size = 24` 档位的右边缘）**，砍掉了后面 24 个低收益草稿，吞吐率显著高于绿色虚线（固定验证全部 48 个位置）。据官方 Q&A 披露，线上开启变长验证后，平均验证草稿数从 5.0 降至 3.2~4.4（省去约 30% 验证算力），而平均接受长度几乎不变（3.29 vs 3.20），端到端吞吐提升 **4%~7%**。

---

## 6.4 当变长验证遇上异步调度（ZOS）：N-2 轮定总盘，N-1 轮 GPU 分名额

最后，我们看一个把第 5 章（ZOS）和第 6 章（变长验证）拧在一起的精妙设计：

开启异步调度 ZOS 后，当 CPU 在跑 `schedule(N+1)` 时，**第 $N$ 步（也就是上一轮）的 `ConfidenceHead` 还在 GPU 上没算完呢！** CPU 手里只有更早一轮（**第 $N-2$ 轮**）拷回来的旧置信度。这该怎么排第 $N+1$ 步的变长验证？

DeepInfer（`mtp_varlen.hpp` L160–L169 注释与 `worker.py` L218–L219）采用了一套**“CPU 用 $N-2$ 轮置信度定总盘子，GPU 用刚出炉的 $N-1$ 轮置信度重分名额”**的两级协作方案：

```{.mermaid caption="图 6-3　异步调度（ZOS）下 CPU 与 GPU 协同完成变长验证（mtp_varlen.hpp & schedule_mtp_varlen.py）"}
flowchart TB
  subgraph CPU_Stage["1. CPU 超前调度阶段（schedule(N+1)）"]
    direction LR
    C1["CPU 读取已落账的 N-2 轮置信度<br/>（prev_accept_rates）"] --> C2["运行 MTPVarlen::schedule()<br/>确定本步最佳总预算 best_bsz"]
    C2 --> C3["根据 best_bsz 锁定 CUDA Graph 档位<br/>并预分配输入张量形状"]
  end

  subgraph GPU_Stage["2. GPU 紧接采样后就地重算阶段（execute_model_step 阶段 1 & 3）"]
    direction LR
    G1["GPU 刚跑完 sample(N)<br/>产出新鲜的 N-1 轮 confidence_rates"] --> G2["TileLang 算子 compute_seqlens_q_by_confidence_rates：<br/>在总预算 best_bsz 不变的前提下，按最新置信度<br/>累乘 + 全局 Top-K 重新分配各请求的 seqlens_q"]
    G2 --> G3["一边直接喂给 forward(N+1)，<br/>一边异步 D2H 拷回 CPU 供阶段 3 采样前更新元数据"]
  end

  CPU_Stage --> GPU_Stage
```

为什么这套两级方案行得通？

- **总预算 `best_bsz` 对单步波动不敏感**：一个批次里几十条请求的整体置信度分布非常平稳，用 $N-2$ 轮和 $N-1$ 轮算出来的最佳总档位 `best_bsz` 几乎完全一样，因此 CPU 可以放心地用它来锁定 CUDA Graph 档位。
- **具体哪条请求分几个名额对最新置信度极敏感**：某条请求在上一轮刚采到了标点符号或确定性词组，置信度可能瞬间飙高。因此，保持总预算 `best_bsz` 不变，让 GPU 在启动 `forward(N+1)` 前跑一个极快的 TileLang 算子 `compute_seqlens_q_by_confidence_rates`（`layers/tilelang_ops/schedule_mtp_varlen.py`），用新鲜出炉的 $N-1$ 轮置信度重新做一次累乘和全局 Top-$K$ 名额分配，既不破坏 CUDA Graph 的静态形状，又拿到了 100% 鲜活的单请求验证长度！

---

## 6.5 本章小结

- **DFlash 草稿架构**：利用主模型中间层特征直接更新草稿 SWA 缓存（零额外 prefill），用 `[golden, noise, ...]` 在一次并行前向里提取 5 个位置的骨干特征，再用极轻量的马尔可夫头（`markov_w1/w2`）和置信度头（`ConfidenceHead`）逐步采样草稿并输出独立置信度。
- **置信度变长验证（`MTPVarlen`）**：将独立置信度做前缀累乘得到边际期望收益，按降序贪心枚举总验证长度 `cur_bsz`，结合 CUDA Graph 阶梯耗时曲线找到期望吞吐量最大的 `best_bsz`。
- **与 ZOS 无缝融合**：CPU 用 $N-2$ 轮置信度拍板总预算 `best_bsz` 和 CUDA Graph 档位，GPU 用刚出炉的 $N-1$ 轮置信度在算子内就地重分每条请求的 `seqlens_q`。
