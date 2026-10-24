# 第 7 章　模型架构、算子与并行：YoCo、Wide-EP 与专家负载均衡

前六章我们把推理引擎的“控制面”（调度、显存、异步流水线、投机验证）摸了个底朝天。这一章我们深入“数据面”，看看 **DeepSeek-V4.1-Flash** 的模型计算图在 `deepinfer/model_executor/` 下是怎么组织的，以及它是如何用 **YoCo 双段架构**、**DP / CP / EP 并行** 和 **EPLB 专家负载均衡** 把多卡集群跑满的。

---

## 7.1 模型代码的四层解耦设计

打开 `deepinfer/model_executor/models/v4_modules/__init__.py`，文件顶部的模块文档把 20 多个模型文件清晰地划分为四种职责角色：

| 职责角色 | 代表文件（`models/v4_modules/`） | 为什么要这样拆？ |
| --- | --- | --- |
| **1. 纯参数容器**<br/>（Parameter Containers） | `containers.py` · `linear.py` · `mla_attn.py` · `moe.py` · `hc_blocks.py` · `engram_parameters.py` | **只管权重长什么样，不写任何 `forward` 逻辑**。严格对齐 HuggingFace 权重的命名、形状与数据类型，让模型加载器（Weight Loader）极其简单。 |
| **2. 运行时适配与前向**<br/>（Runtime Adapters） | `attn.py` · `compress.py` · `index_attn.py` · `mlp.py` · `hc.py` · `engram.py` | **决定运行时怎么算**。在权重加载完成后，把参数容器重排成硬件最喜欢的布局（例如将专家权重打包为 FP8/FP4 或 Mega MoE 格式）并执行前向。 |
| **3. 无状态张量算子**<br/>（Tensor Ops） | `yoco.py` · `mhc_ops.py` · `mhc_modeling.py` · `hc_lazy.py` | 不持有任何权重参数，专门负责跨层张量变换（如 YoCo 序列截断、mHC 多路残差混合与折叠）。 |
| **4. 拓扑推导与组装**<br/>（Topology & Builder） | `cross_layer.py` · `engram_metadata.py` · `builder.py` | 从 `PretrainedConfig` 推导跨层 KV 共享拓扑（Share-KV / Share-Index），校验合法性并拼装完整模型树。 |

在这种架构下，DeepSeek-V4.1-Flash 的一个 Decoder 层融合了多项前沿结构：

- **混合注意力与跨层共享（`attn.py`、`compress.py`、`index_attn.py`、`cross_layer.py`）**：包含滑动窗口注意力（SWA）、门控低秩压缩注意力（Compressed MLA）以及基于 `relu(q @ k) * weights` 打分选 Top-$k$ 块的稀疏索引注意力（Index Attention），且后半段层直接共享前半段层的 KV 或索引（Share-KV / Share-Index）。
- **多路超连接残差主干 mHC（`hc.py`）**：残差流在模型内部被扩展为 `hc_mult` 条并行通道，每个子层前后通过小矩阵做通道混合，直到最后进入 `lm_head` 前才折叠回单路。
- **稀疏 N-gram 记忆表 Engram（`engram.py`）**：根据输入 token 的 N-gram 哈希值从跨卡分片的大表里查出稀疏词向量，经 `DeepEP` 收集后门控融入隐藏状态。

---

## 7.2 YoCo（You Only Cache Once）：为什么后半段层可以跳过一截 Prefill？

在阅读第 2、4 章时，我们多次遇到过 `yoco` 这个词。到底什么是 **YoCo**？打开 `deepinfer/model_executor/models/deepseek.py`（L154–L245）和 `v4_modules/yoco.py`（L15–L60），它的完整机制跃然纸上：

模型的全部 Transformer 层以 `n_prefill_layer`（例如第 20 层）为界，被切分成前后两半：

- **前半段（Producer 层，`0 .. n_prefill_layer - 1`）**：负责计算并产出压缩 KV、索引 KV 以及需要落盘的 SWA 状态。在最后一层 Producer 结束时，还会把经过归一化后的隐藏状态保存为 **`shared_hidden`**。
- **后半段（Consumer 层，`n_prefill_layer .. num_layers - 1`）**：它们自己**不再生成新的长程 KV**，而是直接通过跨层共享读取前半段算好的 `shared_hidden` / KV，或者只看极短的局部 SWA 窗口！

请顺着这个结构想一想：**既然在 Prefill 阶段，我们只需要模型吐出整段 Prompt 最后一个位置的预测 Token（以及为后续 Decode 准备好最后 134 个位置的 Consumer SWA 窗口），那么后半段 Consumer 层还有必要傻乎乎地把前面几千个历史 Token 全算一遍吗？**

**完全没必要！** 因为后半段层既不产出长程 KV，后面的 token 在后半段层里也只需要看最近 `keep_tokens = 134` 个位置的局部窗口！

```{.mermaid caption="图 7-1　YoCo 双段架构与 Prefill 截断机制（deepseek.py L185–L220 & v4_modules/yoco.py）"}
flowchart TB
  subgraph Producer["1. 前半段 Producer 层（第 0 .. n_prefill_layer - 1 层）"]
    direction LR
    P1["输入完整 Prompt Chunk<br/>（例如 4096 个 token）"] --> P2["逐层计算 SWA 与压缩 KV 写入缓存；<br/>在分界点输出全长特征 + shared_hidden"]
  end

  subgraph Gate["2. 分界点截断与后半段 Consumer 层（第 n_prefill_layer .. N-1 层）"]
    direction LR
    G1{"当前 Chunk 含<br/>末尾 134 个位置？"} -- "否（纯中间块）" --> G2["skip_yoco_layers = True<br/>后半段整段跳过！"]
    G1 -- "是（含末尾窗口）" --> G3["截断保留末尾 134 个位置送入后半段层，<br/>经 lm_head 算出末尾 logits 并采样首 token"]
  end

  Producer --> Gate
```

看图 7-1，这就是为什么：
1. 在分块 Prefill（Chunked Prefill）跑一个 32K 的长 Prompt 时，前面十几个 chunk 连 Prompt 的倒数第 134 个位置都没摸到，`skip_yoco_layers` 为真（`deepseek.py` L195–L198），模型算完前半段 Producer 层就直接返回了，**后半段层一行代码都不用跑，直接省掉一半层的计算量**！
2. 到了最后一个 chunk，也只需要把末尾 `keep_tokens = 134` 个 token 喂进后半段 Consumer 层，填满它的局部 SWA 窗口并算出最后一个位置的 logits 即可。
3. 这也解释了第 2 章里为什么开启 YoCo 截断时必须关掉 `prompt_logprobs`——因为前面几千个 prompt token 压根就没进过后半段层和 `lm_head`，自然没有它们的 logprobs！

---

## 7.3 并行策略：为什么放弃 TP，选择 DP + CP + Wide-EP？

查看 `ParallelConfig`（`deepinfer/config.py` L150–L180），你会发现一个惊人的事实：DeepInfer **只支持数据并行（DP）、上下文并行（CP）和专家并行（EP），完全没有张量并行（TP）**！

为什么抛弃了传统大模型推理最常用的 TP？因为 TP 每跑一层注意力和 MLP 都要做两次跨卡 `AllReduce`，在 Decode 小批次下通信延迟极高。而 DeepSeek-V4.1-Flash 的 MLA 共享 KV 让单张卡就能轻松装下完整的注意力头，因此：

- **注意力层走 DP（或处理超长上下文时配合镜像对称的 Mirror CP）**：每张卡跑自己独立的一批请求，拥有独立的 C++ 调度器和 KV Cache 池，卡间在注意力层零通信。
- **MoE 专家层走 Wide-EP（大跨度专家并行）**：所有 DP 卡上的 token 在进入 MoE 层时，通过 `DeepEP` 一次性分发（dispatch）到对应的专家所在卡，算完分组矩阵乘（Grouped GEMM）后再合并（combine）回原卡。

在 `v4_modules/mlp.py`（`DeepseekMoE.forward`，L490–L610）中，一层 MoE 的执行流包含两项极致的通信掩盖与融合优化：

```{.mermaid caption="图 7-2　单层 DeepseekMoE 在专家并行（EP）下的两条执行路径（v4_modules/mlp.py L490–L610）"}
flowchart TB
  subgraph NormalPath["路径 A：标准重叠路径（计算与 DeepEP 通信并行掩盖）"]
    direction LR
    N1["本卡输入 hidden_states<br/>经 gate 选出 Top-k 逻辑专家并映射为物理副本"] --> N2["DeepEP dispatch<br/>将 token 跨卡发往目标专家 rank"]
    N2 --> N3["routed_experts_forward<br/>分组 GEMM 计算路由专家"]
    N1 --> N4["shared_forward（共享专家）<br/>限制 SM 数量（预留通信 SM），与跨卡通信并行执行"]
    N3 --> N5["DeepEP combine<br/>跨卡收回结果并就地加上共享专家输出"]
    N4 --> N5
  end

  subgraph MegaPath["路径 B：Mega MoE 单核融合路径（NVLink 域内极致低延迟）"]
    direction LR
    M1["use_mega_moe = True 时启用"] --> M2["调用 mega_moe_func 单一融合大算子：<br/>将 Gate + Dispatch + Grouped GEMM + Combine 融为一体"]
  end

  NormalPath --- MegaPath
```

---

## 7.4 EPLB：如何用冗余专家解决 MoE 的“木桶效应”？

在专家并行（EP）下，所有卡每一层都要在 `combine` 处互相等待。如果某几个“热门专家”恰好集中在 Rank 0 上，导致 Rank 0 收到的 token 数是其他卡的 3 倍，那么其他 63 张卡就只能干等着 Rank 0 算完——这就是 MoE 推理最头疼的**木桶效应（负载不均衡）**。

为了削平热点，DeepInfer 在 `deepinfer/model_executor/eplb/` 中实现了 **EPLB（Expert Parallelism Load Balancer，专家并行负载均衡）**。它的核心思想非常直观：

1. **区分“逻辑专家”与“物理专家副本”**：
   假设模型原本有 $E = 16$ 个逻辑专家，分布在 $R = 4$ 张卡上。我们在显存里多开辟 $M = 4$ 个冗余槽位，使总物理专家槽位数达到 $E + M = 20$ 个（每张卡放 $20 / 4 = 5$ 个物理专家）。
2. **第一步：给最忙的逻辑专家加副本（`rebalance_experts_global`，`eplb/policy.py` L73–L140）**：
   统计过去一个滑动窗口（例如 100 步或 1000 步）内每个逻辑专家在每一步收到的 token 历史负载。一共做 $M$ 轮贪心选择：每一轮都挑出**“给它多加 1 个副本后，能让历史各步单副本峰值负载之和下降最多”**的那个热门逻辑专家，给它增发 1 个物理副本。当某个逻辑专家拥有 $c$ 个物理副本时，门控路由（Gate）会把发往它的 token 均匀拆成 $c$ 份。
3. **第二步：把物理副本装进各张卡（`_batch_greedy_indices`，`eplb/policy.py` L12–L70）**：
   现在手里有 $E + M = 20$ 个物理副本，要把它们装进 $R = 4$ 个容量为 5 的篮子（Rank）里。算法先把所有物理副本按历史总负载从大到小排序（先放重石头，再塞小沙子），每放一个副本，都挑当前未满、且**放进去之后历史各步最忙卡负载之和（$\sum_t \max_r \text{load}_{t, r}$）最小**的那张卡！

为什么 EPLB 的目标函数要看“历史每一步的最大值之和 $\sum_t \max_r \text{load}_{t, r}$”，而不是简单地看“窗口内的平均负载”？
设想专家 A 只在奇数步忙，专家 B 只在偶数步忙，它们的平均负载都是 50。如果只看平均值，把 A 和 B 放在不同卡上和放在同一张卡上似乎差不多；但按每步最大值优化，就会发现把“同时忙”的专家拆开、把“错峰忙”的专家组合在一起，才能让**每一步**的木桶短板都降到最低！

我们在 `minidi/eplb.py` 中完整实现了这套两阶段贪心算法。在服从 Zipf 幂律分布的偏斜负载下，朴素放置与 EPLB 的对比如图 7-3 所示：

![图 7-3　EPLB 负载均衡效果对比：16 个逻辑专家在 4 张卡上，增加 4 个冗余副本并按步峰值贪心放置后，moe_lb（最慢卡/均值）从 1.38 降至 1.07（由 minidi/eplb.py 生成）](images/plots/ch07_eplb.png)

在线上生产环境中（据官方 Q&A），Prefill 实例每卡配置 4 个冗余专家，统计窗口 100 步、每 1000 步通过 `eplb/shuffle.py`（基于 NCCL P2P 或单机 IPC 共享内存）重排一次权重，实测不均衡度 `moe_lb` 低至 **1.045~1.074**；Decode 实例每卡配置 1 个冗余专家，每 10,000 步重排一次，实测 `moe_lb = 1.090`。更细一点（官方 Q&A L95–L103）：Prefill 的 1.074 对应 EP4DP4、1.045 对应 EP2DP2，冗余专家数会向上取整，使专家总数是 `ep_size` 的整数倍；Decode 的统计窗口是 1,000 步，在 EP64 下每卡 1 个冗余专家取整后为每卡 7 个专家（不含共享专家）。官方 Q&A 同时说明 Wide-EP 能显著降低 MoE 专家部分每步的访存量，并在 `doc/images/wide_ep_decode_pareto.png` 给出了 EP8 与 EP64 的 decode 吞吐对比（L55–L59）。

---

## 7.5 本章小结

- **模型四层解耦**：参数容器对齐 HF 权重名，运行时适配器负责硬件重排与前向，张量算子与拓扑组装器各司其职。
- **YoCo 双段架构**：以 `n_prefill_layer` 为界分为 Producer 与 Consumer，Prefill 时对不含末尾窗口的中间块直接跳过整个后半段层，对末尾块也只截断保留最后 134 个 token 送入后半段。
- **无 TP 的并行哲学**：注意力层走 DP / Mirror CP，MoE 层走 Wide-EP，并通过限制共享专家 SM 数实现计算与通信掩盖，或在 NVLink 域内启用 Mega MoE 单核融合。
- **EPLB 专家负载均衡**：通过历史步峰值贪心复制热门专家、错峰装箱放置物理副本，并定期在线热迁移专家权重，消除 MoE 跨卡通信的木桶效应。
