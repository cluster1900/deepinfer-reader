# 第 7 章　模型、算子与并行

前几章讲的是“引擎”：调度、内存、异步。本章往下看一层：模型怎么组织、用了哪些算子库、多卡怎么切、MoE 专家怎么做负载均衡。这一层代码量最大（TileLang 算子 23,331 行，模型结构 7,051 行，见第 1 章），本书只讲与系统设计相关的部分，不逐个解读 kernel。

## 7.1 模型代码的四种角色

`deepinfer/model_executor/models/v4_modules/__init__.py` 的模块 docstring 是读模型代码最好的地图。它把文件分成四种角色，并要求新文件按角色放：

| 角色 | 文件 | 职责 |
|---|---|---|
| 参数容器 | `containers.py` `linear.py` `norm.py` `rope.py` `mla_attn.py` `moe.py` `hc_blocks.py` `engram_parameters.py` `heads.py` | 按 HF 权重的名字、形状、dtype 分配参数，不写 forward |
| 运行时适配与 forward | `attn.py` `compress.py` `index_attn.py` `mlp.py` `hc.py` `engram.py` `utils.py` | 把容器转换成运行时布局，决定“模型怎么算” |
| 张量算子 | `mhc_ops.py` `mhc_modeling.py` `yoco.py` `hc_lazy.py` | 无参数、无 forward，只操作张量 |
| 由配置推导的拓扑与组装 | `cross_layer.py` `engram_metadata.py` `builder.py` | 从配置推导并校验结构，构建参数树 |

把“参数长什么样”和“怎么计算”分开的好处是：权重加载只需对齐 HF 名字，运行时可以自由重排布局（比如 `mlp.py` 把专家权重重新打包成 FP8/FP4 和 Mega MoE 布局）。

各模块 docstring 给出的结构要点（本书只转述，没有深入验证每条数学细节）：

- **注意力**（`attn.py`）：`V4Attention` 在 MLA 基础上加了压缩 KV、稀疏索引、滑动窗口、YoCo 和跨层共享几条路径；`DFlashAttention` 服务草稿模型。
- **压缩 KV**（`compress.py`）：门控低秩压缩，压缩结果一路喂给稀疏索引器，一路写入主 KV cache。
- **稀疏索引器**（`index_attn.py`）：`MQA` 计算 `relu(q @ k) * weights` 打分，为每个 query 选 top-k 个压缩 KV 块。
- **mHC**（`hc.py`）：残差主干被扩成 `hc_mult` 路，每个子层前后做混合，最后在 lm head 前折回一路。
- **Engram**（`engram.py`）：按 n-gram 哈希从分片表里取稀疏 embedding，跨 rank 收集后门控融合进 hidden state。
- **跨层共享**（`cross_layer.py`）：share-KV、share-index、share-index-K 三种共享拓扑，由配置推导并校验。
- **YoCo**（`yoco.py`）：`n_prefill_layer` 之后的层是“consumer”层，配合 SWA bounded replay 截断 prefill（第 4 章）。

## 7.2 算子库

README 列出的算子来源是 FlashMLA、DeepGEMM、DeepEP、TileKernels、DeepSelect 和仓库自带的 TileLang 算子。在 `deepinfer/` 包里按 import 粗略统计（`grep "import X / from X"`，按文件计数）：

| 库 | 引用文件数 | 典型位置 |
|---|---|---|
| `tile_kernels` | 46 | 线性层、RoPE、asc 算子等 |
| `tilelang` | 60 | `layers/tilelang_ops/` 下的自定义算子（含 `asc/` 昇腾版本） |
| `deep_gemm` | 9 | `layers/linear.py`、`attn.py`、`hc.py`、`index_attn.py`、MoE |
| `deep_ep` | 5 | `mlp.py`（专家 dispatch/combine）、`engram.py`、CP 通信 |
| `flash_mla` | 4 | `attention_backend.py`、`attn.py`、输入元数据 |
| `deep_select` | 4 | sampler、`index_attn.py`（top-k 选择） |

`layers/tilelang_ops/` 下有 35 个条目，第 5、6 章提到的 `compute_seqlens_k_zos`、`schedule_mtp_varlen` 都在这里。Q&A 对开源版和内部版的差别说得很具体：“FlashMLA 少了一个硬改 SASS 的优化，decoding 算子慢 5% - 10%”，“端到端性能接近，最多差 5%”。

## 7.3 并行策略

`ParallelConfig`（`deepinfer/config.py` L150–L180）只有三个维度：数据并行 DP、上下文并行 CP、专家并行 EP（外加 Engram 表的并行度），没有张量并行：

- `data_parallel_size == 0` 时自动取 `max(1, ep // cp)`（L161–L162）；
- `model_parallel_size = context_parallel_size`（L171），`world_size = max(cp * dp, engram_parallel_size)`；
- 断言 `cp * dp` 能被 `ep` 整除（L178）。

注意力部分按 DP 切：每个 DP rank 跑自己的一批请求，各有自己的调度器和 KV cache（第 3 章的跨 rank 投票就是为了让所有 rank 同一步做同一种事）。MoE 部分按 EP 切：每个 rank 持有一部分专家，token 通过 DeepEP 发往专家所在 rank。

Q&A 给出的线上配置：Prefill “EP2~8/DP2~8，在显存足够的情况下用尽可能小的EP/DP”；Decode “EP64~128/DP64~128 (Wide-EP)，在显卡可用的情况下用尽可能大的EP/DP”；“暂未用TP，CP”。还提到 Wide-EP “显著减少了 per-step MoE 专家部分的访存量”。这些是部署经验，不在开源代码里。

```{.mermaid caption="图 7-1　一层 MoE 在 EP 下的数据流（mlp.py DeepseekMoE.forward，非 Mega MoE 路径）"}
flowchart LR
  H["本 rank 的 hidden states<br/>（注意力按 DP 切）"] --> G["gate_forward<br/>top-k 选专家<br/>logical → physical 映射"]
  G --> D["dispatch<br/>DeepEP 发往专家所在 rank"]
  D --> E["routed_experts_forward<br/>分组 GEMM"]
  H --> S["shared_forward<br/>共享专家（限制 SM 数，与通信重叠）"]
  E --> C["combine<br/>DeepEP 收回，并加上共享专家输出"]
  S --> C
  C --> O["输出"]
```

`DeepseekMoE.forward`（`v4_modules/mlp.py` L565 起）的流程正如图 7-1。两个值得注意的细节：

1. **共享专家与 combine 重叠**。共享专家在 `limit_num_sms(DEEPINFER_USERBUFFER_COMM_SMS)` 下计算，给通信留出 SM，结果作为 `bias` 直接交给 `combine` 相加（L493–L511）。
2. **Mega MoE**。`use_mega_moe` 时整层走 `mega_moe_forward`（L528–L563），gate、dispatch、GEMM、combine 融合成一个 `mega_moe_func`。Q&A 说“Mega MoE 是 NVL domain 的通用方案，无论 EP 大小、batch 大小，Mega MoE 都显著比 unfused MoE 快”。

gate 里还有一个调试开关 `DEEPINFER_FORCE_IMBALANCE_RATIO`：大于 0 时用偏向第一个 EP rank 的随机路由替换真实路由，用来压测负载不均衡（L367–L371）。

## 7.4 EPLB：专家负载均衡

EP 下一个 rank 的耗时取决于它承载的专家收到多少 token。热门专家所在的 rank 会成为整步的瓶颈。EPLB 的思路是给热门专家加副本，再把副本均衡地摆到各 rank。

**物理专家与逻辑专家**。`moe.py` 的 docstring 定义了计数语义：logical = `n_routed_experts`，即 gate 的选择范围；extra = `n_redundant_experts`；physical = logical + extra，必须能被 `ep_size` 整除；local = physical // ep_size。冗余数由 `resolve_num_extra_experts` 向上取整（`eplb/__init__.py` L21–L28）：`DEEPINFER_OVERRIDE_NUM_EXTRA_EXPERTS` 给下限，再补到能整除。gate 通过 `to_physical_map=logical_to_physical_map` 和 `logical_count` 把选中的逻辑专家映射到某个副本（`mlp.py` L376–L398）。

**重算放置**的策略在 `eplb/policy.py`，分两步：

1. `rebalance_experts_global`（L73–L140）：每轮挑一个逻辑专家加一个副本。挑法是“给 j 加副本之后，每步的最大每副本负载之和最小”的那个 j，重复 `num_redundant` 轮。
2. `_batch_greedy_indices`（L12–L70）：把副本按历史总负载降序排，逐个放到“放进去后 Σ_step max_rank 负载最小”的未满 rank。注释说明优化目标是让每个历史 step 的最忙 rank 负载之和尽量小，而不是只看总负载。这样能照顾到“不同步热点不同”的情况。

**何时重排**由 `ExpertLoadBalancer`（`eplb/__init__.py` L42 起）决定：

- 每步在 `DeepseekForCausalLM.sample` 里调 `step()`（`deepseek.py` L403–L404），把本步各专家的 token 数存进长度为 `DEEPINFER_EXPERT_COUNT_WINDOW_SIZE`（默认 100）的环形窗口；
- 每 `DEEPINFER_EXPERT_REARRANGEMENT_STEP_INTERVAL` 步重排一次。这个变量用 `EnvVar.int()` 声明，默认 0，**即默认不重排**（`environ.py` L116；`_update_rearrangement` 在间隔为 0 时直接返回，L146–L149）；
- 首次重排提前到 1/4 周期后，“尽早纠正冷启动的不均衡”（L121–L123）；
- NORMAL 引擎默认在 prefill 步不统计（`_skip_rearrangement_on_prefill`，L83–L84）。

重排时各 rank all-reduce 窗口统计，算出新映射，再搬运专家权重：`eplb/shuffle.py` 提供 NCCL batched P2P 的通用实现，以及单机时经 symm_mem/IPC 镜像直接读对端权重的实现。

Q&A 给出的线上参数：Prefill “每卡 4 个冗余专家起步……统计窗 100 步，每 1000 步重排”，实测 moe_lb（最慢 rank / 均值）为 1.074（EP4DP4）、1.045（EP2DP2）；Decode “每卡 1 个冗余专家起步……统计窗 1000 步，每 10000 步重排一次，实测moe_lb = 1.090”。

![图 7-2　EPLB 前后各 rank 的负载（合成的 Zipf 分布，minidi/eplb.py，示意用）](images/plots/ch07_eplb.png)

`minidi/eplb.py` 用 numpy 写了单层版本，逐行对照上面两个函数。图 7-2 是在合成的 Zipf 热度上的效果，`tests/test_algorithms.py` 验证了它比朴素放置的 moe_lb 更低。第 15 章会实现它。
