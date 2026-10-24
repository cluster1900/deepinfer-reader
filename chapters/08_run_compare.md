# 第 8 章　配置开关、确定性推理与主流框架横评

作为第一部分的收官之章，我们将前面七章零散提到的运行时开关、确定性复现条件以及评测工具串联成一份实用的**速查手册**，并在最后把 DeepInfer 与 vLLM、SGLang 放在同一张桌子上，看看三种不同定位的推理引擎在架构取舍上的本质分野。

---

## 8.1 编译构建与离线运行速查

DeepInfer 针对 **NVIDIA SM100（Blackwell）** 与 **华为昇腾 950 NPU** 两个硬件平台构建，要求 Linux、GCC ≥ 13、Python 3.12，以及 CUDA 13 或 CANN 9.2（`README.md` L33–L52）：

```bash
# 1. 指定目标硬件后端（编译与运行阶段均需保持导出）
export DEEPINFER_ACCELERATOR=nvidia   # 昇腾环境设为 ascend
./build.sh

# 2. 运行离线推理示例（默认模型 deepseek-ai/DeepSeek-V4.1-Flash-1017）
python3 examples/offline_inference.py            # 默认 EP=8, DP=8
python3 examples/offline_inference.py --ep 2 --dp 8
python3 examples/offline_inference.py --ep 2 --cp 8
```

通过 Python API 调用时，入口类 `deepinfer.entrypoints.llm.LLM`（`llm.py` L54 起）保持了与 vLLM 高度相似的签名：

```python
from deepinfer.entrypoints.llm import LLM
from deepinfer.sampling_params import SamplingParams

llm = LLM(
    model="deepseek-ai/DeepSeek-V4.1-Flash-1017",
    expert_parallel_size=8,
    gpu_memory_utilization=0.85,
    trust_remote_code=True,
)
outputs = llm.generate(
    ["请用三句话解释什么是投机解码："],
    SamplingParams(temperature=0.6, max_tokens=256, seq_seed=42),
)
```

需要特别注意 **`examples/_config.py` 对引擎默认行为的三处覆盖**（L75–L83）：
引擎底层默认关闭 EPLB 动态重排（`DEEPINFER_EXPERT_REARRANGEMENT_STEP_INTERVAL = 0`），但示例脚本的 `setup_common_env()` 在开启 EPLB 时会自动将冗余专家数设为 `32`、重排间隔设为 `30` 步，并开启 `DEEPINFER_REARRANGE_AFTER_PREFILL_IN_NORMAL_ENGINE=1`。读代码时如果只看 `environ.py` 的默认值，很容易漏掉示例脚本替你打开的这些开关。

---

## 8.2 核心环境变量全景表

DeepInfer 的核心特性大多通过 `deepinfer/environ.py` 中声明的环境变量控制。我们将全书涉及的 12 个关键开关按子系统归纳如下：

| 所属子系统 | 环境变量名 | `environ.py` 默认值 | 功能说明与生产建议 |
| --- | --- | --- | --- |
| **异步与计算图**<br/>（第 5 章） | `DEEPINFER_ZERO_OVERHEAD_SCHEDULING` | `NORMAL`/`COMPLETION` 开<br/>`PROMPT` 关 | 开启零开销异步调度（ZOS），使 CPU 调度第 $N+1$ 步与 GPU 执行第 $N$ 步重叠 |
| **异步与计算图** | `DEEPINFER_ENFORCE_EAGER` | `False` | 设为 `1` 时禁用 CUDA Graph，回退到 PyTorch 逐算子发射（用于调试） |
| **异步与计算图** | `DEEPINFER_CUDAGRAPH_BATCH_SIZE_INTERVAL` | `8` | CUDA Graph 按 `total_q` 捕获的档位步长（小于 8 时按 1, 2, 4 捕获） |
| **异步与计算图** | `DEEPINFER_DISABLE_SYNC_GUARD` | `False` | 设为 `1` 时关闭 `forbid_cuda_sync` 隐式同步报错守卫 |
| **投机与变长验证**<br/>（第 6 章） | `DEEPINFER_USE_NEXTN_DECODE` | `5` | DFlash 单步并行生成的最大候选草稿 token 数（`nextn`） |
| **投机与变长验证** | `DEEPINFER_ENABLE_MTP_VARLEN` | `False` | 开启基于 `ConfidenceHead` 与 CUDA Graph 阶梯耗时的贪心变长验证 |
| **显存与 YoCo**<br/>（第 4、7 章） | `DEEPINFER_MAX_BACKTRACKING_TOKENS` | `max(6, nextn + 1)` | SWA Ring 的回退与边界快照余量（主模型 SWA Ring = `128 + 6 = 134`） |
| **显存与 YoCo** | `DEEPINFER_YOCO_PREFILL_TRUNCATION_MODE` | `"off"` | YoCo 后半段层 Prefill 截断模式；生产推荐设为 `"compact"` |
| **显存与 YoCo** | `DEEPINFER_YOCO_PREFILL_KEEP_TOKENS` | `0` | YoCo 截断保留的尾部 token 数；README 推荐设为 `140`（不小于 134） |
| **专家负载均衡**<br/>（第 7 章） | `DEEPINFER_OVERRIDE_NUM_EXTRA_EXPERTS` | `0` | EPLB 额外复制的冗余物理专家总数下限（向上取整至 `ep_size` 的倍数） |
| **专家负载均衡** | `DEEPINFER_EXPERT_COUNT_WINDOW_SIZE` | `100` | EPLB 统计各逻辑专家历史步负载的滑动窗口步数 |
| **专家负载均衡** | `DEEPINFER_EXPERT_REARRANGEMENT_STEP_INTERVAL` | `0`（不重排） | 每隔多少步触发一次 EPLB 全局重算与跨卡专家权重热迁移 |
| **确定性推理** | `DEEPINFER_BATCH_INVARIANT` | `False` | 开启批次无关的确定性矩阵乘与注意力归约（仅支持 NVIDIA） |

---

## 8.3 怎么做到“逐 Token 完全可复现”？批不变推理与确定性采样

在强化学习（RLHF / GRPO）采样、线上故障排查和精度评测中，工程师经常有一个硬性诉求：**同一条请求、给定相同的随机种子，无论它是单独跑，还是和另外 63 条长短不一的请求拼在一个批次里跑，生成的每一个 token 都必须分毫不差！**

为什么普通的 GPU 推理做不到这一点？因为有三个“不确定性污染源”：

1. **算子归约顺序随批大小变化**：浮点数加法不满足结合律 $(a + b) + c \ne a + (b + c)$。当批次大小 `total_q` 改变时，GEMM 和 FlashMLA 内部选择的分块（Tile）大小和并行归约顺序会发生变化，导致算出的 logits 出现几个 ULP（末位有效数字）的微小抖动，一旦卡在采样阈值边缘就会分岔。
2. **变长验证受同批邻居影响**：如果开启了第 6 章的 `DEEPINFER_ENABLE_MTP_VARLEN=1`，每条请求本步分到几个草稿验证名额 `seqlens_q[i]`，取决于它的置信度在**全批次所有请求**里的排名！邻居一换，`seqlens_q[i]` 就变了。
3. **全局随机数发生器串扰**：如果整个批次共用一个全局 RNG，某条请求抽到第几个随机数取决于排在它前面的请求消费了多少个随机数。

DeepInfer 用一套三合一机制彻底锁死了这三个污染源（`README.md` L78–L92，`worker/runtime_config.py` L61–L69）：

```{.mermaid caption="图 8-1　DeepInfer 实现跨批次严格可复现（Deterministic & Batch-Invariant）的三项条件"}
flowchart LR
  subgraph Conds["锁定三大不确定性污染源"]
    A1["1. 算子级批不变（DEEPINFER_BATCH_INVARIANT=1）<br/>开启 deep_gemm / tile_kernels 确定性分块，FlashMLA 传 batch_invariant=True"]
    B1["2. 关闭跨请求预算竞争（DEEPINFER_ENABLE_MTP_VARLEN=0）<br/>每条请求固定验证 1 + nextn 个位置，不受同批邻居置信度排名影响"]
    C1["3. 每请求每位置独立随机源（SamplingParams.seq_seed）<br/>prepare_sample.cpp 按请求独立种子与采样位置派生 Gumbel 噪声"]
  end

  Out["最终效果：<br/>同一请求在任意组批方式下<br/>生成的 Token 序列 100% 相同"]

  A1 & B1 & C1 --> Out
```

---

## 8.4 读懂 `examples/benchmark.py` 的三个统计口径

在阅读或使用 `examples/benchmark.py`（及 `examples/README.md`）测试吞吐量时，有三个特殊口径必须心里有数：

1. **压测脚本报出的 Decode Throughput 是“假设草稿 100% 接受”的算子峰值吞吐**：
   公式直接按 `avg_SPS × decode_num_requests × (1 + nextn) / cp_size` 计算（即每步固定记账 `1 + 5 = 6` 个 token），衡量的是硬件每秒能验证多少个位置，而不是乘以真实接受率后的有效生成吞吐。
2. **开启 Profiler 时切勿看吞吐数字**：
   开启 PyTorch Profiler 会给密集的小算子带来高达 50% 的额外发射开销（尤其在 `bsz=1` 时），仅用于观察 Kernel 占比，不能用于测算 SPS。
3. **自动剔除 EPLB 统计同步窗口**：
   在计算稳态单步耗时时，压测脚本会自动剔除触发 MoE 负载统计跨卡 `all_gather` 的那些步，反映纯推理步的速度。

---

## 8.5 终极横评：DeepInfer vs vLLM vs SGLang

走完了第一部分的全部源码，我们现在可以站在架构师的视角，把 **DeepInfer** 与社区最主流的两个通用推理框架 **vLLM**、**SGLang** 做一个全方位对比（其中 DeepInfer 列全部经本书源码实证，vLLM 与 SGLang 列基于各自公开架构设计）：

| 架构维度 | DeepInfer（commit `9bd9cf2`） | vLLM（V1 架构） | SGLang |
| --- | --- | --- | --- |
| **核心设计目标** | **单模型极致特化**：专为 DeepSeek-V4.1-Flash 与大规模 Wide-EP 集群打造 | **高通用性生态底座**：支持上百种模型架构与广泛的硬件后端 | **通用高性能与复杂控制流**：兼顾多模型支持、Radix 缓存与结构化生成 |
| **调度器实现语言** | **纯 C++（`csrc/engine/scheduler.hpp`）**，释放 GIL，微秒级完成块管理与抢占 | Python 为主（配合部分 C++/Cython 辅助结构） | Python 为主（运行于独立调度进程/线程） |
| **Prefill 与 Decode 组批** | **严格分离，同一步绝不混跑**；多 DP 卡通过 `ALL(70)/ANY(100)` 两轮投票对齐步型 | 默认采用混合批处理（Chunked Prefill 与 Decode 拼入同一批次） | 默认分離或分块混跑，支持 PD 分离部署 |
| **KV Cache 物理架构** | **双池账本**：分页池（128 tokens/块，单 KV 头压缩）+ 定长 State 池（SWA 134 槽环形缓冲） | 统一分页池（PagedAttention，近期引入混合注意力支持） | 统一分页池 + RadixAttention 前缀树管理 |
| **跨请求前缀缓存** | 引擎内仅保留 `fork_seq` 引用计数；**持久化前缀缓存交由外部 3FS 集群按 128 粒度管理** | 内置基于块哈希的自动前缀缓存（Automatic Prefix Caching） | 内置基于基数树（Radix Tree）的自动前缀复用 |
| **异步调度机制** | **ZOS（零开销异步调度）**：双变量在途计数 + GPU 端 `remap` 与 `compute_seqlens_k_zos` | 支持异步输出处理（Async Output Processor）与多步调度 | Overlap Scheduler（CPU 调度与 GPU 前向重叠） |
| **投机解码与验证** | **原生内置 DFlash + 贪心变长验证**：并行提特征、马尔可夫头采样、按置信度卡 CUDA Graph 档位 | 外挂 Draft 模型 / N-gram / Medusa / Eagle，固定验证长度为主 | 支持 Eagle / MTP 等投机解码方案 |
| **多卡并行哲学** | **彻底抛弃 TP**：注意力走 DP / Mirror CP，MoE 走 Wide-EP + EPLB 在线热迁移冗余专家 | 以 TP / PP 为主，大模型的 MoE 层支持 EP | 支持 TP + DP Attention + DeepEP 专家并行 |

### 第一部分总结：从 DeepInfer 我们学到了什么？

合上第一部分，你会发现 DeepInfer 的每一处“特立独行”，都是模型架构与系统工程深度协同（Co-design）的结果：

- 因为模型有 **SWA + 压缩注意力 + YoCo**，所以显存管理拆成了分页池与定长环形池两套账本，后半段层还能在 Prefill 时直接跳过或截断；
- 因为模型有 **大规模 MoE 且走 Wide-EP**，所以必须抛弃 TP、严格分离 Prefill 与 Decode 步，并用两轮投票和 EPLB 冗余专家重排来消灭跨卡通信的木桶效应；
- 因为模型内置了 **DFlash 投机头与置信度预测**，所以调度器才能结合 CUDA Graph 的阶梯耗时曲线，用贪心算法把验证算力花在刀刃上。

理解了这些精妙的系统思想，从下一章（第 9 章）开始，我们将挽起袖子进入**第二部分**：用几百行纯 Python + NumPy，从零开始亲手写出一个五脏俱全的 CPU 教学引擎 `minidi`！
