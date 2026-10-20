# 第 1 章　DeepInfer 是什么：定位与仓库地图

## 1.1 一句话定位

README 的第一句是：DeepInfer 是支撑 DeepSeek 大规模线上服务的高性能推理框架，这次开源的是其中的**离线推理**部分，带上了规模化服务需要的核心功能（`README.md` L3）。它目前只支持一个模型：**DeepSeek-V4.1-Flash**（L5–L7），默认权重是 Hugging Face 上的 `deepseek-ai/DeepSeek-V4.1-Flash-1017`。

这和 vLLM、SGLang 的定位很不一样。后两者是通用引擎，要支持几百种模型和多种硬件；DeepInfer 是为一个模型、一套部署形态打磨出来的生产引擎，开源时把与内部基础设施耦合的部分拿掉了。读它的价值不在于“拿来就用”，而在于看一个超大规模 MoE 模型在生产中到底需要哪些机制。

## 1.2 README 列出的功能

`README.md` L9–L29 列出的功能，以及本书讲解它的章节：

| 功能（README 原文） | 含义 | 本书章节 |
| --- | --- | --- |
| Optimized kernels, model and sampler runtime | FlashMLA、DeepGEMM、DeepEP、TileKernels、DeepSelect 加上自写 TileLang 算子 | 第 7 章 |
| Parallelism: Attention DP / balanced CP, MoE EP + EPLB | 注意力数据并行、上下文并行，MoE 专家并行与负载均衡 | 第 7 章 |
| DSpark and confidence-scheduled verification | 草稿模型投机解码 + 按置信度决定每个请求验证几个 token | 第 6 章 |
| SWA bounded replay | 滑动窗口层只重算有限长度的后缀 | 第 4、6 章 |
| Asynchronous scheduling (zero-overhead batch scheduler) | CPU 调度与 GPU 执行重叠 | 第 5 章 |
| Batch-invariant inference | 同一请求的结果不随批次组成改变（仅 NVIDIA） | 第 8 章 |
| Deterministic sampling | 相同输入和采样参数得到相同输出（仅 NVIDIA） | 第 8 章 |
| Structured outputs | regex / JSON schema / EBNF 约束生成 | 第 2、8 章 |

支持的硬件只有两种：**NVIDIA SM100 GPU** 和**华为昇腾 950 NPU**（L33–L36）。构建要求 Linux、GCC ≥ 13、Python 3.12，NVIDIA 需要 CUDA 13，昇腾需要 CANN 9.2（L46–L52）。

## 1.3 开源版里没有什么

README L38–L42 明确写了，构成一个完整推理服务所需、但**这次没有放出**的三样东西：

1. KV cache 系统（跨请求复用前缀的持久化缓存）；
2. VE / Prefill / Decode 分离部署（“大部分所需的引擎功能已经实现”）；
3. 数据并行副本之间的负载均衡（prefill 和 decode 都需要）。

官方 Q&A 的说法一致：“推理系统中的 KVCache 管理与负载均衡组件因与内部基础设施耦合较深，相关代码目前尚未完整开源”。所以读代码时会看到不少“虚函数”占位，比如 `LLMEngine.init_pd_split`、`do_save_kvcache`、`send_processed_prompts`（`deepinfer/engine/llm_engine.py` L801–L858），在开源版里函数体是空的或只有一行。这些钩子告诉我们生产版在哪里接入了 PD 分离和 KV 落盘，第 2 章会把它们标在流程图上。

## 1.4 仓库地图

```{.mermaid caption="图 1-1　DeepInfer 仓库的主要目录（commit 8b16149）"}
flowchart TB
  root["DeepInfer/"]
  root --> csrc["csrc/  C++ 引擎核心<br/>scheduler / block_manager / sequence / zos / mtp_varlen"]
  root --> pkg["deepinfer/  Python 包"]
  root --> dc["deep_cluster/  硬件与拓扑探测、NUMA 绑定"]
  root --> vendor["vendor/  6 个子模块<br/>FlashMLA DeepGEMM DeepEP DeepSelect<br/>DeepEP-Ascend DeepGEMM-Ascend"]
  root --> ex["examples/  离线推理、benchmark、sweep"]
  root --> ev["eval/  AIME / HLE / GPQA 评测"]
  root --> ts["tests/"]
  pkg --> ent["entrypoints/llm.py  LLM 类"]
  pkg --> eng["engine/  LLMEngine、参数、统计"]
  pkg --> wk["worker/  Worker、ModelRunner、CacheEngine、CUDA Graph"]
  pkg --> me["model_executor/"]
  me --> models["models/  deepseek.py、decoder_layer.py、v4_modules/"]
  me --> layers["layers/  sampler、linear、tilelang_ops/（cuda 与 asc 两套）"]
  me --> par["parallel_utils/  进程组、CP、userbuffer"]
  me --> eplb["eplb/  专家负载均衡"]
  pkg --> so["structure_output/  xgrammar 约束"]
  pkg --> ve["ve/  视觉编码与多模态"]
```

一个最直观的事实：调度器、块管理器、请求状态这些“引擎骨架”是用 **C++** 写的（`csrc/engine/`，通过 pybind11 暴露为 `deepinfer._C.engine`），Python 侧的 `LLMEngine` 主要负责串流程。这和 vLLM 的 Python 调度器不同，原因在第 5 章讲：当 decode 一步只有十几毫秒时，Python 调度本身就会成为瓶颈。

六个 vendor 子模块在这个提交上固定的版本（`git submodule status`）：

| 子模块 | 提交 | 作用 |
| --- | --- | --- |
| FlashMLA | `2e5429f` | MLA 注意力 kernel |
| DeepGEMM | `1fc9c66` | FP8/FP4 GEMM、MoE 分组 GEMM |
| DeepEP | `93eb6eb` | MoE 专家并行通信（dispatch / combine），也提供 CP 用的 BucketBuffer |
| DeepSelect | `bfa4507` | 稀疏注意力的 top-k 选择 |
| DeepEP-Ascend | `3b25377` | 昇腾版通信 |
| DeepGEMM-Ascend | `3a35326` | 昇腾版 GEMM |

## 1.5 代码量分布

![图 1-2　DeepInfer 各部分代码行数（统计脚本见 data/deepinfer_loc.json，不含 vendor 子模块）](images/plots/ch01_loc.png)

全部 `.py/.cpp/.hpp` 共 60,838 行。最大的一块是 `layers/tilelang_ops/` 的 23,331 行，里面每个算子都有 `cuda/` 和 `asc/` 两份实现，外加一个分发用的同名文件；这部分是“同一份逻辑写两遍硬件版本”的成本。真正的调度骨架 `csrc/` 只有 5,909 行，`engine/` 1,957 行，`worker/` 2,805 行。也就是说，想读懂引擎怎么转，要啃的核心代码大约 1 万行，其余是模型和算子。

## 1.6 它服务的系统长什么样

开源代码只是单个推理实例。官方 Q&A 描述了它所在的整个系统，这能解释代码里很多看似多余的设计：

```{.mermaid caption="图 1-3　官方 Q&A 描述的 E/P/D 分离部署（虚线框是本次未开源的部分）"}
flowchart LR
  U["用户请求"] --> R["DeepSeek Recipe<br/>API 协议 ↔ token"]
  R --> LB["Prefill / Decode / VE<br/>负载均衡器"]
  LB --> VE["Vision Encoder 实例<br/>单卡"]
  LB --> PE["Prefill 实例<br/>EP2~8 / DP2~8"]
  PE -- "KV + SWA state" --> CE["Decode 实例<br/>EP64~128 / DP64~128"]
  VE --> PE
  PE <--> KV[("KVCache 存储<br/>3FS 集群")]
  CE --> KV
  style LB stroke-dasharray: 5 5
  style KV stroke-dasharray: 5 5
  style R stroke-dasharray: 5 5
```

几条与代码直接相关的事实（均来自官方 Q&A）：

- Prefill 用 EP2~8 / DP2~8，“在显存足够的情况下用尽可能小的 EP/DP”；Decode 用 EP64~128 / DP64~128（Wide-EP）；“暂未用 TP，CP”。代码里因此只有 DP、EP、CP 三种并行，没有张量并行（第 7 章）。
- 注意力部分 DP 数等于 EP 数（`#DP=#EP`）。这对应 `ParallelConfig` 里 `data_parallel_size` 缺省时取 `expert_parallel_size // context_parallel_size`（`deepinfer/config.py` L161–L162）。
- 代码里的 `EngineType` 有三种：`NORMAL`（prefill + decode 都做）、`PROMPT`（只做 prefill）、`COMPLETION`（只做 decode）（`deepinfer/engine/engine_type.py`）。开源版的离线推理用 `NORMAL`；`PROMPT`/`COMPLETION` 就是 PD 分离里的 PE 和 CE，官方 Q&A 也用 PE/CE 这两个缩写。
- 官方给出的线上数字：单一集群吞吐 155 万亿 tokens/天，每卡每小时收入 38 美元（官方 Q&A 原话注明“按谷价”）。这些数字本书无法核实，只作为背景。

## 1.7 和 vLLM 的血缘

`deepinfer/` 下有 21 个 Python 文件开头写着 `# Derived from vLLM; modified by DeepSeek.`，例如 `entrypoints/llm.py`、`engine/llm_engine.py`、`sampling_params.py`。README 的致谢段也说 DeepInfer “最初主要基于 vLLM 构建，并参考了 SGLang 和 TensorRT-LLM”（L95）。所以你会看到熟悉的名字：`LLM.generate`、`SamplingParams`、`LLMEngine.step`、`gpu_memory_utilization`、`profile_num_available_blocks`。但名字之下的实现已经大幅改写：调度器搬进了 C++，prefill 和 decode 不在同一步混跑，KV 分成分页块和每请求定长 state 两套。第 8 章会把这些差异集中列一张表。
