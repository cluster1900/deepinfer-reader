# 第 1 章　DeepInfer 是什么：定位、架构与仓库地图

在一头扎进 C++ 调度器或 CUDA/TileLang 算子之前，我们先回答三个最基本的问题：

1. **DeepInfer 和我们熟悉的 vLLM、SGLang 到底有什么本质区别？**
2. **DeepSeek 线上真实的推理集群长什么样，这次开源放出了哪一部分？**
3. **6 万行源码里，哪 1 万行是决定引擎怎么转的核心骨架？**

---

## 1.1 一句话定位：为一个模型量身定做的生产引擎

打开 DeepInfer 的 `README.md`（L3–L7），第一段话就把它的性格交代得明明白白：

> DeepInfer 是支撑 DeepSeek 大规模线上服务的高性能推理框架；本次开源的是其中的**离线推理引擎**，同时保留了大规模服务所需的核心运行时特性。它目前只支持一个模型家族：**DeepSeek-V4.1-Flash**（默认权重 `deepseek-ai/DeepSeek-V4.1-Flash-1017`）。

这一定位决定了它的代码长相与 vLLM、SGLang 截然不同：

- **vLLM / SGLang 是“通用百货商场”**：它们必须兼容上百种模型架构和五花八门的硬件，因此需要庞大的抽象层与插件体系。
- **DeepInfer 是“F1 赛车”**：它只为 **DeepSeek-V4.1-Flash** 这一款模型服务。模型有哪些奇特结构（比如前一半层配滑动窗口注意力 SWA + 门控线性注意力 GLA、后一半层配共享 KV 的 MLA、内置 DFlash 投机草稿头、大规模 MoE 专家并行），引擎的块管理器、调度器和算子就精确地长成什么形状，没有任何多余的抽象层。

因此，读 DeepInfer 的最大价值，不是把它当成一个通用工具箱去跑别的模型，而是**透过它看清：当模型架构演进到混合注意力 + 原生投机解码 + 超大规模 MoE 时，推理引擎必须做出怎样脱胎换骨的配合**。

---

## 1.2 核心特性一览与全书导引

`README.md` L9–L29 列出了开源版的八大核心特性。把英文术语翻译成大白话，它们各自解决的痛点和对应章节如下：

| 核心特性（README 原文） | 它到底解决什么痛点？ | 详解章节 |
| --- | --- | --- |
| **Optimized kernels, model & sampler runtime** | 集成 FlashMLA、DeepGEMM、DeepEP、DeepSelect 与自研 TileLang 算子，榨干 SM100 / 昇腾 950 算力 | 第 7 章 |
| **Parallelism: Attention DP / balanced CP, MoE EP + EPLB** | 放弃张量并行（TP），注意力层走 DP 或镜像对称 CP，MoE 层走 EP 并用 EPLB 复制热门专家防木桶效应 | 第 7、15 章 |
| **DSpark (DFlash) & confidence-scheduled verification** | 单步并行生成 5 个草稿 token，按马尔可夫头与置信度头动态决定每条请求本步验证几个草稿 | 第 6、14 章 |
| **SWA bounded replay** | 滑动窗口层不存全量历史 KV，只维护定长环形缓冲区（134 槽），长序列显存封顶 | 第 4、15 章 |
| **Asynchronous scheduling (ZOS)** | CPU 提前调度第 $N+1$ 步并与 GPU 第 $N$ 步重叠执行，消除步间气泡 | 第 5、13 章 |
| **Batch-invariant inference & Deterministic sampling** | 用固定分块归约算子与每请求独立 Gumbel 随机数，保证同一请求无论怎么凑批输出完全一致（仅 NVIDIA） | 第 8 章 |
| **Structured outputs** | 异步编译正则表达式 / JSON Schema / EBNF 语法自动机，解码时生成词表掩码强制合法输出 | 第 2、8 章 |

目标硬件非常聚焦：仅支持 **NVIDIA SM100（Blackwell 架构）** 与 **华为昇腾 950 NPU**（`README.md` L33–L36），构建要求 Linux、GCC ≥ 13、Python 3.12，以及 CUDA 13 或 CANN 9.2。

---

## 1.3 线上真实系统 vs 本次开源范围

阅读 DeepInfer 源码时，你经常会在 `LLMEngine` 里看到一些函数体为空的钩子方法，比如 `init_pd_split()`、`do_save_kvcache()`、`send_processed_prompts()`（`deepinfer/engine/llm_engine.py` L801–L858）。为什么会有这些“空壳”？

答案在 `README.md` L38–L42 和**官方 Q&A** 里：线上完整系统是一套分布式集群，而本次开源移除了与 DeepSeek 内部基础设施（如 3FS 分布式文件系统、内部 RPC 与全局网关）强耦合的三个外部组件：

1. **跨请求持久化的分布式 KVCache 存储系统**（基于 3FS 集群，以 128 个 token 为粒度保存和检索前缀 KV 与 SWA 状态）；
2. **VE（视觉编码）/ Prefill / Decode 分离部署的跨实例传输层**（但单实例内的 `PROMPT` / `COMPLETION` 引擎角色逻辑已完整保留在代码中）；
3. **跨数据并行（DP）副本的全局请求路由与负载均衡器**。

把官方 Q&A 描述的线上全貌画出来，就能立刻看清当前开源代码在整座大厦里的位置：

```{.mermaid caption="图 1-1　DeepSeek 线上 E/P/D 分离部署全貌（虚线框为未开源的外部基础设施，实线框为 DeepInfer 引擎实例）"}
flowchart TB
  subgraph Infra["1. 外部接入与全局路由层（本次未开源）"]
    direction LR
    U["用户请求"] --> R["DeepSeek Recipe<br/>API 协议转换与分词"] --> LB["全局负载均衡器<br/>VE / Prefill / Decode 路由"]
  end

  subgraph Cluster["2. DeepInfer 推理实例集群（核心引擎已开源）"]
    direction LR
    VE["视觉编码实例 (VE)<br/>单卡独立部署"] -- "视觉特征" --> PE["Prefill 实例 (PROMPT)<br/>小并行：EP 2~8 / DP 2~8"] -- "首 token + 状态" --> CE["Decode 实例 (COMPLETION)<br/>大并行 Wide-EP：64~128"]
  end

  subgraph Storage["3. 分布式缓存底座（本次未开源）"]
    direction LR
    KV1["3FS 分布式 KVCache 存储集群"] --- KV2["按 128 tokens 粒度供 PE 写入、CE 读取前缀 KV 与 SWA 快照"]
  end

  Infra --> Cluster --> Storage

  style R stroke-dasharray: 5 5
  style LB stroke-dasharray: 5 5
  style KV1 stroke-dasharray: 5 5
  style KV2 stroke-dasharray: 5 5
```

理解这张图，就能解开源码里的三个设计之谜（均据官方 Q&A 与源码交叉印证）：

- **为什么代码里没有张量并行（TP）？** 官方 Q&A 指出，DeepSeek-V4.1-Flash 的 MLA 共享 KV 与紧凑权重设计使单卡足以放下一份注意力与专家分片。线上 Prefill 实例采用小规模 `EP 2~8 / DP 2~8` 降低通信开销；Decode 实例采用 `EP 64~128 / DP 64~128`（Wide-EP）凑大全局批大小以喂饱 MoE 专家。注意力层满足 `#DP = #EP`（未开启 CP 时），完全不需要切分单层权重的 TP。
- **为什么有三种 `EngineType`？** `deepinfer/engine/engine_type.py` 定义了 `NORMAL`（既做 prefill 又做 decode，用于离线推理和单实例测试）、`PROMPT`（对应线上 PE 实例，只跑 prefill）和 `COMPLETION`（对应线上 CE 实例，只跑 decode）。
- **为什么会有 128 粒度的状态拷贝钩子？** 线上 PE 与 CE 共享基于 3FS 的 KVCache 存储，为了兼顾 I/O 效率与命中率，KV 与 SWA 状态都按 **128 个 token**（`KVSTORE_GRAIN = 128`）对齐落盘。这直接影响了第 4 章将要讲到的 SWA 环形缓冲区大小。

---

## 1.4 仓库目录地图：去哪里找哪段逻辑？

DeepInfer 仓库结构清晰，分为 **C++ 引擎核心（`csrc/`）**、**Python 运行时与模型层（`deepinfer/`）**、**硬件拓扑工具（`deep_cluster/`）** 和 **外部高性能算子库（`vendor/`）** 四大块：

```{.mermaid caption="图 1-2　DeepInfer 仓库目录地图（commit 8b16149）"}
flowchart LR
  subgraph Core["1. 引擎与运行时骨架"]
    csrc["csrc/engine/（C++ 调度核心）<br/>scheduler · block_manager<br/>sequence · zos · mtp_varlen"]
    eng["deepinfer/entrypoints & engine<br/>LLM 入口 · LLMEngine 主循环"]
    wk["deepinfer/worker/<br/>Worker · ModelRunner · CacheEngine"]
  end

  subgraph ModelExec["2. 模型、并行与算子层"]
    models["models/<br/>deepseek.py · v4_modules/<br/>mtp/dflash.py（DFlash 草稿）"]
    layers["layers/<br/>sampler.py · linear.py<br/>tilelang_ops/（cuda 与 asc）"]
    par["parallel_utils/ & eplb/<br/>DP / CP / EP · 专家负载均衡"]
  end

  subgraph Support["3. 子模块与外围组件"]
    vendor["vendor/（6 个算子通信子模块）<br/>FlashMLA · DeepGEMM<br/>DeepEP · DeepSelect"]
    aux["structure_output/（xgrammar）<br/>ve/（多模态）· deep_cluster/"]
    ex["examples/ · eval/ · tests/<br/>离线推理、压测与评测集"]
  end

  csrc --> models
  eng --> layers
  wk --> par
  models --> vendor
  layers --> aux
  par --> ex
```

一个最值得注意的架构决策是：**调度器（`Scheduler`）、块管理器（`BlockManager`）、序列状态（`Sequence`）、异步调度输入修正（`zos.cpp`）和变长验证预算分配（`mtp_varlen.hpp`）全部用 C++ 编写**，通过 `pybind11` 编译为 `deepinfer._C.engine` 供 Python 调用。为什么不用 Python 写调度器？因为当 GPU 跑一步 decode 只需几毫秒到十几毫秒时，Python 解释器遍历几百个请求对象、维护块表的开销就会超过 GPU 计算本身。

六个 `vendor/` 子模块在当前提交上锁定的版本与职责如下：

| 子模块 | 锁定提交 | 在 DeepInfer 中的角色 |
| --- | --- | --- |
| `FlashMLA` | `2e5429f` | 多头潜在注意力（MLA）与稀疏 Top-$k$ 注意力的高性能 CUDA kernel |
| `DeepGEMM` / `DeepGEMM-Ascend` | `1fc9c66` / `3a35326` | FP8 / BF16 矩阵乘与 MoE 分组 GEMM（NVIDIA 与昇腾双后端） |
| `DeepEP` / `DeepEP-Ascend` | `93eb6eb` / `3b25377` | MoE 跨卡 token 分发（dispatch）与合并（combine），兼供 CP 的 `BucketBuffer` |
| `DeepSelect` | `bfa4507` | 稀疏注意力（DSA / NSA）中的快速 Top-$k$ 块选择算子 |

---

## 1.5 代码量分布：抓住核心的 1 万行

如果我们统计仓库内所有 `.py`、`.cpp`、`.hpp` 文件（不含 `vendor/` 子模块），总行数为 **60,838 行**：

![图 1-3　DeepInfer 各模块代码行数统计（数据见 data/deepinfer_loc.json，不含 vendor 子模块）](images/plots/ch01_loc.png)

看图 1-3 能立刻消除阅读大型代码库的畏惧感：

- 占比最大的 `layers/tilelang_ops/`（**23,331 行**，占全仓近 40%）是 TileLang 自定义算子库，因为要同时支持 NVIDIA CUDA（`cuda/`）和昇腾 Ascend（`asc/`），每个算子都写了两套硬件实现外加一层路由包装。
- 真正决定推理引擎怎么调度的核心代码——`csrc/`（**5,909 行**）、`engine/`（**1,957 行**）和 `worker/`（**2,805 行**）——加起来只有 **10,671 行**！

换句话说，只要读懂这 1 万行核心骨架，你就能彻底掌握 DeepInfer 的精髓。

---

## 1.6 与 vLLM 的血缘关系

在 `deepinfer/` 目录下，有 21 个 Python 文件顶部保留着一行声明：`# Derived from vLLM; modified by DeepSeek.`（例如 `entrypoints/llm.py`、`engine/llm_engine.py`、`sampling_params.py`）。`README.md` 的致谢部分也坦诚说明：DeepInfer 最初基于早期 vLLM 演化而来，并参考了 SGLang 与 TensorRT-LLM 的设计。

因此，你在阅读时会看到很多亲切的老面孔：`LLM.generate`、`SamplingParams`、`LLMEngine.step`、`gpu_memory_utilization`、`profile_num_available_blocks`。但在这些熟悉的外壳之下，内部引擎已经被彻底重造：

1. **调度器下沉到 C++**，且同一个批次严格区分 prefill 步与 decode 步，绝不混跑；
2. **显存池一分为二**：全注意力层用分页块表（Paged KV），而滑动窗口层（SWA）、键值对压缩层与 DFlash 草稿层则使用每请求定长的环形状态槽（State Pool）；
3. **原生深度集成 DFlash 投机解码与零开销异步调度（ZOS）**，连 GPU 端修正 `seqlens_k` 的算子都是专门定制的。

下一章，我们就顺着 `llm.generate(...)` 的调用链，陪一个请求走完它在 DeepInfer 里的一生。
