# 前言：这本书怎么读

## 0.1 读的是哪一份代码

本书解读的是 DeepSeek 开源的推理框架 DeepInfer，仓库 `https://github.com/deepseek-ai/DeepInfer`，固定在提交 **`8b16149741a4935e1dec8cea27853f273b374c5b`**（2026-10-20，提交信息 `update Q&A documentation`）。这个仓库目前只有两个提交，上一个是 `d585e65 initial commit`。书中出现的每一个路径、行号、默认值，都是在这个提交上逐条查过的。行号写成 `scheduler.hpp L275`，指 `csrc/engine/scheduler.hpp` 第 275 行；没有写目录时，按上下文补全。

另一个证据来源是仓库自带的官方文档 `doc/DeepSeek V4.1 Flash 推理部署 Q&A.pdf`（下文简称“官方 Q&A”）。它讲了很多代码里看不到的部署事实，比如集群吞吐、KVCache 存储、负载均衡。凡是引用它的数字，书里都会标“官方 Q&A”，表示这是 DeepSeek 的说法，本书没有、也没法独立复现。

本书**没有**在 GPU 上运行 DeepInfer。它要求 NVIDIA SM100（Blackwell 一代）或华为昇腾 950，还要下载完整的 DeepSeek-V4.1-Flash 权重。所以书中关于 DeepInfer 的结论都是“读代码得出”，关于性能的数字都来自官方文档；凡是“实测”，指的都是第二部分的 CPU 教学引擎 `minidi`。

## 0.2 两部分各解决什么问题

第一部分（第 1–8 章）回答“DeepInfer 是怎么工作的”。顺序是先给全景，再跟着一个请求走完一生，然后逐个拆开调度器、显存管理、异步调度、投机解码、模型与并行，最后讲配置、运行和与 vLLM / SGLang 的关系。

第二部分（第 9–16 章）回答“一个人怎么从 0 到 1 写出一个类似的东西”。我们用纯 Python + NumPy 写一个可以在笔记本 CPU 上跑的 `minidi`，按阶段加功能：分页 KV、调度器、抢占、异步调度、投机解码与变长验证、滑动窗口 ring、专家负载均衡。每个阶段都有可运行代码和测试（仓库 `tests/` 下 63 个 pytest 用例），并指明它对应 DeepInfer 的哪段代码、简化掉了什么。

```{.mermaid caption="图 0-1　全书结构：第一部分读真实代码，第二部分照着它从零实现"}
flowchart LR
  subgraph P1["第一部分：解读 DeepInfer"]
    C1["1 定位与仓库地图"] --> C2["2 请求的一生"]
    C2 --> C3["3 调度器"]
    C3 --> C4["4 显存与 KV Cache"]
    C4 --> C5["5 异步调度 ZOS"]
    C5 --> C6["6 投机解码 DSpark"]
    C6 --> C7["7 模型、算子与并行"]
    C7 --> C8["8 配置、运行与对比"]
  end
  subgraph P2["第二部分：从 0 到 1 写 minidi"]
    D9["9 路线图"] --> D10["10 最小引擎"]
    D10 --> D11["11 分页 KV"]
    D11 --> D12["12 调度与抢占"]
    D12 --> D13["13 异步调度"]
    D13 --> D14["14 投机解码"]
    D14 --> D15["15 SWA 与 EPLB"]
    D15 --> D16["16 走向 GPU"]
  end
  P1 -.每阶段对照.-> P2
```

## 0.3 读者需要什么基础

会读 Python，大致知道 Transformer 的注意力在算什么、KV Cache 为什么能省计算，就可以读。C++ 部分只用到结构体、`std::vector`、`std::deque` 和 lambda，书里会把关键片段翻成文字。不需要 CUDA 经验，第 7 章涉及算子时只讲“它解决什么问题、在哪个文件”，不展开 kernel 写法。

## 0.4 本书怎么核对

每个结论至少满足下面一条：

1. 能指到 DeepInfer 源码的具体文件和行；
2. 来自官方 Q&A，并明确标注；
3. 在 `minidi` 上有测试或实跑脚本（`tests/`、`scripts/figures/make_plots.py`）。

有些地方读代码读不出确定结论，书里会直接写“没有追到”或“本书不下结论”，不拿猜测填空。比如第 3 章 decode 抢占的实际顺序和代码注释的说法不一致，第 15 章 SWA ring 的回退余量比我们推导的下界多 1 个位置，这两处都只陈述代码事实和推导，不猜动机。

## 0.5 动手

```bash
git clone https://github.com/cluster1900/deepinfer-reader
cd deepinfer-reader
python -m venv .venv && . .venv/bin/activate
pip install -r requirements-examples.txt
python -m pytest            # 63 个用例，CPU 上 1 秒内跑完
python examples/quickstart.py
```

`requirements-examples.txt` 只有 numpy、matplotlib、pytest 三个依赖，版本已固定。
