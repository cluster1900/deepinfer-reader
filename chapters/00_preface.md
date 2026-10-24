# 前言：为什么要读 DeepInfer，以及这本书怎么读

如果你写过或者读过 LLM 推理引擎的代码，大概率经历过这样的困惑：

- 看论文时，PagedAttention、投机解码（Speculative Decoding）、异步调度（Zero-Overhead Scheduling）、MoE 专家并行（EP）各自都很清晰；
- 可一旦把它们塞进**同一个生产级引擎**里，所有的边界条件就会撞在一起：CPU 在调度第 $N+1$ 步时，第 $N$ 步在 GPU 上到底接受了几个草稿 token 还不知道，KV 块怎么预留？滑动窗口注意力（SWA）只保留最近 128 个位置，投机解码万一拒绝了 4 个草稿 token，被覆盖的窗口怎么回退？多卡 DP 跑 MoE 时，有的卡想做 prefill、有的卡想做 decode，集合通信怎么对齐？

DeepSeek 开源的 **DeepInfer** 正好是一份把这些问题全部摊在桌面上的实战答卷。与追求“支持几百种模型”的通用框架不同，DeepInfer 只服务一个目标——把 **DeepSeek-V4.1-Flash** 这个混合注意力 + 大规模 MoE + 内置 DFlash 投机解码的模型跑出极致吞吐。代码总量约 6 万行，其中真正的 C++/Python 引擎调度骨架只有 **1 万行左右**，非常适合从头到尾读透。

---

## 0.1 本书怎么组织：先拆真引擎，再亲手造一个

为了让源码阅读既不沦为“枯燥的行号流水账”，也不变成“脱离代码的空谈”，本书采用**“左手指源码、右手写引擎”**的双线结构：

- **第一部分（第 1–8 章）：拆解真实的 DeepInfer**。我们先用一张全景图看清 DeepInfer 的定位与生产部署形态（第 1 章），再跟着一次请求走完从入队到吐出 token 的一生（第 2 章），随后逐一攻克五个核心子系统：**C++ 调度器**（第 3 章）、**混合 KV Cache 与 SWA 状态池**（第 4 章）、**零开销异步调度 ZOS**（第 5 章）、**DFlash 投机解码与置信度变长验证**（第 6 章）、**YoCo 双段架构与 DP/CP/EP 并行**（第 7 章），最后总结配置开关与框架对比（第 8 章）。
- **第二部分（第 9–16 章）：从 0 到 1 实现教学引擎 `minidi`**。光看懂别人的代码还不够，我们在 `minidi/` 目录下用 **约 750 行纯 Python + NumPy**，按 7 个阶段亲手搭出一个能在笔记本 CPU 上运行的微型推理引擎。第一部分里读到的每一条硬核机制——块表分配、双指针抢占、乐观推进与回填、ZOS 在途位置修正、贪心变长验证、SWA 环形缓冲区回退、EPLB 专家复制与放置——都在 `minidi` 里有对应的可运行实现和单元测试（共 64 个 `pytest` 用例，1 秒内跑完）。

```{.mermaid caption="图 0-1　全书阅读地图：第一部分拆解 DeepInfer 源码，第二部分在 minidi 中一对一复现"}
flowchart TB
  subgraph P1["第一部分：解读 DeepInfer 生产源码（第 1–8 章）"]
    direction LR
    C1["第 1–2 章<br/>全局定位与请求生命周期"] --> C2["第 3–4 章<br/>C++ 调度器与双池显存"]
    C2 --> C3["第 5–6 章<br/>ZOS 异步与 DFlash 投机"]
    C3 --> C4["第 7–8 章<br/>YoCo 并行、EPLB 与横评"]
  end

  subgraph P2["第二部分：从 0 到 1 手写 CPU 教学引擎 minidi（第 9–16 章）"]
    direction LR
    D1["第 9–10 章<br/>路线图与最小 step 引擎"] --> D2["第 11–12 章<br/>分页 KV 与双指针抢占"]
    D2 --> D3["第 13–14 章<br/>ZOSEngine 与变长验证"]
    D3 --> D4["第 15–16 章<br/>SWA / EPLB 与走向 GPU"]
  end

  P1 -. "机制与行号一一对照" .-> P2
```

---

## 0.2 每一章的叙述方式

为了符合人类的阅读直觉，本书每一章都遵循四步展开法：

1. **先问“为什么”**：如果不这么设计，最朴素的做法会在什么场景下崩溃、变慢或浪费显存？
2. **用小数字推演**：在看代码之前，先用 `block_size = 4`、窗口大小 `W = 4`、草稿数 `k = 3` 这样一目了然的小数字，把指针移动和张量形状算清楚。
3. **对照真实源码**：带着算清楚的直觉去看 DeepInfer 对应的 C++ / Python 函数，每一处分支、每一个常量都能和前面的推演严丝合缝地对上。
4. **动手验证与边界分析**：在 `minidi` 中跑通对应逻辑，探讨极端负载下的边界情况与工程取舍。

---

## 0.3 版本基准与证据约定

本书解读的 DeepInfer 源码固定在官方仓库 `https://github.com/deepseek-ai/DeepInfer` 的提交 **`9bd9cf243fc53ef9afbf8f04d391d989117d6ab9`**（2026-10-21，`Bump DeepEP to f6aa38fe and drop the SSH->HTTPS rewrite (#157)`）。书中所有文件路径、函数名、行号和默认值均在此提交上逐行核对。

- **行号标记**：形如 `scheduler.hpp L275` 指 `csrc/engine/scheduler.hpp` 第 275 行；省略目录前缀时按当前章节上下文补全。
- **官方 Q&A**：仓库自带的 `doc/inference_deployment_qa.md`（书中简称“官方 Q&A”；上游在 `8b16149` 以 PDF `doc/DeepSeek V4.1 Flash 推理部署 Q&A.pdf` 首发，`774ce24` 改为英文 Markdown 并删除 PDF，同时新增 `doc/images/architecture_overview.png`、`wide_ep_decode_pareto.png`）披露了大量线上生产部署细节（如 E/P/D 分离配比、3FS 分布式 KV 存储、128 粒度落盘原因等）。凡引用自该文档的数据与事实，书中均明确标注“官方 Q&A”。
- **实验边界**：DeepInfer 生产运行依赖 NVIDIA SM100（Blackwell）或华为昇腾 950 硬件及完整模型权重。本书对 DeepInfer 本身的分析全部基于静态源码与官方 Q&A 交叉验证；书中所有实验曲线与数据图则由第二部分的 `minidi` 引擎在 CPU 上真实运行生成。

---

## 0.4 三分钟跑通本书配套代码

阅读第一部分或第二部分时，建议随时在终端里跑一跑 `minidi` 的测试和演示脚本：

```bash
git clone https://github.com/cluster1900/deepinfer-reader
cd deepinfer-reader
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-examples.txt

# 1. 运行全部 64 个单元测试（验证分页 KV、调度抢占、ZOS、投机解码、SWA、EPLB）
python3 -m pytest

# 2. 运行快速演示脚本，直观看到各模块输出
python3 examples/quickstart.py
```

`requirements-examples.txt` 只包含 `numpy`、`matplotlib`、`pytest` 三个轻量依赖，在任何笔记本 CPU 上都能在 1 秒内跑完测试。现在，让我们翻开第 1 章，先看清 DeepInfer 的全局面貌。
