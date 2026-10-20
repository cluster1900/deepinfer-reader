# 第 9 章　从 0 到 1 路线图：如何用 800 行代码复刻核心机制？

读完第一部分，你脑海里已经有了一张完整的 DeepInfer 架构图。但“看懂别人的代码”和“自己能写出来”之间，往往隔着一层窗户纸：

- 当你自己写 `process_outputs` 时，稍微弄混 `num_cached` 和 `get_len()` 的加减顺序，下一个 chunk 的位置编码就会全盘错位；
- 当你自己写异步调度 `ZOSEngine` 时，如果不处理“在途请求被抢占”的边界情况，空闲块计数器就会悄悄泄漏甚至变成负数；
- 当你自己写分块 Prefill 加抢占重算时，如果无脑把被抢占的请求塞回等待队列最前面，就会一头撞上“半截 chunk 占着块等调度、队头重算请求等着块分配”的死锁！

第二部分（第 9–16 章）的目标，就是带你**亲手捅破这层窗户纸**。我们不写繁琐的 CUDA 算子，也不依赖昂贵的 GPU，而是用 **纯 Python + NumPy** 在 `minidi/` 目录下从零搭建一个不到 900 行的教学推理引擎。DeepInfer 控制平面里的每一个核心决策，我们都在 CPU 上真刀真枪地实现一遍、测试一遍。

---

## 9.1 我们的四条开发铁律

为了让这不到 900 行代码既通俗易懂，又能严丝合缝地检验我们对 DeepInfer 的理解，我们在编写 `minidi` 时严守四条铁律：

1. **黄金基准对照（Golden Reference）**：
   我们在 `minidi/engine.py` 里先写了一个只有 10 行代码、完全不用 KV Cache、每次都把整段序列从头算一遍的朴素生成函数 `reference_generate`。**无论后面我们加上分页块表、分块 Prefill、双指针抢占、异步调度（ZOS）还是投机解码，引擎最终吐出的每一个 token 都必须与 `reference_generate` 100% 逐字相等！**
2. **用 `float64` + 确定性位置种子消除浮点噪声**：
   教学引擎最怕“逻辑写错了，但以为是浮点误差”。因此 `TinyLM` 全程使用 `float64` 双精度计算，分页前向与全量前向的 logits 误差严格小于 $10^{-10}$，随机采样种子由 `(seq.seed, position)` 唯一确定。
3. **小参数放大边界冲突**：
   DeepInfer 的物理块大小是 128，滑动窗口是 128，要跑几千个 token 才会触发跨块和抢占。而在 `minidi` 中，我们把默认块大小设为 `block_size = 4`、滑动窗口设为 `W = 4`——短短十几个 token 就能频繁触发跨块分配、块表缩容、满池抢占与环形缓冲区回绕，让所有潜伏的 Bug 无处遁形。
4. **每一步都和 DeepInfer 真实行号一对一挂钩**：
   `minidi/` 下每个文件、每个函数的注释都直接标明了它对应 DeepInfer 哪个 C++/Python 文件的第几行，让你随时可以左右对照阅读。

---

## 9.2 七步通关路线图

我们将 `minidi` 的构建过程拆解为循序渐进的 7 个阶段，每个阶段对应本书的一章和一组独立的 `pytest` 单元测试：

```{.mermaid caption="图 9-1　minidi 从 0 到 1 的七个演进阶段与代码模块映射"}
flowchart LR
  subgraph Base["基础底座（第 10–12 章）"]
    S1["阶段 1（第 10 章）<br/>TinyLM 与最小 step 循环<br/>model.py · engine.py"] --> S2["阶段 2（第 11 章）<br/>PagedKV 与 BlockManager<br/>blocks.py · sequence.py"]
    S2 --> S3["阶段 3–4（第 12 章）<br/>分块 Prefill 与 LCFS 抢占<br/>scheduler.py"]
  end

  subgraph Adv["进阶性能机制（第 13–15 章）"]
    S5["阶段 5（第 13 章）<br/>零开销异步调度 ZOSEngine<br/>zos.py"]
    S6["阶段 6（第 14 章）<br/>无损投机解码与贪心变长验证<br/>engine.py · varlen.py"]
    S7["阶段 7（第 15 章）<br/>SWA 环形回退 · EPLB · 投票<br/>swa.py · eplb.py · vote.py"]
  end

  S3 --> S5 & S6 & S7
```

各阶段的代码规模与它们所复刻的 DeepInfer 原型如下表所示：

| 阶段 | `minidi/` 模块文件 | 代码行数 | 复刻的 DeepInfer 源码位置 | 对应章节 |
| --- | --- | --- | --- | --- |
| **阶段 1** | `model.py` (`TinyLM`) + `engine.py` | 93 + 134 行 | `models/deepseek.py` · `engine/llm_engine.py` | 第 10 章 |
| **阶段 2** | `blocks.py` + `sequence.py` | 61 + 81 行 | `csrc/engine/block_manager.hpp` · `sequence.hpp` | 第 11 章 |
| **阶段 3–4** | `scheduler.py` | 206 行 | `csrc/engine/scheduler.hpp`（分块、双指针抢占、缩容） | 第 12 章 |
| **阶段 5** | `zos.py` (`ZOSEngine`) | 105 行 | `worker/worker.py` · `csrc/engine/zos.cpp` | 第 13 章 |
| **阶段 6** | `varlen.py` + `engine.py` (投机分支) | 56 行 | `layers/sampler.py` · `csrc/engine/mtp_varlen.hpp` | 第 14 章 |
| **阶段 7** | `swa.py` + `eplb.py` + `vote.py` | 44 + 61 + 25 行 | `cache_engine.py` · `eplb/policy.py` · `scheduler.hpp` | 第 15 章 |

整个 `minidi/` 目录加起来只有 **869 行 Python 代码**（含详细中文注释），配套 `tests/` 下的 **64 个单元测试用例**。

---

## 9.3 先跑起来看看效果

在开始第 10 章之前，你可以先在终端运行 `examples/quickstart.py`，看看同一批 8 条并发请求在 **同步引擎**、**异步引擎（ZOS）** 和 **投机解码引擎** 下跑出来的真实表现：

```bash
$ python3 examples/quickstart.py
[同步]   一致=True  steps=36  prefill 步=11  抢占=2
[ZOS]    一致=True  steps=38
[投机]   一致=True  steps=32  draft 接受率=0.39
```

注意看这三行输出里藏着的三个细节：
1. 三种模式的 `一致=True`，证明无论是中途经历了 2 次显存抢占重算、还是超前一步盲调度的 ZOS、抑或是每次猜 3 个草稿的投机解码，它们的输出都与不带缓存的 `reference_generate` 100% 相同！
2. 为什么 `[ZOS]` 用了 38 步，比 `[同步]` 的 36 步多了 2 步？因为当请求在第 $N$ 步采到停止位时，超前调度的 ZOS 会在不知情的情况下多排一步（随后在落账时自动作废，见第 13 章）。
3. 为什么 `[投机]` 只用了 32 步？因为草稿模型猜中的 39% 的草稿帮主模型省掉了好几步自回归迭代（见第 14 章）。

准备好了吗？让我们翻开第 10 章，从第一行代码写起。
