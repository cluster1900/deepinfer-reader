# 第 9 章　路线图：一个人怎么从 0 到 1

第一部分读完，你大概会觉得 DeepInfer 太大了：6 万行代码、两种硬件、几十个 kernel。第二部分的目标不是复刻它，而是**把它的每一个系统设计决策，用最少的代码在 CPU 上做一遍**。做完以后再回去读源码，每个 `if` 为什么在那里会清楚得多。

## 9.1 原则

1. **先正确，再快**。每一阶段都有一个“参考答案”：不带缓存、一次一个请求的朴素生成 `reference_generate`（`minidi/engine.py`）。引擎不管怎么调度、抢占、异步，输出必须与参考逐 token 一致。这是整个第二部分最重要的一条测试。
2. **一次只加一个机制**。分页 KV、调度、抢占、异步、投机，每加一个都先让全部测试通过再继续。
3. **对照真实代码**。每个模块的 docstring 都写着它对应 DeepInfer 的哪个文件、哪一行，简化了什么。
4. **用 float64 和固定种子**。浮点误差会掩盖逻辑错误。`TinyLM` 用 float64 计算，分页和不分页的 logits 可以做到 `allclose`，token 可以逐个相等。

## 9.2 阶段

```{.mermaid caption="图 9-1　minidi 的七个阶段与对应测试"}
flowchart TB
  S1["阶段 1　TinyLM + 参考生成<br/>model.py"] --> S2["阶段 2　分页 KV + 块管理<br/>model.py PagedKV, blocks.py<br/>test_model_blocks.py"]
  S2 --> S3["阶段 3　调度器：prefill/decode 分离、chunk<br/>scheduler.py<br/>test_scheduler.py"]
  S3 --> S4["阶段 4　抢占与重算<br/>scheduler.py _preempt<br/>test_engine.py（含抢占）"]
  S4 --> S5["阶段 5　异步调度<br/>zos.py<br/>test_zos_matches_sync"]
  S4 --> S6["阶段 6　投机解码 + 变长验证<br/>engine.py nextn, varlen.py"]
  S4 --> S7["阶段 7　SWA ring、EPLB、投票<br/>swa.py eplb.py vote.py<br/>test_algorithms.py"]
```

| 阶段 | 代码 | 行数 | 对应 DeepInfer | 章节 |
|---|---|---|---|---|
| 1 | `model.py` TinyLM | 93（含 PagedKV） | `models/deepseek.py` | 10 |
| 2 | `model.py` PagedKV、`blocks.py`、`sequence.py` | 61 + 81 | `block_manager.hpp`、`sequence.hpp` | 11 |
| 3–4 | `scheduler.py`、`engine.py` | 203 + 134 | `scheduler.hpp`、`llm_engine.py` | 10、12 |
| 5 | `zos.py` | 105 | `worker.py`、`zos.cpp` | 13 |
| 6 | `engine.py` 投机部分、`varlen.py` | 56 | `sampler.py`、`mtp_varlen.hpp` | 14 |
| 7 | `swa.py`、`eplb.py`、`vote.py` | 44 + 61 + 25 | `cache_engine.py`、`eplb/policy.py`、`scheduler.hpp` | 15 |

`minidi/` 一共 867 行（含注释和 docstring），加上 `tests/` 的 63 个用例。

## 9.3 有意不做的事

- **不写 kernel**。注意力就是 NumPy 矩阵乘。第 16 章讨论走向 GPU 时哪些地方要换成真正的 kernel。
- **不做多进程**。DP 投票和 EPLB 只实现决策算法，不实现通信。
- **不做前缀缓存**。DeepInfer 开源版也没有（第 4 章）。
- **不做结构化输出、多模态**。它们与本书关注的调度和内存关系不大。

## 9.4 环境

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements-examples.txt
python -m pytest
python examples/quickstart.py
```

`quickstart.py` 用同一批 8 个请求分别跑同步引擎、异步引擎和投机解码，打印是否与参考一致、用了多少步、抢占几次、draft 接受率。在本书写作环境（Python 3.12、numpy 2.3.4）的输出是：

```text
[同步]   一致=True  steps=36  prefill 步=11  抢占=2
[ZOS]    一致=True  steps=38
[投机]   一致=True  steps=32  draft 接受率=0.39
```

ZOS 比同步多 2 步，原因是它在不知道请求已结束时多排了步（第 13 章）。
