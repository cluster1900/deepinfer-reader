# 第 16 章　走向 GPU：minidi 和 DeepInfer 之间还差什么

minidi 里所有的“系统决策”都能在 DeepInfer 中找到对应。但从一个 CPU 教学引擎走到生产系统，中间的大部分工作量不在这些决策上。本章按第一部分的顺序，列出每一层还差什么，以及在 DeepInfer 的哪里能看到答案。

## 16.1 差距清单

| 层 | minidi | DeepInfer | 去哪看 |
|---|---|---|---|
| 模型 | 单层单头 NumPy | MLA + 压缩 KV + 稀疏索引 + mHC + MoE + Engram | `v4_modules/`（第 7 章） |
| 注意力 kernel | 矩阵乘 + mask | FlashMLA，分页 KV 直接在 kernel 里按块表寻址 | `attention_backend.py`、FlashMLA |
| KV 格式 | float64，K/V 分开 | 单 KV head，528 B/token（e4m3 + ue8m0 scale），或 FP4 288 B | `cache_common`（第 4 章） |
| 调度器语言 | Python | C++，pybind 暴露，释放 GIL | `csrc/engine/` |
| 块分配 | 单线程 | 互斥锁同时锁 block 和 state manager | `block_manager.hpp` |
| 输入准备 | Python 列表 | C++ 在 pinned 内存里拼元数据，异步拷到 GPU | `zos.cpp`、`prepare_sample.cpp` |
| 异步 | 字典模拟 | CUDA event + GPU 侧 remap 算子 + 禁止隐式同步守卫 | `worker.py`、`utils.py`（第 5 章） |
| launch 开销 | 无 | CUDA Graph，按 token 数分档捕获 | `cuda_graph_runner.py` |
| 投机 | 加噪草稿 + 贪心验证 | MTP/DFlash 草稿 + 接受拒绝采样 + 置信度变长验证 | `sampler.py`、`mtp_varlen.hpp`（第 6 章） |
| 并行 | 无 | DP 注意力 + EP MoE（DeepEP）+ CP，跨 rank 投票 | `parallel_utils/`、`scheduler.hpp` |
| EPLB | 决策算法 | 统计窗口、周期重排、NVLink/IPC 权重搬运 | `eplb/` |

## 16.2 如果要往前走一步

按收益和难度，建议的顺序：

1. **换成 PyTorch，跑在一张普通 GPU 上**。保留 minidi 的调度器和块管理，把 `TinyLM` 换成一个小的 HF 模型，注意力先用“按块表 gather 出连续 K/V 再算”的朴素方法。参考答案仍然是不带缓存的 HF `generate`，测试照搬。
2. **把隐式同步找出来**。在 forward 和采样外面包一层 `torch.cuda.set_sync_debug_mode("error")`，像 DeepInfer 的 `forbid_cuda_sync` 一样。第一次打开时通常会发现好几处 `.item()` 和 `.tolist()`。
3. **CUDA Graph**。按 `build_graph_batch_sizes` 的方式分档捕获 decode，输入 buffer 预分配、padding 补齐。这一步之后 decode 才真正是“GPU 受限”的，ZOS 才有意义。
4. **ZOS**。把 minidi 的 `ZOSEngine` 结构搬过去，`prev_tokens` 换成 GPU 张量，remap 用 `index_select`。
5. **分页注意力 kernel**。用 Triton 或 TileLang 写一个按块表寻址的 decode 注意力，替换第 1 步的 gather。

做到第 4 步，你就有了一个结构上与 DeepInfer 引擎层同构的系统。第 5 步以后是 kernel 工程，那是另一本书的内容。

## 16.3 读源码的建议

1. 先读 `csrc/engine/scheduler.hpp`。它是整个系统的“控制平面”，1,329 行，注释大多是中文。
2. 再读 `deepinfer/worker/worker.py` 的 `execute_model_step`，理解一步里 CPU 和 GPU 怎么交错。
3. 带着具体问题读 `model_executor/`，不要从头读。比如“压缩 KV 写进了哪个 cache”“EPLB 的映射在 gate 里怎么用”。
4. 看 `tests/`。`test_cuda_sync_guards.py` 这种测试本身就说明了设计意图。

每读完一块，就回到 minidi 里找对应的那几十行，改一改，看测试哪里失败。这是本书第二部分想交给你的方法。
