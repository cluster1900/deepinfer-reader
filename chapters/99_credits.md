# 致谢与引用

- **DeepInfer**：DeepSeek-AI，`https://github.com/deepseek-ai/DeepInfer`，Apache License 2.0。本书解读的版本是 commit `8b16149741a4935e1dec8cea27853f273b374c5b`。书中引用的代码片段与文档摘录版权归原作者所有，仅用于评论和教学。
- **官方 Q&A**：`doc/DeepSeek V4.1 Flash 推理部署 Q&A.pdf`（同一 commit）。书中所有标注“官方 Q&A”的数字均来自该文档，本书未独立复现。
- DeepInfer README 致谢了 vLLM、SGLang 和 TensorRT-LLM；Q&A 提到异步调度参考了 SGLang 和 Nanoflow。
- 第二部分的 `minidi` 与全部测试、图表脚本为本书原创，以 MIT 协议发布；书稿正文以 CC BY 4.0 发布。

## 已知的不确定之处

1. `scheduler.hpp` decode 部分的头部注释（L556–L557）说“抢占 output 最短”，代码（L625 排序 + 从头部抢占）抢的是 output 最长的请求。本书按代码描述。
2. `_init_cache` 的 docstring（`llm_engine.py` L426–L427）说块数取各 rank 的最小值，代码用 all_reduce SUM 后除以 world_size，实际是平均值。本书按代码描述。
3. SWA ring 的回退余量：DeepInfer 取至少 `nextn + 1`，本书推导的下界是 `nextn`，多出的 1 个位置原因没有追到。
4. 本书没有在 SM100 或昇腾 950 上运行 DeepInfer，所有性能数字来自官方文档。
5. 第 8 章对比表中 vLLM 和 SGLang 两列只写了广为人知的设计，没有对照它们的源码核验。

如发现错误，欢迎在 `https://github.com/cluster1900/deepinfer-reader/issues` 提出。
