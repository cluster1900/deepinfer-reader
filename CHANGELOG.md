# 更新日志

## v0.1.0（2026-10-20）

首个完整版本。

- **第一部分：解读 DeepInfer**（第 1–8 章），对照 commit `8b16149741a4935e1dec8cea27853f273b374c5b` 逐条核对路径与行号：
  仓库地图与代码规模；请求生命周期；C++ 调度器（prefill/decode 分离、跨 rank ALL/ANY 投票、FCFS、按输出长度抢占）；
  显存与 KV Cache（块数计算、0.8 容量闸门、528 B/token 的 V41 格式、SWA 状态槽）；异步调度 ZOS 与 CUDA Graph；
  DSpark 置信度变长验证；模型模块角色、算子库、DP/EP/CP 并行与 EPLB；配置、运行、评测、批不变与 vLLM/SGLang 对比。
- **第二部分：个人从 0 到 1**（第 9–16 章）：CPU 教学引擎 `minidi`（867 行）与 63 个 pytest 用例，
  覆盖分页 KV、调度与抢占、异步调度、投机解码与变长验证、SWA 环形缓冲、EPLB、步型投票；记录了开发中遇到的三个真实死锁/计数错误及修法。
- 28 张图：21 张 mermaid 结构图，7 张由 `scripts/figures/make_plots.py` 从 `minidi` 实跑生成的数据图。
- 输出网页、EPUB、PDF；CI 门禁检查章节结构、图片、疑似密钥、固定 commit，并运行全部测试与示例。
- 已知的不确定之处列在书末“致谢与引用”。
