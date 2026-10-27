# 致谢、引用与源码勘误记录

## 1. 致谢与版本声明

- **DeepInfer 开源项目**：由 DeepSeek-AI 团队开源（仓库地址：`https://github.com/deepseek-ai/DeepInfer`，Apache License 2.0 协议）。本书所有源码路径、函数名、行号及默认配置均严格锁定并核对自提交 **`553741557a57df44a46fb3ab72b6aafeaf945337`**（2026-10-25，`fix: validate boolean environment variables with strict parsing (#218)`）。书中引用的源码片段与文档摘录版权归原作者所有，仅作技术解析与教学之用。
- **官方技术问答文档（官方 Q&A）**：即仓库内附带的 `doc/inference_deployment_qa.md`（`774ce24` 起取代原 PDF 版 `doc/DeepSeek V4.1 Flash 推理部署 Q&A.pdf`）。书中凡涉及线上 E/P/D 分离配比、3FS 分布式 KVCache 存储、线上吞吐与 EPLB 实测指标等生产环境数据，均明确标注“官方 Q&A”以示出处。
- **开源社区传承**：DeepInfer 的 `README.md` 致谢了 vLLM、SGLang 与 TensorRT-LLM；官方 Q&A 亦指出其异步调度（ZOS）参考了 SGLang 与 Nanoflow 的思路。
- **本书配套代码与许可**：本书第二部分实现的 CPU 教学引擎 `minidi/`、全部单元测试 `tests/`、实验绘图脚本 `scripts/` 及构建管线均为本书原创，以 **MIT 协议** 开源；全书正文与图表以 **CC BY 4.0 协议** 发布。

---

## 2. DeepInfer 源码中的两处“注释与实现不一致”备忘

在逐行核对 commit `553741557a57df44a46fb3ab72b6aafeaf945337` 的过程中，我们发现上游源码中有两处函数注释与真实代码逻辑相反或不一致，特此记录供读者查阅源码时参考（本书正文及 `minidi` 均严格以**真实可执行代码**为准）：

1. **`_schedule_decode` 的抢占方向（`csrc/engine/scheduler.hpp` L556 vs L625–L644）**：
   - *注释说法*：第 556–557 行函数头注释写着“block 不足时抢占 output 最短的 seq”。
   - *代码真实行为*：第 625–626 行按 `a->get_output_len() > b->get_output_len()` 将 `running` 降序排列（`running[0]` 输出最长，`running[r-1]` 输出最短），随后右指针从 `r - 1`（输出最短）开始排入批次（即第 622 行注释所称的 `// LCFS`），而左指针在块不足时从 `l = 0`（**输出最长**）开始调用 `_preempt(running[l])`。这一行为与 `csrc/engine/sequence.hpp` L56 对 `insufficient_system_resource` 的注释（“如果 kvcache 即将溢出，则占用 kvcache 最多的任务，会被我们强制终止”）完全一致。
2. **`profile_num_available_blocks` 的多卡块数汇总方式（`deepinfer/engine/llm_engine.py` L426 vs `deepinfer/worker/base.py` L265–L266）**：
   - *注释说法*：`LLMEngine._init_cache` 的 docstring（L426–L427）写着取所有 worker 可用块数的**最小值（minimum）**。
   - *代码真实行为*：`Worker.profile_num_available_blocks`（`base.py` L265–L266）调用 `torch.distributed.all_reduce(num_gpu_blocks_tensor, op=ReduceOp.SUM)` 后整除 `world_size` 并按偶数对齐（`// 2 * 2`），实际计算的是各卡可用块数的**平均值**。

---

## 3. 反馈与勘误

如果你在阅读本书或运行 `minidi` 的过程中发现任何笔误、技术疏漏或有改进建议，非常欢迎在仓库 Issue 区提出交流：`https://github.com/cluster1900/deepinfer-reader/issues`。
