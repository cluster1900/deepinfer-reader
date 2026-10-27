# Acknowledgments, References, and Source Code Errata Notes

## 1. Acknowledgments and Version Statement

- **The DeepInfer Open-Source Project**: Open-sourced by the DeepSeek-AI team (repository: `https://github.com/deepseek-ai/DeepInfer`, licensed under Apache License 2.0). All source file paths, function names, line numbers, and default configurations in this book are strictly locked to and verified against commit **`553741557a57df44a46fb3ab72b6aafeaf945337`** (2026-10-25, `fix: validate boolean environment variables with strict parsing (#218)`). Copyright of quoted source snippets and documentation excerpts belongs to their original authors and is used solely for technical analysis and educational purposes.
- **Official Technical Q&A Document (Official Q&A)**: Refers to `doc/inference_deployment_qa.md` included in the repository (since `774ce24` it replaces the earlier PDF `doc/DeepSeek V4.1 Flash 推理部署 Q&A.pdf`). Wherever this book cites production deployment figures—such as online E/P/D disaggregation ratios, 3FS distributed KVCache storage, production throughput numbers, or empirical EPLB metrics—it explicitly labels them as "Official Q&A" to indicate their source.
- **Open-Source Community Lineage**: DeepInfer's `README.md` acknowledges vLLM, SGLang, and TensorRT-LLM; the Official Q&A also notes that its zero-overhead asynchronous scheduling (ZOS) draws inspiration from SGLang and Nanoflow.
- **Companion Code and Licensing**: The CPU pedagogical engine `minidi/` implemented in Part II of this book, along with all unit tests in `tests/`, plotting scripts in `scripts/`, and the build pipeline, are original works of this book open-sourced under the **MIT License**; the text and diagrams of the book are released under the **CC BY 4.0 License**.

---

## 2. Notes on Two "Comment vs. Implementation Discrepancies" in DeepInfer's Source Code

During our line-by-line verification of commit `553741557a57df44a46fb3ab72b6aafeaf945337`, we found two places in the upstream codebase where a function comment contradicts or diverges from the actual executable logic. We record them here as a reference for readers browsing the upstream code (both the main text of this book and `minidi` strictly follow the **actual executable code**):

1. **Preemption direction in `_schedule_decode` (`csrc/engine/scheduler.hpp` L556 vs. L625–L644)**:
   - *What the comment says*: The function header comment on lines 556–557 states that when blocks are insufficient, it preempts the sequence with the *shortest* output ("block 不足时抢占 output 最短的 seq").
   - *What the code actually does*: Lines 625–626 sort `running` in descending order via `a->get_output_len() > b->get_output_len()` (so `running[0]` has the longest output and `running[r-1]` has the shortest output). The right pointer then schedules sequences into the batch starting from `r - 1` (shortest output, matching the `// LCFS` comment on line 622), whereas when blocks run short, the left pointer calls `_preempt(running[l])` starting from `l = 0` (**longest output**). This actual behavior is completely consistent with the comment on `insufficient_system_resource` in `csrc/engine/sequence.hpp` L56 ("if kvcache is about to overflow, the task occupying the most kvcache will be forcibly terminated by us").
2. **Multi-GPU block count aggregation in `profile_num_available_blocks` (`deepinfer/engine/llm_engine.py` L426 vs. `deepinfer/worker/base.py` L265–L266)**:
   - *What the comment says*: The docstring of `LLMEngine._init_cache` (L426–L427) states that it takes the **minimum** number of available blocks across all workers.
   - *What the code actually does*: `Worker.profile_num_available_blocks` (`base.py` L265–L266) calls `torch.distributed.all_reduce(num_gpu_blocks_tensor, op=ReduceOp.SUM)`, divides by `world_size`, and aligns down to an even number (`// 2 * 2`), which actually computes the **average** number of available blocks across all GPUs.

---

## 3. Feedback and Errata

If you spot any typos, technical oversights, or have suggestions for improvement while reading this book or running `minidi`, you are warmly welcome to open an issue for discussion at `https://github.com/cluster1900/deepinfer-reader/issues`.
