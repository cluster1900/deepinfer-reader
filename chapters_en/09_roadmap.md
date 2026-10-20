# Chapter 9  From 0 to 1 Roadmap: Replicating the Core Mechanisms in 800 Lines of Code

After finishing Part I, you now hold a complete mental map of DeepInfer's architecture. Yet between "understanding someone else's code" and "being able to write it yourself from scratch," there is often a thin but stubborn barrier:

- When writing `process_outputs` yourself, slightly mixing up the order of additions and subtractions between `num_cached` and `get_len()` will throw off the positional encodings of the very next chunk;
- When writing the asynchronous scheduler `ZOSEngine` yourself, failing to handle the edge case where an in-flight request gets preempted will silently leak free blocks—or even drive the free-block counter negative;
- When combining Chunked Prefill with preemption and recomputation, naively pushing a preempted request back to the very front of the waiting queue will slam straight into a deadlock where a half-finished chunk holds onto blocks waiting to be scheduled while the preempted request at the head of the queue waits for blocks to be freed!

The goal of Part II (Chapters 9–16) is to help you **break through that barrier with your own hands**. Instead of writing verbose CUDA kernels or relying on expensive GPUs, we use **pure Python + NumPy** to build an educational inference engine of under 900 lines from scratch inside the `minidi/` directory. Every core control-plane decision in DeepInfer is implemented and tested for real on the CPU.

---

## 9.1 Our Four Iron Rules of Development

To keep these <900 lines of code both approachable and rigorous enough to validate our understanding of DeepInfer, we enforce four iron rules throughout `minidi`:

1. **Golden Reference Verification**:
   In `minidi/engine.py`, we first write a 10-line naive generation function `reference_generate` that uses no KV Cache at all and recomputes the entire sequence from scratch at every step. **Whether we subsequently add paged block tables, Chunked Prefill, two-pointer preemption, asynchronous scheduling (ZOS), or speculative decoding, every single token emitted by the engine must match `reference_generate` 100% token for token!**
2. **Eliminating Floating-Point Noise with `float64` + Deterministic Position Seeds**:
   Nothing is worse in an educational engine than writing a bug and mistaking it for floating-point drift. Therefore, `TinyLM` runs entirely in double precision (`float64`)—keeping the logits discrepancy between paged forward passes and full-sequence forward passes strictly below $10^{-10}$—and random sampling seeds are uniquely determined by `(seq.seed, position)`.
3. **Tiny Hyperparameters to Amplify Boundary Collisions**:
   In DeepInfer, the physical block size is 128 and the sliding window is 128, so it takes thousands of tokens to trigger cross-block allocations and preemptions. In `minidi`, we set the default block size to `block_size = 4` and the sliding window to `W = 4`—just a dozen tokens will repeatedly trigger cross-block allocation, block-table shrinking, full-pool preemption, and circular buffer wrap-around, leaving latent bugs nowhere to hide.
4. **One-to-One Line-Number Mapping to DeepInfer**:
   Every file and function in `minidi/` includes comments pointing directly to the exact C++ or Python file and line range in DeepInfer that it mirrors, making side-by-side reading effortless.

---

## 9.2 The Seven-Stage Roadmap

We break the construction of `minidi` into 7 progressive stages, each corresponding to a chapter in Part II and an independent suite of `pytest` unit tests:

```{.mermaid caption="Figure 9-1  The seven evolutionary stages of minidi from 0 to 1 and their module mappings"}
flowchart LR
  subgraph Base["Foundational Core (Chapters 10–12)"]
    S1["Stage 1 (Chapter 10)<br/>TinyLM & Minimal step Loop<br/>model.py · engine.py"] --> S2["Stage 2 (Chapter 11)<br/>PagedKV & BlockManager<br/>blocks.py · sequence.py"]
    S2 --> S3["Stages 3–4 (Chapter 12)<br/>Chunked Prefill & LCFS Preemption<br/>scheduler.py"]
  end

  subgraph Adv["Advanced Performance Mechanisms (Chapters 13–15)"]
    S5["Stage 5 (Chapter 13)<br/>Zero-Overhead Async ZOSEngine<br/>zos.py"]
    S6["Stage 6 (Chapter 14)<br/>Lossless Speculative & Varlen Verify<br/>engine.py · varlen.py"]
    S7["Stage 7 (Chapter 15)<br/>SWA Ring Rewind · EPLB · Voting<br/>swa.py · eplb.py · vote.py"]
  end

  S3 --> S5 & S6 & S7
```

The code size of each stage and the DeepInfer counterparts they replicate are summarized below:

| Stage | `minidi/` Module Files | Lines of Code | Replicated DeepInfer Source Location | Chapter |
| --- | --- | --- | --- | --- |
| **Stage 1** | `model.py` (`TinyLM`) + `engine.py` | 93 + 134 lines | `models/deepseek.py` · `engine/llm_engine.py` | Chapter 10 |
| **Stage 2** | `blocks.py` + `sequence.py` | 61 + 81 lines | `csrc/engine/block_manager.hpp` · `sequence.hpp` | Chapter 11 |
| **Stages 3–4** | `scheduler.py` | 206 lines | `csrc/engine/scheduler.hpp` (chunking, two-pointer preemption, shrinking) | Chapter 12 |
| **Stage 5** | `zos.py` (`ZOSEngine`) | 105 lines | `worker/worker.py` · `csrc/engine/zos.cpp` | Chapter 13 |
| **Stage 6** | `varlen.py` + `engine.py` (speculative branch) | 56 lines | `layers/sampler.py` · `csrc/engine/mtp_varlen.hpp` | Chapter 14 |
| **Stage 7** | `swa.py` + `eplb.py` + `vote.py` | 44 + 61 + 25 lines | `cache_engine.py` · `eplb/policy.py` · `scheduler.hpp` | Chapter 15 |

All told, the entire `minidi/` directory totals just **869 lines of Python code** (including detailed comments), backed by **64 unit test cases** under `tests/`.

---

## 9.3 Taking It for a Spin First

Before diving into Chapter 10, you can run `examples/quickstart.py` in your terminal to watch the exact same batch of 8 concurrent requests execute across the **synchronous engine**, **asynchronous engine (ZOS)**, and **speculative decoding engine**:

```bash
$ python3 examples/quickstart.py
[Sync]   match=True  steps=36  prefill_steps=11  preempted=2
[ZOS]    match=True  steps=38
[Spec]   match=True  steps=32  draft_accept_rate=0.39
```

Notice the three insights packed into those three lines of output:
1. All three modes report `match=True`, proving that whether requests undergo 2 mid-flight memory preemptions and recomputations, run under 1-step-ahead blind ZOS scheduling, or speculate 3 draft tokens per step, their final outputs match the cache-free `reference_generate` 100%!
2. Why did `[ZOS]` take 38 steps—2 more than the 36 steps of `[Sync]`? Because when a request samples a stop condition at step $N$, the 1-step-ahead ZOS scheduler blindly schedules one extra step before finding out (which is subsequently discarded during output settlement; see Chapter 13).
3. Why did `[Spec]` need only 32 steps? Because the 39% of draft tokens guessed correctly by the draft model saved the main model several autoregressive iterations (see Chapter 14).

Ready? Let's turn to Chapter 10 and write the first line of code.
