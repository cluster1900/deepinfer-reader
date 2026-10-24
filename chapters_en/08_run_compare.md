# Chapter 8  Configuration Flags, Deterministic Inference, and Comparison with Mainstream Frameworks

As the closing chapter of Part I, we bring together the runtime flags, deterministic reproducibility requirements, and benchmarking conventions scattered across the first seven chapters into a practical **quick-reference guide**. Finally, we place DeepInfer side by side with vLLM and SGLang to examine the fundamental architectural trade-offs among three inference engines built with very different goals.

---

## 8.1 Quick Reference for Building and Offline Inference

DeepInfer is built for two hardware platforms—**NVIDIA SM100 (Blackwell)** and **Huawei Ascend 950 NPU**—requiring Linux, GCC ≥ 13, Python 3.12, and either CUDA 13 or CANN 9.2 (`README.md` L33–L52):

```bash
# 1. Specify the target hardware backend (must remain exported during both build and runtime)
export DEEPINFER_ACCELERATOR=nvidia   # Set to 'ascend' on Huawei Ascend environments
./build.sh

# 2. Run offline inference examples (default model: deepseek-ai/DeepSeek-V4.1-Flash-1017)
python3 examples/offline_inference.py            # Default: EP=8, DP=8
python3 examples/offline_inference.py --ep 2 --dp 8
python3 examples/offline_inference.py --ep 2 --cp 8
```

When invoked via the Python API, the entry class `deepinfer.entrypoints.llm.LLM` (`llm.py` L54+) maintains a signature closely aligned with vLLM:

```python
from deepinfer.entrypoints.llm import LLM
from deepinfer.sampling_params import SamplingParams

llm = LLM(
    model="deepseek-ai/DeepSeek-V4.1-Flash-1017",
    expert_parallel_size=8,
    gpu_memory_utilization=0.85,
    trust_remote_code=True,
)
outputs = llm.generate(
    ["Explain speculative decoding in three sentences:"],
    SamplingParams(temperature=0.6, max_tokens=256, seq_seed=42),
)
```

Pay special attention to **three overrides that `examples/_config.py` applies on top of the engine's default settings** (L75–L83):
By default, the underlying engine disables dynamic EPLB rearrangement (`DEEPINFER_EXPERT_REARRANGEMENT_STEP_INTERVAL = 0`). However, when EPLB is enabled in the example scripts, `setup_common_env()` automatically sets the number of redundant experts to `32`, sets the rearrangement interval to `30` steps, and enables `DEEPINFER_REARRANGE_AFTER_PREFILL_IN_NORMAL_ENGINE=1`. If you only read the default values in `environ.py`, it is easy to miss these flags that the example runner turns on for you.

---

## 8.2 Comprehensive Reference of Core Environment Variables

Most of DeepInfer's core features are controlled via environment variables declared in `deepinfer/environ.py`. Below, we summarize the 13 key switches covered throughout this book, grouped by subsystem:

| Subsystem | Environment Variable | Default in `environ.py` | Description & Production Recommendation |
| --- | --- | --- | --- |
| **Async & CUDA Graph**<br/>(Chapter 5) | `DEEPINFER_ZERO_OVERHEAD_SCHEDULING` | On for `NORMAL`/`COMPLETION`,<br/>Off for `PROMPT` | Enables Zero-Overhead Asynchronous Scheduling (ZOS), overlapping CPU scheduling of step $N+1$ with GPU execution of step $N$ |
| **Async & CUDA Graph** | `DEEPINFER_ENFORCE_EAGER` | `False` | Set to `1` to disable CUDA Graph and fall back to eager kernel-by-kernel PyTorch launch (useful for debugging) |
| **Async & CUDA Graph** | `DEEPINFER_CUDAGRAPH_BATCH_SIZE_INTERVAL` | `8` | Step size of `total_q` buckets captured by CUDA Graph (below 8, captures buckets 1, 2, 4) |
| **Async & CUDA Graph** | `DEEPINFER_DISABLE_SYNC_GUARD` | `False` | Set to `1` to disable the `forbid_cuda_sync` guard that raises errors on implicit host-device synchronizations |
| **Speculative & Varlen**<br/>(Chapter 6) | `DEEPINFER_USE_NEXTN_DECODE` | `5` | Maximum number of candidate draft tokens (`nextn`) generated in parallel per step by DFlash |
| **Speculative & Varlen** | `DEEPINFER_ENABLE_MTP_VARLEN` | `False` | Enables greedy variable-length verification driven by `ConfidenceHead` and CUDA Graph step latencies |
| **Memory & YoCo**<br/>(Chapters 4 & 7) | `DEEPINFER_MAX_BACKTRACKING_TOKENS` | `max(6, nextn + 1)` | Backtracking and boundary snapshot margin for the SWA Ring (main model SWA Ring = `128 + 6 = 134`) |
| **Memory & YoCo** | `DEEPINFER_YOCO_PREFILL_TRUNCATION_MODE` | `"off"` | Prefill truncation mode for second-half YoCo layers; `"compact"` is recommended in production |
| **Memory & YoCo** | `DEEPINFER_YOCO_PREFILL_KEEP_TOKENS` | `0` | Number of trailing tokens retained during YoCo truncation; README recommends `140` (must be $\ge 134$) |
| **Expert Load Balancing**<br/>(Chapter 7) | `DEEPINFER_OVERRIDE_NUM_EXTRA_EXPERTS` | `0` | Lower bound on total redundant physical experts replicated by EPLB (rounded up to a multiple of `ep_size`) |
| **Expert Load Balancing** | `DEEPINFER_EXPERT_COUNT_WINDOW_SIZE` | `100` | Sliding window size (in steps) over which EPLB tracks historical per-step token loads for logical experts |
| **Expert Load Balancing** | `DEEPINFER_EXPERT_REARRANGEMENT_STEP_INTERVAL` | `0` (disabled) | Step interval at which EPLB triggers global rebalancing and cross-GPU hot migration of expert weights |
| **Deterministic Inference** | `DEEPINFER_BATCH_INVARIANT` | `False` | Enables batch-invariant deterministic matrix multiplication and attention reduction (NVIDIA only) |

---

## 8.3 Achieving Token-by-Token Reproducibility: Batch-Invariant Inference and Deterministic Sampling

In reinforcement learning (RLHF / GRPO) rollouts, production debugging, and accuracy evaluations, engineers often face a strict requirement: **given the same request and random seed, whether the request runs alone or batched together with 63 other requests of varying lengths, every single generated token must be bit-for-bit identical!**

Why does standard GPU inference fail to guarantee this? Because of three "sources of non-determinism contamination":

1. **Kernel reduction order varies with batch size**: Floating-point addition is not associative: $(a + b) + c \ne a + (b + c)$. When the batch size `total_q` changes, GEMM and FlashMLA kernels choose different tile sizes and parallel reduction orders, introducing tiny fluctuations of a few ULPs (units in the last place) in the output logits. Whenever a logit sits right on a sampling threshold boundary, the trajectory bifurcates.
2. **Variable-length verification depends on co-batched neighbors**: If `DEEPINFER_ENABLE_MTP_VARLEN=1` (Chapter 6) is enabled, how many draft verification slots `seqlens_q[i]` a request receives in a given step depends on how its confidence ranks against **all other requests in the batch**! Change its neighbors, and `seqlens_q[i]` changes.
3. **Global random number generator cross-talk**: If an entire batch shares a single global RNG state, which random number a request draws depends on how many random numbers were consumed by requests ordered ahead of it.

DeepInfer locks down all three contamination sources with a three-part mechanism (`README.md` L78–L92, `worker/runtime_config.py` L61–L69):

```{.mermaid caption="Figure 8-1  Three conditions required for strict cross-batch reproducibility (Deterministic & Batch-Invariant) in DeepInfer"}
flowchart LR
  subgraph Conds["Locking Down the Three Sources of Non-Determinism"]
    A1["1. Kernel-Level Batch Invariance (DEEPINFER_BATCH_INVARIANT=1)<br/>Enable deterministic tiling in deep_gemm / tile_kernels & pass batch_invariant=True to FlashMLA"]
    B1["2. Disable Cross-Request Budget Competition (DEEPINFER_ENABLE_MTP_VARLEN=0)<br/>Verify a fixed 1 + nextn positions per request, independent of neighbor confidence rankings"]
    C1["3. Per-Request Per-Position Independent RNG (SamplingParams.seq_seed)<br/>prepare_sample.cpp derives Gumbel noise from each request's seed and sampling position"]
  end

  Out["Final Result:<br/>100% identical token sequences<br/>under any batching configuration"]

  A1 & B1 & C1 --> Out
```

---

## 8.4 Understanding the Three Measurement Conventions in `examples/benchmark.py`

When reading or running `examples/benchmark.py` (and `examples/README.md`) to measure throughput, keep three specific measurement conventions in mind:

1. **The Decode Throughput reported by the benchmark script is the peak kernel verification throughput assuming "100% draft acceptance"**:
   The formula computes `avg_SPS × decode_num_requests × (1 + nextn) / cp_size` directly (crediting a fixed `1 + 5 = 6` tokens per step). It measures how many positions the hardware can verify per second, rather than the effective token generation throughput multiplied by the actual acceptance rate.
2. **Never trust throughput numbers when the Profiler is enabled**:
   Enabling the PyTorch Profiler adds up to 50% extra launch overhead across dense sequences of small kernels (especially at `bsz=1`). Use it only to inspect relative kernel time breakdowns, never to measure SPS.
3. **EPLB statistics synchronization steps are automatically excluded**:
   When calculating steady-state per-step latency, the benchmark script automatically filters out steps that trigger cross-GPU `all_gather` collectives for MoE load statistics, reflecting pure inference step speed.

---

## 8.5 Head-to-Head Comparison: DeepInfer vs. vLLM vs. SGLang

Having walked through the entire codebase in Part I, we can now step back and compare **DeepInfer** against the community's two leading general-purpose inference frameworks—**vLLM** and **SGLang**—from an architect's perspective (all entries in the DeepInfer column are verified directly against the source code in this book; the vLLM and SGLang columns reflect their respective public architectures):

| Architectural Dimension | DeepInfer (commit `9bd9cf2`) | vLLM (V1 Architecture) | SGLang |
| --- | --- | --- | --- |
| **Core Design Goal** | **Extreme single-model specialization**: purpose-built for DeepSeek-V4.1-Flash and large-scale Wide-EP clusters | **Universal ecosystem foundation**: supports 100+ model architectures and a wide range of hardware backends | **General high performance & complex control flow**: balances multi-model support, Radix caching, and structured generation |
| **Scheduler Implementation Language** | **Pure C++ (`csrc/engine/scheduler.hpp`)**, releasing the GIL for microsecond-level block management and preemption | Primarily Python (assisted by selected C++/Cython data structures) | Primarily Python (running in an isolated scheduler process/thread) |
| **Prefill & Decode Batching** | **Strictly separated; never mixed in the same step**; multi-DP ranks align step types via `ALL(70)/ANY(100)` two-round voting | Hybrid batching by default (Chunked Prefill and Decode mixed within the same batch) | Separated by default or chunked-mixed; supports PD-disaggregated deployment |
| **KV Cache Physical Architecture** | **Dual-pool ledger**: paged pool (128 tokens/block, single-KV-head compression) + fixed-size State pool (134-slot SWA circular buffer) | Unified paged pool (PagedAttention, with recent hybrid attention extensions) | Unified paged pool + RadixAttention prefix tree management |
| **Cross-Request Prefix Caching** | Retains only `fork_seq` ref-counting inside the engine; **persistent prefix caching is offloaded to an external 3FS cluster at 128-token granularity** | Built-in block-hash-based Automatic Prefix Caching (APC) | Built-in Radix-Tree-based automatic prefix reuse |
| **Asynchronous Scheduling** | **ZOS (Zero-Overhead Scheduling)**: dual-variable in-flight accounting + GPU-side `remap` and `compute_seqlens_k_zos` | Supports Async Output Processor and multi-step scheduling | Overlap Scheduler (overlapping CPU scheduling with GPU forward pass) |
| **Speculative Decoding & Verification** | **Native built-in DFlash + greedy variable-length verification**: parallel feature extraction, Markov head sampling, confidence-gated CUDA Graph bucketing | External Draft model / N-gram / Medusa / Eagle, primarily fixed verification length | Supports Eagle / MTP speculative decoding schemes |
| **Multi-GPU Parallelism Philosophy** | **Completely drops TP**: DP / Mirror CP for attention, Wide-EP + online EPLB redundant-expert hot migration for MoE | Primarily TP / PP, with EP support for MoE layers in large models | Supports TP + DP Attention + DeepEP expert parallelism |

### Part I Summary: What Have We Learned from DeepInfer?

Closing Part I, you can see that every seemingly unconventional design choice in DeepInfer is the result of deep **model–system co-design**:

- Because the model uses **SWA + compressed attention + YoCo**, memory management is split into two ledgers—a paged pool and a fixed-size circular pool—and the second half of layers can be skipped entirely or truncated during Prefill;
- Because the model features **massive MoE layers running under Wide-EP**, it drops TP, strictly separates Prefill and Decode steps, and employs two-round voting plus EPLB redundant-expert rearrangement to eliminate cross-GPU straggler effects;
- Because the model natively integrates a **DFlash speculative head and confidence predictor**, the scheduler can pair cumulative confidence scores with the step-function latency curve of CUDA Graphs, using a greedy algorithm to spend verification compute only where it pays off.

Armed with these system insights, starting in the next chapter (Chapter 9) we roll up our sleeves and enter **Part II**: writing a complete, self-contained educational CPU inference engine—`minidi`—from scratch in just a few hundred lines of pure Python + NumPy!
