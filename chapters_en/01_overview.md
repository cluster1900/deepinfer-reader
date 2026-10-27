# Chapter 1  What Is DeepInfer: Positioning, Architecture, and Repository Map

Before diving headfirst into C++ schedulers or CUDA/TileLang kernels, let us answer three fundamental questions:

1. **How does DeepInfer fundamentally differ from familiar engines like vLLM and SGLang?**
2. **What does DeepSeek's real online inference cluster look like, and which slice of it is open-sourced here?**
3. **Out of 60,000 lines of source code, which 10,000 lines form the core skeleton that drives the engine?**

---

## 1.1 One-Sentence Positioning: A Production Engine Custom-Tailored for a Single Model

Open DeepInfer's `README.md` (L3–L7), and the opening paragraph states its identity plainly:

> DeepInfer is the high-performance inference framework powering DeepSeek's large-scale online serving; this open-source release provides its **offline inference engine** while retaining the core runtime features required for large-scale production serving. It currently supports a single model family: **DeepSeek-V4.1-Flash** (default checkpoint `deepseek-ai/DeepSeek-V4.1-Flash-1017`).

This positioning makes its codebase look radically different from vLLM or SGLang:

- **vLLM and SGLang are "general-purpose department stores"**: They must accommodate hundreds of model architectures and diverse hardware generations, requiring thick abstraction layers and plugin registries.
- **DeepInfer is a "Formula 1 race car"**: It is built exclusively for **DeepSeek-V4.1-Flash**. Whatever bespoke architectural traits the model has—such as Sliding Window Attention (SWA) + Gated Linear Attention (GLA) in its first half of layers, shared-KV Multi-head Latent Attention (MLA) in its second half, built-in DFlash speculative draft heads, and large-scale MoE Expert Parallelism—the engine's block manager, scheduler, and kernels are sculpted to match those exact contours, with zero redundant abstraction layers.

Consequently, the greatest value of reading DeepInfer is not treating it as a general-purpose toolbox to run other models, but **seeing through it how an inference engine must fundamentally reinvent itself when model architectures evolve toward hybrid attention + native speculative decoding + ultra-large-scale MoE**.

---

## 1.2 Core Features at a Glance and Chapter Guide

`README.md` L9–L29 lists eight core features of the open-source release. Translating the jargon into plain engineering terms, here is the pain point each feature solves and where we cover it in detail:

| Core Feature (README) | What Pain Point Does It Actually Solve? | Chapters |
| --- | --- | --- |
| **Optimized kernels, model & sampler runtime** | Integrates FlashMLA, DeepGEMM, DeepEP, DeepSelect, and custom TileLang kernels to saturate SM100 / Ascend 950 compute | Ch 7 |
| **Parallelism: Attention DP / balanced CP, MoE EP + EPLB** | Drops Tensor Parallelism (TP); runs DP or mirror-balanced CP for attention, and EP with EPLB hot-expert replication for MoE to prevent stragglers | Ch 7, 15 |
| **DSpark (DFlash) & confidence-scheduled verification** | Generates 5 draft tokens in parallel per step and uses Markov + confidence heads to dynamically decide how many drafts each request verifies | Ch 6, 14 |
| **SWA bounded replay** | Sliding-window layers avoid storing full history KV and instead maintain a fixed-size ring buffer (134 slots), capping long-context memory | Ch 4, 15 |
| **Asynchronous scheduling (ZOS)** | Overlaps CPU scheduling of step $N+1$ with GPU execution of step $N$, eliminating inter-step bubbles | Ch 5, 13 |
| **Batch-invariant inference & Deterministic sampling** | Uses fixed-split reduction kernels and per-request Gumbel RNGs so a request produces identical outputs regardless of batch composition (NVIDIA only) | Ch 8 |
| **Structured outputs** | Asynchronously compiles regex / JSON Schema / EBNF grammar automata in background threads and applies vocabulary bitmasks during decoding | Ch 2, 8 |

Target hardware is equally focused: only **NVIDIA SM100 (Blackwell architecture)** and **Huawei Ascend 950 NPU** are supported (`README.md` L33–L36), requiring Linux, GCC ≥ 13, Python 3.12, and CUDA 13 or CANN 9.2.

---

## 1.3 Production Online System vs. Open-Source Scope

While reading DeepInfer's source code, you will frequently encounter empty hook methods inside `LLMEngine`, such as `init_pd_split()`, `do_save_kvcache()`, and `send_processed_prompts()` (`deepinfer/engine/llm_engine.py` L801–L858). Why do these no-op shells exist?

The answer lies in `README.md` L38–L42 and the **Official Q&A**: the full production system is a distributed cluster, and this open-source release strips out three external components that are tightly coupled to DeepSeek's internal infrastructure (such as the 3FS distributed file system, internal RPCs, and global gateways):

1. **Cross-request persistent distributed KVCache storage** (backed by a 3FS cluster, saving and retrieving prefix KV and SWA states at a 128-token granularity);
2. **Cross-instance transport for disaggregated VE (Vision Encoding) / Prefill / Decode deployment** (though the intra-instance `PROMPT` / `COMPLETION` engine-role logic remains fully intact in the codebase);
3. **Global request routing and load balancing across Data Parallel (DP) replicas**.

Plotting the full production topology described in the Official Q&A immediately reveals where the open-source codebase sits within the larger edifice:

```{.mermaid caption="Figure 1-1  DeepSeek Production E/P/D Disaggregated Deployment Overview (dashed boxes indicate un-open-sourced external infrastructure; solid boxes indicate DeepInfer engine instances)"}
flowchart TB
  subgraph Infra["1. External Ingress & Global Routing Layer (Not Open-Sourced)"]
    direction LR
    U["User Request"] --> R["DeepSeek Recipe<br/>API Protocol & Tokenization"] --> LB["Global Load Balancer<br/>VE / Prefill / Decode Routing"]
  end

  subgraph Cluster["2. DeepInfer Inference Instance Cluster (Core Engine Open-Sourced)"]
    direction LR
    VE["Vision Encode Instance (VE)<br/>Single-GPU Standalone"] -- "Visual Features" --> PE["Prefill Instance (PROMPT)<br/>Small Parallelism: EP 2~8 / DP 2~8"] -- "First Token + State" --> CE["Decode Instance (COMPLETION)<br/>Large Wide-EP: 64~128"]
  end

  subgraph Storage["3. Distributed Cache Substrate (Not Open-Sourced)"]
    direction LR
    KV1["3FS Distributed KVCache Storage Cluster"] --- KV2["128-Token Granularity: PE Writes & CE Reads<br/>Prefix KV and SWA Snapshots"]
  end

  Infra --> Cluster --> Storage

  style R stroke-dasharray: 5 5
  style LB stroke-dasharray: 5 5
  style KV1 stroke-dasharray: 5 5
  style KV2 stroke-dasharray: 5 5
```

Understanding Figure 1-1 unlocks three architectural mysteries in the source code (all cross-verified between the Official Q&A and the codebase):

- **Why is there no Tensor Parallelism (TP) in the codebase?** As explained in the Official Q&A, DeepSeek-V4.1-Flash's shared-KV MLA and compact weight design allow a single GPU to hold an entire attention replica and expert shard. Online Prefill instances use small-scale `EP 2~8 / DP 2~8` to minimize communication overhead, while Decode instances use `EP 64~128 / DP 64~128` (Wide-EP) to amass a large global batch size that keeps MoE experts saturated. Because attention layers satisfy `#DP = #EP` (when CP is disabled), weight-splitting TP within a single layer is completely unnecessary.
- **Why are there three `EngineType` modes?** `deepinfer/engine/engine_type.py` defines `NORMAL` (runs both prefill and decode, used for offline inference and single-instance testing), `PROMPT` (corresponds to online PE instances, running only prefill), and `COMPLETION` (corresponds to online CE instances, running only decode).
- **Why are there state-copy hooks aligned to 128 tokens?** Online PE and CE instances share a 3FS-backed KVCache store. To balance I/O efficiency and cache hit rates, both KV blocks and SWA states are persisted at **128-token boundaries** (`KVSTORE_GRAIN = 128`). As we will see in Chapter 4, this directly dictates the size of the SWA ring buffer.

---

## 1.4 Repository Directory Map: Where to Find Each Piece of Logic

DeepInfer's repository is cleanly organized into four major areas: the **C++ Engine Core (`csrc/`)**, the **Python Runtime & Model Layer (`deepinfer/`)**, **Hardware Topology Utilities (`deep_cluster/`)**, and **External High-Performance Kernel Libraries (`vendor/`)**:

```{.mermaid caption="Figure 1-2  DeepInfer Repository Directory Map (commit 5537415)"}
flowchart LR
  subgraph Core["1. Engine & Runtime Backbone"]
    csrc["csrc/engine/ (C++ Scheduling Core)<br/>scheduler · block_manager<br/>sequence · zos · mtp_varlen"]
    eng["deepinfer/entrypoints & engine<br/>LLM Entrypoint · LLMEngine Main Loop"]
    wk["deepinfer/worker/<br/>Worker · ModelRunner · CacheEngine"]
  end

  subgraph ModelExec["2. Model, Parallelism & Kernel Layer"]
    models["models/<br/>deepseek.py · v4_modules/<br/>mtp/dflash.py (DFlash Draft)"]
    layers["layers/<br/>sampler.py · linear.py<br/>tilelang_ops/ (cuda & asc)"]
    par["parallel_utils/ & eplb/<br/>DP / CP / EP · Expert Load Balancing"]
  end

  subgraph Support["3. Submodules & Peripheral Components"]
    vendor["vendor/ (6 Kernel & Comm Submodules)<br/>FlashMLA · DeepGEMM<br/>DeepEP · DeepSelect"]
    aux["structure_output/ (xgrammar)<br/>ve/ (Multimodal) · deep_cluster/"]
    ex["examples/ · eval/ · tests/<br/>Offline Inference, Benchmarks & Evals"]
  end

  csrc --> models
  eng --> layers
  wk --> par
  models --> vendor
  layers --> aux
  par --> ex
```

One architectural decision stands out above all others: **the scheduler (`Scheduler`), block manager (`BlockManager`), sequence state (`Sequence`), asynchronous scheduling input correction (`zos.cpp`), and variable-length verification budget allocation (`mtp_varlen.hpp`) are all written in C++**, compiled via `pybind11` into `deepinfer._C.engine` for Python to invoke. Why not write the scheduler in Python? Because when a GPU finishes a decode step in just a few milliseconds, the overhead of the Python interpreter iterating over hundreds of request objects and mutating block tables would exceed the GPU execution time itself.

The six `vendor/` submodules pinned at the current commit and their roles are summarized below:

| Submodule | Pinned Commit | Role in DeepInfer |
| --- | --- | --- |
| `FlashMLA` | `2e5429f` | High-performance CUDA kernels for Multi-head Latent Attention (MLA) and sparse Top-$k$ attention |
| `DeepGEMM` / `DeepGEMM-Ascend` | `1fc9c66` / `3a35326` | FP8 / BF16 matrix multiplication and MoE grouped GEMM (dual backends for NVIDIA and Ascend) |
| `DeepEP` / `DeepEP-Ascend` | `f6aa38f` / `3b25377` | Cross-GPU MoE token dispatch and combine collectives, plus `BucketBuffer` for Context Parallelism (CP) |
| `DeepSelect` | `bfa4507` | Fast Top-$k$ block selection kernel for sparse attention (DSA / NSA) |

> **Baseline update note (`9bd9cf2`, upstream #157)**: `vendor/DeepEP` moves from `93eb6eb` to `f6aa38f`. DeepEP has only two commits in between: `b5a33ee` (removes formatting and lint checks) and `f6aa38f` (switches the DeepJIT submodule to an HTTPS URL). Neither touches the dispatch / combine kernels, so this book's analysis of EP communication is unaffected. Because DeepEP now records an HTTPS DeepJIT URL, `scripts/build_vendor.sh` drops `GIT_REWRITE=(-c url."https://github.com/".insteadOf="git@github.com:")` and runs both `submodule update --init` calls directly, so build machines without an SSH key can still fetch every submodule.

---

## 1.5 Lines of Code Distribution: Focus on the Core 10,000 Lines

Counting all `.py`, `.cpp`, and `.hpp` files in the repository (excluding `vendor/` submodules) yields a total of **60,838 lines of code**:

![Figure 1-3  Lines of Code Breakdown Across DeepInfer Modules (data from data/deepinfer_loc.json, excluding vendor submodules)](images/plots_en/ch01_loc.png)

Looking at Figure 1-3 immediately takes the intimidation out of reading a large codebase:

- The largest slice, `layers/tilelang_ops/` (**23,331 lines**, nearly 40% of the repository), is the custom TileLang operator library; because it supports both NVIDIA CUDA (`cuda/`) and Huawei Ascend (`asc/`), every operator has two hardware implementations plus a dispatch wrapper.
- The core code that actually governs how the inference engine schedules and runs—`csrc/` (**5,909 lines**), `engine/` (**1,957 lines**), and `worker/` (**2,805 lines**)—adds up to just **10,671 lines**!

In other words, once you master this 10,000-line core skeleton, you have mastered the essence of DeepInfer.

---

## 1.6 Lineage and Evolution from vLLM

Under the `deepinfer/` directory, 21 Python files retain a header comment at the top: `# Derived from vLLM; modified by DeepSeek.` (for example, `entrypoints/llm.py`, `engine/llm_engine.py`, and `sampling_params.py`). The acknowledgments section of `README.md` also candidly notes that DeepInfer originally evolved from an early version of vLLM while drawing inspiration from SGLang and TensorRT-LLM.

As a result, you will encounter many familiar faces as you read: `LLM.generate`, `SamplingParams`, `LLMEngine.step`, `gpu_memory_utilization`, and `profile_num_available_blocks`. Beneath those familiar entrypoints, however, the internal engine has been completely rebuilt from the ground up:

1. **The scheduler is pushed down into C++**, and every batch strictly separates prefill steps from decode steps—never mixing the two;
2. **GPU memory management is split into two distinct pools**: full/compressed attention layers use a paged block table (Paged KV), whereas Sliding Window Attention (SWA), KV compression boundary states, and DFlash draft layers use fixed-size per-request ring-buffer slots (State Pool);
3. **DFlash speculative decoding and Zero-Overhead Scheduling (ZOS) are natively and deeply co-designed**, right down to custom GPU kernels that patch `seqlens_k` in place on device memory.

In the next chapter, we will follow the call chain of `llm.generate(...)` and accompany a single request through its entire lifecycle inside DeepInfer.
