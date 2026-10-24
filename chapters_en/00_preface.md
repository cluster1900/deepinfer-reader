# Preface: Why Read DeepInfer, and How to Read This Book

If you have ever written or read the source code of an LLM inference engine, you have likely experienced this exact frustration:

- On paper, PagedAttention, Speculative Decoding, Zero-Overhead Scheduling (ZOS), and MoE Expert Parallelism (EP) are each crystal clear in isolation.
- Yet the moment you pack all of them into **a single production-grade engine**, their boundary conditions collide head-on: When the CPU schedules step $N+1$, it still does not know how many draft tokens step $N$ accepted on the GPU—so how should KV blocks be reserved? Sliding Window Attention (SWA) keeps only the most recent 128 positions—so if speculative decoding rejects 4 draft tokens, how do you roll back an overwritten window? When running MoE across multiple Data Parallel (DP) ranks, what happens if some ranks want to run prefill while others want to run decode—how do collective communications stay aligned?

**DeepInfer**, open-sourced by DeepSeek, is a battle-tested answer sheet that lays every single one of these problems right out on the table. Unlike general-purpose frameworks that strive to support hundreds of model architectures, DeepInfer serves a single, razor-focused goal: squeezing maximum throughput out of **DeepSeek-V4.1-Flash**—a model combining hybrid attention, large-scale MoE, and built-in DFlash speculative decoding. The entire repository is roughly 60,000 lines of code, and the actual C++/Python scheduling and runtime backbone is **only around 10,000 lines**, making it ideal to read from cover to cover.

---

## 0.1 How This Book Is Organized: Deconstruct a Real Engine, Then Build One Yourself

To ensure our source-code walkthrough neither degenerates into a dry laundry list of line numbers nor drifts into high-level hand-waving detached from code, this book adopts a dual-track structure—**"left hand pointing at production source code, right hand building an engine from scratch"**:

- **Part I (Chapters 1–8): Deconstructing Production DeepInfer**. We start with a panoramic view of DeepInfer's positioning and production deployment topology (Chapter 1), then follow a single request through its entire lifecycle from enqueueing to token emission (Chapter 2). Next, we conquer five core subsystems one by one: the **C++ Scheduler** (Chapter 3), the **Hybrid KV Cache & SWA State Pool** (Chapter 4), **Zero-Overhead Asynchronous Scheduling (ZOS)** (Chapter 5), **DFlash Speculative Decoding & Confidence-Scheduled Variable-Length Verification** (Chapter 6), and the **YoCo Two-Stage Architecture & DP/CP/EP Parallelism** (Chapter 7), before wrapping up with configuration switches and a cross-framework comparison (Chapter 8).
- **Part II (Chapters 9–16): Building the Educational Engine `minidi` from 0 to 1**. Reading someone else's code is never enough. Under the `minidi/` directory, using **roughly 750 lines of pure Python + NumPy**, we build a miniature inference engine across 7 stages that runs right on a laptop CPU. Every hardcore mechanism you read about in Part I—block-table allocation, two-pointer preemption, optimistic advancement and commit rollback, ZOS in-flight position correction, greedy variable-length verification, SWA ring-buffer rollback, and EPLB expert replication and placement—has a direct, runnable implementation and unit test suite in `minidi` (64 `pytest` test cases in total, finishing in under 1 second).

```{.mermaid caption="Figure 0-1  Reading Map of This Book: Part I Deconstructs DeepInfer Source Code, Part II Reproduces It 1-to-1 in minidi"}
flowchart TB
  subgraph P1["Part I: Deconstructing DeepInfer Production Source (Ch 1–8)"]
    direction LR
    C1["Ch 1–2<br/>Global Positioning &<br/>Request Lifecycle"] --> C2["Ch 3–4<br/>C++ Scheduler &<br/>Dual Memory Pools"]
    C2 --> C3["Ch 5–6<br/>ZOS Async Pipeline &<br/>DFlash Speculation"]
    C3 --> C4["Ch 7–8<br/>YoCo Parallelism,<br/>EPLB & Comparison"]
  end

  subgraph P2["Part II: Building the CPU Educational Engine minidi from 0 to 1 (Ch 9–16)"]
    direction LR
    D1["Ch 9–10<br/>Roadmap & Minimal<br/>Step Engine"] --> D2["Ch 11–12<br/>Paged KV &<br/>Two-Pointer Preemption"]
    D2 --> D3["Ch 13–14<br/>ZOSEngine &<br/>Varlen Verification"]
    D3 --> D4["Ch 15–16<br/>SWA / EPLB &<br/>Path to GPU"]
  end

  P1 -. "1-to-1 Mechanism & Line-Number Mapping" .-> P2
```

---

## 0.2 Narrative Style of Each Chapter

To match natural human intuition when learning systems, every chapter in this book follows a four-step progression:

1. **Ask "Why" First**: If we did not design it this way, under what exact workload would the naive approach crash, stall, or waste GPU memory?
2. **Walk Through with Small Numbers**: Before looking at code, we trace pointer movements and tensor shapes by hand using crystal-clear toy numbers like `block_size = 4`, window size `W = 4`, and draft count `k = 3`.
3. **Inspect the Real Source Code**: Armed with concrete arithmetic intuition, we examine the corresponding C++ / Python functions in DeepInfer, watching every branch and constant snap cleanly into place.
4. **Verify Hands-On and Analyze Edge Cases**: We run the corresponding logic in `minidi` and explore boundary conditions and engineering trade-offs under extreme load.

---

## 0.3 Version Baseline and Evidence Conventions

All DeepInfer source code analyzed in this book is pinned to commit **`9bd9cf243fc53ef9afbf8f04d391d989117d6ab9`** (2026-10-21, `Bump DeepEP to f6aa38fe and drop the SSH->HTTPS rewrite (#157)`) of the official repository `https://github.com/deepseek-ai/DeepInfer`. Every file path, function name, line number, and default value in this book has been verified line by line against this commit.

- **Line-number notation**: Citations like `scheduler.hpp L275` refer to line 275 of `csrc/engine/scheduler.hpp`; when directory prefixes are omitted, they follow the context of the current chapter.
- **Official Q&A**: The repository includes `doc/inference_deployment_qa.md` (referred to throughout this book as the **"Official Q&A"**; upstream first shipped it as the PDF `doc/DeepSeek V4.1 Flash 推理部署 Q&A.pdf` in `8b16149`, then replaced it in `774ce24` with an English Markdown file and added `doc/images/architecture_overview.png` and `wide_ep_decode_pareto.png`), which discloses extensive production deployment details (such as E/P/D disaggregation ratios, 3FS distributed KV storage, and the rationale behind 128-token persistence granularity). All facts and figures cited from that document are explicitly labeled as "Official Q&A".
- **Experimental boundary**: Running production DeepInfer requires NVIDIA SM100 (Blackwell) or Huawei Ascend 950 hardware along with full model weights. Our analysis of DeepInfer itself relies strictly on static source verification cross-checked against the Official Q&A, whereas all experimental curves and plots in this book are generated by actually executing the `minidi` engine on a CPU in Part II.

---

## 0.4 Run the Companion Code in Three Minutes

As you read Part I or Part II, we encourage you to keep a terminal open and run the `minidi` test suite and demo script along the way:

```bash
git clone https://github.com/cluster1900/deepinfer-reader
cd deepinfer-reader
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-examples.txt

# 1. Run all 64 unit tests (verifying paged KV, preemption, ZOS, speculative decoding, SWA, and EPLB)
python3 -m pytest

# 2. Run the quickstart demo script to inspect module outputs interactively
python3 examples/quickstart.py
```

`requirements-examples.txt` depends only on three lightweight packages—`numpy`, `matplotlib`, and `pytest`—and finishes all unit tests in under 1 second on any laptop CPU. Now, let us turn to Chapter 1 and survey the global landscape of DeepInfer.
