# DeepInfer Source Code Walkthrough & From 0 to 1

<p align="center">
  🌐 <b>English</b> &nbsp;|&nbsp; <a href="README_zh.md">🇨🇳 <b>简体中文</b></a>
</p>

<p align="center">
  Author: <b>Hawk Wu</b><br/>
  🌐 <b>English Edition (Default)</b>:
  <a href="https://cluster1900.github.io/deepinfer-reader/"><b>📖 Read Online (Interactive Web)</b></a> ·
  <a href="https://cluster1900.github.io/deepinfer-reader/deepinfer-reader-en.epub"><b>📱 Download EPUB (EN)</b></a> ·
  <a href="https://cluster1900.github.io/deepinfer-reader/deepinfer-reader-en.pdf"><b>📄 Download PDF (EN)</b></a>
  <br/>
  🇨🇳 <b>中文版 (Chinese Edition)</b>:
  <a href="https://cluster1900.github.io/deepinfer-reader/index-zh.html"><b>📖 在线阅读（交互式网页版）</b></a> ·
  <a href="https://cluster1900.github.io/deepinfer-reader/deepinfer-reader.epub"><b>📱 下载中文 EPUB</b></a> ·
  <a href="https://cluster1900.github.io/deepinfer-reader/deepinfer-reader.pdf"><b>📄 下载中文 PDF</b></a> ·
  <a href="https://github.com/cluster1900/deepinfer-reader/releases"><b>📦 GitHub Releases</b></a>
</p>

A hands-on, code-first technical book that deconstructs DeepSeek's open-source production LLM inference engine **[DeepInfer](https://github.com/deepseek-ai/DeepInfer)** (verified line-by-line against commit `553741557a57df44a46fb3ab72b6aafeaf945337`) and guides you through building an isomorphic teaching engine, `minidi`, from scratch in pure Python + NumPy. Available in both **English (`chapters_en/`)** and **Chinese (`chapters/`)**.

Every chapter follows a four-step pedagogical rhythm: **Real-World Pain Point → Small-Number Visual Walkthrough → Line-by-Line C++/Python Source Verification → Hands-On Code & Failure Post-Mortems**.

---

## Highlights

- **Part I (Chapters 1–8): Deconstructing Production DeepInfer Source Code**
  - **Positioning & Deployment Topology**: Why does DeepInfer drop Tensor Parallelism (TP) in favor of DP/CP + Wide-EP + E/P/D disaggregation? Which 10,000 lines out of the 60,000-line codebase form the core scheduling and runtime backbone?
  - **C++ Scheduler & Two-Pointer Preemption**: Why are Prefill steps and Decode steps strictly separated? How do multiple DP ranks align step types via `ALL(70)` / `ANY(100)` two-round voting? Why does Decode sort by descending `output_len`, advance the right pointer to protect newcomers (LCFS), and advance the left pointer to preempt the longest-output sequences first?
  - **Dual-Pool GPU Memory Ledger**: Why does compressed KV use a 128-token paged block pool while Sliding Window Attention (SWA) uses a fixed-size ring buffer? Why is the compressed-state ring `ratio + 5` while the main-model SWA ring is `128 + 6 = 134`?
  - **Zero-Overhead Async Scheduling (ZOS)**: When the CPU schedules step $N+1$ before knowing step $N$'s sampling results, how do dual in-flight counters, GPU-side `remap`, and the `compute_seqlens_k_zos` formula keep attention context lengths exact?
  - **DFlash Speculative Decoding & Variable-Length Verification**: How does DFlash extract 5 draft features in a single parallel forward pass and sample sequentially via a 2-layer Markov head (`markov_w1/w2`) and `ConfidenceHead`? How does greedy budget allocation maximize accepted tokens per millisecond against a stepped CUDA Graph latency curve?
  - **YoCo Dual-Stage Architecture & EPLB Load Balancing**: Why can non-final chunks in chunked prefill skip the entire second-half Consumer stack? How does EPLB eliminate MoE stragglers via peak-load greedy replication and hierarchical placement?
- **Part II (Chapters 9–16): Building the CPU Teaching Engine `minidi` from 0 to 1**
  - In **~870 lines of pure Python + NumPy** (zero GPU dependencies), build `TinyLM`, `PagedKV`, `BlockManager`, `Scheduler`, `ZOSEngine`, speculative decoding with variable-length verification, `SWARing`, `EPLB`, and multi-rank voting across 7 progressive stages.
  - Backed by **64 `pytest` unit tests** (running in under 1 second on a laptop CPU), including post-mortems of real engineering traps encountered during development: missing the 80% capacity gate deadlock, head-of-line deadlock between half-prefilled chunks and preempted sequences, and in-flight preemption counter corruption.

---

## Table of Contents

| Part | English Edition (`chapters_en/`) | 中文版 (`chapters/`) | Core Topics & Companion Code |
| --- | --- | --- | --- |
| **Preface** | [Preface](chapters_en/00_preface.md) | [前言](chapters/00_preface.md) | Why read DeepInfer, dual-track structure, and 4-step reading method |
| **Part I**<br/>Inside DeepInfer | [Chapter 1](chapters_en/01_overview.md) | [第 1 章](chapters/01_overview.md) | **Positioning & Repo Map**: E/P/D disaggregation, submodules & the 10K-line core · `data/deepinfer_loc.json` |
| | [Chapter 2](chapters_en/02_lifecycle.md) | [第 2 章](chapters/02_lifecycle.md) | **The Life of a Request**: 4-phase startup, state machine & the `num_tokens = len - 1` invariant |
| | [Chapter 3](chapters_en/03_scheduler.md) | [第 3 章](chapters/03_scheduler.md) | **The C++ Scheduler**: Step separation, 2-round voting, FCFS chunking, LCFS preemption & tail-block reclamation · `csrc/engine/scheduler.hpp` |
| | [Chapter 4](chapters_en/04_memory.md) | [第 4 章](chapters/04_memory.md) | **GPU Memory & KV Cache**: Paged block pool vs. fixed state pool, deriving `134` and `ratio+5` · `cache_engine.py` |
| | [Chapter 5](chapters_en/05_zos.md) | [第 5 章](chapters/05_zos.md) | **Zero-Overhead Scheduling (ZOS)**: 4-stage pipeline, dual in-flight counters & `seqlens_k` patching · `worker.py` · `zos.cpp` |
| | [Chapter 6](chapters_en/06_varlen.md) | [第 6 章](chapters/06_varlen.md) | **DFlash & Variable-Length Verification**: Parallel feature extraction + Markov head + greedy bucket selection · `dflash.py` · `mtp_varlen.hpp` |
| | [Chapter 7](chapters_en/07_model_parallel.md) | [第 7 章](chapters/07_model_parallel.md) | **Model, Kernels & Parallelism**: YoCo dual-stage truncation, Wide-EP, Mega MoE & EPLB · `v4_modules/` · `eplb/policy.py` |
| | [Chapter 8](chapters_en/08_run_compare.md) | [第 8 章](chapters/08_run_compare.md) | **Configuration, Determinism & Comparison**: 12 key flags, batch invariance & comparison with vLLM/SGLang · `environ.py` |
| **Part II**<br/>Building `minidi` | [Chapter 9](chapters_en/09_roadmap.md) | [第 9 章](chapters/09_roadmap.md) | **From 0 to 1 Roadmap**: 4 engineering rules & 7-stage evolution blueprint · `examples/quickstart.py` |
| | [Chapter 10](chapters_en/10_minimal_engine.md) | [第 10 章](chapters/10_minimal_engine.md) | **Stage 1 — Minimal Engine**: `TinyLM`, flat varlen batching, `Engine.step` & position-seeded sampling · `minidi/model.py` · `engine.py` |
| | [Chapter 11](chapters_en/11_paged_kv.md) | [第 11 章](chapters/11_paged_kv.md) | **Stage 2 — Paged KV & Block Manager**: Block tables, FIFO free list & the 80% capacity gate deadlock · `minidi/blocks.py` · `sequence.py` |
| | [Chapter 12](chapters_en/12_scheduler.md) | [第 12 章](chapters/12_scheduler.md) | **Stage 3 — Scheduler & Preemption**: Two-pointer preemption, tail-block reclamation & chunked-prefill deadlock fix · `minidi/scheduler.py` |
| | [Chapter 13](chapters_en/13_zos.md) | [第 13 章](chapters/13_zos.md) | **Stage 4 — Async `ZOSEngine`**: `self.prev` isolation & in-flight preemption guardrails · `minidi/zos.py` |
| | [Chapter 14](chapters_en/14_speculative.md) | [第 14 章](chapters/14_speculative.md) | **Stage 5 — Speculative Decoding & Varlen**: Single-step parallel verification, rollback & greedy optimality · `minidi/varlen.py` |
| | [Chapter 15](chapters_en/15_swa_eplb.md) | [第 15 章](chapters/15_swa_eplb.md) | **Stage 6 & 7 — SWA Ring, EPLB & Voting**: $W+k-1$ rollback bound, greedy bin packing & step voting · `minidi/swa.py` · `eplb.py` · `vote.py` |
| | [Chapter 16](chapters_en/16_toward_gpu.md) | [第 16 章](chapters/16_toward_gpu.md) | **Toward Production GPU**: Gap analysis & 5-step roadmap to GPU kernels |
| **Appendix** | [Credits & Notes](chapters_en/99_credits.md) | [致谢与勘误](chapters/99_credits.md) | Acknowledgments, references & 2 verified comment-vs-code discrepancies in upstream |

---

## 3-Minute Quickstart

```bash
# 1. Create a virtual environment and install lightweight dependencies (numpy, matplotlib, pytest)
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-examples.txt

# 2. Run all 64 unit tests (finishes in < 1 second on CPU)
python3 -m pytest

# 3. Run the quickstart demo comparing sync, async ZOS, and speculative decoding engines
python3 examples/quickstart.py

# 4. Run minidi on CPU to generate all bilingual data plots
python3 scripts/figures/make_plots.py
```

---

## Build the Book (English & Chinese Web / EPUB / PDF)

```bash
# Prerequisites: pandoc >= 3.0, Node >= 22, Chrome/Chromium, CJK fonts, poppler-utils
PUPPETEER_SKIP_DOWNLOAD=1 npm ci
python3 scripts/check_book.py      # Quality gate: structure, figure counts, pinned commit SHA
python3 build.py                   # Builds bilingual HTML, EPUB, and PDF into dist/
python3 scripts/check_outputs.py dist --min-figures 28
```

---

## License

- Companion code (`minidi/`, `tests/`, `examples/`, `scripts/`) is licensed under the **MIT License** (see [`LICENSE`](LICENSE)).
- Book prose and original diagrams are licensed under **CC BY 4.0**.
- Quoted DeepInfer source snippets remain under copyright of DeepSeek-AI (Apache License 2.0).
