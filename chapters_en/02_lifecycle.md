# Chapter 2  The Life of a Request: From `generate` to Token Commit

When studying a complex system, the biggest pitfall is getting bogged down in the minutiae of a local helper function right away, losing sight of when and by whom it is called.

In this chapter, we set aside kernel-level details and instead attach a GPS tracker to a single `llm.generate(["Hello"])` call, following its complete lifecycle across **engine startup → request enqueueing → single-step scheduling & forward pass → sampling & token commit → deferred resource release**. Once we complete this journey, every module discussed in subsequent chapters will have a clear coordinate on our global roadmap.

---

## 2.1 Phase 1: Engine Startup (`LLM` and `LLMEngine.__init__`)

When you write `llm = LLM(model="deepseek-ai/DeepSeek-V4.1-Flash-1017")`, the entrypoint class `LLM.__init__` (`deepinfer/entrypoints/llm.py` L54–L87) assembles `EngineArgs` and instantiates the core `LLMEngine` object. Notice a crucial difference in default behavior between online and offline modes:

- **Online serving defaults to "abort on preemption"**: `EngineArgs.stop_preempted_request` defaults to `True` (`engine/arg_utils.py` L40). When GPU memory blocks run short, a preempted request is immediately terminated with `finish_reason = "insufficient_system_resource"`.
- **Offline `LLM` defaults to "recompute on preemption"**: `LLM.__init__` overrides this setting via `kwargs.setdefault("stop_preempted_request", False)` (L68), so that when memory is tight during offline batch processing, preempted requests are returned to the waiting queue to be prefilled again, ensuring no batch job is dropped (Chapter 3 explains why in detail).

Control then enters `LLMEngine.__init__` (`deepinfer/engine/llm_engine.py` L155–L305), which completes initialization across two macro-stages:

```{.mermaid caption="Figure 2-1  The Six Initialization Steps of LLMEngine (llm_engine.py L155–L305, L416–L458)"}
flowchart TB
  subgraph Row1["Stage 1: Tokenizer, CP Worker Spawning & Memory Capacity Profiling"]
    direction LR
    A1["1. Load Tokenizer & Multimodal Processor<br/>Spawn CP Subprocesses (rank 0 is driver)"] --> A2["2. Load Model Weights (load_model)<br/>Run profile_run to Measure Peak Memory"] --> A3["3. Enforce 80% Capacity Gate:<br/>max_model_len ≤ 0.8 × Total Block Capacity"]
  end

  subgraph Row2["Stage 2: Dual-Pool Allocation, Graph Warmup & C++ Scheduler Construction"]
    direction LR
    B1["4. init_cache_engine<br/>Allocate GPU KV Pool & State Pool"] --> B2["5. warm_up_model: Capture CUDA Graphs<br/>Profile Varlen Verification Decode Latency"] --> B3["6. Instantiate Two C++ BlockManagers,<br/>Scheduler & DeferredRelease"]
  end

  Row1 --> Row2
```

Looking at Figure 2-1, keep in mind three foundational designs that run through the entire book:

1. **How are processes organized?** Each Data Parallel (DP) rank is a completely independent `LLMEngine` process (offline scripts launch them via `launch_dp` in `examples/_launch.py`). Within a single DP process, child processes are spawned only when Context Parallelism (CP) is enabled: the `Worker` for `cp_rank = 0` lives directly inside the main process (acting as the `driver_worker`), while `Worker` instances for `cp_rank >= 1` run as child processes that receive method names and arguments broadcast from the driver via `cp_broadcast_object` (`worker/base.py` L290–L302).
2. **Why are there two `BlockManager` instances?** `self.block_manager` manages the paged KV pool sliced into fixed token blocks, whereas `self.state_manager` (L265–L266) manages the fixed-size state pool where each active request exclusively occupies a single slot (holding SWA sliding windows, KV compression boundary states, etc.). A request must acquire resources from both managers simultaneously before it can run (see Chapter 4).
3. **Why enforce an "80% capacity gate"?** Inside `_init_cache` (L445–L451), if the maximum sequence length of a single request (`runtime_max_model_len`) exceeds 80% of the total GPU paged KV capacity, the engine immediately raises a `ValueError` and refuses to start. This reserves headroom for concurrent requests and in-flight token reservations, preventing a single ultra-long sequence from deadlocking the entire memory pool (see Chapters 4 and 12).

---

## 2.2 Phase 2: Request Enqueueing and State Machine (`add_request`)

When `llm.generate(prompts, sampling_params)` is called, each prompt is tokenized and passed to `LLMEngine.add_request` (`llm_engine.py` L472–L631). Before a request ever reaches the C++ scheduler, the Python layer performs validation and preprocessing:

- **Sanity checks**: An empty prompt raises a `ValueError` (L553); if any out-of-vocabulary token ID is detected, the engine logs an error and **silently drops the request (returning `None`)** so a single malformed request cannot crash the entire batch (L580–L589).
- **Mutual exclusion between YoCo and `prompt_logprobs`**: When YoCo prefill truncation is enabled (Chapter 7), historical prefix tokens never compute logits in the second half of the model layers, so `prompt_logprobs` is forcibly disabled with a warning (L529–L538).
- **Asynchronous compilation for Structured Outputs (SO)**: If a request carries a JSON Schema or regex constraint (`so_args`), it **does not enter the scheduler immediately**. Instead, it is submitted to a background thread pool via `so_matcher_builder.submit(seq, so_args, _enqueue)` to compile the grammar automaton; only after compilation finishes does the callback `_enqueue` invoke `scheduler.add_sequence` (L606–L631). This implements the "decoupling of grammar compilation from inference" highlighted in the Official Q&A.

Once inside the C++ layer, each request is wrapped in a `Sequence` object (`csrc/engine/sequence.hpp`), whose lifecycle transitions among the following states:

```{.mermaid caption="Figure 2-2  Sequence State Machine (csrc/engine/sequence.hpp L30–L64)"}
flowchart LR
  subgraph Active["Active States (status < 3)"]
    direction TB
    W["WAITING (0)<br/>Awaiting prefill / Chunk incomplete /<br/>Preempted for recompute"]
    R["RUNNING (1)<br/>First or final prefill chunk scheduled,<br/>actively executing"]
    W -- "Final prefill chunk scheduled" --> R
    R -- "Preempted & recompute allowed" --> W
    W -- "Intermediate chunk done,<br/>pushed back to front of queue" --> W
  end

  subgraph Term["Terminal States (status ≥ 3, is_finished() == true)"]
    direction TB
    F1["FINISHED_STOPPED (3) → 'stop' (sampled EOS)"]
    F2["FINISHED_LENGTH_CAPPED (4) → 'length' (hit max_tokens or max_model_len)"]
    F3["FINISHED_ABORTED (5) / IGNORED (6) → Cancelled or prompt too long"]
    F4["FINISHED_PREEMPTED (7) → 'insufficient_system_resource' (abort on preemption)"]
    F5["FINISHED_SO_FAILED (8) / VE_FAILED (9) → Grammar or vision encoding failed"]
  end

  Active --> Term
```

Notice a subtle detail: although the `SequenceStatus` enum defines `SWAPPED = 2`, not a single line of code in the repository ever sets a sequence's status to `SWAPPED`—DeepInfer completely removed the slow path of swapping KV blocks out to host CPU memory. When GPU memory runs out, requests are either terminated outright (online) or have their KV discarded to be prefilled again later (offline). Checking whether a request has finished in C++ takes just a single comparison: `status >= SequenceStatus::FINISHED_STOPPED` (L44).

---

## 2.3 Phase 3: What Happens Inside a Single Step (`LLMEngine.step`)

After requests are enqueued, `LLM._run_engine` (`llm.py` L168–L194) repeatedly calls `LLMEngine.step()` in a loop until all requests finish. The cross-layer call sequence within a single step is shown below:

```{.mermaid caption="Figure 2-3  Single-Step Call Sequence of LLMEngine.step() (Synchronous View; See Chapter 5 for Async ZOS Overlap)"}
sequenceDiagram
  participant L as LLM._run_engine
  participant E as LLMEngine (Python)
  participant S as Scheduler (C++)
  participant W as Worker / ModelRunner
  L->>E: step()
  E->>S: schedule(sequence_metadata_pool)
  S-->>E: seq_metadata_list + SchedulerOutputs
  E->>W: execute_model_step(is_prompt, seq_metadata_list, ...)
  W->>W: 1. execute_main_model (GPU model forward + DFlash cache update)
  W->>W: 2. execute_sample (GPU main sampling + DFlash draft generation & sampling)
  W->>E: 3. pre_sampler_sync() (CPU reclaims previous step resources while awaiting GPU)
  W->>E: 4. post_process() → _process_model_outputs()
  E->>S: process_MTPInfo() + process_outputs() (commit KV counts & append tokens)
  E->>E: build_request_outputs() (assemble outputs from object pool)
  E-->>L: Return step RequestOutput list
```

Here lies an exceptionally clever engineering trick—**doing CPU housekeeping while the GPU is busy (`pre_sampler_sync` and `DeferredRelease`)**:

In step 2 of Figure 2-3, the Worker launches the sampling kernels on the GPU and must then wait for the GPU to copy the sampled token IDs back to host memory before running step 4 (`post_process`). During this exact window while waiting for the GPU, the Worker immediately invokes the engine callback `pre_sampler_sync()` (`worker.py` L162–L165, `llm_engine.py` L971–L982) to knock out three pieces of CPU housekeeping in one go:

1. Calls `deferred_release.release()`: returns the **previous step's** `SequenceMetadata` and `RequestOutput` objects back to their object pools and releases temporary references on forked blocks;
2. Executes `finalize_seq` for finished requests (actually invoking `free_seq` to return KV blocks and State slots to their free lists);
3. Calls `so_matcher_builder.pump()`, pushing requests whose grammar automata just finished compiling in background threads into the C++ scheduler.

As a result, the CPU overhead of memory reclamation and object-pool management is completely hidden inside the shadow of GPU sampling, adding zero latency to the critical path.

---

## 2.4 The Core Invariant: `cache.num_tokens` and Two-Phase Commit

Before reading Chapters 3, 5, and 6, you must understand the exact relationship between the two most important counters on a `Sequence`:

- `seq.get_len()` (i.e., `seq.input.tokens.size()`): the **total number of known tokens** currently in the sequence (including the prompt plus all output tokens generated and accepted so far).
- `seq.cache.num_tokens`: the **number of tokens that have already passed through the model forward pass and had their KV written into the KV Cache**.

Pause and consider a fundamental question: **For a request sitting in the `running` queue awaiting its next decode step, are `cache.num_tokens` and `get_len()` equal?**

The answer is **no—they always differ by 1**:

$$\text{cache.num\_tokens} = \text{seq.get\_len()} - 1$$

Why? Because the newest token (`last_output_token`) that was just sampled on the GPU at the end of the previous step has already been appended to the end of `seq.input.tokens` (increasing `get_len()` by 1), **but it has not yet been fed into the model as an input for a forward pass**, so its K and V vectors have not been computed yet! Only when the next decode step feeds that token into the model will its KV be written into the cache.

Once you grasp this "always off by 1" invariant, how `Scheduler::process_outputs` (`scheduler.hpp` L872–L1022) updates the ledger becomes crystal clear:

```{.mermaid caption="Figure 2-4  Per-Request Ledger Update Flow in process_outputs (scheduler.hpp L898–L1016)"}
flowchart TB
  subgraph Row1["Phase 1: In-Flight Settlement, Guard Checks & Optimistic Advance (advance_optimistically)"]
    direction LR
    P1["1. finish_inflight_step()<br/>Decrement step in-flight count"] --> P2["2. If finished or preempted in-flight:<br/>Free slots & discard step outputs"] --> Q1["3. advance_optimistically(q)<br/>Save old value & set num_tokens += q"]
  end

  subgraph Row2["Phase 2: Chunk Check, Token Appending & Exact Rollback Commit (commit_processed_tokens)"]
    direction LR
    Q2["4. If num_tokens < len (chunk incomplete):<br/>Reset to WAITING & push to queue front"] --> R1["5. If complete: move Prompt seq to running;<br/>append accepted drafts + main token & check stop"] --> R3["6. Decode step runs commit_processed_tokens:<br/>num_tokens = old + appended (preserving = len - 1)"]
  end

  Row1 --> Row2
```

Let us trace both a **Prefill step** and a **Decode step** through Figure 2-4 using concrete numbers:

1. **The step where Prefill completes**:
   - Suppose a prompt has length $10$ (`get_len() = 10`, `num_tokens = 0`) and finishes in a single prefill step (`seqlen_q = 10`).
   - Calling `advance_optimistically(10)` advances `num_tokens` to $10$.
   - Next, the first output token sampled by the model is appended to the end of the sequence, so `get_len()` becomes $11$.
   - No further adjustment is needed—the invariant $\text{num\_tokens} (10) = \text{get\_len()} (11) - 1$ holds naturally! At this exact moment, the request is moved into the `running` queue (L927).
2. **A Decode step with speculative decoding**:
   - Entering the step, `get_len() = 11` and `num_tokens = 10` (the 11th token has not been fed through the model yet), accompanied by $5$ candidate draft tokens.
   - During this step, the model forward pass verifies `seqlen_q = 1 + 5 = 6` tokens in parallel (1 main input token + 5 draft tokens).
   - Calling `advance_optimistically(6)` (`sequence.hpp` L216) saves the pre-step value `previous_num_tokens = 10` and optimistically advances `num_tokens` to $10 + 6 = 16$.
   - Suppose verification **accepts the first 2 of the 5 drafts, plus 1 bonus token sampled by the target model, appending a total of `token_cnt = 3` new tokens** to the sequence. Now `get_len()` grows from $11$ to $11 + 3 = 14$.
   - The remaining 3 rejected drafts wrote KV entries during the forward pass, but those entries are invalid garbage! So at the end of the step, the scheduler calls `commit_processed_tokens(3)` (`sequence.hpp` L220):
     $$\text{num\_tokens} = \text{previous\_num\_tokens} (10) + \text{token\_cnt} (3) = 13$$
   - Look at that: $\text{num\_tokens} (13) = \text{get\_len()} (14) - 1$ holds once again! The KV slots written by the 3 rejected drafts do not need to be zeroed out; the next decode step will simply overwrite them starting at position 13.

---

## 2.5 Chapter Summary: Division of Labor Between Python and C++

We can condense the path traversed in this chapter into a four-layer architectural diagram:

```{.mermaid caption="Figure 2-5  Four-Layer Division of Labor and Language Boundary in DeepInfer"}
flowchart TB
  subgraph L1["1. Python Entrypoint & Orchestration Layer (entrypoints / engine)"]
    direction LR
    LLM["LLM (Offline Batch Loop)"] --> Engine["LLMEngine (Lifecycle, Async SO Compile, DeferredRelease)"]
  end

  subgraph L2["2. C++ High-Speed Scheduling & Metadata Layer (csrc/engine, pybind11)"]
    direction LR
    Sched["Scheduler (Queues, Step Voting, Preemption, Ledger Commit)"]
    BM["BlockManager × 2 (Paged KV Pool & Fixed-Slot State Pool)"]
    Prep["zos.cpp / prepare_tensor_input_v4 / mtp_varlen"]
    Sched --> BM
    Sched --> Prep
  end

  subgraph L3["3. Python Worker & Model Execution Layer (worker / model_executor)"]
    direction LR
    Worker["Worker (Multi-Process Sync, CUDA Graph, ZOS Pipeline)"] --> Runner["ModelRunner → DeepseekForCausalLM + Sampler + DFlash"]
  end

  subgraph L4["4. Low-Level Kernels & Communication Libraries (vendor + tilelang_ops)"]
    direction LR
    Kernels["FlashMLA · DeepGEMM · DeepEP · DeepSelect · TileLang (CUDA / Ascend)"]
  end

  L1 --> L2
  L1 --> L3
  L3 --> L2
  L3 --> L4
```

Remember this division of labor: **C++ handles all high-frequency loops and state ledgers (at microsecond scale), Python handles workflow orchestration and tensor plumbing, and low-level kernels handle matrix multiplications and collective communication**. In the next chapter, we lift the lid on `csrc/engine/scheduler.hpp` to see how the C++ scheduler selects the batch of requests at every single step.
