# Chapter 3  The Scheduler: Who Runs Each Step, How Much to Run, and Whom to Preempt

If an inference engine is a restaurant kitchen, the GPU is the stove, and the KV Cache is the cutting-board space, then **the `Scheduler` is the head chef**. After every dish leaves the wok, the head chef has just a few microseconds to make three decisions:

1. **For this next batch, are we prepping raw ingredients (Prefill) or continuing to stir-fry dishes already on the stove (Decode)?**
2. **If we are prepping raw ingredients, how much do we chop at once, and what if the cutting board runs out of space?**
3. **If we are stir-frying and discover mid-step that the cutting board cannot fit the new garnish, which table's dish gets pulled off first (Preemption)?**

DeepInfer's scheduler is implemented entirely in C++ inside `csrc/engine/scheduler.hpp` (1,329 lines). In this chapter, we follow its core entrypoint `schedule()` and thoroughly unpack the design trade-offs and source-code mechanics behind all three questions.

---

## 3.1 Three Queues: Where Do Requests Live Inside the Scheduler?

Open `scheduler.hpp` L1124–L1126, and you will see that the scheduler holds three queues:

| Queue Name | C++ Container Type | What Kind of Requests Live Here? |
| --- | --- | --- |
| `waiting` | `std::deque<SequencePtr>` | Brand-new requests awaiting prefill, chunked-prefill requests that are only partially prefilled, and decode requests preempted back for recomputation |
| `running` | `std::vector<SequencePtr>` | Active requests that have completed prefill and are decoding new tokens step by step |
| `prompted` | `std::deque<tuple<SequencePtr, bool>>` | Requests that **already possess prefix KV** and can enter decode directly without local prefill (the boolean flags whether the request was cloned via `fork_seq`) |

While the first two queues are familiar, the third queue—`prompted`—serves as a dedicated "fast lane" for two specific scenarios:

- **Online PD Disaggregation**: Once a Prefill instance (PE) finishes computing a request's prompt, it transfers the KV Cache and first token to a Decode instance (CE). The CE places the request straight into `prompted` so it can join `running` on the very next step and begin decoding.
- **Local Prefix Cloning (`fork_seq`, L1062–L1122)**: During benchmarking or parallel sampling, you can prefill a long prompt once on a parent sequence and then call `fork_seq` to zero-copy clone dozens of child sequences. Each child sequence shares the parent's prefix KV blocks via `pin_blocks` (incrementing block reference counts by 1) and enters `prompted`; when it actually steps into decode, only the final partially filled block is duplicated (Copy-on-Write).

---

## 3.2 Iron Rule 1: Prefill Steps and Decode Steps Never Mix

Many mainstream open-source engines (such as vLLM V1) default to "hybrid chunked-prefill batching": packing a chunk of a long prompt together with dozens of single-token decode requests into the exact same forward pass.

Look at DeepInfer's `_schedule()` (`scheduler.hpp` L275–L345), however, and you will find a diametrically opposite iron rule: **every step across the entire parallel group runs either exclusively Prefill (`prompt_run = true`) or exclusively Decode (`prompt_run = false`)—never mixing the two in the same batch!**

Why does DeepInfer separate them so strictly? There are two reasons:

1. **Radically different kernel profiles and CUDA Graph shapes**: In a Decode step, each request has only $1 \sim 6$ query tokens, running inside tightly tailored CUDA Graphs with FlashMLA decode kernels specialized for short queries. In a Prefill step, there are thousands of query tokens executing compute-bound large GEMMs and YoCo cross-layer caching. Separating them lets both kernel regimes operate at peak efficiency without interfering with each other.
2. **Multi-GPU DP running MoE requires lockstep synchronization**: When Expert Parallelism (EP) is enabled, requests across different DP ranks must execute an all-to-all (`dispatch` / `combine`) collective via `DeepEP` at every MoE layer. If Rank 0 were running a heavy Prefill while Rank 1 were running a fast Decode inside a CUDA Graph, the two ranks would desynchronize or stall waiting at the collective barrier.

Consequently, under the `NORMAL` engine mode used for offline inference, **all DP ranks must vote via `allreduce` at fixed step intervals to jointly decide whether the next step is a collective Prefill step or a collective Decode step**:

```{.mermaid caption="Figure 3-1  Multi-GPU Step-Type Decision and Two-Round Voting in NORMAL Engine Mode (scheduler.hpp L295–L344)"}
flowchart TB
  subgraph Check["Phase 1: Check Voting Interval and Local Prefill Willingness"]
    direction LR
    S0["Check if decode_step hits<br/>ALL(70) or ANY(100) interval"] --> W["Compute local flag want:<br/>KV usage ≤ 0.9 & waiting schedulable"] --> V1{"Round 1 allreduce Vote<br/>ALL requires all want / ANY requires any want"}
  end

  subgraph Vote2["Phase 2: Trial Scheduling and Round-2 Confirmation (Fallback to Decode if Rejected)"]
    direction LR
    SP["If Round 1 passes: ranks with want=true<br/>run _schedule_prefill() trial"] --> V2{"Round 2 allreduce<br/>Did at least one rank schedule ≥1 seq?"}
    V2 -- "Yes" --> R_YES["All ranks enter Prefill step (idle ranks run dummy step)<br/>and reset decode_step = 0 across all ranks"]
    V2 -- "No" --> D_RUN["All ranks fall back to Decode step:<br/>run _schedule_decode(), decode_step += 1"]
  end

  Check --> Vote2
```

Tracing Figure 3-1 reveals how carefully this voting protocol is engineered (constants defined in `deepinfer/environ.py` L174–L177):

- **Why have both an `ALL` point at 70 steps and an `ANY` point at 100 steps?** Active decode requests are highly sensitive to time-between-tokens (TBT). Every 70 decode steps (`DEEPINFER_PREFILL_ATTEMPT_INTERVAL = 70`), if every DP rank has new requests waiting (unanimous `want`), the whole group switches over to run prefill together. If load is skewed and only a single rank has waiting requests—failing unanimity—then at step 100 (`DEEPINFER_PREFILL_ANY_ATTEMPT_INTERVAL = 100`), the `ANY` check forces a switch, and the remaining ranks execute a dummy empty prefill step to keep collective communications aligned.
- **Why do Prefill steps run back-to-back once started?** Whenever a prefill step executes, `schedule()` (L676–L677) resets `decode_step` to 0. On the very next step, `decode_step == 0` immediately hits `0 % 70 == 0` (`ALL` point)! As long as the waiting queues are not yet drained and KV block usage remains below the 90% watermark (`GPU_CACHE_PREFILL_WATERMARK = 0.9f`), the engine chains consecutive prefill steps to drain the backlog in one go.
- **Why is a second voting round necessary?** Imagine an edge case: a rank sees a non-empty `waiting` queue (so it votes `want = true` in Round 1), but once inside `_schedule_prefill`, it discovers that its `running` concurrency slots are already full (`max_num_seqs` reached), so it actually schedules zero sequences. Without Round 2 confirming that at least one rank scheduled a non-empty batch, all ranks would waste an empty prefill step and reset `decode_step` to 0, causing the exact same vote and empty step to repeat on the next iteration in an infinite livelock (see comments at L328–L333).

---

## 3.3 Prefill Scheduling: Strict FCFS and Chunk Truncation

Once the vote selects a Prefill step, `_schedule_prefill` (`scheduler.hpp` L351–L554) begins pulling requests from the front of the `waiting` queue. Its guiding principle is four words: **strict FCFS (First-Come, First-Served)**—the moment the head-of-line request cannot be scheduled due to insufficient resources, the loop executes `break` immediately, **never skipping over the head of the queue to cherry-pick smaller requests behind it**, thereby completely preventing starvation of long prompts.

The admission and chunking workflow for the head-of-line request `seq` proceeds in four steps:

```{.mermaid caption="Figure 3-2  Four-Step Admission and Chunking Flow in _schedule_prefill (scheduler.hpp L362–L550)"}
flowchart TB
  subgraph Row1["Phase 1: Queue-Head Cleanup, Readiness & Capacity Check (Break Immediately on LATER for Strict FCFS)"]
    direction LR
    A1["1. Peek waiting front seq;<br/>move to ignored if aborted or too long"] --> A2["2. Check multimodal schedulable<br/>& _can_allocate_with_state"] --> A3["3. If LATER, break immediately;<br/>if OK, proceed to budget slicing"]
  end

  subgraph Row2["Phase 2: Batch Budget Slicing, Upfront Full-Block Allocation & Queue Pop"]
    direction LR
    B1["4. Verify concurrency ≤ max_num_seqs<br/>& cumulative seqlen_k within limit"] --> B2["5. Compute step seqlen_q:<br/>align to 2×cp & avoid splitting image tokens"] --> B3["6. Allocate all blocks upfront on first chunk;<br/>pop front, and stop if chunked"]
  end

  Row1 --> Row2
```

Four critical engineering details stand out in this routine:

1. **Chunked computation, but upfront full-length block allocation**: Even if an 8,000-token prompt is sliced into multiple chunks across several steps, when its **very first chunk** is scheduled, `acquire.allocate_seq(*seq)` (L518–L520) allocates all KV blocks and the State slot required for the entire 8,000-token prompt upfront! Subsequent chunks (chunk 2, chunk 3, etc.) allocate zero additional blocks. The benefit is decisive: **once a request begins its first chunk, it can never deadlock mid-prefill on a later chunk due to running out of memory blocks**.
2. **Dual budget caps to bound step latency**: A prefill step caps not only the total number of new query tokens (`max_num_batched_tokens`) but also—via `prefill_max_sum_seqlen_k` (L358, L444–L449)—the cumulative attention context length $\sum \text{seqlen\_k}$ across all requests in the batch, preventing attention latency from blowing up on the final chunks of ultra-long sequences.
3. **CP mirror alignment and multimodal token protection**: When Context Parallelism (CP) is enabled, chunk sizes are rounded down to a multiple of `2 * cp_size` (L470–L473) because DeepInfer uses head-tail symmetric Mirror CP (Chapter 7). For multimodal requests, `_adjust_chunk_size` (L235–L260) nudges the split point so it never slices through the middle of a contiguous run of `IMAGE_PAD` tokens belonging to the same image.
4. **How do incomplete chunks resume?** If a request was chunked (`is_chunked == true`), after the step finishes `process_outputs` sees `cache.num_tokens < seq.get_len()`, resets its status to `WAITING`, and pushes it right back to the **front** of the `waiting` queue (`waiting.push_front`, L950), guaranteeing that the next prefill step immediately picks up where it left off.

---

## 3.4 Decode Scheduling: Why Is It Called LCFS, and Who Gets Preempted When Blocks Run Out?

When control enters `_schedule_decode` (`scheduler.hpp` L559–L660), the scheduler first removes any finished sequences from `running` and merges newly ready sequences from `prompted` into `running`.

Now comes the climax: every request in `running` wants to step forward (generating 1 main token plus verifying up to 5 draft tokens). Whenever a request's last KV block is full, it must request 1 new physical block from `BlockManager`. **What if the free block pool is exhausted and there are not enough blocks for everyone—whom do we protect, and whom do we preempt?**

Let us look directly at the core two-pointer algorithm in `scheduler.hpp` L622–L647:

```cpp
// LCFS (distilled core logic from scheduler.hpp L622-L647)
std::stable_sort(running.begin(), running.end(),
                 [](const SequencePtr& a, const SequencePtr& b) {
                     return a->get_output_len() > b->get_output_len();
                 });
size_t l = 0, r = running.size();
while (l < r && scheduled.size() < max_num_seqs) {
    SequencePtr seq = running[r - 1];                  // Pick from the right end (shortest output) first
    while (!_can_append_slot(*seq) && r - l > 1) {
        _preempt(running[l], ...);                     // When blocks run short, preempt from the left end (longest output)!
        l++;
    }
    if (_can_append_slot(*seq)) {
        _append_slot(*seq, ...);
        scheduled.push_back(seq);
    } else {
        _preempt(seq, ...);
    }
    r--;
}
```

Visualizing this loop as a two-pointer scan makes the mechanics unmistakable:

```{.mermaid caption="Figure 3-3  Two-Pointer Mechanism in Decode Scheduling: Right Pointer Prioritizes Newest Requests (LCFS), Left Pointer Preempts Longest-Output Heavyweights"}
flowchart TB
  subgraph Sorted["1. Sort running Queue in Descending Order of Generated Length (output_len)"]
    direction LR
    S0["running[0]: out=900<br/>(Earliest in decode, holds most blocks)"] --- S1["running[1]<br/>out=420"] --- S2["running[2]<br/>out=130"] --- S3["running[3]<br/>out=12"] --- S4["running[4]: out=3<br/>(Latest in decode, holds fewest blocks)"]
  end

  subgraph Ptrs["2. Converging Left and Right Pointers (while l < r)"]
    direction LR
    LP["Left pointer l (moves right from 0):<br/>When free blocks run short, _preempt(running[l]) first;<br/>evicting 1 long-output heavyweight frees many KV blocks"] --- RP["Right pointer r-1 (moves left from end):<br/>Each iteration picks running[r-1] for _append_slot first,<br/>implementing LCFS (newest short decode requests go first)"]
  end

  Sorted --> Ptrs
```

Anyone reading this code for the first time naturally asks two questions:

### Question 1: Why does line 622 have the comment `// LCFS` (Last-Come, First-Served)?
Look at the right pointer `r - 1` in Figure 3-3: once `running` is sorted in descending order of generated token count `output_len`, which request has the smallest `output_len`? Clearly **the request that just finished prefill and entered the decode phase most recently** (e.g., `output_len = 3`)! Because the right pointer `r - 1` scans from the rightmost end toward the left to admit requests into the batch, **the request that arrived in decode last is the first to receive block allocation and scheduling**—textbook **LCFS (Last-Come, First-Served)**.

### Question 2: When blocks run short, why does left pointer `l` preempt from `running[0]` (the longest output)?
Note that the function header has a stale comment at L556 claiming it preempts the sequence with the shortest output, but the actual code (`a->get_output_len() > b->get_output_len()` at L625 paired with `_preempt(running[l])`) and the official comment on `insufficient_system_resource` at `sequence.hpp` L56 state the true behavior explicitly:

> "If the kvcache is about to overflow, **the task occupying the most kvcache will be forcibly terminated by us**."

Why prioritize preempting (or even terminating) the longest-output request in online serving? Do the math: if you preempt a brand-new request on the right that has generated only 3 tokens, evicting it frees only a handful of blocks—you might have to kill a dozen requests and still not scrape together enough blocks for the current step. By contrast, `running[0]` on the left with `output_len = 900` is sitting on a large pile of blocks; **preempting that single heavyweight frees enough blocks to keep dozens of shorter decode requests on the right alive**, maximizing overall request completion rate and system throughput!

### What happens after preemption? (`_preempt`, L766–L790)
- **Online serving mode (`stop_preempted_request = true`)** and **all requests with Structured Output (SO)**: The sequence is immediately marked `FINISHED_PREEMPTED`, and `_free_seq` releases all of its KV blocks and State slots without recomputation (since online CE instances have no prefill pipeline, and structured-output grammar automata do not support rollback).
- **Offline batch mode (`stop_preempted_request = false`)**: All KV blocks and State slots held by the sequence are freed, `reset_for_recompute()` resets `cache.num_tokens` to 0 (while preserving all tokens already generated inside `input.tokens`), and its status reverts to `WAITING` at the front of the `waiting` queue so the next prefill cycle recomputes the entire sequence from scratch.

---

## 3.5 A Hidden Gem in `_append_slot`: Returning Redundant Blocks After Draft Rejection

Inside `_append_slot` (`scheduler.hpp` L817–L828), which allocates blocks for each decode request during `_schedule_decode`, sits a subtle two-branch routine that is easy to overlook yet vital for speculative decoding:

```cpp
void _append_slot(Sequence& seq, std::vector<std::pair<int, int>>& clone_blocks) {
    const size_t num_blocks = seq.get_num_blocks();
    auto& block_table = seq.cache.block_table;
    if (block_table.size() < num_blocks) {
        // Branch 1: Not enough blocks—allocate and append new blocks
        const size_t num_new_blocks = num_blocks - block_table.size();
        for (size_t i = 0; i < num_new_blocks; ++i)
            block_table.push_back(block_manager.allocate());
    } else if (block_table.size() > num_blocks) {
        // Branch 2: Holding more blocks than currently needed! Return redundant tail blocks
        std::vector<int> redundant_blocks(block_table.begin() + num_blocks, block_table.end());
        block_table.resize(num_blocks);
        block_manager.free_blocks_nolock(redundant_blocks);
    }
}
```

Stop and think for a second: **As a sequence decodes forward, its text only grows longer—how could `block_table.size() > num_blocks` (holding more blocks than needed) ever happen?**

The culprit is **speculative decoding + variable-length verification**! Suppose `block_size = 64`:
1. When scheduling step $N$, a sequence has 62 tokens plus 5 candidate draft tokens, requiring $62 + 5 = 67$ positions—crossing the 64-token boundary. So `_append_slot` allocates a **2nd physical block** (`block_table.size() == 2`).
2. Step $N$ finishes verification, and all 5 draft tokens are rejected! The sequence grows by only 1 main sampled token, reaching length $63$.
3. On step $N+1$, suppose the variable-length verification scheduler decides to verify 0 draft tokens for this request (or just 1 draft token, keeping the total requirement $\le 64$). Now `seq.get_num_blocks()` evaluates to **1 block**!
4. At that moment, the `block_table.size() (2) > num_blocks (1)` branch fires and immediately returns the 2nd physical block—reserved for drafts on the previous step but ultimately unused—back to the free pool instead of hoarding it. We faithfully reproduce this exact reclamation branch in `minidi/scheduler.py` (L144–L146).

---

## 3.6 Chapter Summary

The table below distills the core design choices of DeepInfer's C++ scheduler:

| Scheduling Dimension | DeepInfer's Mechanism | Engineering Benefit & Trade-Off |
| --- | --- | --- |
| **Step Organization** | Strict separation of Prefill and Decode steps; multi-GPU synchronization via two-round `ALL(70)` / `ANY(100)` voting | Zero interference with Decode kernels / CUDA Graphs and strict alignment of MoE EP collectives; trade-off is up to a 70-step wait for new arrivals in `NORMAL` mode |
| **Prefill Admission** | Strict FCFS (`break` immediately if queue head cannot fit); upfront allocation of full-length KV blocks and State slot on chunk 1 | Completely eliminates long-prompt starvation and mid-chunk memory deadlocks; trade-off is potential head-of-line blocking |
| **Decode Admission & Preemption** | Sort by `output_len` descending: right pointer (shortest output, LCFS) allocates first; left pointer (longest output) is preempted first | Evicting a single long sequence frees a large batch of physical blocks to keep many short sequences alive, maximizing system throughput |
| **Dynamic Block Sizing** | `_append_slot` handles both `< num_blocks` (expansion) and `> num_blocks` (shrinking & returning blocks) | Promptly reclaims over-reserved tail blocks when speculative drafts are rejected, preventing memory leaks |

Now that we know how the head chef schedules orders, let us walk into the kitchen pantry in Chapter 4 to see how the **Paged KV Pool** and the **Fixed-Size State Pool** actually manage GPU memory.
