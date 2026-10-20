# Chapter 12  Building the Scheduler and Preemption (`Scheduler`): Two Pointers, Block Shrinking, and Head-of-Line Deadlock

In this chapter, we open `minidi/scheduler.py` (206 lines in total) and implement every piece of the core DeepInfer C++ scheduler logic we studied in Chapter 3 in clean Python—**strict separation between Prefill and Decode steps, FCFS chunked Prefill, LCFS two-pointer Decode preemption, tail-block shrinking after draft rejection, and two-phase ledger commit**—while dissecting the second real-world deadlock we encountered when testing "preemption with recomputation."

---

## 12.1 Step-Type Switching: Why Does Each Step Do Only One Thing?

In `Scheduler.schedule()` (`minidi/scheduler.py` L71–L84), we replicate the single-GPU step-type switching rule of DeepInfer's `NORMAL` engine:

```python
def schedule(self) -> SchedulerOutputs:
    out = None
    # Only attempt Prefill when decode_step is a multiple of the attempt interval and KV usage <= 0.9
    if self.decode_step % self.prefill_attempt_interval == 0:
        want = self.bm.usage() <= PREFILL_WATERMARK and bool(self.waiting)
        if want:
            out = self._schedule_prefill()
    if out is None:
        out = self._schedule_decode()
    if out.prompt_run:
        self.decode_step = 0                     # Once a Prefill step runs, reset the counter to 0 so the next step can continue Prefill
    else:
        self.running = list(out.scheduled)
        self.decode_step += 1
    return out
```

Notice lines 79–80: as soon as a Prefill step is scheduled (`out.prompt_run == True`), `self.decode_step` is reset to `0`. This means that as long as there are still requests waiting for prefill in the `waiting` queue and KV block usage has not yet hit the 90% watermark, `0 % prefill_attempt_interval == 0` still holds on the next step—so the scheduler will continuously drain the backlog of Prefill work before switching back to Decode.

---

## 12.2 `_schedule_prefill`: Strict First-Come-First-Served and Chunking Rules

In `_schedule_prefill` (`minidi/scheduler.py` L86–L125), we inspect requests one by one starting from the head of `self.waiting`. To maintain 100% behavioral parity with `scheduler.hpp` (L351–L554), five ironclad rules govern this loop:

1. **Remove aborted or over-length requests immediately without blocking the queue**: If the head-of-line request has already finished, or its total length exceeds `prompt_limit`, or `bm.can_allocate` returns `AllocStatus.NEVER` (it will never fit), mark it as `FINISHED_IGNORED`, pop it into `out.ignored`, and move on to the next request.
2. **`break` immediately on temporary block shortage (`LATER`)**: If there are currently not enough free blocks for the head-of-line request (returning `AllocStatus.LATER`), terminate the loop on the spot—**never skip over it to cherry-pick smaller requests behind it**, strictly enforcing First-Come-First-Served (FCFS).
3. **Allocate all blocks for the entire prompt on the first chunk**: When `not seq.block_table`, immediately call `seq.block_table = self.bm.allocate(seq.get_num_blocks())` to grab all physical blocks for the entire prompt at once.
4. **Only transition to `RUNNING` and consume a concurrency slot on the final chunk**: If the request is split into chunks due to the `max_num_batched_tokens` budget (`is_chunked = q < todo`), it must be returned to the front of `waiting` after this step completes; only when `not is_chunked` (this step finishes the entire prompt) do we increment `num_curr += 1` and set its status to `Status.RUNNING`.
5. **The `NORMAL` engine stops packing once a request is chunked**: As soon as the current request undergoes chunk truncation (`if is_chunked: break`), no subsequent requests are packed into the current step's batch.

---

## 12.3 `_schedule_decode`: Two-Pointer Preemption and Redundant Tail-Block Reclamation

Upon entering `_schedule_decode` (`minidi/scheduler.py` L127–L153), we first free the physical blocks of requests that finished in the previous step, and then implement the **LCFS scheduling + longest-output-first preemption + dynamic tail-block resizing** from Chapter 3 in just 18 lines of code:

```python
running = sorted(self.running, key=lambda s: s.output_len(), reverse=True)
l, r = 0, len(running)
while l < r and len(out.scheduled) < self.max_num_seqs:
    seq = running[r - 1]                              # Right pointer: schedule starting from the shortest output (latest to enter decode)
    need = seq.get_num_blocks() - len(seq.block_table)
    while need > self.bm.num_free() and r - l > 1:
        self._preempt(running[l], out)                # Left pointer: when blocks run short, preempt the request with the longest output first!
        l += 1
    if need <= self.bm.num_free():
        if need > 0:
            seq.block_table += self.bm.allocate(need) # Branch 1: expand and append new blocks
        elif need < 0:                                # Branch 2: reclaim redundant tail blocks after draft rejection (matching L821-L825)
            self.bm.free_blocks(seq.block_table[need:])
            del seq.block_table[need:]
        out.scheduled.append(seq)
    else:
        self._preempt(seq, out)
    r -= 1
```

Take a look at two highlights in this code:
- **Right pointer protects the newest, left pointer evicts the longest**: `tests/test_scheduler.py::test_lcfs_preempts_longest_output_first` constructs 3 requests with output lengths `4, 0, 8` that exactly fill the physical block pool. When all three need a new block on the next step, the test verifies that the very first sequence preempted by the left pointer is `seq2`, which has the longest output (`out=8`).
- **`elif need < 0:` precisely reclaims rejected draft blocks**: As analyzed in Section 3.5, when the previous step reserved a new block across a block boundary for speculative draft tokens, and draft rejection during verification causes the required block count for the current step to drop back down, the `need < 0` branch immediately returns the excess tail blocks to the `BlockManager` (specifically verified by `test_append_slot_frees_redundant_blocks_after_draft_rejection`).

---

## 12.4 Real-World Pitfall: When "Chunked Prefill" Meets "Preemption with Recomputation," Why Do They Deadlock at the Head of the Queue?

After implementing `_preempt`, we ran randomized stress tests in offline recomputation mode (`stop_preempted_request = False`) and hit the second deadlock during the development of `minidi`. This is a classic scheduling deadlock that every inference engine author should watch out for!

### How Did the Deadlock Happen?
Initially, we copied line 784 of `scheduler.hpp` verbatim, pushing any preempted-for-recomputation request straight back to the very front of the `waiting` queue inside `_preempt` (`self.waiting.appendleft(seq)`). In one of our stress tests, this triggered a four-step chain reaction:

```{.mermaid caption="Figure 12-1  Head-of-line deadlock between a partially chunked request and a preempted recompute request at the front of the waiting queue"}
flowchart TB
  subgraph S12["Phase 1: Partial Chunk Holds Blocks + Decode Preemption Cuts in Line"]
    direction LR
    A1["1. Req A (needs 10 blocks) finishes chunk 1;<br/>holds 10 blocks, returns to waiting[0] unfinished"] --> A2["2. Next Decode step runs out of blocks, preempts Req B!<br/>B frees its blocks and is pushed to the very front of waiting"] --> A3["3. Queue is now: [B (0 blocks held, needs 6), A (holds 10)]<br/>while the free pool only has 4 blocks left"]
  end

  subgraph S34["Phase 2: Head-of-Line Deadlock vs. minidi Fix"]
    direction LR
    B1["Deadlock: _schedule_prefill checks head B, gets LATER, and breaks;<br/>A sits behind B holding 10 blocks, never scheduled to finish and free them!"] --> B2["Fix (scheduler.py L165): insert recompute requests behind block-holding partial chunks,<br/>letting holder A finish and free its blocks first to eliminate the deadlock!"]
  end

  S12 --> S34
```

Look closely at Figure 12-1:
- **Request B** sits at `waiting[0]`. It holds no blocks and asks the block manager for 6 blocks, but the free pool only has 4 free blocks left. So `can_allocate` returns `LATER`, and strict FCFS forces the loop to `break` immediately.
- **Request A** sits at `waiting[1]`. It is already holding 10 physical blocks in its hand! If only we let it finish its remaining chunk and proceed through Decode to completion, those 10 blocks would be freed. Yet because it is stuck behind blockless Request B, it can never be scheduled!

### Why Doesn't DeepInfer Hit This in Production, and How Does `minidi` Fix It?
In DeepInfer's production environment, this deadlock naturally cannot occur for two reasons:
1. Production deployments use a **PD-disaggregated architecture** (`PROMPT` engines only run prefill, while `COMPLETION` engines only run decode), so the two stages never compete for blocks inside the same instance;
2. Even in a single-instance deployment, `EngineArgs.stop_preempted_request` defaults to **`True`** (preempted requests are aborted immediately rather than pushed back into `waiting` for recomputation).

In `minidi`, which supports offline recomputation (`stop_preempted_request = False`), our fix is both simple and complete (`minidi/scheduler.py` L165–L171): when inserting a preempted request back into `waiting`, **place it behind all partially chunked requests that are already holding physical blocks**:

```python
# minidi/scheduler.py L165-L171 (_preempt recomputation branch)
i = 0
while i < len(self.waiting) and self.waiting[i].block_table:
    i += 1
self.waiting.insert(i, seq)   # Insert behind partial-chunk requests that already hold blocks, letting block holders finish and free first!
```

As long as partially chunked Request A—which already holds physical blocks—stays at the head of the queue to finish first, the deadlock vanishes completely.

---

## 12.5 Live Execution Trace: Seeing the 0.9 Watermark and Preemption Recomputation in Action

Now we run `trace()` in `scripts/figures/make_plots.py`, feeding 10 requests of varying lengths into a `minidi.Engine` with only 19 usable physical blocks, and plot the live trajectory of batch size, step type (orange = Prefill step, blue = Decode step), and KV block usage across every step:

![Figure 12-2  Full execution trace of 10 requests in minidi: Prefill pauses when KV usage hits the 0.9 watermark, and resumes after a full pool triggers 4 preemptions until all requests complete](images/plots_en/ch12_trace.png)

Looking at Figure 12-2, every rule we covered in Chapter 3 and this chapter comes to life:
1. **Opening phase (Steps 0–6)**: While KV usage is below the 0.9 watermark, the engine runs consecutive orange Prefill steps, rapidly admitting the first wave of requests into `running` as block usage climbs past 0.9;
2. **Middle phase (Steps 7–25)**: Because usage is above or near 0.9, the engine focuses on blue Decode steps. When multiple requests grow simultaneously and exhaust all 19 blocks, **4 preemptions** are triggered (visible as sharp drops in the black usage curve). As soon as usage drops below 0.9, orange Prefill steps immediately interleave to finish both newly waiting requests and preempted requests!
3. Across all 50+ steps, orange bars (Prefill) and blue bars (Decode) remain strictly separated, and the final outputs of all 10 requests match `reference_generate` 100% token for token.
