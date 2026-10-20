# Chapter 13  Building Zero-Overhead Asynchronous Scheduling (`ZOSEngine`): Blind Lookahead Scheduling and In-Flight Preemption

In Chapter 5, we walked through the four-stage asynchronous pipeline of DeepInfer's `Worker.execute_model_step` (`worker.py` L89–L189). At the time, you might have wondered: "Without real asynchronous GPU streams on a CPU, how can we reproduce and verify the logic of ZOS?"

The answer is remarkably simple: what makes asynchronous scheduling tricky is not hardware concurrency per se, but **information delay**—specifically, **at the exact moment the CPU executes `schedule(N+1)`, it is strictly forbidden from peeking at the sampling results of step $N$**!

As long as our pure-CPU `ZOSEngine` (`minidi/zos.py`) strictly enforces this information-isolation rule—locking step $N$'s sampling results inside a black box named `self.prev` and mandating that `_post_process(prev)` may only open that box *after* step $N+1$ has completed scheduling and launched its forward pass—every single asynchronous edge case that DeepInfer faces in production (lookahead block-table reservation, `remap` wiring, wasted tokens after EOS, and in-flight preemption) reproduces with 100% fidelity in our CPU code!

---

## 13.1 Replicating the Four-Stage `execute_model_step` Pipeline in 30 Lines of Code

Open `minidi/zos.py` (L18–L64). `ZOSEngine` inherits from our synchronous `Engine` from Chapter 10, and its `step()` method follows the exact four-stage order from Chapter 5:

```python
class ZOSEngine(Engine):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.prev = None          # Simulates step N state "still in flight on the GPU stream": (SchedulerOutputs, sampled_dict)
        self.wasted = 0           # Counts tokens computed in vain because step N already sampled EOS while the CPU did not yet know

    def step(self):
        # Stage 0: Lookahead-schedule step N+1 (step N's sampled tokens are still locked inside self.prev; never call process_outputs yet!)
        out = self.scheduler.schedule()
        prev_tokens = self.prev[1] if self.prev else {}

        # Stage 1: Assemble and run step N+1 forward pass (corresponds to _correct_seqlens_qk + execute_main_model)
        batch = []
        for seq, q in zip(out.scheduled, out.seqlens_q):
            if out.prompt_run:
                batch.append((seq.tokens[seq.num_cached : seq.num_cached + q], seq.num_cached, seq.block_table))
            elif seq.seq_id in prev_tokens and seq.inflight > 0:
                # Hit remap! Grab the input token directly from the previous step's "GPU memory" (prev_tokens) and shift the write offset by inflight!
                batch.append(([prev_tokens[seq.seq_id]], seq.num_cached + seq.inflight, seq.block_table))
            else:
                batch.append(([seq.tokens[-1]], seq.num_cached, seq.block_table))
        logits = self.model.paged_forward(self.kv, batch) if batch else None

        # Stage 2: Wait for step N sampling to finish and commit to the ledger (corresponds to _post_process_step(self.prev_step_state))
        prev, self.prev = self.prev, None
        self._post_process(prev)

        # Stage 3: Execute step N+1 sampling
        sampled, off = {}, 0
        for seq, q, (qt, start, _) in zip(out.scheduled, out.seqlens_q, batch):
            sampled[seq.seq_id] = sample(logits[off + q - 1], seq, start + q)
            off += q

        # Stage 4: Commit Prefill steps immediately; for Decode steps, save into self.prev to be settled in Stage 2 of the next step!
        if out.prompt_run:
            toks = [[sampled[s.seq_id]] if s.num_cached + q >= s.get_len() else []
                    for s, q in zip(out.scheduled, out.seqlens_q)]
            self.scheduler.process_outputs(out, toks)
        elif out.scheduled:
            for s in out.scheduled:
                s.inflight += 1
            self.prev = (out, sampled)
        return out
```

Look closely at the `elif seq.seq_id in prev_tokens and seq.inflight > 0:` branch in Stage 1:
- In a synchronous engine, the token sampled at step $N$ has already been appended to `seq.tokens[-1]`, and `seq.num_cached` has already been incremented by 1;
- In `ZOSEngine`, however, because Stage 2's `self._post_process(prev)` runs **after** Stage 1's `paged_forward`, at the moment Stage 1 executes the forward pass, `seq.tokens` does not yet contain step $N$'s sampled token, and `seq.num_cached` is still sitting at step $N-1$'s position!
- Therefore, any sequence that was in flight during the previous step (`seq.inflight > 0`) must fetch its input token directly from `prev_tokens[seq.seq_id]` (which corresponds to GPU-side `remap` in DeepInfer) and set its KV write start position to **`seq.num_cached + seq.inflight`**!

---

## 13.2 Real-World Pitfall: What Happens When an "In-Flight Request" Is Preempted at Step `N+1`?

When we wrote our first version of `ZOSEngine`, we naturally wrote the following lines inside `_post_process(prev)`:

```python
# Buggy first version:
for s in out.scheduled:
    s.inflight -= 1
```

Looks completely bulletproof, right? At the end of the previous step we did `s.inflight += 1`, and when settling the ledger on the next step we do `s.inflight -= 1`—plus one, minus one, back to zero.

Yet when we ran randomized small-block-pool tests in `test_zos_matches_sync`, the test suddenly crashed: **for one request, `inflight` became `-1`, causing its KV write position on the following step to move backward by 1 slot and corrupting the output!**

Tracing the logs and reconstructing the crime scene immediately revealed what that seemingly peculiar guard in DeepInfer's `scheduler.hpp` L917–L925 is actually protecting against:

```{.mermaid caption="Figure 13-1  Timing conflict and fix when an in-flight sequence is preempted during the next scheduling step"}
sequenceDiagram
  participant S as Scheduler / ZOSEngine
  participant A as Request A (Sequence)
  participant P as In-Flight Box (self.prev)
  S->>P: 1. Step N Decode finishes forward & sampling, stored in self.prev
  S->>A: 2. Mark in-flight: A.inflight = 1 (status remains RUNNING)
  Note over S,A: 3. Enter Step N+1: run schedule(N+1) first, free blocks run short!
  S->>A: 4. Left pointer preempts Req A: calls reset_for_recompute()<br/>resets A.inflight to 0, sets status back to WAITING, frees all physical blocks!
  P-->>S: 5. Stage 2 opens self.prev to settle Step N outputs: _post_process(N)
  Note over S,A: Fatal Traps:<br/>Trap 1: Unconditionally running A.inflight -= 1 turns the counter from 0 into -1!<br/>Trap 2: Appending Step N's token into A treats WAITING Req A as an active decode sequence!
  S->>A: Proper Guardrails:<br/>1. Decrement by 1 only when A.inflight > 0,<br/>2. If process_outputs sees a decode step with A.status == WAITING, discard Step N's output!
```

Look at steps 4 and 5 in Figure 13-1:
1. At the end of step $N$, Request A is in flight (`A.inflight = 1`).
2. Step $N+1$ begins by immediately running `schedule(N+1)`. Unluckily, physical blocks run short on this step, and the two-pointer preemption algorithm picks Request A for preemption! `_preempt(A)` calls `A.reset_for_recompute()`, which immediately resets `A.inflight` to `0`, frees all of A's physical blocks, sets its status back to `Status.WAITING`, and pushes it back into the waiting queue.
3. Next, in Stage 2 of step $N+1$, `_post_process(prev)` arrives to settle the ledger using step $N$'s old schedule list `out.scheduled` (which still contains Request A!):
   - If we blindly decrement by 1 without checking `if s.inflight > 0:`, `A.inflight` drops from `0` to `-1`!
   - If `process_outputs` does not check `if not out.prompt_run and seq.status == Status.WAITING: continue`, it will force-feed the now-invalid token from step $N$ into Request A (which is sitting in `waiting` awaiting recomputation) and set `A.num_cached` to `len - 1` (even though A's block table is now completely empty!), crashing on the very next step!

After adding those two guardrails (`minidi/zos.py` L30–L31 and `minidi/scheduler.py` L177–L178), look back at `csrc/engine/scheduler.hpp` L917–L925 from Chapter 5:

```cpp
if (!prompt_run && seq.status == SequenceStatus::WAITING) {
    // Note: During this in-flight decode step, the sequence was preempted back to waiting; discard its output directly
    continue;
}
```

That "Aha! The DeepSeek engineers wrote this exact `if` because they stepped into the exact same trap!" moment of resonance is the greatest joy of building an inference engine with your own hands.

---

## 13.3 Final Drain (`flush_inflight_reqs`) and Full Verification

When all requests have finished and `scheduler.has_unfinished()` becomes `False` to exit the main loop, don't forget that the `self.prev` black box is still holding the uncommitted sampling results from the **very last step**! Therefore, `ZOSEngine.generate` (`minidi/zos.py` L74) must make one final call to `self._post_process(self.prev)` after exiting the loop—matching DeepInfer's call to `flush_inflight_reqs()` at the end of `LLM._run_engine` (`llm.py` L184).

In `tests/test_engine.py::test_zos_matches_sync`, we run comparison tests across 20 random seeds with tight physical block pools (14–39 blocks, triggering frequent preemptions) and random EOS tokens:
1. Assert that all sequences generated by `ZOSEngine` and the synchronous `Engine` are 100% identical token for token;
2. Assert that upon completion, `z.bm.num_free() == num_blocks - 1` (aside from reserved block 0, not a single physical block is leaked due to asynchronous preemption).
