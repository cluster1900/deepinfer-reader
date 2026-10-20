# Chapter 15  Building the SWA Ring Buffer, EPLB, and Multi-GPU Step-Type Voting

In Chapter 14, we saw that when speculative draft tokens are rejected, paged KV only needs to rewind its `num_cached` pointer without erasing stale data.

However, inside the **fixed-length ring buffers** used by **Sliding Window Attention (SWA)** and **gated KV compression states**, the situation is completely different! Because a ring buffer maps logical position $p$ to physical slot $p \bmod \text{ring}$ via modulo arithmetic, if the ring buffer length $\text{ring}$ were set equal to the window size $W$, writing candidate draft tokens during the current step would **directly overwrite valid historical window entries from a few steps earlier**! Once those drafts are rejected, the overwritten historical data would be lost forever!

In this chapter, we open `minidi/swa.py`, `minidi/eplb.py`, and `minidi/vote.py` to build these three core algorithmic modules from scratch and empirically verify the ring-buffer backtracking slack formula in code.

---

## 15.1 Derivation and Empirical Verification of SWA Ring Buffer Backtracking Slack (`minidi/swa.py`)

Let's first derive, using a strict inequality, the minimum ring buffer size `ring` required for single-step forward execution and speculative rollback:

- Suppose the sliding window size is $W$ (so attention at any query position $q$ needs to read the $W$ KV positions $[q - W + 1, \dots, q]$).
- Before a Decode step begins, the sequence has already confirmed and committed $L$ tokens (positions $0 \dots L-1$).
- During the current speculative verification step, we write $k = 1 + \text{nextn}$ new positions into the ring buffer at once, starting at position $L$: $L, L+1, \dots, L+k-1$.
- Note: **High-performance industrial attention kernels (such as FlashMLA) first call `store_kv_cache` to write the KV of all $k$ new positions into the ring buffer before launching the attention kernel to compute queries for those $k$ positions in parallel!**
- This means that when the very first query of the current step (at position $L$) computes attention, the oldest historical position it requires—**`L - W + 1`**—must never collide modulo `ring` with, or be overwritten by, the last position written in the current step—**`L + k - 1`**!
- At the same time, if all candidate drafts are rejected after verification (keeping only position $L$), when the next step resumes from position $L+1$, its window start **`L - W + 2`** is also preserved within that same historical range.
- For the last written position $L + k - 1$ wrapping around the ring (i.e., $L + k - 1 - \text{ring}$) to fall strictly to the left of the oldest required position $L - W + 1$, we must have:
  $$L + k - 1 - \text{ring} < L - W + 1 \iff \text{ring} > W + k - 2 \iff \text{ring} \ge W + k - 1 = W + \text{nextn}$$

In other words, **the minimum safe backtracking slack for single-step forward execution and speculative rollback is $\text{extra} = \text{ring} - W = k - 1 = \text{nextn}$**!

In `minidi/swa.py` (44 lines in total), we implemented the `SWARing` class, which records in each slot the exact logical position currently stored there (`pos_of_slot[slot]`). If reading a window ever discovers that the logical position in a required slot has been overwritten by newer data, it immediately raises a `RuntimeError`. We run a side-by-side empirical test with $W = 4, k = 3$ (1 main token + 2 drafts, so $\text{nextn} = 2$) and committed length $L = 10$; the result is shown in Figure 15-1:

![Figure 15-1  Empirical comparison of SWA ring buffer backtracking slack (W=4, k=3, L=10): Top panel with slack k-1=2 (ring=6) preserves the full window; bottom panel with slack k-2=1 (ring=5) overwrites position 7 when writing position 12 in the current step](images/plots_en/ch15_swa_ring.png)

Look at Figure 15-1:
- **Top panel (`ring = W + (k - 1) = 4 + 2 = 6`)**: The current step writes positions `10, 11, 12` (orange, landing in slots `4, 5, 0`). Position `12` overwrites slot `0`, which held the already-expired position `6`; meanwhile, the window `7, 8, 9, 10` required by the query at position `10` (blue slots `1, 2, 3` and orange slot `4`) remains completely intact! Even if drafts `11, 12` are subsequently rejected and the next step rewrites starting from position `11` (slot `5`), the historical window is unharmed.
- **Bottom panel (`ring = W + (k - 2) = 4 + 1 = 5`, just 1 slot smaller)**: When the current step writes positions `10, 11, 12`, position `12` (`12 % 5 == 2`) crashes directly into **slot 2** (red), overwriting the valid historical **position 7** that lived in slot 2! When position `10` tries to read its window `[7, 8, 9, 10]`, it immediately reads corrupted data and raises an error!

### Complete Corroboration with DeepInfer's Source Code
This derivation matches the real DeepInfer source code we inspected in Section 4.3 down to the exact constant:
1. **Two places in DeepInfer that use the exact lower bound $W + \text{nextn}$**:
   - DeepInfer's compression boundary state ring (`get_inference_compress_block_size`, `cache_engine.py` L131–L133 and `compress.py` L98) uses the exact formula `compress_ratio + DEEPINFER_MAX_BACKTRACKING_TOKENS - 1 = ratio + nextn`;
   - The DFlash draft model's own SWA layer (`get_kv_index.py` L183) likewise uses `window_size + dflash_block_size = 128 + 5 = 133` effective window slots.
2. **Why the main model's SWA Ring uses $W + \text{nextn} + 1 = 128 + 6 = 134$**:
   As analyzed in Section 4.3, beyond satisfying the single-step forward and rollback lower bound (133 slots), the main model's SWA ring also supports state snapshot persistence across 128-token boundaries (`do_copy_state` & `select_state_slice`), loading strictly preceding 128 historical positions during Prefill (`make_swa_block_load_plan`), and YoCo prefill tail retention (`keep_tokens = 134`), so it directly adds the full single-step maximum length `DEEPINFER_MAX_BACKTRACKING_TOKENS = nextn + 1 = 6`.

---

## 15.2 Building the EPLB Expert Replication and Packing Algorithm in 60 Lines (`minidi/eplb.py`)

Next, let's open `minidi/eplb.py`, which distills `deepinfer/model_executor/eplb/policy.py` from Section 7.4 into two clean NumPy functions:

```{.mermaid caption="Figure 15-2  Two-stage expert load-balancing algorithm replicated in minidi/eplb.py"}
flowchart LR
  subgraph Stage1["Stage 1: replicate(tokens, num_physical)<br/>(corresponds to rebalance_experts_global)"]
    direction TB
    R1["Input historical step load matrix tokens [T, E]<br/>Start with 1 replica per logical expert (count = 1)"] --> R2["Loop M = num_physical - E rounds:<br/>enumerate which logical expert j, if given +1 replica,<br/>minimizes Σ_t max_e (tokens[t,e] / count[e])"]
    R2 --> R3["Produce physical-to-logical map p2l [E + M]<br/>and split each logical expert's load evenly across its replicas"]
  end

  subgraph Stage2["Stage 2: place(tokens, p2l, num_ranks)<br/>(corresponds to _batch_greedy_indices)"]
    direction TB
    P1["Sort all E + M physical expert replicas<br/>in descending order of total window load"] --> P2["Greedily place each into a non-full rank (cap (E+M)//R)<br/>that minimizes the sum of historical step peaks Σ_t max_r load[t,r]"]
    P2 --> P3["Output per-EP-rank physical replica lists ranks"]
  end

  Stage1 --> Stage2
```

Look at the core implementation in `minidi/eplb.py` L16–L52—even the tiny tie-breaking bias `+ np.arange(E) * 1e-4` matches DeepInfer's original code:

```python
def replicate(tokens: np.ndarray, num_physical: int) -> np.ndarray:
    t, e = tokens.shape
    biased = tokens.astype(float) + np.arange(e) * 1e-4
    cnt = np.ones(e, dtype=int)
    for _ in range(num_physical - e):
        # Evaluate the sum of per-step peak loads after giving +1 replica to each logical expert k, and pick the best drop!
        best_k, best_score = 0, np.inf
        for k in range(e):
            c = cnt.copy(); c[k] += 1
            score = (biased / c).max(axis=1).sum()
            if score < best_score:
                best_score, best_k = score, k
        cnt[best_k] += 1
    return np.repeat(np.arange(e), cnt)                   # physical_to_logical_map
```

In `tests/test_algorithms.py::test_eplb_improves_balance`, we test 16 Zipf-distributed logical experts across 4 GPUs with 4 redundant replicas, verifying that the cross-GPU load balance metric `moe_lb` achieved by EPLB significantly outperforms naive uniform placement (which is also the data source for Figure 7-3 in Chapter 7).

---

## 15.3 Building Two-Round Multi-DP Step-Type Voting in 25 Lines (`minidi/vote.py`)

Finally, let's look at `minidi/vote.py` (25 lines in total), which faithfully replicates the cross-DP `ALL(70)` / `ANY(100)` two-round voting protocol from `scheduler.hpp` L296–L344 (Figure 3-1 in Section 3.2):

```python
def decide(decode_step: int, wants: list[bool], can_schedule: list[bool],
           all_interval: int = 70, any_interval: int = 100):
    n = len(wants)
    is_any = decode_step != 0 and decode_step % any_interval == 0
    is_all = decode_step % all_interval == 0
    if not (is_any or is_all):
        return "decode", ["decode"] * n
    # Round 1 vote: ANY point requires at least one rank wanting prefill; ALL point requires every rank wanting prefill
    do_prefill = any(wants) if is_any else all(wants)
    if not do_prefill:
        return "decode", ["decode"] * n
    # Round 2 vote: after attempting _schedule_prefill, check whether at least one rank actually scheduled a request
    got = [w and c for w, c in zip(wants, can_schedule)]
    if not any(got):
        return "decode", ["decode"] * n                   # Everyone came back empty-handed; fall back to decode to prevent an infinite loop!
    return "prefill", ["prefill" if g else "empty-prefill" for g in got]
```

`tests/test_algorithms.py::test_vote_points` systematically verifies all five key scenarios: normal steps going straight to Decode, single-rank veto at an `ALL(70)` point, single-rank approval at an `ANY(100)` point with idle ranks running `empty-prefill` dummy steps, and safe fallback to Decode when Round 2 voting finds that no rank could actually allocate blocks for prefill.
