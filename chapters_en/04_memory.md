# Chapter 4  GPU Memory and KV Cache: Two Separate Ledgers for Paged Blocks and Fixed-Size State Slots

In a standard Transformer inference engine (such as original vLLM), every layer's KV Cache looks identical: each time a sequence grows by 1 token, every layer appends 1 K and V entry into a paged block.

In **DeepSeek-V4.1-Flash**, however, that one-size-fits-all paging scheme breaks down. A single layer of this model can simultaneously involve four fundamentally different cache objects:

1. **Compressed Attention KV (`ca_cache`)**: Gated compression condenses every `ratio` consecutive tokens into a single latent vector, growing linearly with sequence length;
2. **Sparse Index KV (`ica_cache`)**: Ultra-low-bit index keys scored by the sparse attention selector (`DeepSelect`), also growing linearly with sequence length;
3. **Sliding Window Attention KV (`swa_cache`)**: Both the main model's SWA layers and the DFlash draft model attend only to the most recent $W = 128$ tokens, regardless of whether the total sequence length is 500 or 50,000;
4. **Compression Boundary Accumulators (`ca_state` / `ica_state`)**: Fixed-size buffers storing partial token states at compression-window boundaries before a full compression stride is reached.

If you forced items 3 and 4—which have **fixed size and circular overwrite semantics**—into a full-length paged block table, you would waste massive amounts of GPU memory and twist block reclamation into knots. DeepInfer's core design therefore **maintains two independent memory ledgers**:

- **Ledger 1: `block_manager` (Paged KV Pool)**—grows dynamically in fixed token blocks to manage `ca_cache` and `ica_cache`;
- **Ledger 2: `state_manager` (Fixed-Size State Pool)**—assigns each active request 1 dedicated fixed-size slot (`state_id`), managing `swa_cache` and compression boundary states as ring buffers.

---

## 4.1 Startup Profiling: How Many Paged Blocks Can the GPU Actually Carve Out?

Before allocating either ledger, the engine probes available GPU memory inside `Worker.profile_num_available_blocks` (`deepinfer/worker/base.py` L212–L268):

```{.mermaid caption="Figure 4-1  Profiling and Computing Available Paged KV Blocks (deepinfer/worker/base.py L212–L268)"}
flowchart TB
  subgraph Step1["Stage 1: Measure Peak Activation Memory on a Worst-Case Batch Forward Pass"]
    direction LR
    P1["Clear cache & reset stats<br/>empty_cache + reset_peak"] --> P2["Run profile_run()<br/>Execute dummy forward at max batch"]
    P2 --> P3["Record forward peak memory:<br/>PyTorch peak + non-PyTorch GPU memory"]
  end

  subgraph Step2["Stage 2: Deduct Fixed-Size State Pool First, Then Convert Remainder to Paged Blocks"]
    direction LR
    Q1["Usable Budget = Total VRAM × gpu_memory_utilization − peak"] --> Q2["Deduct Fixed State Pool first:<br/>subtract state_cache bytes × 2 (main + copy buffer)"]
    Q2 --> Q3["Local blocks = remaining bytes // block_bytes;<br/>all_reduce(SUM) average across ranks & align to even"]
  end

  Step1 --> Step2
```

Two engineering details in Figure 4-1 are worth noting:

1. **The fixed-size State Pool is deducted first, and provisioned at `× 2` capacity**: Because the State Pool size depends solely on the maximum concurrency `num_state_seqs`, its footprint is subtracted upfront—multiplied by 2 because alongside the primary `state_cache`, the engine allocates an identically sized `state_cache_to_copy` staging buffer for asynchronous snapshot persistence (`base.py` L244–L249). All remaining memory budget is then sliced into paged KV blocks.
2. **Multi-GPU averaging and even-number alignment**: Lines 265–266 compute a uniform block count across all ranks via `all_reduce(SUM)` divided by `world_size`, followed by `// 2 * 2` to round down to an even integer (note: while a comment in `llm_engine.py` L426 says minimum across ranks, the actual code computes the mean aligned to an even integer).

---

## 4.2 Ledger 1: How Large Is a Paged KV Block, and How Many Bytes Does Each Token Take?

In DeepInfer, the number of raw tokens represented by a single paged block is locked by model metadata: `ModelMetadata.resolved_tokens_per_block` always returns **`128`** (`model_metadata_utils.py` L134–L135).

In other words, **logically, one physical block corresponds to 128 raw tokens**. Because compressed attention layers emit only one compressed KV entry per `ratio` tokens, however, the second dimension of the physical tensor is `128 // ratio`. Cross-referencing `deepinfer/worker/cache_engine.py` (L206–L219) and `layers/tilelang_ops/cache_common.py` (L109–L127) gives the exact tensor shapes and byte accounting across both pools:

| Ledger | Cache Name | Physical Tensor Shape (`cache_engine.py`) | Data Format & Bytes per Entry |
| --- | --- | --- | --- |
| **Paged KV Pool**<br/>(`block_manager`) | `ca_cache`<br/>(Compressed Attention KV) | `[num_blocks, 128 // ratio, 1, bytes]` | **MXFP8**: 512 `e4m3` + 16 `ue8m0` scales = **528 B**<br/>**FP4 (per-16)**: 512 `e2m1` (256 B) + 32 `e4m3` scales = **288 B** |
| **Paged KV Pool**<br/>(`block_manager`) | `ica_cache`<br/>(Sparse Index KV) | `[num_blocks, 128 // ratio, 1, idx_bytes]` | **MXFP4**: `index_head_dim / 2` data bytes + `index_head_dim / 32` `ue8m0` scale bytes |
| **Fixed State Pool**<br/>(`state_manager`) | `swa_cache`<br/>(Sliding Window KV) | `[num_state_seqs, swa_block_size, 1, 528]` | **MXFP8**: fixed **528 B** per window slot, ring buffer length `swa_block_size = 134` |
| **Fixed State Pool**<br/>(`state_manager`) | `ca_state` / `ica_state`<br/>(Compression Boundary State) | `[num_state_seqs, ring_size, 2, dim]` | **FP32**: stores residual vectors at compression window boundaries, ring length `ring_size = ratio + 5` |

Notice the striking **`1`** in the third dimension of the tensor shapes: `_kvcache_layout` (L624) explicitly asserts `num_key_value_heads == 1`—the hallmark of DeepSeek's **MLA (Multi-head Latent Attention)** architecture, where all attention heads share a single compressed latent KV vector, shrinking KV memory footprint by more than an order of magnitude compared to standard multi-head attention!

---

## 4.3 Ledger 2: Why Is the Compression Ring `W + 5`, While the SWA Ring Is `128 + 6 = 134`?

Now let us examine the fixed-size state pool (`state_manager`). Every active request entering the system is assigned a dedicated `state_id` that indexes into both its SWA ring buffer and its compression-state ring buffer:

```{.mermaid caption="Figure 4-2  A Single Active Sequence Simultaneously Holds a Paged Block Table (block_table) and a Fixed State Slot (state_id)"}
flowchart TB
  Seq["Active Sequence<br/>cache.block_table = [7, 3, 12] · cache.state_id = 5"]

  subgraph Paged["Ledger 1: block_manager (Dynamically Appends Blocks with Sequence Length; Block 0 Reserved)"]
    direction LR
    B7["Physical Block 7<br/>tokens 0..127"] --> B3["Physical Block 3<br/>tokens 128..255"] --> B12["Physical Block 12<br/>tokens 256.."]
  end

  subgraph State["Ledger 2: state_manager (1 Fixed Slot per Sequence, Circularly Overwritten via Modulo Indexing)"]
    direction LR
    S5A["Slot 5 · Stored Region<br/>First 20 Producer SWA Layers (134 slots × 528 B)"] --- S5B["Slot 5 · Dropped Region<br/>YoCo Consumer SWA + Compression Ring (ratio + 5 slots)"]
  end

  Seq --> B7
  Seq --> S5A
```

Here lies a deeply illuminating question that puzzles almost everyone reading the source code for the first time:

> Given a sliding window size of $W = 128$ and $\text{nextn} = 5$ speculative draft tokens (meaning a single decode step feeds up to $k = 1 + 5 = 6$ tokens into the model), why does the **Compression State Ring** have length $\text{ratio} + 5$, whereas the **SWA Ring** has length $128 + 6 = 134$? Where does that extra **`1`** come from?

Let us first lay out the two formulas from the source code (`deepinfer/environ.py` L139 and `deepinfer/worker/cache_engine.py` L88–L133):

1. The environment variable `DEEPINFER_MAX_BACKTRACKING_TOKENS` defaults to `nextn + 1 = 6` (the maximum number of tokens $k = 1 + \text{nextn}$ fed into the model in a single decode step).
2. **Compression State Ring size** (`get_inference_compress_block_size`, `cache_engine.py` L131–L133 and `compress.py` L98):
   $$\text{compress\_ring\_size} = \text{compress\_ratio} + \text{DEEPINFER\_MAX\_BACKTRACKING\_TOKENS} - 1 = \text{ratio} + 5$$
3. **SWA Ring size** (`get_inference_swa_block_size`, `cache_engine.py` L88–L93):
   $$\text{swa\_block\_size} = \text{window\_size} + \text{DEEPINFER\_MAX\_BACKTRACKING\_TOKENS} = 128 + 6 = 134$$

Why does one formula subtract 1 while the other does not? Walking through the indexing math with small numbers makes the answer crystal clear:

### Step 1: The theoretical minimum lower bound for single-step forward + speculative rollback is `W + k - 1 = W + nextn`
Suppose the window size is $W$, and starting at position $L$, the current decode step writes $k = 1 + \text{nextn} = 6$ new positions into the ring buffer in one shot: $L, L+1, \dots, L+k-1$.
- Within this step, the very first query token (at position $L$) must attend back to the most recent $W - 1$ historical positions before it: $[L - (W - 1), \dots, L - 1]$.
- Meanwhile, if all 5 draft tokens are rejected during verification, the next step will roll back and resume from position $L+1$—meaning those $W - 1$ historical positions plus position $L$ (a total of $W$ valid positions) must never be overwritten by the 5 rejected draft tokens!
- Therefore, the total number of positions that must coexist peacefully inside the ring buffer at the same time is:
  $$\underbrace{(W - 1)\text{ prior history positions}}_{L-(W-1)\,\dots\,L-1} + \underbrace{k\text{ newly written positions in this step}}_{L\,\dots\,L+k-1} = W + k - 1 = W + \text{nextn}$$
- This exact bound is why the **Compression State Ring** uses `compress_ratio + 5` (explicitly subtracting 1) and why the **DFlash draft-layer SWA** uses `128 + 5 = 133` (`get_kv_index.py` L183)! In Chapter 15, we will use `minidi` to plot the exact slot-overwrite diagram showing how having even 1 fewer slot corrupts historical state upon draft rejection.

### Step 2: Why doesn't the main-model SWA Ring subtract 1, insisting on `128 + 6 = 134`?
Because in addition to single-step forward execution and speculative rollback, the main model's SWA ring buffer must also support **asynchronous snapshot persistence across 128-token boundaries (KVCache Store)** and **prefill history window loading** (see Official Q&A L161–L191 and source code):

1. **State snapshotting when Decode crosses a 128-token boundary (`llm_engine.py` L930 `do_copy_state` and `state_cache.py` L44 `select_state_slice`)**:
   In production, the engine copies out a request's SWA state for persistence only when its sequence length crosses an **integer multiple of 128** (`KVSTORE_GRAIN = 128`). Consider the worst-case phase alignment: prior to the current step, a sequence has cached $127$ tokens (positions $0 \dots 126$), and the current decode step writes $k = 6$ tokens (positions $127 \dots 132$), crossing the $128$ boundary!
   At the end of the step, `do_copy_state` triggers and must slice out the **entire 128-slot historical window** up to position $127$ (positions $0 \dots 127$) for persistence—even though positions $128 \dots 132$ (5 overshoot positions) have already been written into the exact same ring buffer during the forward pass! Provisioning $128 + 6 = 134$ slots guarantees that regardless of where the 6-token decode step lands relative to the 128-token boundary, the full 128-token pre-boundary window remains completely intact inside the ring waiting to be copied!
2. **Prefill historical window loading (`prepare_tensor_input_v4.cpp` L378–L383 `make_swa_block_load_plan`)**:
   When loading historical SWA states into the ring before a new chunk runs, the load plan reads a full 128 strictly prior historical positions `[history_end - 128, history_end)`, which together with up to 6 backtracking margin slots occupies all $128 + 6 = 134$ slots.
3. **YoCo prefill truncation and prefix-cache hit ceiling (`prepare_tensor_input_v4.cpp` L355 and Official Q&A L189–L191)**:
   To ensure YoCo's second-half layers and SWA Bounded Replay always have a complete window available, the uncached tail length retained during prefix matching is fixed at `window_size + max_backtracking_tokens = 128 + 6 = 134`.

Furthermore, inside `StateBlock` (`cache_engine.py` L221–L241), each slot is partitioned into a **Stored region** in the first half (the first 20 Producer SWA layers, `20 × 134 × 528 B ≈ 1.35 MiB` per request, persisted across requests) and a **Dropped region** in the second half (YoCo Consumer SWA layers and compression boundary states, which can be reconstructed by replaying the last 128 tokens and therefore never need to be persisted to disk).

---

## 4.4 Dissecting `BlockManager`: Why Is Block 0 Reserved, and Why Acquire Two Locks?

Whether managing the Paged KV Pool or the Fixed-Size State Pool, both ledgers are powered by the exact same 242-line C++ class: `BlockManager` (`csrc/engine/block_manager.hpp`). Its core data structure is an array-backed doubly linked free list `Blocks` (L27–L117):

```{.mermaid caption="Figure 4-3  Lifecycle and Reference Counting of a Physical Block in BlockManager (block_manager.hpp L27–L165)"}
flowchart TB
  subgraph InitPool["1. Free Pool (FIFO Doubly Linked List; Block 0 Permanently Reserved as Null Sentinel)"]
    direction LR
    Init["Init: enqueue blocks 1..N-1 in order"] --> Free["Free State (Free)<br/>ref_count = 0; allocate() pops from head"]
  end

  subgraph ActivePool["2. Active Reference Pool (Supports Zero-Copy Prefix Sharing via fork_seq)"]
    direction LR
    Used1["Exclusive Use (Used)<br/>ref_count = 1"] -- "pin_block() increments ref_count" --> Shared["Shared Across Sequences (Shared)<br/>ref_count > 1"]
    Shared -- "free() decrements ref_count" --> Used1
    Used1 -- "free() hits 0 & appends to free list tail" --> Ret["Returned to Free Pool"]
  end

  InitPool --> ActivePool
```

Keep three key takeaways in mind when reading `block_manager.hpp`:

1. **Why is physical block 0 never allocated?** An array-backed doubly linked list needs a sentinel index analogous to `nullptr` to mark the end of the list. DeepInfer reserves index `0` as that null sentinel (L27 comment: "gpu block 0 is unused, so 0 represents null here"), making the allocatable physical block IDs `1 .. num_blocks - 1`. Conveniently, when `clone_blocks` passes `(src_block, dst_block)` pairs, `dst_block = 0` also serves as a natural sentinel meaning "no clone needed".
2. **Three-state allocation check `can_allocate` (L127–L135)**:
   - If the required block count is `>= num_blocks` (exceeding total pool capacity), it returns `AllocStatus::NEVER`, and the scheduler immediately marks the sequence `FINISHED_IGNORED`;
   - If the required block count is `<= get_num_free_blocks()`, it returns `AllocStatus::OK`;
   - Otherwise, it returns `AllocStatus::LATER`, prompting the scheduler to pause prefill batch expansion and wait for active sequences to free blocks.
3. **Why does a single-threaded scheduler need mutex locks?** In `scheduler.hpp` L189–L206, every scheduling step uses `BlockManagerMutexAcquire` to lock both ledgers in a strict, fixed order: **lock `state_manager` first, then lock `block_manager`**. Why? Because `_schedule` releases the Python GIL right at entry via `py::gil_scoped_release` (`scheduler.hpp` L276)! While the GIL is released, background Structured Output (SO) compilation threads or multimodal preprocessing threads may concurrently invoke `free` or `fork`; locking in a fixed order guarantees thread safety while eliminating any possibility of deadlock.

---

## 4.5 Chapter Summary

- **Two specialized memory ledgers**: Sequence-length-proportional compressed KV (`ca_cache`) and sparse index KV (`ica_cache`) live in the 128-token Paged KV Pool, whereas fixed-window SWA KV (`swa_cache`) and compression boundary states live in the 1-slot-per-request Fixed State Pool.
- **Exact mathematics of the two ring buffer sizes**: The Compression State Ring uses $\text{ratio} + 5$ (matching the single-step forward + speculative rollback lower bound $W + k - 1$), whereas the main-model SWA Ring uses $128 + 6 = 134$ (adding 1 slot to accommodate 128-token boundary snapshot persistence, prefill history loading, and YoCo tail retention).
- **Lightweight C++ block management**: Uses block 0 as a null sentinel, an array-backed doubly linked list for $O(1)$ allocation and deallocation, and reference counting for zero-copy prefix sharing via `fork_seq`.

Now that we understand how GPU memory is accounted for, let us turn to Chapter 5 to see how DeepInfer hides CPU scheduling overhead completely inside the shadow of GPU execution using **Zero-Overhead Scheduling (ZOS)**.
