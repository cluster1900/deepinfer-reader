# Chapter 11  Building Paged KV and the Block Manager from Scratch (`PagedKV` & `BlockManager`)

In `paged_forward` from Chapter 10, we relied on two core calls: `kv.write(block_table, start, k_new, v_new)` and `kv.read(block_table, end)`.

In this chapter, we open `minidi/model.py` (L49–L67) and `minidi/blocks.py` (L1–L61) to implement the paged KV memory pool and block manager from scratch, and recount the first major pitfall we hit while building `minidi`—**why omitting the "80% capacity gate" causes Chunked Prefill to deadlock the entire engine!**

---

## 11.1 Two-Step Mapping from Logical Positions to Physical Slots (`PagedKV`)

Imagine holding a large array of shape `[num_blocks, block_size, dim]` representing the physical memory pool on a GPU. So that tiny test cases can cross block boundaries frequently, `minidi` defaults to `block_size = 4` (whereas DeepInfer fixes it at `128` in production).

When a request has `block_table = [1, 2, 4]`, which physical block and which slot within that block store its $p$-th logical position (say, $p = 0, 1, \dots, 9$)? All it takes is integer division and modulo:

$$\text{Physical block ID } b = \text{block\_table}\bigl[p \mathbin{//} \text{block\_size}\bigr], \qquad \text{In-block slot } s = p \bmod \text{block\_size}$$

Using `scripts/figures/make_plots.py` to drive `minidi.blocks.BlockManager` as Request A (10 tokens) and Request B (5 tokens) grow interleaved, Figure 11-1 visualizes the resulting physical pool layout:

![Figure 11-1  Physical block pool and block table mapping as two requests grow interleaved (block_size=4, block 0 permanently reserved as the null sentinel)](images/plots_en/ch11_block_table.png)

Look at Figure 11-1:
- Request A first runs an 8-token prefill and is allocated physical blocks `[1, 2]`; next, Request B runs a 4-token prefill and is allocated physical block `[3]`;
- Later during decode, Request A grows to 10 tokens, crossing the 8-token boundary, and is allocated an additional physical block `4` (updating its block table to `[1, 2, 4]`), where block 4 has its first 2 slots filled and its last 2 slots empty;
- When Request B grows to 5 tokens, it is allocated physical block `5` (updating its block table to `[3, 5]`).

Expressing this mapping as vectorized NumPy code, `PagedKV` (`minidi/model.py` L49–L67) takes only 18 lines:

```python
class PagedKV:
    """Paged KV pool: physical block 0 is permanently reserved (matching DeepInfer block_manager.hpp L27)."""
    def __init__(self, num_blocks: int, block_size: int, dim: int):
        self.block_size = block_size
        self.k = np.zeros((num_blocks, block_size, dim))
        self.v = np.zeros((num_blocks, block_size, dim))

    def _phys_indices(self, block_table: list[int], positions: np.ndarray):
        bt = np.asarray(block_table, dtype=int)
        phys_blocks = bt[positions // self.block_size]
        slots = positions % self.block_size
        assert (phys_blocks > 0).all(), "Fatal error: attempted to read/write reserved sentinel block 0!"
        return phys_blocks, slots

    def write(self, block_table: list[int], start: int, k: np.ndarray, v: np.ndarray):
        pos = np.arange(start, start + len(k))
        b, s = self._phys_indices(block_table, pos)
        self.k[b, s], self.v[b, s] = k, v

    def read(self, block_table: list[int], end: int):
        pos = np.arange(end)
        b, s = self._phys_indices(block_table, pos)
        return self.k[b, s], self.v[b, s]
```

Notice the assertion `assert (phys_blocks > 0).all()`! Just like DeepInfer, we **permanently reserve physical block 0**. Beyond faithfully mirroring the C++ null-sentinel design, this gives us a priceless debugging tripwire: if the scheduler ever under-allocates blocks or accidentally leaves an uninitialized `0` in a block table, `PagedKV.write` immediately fires an assertion error instead of silently corrupting another request's KV cache.

---

## 11.2 Implementing `BlockManager` in 60 Lines: FIFO Free List and Reference Counting

Next, let's examine `minidi/blocks.py`, which mirrors `csrc/engine/block_manager.hpp` from Chapter 4:

1. **FIFO Free Queue (`OrderedDict` simulating a doubly linked list)**:
   At initialization, blocks `1 .. num_blocks - 1` are appended in order to `self.free`. Allocating a block (`allocate`) pops from the front (`popitem(last=False)`), while freeing a block (`free_blocks`) whose reference count reaches zero appends it to the back. Why use FIFO (queue) instead of LIFO (stack)? Not only does it match `_pop_front` / `_push_back` in C++, but in unit tests it prevents a freshly freed block from being immediately reused on the very next line—making Use-After-Free bugs much easier to catch.
2. **Reference Counting (`ref_count`) for Zero-Copy Sharing**:
   Calling `pin(blocks)` increments the reference count of allocated blocks by 1 (corresponding to `fork_seq` prefix sharing); calling `free_blocks(blocks)` decrements the reference count by 1 each time, returning a physical block to the free queue only when `ref_count == 0`.
3. **Tri-State Allocation Check `can_allocate(n)`**:
   ```python
   def can_allocate(self, n: int) -> AllocStatus:
       if n >= self.num_blocks:           # Note '>=': block 0 is reserved, leaving at most num_blocks - 1 usable blocks!
           return AllocStatus.NEVER
       return AllocStatus.OK if n <= len(self.free) else AllocStatus.LATER
   ```

---

## 11.3 How Many Blocks Does a Single Request Need? (`Sequence.get_num_blocks`)

In `minidi/sequence.py` (L69–L71), the formula that computes how many physical blocks a request currently needs is only two lines long, yet every single term is essential:

```python
def get_num_blocks(self) -> int:
    n = self.get_len() + self.inflight + len(self.draft_tokens)
    return (n + self.block_size - 1) // self.block_size
```

This formula is identical to DeepInfer's `Sequence::get_num_blocks()` (`csrc/engine/sequence.hpp` L292). Why must all three terms be summed before taking the ceiling division?

- `self.get_len()`: All confirmed tokens currently in the sequence (including the trailing token just sampled in the previous step, which will enter the model as the main query input and write its KV in this step);
- `self.inflight`: Under asynchronous scheduling (ZOS, Chapter 13), the number of in-flight tokens from the previous step that are still executing on the GPU and have not yet been finalized;
- `len(self.draft_tokens)`: Under speculative decoding (Chapter 14), the number of candidate draft tokens that will be fed into the model alongside the main input token for verification and KV writing in this step.

If you omit either of the last two terms, the moment a sequence's length lands on an exact multiple of `block_size`, `kv.write` inside `paged_forward` will index past the end of `block_table` and crash!

---

## 11.4 Real-World War Story: Why Missing the 80% Capacity Gate Causes a Deadlock

When reading `LLMEngine._init_cache` (`llm_engine.py` L445–L451) in Chapter 2, we saw a short check: if `runtime_max_model_len > block_size * num_gpu_blocks * 0.8`, the engine refuses to start. At the time, you might have wondered: "Shouldn't a single request be fine as long as it doesn't exceed 100% of available blocks? Why cap it at **80% (`0.8`)**?"

When we first wrote `minidi`, we didn't include that 80% gate either. Then, during randomized stress testing, the engine **froze in an infinite loop** under a specific parameter combination! Reconstructing that exact crime scene makes the interplay between the `0.8` capacity gate and the `0.9` watermark immediately obvious:

```{.mermaid caption="Figure 11-2  How omitting the 80% capacity gate causes an oversized Chunked Prefill to cross the 0.9 watermark and deadlock"}
flowchart TB
  subgraph Trap["Deadlock Walkthrough (num_blocks=20, block_size=4 tokens, chunk budget=16 tokens)"]
    direction LR
    T1["1. A 74-token request arrives,<br/>needing 19 blocks (19/20 = 95% of pool)"] --> T2["2. Chunk 1 is scheduled (first 16 tokens),<br/>allocating all 19 physical blocks up front!<br/>With 58 tokens left, it returns to head of waiting"]
    T2 --> T3["3. Next step tries to schedule Chunk 2:<br/>KV usage = 95% > 0.9 watermark!<br/>Scheduler refuses _schedule_prefill & runs empty Decode!"]
  end

  Dead["Result: The half-prefilled request at head of waiting holds 95% of blocks waiting for Prefill,<br/>while Prefill is permanently blocked because KV usage > 90% — permanent deadlock!"]
  Trap --> Dead
```

Look at Figure 11-2! Because DeepInfer allocates **all physical blocks required by the entire prompt up front** when scheduling the **first chunk** of a Chunked Prefill (`scheduler.hpp` L518–L520), if a single request consumes 95% of the block pool in one gulp, then as soon as its first chunk finishes, KV cache utilization has already breached `PREFILL_WATERMARK = 0.9` (the 90% watermark)! On the very next step, the scheduler sees "KV utilization > 0.9, Prefill forbidden" and refuses to call `_schedule_prefill()`. Meanwhile, the half-prefilled request holding 95% of the memory blocks sits stranded at the head of `waiting` waiting for its second chunk—it can never enter `running`, and it can never free its blocks!

Once we added DeepInfer's 80% capacity gate (`minidi/engine.py` L43–L46):

```python
max_seq_len = int(block_size * num_blocks * 0.8)
if self.scheduler.max_model_len > max_seq_len:
    raise ValueError(f"max_model_len={self.scheduler.max_model_len} > 0.8*capacity={max_seq_len}")
```

a single request can occupy at most 80% of total blocks, leaving at least a 10% safety buffer below the 90% Prefill circuit-breaker watermark and guaranteeing that any request admitted for its first chunk will always be able to finish its remaining chunks! That is the kind of design wisdom you only truly appreciate after building the system with your own hands.
