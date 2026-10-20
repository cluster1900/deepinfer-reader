# Chapter 10  The Minimal Engine: Toy Model, Flat Variable-Length Batching, and the `step` Loop

Every skyscraper starts from the ground up. To verify whether an inference engine's KV Cache and scheduler are implemented correctly, we first need a language model that actually computes causal attention and emits logits, paired with a minimal `step()` main loop.

In this chapter, we walk through `minidi/model.py` and `minidi/engine.py` to accomplish three things:
1. Build a 40-line single-layer causal attention model `TinyLM` to serve as the "golden yardstick" for the entire book;
2. Replace traditional 3D padded batches with the **flat variable-length batch (Flat Varlen Batch)** format used in production engines;
3. Construct the five-step skeleton of `Engine.step()` and implement **batch-invariant deterministic sampling**.

---

## 10.1 The Golden Yardstick: Why Does `TinyLM` Need Only 40 Lines of Code?

A natural question arises: if our goal is to test the inference engine, why not just return random numbers as fake model logits instead of writing a real attention model with Q, K, and V projections and a causal mask?

Because **fake logits cannot catch KV Cache addressing bugs**! If the model output does not depend on historical context, even if you point Request A's KV blocks to Request B's memory or write position 5 into slot 6, a dummy model will happily run without complaint. In a genuine causal attention model, by contrast, **every floating-point number at the current position depends strictly on the entire history of K and V from position $0$ up to the current position**—if block-table indexing or cache accounting is off by even a single slot, the resulting logits diverge immediately and get caught red-handed by unit tests.

Open `minidi/model.py` (L14–L46); our `TinyLM` is tiny yet complete:

```python
class TinyLM:
    def __init__(self, vocab_size=64, dim=16, max_len=256, seed=0, noise=0.0):
        rng = np.random.default_rng(seed)
        self.vocab_size, self.dim = vocab_size, dim
        # Token embedding table + sinusoidal positional encoding + single-head attention weights (Wq, Wk, Wv, Wo)
        self.emb = rng.normal(0, 0.5, (vocab_size, dim))
        pos = np.arange(max_len)[:, None] / (10000 ** (np.arange(dim)[None, :] / dim))
        self.pos = np.where(np.arange(dim)[None, :] % 2 == 0, np.sin(pos), np.cos(pos))
        self.wq, self.wk, self.wv, self.wo = [rng.normal(0, 0.3, (dim, dim)) for _ in range(4)]

    def reference_logits(self, tokens: list[int]) -> np.ndarray:
        """Golden reference implementation: uses no KV Cache, computing causal attention over [0..n-1] from scratch."""
        n = len(tokens)
        h = self.emb[tokens] + self.pos[:n]                  # [n, dim]
        q, k, v = h @ self.wq, h @ self.wk, h @ self.wv      # [n, dim]
        att = (q @ k.T) / np.sqrt(self.dim)                  # [n, n]
        mask = np.tril(np.ones((n, n), dtype=bool))          # Lower-triangular causal mask: pos i attends only to 0..i
        att = np.where(mask, att, -np.inf)
        att = np.exp(att - att.max(axis=-1, keepdims=True))
        att /= att.sum(axis=-1, keepdims=True)
        return (h + att @ v) @ self.wo                       # [n, vocab_size]
```

Look at `reference_logits(tokens)`: it is completely stateless. Feed it `[t_0, t_1, ..., t_{n-1}]`, and it faithfully computes `n` rows of logits, where the final row `logits[-1]` gives the probability distribution for predicting the next token. This serves as our **ground truth** against which every sophisticated optimization is verified.

---

## 10.2 Goodbye Padding: How Production Engines Compute Attention with Flat Variable-Length Batches

In introductory tutorials, batching multiple requests together usually means zero-padding shorter sequences to match the longest sequence in the batch, forming a 3D tensor `[batch_size, max_seq_len, dim]`. In a real inference engine, however, Request A might be computing a prefill chunk of length 15, Request B might be decoding a single token of length 1, and Request C might be verifying 4 speculative draft tokens—if you padded them to the same length, up to 90% of compute would be wasted on padding tokens!

For this reason, both DeepInfer (`InputMetadata`) and FlashMLA adopt **flat variable-length batching (Flat Varlen Batch)**: all new query tokens across all requests in the current step are concatenated end-to-end into a 2D tensor `[total_q, dim]`, accompanied by a 3-tuple of addressing metadata per request:

1. `query_tokens` (of length `q = seqlen_q`): the list of new tokens fed into the model for this request in the current step;
2. `start_pos`: the absolute index in the original sequence corresponding to index 0 of these new tokens (i.e., `num_cached`, the number of tokens already stored in the KV Cache prior to this step);
3. `block_table`: the list of physical block IDs allocated to this request.

In `TinyLM.paged_forward(kv, batch)` (`minidi/model.py` L69–L93), whether the engine is running full Prefill, Chunked Prefill, standard Decode, or speculative verification, execution follows the exact same five-step flow:

```python
def paged_forward(self, kv: PagedKV, batch: list[tuple[list[int], int, list[int]]]) -> np.ndarray:
    out = []
    for query_tokens, start, block_table in batch:
        n = len(query_tokens)                                 # New query length in this step: q = seqlen_q
        positions = np.arange(start, start + n)               # Absolute positions: start .. start+n-1
        h = self.emb[query_tokens] + self.pos[positions]      # 1. Look up token embeddings and add positional encodings
        q, k_new, v_new = h @ self.wq, h @ self.wk, h @ self.wv
        kv.write(block_table, start, k_new, v_new)            # 2. Write newly computed K/V into physical blocks first!
        k_all, v_all = kv.read(block_table, start + n)        # 3. Read all historical K/V across 0 .. start+n-1
        att = (q @ k_all.T) / np.sqrt(self.dim)               # 4. Rectangular attention matrix of shape [q, seqlen_k]
        # 5. Rectangular causal mask: the i-th query (at absolute position start+i) attends only to keys 0 .. start+i
        mask = np.arange(start + n)[None, :] <= (start + np.arange(n))[:, None]
        att = np.where(mask, att, -np.inf)
        ...
        out.append((h + att @ v_all) @ self.wo)
    return np.concatenate(out, axis=0)                        # Return flattened [total_q, vocab_size]
```

Pay special attention to the rectangular causal mask in Step 5: `np.arange(start + n)[None, :] <= (start + np.arange(n))[:, None]`:
- During **standard Decode** (`n = 1`), the single query sits at absolute position `start` and attends to all historical keys `0 .. start`;
- During **Chunked Prefill** (e.g., the 2nd chunk with `start = 16, n = 14`) or **speculative decoding verification** (`n = 1 + nextn = 4`), the $i$-th query (at absolute position `start + i`) can see all previously cached keys `0 .. start - 1` as well as earlier queries in the current step `start .. start + i`, while remaining masked from later queries `start + i + 1 .. start + n - 1`!

That is why a single `paged_forward` function effortlessly handles every stage in this book.

---

## 10.3 The Five-Step Main Loop of `Engine.step()`

With `TinyLM.paged_forward` in place, we can write the five-step loop in `minidi/engine.py` (L53–L92) that is completely isomorphic to `LLMEngine.step()` from Chapter 2:

```{.mermaid caption="Figure 10-1  The five-step closed loop of Engine.step() in the minidi synchronous engine (minidi/engine.py L53–L92)"}
flowchart TB
  subgraph Row1["Phase 1: Scheduling Decision, Flat Batch Assembly, and Paged Forward Pass"]
    direction LR
    S1["1. scheduler.schedule()<br/>Decide Prefill or Decode,<br/>select requests & seqlens_q"] --> S2["2. Assemble flat batch<br/>(query_tokens, start, block_table)"] --> F1["3. model.paged_forward()<br/>Write new K/V to blocks, then read history for attention"]
  end

  subgraph Row2["Phase 2: Position-Deterministic Sampling and Two-Stage Ledger Settlement"]
    direction LR
    F2["4. Sample by (seed, pos):<br/>Prefill tail chunk samples last row;<br/>Decode verifies drafts row by row"] --> P1["5. process_outputs():<br/>Optimistic advance -> append & check stop -> set num_cached = len - 1"] --> P2["Next queue transition:<br/>incomplete -> waiting / active -> running / finished -> free blocks"]
  end

  Row1 --> Row2
```

Looking at Steps 2 and 4 in Figure 10-1, keep two error-prone boundary invariants firmly in mind:

- **When assembling the batch, `start_pos` is always equal to `seq.num_cached`**:
  In a Prefill step, since `seq.tokens` holds the complete prompt, we slice `seq.tokens[seq.num_cached : seq.num_cached + q]` and feed it into the model. In a Decode step, thanks to the invariant `seq.num_cached == seq.get_len() - 1` from Chapter 2, `seq.tokens[-1]` is guaranteed to be the newly sampled token from the previous step sitting at index `num_cached` that has not yet been passed through the model!
- **Intermediate chunks in Chunked Prefill never sample**:
  If a prompt of length 30 is split into two chunks of `16 + 14`, after finishing the first chunk (`q = 16`), `seq.num_cached + q` (16) is still strictly less than `seq.get_len()` (30). At this point, all 16 rows of logits computed by the model are discarded, and an empty list `[]` is passed to `process_outputs`—the sole purpose of that forward pass was to populate the K/V cache for the first 16 positions. Only when the second chunk finishes the remaining 14 positions (`16 + 14 >= 30`) do we take the final row of logits and sample the 1st output token!

---

## 10.4 Implementing Batch-Invariant Deterministic Sampling in 5 Lines of Code

In Chapter 8, we saw how DeepInfer assigns each request a `seq_seed` and derives random numbers from the sampling position so that stochastic sampling remains unaffected by co-batched requests. In `minidi/engine.py` (L18–L26), we achieve the exact same mathematical property in just 5 lines of NumPy:

```python
def sample(logits_row: np.ndarray, seq: Sequence, position: int) -> int:
    if seq.temperature <= 0:
        return int(np.argmax(logits_row))
    # Key invariant: RNG state is uniquely determined by (request's seed, absolute sampling position)!
    rng = np.random.default_rng([seq.seed, position])
    probs = np.exp((logits_row - logits_row.max()) / seq.temperature)
    probs /= probs.sum()
    return int(rng.choice(len(probs), p=probs))
```

Why does seeding with `[seq.seed, position]` guarantee batch invariance?
Because for position $p$ of Request $A$, whether Request $A$ runs alone in the batch or behind 10 other requests—and regardless of whether it was preempted and recomputed along the way—the pair `[seq.seed, position]` when generating the token at position $p$ is always identical, and therefore draws the exact same random number!

In `tests/test_engine.py::test_seeded_sampling_is_batch_invariant`, we run 6 stochastic requests (`temperature = 0.8`) under two extreme configurations:
- Run 1: Large block pool and large batch size (`max_num_seqs = 8, max_num_batched_tokens = 64`), enqueued in forward order;
- Run 2: Tight block pool and tiny batch size (`max_num_seqs = 2, max_num_batched_tokens = 8`), enqueued in reverse order, triggering frequent chunking and preemptions mid-flight.

The unit test asserts that once restored to their original request order, the outputs of both runs match **100% token for token**. With this rock-solid minimal loop in place, the next chapter dives into how `PagedKV` and `BlockManager` turn scattered physical blocks into contiguous logical sequences.
