# Chapter 14  Building Speculative Decoding and Greedy Variable-Length Verification (`nextn` & `varlen.py`)

In this chapter, we implement on CPU the two core speculative decoding mechanisms dissected in Chapter 6—**speculative draft generation with single-step parallel verification and rollback** (`minidi/engine.py`), and **the greedy variable-length verification algorithm based on cumulative confidence and step-time curves** (`minidi/varlen.py`)—and use both a mathematical proof and brute-force enumeration tests to answer a fundamental question: **Why is globally sorting by cumulative acceptance rate and greedily taking a prefix guaranteed to find the global optimum?**

---

## 14.1 Implementing Speculative Draft Generation and Single-Step Verification in 25 Lines

To simulate draft models of varying quality in `minidi` (analogous to DeepInfer's DFlash module), we added a controllable `noise` parameter to `TinyLM`. When `noise = 0.0`, the draft model is identical to the main model (100% draft acceptance rate); when `noise > 0`, Gaussian perturbations are added to the draft model's weights—the larger the noise, the higher the probability of a wrong guess.

Before a Decode step begins, `Engine._propose(seq)` (`minidi/engine.py` L48–L51) lets the draft model rapidly guess `nextn` candidate draft tokens and stores them in `seq.draft_tokens`. Then, in `Engine.step()`, the main model concatenates **`[seq.tokens[-1]] + seq.draft_tokens`** (`q = 1 + nextn` tokens in total) and computes `q` rows of logits in parallel via a single `paged_forward` call!

Once the main model has those `q` rows of logits, how does it decide how many draft tokens to accept? Look at the verification loop in `minidi/engine.py` L70–L80:

```python
# rows has shape [q, vocab_size], where q = 1 + len(seq.draft_tokens)
accepted = []
for j in range(q):
    # Row j of logits is the main model's true prediction for position start+j+1 after seeing the main input token and the first j draft tokens!
    t = sample(rows[j], seq, start + j + 1)
    accepted.append(t)
    # If we are in the first q-1 rows (checking draft j) and the main model's sampled token t disagrees with draft_tokens[j]:
    if j < q - 1 and t != seq.draft_tokens[j]:
        break                                   # Immediately reject all subsequent drafts! But keep the bonus correction token t just sampled by the main model!
```

Using a concrete scenario where $\text{nextn} = 3$ and the 2nd draft token is guessed wrong, let's trace the inputs and outputs of those 8 lines of code:

```{.mermaid caption="Figure 14-1  Single-step speculative verification and automatic bonus correction token (with nextn = 3 and the 2nd draft rejected)"}
flowchart TB
  subgraph Input["1. 4 Positions Fed in Parallel to Main Model in One Step (start .. start+3)"]
    direction LR
    I0["Pos start: Main input t<br/>(last confirmed token)"] --> I1["Pos start+1: Draft d₁"] --> I2["Pos start+2: Draft d₂"] --> I3["Pos start+3: Draft d₃"]
  end

  subgraph Verify["2. Main Model Runs One paged_forward to Produce 4 Logit Rows & Verifies Row by Row"]
    direction LR
    R0["Row 0 Logits (after seeing t)<br/>Main samples d₁ == Draft d₁ ✔<br/>(1st draft accepted!)"] --> R1["Row 1 Logits (after seeing t, d₁)<br/>Main samples x ≠ Draft d₂ ✘<br/>(2nd draft rejected; keep correction token x & break!)"] --> R2["Rows 2 & 3 Logits (after seeing d₂, d₃)<br/>Premise d₂ was wrong; discard both!"]
  end

  subgraph Commit["3. process_outputs Ledger Commit and Automatic KV Rollback"]
    direction LR
    C1["Append 2 valid tokens this step: [d₁, x]<br/>(Even if 0 drafts pass, 1 main token is guaranteed)"] --> C2["Reset num_cached = get_len() - 1:<br/>KV slots written by d₂, d₃ are invalidated & overwritten next step!"]
  end

  Input --> Verify --> Commit
```

Looking at Figure 14-1, you can see the two most elegant properties of speculative decoding:

1. **Guaranteed progress on every step (at least 1 token, up to $1 + \text{nextn}$ tokens)**:
   - Even if the very first draft $d_1$ is wrong, Row 0 of the logits was computed by the main model after seeing the true prefix $t$, so the correction token $x$ sampled from Row 0 is a 100% valid next token!
   - When the first draft $d_1$ is right, Row 1 of the logits becomes the main model's distribution after seeing the true prefix $(t, d_1)$! Even if the second draft $d_2$ is wrong, the correction token $x$ sampled from Row 1 is a free 2nd valid token!
   - If all 3 drafts $d_1, d_2, d_3$ are right, Row 3 of the logits (after seeing $t, d_1, d_2, d_3$) gives you 1 extra bonus token, yielding $3 + 1 = 4$ tokens in a single step!
2. **Rollback under paged KV is "zero-cost"**:
   During `paged_forward`, the K and V vectors for all 4 positions ($t, d_1, d_2, d_3$) were already written into physical blocks. Now that $d_2, d_3$ have been rejected, do we need to go into the physical blocks and erase them? **Not at all!** Because the final line of `process_outputs`—`seq.num_cached = seq.get_len() - 1`—rewinds the cached-token pointer to the exact valid boundary (advancing by 2 slots instead of 4), the next Decode step will naturally start writing from the new `num_cached` and overwrite the stale entries. And if $d_2, d_3$ happened to cross a block boundary and allocate an extra physical block, the `elif need < 0:` branch in `_schedule_decode` on the next step will automatically return that redundant block to the block manager!

In `tests/test_engine.py::test_speculative_decoding_is_lossless`, we run full generation with `noise = 0.0` (100% hit rate), `0.2` (high hit rate), and `0.6` (frequent rejections), asserting that every output is 100% identical token for token to `reference_generate` without speculative decoding.

---

## 14.2 Implementing `MTPVarlen` Greedy Variable-Length Verification in 35 Lines and Proving Its Global Optimality

Next, let's examine `minidi/varlen.py` (56 lines in total). It replicates line by line the C++ function `MTPVarlen::schedule` (`csrc/engine/mtp_varlen.hpp` L150–L267) from Chapter 6:

```python
def cumulative(conf: list[float]) -> list[float]:
    """Corresponds to scheduler.hpp L1242-L1250: compute the prefix product of independent per-position draft confidences."""
    out, p = [], 1.0
    for c in conf:
        p *= c
        out.append(p)
    return out

def schedule_varlen(acc_rates: list[list[float]], curve: list[tuple[int, float]],
                    slope_ms: float = 0.0, sampler_ms: float = 0.0):
    """Corresponds to mtp_varlen.hpp L150-L267: greedily find the total budget best_bsz that maximizes expected_tokens / step_time."""
    n = len(acc_rates)
    # Sort all candidate draft tokens across all requests by (cumulative acceptance rate descending, request index ascending)
    queue = sorted(((r, i) for i, rs in enumerate(acc_rates) for r in rs), key=lambda x: (-x[0], x[1]))
    best_tp, best_bsz = 0.0, 0
    expected, bsz = float(n), n               # Baseline: every request must verify 1 main input token

    def update():
        nonlocal best_tp, best_bsz
        t = step_time(bsz, curve, slope_ms, sampler_ms)
        if t is not None and expected / t > best_tp:
            best_tp, best_bsz = expected / t, bsz

    update()
    for r, _ in queue:                        # Greedily add 1 draft token with the highest marginal gain at a time
        expected += r
        bsz += 1
        update()

    seqlens_q = [1] * n
    for _, i in queue[: best_bsz - n]:
        seqlens_q[i] += 1
    return seqlens_q, best_bsz, best_tp
```

### Why Is This Simple `O(K log K)` Greedy Algorithm Guaranteed to Find the Global Optimum?

Suppose a batch has $B$ requests, and each request can choose a prefix of $k_i \in \{0, 1, \dots, 5\}$ draft tokens. The total number of combinations is $6^B$ (an astronomical $6^{64}$ when $B=64$). Why doesn't `schedule_varlen` need dynamic programming, and how can a simple sort-and-scan find the exact global optimum?

Because the problem simultaneously satisfies two clean mathematical properties:
1. **The denominator depends only on the total token count**: The single-step latency `step_time(bsz)` depends solely on the total number of verified tokens $\text{bsz} = B + \sum_{i=1}^B k_i$, regardless of which requests those tokens are assigned to. Therefore, if for every fixed total count $\text{bsz}$ we can pick the subset of $\sum k_i = \text{bsz} - B$ draft tokens that maximizes the numerator (expected total tokens produced), and then take the maximum across all possible values of $\text{bsz}$, we are guaranteed to find the global optimum!
2. **Monotonicity of the numerator guarantees that the "globally largest subset" is automatically a "valid per-request prefix"**: Because the cumulative acceptance rates within each request $i$ are monotonically non-increasing, $p_{i, 1} \ge p_{i, 2} \ge \dots \ge p_{i, 5} \ge 0$ (each factor in the product lies in $[0, 1]$), whenever we pick the top $M = \text{bsz} - B$ largest elements across all candidate drafts globally, if the $j$-th draft of request $i$ is selected, then its preceding drafts $1 \dots j-1$—whose values are greater than or equal to it—**must also have been selected**!

To verify this mathematical proof in code, we wrote a brute-force enumerator in `tests/test_algorithms.py::test_varlen_greedy_is_optimal_over_prefixes` that exhaustively checks all $5^4 = 625$ valid prefix combinations across 4 requests, confirming that `best_tp` returned by `schedule_varlen` matches the brute-force global maximum to the last decimal place!
