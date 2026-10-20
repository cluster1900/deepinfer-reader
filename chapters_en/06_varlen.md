# Chapter 6  DFlash Speculative Decoding and Confidence-Gated Variable-Length Verification

During the Decode phase, if running a heavy forward pass of the main model generates only a single token per step, the GPU's arithmetic logic units (ALUs) spend most of their time waiting for memory bandwidth to stream massive model weights into registers, resulting in dismal compute utilization.

The breakthrough behind **Speculative Decoding** is well known: first use a lightweight "draft module (Draft Model)" to speculate the next $\text{nextn} = 5$ candidate draft tokens in one go, and then let the main model verify `1 + 5 = 6` positions simultaneously in a single forward pass. Whenever several drafts pass verification, a single step emits multiple tokens—and through standard rejection sampling, the final generated text distribution is **100% mathematically lossless and equivalent** to token-by-token decoding.

Yet in ultra-large-scale production deployments, traditional speculative decoding suffers from two pain points:

1. **What if the draft model itself is slow?** If the draft model has to run 5 serial autoregressive steps to guess 5 draft tokens, the latency of drafting eats up a large fraction of the speedup.
2. **What if verifying a fixed 5 drafts is too wasteful?** The 5th draft can only be accepted if all first 4 drafts are guessed correctly. If a request is already uncertain at the 2nd draft (extremely low confidence), forcing the main model to verify all 5 drafts for it merely bloats the step's total token count `total_q`, pushing execution into a slower CUDA Graph bucket!

DeepInfer solves the first pain point with the **DFlash (also referred to as DSpark in the official Q&A) parallel draft architecture**, and addresses the second with **confidence-driven greedy variable-length verification (`MTPVarlen`)**. In this chapter, we dissect both mechanisms in detail.

---

## 6.1 The DFlash Draft Model: One Parallel Forward Pass for Features, Step-by-Step Sampling via a Markov Head

Open `deepinfer/model_executor/models/mtp/dflash.py` (L90–L346) and `layers/sampler.py`, and you will see that the `nextn` (MTP) speculative module in DeepInfer is not a traditional Transformer layer executed serially step by step, but a purpose-built **`DFlashDraftModel`** (L98–L104 explicitly asserts that only `DFlashNextPredictHeadHC` is supported).

Its workflow is divided into three ingenious steps, all completed seamlessly within a single Decode step:

```{.mermaid caption="Figure 6-1  Three-step pipeline of the DFlash speculative draft module: parallel feature extraction + serial Markov sampling (dflash.py L140–L345)"}
flowchart TB
  subgraph Step12["Stage 1: Main Model Updates Cache + Parallel Feature Extraction Across 5 Draft Positions"]
    direction LR
    M1["1. Main model forward() extracts target-layer features<br/>and writes to draft SWA ring via main_kv_proj"] --> D1["2. Construct 5 inputs [golden, noise x 4],<br/>run DFlash layer (window 133) in one parallel pass"]
  end

  subgraph Step3["Stage 2: Lightweight Markov Head + Confidence Head Stepwise Sampling (sample_drafts, step_i = 0..4)"]
    direction LR
    S1["3. step 0 samples draft[0] directly;<br/>step >= 1 injects prev draft embedding via markov_w1/w2"] --> S2["4. Sample draft[i] via shared lm_head and run ConfidenceHead<br/>to output independent confidence_rates[i]"]
  end

  Step12 --> Step3
```

Cross-referencing the source code in `dflash.py`, these three steps combine speed and accuracy to the extreme:

1. **Zero-overhead sharing of main model features (`update_main_cache`, `deepseek.py` L223 & `dflash.py` L140–L175)**:
   DFlash does not need its own separate prefill pass. While the main model runs its forward pass, it concatenates the hidden states from several target layers (`dspark_target_layer_ids`), projects them into K and V via `main_kv_proj`, and writes them directly into DFlash's own SWA circular buffer.
2. **Filling placeholders with noise to run all draft positions in a single parallel pass (`_embed_draft_init_tokens` & `forward_drafts`, L230–L277)**:
   Upon entering the sampling stage, the main model first samples 1 guaranteed-accepted **`golden_token`** from the last verified position of the previous round. Next, DFlash constructs an initial sequence of length `dflash_block_size = nextn = 5` for each request: position 0 holds the newly sampled `golden_token`, while positions $1 \dots 4$ are all filled with a fixed placeholder token `dflash_noise_token_id`! It then feeds all 5 positions **into DFlash's Transformer layer in a single parallel batch** (`forward_drafts`), running one forward pass over a sliding window of `128 + 5 = 133` (accelerated by a dedicated `DFlashCudaGraphRunner`) to obtain the backbone features `hidden_states` for all 5 positions.
3. **Injecting autoregressive dependency via the Markov Head + scoring via the ConfidenceHead (`sample_drafts`, L279–L345)**:
   Since positions $1 \dots 4$ were fed blind noise tokens as input, how does the $i$-th draft know which word was actually sampled at the $(i-1)$-th draft?
   `sample_drafts` resolves this with an ultra-lightweight loop `for step_i in range(5):`:
   - When `step_i == 0`, it feeds the `hidden_states` at position 0 directly into the shared vocabulary head `lm_head` to sample the first draft token `draft[0]`;
   - When `step_i >= 1`, it takes the token embedding of the draft word `draft[step_i - 1]` just sampled in the previous step, passes it through a lightweight two-layer projection `markov_w2(act(markov_w1(norm(prev_emb))))` to obtain `markov_out`, adds it to the backbone features at the current position (`combined_states = hidden_states_i + markov_out`), and feeds the sum into `lm_head` to sample `draft[step_i]`!
   - At the same time, it feeds `combined_states` into the **`ConfidenceHead`** (L312–L315) to immediately predict the independent probability `confidence_rates[:, step_i]` that the current draft token will be accepted by the main model!

In this way, the heavy Transformer attention layer runs only **1 parallel forward pass**, while serial autoregression is confined to two tiny matrices (the Markov Head). This preserves conditional dependencies across multi-step drafts while keeping draft generation latency to an absolute minimum.

---

## 6.2 Why Variable-Length Verification? From Independent Confidences to Cumulative Acceptance Rates

Now every request holds 5 draft tokens along with 5 independent confidence scores $c_1, c_2, c_3, c_4, c_5 \in [0, 1]$ produced by the `ConfidenceHead`.

Note a crucial mathematical fact (see the comment in `process_MTPInfo` at `scheduler.hpp` L1242–L1250):

> "The confidence rate returned by modeling is independent for each token; a **cumulative product** must be computed before use."

Why is a cumulative product mandatory? Because under speculative decoding verification rules, the $j$-th draft has a chance to be accepted **only if drafts $1, 2, \dots, j-1$ are all accepted**! Consequently, the **marginal expected gain $p_j$**—the probability that the $j$-th draft ultimately contributes 1 valid output token—equals the product of the first $j$ independent confidences:

$$p_j = \prod_{m=1}^{j} c_m \quad (p_1 \ge p_2 \ge p_3 \ge p_4 \ge p_5)$$

Let's look at a concrete numerical example: suppose a request's 5 independent confidences are $[0.9, 0.8, 0.5, 0.5, 0.4]$. After taking the cumulative product, the expected acceptance rates become:

$$[0.90,\; 0.72,\; 0.36,\; 0.18,\; 0.072]$$

Notice that verifying the 1st draft gives a $0.90$ chance of gaining an extra token, whereas verifying the 5th draft yields an expected gain of only $0.072$! If there are 8 requests in the batch, verifying all drafts unconditionally requires computing $8 \times (1 + 5) = 48$ tokens. By contrast, if we prune away those low-value drafts with expected gains of only $0.07$ or $0.18$ and cap the step's total token count `total_q` at **32** or fewer, we can fit into the smaller `graph_size = 32` CUDA Graph bucket—dramatically reducing step latency and producing *more* valid tokens per unit time!

---

## 6.3 `MTPVarlen::schedule`: Greedily Searching for Peak Throughput

This algorithm for "pruning low-yield drafts to hit the optimal graph bucket" is implemented in `csrc/engine/mtp_varlen.hpp` (L150–L267). Its logic is remarkably intuitive and can be broken down into four steps:

1. **Mandatory baseline tokens**: Suppose there are $B$ active requests in this step. Each request must verify at least 1 main input token (the token sampled in the previous step), which is 100% guaranteed to produce 1 new token. Thus, the initial total batch size is $\text{cur\_bsz} = B$, and the initial expected token yield is $\text{cur\_expected} = B$.
2. **Sort all candidate drafts globally by cumulative acceptance rate in descending order**: Push the cumulative acceptance rates $p_{i, j}$ of all 5 drafts across each of the $B$ requests into a single global array `queue`, and sort `queue` in descending order of $p_{i, j}$ (L188–L214).
   - *Quiz: After sorting globally by value, could we ever encounter an invalid state where a request's 3rd draft is selected, but its 2nd draft is not?*
   - *Answer: Never!* Because within any single request, the cumulative acceptance rates are monotonically non-increasing ($p_{i, 1} \ge p_{i, 2} \ge \dots \ge p_{i, 5}$). When taking a prefix from largest to smallest, an earlier draft will always be selected before a later draft of the same request, naturally forming a valid prefix!
3. **Add drafts one by one and evaluate expected throughput**: Traverse `queue` from highest to lowest. Each time the $k$-th high-value draft is added, the total token count becomes $\text{cur\_bsz} = B + k$, and the expected output increases by $\text{cur\_expected} += p_{(k)}$. Next, look up the estimated execution time $\text{time\_ms}(\text{cur\_bsz})$ for the CUDA Graph bucket corresponding to $\text{cur\_bsz}$, and compute the expected throughput:
   $$\text{Throughput}(\text{cur\_bsz}) = \frac{\text{cur\_expected}}{\text{time\_ms}(\text{cur\_bsz})}$$
   Record the optimal total token budget **`best_bsz`** that maximizes $\text{Throughput}$ (L235–L259).
4. **Distribute quotas according to `best_bsz`**: Assign the top `best_bsz - B` selected drafts in `queue` back to their respective requests, yielding each request's actual verification length `seqlens_q[i]` for this step (L263–L266).

Here, the single-step runtime estimation formula is (L243):

$$\text{time\_ms} = \text{baseline\_ms}(\text{graph\_size}) + \text{slope\_ms} \times \text{graph\_size} + \text{sampler\_ms}$$

- `baseline_ms`: The baseline step-function latency measured during engine startup (`warm_up_model`) by timing each CUDA Graph bucket 5 times, dropping the min/max outliers, and averaging (`model_runner.py` L562 `profile_decode_perf`);
- `slope_ms`: The median slope dynamically tracked at runtime by `ForwardRuntimeEstimator` (L44–L131) over the most recent 5,000 steps as `(measured forward_ms - baseline_ms) / graph_size`, compensating for the growth in attention latency as context length increases;
- `sampler_ms`: The median sampler latency over the last 9 steps, tracked by `SamplerRuntimeEstimator` (L20–L42).

By running an example with 8 requests and 5 drafts per request through `minidi/varlen.py` (which is line-by-line isomorphic to the C++ implementation), we can visualize this throughput curve with its step jumps:

![Figure 6-2  Greedy optimization in variable-length verification: throughput exhibits sawtooth drops at CUDA Graph bucket boundaries, and the algorithm automatically locks onto the global peak at total_q=24 (generated by minidi/varlen.py)](images/plots_en/ch06_varlen.png)

Look at Figure 6-2: whenever `total_q` crosses a CUDA Graph bucket boundary at 16, 24, 32, or 48, the denominator latency jumps up a tier and the curve dips sharply. The greedy algorithm pinpoints the total budget right at **`total_q = 24` (the right edge of the `graph_size = 24` bucket)**, pruning the remaining 24 low-yield drafts and achieving a significantly higher throughput than the green dashed line (which verifies all 48 positions unconditionally). According to the official Q&A, enabling variable-length verification in production reduces the average number of verified drafts from 5.0 to 3.2–4.4 (saving roughly 30% of verification compute), while the average accepted length remains virtually unchanged (3.29 vs. 3.20), delivering a **4%–7%** end-to-end throughput gain.

---

## 6.4 When Variable-Length Verification Meets Asynchronous Scheduling (ZOS): Fixing the Total Budget at Step `N-2`, Reallocating Quotas on GPU at Step `N-1`

Finally, let's examine an elegant design that weaves together Chapter 5 (ZOS) and Chapter 6 (variable-length verification):

When asynchronous scheduling (ZOS) is enabled and the CPU is running `schedule(N+1)`, **the `ConfidenceHead` for step $N$ (the previous round) has not finished executing on the GPU yet!** The CPU only has the older confidence scores copied back from an even earlier round (**step $N-2$**). How can it schedule variable-length verification for step $N+1$?

DeepInfer (comments in `mtp_varlen.hpp` L160–L169 and `worker.py` L218–L219) adopts a two-level cooperative scheme: **"the CPU fixes the total budget using step $N-2$ confidences, while the GPU reallocates per-request quotas in place using the freshly computed step $N-1$ confidences"**:

```{.mermaid caption="Figure 6-3  CPU-GPU cooperative variable-length verification under asynchronous scheduling (ZOS) (mtp_varlen.hpp & schedule_mtp_varlen.py)"}
flowchart TB
  subgraph CPU_Stage["1. CPU Ahead-of-Time Scheduling Stage (schedule(N+1))"]
    direction LR
    C1["CPU reads finalized step N-2 confidences<br/>(prev_accept_rates)"] --> C2["Run MTPVarlen::schedule() to<br/>determine optimal total budget best_bsz"]
    C2 --> C3["Lock CUDA Graph bucket by best_bsz<br/>and pre-allocate input tensor shapes"]
  end

  subgraph GPU_Stage["2. GPU In-Place Reallocation Immediately After Sampling (execute_model_step Stages 1 & 3)"]
    direction LR
    G1["GPU finishes sample(N) and produces<br/>fresh step N-1 confidence_rates"] --> G2["TileLang kernel compute_seqlens_q_by_confidence_rates:<br/>keeping best_bsz fixed, recomputes cumulative products<br/>+ global Top-K to reassign each request's seqlens_q"]
    G2 --> G3["Feed directly into forward(N+1) while<br/>copying asynchronously (D2H) to CPU for Stage 3 metadata"]
  end

  CPU_Stage --> GPU_Stage
```

Why does this two-level scheme work so well?

- **The total budget `best_bsz` is insensitive to single-step fluctuations**: Across dozens of requests in a batch, the aggregate distribution of confidence scores is statistically stable. The optimal total bucket `best_bsz` computed from step $N-2$ is almost always identical to that from step $N-1$, so the CPU can safely use it to lock in the CUDA Graph bucket.
- **Which specific request gets how many draft slots is highly sensitive to the latest confidences**: A request might have just sampled a punctuation mark or a deterministic phrase in the last step, causing its confidence to spike instantly. Therefore, by holding the total budget `best_bsz` constant and having the GPU launch a fast TileLang kernel `compute_seqlens_q_by_confidence_rates` (`layers/tilelang_ops/schedule_mtp_varlen.py`) right before `forward(N+1)` to recompute cumulative products and global Top-$K$ quota assignments using the fresh step $N-1$ confidences, DeepInfer preserves the static tensor shapes required by CUDA Graph while still enjoying 100% up-to-date per-request verification lengths!

---

## 6.5 Chapter Summary

- **DFlash Draft Architecture**: Directly updates the draft SWA cache using intermediate-layer features from the main model (zero extra prefill), extracts backbone features for 5 positions in a single parallel forward pass using `[golden, noise, ...]`, and samples drafts step by step while outputting independent confidence scores via an ultra-lightweight Markov Head (`markov_w1/w2`) and `ConfidenceHead`.
- **Confidence-Gated Variable-Length Verification (`MTPVarlen`)**: Takes the prefix cumulative product of independent confidences to obtain marginal expected gains, greedily enumerates total verification length `cur_bsz` in descending order, and combines this with the step-function latency curve of CUDA Graphs to find the `best_bsz` that maximizes expected throughput.
- **Seamless Integration with ZOS**: The CPU locks in the total budget `best_bsz` and CUDA Graph bucket using step $N-2$ confidences, while the GPU redistributes `seqlens_q` across individual requests in place inside a custom kernel using the freshly produced step $N-1$ confidences.
