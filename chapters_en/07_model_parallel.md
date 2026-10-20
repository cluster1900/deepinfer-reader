# Chapter 7  Model Architecture, Kernels, and Parallelism: YoCo, Wide-EP, and Expert Load Balancing

In the first six chapters, we thoroughly explored the "control plane" of the inference engine (scheduling, memory management, asynchronous pipelining, and speculative verification). In this chapter, we dive into the "data plane" to see how the computation graph of **DeepSeek-V4.1-Flash** is organized under `deepinfer/model_executor/`, and how it saturates multi-GPU clusters using the **YoCo dual-stage architecture**, **DP / CP / EP parallelism**, and **EPLB expert load balancing**.

---

## 7.1 Four-Layer Decoupled Design of the Model Code

Open `deepinfer/model_executor/models/v4_modules/__init__.py`. The module docstring at the top of the file cleanly divides more than 20 model files into four distinct architectural roles:

| Architectural Role | Representative Files (`models/v4_modules/`) | Why Decouple This Way? |
| --- | --- | --- |
| **1. Parameter Containers** | `containers.py` · `linear.py` · `mla_attn.py` · `moe.py` · `hc_blocks.py` · `engram_parameters.py` | **Only define weight shapes and types, with zero `forward` logic.** Strictly align with HuggingFace weight names, shapes, and data types, keeping the weight loader dead simple. |
| **2. Runtime Adapters & Forward** | `attn.py` · `compress.py` · `index_attn.py` · `mlp.py` · `hc.py` · `engram.py` | **Decide how computation runs at runtime.** Once weights are loaded, rearrange parameter containers into hardware-friendly layouts (e.g., packing expert weights into FP8/FP4 or Mega MoE formats) and execute the forward pass. |
| **3. Stateless Tensor Ops** | `yoco.py` · `mhc_ops.py` · `mhc_modeling.py` · `hc_lazy.py` | Hold no weight parameters; dedicated to cross-layer tensor transformations (such as YoCo sequence truncation and mHC multi-channel residual mixing and collapsing). |
| **4. Topology & Builder** | `cross_layer.py` · `engram_metadata.py` · `builder.py` | Derive cross-layer KV sharing topologies (Share-KV / Share-Index) from `PretrainedConfig`, validate invariants, and assemble the complete model tree. |

Under this architecture, a single Decoder layer in DeepSeek-V4.1-Flash integrates several frontier mechanisms:

- **Hybrid Attention and Cross-Layer Sharing (`attn.py`, `compress.py`, `index_attn.py`, `cross_layer.py`)**: Combines Sliding Window Attention (SWA), gated low-rank compressed attention (Compressed MLA), and sparse Index Attention that scores and selects Top-$k$ blocks via `relu(q @ k) * weights`. Moreover, layers in the second half directly share KV caches or index structures computed by layers in the first half (Share-KV / Share-Index).
- **Multi-Channel Hyper-Connection Residual Backbone (`hc.py`, mHC)**: Inside the model, the residual stream is expanded into `hc_mult` parallel channels; small matrices mix channels before and after each sublayer, collapsing them back into a single stream right before entering `lm_head`.
- **Sparse N-gram Memory Table (`engram.py`, Engram)**: Computes N-gram hashes of input tokens to look up sparse embeddings from a large table sharded across GPUs, gathering them via `DeepEP` and fusing them into the hidden states through a learned gate.

---

## 7.2 YoCo (You Only Cache Once): Why Can the Second Half of Layers Skip Most of Prefill?

While reading Chapters 2 and 4, we encountered the term `yoco` several times. What exactly is **YoCo**? Open `deepinfer/model_executor/models/deepseek.py` (L154–L245) and `v4_modules/yoco.py` (L15–L60), and its full mechanism becomes crystal clear:

Using `n_prefill_layer` (for example, layer 20) as the boundary, all Transformer layers in the model are split into two halves:

- **First Half (Producer Layers, `0 .. n_prefill_layer - 1`)**: Responsible for computing and producing compressed KV, index KV, and the SWA states that need to be persisted. At the end of the final Producer layer, the normalized hidden states are also saved as **`shared_hidden`**.
- **Second Half (Consumer Layers, `n_prefill_layer .. num_layers - 1`)**: They **no longer generate any new long-range KV caches** of their own; instead, they directly read the `shared_hidden` / KV computed by the first half via cross-layer sharing, or attend only to a very short local SWA window!

Now follow this structure to its logical conclusion: **Since during the Prefill phase we only need the model to emit the predicted token at the very last position of the prompt (and populate the Consumer SWA window for the final 134 positions to prepare for subsequent Decode steps), do the second-half Consumer layers really need to compute representations for thousands of earlier prompt tokens?**

**Not at all!** Because the second-half layers do not produce long-range KV caches, and subsequent tokens in the second-half layers only ever look at a local window of the most recent `keep_tokens = 134` positions!

```{.mermaid caption="Figure 7-1  YoCo dual-stage architecture and Prefill truncation mechanism (deepseek.py L185–L220 & v4_modules/yoco.py)"}
flowchart TB
  subgraph Producer["1. First-Half Producer Layers (Layers 0 .. n_prefill_layer - 1)"]
    direction LR
    P1["Input full Prompt Chunk<br/>(e.g., 4096 tokens)"] --> P2["Compute SWA & compressed KV layer by layer into cache;<br/>output full-length features + shared_hidden at boundary"]
  end

  subgraph Gate["2. Boundary Truncation & Second-Half Consumer Layers (Layers n_prefill_layer .. N-1)"]
    direction LR
    G1{"Does current chunk contain<br/>the last 134 positions?"} -- "No (middle chunk)" --> G2["skip_yoco_layers = True<br/>Skip the entire second half!"]
    G1 -- "Yes (tail window)" --> G3["Truncate to keep only the last 134 positions,<br/>run Consumer layers & lm_head to sample first token"]
  end

  Producer --> Gate
```

Looking at Figure 7-1, this explains three critical behaviors:
1. When running Chunked Prefill on a long 32K prompt, the first dozen chunks do not even touch the last 134 positions of the prompt. For those chunks, `skip_yoco_layers` evaluates to `True` (`deepseek.py` L195–L198), so the model returns immediately after finishing the first-half Producer layers—**not a single line of code in the second-half layers is executed, cutting layer compute in half!**
2. When execution reaches the final chunk, only the trailing `keep_tokens = 134` tokens need to be fed into the second-half Consumer layers to fill their local SWA windows and compute the logits for the final position.
3. This also explains why in Chapter 2 `prompt_logprobs` must be disabled whenever YoCo truncation is active—thousands of earlier prompt tokens never pass through the second-half layers or `lm_head`, so their logprobs simply do not exist!

---

## 7.3 Parallelism Strategy: Why Drop TP in Favor of DP + CP + Wide-EP?

Inspecting `ParallelConfig` (`deepinfer/config.py` L150–L180) reveals a striking architectural choice: DeepInfer **supports only Data Parallelism (DP), Context Parallelism (CP), and Expert Parallelism (EP)—with zero Tensor Parallelism (TP)**!

Why abandon TP, the most common parallelism strategy in traditional LLM inference? Because TP requires two cross-GPU `AllReduce` collectives per layer (one for attention and one for MLP), which incurs prohibitive communication latency at small Decode batch sizes. Meanwhile, DeepSeek-V4.1-Flash's MLA shared-KV design makes the attention heads compact enough to fit comfortably on a single GPU. Consequently:

- **Attention layers use DP (or symmetric Mirror CP for ultra-long contexts)**: Each GPU processes its own independent batch of requests with its own C++ scheduler and KV Cache pool, requiring zero cross-GPU communication during attention.
- **MoE expert layers use Wide-EP (wide-scale Expert Parallelism)**: When tokens across all DP ranks enter an MoE layer, `DeepEP` dispatches them all-to-all to the GPUs hosting their selected experts, executes Grouped GEMM, and combines the results back to the originating GPUs.

In `v4_modules/mlp.py` (`DeepseekMoE.forward`, L490–L610), the execution flow of a single MoE layer incorporates two aggressive communication-overlapping and kernel-fusion optimizations:

```{.mermaid caption="Figure 7-2  Two execution paths for a single DeepseekMoE layer under Expert Parallelism (EP) (v4_modules/mlp.py L490–L610)"}
flowchart TB
  subgraph NormalPath["Path A: Standard Overlap Path (Overlapping Compute with DeepEP Communication)"]
    direction LR
    N1["Local hidden_states pass through gate to select<br/>Top-k logical experts & map to physical replicas"] --> N2["DeepEP dispatch sends tokens<br/>across GPUs to target expert ranks"]
    N2 --> N3["routed_experts_forward:<br/>Grouped GEMM for routed experts"]
    N1 --> N4["shared_forward (shared experts):<br/>cap SM count (reserving comm SMs) to overlap with DeepEP"]
    N3 --> N5["DeepEP combine gathers results across GPUs<br/>and adds shared expert output in place"]
    N4 --> N5
  end

  subgraph MegaPath["Path B: Mega MoE Single-Kernel Fused Path (Ultra-Low Latency Within NVLink Domain)"]
    direction LR
    M1["Enabled when use_mega_moe = True"] --> M2["Invoke mega_moe_func monolithic fused kernel:<br/>fuses Gate + Dispatch + Grouped GEMM + Combine into one"]
  end

  NormalPath --- MegaPath
```

---

## 7.4 EPLB: How Redundant Experts Solve the Straggler Effect in MoE

Under Expert Parallelism (EP), all GPUs must synchronize at `combine` in every layer. If several "hot experts" happen to reside on Rank 0, causing Rank 0 to receive $3\times$ as many tokens as other GPUs, then the other 63 GPUs have no choice but to sit idle waiting for Rank 0 to finish—this is the notorious **straggler effect (load imbalance)** in MoE inference.

To flatten hot spots, DeepInfer implements **EPLB (Expert Parallelism Load Balancer)** in `deepinfer/model_executor/eplb/`. Its core idea is straightforward:

1. **Distinguish between "logical experts" and "physical expert replicas"**:
   Suppose the model originally has $E = 16$ logical experts distributed across $R = 4$ GPUs. We allocate $M = 4$ extra redundant slots in GPU memory, bringing the total number of physical expert slots to $E + M = 20$ (so each GPU hosts $20 / 4 = 5$ physical experts).
2. **Step 1: Allocate extra replicas to the busiest logical experts (`rebalance_experts_global`, `eplb/policy.py` L73–L140)**:
   Record the historical token load received by each logical expert at each step over a sliding window (e.g., 100 or 1,000 steps). Run $M$ rounds of greedy selection: in each round, pick the hot logical expert for which **adding 1 more replica yields the largest reduction in the sum of per-step single-replica peak loads across the historical window**, and grant it 1 additional physical replica. When a logical expert has $c$ physical replicas, the router gate splits incoming tokens evenly into $c$ shares.
3. **Step 2: Pack physical replicas onto GPUs (`_batch_greedy_indices`, `eplb/policy.py` L12–L70)**:
   We now have $E + M = 20$ physical replicas to pack into $R = 4$ bins (Ranks) of capacity 5 each. The algorithm first sorts all physical replicas by their total historical load in descending order (packing heavy boulders first, then filling gaps with small pebbles). As each replica is placed, it chooses among the non-full GPUs the one that **minimizes the sum of per-step maximum GPU loads across the historical window ($\sum_t \max_r \text{load}_{t, r}$)**!

Why does EPLB's objective function minimize the "sum of per-step maximum loads $\sum_t \max_r \text{load}_{t, r}$" rather than simply balancing the "average load over the window"?
Imagine Expert A is busy only on odd steps and Expert B is busy only on even steps, both with an average load of 50. If you only look at averages, placing A and B on the same GPU looks no different from placing them on separate GPUs. By optimizing the per-step maximum instead, the algorithm separates experts that spike simultaneously and co-locates experts whose peaks are out of phase—minimizing the straggler bottleneck on **every single step**!

We implemented this complete two-stage greedy algorithm in `minidi/eplb.py`. Under a skewed workload following a Zipf power-law distribution, Figure 7-3 compares naive placement against EPLB:

![Figure 7-3  EPLB load balancing comparison: with 16 logical experts across 4 GPUs, adding 4 redundant replicas and packing greedily by step-peak load reduces moe_lb (slowest rank / mean) from 1.38 to 1.07 (generated by minidi/eplb.py)](images/plots_en/ch07_eplb.png)

In production deployments (according to the official Q&A), Prefill instances configure 4 redundant experts per GPU with a 100-step statistics window and rearrange weights every 1,000 steps via `eplb/shuffle.py` (using NCCL P2P or intra-node IPC shared memory), achieving a measured imbalance ratio `moe_lb` as low as **1.045–1.074**; Decode instances configure 1 redundant expert per GPU and rearrange every 10,000 steps, achieving a measured `moe_lb = 1.090`.

---

## 7.5 Chapter Summary

- **Four-Layer Model Decoupling**: Parameter containers align with HuggingFace weight names, runtime adapters handle hardware layout rearrangement and forward execution, and stateless tensor ops and topology builders each fulfill a focused role.
- **YoCo Dual-Stage Architecture**: Splits layers at `n_prefill_layer` into Producer and Consumer stages; during Prefill, middle chunks that do not touch the tail window skip the entire second half of layers, while the final chunk is truncated so only the last 134 tokens enter the second half.
- **TP-Free Parallelism Philosophy**: Attention layers run DP / Mirror CP, while MoE layers run Wide-EP—overlapping communication with compute by capping the SM count of shared experts, or fusing the entire MoE layer into a single Mega MoE kernel within an NVLink domain.
- **EPLB Expert Load Balancing**: Greedily replicates hot logical experts and packs physical replicas out of phase based on historical step-peak loads, periodically hot-migrating expert weights online to eliminate the straggler effect in cross-GPU MoE communication.
