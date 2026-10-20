# 第 4 章　显存与 KV Cache：两套账本

DeepSeek-V4.1-Flash 的注意力不是标准多头注意力。从 `cache_engine.py` 的布局代码能看出，每层可能同时有：压缩后的 KV（compressed attention）、稀疏注意力的索引 KV（index attention）、滑动窗口 KV（SWA），以及压缩时需要的边界累加器（state）。DeepInfer 把这些东西分进**两套账本**：按 token 增长的放进分页块，按请求定长的放进 state 槽。

## 4.1 先算出能有多少块

`Worker.profile_num_available_blocks`（`deepinfer/worker/base.py` L212–L268）的计算过程：

```{.mermaid caption="图 4-1　KV 块数的推导（base.py L212–L268）"}
flowchart TB
  A["empty_cache、reset_peak_memory_stats"] --> B["profile_run：用最大批次跑一次 forward"]
  B --> C["peak = 峰值 torch 显存 + max(非 torch 显存, 0)"]
  C --> D["可用 = 总显存 × gpu_memory_utilization − peak"]
  D --> E["再减去 state cache：<br/>state_shape 字节数 × 2（store + copy 两份）"]
  E --> F["num_gpu_blocks = 可用 // 每块字节数"]
  F --> G["all_reduce(SUM) 全部 rank"]
  G --> H["// world_size // 2 × 2"]
```

最后一步需要注意：代码是 `all_reduce(SUM)` 再整除 `world_size`（L265–L266），得到的是**各 rank 块数的平均值**（向下取到偶数），而 `LLMEngine._init_cache` 的 docstring 写的是“take the minimum number of blocks across all workers”（`llm_engine.py` L426–L427）。两者不一致，以代码为准。在同构 GPU、同样的模型切分下，各 rank 的结果应该很接近，所以平均和最小差别不大；但如果某个 rank 因为别的进程占了显存而偏小，平均值可能超过它的真实可用量。本书没有在 GPU 上验证过这种情况会不会 OOM，只陈述代码事实。

`gpu_memory_utilization` 默认 0.9（`arg_utils.py` L34），离线示例用 0.85（`examples/offline_inference.py`）。`LLM` 的 docstring 里还有一句中文提醒：开 CUDA Graph 时如果遇到 `illegal memory access`，“可能是 cudagraph OOM 了，可以调低 gpu_memory_utilization 再试试”（`llm.py` L50–L51）。

## 4.2 块有多大

块大小不是参数，而是由模型决定：`ModelMetadata.resolved_tokens_per_block` 直接返回 `128`（`model_executor/model_metadata_utils.py` L134–L135）。`EngineArgs.tokens_per_block` 默认 `-1`，如果用户显式给了别的值，`create_engine_configs` 会断言两者相等（`arg_utils.py` L85–L86）。也就是说，**一个块代表 128 个原始 token**，但每层实际存的条目数要除以该层的压缩比：

| 缓存 | 形状（`cache_engine.py` L206–L219 注释） | 每 token 字节 |
| --- | --- | --- |
| `ca_cache` 压缩注意力 KV | `[num_blocks, block_size // ratio, 1, bytes]` uint8 | MXFP8：528；FP4（per-16）：288 |
| `ica_cache` 索引注意力 KV | `[num_blocks, block_size // ratio, 1, index_bytes]` uint8 | `index_head_dim/2 + index_head_dim/32`（FP4 + UE8M0 scale，L73–L80） |
| `swa_cache` 滑动窗口 KV | `[num_state_seqs, swa_block_size, 1, 528]` uint8 | 528 |
| `ca_state` / `ica_state` | `[num_state_seqs, ring_size, 2, dim]` fp32 | — |

528 和 288 可以从格式定义算出来（`layers/tilelang_ops/cache_common.py` L109–L127）：

- `V41_FORMAT`：512 个 e4m3 元素，每 32 个一个 ue8m0 scale，512 + 512/32 = **528** 字节；
- `V41_FP4_FORMAT`：512 个 e2m1（4 bit）元素，每 16 个一个 e4m3 scale，256 + 32 = **288** 字节。

官方 Q&A 的“低精度计算”一节对应：Main KV 用 per-16 的 e4m3 scale（“相比 NVFP4 无 global scale”），Indexer KV 用 per-32 的 ue8m0（MXFP4）。另外 `_kvcache_layout` 断言 `num_key_value_heads == 1`（L624），即所有头共享一份 KV，这是 MLA 类注意力的特征。

## 4.3 为什么 SWA 不进分页块

滑动窗口层每个请求只需要最近 `window` 个位置的 KV，和上下文多长无关。如果放进分页块，要么每个请求按全长分块（浪费），要么频繁释放窗口外的块（复杂）。DeepInfer 的选择是：**每个请求一个定长 state 槽**，SWA KV 写成环形缓冲区，位置 `p` 写到 `p % ring`。

```{.mermaid caption="图 4-2　两套账本：分页块按 token 增长，state 槽每请求一个"}
flowchart LR
  subgraph BM["block_manager：num_gpu_blocks 个块（块 0 保留）"]
    b1["块 7"] --- b2["块 3"] --- b3["块 12"] --- b4["..."]
  end
  subgraph SM["state_manager：num_state_seqs 个槽"]
    s1["槽 5：SWA ring（各层）+ 压缩 state"]
  end
  Seq["Sequence<br/>cache.block_table = [7, 3, 12, ...]<br/>cache.state_id = 5"] --> b1
  Seq --> s1
```

两套账本用的是**同一个 C++ 类** `BlockManager`：`state_manager = BlockManager(cache_config.num_state_seqs)`（`llm_engine.py` L266），只是一个“块”在这里代表一个请求的整个 state 槽。槽的数量是 `int(max_num_seqs * DEEPINFER_NUM_STATE_SLOTS_RATIO) + 1`，比例默认 1.2（`cache_engine.py` L201–L202，`environ.py` L162），多出来的 20% 给 fork 和异步调度中“旧请求还没释放、新请求已经要用”的重叠期。

ring 有多长？`get_inference_swa_block_size(window) = window + DEEPINFER_MAX_BACKTRACKING_TOKENS`（`cache_engine.py` L88–L93），而 `MAX_BACKTRACKING_TOKENS` 至少是 `nextn + 1`（`environ.py` L132）。多出来的部分是给投机解码的：一次 decode 会把主 token 和所有草稿 token 的 KV 都写进 ring，被拒绝的草稿要“回退”。官方 Q&A 给出 V4.1 的推理视角：`ring = 窗口 128 + 最大回退 6 = 134`。第 15 章会在 `minidi` 里推导这个余量的下界，并用测试验证余量不够时窗口会被覆盖。

PE（prefill 引擎）为了能在 128 的整数倍处落盘，ring 更长：官方 Q&A 写的是 `396 = 134 + PREFILL_STORE_EXTRA_SIZE(128) + YOCO_PREFILL_KEEP_TOKENS(134)`，对应代码 `get_kvstore_swa_block_size`（`cache_engine.py` L101–L110）。开源版默认 `DEEPINFER_PREFILL_STORE_EXTRA_STATE` 关闭，所以离线推理用的是 134 那一档（具体窗口值取决于模型 config）。

## 4.4 state 槽的内部分区

`StateBlock` 的类注释（`cache_engine.py` L221–L241）给出了一个槽的布局：

```text
[stored region: SWA]  [dropped region: YOCO consumer SWA | compression state]
```

- stored region：需要落盘、跨请求复用的 SWA 窗口；
- dropped region：YoCo consumer 层的 SWA（恢复靠 prefill 重算）和压缩 state（压缩窗口边界与落盘粒度 128 对齐，不跨边界依赖），不落盘。

官方 Q&A 给了 V4.1 的数字：落盘的只有前 20 层，`state_size_to_disk = 20 层 × 134 token × 528 B = 1,415,040 B/seq`，约 1.35 MiB。开源版没有落盘系统，但布局保留了这个分区，所以 `do_copy_state` / `do_save_kvcache` 这些钩子位置都还在。

## 4.5 BlockManager：链表 + 引用计数

`csrc/engine/block_manager.hpp` 只有 242 行，核心是 `Blocks`（L27–L117）：

- **块 0 保留**：“gpu blocks 0 是不用的，所以这里 0 代表空”（L27）。构造时空闲链表从 1 到 n-1，链表的 `next==0` 表示到头。后面看到 `clone_blocks` 里的 `(block, 0)` 对，0 就是“不需要复制目标”的意思（`scheduler.hpp` L431–L434）。
- **空闲块是 FIFO 链表**：`allocate` 从表头 `_pop_front`，`free` 在引用计数归零时 `_push_back` 到表尾。用 `prev/next` 下标实现双向链表，删除任意节点 O(1)。
- **引用计数**：`pin_block` 给已分配的块 +1，fork 共享前缀时用；`free` 减到 0 才回到空闲链表。
- **三态分配判断** `can_allocate_nolock`（L127–L135）：要的块数 `>= num_blocks` 返回 `NEVER`（因为块 0 不可用，最多只有 `num_blocks-1` 块），空闲够返回 `OK`，否则 `LATER`。调度器拿到 `LATER` 就停止本轮 prefill，拿到 `NEVER` 就把请求判为 `FINISHED_IGNORED`。

```{.mermaid caption="图 4-3　块的生命周期"}
stateDiagram-v2
  [*] --> Free: 构造（1..n-1）
  Free --> Used: allocate（ref=1，从链表头取）
  Used --> Used: pin_block（ref+1，fork 共享）
  Used --> Used: free（ref-1，仍 >0）
  Used --> Free: free（ref 归零，挂到链表尾）
```

`BlockManager` 继承 `Lockable`，带一把互斥锁。调度时用 RAII 的 `BlockManagerMutexAcquire` 同时锁住 `state_manager` 和 `block_manager`（L189–L206），顺序固定为先 state 后 block，避免两把锁交叉导致死锁。为什么一个单线程调度器还需要锁？因为 `_schedule` 一开头就 `py::gil_scoped_release` 释放了 GIL（`scheduler.hpp` L276），结构化输出的编译线程池、多模态处理线程等 Python 线程可以同时运行，其中 fork、free 等操作也会碰到块管理器。

## 4.6 KV 使用率怎么算

`get_gpu_cache_usage`（`scheduler.hpp` L830–L835）= 已用块数 / `num_blocks`。分母包含了保留的块 0，所以“全部可用块用光”时这个值是 `(n-1)/n`，不是 1.0。prefill 的 0.9 水位线就是拿这个值比较的。

## 4.7 开源版没有前缀缓存

`csrc/engine/` 里没有任何哈希相关代码（`grep -rn hash csrc/engine` 无结果）。`_schedule_prefill` 里有一个变量叫 `prefix_caching_hit`，但它只是标记“这个请求是新分配块的，还是 fork 来的”，不涉及跨请求的前缀匹配。跨请求的前缀复用就是 README 说的“KV cache system”，没有开源。官方 Q&A 描述了它在生产中的做法：Block KVCache 存在独立的 3FS 集群（约 100 PiB SSD），命中长度由 Block KVCache 决定，SWA KVCache 只决定最后一块是否要重算；SWA 未命中时通过 **SWA Bounded Replay** 只重算最后 128 个 token。这部分只能读文档，代码里看不到。
