# 第 4 章　显存与 KV Cache：分页块池与定长状态池的两套账本

在标准的 Transformer 推理引擎（如原版 vLLM）里，所有层的 KV Cache 都长得一模一样：序列每增长 1 个 token，所有层都往分页块（Paged Block）里追加 1 个位置的 K 和 V。

但在 **DeepSeek-V4.1-Flash** 中，这种“一刀切”的分页方案行不通了。因为它的每一层可能同时包含四种截然不同的缓存对象：

1. **压缩注意力 KV（`ca_cache`）**：把连续若干个 token 门控压缩成一个潜在向量，随序列长度线性增长；
2. **稀疏索引 KV（`ica_cache`）**：给稀疏注意力选择器（DeepSelect）打分用的超低比特索引，同样随序列长度线性增长；
3. **滑动窗口注意力 KV（`swa_cache`）**：无论是主模型的 SWA 层还是 DFlash 草稿模型，都只看最近 $W = 128$ 个 token，跟序列总长是 500 还是 50,000 毫无关系；
4. **压缩边界累加器（`ca_state` / `ica_state`）**：在相邻压缩窗口交界处暂存未凑齐一个压缩步长的残余状态，大小同样是固定的。

如果把第 3、4 种“固定大小、循环覆写”的数据也硬塞进按全长分配的分页块表里，不仅会白白浪费海量显存，还会让块回收逻辑变得极其拧巴。因此，DeepInfer 的核心设计是**建立两套独立的显存账本**：

- **账本一：`block_manager`（分页 KV 池）**——按 token 块动态增长，管理 `ca_cache` 和 `ica_cache`；
- **账本二：`state_manager`（定长状态池）**——每条活跃请求独占 1 个定长槽位（`state_id`），以环形缓冲区（Ring Buffer）方式管理 `swa_cache` 与压缩状态。

---

## 4.1 启动探底：GPU 到底能切出多少个分页块？

在分配这两本账之前，引擎先要在 `Worker.profile_num_available_blocks`（`deepinfer/worker/base.py` L212–L268）里摸清家底：

```{.mermaid caption="图 4-1　可用 KV 分页块数的探测与计算流程（deepinfer/worker/base.py L212–L268）"}
flowchart TB
  subgraph Step1["第一阶段：实测最大批次前向的峰值显存"]
    direction LR
    P1["清空缓存并重置统计<br/>empty_cache + reset_peak"] --> P2["执行 profile_run()<br/>用最大批次跑一次假前向"]
    P2 --> P3["记录前向峰值显存 peak：<br/>PyTorch 峰值 + 非 PyTorch 显存占用"]
  end

  subgraph Step2["第二阶段：扣减定长 State 池并折算分页块数"]
    direction LR
    Q1["可支配总池 = 总显存 × gpu_memory_utilization − peak"] --> Q2["优先扣除定长 State 池：<br/>减去 state_cache 字节数 × 2（主池 + 拷贝缓冲）"]
    Q2 --> Q3["单卡块数 = 剩余字节 // 单块字节数；<br/>跨卡 all_reduce(SUM) 取均值并向下对齐到偶数"]
  end

  Step1 --> Step2
```

这里有两个值得留意的工程细节：

1. **定长 State 池优先扣除，且留双份（`× 2`）**：State 池的大小只取决于最大并发请求数 `num_state_seqs`，因此先把它的显存扣掉（乘 2 是因为除了主状态池 `state_cache`，还要留一份同尺寸的 `state_cache_to_copy` 供落盘拷贝使用，见 `base.py` L244–L249），剩下的显存再全部切成分页 KV 块。
2. **多卡取平均并对齐偶数**：代码第 265–266 行通过 `all_reduce(SUM)` 除以 `world_size` 再 `// 2 * 2` 得到全组统一的块数（注：`llm_engine.py` L426 的注释写的是取最小值，但实际代码执行的是取均值并对齐偶数）。

---

## 4.2 账本一：分页 KV 块有多大？每个 Token 占多少字节？

在 DeepInfer 中，一个分页块代表多少个原始 token 是由模型元数据锁定的：`ModelMetadata.resolved_tokens_per_block` 固定返回 **`128`**（`model_metadata_utils.py` L134–L135）。

也就是说，**逻辑上一个物理块对应 128 个原始 token**。但由于压缩注意力层每 `ratio` 个 token 才产出一个压缩 KV 条目，物理张量的第二维长度实际上是 `128 // ratio`。我们对照 `deepinfer/worker/cache_engine.py`（L206–L219）和 `layers/tilelang_ops/cache_common.py`（L109–L127），可以把两套池子的张量形状和字节账算得一清二楚：

| 所属账本 | 缓存名称 | 物理张量形状（`cache_engine.py`） | 数值格式与每条目字节数 |
| --- | --- | --- | --- |
| **分页 KV 池**<br/>（`block_manager`） | `ca_cache`<br/>（压缩注意力 KV） | `[num_blocks, 128 // ratio, 1, bytes]` | **MXFP8**：512 个 `e4m3` + 16 个 `ue8m0` 缩放因子 = **528 B**<br/>**FP4 (per-16)**：512 个 `e2m1`(256B) + 32 个 `e4m3` 缩放因子 = **288 B** |
| **分页 KV 池**<br/>（`block_manager`） | `ica_cache`<br/>（稀疏索引 KV） | `[num_blocks, 128 // ratio, 1, idx_bytes]` | **MXFP4**：`index_head_dim / 2` 字节数据 + `index_head_dim / 32` 字节 `ue8m0` 缩放因子 |
| **定长 State 池**<br/>（`state_manager`） | `swa_cache`<br/>（滑动窗口 KV） | `[num_state_seqs, swa_block_size, 1, 528]` | **MXFP8**：每个窗口槽位固定 **528 B**，环形缓冲区长度 `swa_block_size = 134` |
| **定长 State 池**<br/>（`state_manager`） | `ca_state` / `ica_state`<br/>（压缩边界状态） | `[num_state_seqs, ring_size, 2, dim]` | **FP32**：保存压缩窗口交界处的残余向量，环形长度 `ring_size = ratio + 5` |

注意表中第三列那个醒目的 **`1`**：`_kvcache_layout`（L624）明确断言 `num_key_value_heads == 1`——这就是 DeepSeek 标志性的 **MLA（多头潜在注意力）** 架构：所有注意力头共享同一份压缩后的潜在 KV 向量，把 KV 显存占用压缩到了传统多头注意力的几十分之一！

---

## 4.3 账本二：为什么压缩 Ring 是 `W + 5`，而 SWA Ring 是 `128 + 6 = 134`？

现在我们来看定长状态池（`state_manager`）。每条进入系统的请求都会分到一个独立的 `state_id`，同时指向自己的 SWA 环形缓冲区和压缩状态环形缓冲区：

```{.mermaid caption="图 4-2　单条请求同时持有分页块表（block_table）与定长状态槽（state_id）"}
flowchart TB
  Seq["活跃请求 Sequence<br/>cache.block_table = [7, 3, 12] · cache.state_id = 5"]

  subgraph Paged["账本一：block_manager（随序列长度动态追加块，块 0 保留）"]
    direction LR
    B7["物理块 7<br/>tokens 0..127"] --> B3["物理块 3<br/>tokens 128..255"] --> B12["物理块 12<br/>tokens 256.."]
  end

  subgraph State["账本二：state_manager（每请求独占 1 个定长槽，按下标取模循环覆写）"]
    direction LR
    S5A["槽 5 · Stored 区<br/>前 20 层 SWA Ring（134 槽 × 528 B）"] --- S5B["槽 5 · Dropped 区<br/>YoCo 后半段 SWA + 压缩状态 Ring（ratio + 5 槽）"]
  end

  Seq --> B7
  Seq --> S5A
```

这里有一个极其深刻、很多读者第一次看源码都会疑惑的问题：

> 既然滑动窗口大小是 $W = 128$，投机解码草稿数是 $\text{nextn} = 5$（即一步 decode 最多送入 $k = 1 + 5 = 6$ 个 token），为什么**压缩状态 Ring** 的长度是 $\text{ratio} + 5$，而 **SWA Ring** 的长度却是 $128 + 6 = 134$？那个相差的 **`1`** 到底是从哪儿来的？

我们先把源码里的两个公式摆出来（见 `deepinfer/environ.py` L132 与 `deepinfer/worker/cache_engine.py` L88–L133）：

1. 环境变量 `DEEPINFER_MAX_BACKTRACKING_TOKENS` 默认等于 `nextn + 1 = 6`（即单步 decode 送入模型的最大 token 数 $k = 1 + \text{nextn}$）。
2. **压缩状态 Ring 大小**（`get_inference_compress_block_size`，`cache_engine.py` L131–L133 及 `compress.py` L98）：
   $$\text{compress\_ring\_size} = \text{compress\_ratio} + \text{DEEPINFER\_MAX\_BACKTRACKING\_TOKENS} - 1 = \text{ratio} + 5$$
3. **SWA Ring 大小**（`get_inference_swa_block_size`，`cache_engine.py` L88–L93）：
   $$\text{swa\_block\_size} = \text{window\_size} + \text{DEEPINFER\_MAX\_BACKTRACKING\_TOKENS} = 128 + 6 = 134$$

为什么两边一个减了 1、一个没减 1？我们用小数字推演一下就彻底明白了：

### 第一步：单步前向与投机回退的理论最小下界是 `W + k - 1 = W + nextn`
假设窗口大小是 $W$，本步从位置 $L$ 开始一口气往环形缓冲区里写入 $k = 1 + \text{nextn} = 6$ 个新位置：$L, L+1, \dots, L+k-1$。
- 在这一步内部，排在最前面的第 1 个 query（位置 $L$）需要回头看它前面最近的 $W - 1$ 个历史位置：$[L - (W - 1), \dots, L - 1]$。
- 同时，万一本步验证后 5 个草稿全部被拒绝，下一步要退回到位置 $L+1$ 重新开始，那 $W - 1$ 个旧历史加上位置 $L$（共 $W$ 个有效位置）也绝不能被那 5 个废草稿覆盖！
- 因此，在环形缓冲区里必须同时共存的位置总数是：
  $$\underbrace{(W - 1)\text{ 个旧历史}}_{L-(W-1)\,\dots\,L-1} + \underbrace{k\text{ 个本步新写入位置}}_{L\,\dots\,L+k-1} = W + k - 1 = W + \text{nextn}$$
- 这正是**压缩状态 Ring** 取 `compress_ratio + 5`（显式减 1）以及 **DFlash 草稿层 SWA** 取 `128 + 5 = 133`（`get_kv_index.py` L183）的原因！第 15 章我们会用 `minidi` 画出槽位覆盖图，直观展示少 1 个槽位就会发生历史覆盖。

### 第二步：为什么主模型 SWA Ring 不减这 1，非要取 `128 + 6 = 134`？
因为主模型的 SWA 环形缓冲区除了应付单步前向和投机回退，还要承担**按 128 边界异步快照落盘（KVCache Store）**与**预填充历史加载**三项额外任务（见官方 Q&A L189–L235 及源码）：

1. **Decode 跨 128 边界时的状态快照（`llm_engine.py` L930 `do_copy_state` 与 `state_cache.py` L44 `select_state_slice`）**：
   线上系统只在序列长度跨过 **128 的整数倍**（`KVSTORE_GRAIN = 128`）时，才会把当前请求的 SWA 状态拷贝出来落盘。设想最极端的情况：某条请求在本步之前已经缓存了 $127$ 个 token（位置 $0 \dots 126$），本步 decode 一口气写入了 $k = 6$ 个 token（位置 $127 \dots 132$），跨过了 $128$ 的边界！
   在本步结束时，`do_copy_state` 触发，需要从环形缓冲区里完整切出截至位置 $127$ 的那 **整整 128 个历史槽位**（位置 $0 \dots 127$）去落盘；而此时位置 $128 \dots 132$（共 5 个超调位置）也已经写进了同一个环形缓冲区里！如果还要留 1 个对齐安全位或算上本步写入的全部 6 个偏移，环形缓冲区拥有 $128 + 6 = 134$ 个槽位，就能保证无论单步 6 个 token 落在 128 边界的哪个相位上，边界前完整的 128 个 token 窗口都原封不动地躺在 Ring 里等待拷贝！
2. **Prefill 历史窗口加载（`prepare_tensor_input_v4.cpp` L378–L383 `make_swa_block_load_plan`）**：
   当新 chunk 开始前需要把历史 SWA 状态载入 Ring 时，加载计划直接读取 `[history_end - 128, history_end)` 整整 128 个严格先前的历史位置，再加上本步最多 6 个回退余量，正好占满 $128 + 6 = 134$ 个槽位。
3. **YoCo 预填充截断与前缀缓存命中上限（`prepare_tensor_input_v4.cpp` L355 及官方 Q&A L231–L235）**：
   为了确保 YoCo 后半段层和 SWA Bounded Replay 有完整的窗口可用，未命中的尾部保留长度固定为 `window_size + max_backtracking_tokens = 128 + 6 = 134`。

另外，在 `StateBlock`（`cache_engine.py` L221–L241）内部，槽位还被精心地划分为前半段 **Stored 区**（前 20 层 Producer SWA，每请求 `20 × 134 × 528 B ≈ 1.35 MiB`，需要跨请求落盘复用）和后半段 **Dropped 区**（YoCo Consumer 层的 SWA 与压缩状态，可通过重算最后 128 个 token 恢复，无需落盘）。

---

## 4.4 `BlockManager` 源码剖析：为什么块 0 不用？为什么需要双锁？

不管是分页 KV 池还是定长 State 池，底层用的都是同一个仅有 242 行 C++ 代码的 `BlockManager`（`csrc/engine/block_manager.hpp`）。它的核心数据结构是一个用数组下标模拟的双向空闲链表 `Blocks`（L27–L117）：

```{.mermaid caption="图 4-3　BlockManager 中单个物理块的生命周期与引用计数流转（block_manager.hpp L27–L165）"}
flowchart TB
  subgraph InitPool["1. 空闲池（FIFO 双向链表，块 0 永久保留为空指针哨兵）"]
    direction LR
    Init["初始化：块 1..N-1 依次入队"] --> Free["空闲状态 (Free)<br/>ref_count = 0，allocate() 从表头弹出"]
  end

  subgraph ActivePool["2. 活跃引用池（支持 fork_seq 零拷贝共享前缀）"]
    direction LR
    Used1["独占使用 (Used)<br/>ref_count = 1"] -- "pin_block() 引用加 1" --> Shared["多请求共享 (Shared)<br/>ref_count > 1"]
    Shared -- "free() 引用减 1" --> Used1
    Used1 -- "free() 归 0 挂回空闲链表尾" --> Ret["归还空闲池"]
  end

  InitPool --> ActivePool
```

读 `block_manager.hpp` 时记住三个要点：

1. **为什么物理块 0 永远不用？** 数组实现的双向链表需要一个类似 `nullptr` 的哨兵值来表示“链表到头了”。DeepInfer 直接把下标 `0` 当作空指针哨兵（L27 注释：“gpu blocks 0 是不用的，所以这里 0 代表空”），真正可分配的物理块编号是 `1 .. num_blocks - 1`。同时，在 `clone_blocks` 传递 `(src_block, dst_block)` 时，`dst_block = 0` 也天然代表“无需克隆”。
2. **三态分配检查 `can_allocate`（L127–L135）**：
   - 若请求所需的块数 `>= num_blocks`（超过总容量上限），返回 `AllocStatus::NEVER`，调度器直接将该请求标为 `FINISHED_IGNORED`；
   - 若所需块数 `<= get_num_free_blocks()`，返回 `AllocStatus::OK`；
   - 否则返回 `AllocStatus::LATER`，调度器暂停本轮 prefill 扩批，等待在途请求释放块。
3. **为什么单线程调度器还要加互斥锁？** 在 `scheduler.hpp` L189–L206 中，每次调度都会用 `BlockManagerMutexAcquire` 按**先锁 `state_manager`、再锁 `block_manager`** 的固定顺序同时锁住两本账。原因在于 `_schedule` 一进门就调用了 `py::gil_scoped_release` 释放了 Python GIL（`scheduler.hpp` L276）！此时后台的结构化输出（SO）编译线程或多模态预处理线程可能正在并发调用 `free` 或 `fork`，固定加锁顺序既保证了线程安全，又彻底消除了死锁风险。

---

## 4.5 本章小结

- **两套显存账本各司其职**：随长度增长的压缩 KV（`ca_cache`）与索引 KV（`ica_cache`）走 128-token 粒度的分页块池；固定窗口的 SWA KV（`swa_cache`）与压缩边界状态走每请求一槽的定长 State 池。
- **两个环形缓冲区长度的精确含义**：压缩状态 Ring 取 $\text{ratio} + 5$（满足单步前向与投机回退下界 $W + k - 1$），而主模型 SWA Ring 取 $128 + 6 = 134$（额外兼容跨 128 边界落盘快照、预填充历史窗口加载与 YoCo 尾部保留）。
- **轻量级 C++ 块管理**：用块 0 作哨兵、用数组双向链表实现 $O(1)$ 分配与释放、用引用计数支持 `fork_seq` 零拷贝共享前缀。

理解了显存是怎么管账的，下一章我们来看 DeepInfer 是怎么用 **零开销异步调度（ZOS）** 把 CPU 的调度开销藏进 GPU 的计算阴影里的。
