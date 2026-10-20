# 第 11 章　分页 KV 与块管理

## 11.1 物理池与块表

`PagedKV` 是一块 `[num_blocks, block_size, dim]` 的物理池（K、V 各一份）。请求的第 p 个位置存在 `block_table[p // block_size]` 号块的 `p % block_size` 槽。请求之间不需要连续空间，碎片只出现在每个请求最后一个块里。

![图 11-1　两个请求交替增长时的块表（minidi BlockManager 实际分配；块 0 保留）](images/plots/ch11_block_table.png)

图 11-1 里 A 先拿到块 1、2，B 拿到块 3，A 涨到 10 个 token 时才拿到块 4。逻辑上连续的位置，物理上可以分散在任何地方，只要块表记住顺序。

DeepInfer 的块大小固定 128（第 4 章），minidi 默认 4，这样很小的例子也能用到很多块，容易暴露边界错误。

**块 0 保留**。DeepInfer 的 `Blocks` 约定“0 代表空”，`PagedKV.write` 里断言 `(b > 0).all()`：一旦块表不够长、索引落到填充值 0 上，测试会立刻报“写到了保留块 0”，而不是悄悄写坏别人的数据。

`test_paged_forward_matches_reference_under_any_chunking` 对块大小 1、3、4、16 各做一次：块表用打乱顺序的物理块，把 37 个 token 随机切成 1–8 个一段依次 `paged_forward`，断言每个位置的 logits 与 `reference_logits` 的误差在 1e-10 以内。

## 11.2 块管理器

`BlockManager`（`minidi/blocks.py`）对照 `block_manager.hpp`：

- **空闲链表**：用 `OrderedDict` 模拟 C++ 的双向链表。`allocate` 从表头取（`pop_front`），引用计数归零的块挂回表尾（`push_back`），所以分配顺序是 FIFO。`test_block0_reserved_and_fifo_order` 固定了这个顺序。
- **引用计数**：`pin` 给已分配的块加引用，`free_blocks` 减引用，归零才回到空闲链表。DeepInfer 用它在 `fork_seq` 时共享前缀块。`test_pin_keeps_block_until_last_free` 验证共享块要等最后一个持有者释放。
- **三态判断**：`can_allocate(n)` 返回 OK / LATER / NEVER。要的块数大于等于总块数时永远放不下，返回 NEVER；否则看当前空闲块够不够。调度器对 LATER 和 NEVER 的处理完全不同（第 12 章）。

## 11.3 一个请求要几个块

`Sequence.get_num_blocks`：

```python
n = self.get_len() + self.inflight + len(self.draft_tokens)
return (n + self.block_size - 1) // self.block_size
```

三项分别是：已确定的 token、异步调度下还在途的 token（第 13 章）、本步要验证的 draft（第 14 章）。和 DeepInfer `Sequence::get_num_blocks`（`sequence.hpp` L292）一致。少算任何一项，都会在 forward 时写出块表范围，被块 0 断言抓住。

## 11.4 容量闸门：一个真实踩过的坑

最初的 minidi 没有容量检查。某组测试里，一个很长的 prompt 走 chunked prefill，第一个 chunk 就按整段 prompt 分配了全部块（DeepInfer 也是首个 chunk 一次分配全部块，`scheduler.hpp` L518–L520），使用率一下超过 0.9 的 prefill 水位线。之后调度器再也不尝试 prefill，这个请求剩下的 chunk 永远排不上，引擎卡死。

DeepInfer 在 `_init_cache` 里有一道闸门（`llm_engine.py` L445–L451）：`max_model_len` 不能超过 KV 总容量的 80%。minidi 照搬：

```python
max_seq_len = int(block_size * num_blocks * 0.8)
if self.scheduler.max_model_len > max_seq_len:
    raise ValueError(...)
```

`test_kv_capacity_guard` 验证配置过小时直接报错。这个例子说明，DeepInfer 里一些看起来随意的常数（0.8、0.9）是彼此配合的：单请求最多占 80%，prefill 在 90% 以下才排，中间留出的余量保证了长请求总能做完。
