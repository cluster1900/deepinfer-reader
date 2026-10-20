# 第 11 章　手写分页 KV 与块管理器（`PagedKV` & `BlockManager`）

在第 10 章的 `paged_forward` 里，我们用到了两个核心调用：`kv.write(block_table, start, k_new, v_new)` 和 `kv.read(block_table, end)`。

这一章我们就打开 `minidi/model.py`（L49–L67）和 `minidi/blocks.py`（L1–L61），亲手实现分页 KV 显存池和块管理器，并复盘我们在开发 `minidi` 时踩到的第一个真实大坑——**为什么没有“80% 容量闸门”，分块 Prefill 就会把整个引擎死锁？**

---

## 11.1 逻辑位置到物理槽位的两步映射（`PagedKV`）

想象你手里有一块形状为 `[num_blocks, block_size, dim]` 的大数组，代表整张 GPU 上的物理显存池。为了在很小的测试例子里就能频繁跨越块边界，我们在 `minidi` 中默认取 `block_size = 4`（DeepInfer 生产中固定为 `128`）。

当某条请求的块表是 `block_table = [1, 2, 4]` 时，它的第 $p$ 个逻辑位置（例如 $p = 0, 1, \dots, 9$）到底存在哪个物理块的哪个槽位里？只需要最简单的整除和取模：

$$\text{物理块号 } b = \text{block\_table}\bigl[p \mathbin{//} \text{block\_size}\bigr], \qquad \text{块内槽位 } s = p \bmod \text{block\_size}$$

我们用 `scripts/figures/make_plots.py` 驱动 `minidi.blocks.BlockManager`，让请求 A（10 个 token）和请求 B（5 个 token）交错增长，画出的真实物理池分布如图 11-1 所示：

![图 11-1　两条请求交错增长时的物理块池与块表映射（block_size=4，块 0 永久保留为空指针哨兵）](images/plots/ch11_block_table.png)

看图 11-1：
- 请求 A 先做 8 个 token 的 prefill，分到了物理块 `[1, 2]`；接着请求 B 做 4 个 token 的 prefill，分到了物理块 `[3]`；
- 随后在 decode 阶段，请求 A 涨到了 10 个 token，跨过了第 8 个位置的边界，于是追加分到了物理块 `4`（块表变成 `[1, 2, 4]`），其中块 4 只用了前 2 个槽位，空着后 2 个槽位；
- 请求 B 涨到 5 个 token 时，追加分到了物理块 `5`（块表变成 `[3, 5]`）。

把这套映射写成 NumPy 向量化代码，`PagedKV`（`minidi/model.py` L49–L67）一共只有 18 行：

```python
class PagedKV:
    """分页 KV 池：物理块 0 保留不用（与 DeepInfer block_manager.hpp L27 一致）。"""
    def __init__(self, num_blocks: int, block_size: int, dim: int):
        self.block_size = block_size
        self.k = np.zeros((num_blocks, block_size, dim))
        self.v = np.zeros((num_blocks, block_size, dim))

    def _ phys_indices(self, block_table: list[int], positions: np.ndarray):
        bt = np.asarray(block_table, dtype=int)
        phys_blocks = bt[positions // self.block_size]
        slots = positions % self.block_size
        assert (phys_blocks > 0).all(), "致命错误：试图读写保留的哨兵物理块 0！"
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

注意看 `assert (phys_blocks > 0).all()` 这一行！我们和 DeepInfer 一样把**物理块 0 保留不用**。这不仅忠实复刻了 C++ 空链表哨兵的设计，还给我们送了一个极其好用的调试探针：只要调度器少分配了块、或者谁不小心把未初始化的 0 号块塞进了块表，`PagedKV.write` 就会在第一时间抛出断言错误，绝不会静默写坏其他请求的数据。

---

## 11.2 60 行实现 `BlockManager`：FIFO 空闲链表与引用计数

接下来看 `minidi/blocks.py`。它完整对应第 4 章读过的 `csrc/engine/block_manager.hpp`：

1. **FIFO 空闲队列（`OrderedDict` 模拟双向链表）**：
   初始化时，块 `1 .. num_blocks - 1` 依次进入空闲队列 `self.free`。分配块（`allocate`）时从队头弹出（`popitem(last=False)`），释放块（`free_blocks`）且引用计数归零时挂回队尾。为什么要用先进先出（FIFO）而不是栈（LIFO）？这和 C++ 的 `_pop_front` / `_push_back` 完全一致，而且在测试中能让刚释放的块不会立刻被下一行代码重用，更容易暴露“释放后误用（Use-After-Free）”的 Bug。
2. **引用计数（`ref_count`）支持零拷贝共享**：
   调用 `pin(blocks)` 可将已分配块的引用计数加 1（对应 `fork_seq` 共享前缀）；调用 `free_blocks(blocks)` 时每次将引用计数减 1，只有当 `ref_count == 0` 时才真正把物理块归还给空闲队列。
3. **三态分配检查 `can_allocate(n)`**：
   ```python
   def can_allocate(self, n: int) -> AllocStatus:
       if n >= self.num_blocks:           # 注意是 >=，因为块 0 保留，最多只有 num_blocks - 1 个可用块！
           return AllocStatus.NEVER
       return AllocStatus.OK if n <= len(self.free) else AllocStatus.LATER
   ```

---

## 11.3 单条请求到底要申请几个块？（`Sequence.get_num_blocks`）

在 `minidi/sequence.py`（L69–L71）中，计算一条请求当前需要持有多少个物理块的公式只有两行，但每一个加项都价值千金：

```python
def get_num_blocks(self) -> int:
    n = self.get_len() + self.inflight + len(self.draft_tokens)
    return (n + self.block_size - 1) // self.block_size
```

这行公式与 DeepInfer 的 `Sequence::get_num_blocks()`（`csrc/engine/sequence.hpp` L292）完全一致。为什么要把这三项加在一起向上取整？

- `self.get_len()`：当前序列里已经确认的所有 token（包括末尾那个刚采出来、本步要作为主输入送进去写 KV 的 token）；
- `self.inflight`：在异步调度（ZOS，第 13 章）下，上一步还在 GPU 上飞、尚未落账的在途 token 数；
- `len(self.draft_tokens)`：在投机解码（第 14 章）下，本步要紧跟着主输入一起送进模型验证并写入 KV 的候选草稿 token 数。

只要漏加后面两项中的任何一项，当序列长度刚好卡在 `block_size` 的整数倍边界上时，`paged_forward` 里的 `kv.write` 就会直接越过 `block_table` 的末尾报错！

---

## 11.4 真实踩坑录：为什么没有“80% 容量闸门”就会死锁？

在第 2 章读 `LLMEngine._init_cache`（`llm_engine.py` L445–L451）时，我们看到过一段只有几行的检查：如果 `runtime_max_model_len > block_size * num_gpu_blocks * 0.8`，引擎就拒绝启动。当时你可能会想：“只要单条请求不超过 100% 的可用块数不就能跑吗？为什么要硬卡在 **80%（0.8）**？”

在最初编写 `minidi` 时，我们也没有写这道 80% 闸门。结果在跑随机压力测试时，引擎在某组参数下**彻底卡死（无限循环）**了！我们把当时的现场还原出来，你一下子就能看懂 `0.8` 容量闸门和 `0.9` 水位线之间精妙的配合：

```{.mermaid caption="图 11-2　缺少 80% 容量闸门时，超长分块 Prefill 跨越 0.9 水位线引发的死锁现场"}
flowchart TB
  subgraph Trap["死锁复盘（假设总块数 num_blocks=20，每块 4 tokens，单次预算 16 tokens）"]
    direction LR
    T1["1. 来了一条长 74 tokens 的请求<br/>需要 19 个块（占总块数 19/20 = 95%）"] --> T2["2. 第 1 个 chunk 调度成功（先算前 16 个 token）<br/>一次性拿走全部 19 个物理块！<br/>跑完后还剩 58 个 token，退回 waiting 队头"]
    T2 --> T3["3. 下一步准备继续跑它的第 2 个 chunk：<br/>一查 KV 使用率 = 95% > 0.9 水位线！<br/>调度器拒绝执行 _schedule_prefill，切去跑空 Decode！"]
  end

  Dead["结果：waiting 队头的半截请求占着 95% 的块等 Prefill，<br/>而 Prefill 开关因为使用率 > 90% 永远打不开，系统永久死锁！"]
  Trap --> Dead
```

看图 11-2！因为 DeepInfer 在跑分块 Prefill 的**第一个 chunk** 时，就会一次性把整条 Prompt 所需的全部物理块分配到手（`scheduler.hpp` L518–L520）；如果单条请求一口气吃掉了 95% 的块，那么第一个 chunk 刚跑完，KV 使用率就已经突破了 `PREFILL_WATERMARK = 0.9`（90% 水位线）！到了下一步，调度器一看“KV 使用率 > 0.9，禁止做 Prefill”，于是再也不去调 `_schedule_prefill()`，而那个占着 95% 显存块的半截请求还躺在 `waiting` 队头苦苦等着跑第二个 chunk，永远进不了 `running`，也永远释放不了块！

当我们把 DeepInfer 的 80% 容量闸门（`minidi/engine.py` L43–L46）补上之后：

```python
max_seq_len = int(block_size * num_blocks * 0.8)
if self.scheduler.max_model_len > max_seq_len:
    raise ValueError(f"max_model_len={self.scheduler.max_model_len} > 0.8*容量={max_seq_len}")
```

单条请求最多只能占总块数的 80%，距离 90% 的 Prefill 熔断水位线永远留有至少 10% 的安全余量，保证了任何获准开始第一个 chunk 的请求都一定能把后续的 chunk 跑完！这就是亲手写一遍代码才能体会到的设计深意。
