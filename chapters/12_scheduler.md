# 第 12 章　手写调度器与抢占（`Scheduler`）：双指针、块缩容与队头死锁

这一章我们打开 `minidi/scheduler.py`（共 206 行），把第 3 章读过的 DeepInfer C++ 调度器核心逻辑——**Prefill/Decode 步严格分离、FCFS 分块 Prefill、LCFS 双指针 Decode 抢占、草稿被拒后的尾块缩容回收、以及两阶段账本提交**——一行不落地用 Python 写出来，并剖析我们在测试“抢占后重算（Recompute）”时踩到的第二个真实死锁。

---

## 12.1 步型切换：为什么每一步只做一种事？

在 `Scheduler.schedule()`（`minidi/scheduler.py` L71–L84）中，我们复刻了 DeepInfer `NORMAL` 引擎的单卡步型切换规则：

```python
def schedule(self) -> SchedulerOutputs:
    out = None
    # 只有当 decode_step 落在尝试间隔的整数倍上、且 KV 使用率 <= 0.9 时，才尝试做 Prefill
    if self.decode_step % self.prefill_attempt_interval == 0:
        want = self.bm.usage() <= PREFILL_WATERMARK and bool(self.waiting)
        if want:
            out = self._schedule_prefill()
    if out is None:
        out = self._schedule_decode()
    if out.prompt_run:
        self.decode_step = 0                     # 只要跑了 Prefill，计数器归零，下一步可连续 Prefill
    else:
        self.running = list(out.scheduled)
        self.decode_step += 1
    return out
```

注意第 79–80 行：一旦成功排出了一个 Prefill 步（`out.prompt_run == True`），`self.decode_step` 就会被重置为 `0`。这意味着只要 `waiting` 队列里还有等待预填充的请求，且 KV 块使用率还没触及 90% 的红线，下一步 `0 % prefill_attempt_interval == 0` 依然成立，调度器会连续不断地把积压的 Prefill 做完，然后再切回 Decode。

---

## 12.2 `_schedule_prefill`：严格先来先服务与分块规则

在 `_schedule_prefill`（`minidi/scheduler.py` L86–L125）中，我们从 `self.waiting` 队头开始逐个检查请求。为了和 `scheduler.hpp`（L351–L554）保持 100% 行为一致，这里有五条铁律：

1. **已取消或超长请求直接摘除，不挡道**：如果队头请求已经结束，或者总长超过了 `prompt_limit`，或者 `bm.can_allocate` 返回 `AllocStatus.NEVER`（永远放不下），则将其标记为 `FINISHED_IGNORED` 并弹出放入 `out.ignored`，继续看下一个请求。
2. **遇到暂时缺块（`LATER`）立刻 `break`**：如果队头请求需要的块目前不够（返回 `AllocStatus.LATER`），循环当场终止，**绝不跳过它去挑后面能放下的小请求**，严格保证先来先服务（FCFS）。
3. **首个 chunk 一次性拿齐全部块**：当 `not seq.block_table` 时，立即调用 `seq.block_table = self.bm.allocate(seq.get_num_blocks())` 把整段 Prompt 的块一次拿齐。
4. **只有最后一个 chunk 才把状态改为 `RUNNING` 并占用并发名额**：如果本步受 `max_num_batched_tokens` 预算限制切了分块（`is_chunked = q < todo`），请求在本步跑完后还要退回 `waiting` 队头；只有在 `not is_chunked`（本步能把整个 Prompt 跑完）时，才令 `num_curr += 1` 并将状态设为 `Status.RUNNING`。
5. **`NORMAL` 引擎切了 chunk 就收手**：一旦当前请求发生了分块截断（`if is_chunked: break`），本步不再继续往批次里塞后面的请求。

---

## 12.3 `_schedule_decode`：双指针抢占与多余尾块回收

进入 `_schedule_decode`（`minidi/scheduler.py` L127–L153）后，我们先释放上一轮已结束请求的物理块，然后用 18 行代码实现第 3 章讲过的 **LCFS 排入 + 最长输出优先抢占 + 尾块动态伸缩**：

```python
running = sorted(self.running, key=lambda s: s.output_len(), reverse=True)
l, r = 0, len(running)
while l < r and len(out.scheduled) < self.max_num_seqs:
    seq = running[r - 1]                              # 右指针：从输出最短（最晚进入 decode）开始排
    need = seq.get_num_blocks() - len(seq.block_table)
    while need > self.bm.num_free() and r - l > 1:
        self._preempt(running[l], out)                # 左指针：块不够时，优先抢占输出最长的请求！
        l += 1
    if need <= self.bm.num_free():
        if need > 0:
            seq.block_table += self.bm.allocate(need) # 分支 1：扩容追加新块
        elif need < 0:                                # 分支 2：草稿被拒后回收多余尾块（对应 L821-L825）
            self.bm.free_blocks(seq.block_table[need:])
            del seq.block_table[need:]
        out.scheduled.append(seq)
    else:
        self._preempt(seq, out)
    r -= 1
```

请看代码里的两个亮点：
- **右指针保新、左指针杀长**：`tests/test_scheduler.py::test_lcfs_preempts_longest_output_first` 构造了 3 条输出长度分别为 `4, 0, 8` 的请求把物理块池刚好占满，当下一步大家都需要新块时，测试验证第一个被左指针抢占的正是输出最长（`out=8`）的 `seq2`。
- **`elif need < 0:` 精准回收草稿废块**：正如第 3.5 节剖析的，当上一步因为投机草稿跨块预留了新块、而验证时草稿被拒导致本步所需块数回落时，`need < 0` 分支会立刻把多占的尾块归还给 `BlockManager`（由 `test_append_slot_frees_redundant_blocks_after_draft_rejection` 专门验证）。

---

## 12.4 真实踩坑录：当“分块 Prefill”撞上“抢占后重算”，为什么会队头互等？

在实现了 `_preempt` 之后，我们在离线重算模式（`stop_preempted_request = False`）下跑随机压力测试，遇到了开发 `minidi` 过程中的第二个死锁。这个死锁非常经典，值得所有写推理调度器的人警惕！

### 死锁是怎么发生的？
最初，我们完全照搬 `scheduler.hpp` L784 的写法，在 `_preempt` 里把被抢占重算的请求直接塞回 `waiting` 队列的最前面（`self.waiting.appendleft(seq)`）。结果在某次测试中出现了以下四步连环套：

```{.mermaid caption="图 12-1　“半截分块请求”与“被抢占重算请求”在 waiting 队头发生的死锁过程"}
flowchart TB
  subgraph S12["第一阶段：半截 Chunk 占块 + Decode 触发抢占插队"]
    direction LR
    A1["1. 请求 A（需 10 块）跑完第 1 个 chunk<br/>已拿走 10 块，因未跑完退回 waiting[0]"] --> A2["2. 下一步 Decode 块不够，抢占请求 B！<br/>B 释放块重置后若直接插到 waiting 最前面"] --> A3["3. 此时队列变成：[B(无块,需6块), A(已握10块)]<br/>而空闲池只有 4 个块"]
  end

  subgraph S34["第二阶段：队头互等死锁 vs minidi 修复方案"]
    direction LR
    B1["死锁：_schedule_prefill 查队头 B 缺块返回 LATER 即 break；<br/>排在后面的 A 握着 10 块却永远轮不到跑完释放！"] --> B2["修复（scheduler.py L165）：重算请求排在已持块的半截请求后面，<br/>优先让持块者 A 跑完释放，彻底消除死锁！"]
  end

  S12 --> S34
```

你看图 12-1：
- **请求 B** 排在 `waiting[0]`，它手里没有块，向块管理器要 6 个块，但池子里只有 4 个空闲块，于是 `can_allocate` 返回 `LATER`，严格 FCFS 规则要求循环立刻 `break`；
- **请求 A** 排在 `waiting[1]`，它手里明明已经握着 10 个物理块，只要让它把剩下的半截 chunk 跑完进入 Decode 直至结束，那 10 个块就能释放出来；可因为它被没有块的请求 B 挡在了身后，一辈子也排不上号！

### 为什么 DeepInfer 线上不会踩这个坑，而 `minidi` 怎么修？
在 DeepInfer 的生产环境里，这个死锁天然不会发生，原因有二：
1. 线上服务采用 **PD 分离架构**（`PROMPT` 引擎只做 prefill，`COMPLETION` 引擎只做 decode），两边压根不在同一个实例里抢块；
2. 即使在单实例服务下，`EngineArgs.stop_preempted_request` 默认也是 **`True`**（抢占即终止，不塞回 `waiting` 重算）。

而在支持离线重算（`stop_preempted_request = False`）的 `minidi` 中，我们的修复方法既简单又彻底（`minidi/scheduler.py` L165–L171）：当把被抢占重算的请求塞回 `waiting` 队列时，**让它排在所有“已经分到了物理块、正做到一半的半截 chunk 请求”后面**：

```python
# minidi/scheduler.py L165-L171 (_preempt 重算分支)
i = 0
while i < len(self.waiting) and self.waiting[i].block_table:
    i += 1
self.waiting.insert(i, seq)   # 排在已持有块的半截 chunk 后面，优先让持块者跑完释放！
```

只要让已经握着物理块的半截请求 A 保持在队头优先跑完，死锁就烟消云散了。

---

## 12.5 实跑轨迹：亲眼看见 0.9 水位线与抢占重算

现在我们运行 `scripts/figures/make_plots.py` 中的 `trace()`，把 10 条长短不一的请求丢进只有 19 个可用物理块的 `minidi.Engine` 里，画出每一步的批大小、步型（橙色 = Prefill 步，蓝色 = Decode 步）以及 KV 块使用率随时间变化的真实轨迹：

![图 12-2　minidi 实跑 10 条请求的全程轨迹：KV 使用率触及 0.9 红线时暂停 Prefill，满池触发 4 次抢占后重新 Prefill 直至全部完成](images/plots/ch12_trace.png)

看着图 12-2，我们在第 3 章和本章讲过的所有规则都活了起来：
1. 开局阶段（Step 0–6）：KV 使用率低于 0.9 红线，引擎连续跑橙色的 Prefill 步，迅速把第一批请求送入 `running`，使用率一路攀升至 0.9 以上；
2. 中途阶段（Step 7–25）：因为使用率高于或接近 0.9，引擎专心跑蓝色的 Decode 步；当多条请求同时增长撑满 19 个块时，触发了 **4 次抢占**（黑线使用率陡然下挫），随后一旦使用率回落到 0.9 以下，橙色的 Prefill 步立刻插进来把等待中的新请求和被抢占的请求重新做完！
3. 全程 50 多步里，橙色柱（Prefill）和蓝色柱（Decode）泾渭分明，且最终 10 条请求的输出与 `reference_generate` 100% 逐 token 一致。
