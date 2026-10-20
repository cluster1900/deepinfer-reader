# 第 12 章　调度器与抢占

本章实现 `minidi/scheduler.py`。它是 DeepInfer `scheduler.hpp` 单 rank、NORMAL 引擎行为的缩小版。多 rank 投票单独放在 `vote.py`（第 15 章）。

## 12.1 一步只做一种事

```python
def schedule(self):
    out = None
    if self.decode_step % self.prefill_attempt_interval == 0:
        want = self.bm.usage() <= PREFILL_WATERMARK and bool(self.waiting)
        if want:
            out = self._schedule_prefill()
    if out is None:
        out = self._schedule_decode()
    ...
```

和 DeepInfer 一样，prefill 和 decode 从不在同一步：每隔 `prefill_attempt_interval` 个 decode 步，并且 KV 使用率不超过 0.9 时，才尝试 prefill；排不出任何 prefill 就退回 decode。`test_prompt_and_decode_never_mixed` 检查每一步的输出要么全是 prefill 要么全是 decode。

## 12.2 prefill：严格 FCFS

`_schedule_prefill` 从 `waiting` 队头一个个看：

| 情况 | 处理 | 理由 |
|---|---|---|
| prompt 超过 `prompt_limit` | 标 IGNORED，出队，继续看下一个 | 永远做不了，别挡路 |
| `can_allocate` = NEVER | 标 IGNORED，出队，继续 | 同上 |
| `can_allocate` = LATER | **整个循环停下** | 不跳过队头，保证先来先服务 |
| 并发数已满 | 停下 | `max_num_seqs` |
| token 预算不够 | 切 chunk，排上这一段，然后停下 | NORMAL 引擎切了 chunk 就收手 |

LATER 时停下而不是跳过，是 DeepInfer 的选择（`scheduler.hpp` L386–L388：`_can_allocate_with_state` 返回 LATER 就 `break`）。跳过能提高利用率，但大请求可能被小请求一直插队、永远饿死。`test_too_long_prompt_ignored`、`test_no_chunk_when_disabled`、`test_chunked_prefill_stops_after_first_chunk_and_requeues_front` 分别覆盖这几条。

首个 chunk 就分配整段 prompt 所需的全部块。这样后续 chunk 不会因为没块而卡在半路，代价是占用提前发生（第 11 章的容量闸门就是为此）。

## 12.3 decode：从短的排起，从长的抢起

```python
running = sorted(self.running, key=lambda s: s.output_len(), reverse=True)
l, r = 0, len(running)
while l < r and len(out.scheduled) < self.max_num_seqs:
    seq = running[r - 1]                          # 从尾部（输出最短）开始排
    need = seq.get_num_blocks() - len(seq.block_table)
    while need > self.bm.num_free() and r - l > 1:
        self._preempt(running[l], out)            # 从头部（输出最长）开始抢
        l += 1
    ...
```

这与第 3 章读到的 DeepInfer 实际行为一致：按 `output_len` 降序排序后，从尾部排入，从头部抢占。抢占输出最长的请求，重算代价最大，但它最可能快结束；DeepInfer 头部注释说的是“抢占 output 最短”，和代码相反（第 3 章）。minidi 跟代码走。`test_lcfs_preempts_longest_output_first` 构造了三个请求恰好占满所有块的场景，断言被抢的是输出最长的那个。

被抢占后有两种结局，由 `stop_preempted_request` 决定：直接以 `FINISHED_PREEMPTED`（“insufficient_system_resource”）结束，或者 `reset_for_recompute` 后回到 `waiting` 重新 prefill。重算保留已生成的 token，只是 KV 作废，所以重新 prefill 的是“prompt + 已生成部分”。`test_stop_preempted_request` 覆盖前者。

## 12.4 落账：process_outputs

```python
seq.num_cached += q                           # 乐观推进
if seq.num_cached >= seq.get_len():           # 这段算完了
    if out.prompt_run:
        self.running.append(seq)              # prefill 完成才进 running
    for t in toks: ...append, _check_stop...
    seq.num_cached = seq.get_len() - 1        # 回填
else:
    self.waiting.appendleft(seq)              # chunk 没跑完，放回队头
```

两个细节：

- **prefill 完成的请求在这里才进 running**，和 DeepInfer 一样。调度时它还在 `waiting` 之外的“悬空”状态，这决定了异步调度为什么必须当场处理 prompt 步（第 13 章）。
- **回填**。最后一个新 token 还没有进过模型，它的 KV 下一步才写，所以已缓存长度是 `长度 - 1`。投机解码时被拒绝的 draft 写过 KV，也靠这一行作废（第 14 章）。

## 12.5 坑：队头互等

这是开发 minidi 时真实遇到的死锁。最初 `_preempt` 把重算请求直接 `appendleft` 放到队头。某组测试里：

1. 请求 A 的 prefill 做了一半（已分到全部块，剩下的 chunk 在 `waiting` 队头）；
2. decode 时块不够，抢占了请求 B，B 被放到 `waiting` 最前面，排在 A 前面；
3. 下次 prefill，B 在队头，需要的块被 A 占着，`can_allocate` 返回 LATER，整个循环停下；
4. A 排在 B 后面，永远轮不到；A 不做完就不释放块。

两边互等，引擎卡住。修法是把重算请求插到“已经分到块的请求”之后：

```python
i = 0
while i < len(self.waiting) and self.waiting[i].block_table:
    i += 1
self.waiting.insert(i, seq)
```

**这是 minidi 的问题和修法，本书没有在 DeepInfer 上复现同样的场景**，也不断言 DeepInfer 是否存在同类问题。DeepInfer 的 `_preempt` 同样把重算请求 `waiting.push_front`（`scheduler.hpp` L784），它是否有其他机制避免这种情况，本书没有追到。

## 12.6 实跑

![图 12-1　minidi 实跑轨迹：每步请求数与 KV 使用率](images/plots/ch12_trace.png)

图 12-1 是 `make_plots.py` 实跑的结果：10 个请求、19 个可用块、每步 token 预算 16。橙色是 prefill 步，蓝色是 decode 步，黑线是块使用率，红色虚线是 0.9 水位。可以看到：

- 使用率接近或超过 0.9 时不再插入 prefill；
- 抢占发生后使用率骤降，随后又有 prefill 步把被抢的请求重新做一遍；
- prefill 步和 decode 步严格交替出现，从不混在一起。
