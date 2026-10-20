# 第 13 章　异步调度：先调度，后知道结果

第 5 章讲了 DeepInfer 的 ZOS。本章在 CPU 上把它真的实现一遍：`minidi/zos.py` 的 `ZOSEngine`。CPU 上没有“GPU 在后台跑”，但我们可以严格模拟信息流：**调度第 N+1 步时，不允许看第 N 步的采样结果**。只要这个约束成立，DeepInfer 要处理的所有麻烦就都会出现。

## 13.1 step 的四个阶段

```python
def step(self):
    out = self.scheduler.schedule()              # 此时上一步的 token 还没回来
    prev_tokens = self.prev[1] if self.prev else {}
    for seq, q in zip(out.scheduled, out.seqlens_q):
        if out.prompt_run: ...
        elif seq.seq_id in prev_tokens and seq.inflight > 0:
            # remap 命中：输入 token 用上一步的采样结果，位置往后错一格
            batch.append(([prev_tokens[seq.seq_id]], seq.num_cached + seq.inflight, seq.block_table))
        else: ...
    logits = self.model.paged_forward(self.kv, batch)     # 1. forward 本步
    prev, self.prev = self.prev, None
    self._post_process(prev)                              # 2. 处理上一步
    sampled = {...}                                       # 3. sample 本步
    if out.prompt_run:                                    # 4. prompt 步当场处理
        self.scheduler.process_outputs(out, toks)
    elif out.scheduled:
        for s in out.scheduled: s.inflight += 1
        self.prev = (out, sampled)                        #    decode 步留到下一步
```

和 `Worker.execute_model_step` 的四个阶段一一对应（第 5 章图 5-2）。`self.prev` 就是“还在 GPU 上”的上一步。

## 13.2 三个必须处理的问题

**1. 长度未知 → 在途计数。** decode 步结束时 `inflight += 1`，表示这个请求有 1 个 token 还没落账。`get_num_blocks` 把 `inflight` 算进去（第 11 章），所以调度器会提前为它预留块；本步的写入位置是 `num_cached + inflight`。

**2. 输入未知 → remap。** 本步与上一步都有的请求，输入 token 直接取 `prev_tokens[seq_id]`，即上一步的采样结果。在 DeepInfer 里这一步发生在 GPU 上，CPU 只提供 `remap` 下标（`zos.cpp`）；minidi 里是一个字典查找，但关键是**调度器本身从没读过这个值**。

**3. 结束未知 → 作废。** 第 N 步采到 EOS 的请求，在调度第 N+1 步时还没被标记结束，会被多排一步。`_post_process` 把它标成结束以后，第 N+1 步的结果在 `process_outputs` 里因 `seq.is_finished()` 被跳过。`ZOSEngine.wasted` 统计这类白算的 token。这就是第 9 章 quickstart 里 ZOS 比同步多 2 步的原因。

## 13.3 坑：在途时被抢占

第一版 `_post_process` 无条件 `s.inflight -= 1`。测试在一组小块数配置下报了负数的在途计数。过程是：

1. 第 N 步，请求 A 在途（`inflight = 1`）；
2. 调度第 N+1 步时块不够，A 被抢占，`reset_for_recompute` 把 `inflight` 清零，A 回到 `waiting`；
3. 第 N 步的结果回来，`_post_process` 又把 A 的 `inflight` 减 1，变成 -1。

```{.mermaid caption="图 13-1　在途抢占：第 N 步结果回来时，请求已经被抢占"}
sequenceDiagram
  participant S as 调度器
  participant A as 请求 A
  participant G as “GPU”（self.prev）
  S->>G: decode 第 N 步（A 在批里）
  S->>A: inflight = 1
  S->>S: 调度第 N+1 步：块不够
  S->>A: 抢占：reset_for_recompute，inflight = 0，回 waiting
  G-->>S: 第 N 步结果回来（_post_process）
  Note over S,A: 若无条件 inflight -= 1 → -1<br/>若照常落账 → 重复入队、长度错乱
  S->>S: 正确做法：A 是 WAITING，整步结果作废
```

修法分两处，都与 DeepInfer 对应：

```python
# zos.py _post_process
if s.inflight > 0:            # 在途时被抢占的请求，_preempt 已经把计数清零
    s.inflight -= 1

# scheduler.py process_outputs
if not out.prompt_run and seq.status == Status.WAITING:
    continue                  # 本步在途时被抢占：输出作废
```

第二处正是 DeepInfer `process_outputs` 里“decode 步、状态却是 WAITING”就作废的分支（`scheduler.hpp` L917–L925）。读源码时这个分支看起来很奇怪，自己踩一遍就明白了。

## 13.4 收尾

所有请求都结束后，`self.prev` 里可能还有最后一步没处理。`generate` 末尾调一次 `_post_process(self.prev)`，对应 DeepInfer 的 `flush_inflight_reqs`。

## 13.5 测试

```python
@pytest.mark.parametrize("seed", range(20))
def test_zos_matches_sync(seed):
    ...
    z = ZOSEngine(M, **kw)
    assert z.generate(ps, max_tokens=10) == Engine(M, **kw).generate(ps, max_tokens=10)
    assert z.bm.num_free() == nb - 1
```

20 组随机配置（块数 14–39，随机 EOS，会触发抢占），ZOS 与同步版逐 token 相同，结束后所有块都还回来。第二个断言抓的是块泄漏：在途抢占处理不对时，常见症状不是输出错，而是块没还。

`test_timeline_zos_hides_cpu` 检查第 5 章图 5-1 的时间线模型：同步每步 12 个单位，ZOS 稳态每步 7 个单位（GPU 时间）。

教学版 ZOS 不与投机解码组合（构造函数里断言 `nextn == 0`）。两者组合时，上一步接受了几个 draft 也是未知的，DeepInfer 要用 `compute_seqlens_k_zos` 在 GPU 上修正 `seqlens_k`，还要用两轮接受率（第 6 章），复杂度高很多。
