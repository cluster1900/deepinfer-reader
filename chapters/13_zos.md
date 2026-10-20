# 第 13 章　手写零开销异步调度（`ZOSEngine`）：超前盲调与在途抢占

在第 5 章里，我们读了 DeepInfer `Worker.execute_model_step`（`worker.py` L89–L189）的四阶段异步流水线。当时你可能会想：“在 CPU 上又没有真正的 GPU 异步流，怎么复现和验证 ZOS 的逻辑？”

答案非常简单：异步调度之所以复杂，本质原因不是硬件并发，而是**信息延迟**——即 **CPU 在执行 `schedule(N+1)` 的那一刻，绝对不允许偷看第 $N$ 步的采样结果**！

只要我们在纯 CPU 的 `ZOSEngine`（`minidi/zos.py`）里严格遵守这一条信息隔离铁律——把第 $N$ 步的采样结果锁进一个名为 `self.prev` 的黑盒子里，规定必须在第 $N+1$ 步完成调度和前向发射之后才准打开这个黑盒子执行 `_post_process(prev)`——那么 DeepInfer 在生产中遇到的所有异步难题（块表超前预留、`remap` 接线、EOS 白算作废、在途被抢占）都会分毫不差地在我们的 CPU 代码里重现！

---

## 13.1 30 行代码复刻 `execute_model_step` 四阶段流水线

打开 `minidi/zos.py`（L18–L64），`ZOSEngine` 继承自第 10 章的同步 `Engine`，它的 `step()` 方法严格按照第 5 章的四阶段顺序编写：

```python
class ZOSEngine(Engine):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.prev = None          # 模拟“还在 GPU 流上未同步”的第 N 步状态：(SchedulerOutputs, sampled_dict)
        self.wasted = 0           # 统计因第 N 步已采到 EOS 但 CPU 尚不知情而白算的 token 数

    def step(self):
        # 阶段 0：超前调度第 N+1 步（此时第 N 步的采样结果还锁在 self.prev 里，绝不触发 process_outputs！）
        out = self.scheduler.schedule()
        prev_tokens = self.prev[1] if self.prev else {}

        # 阶段 1：组装并执行第 N+1 步前向（对应 _correct_seqlens_qk + execute_main_model）
        batch = []
        for seq, q in zip(out.scheduled, out.seqlens_q):
            if out.prompt_run:
                batch.append((seq.tokens[seq.num_cached : seq.num_cached + q], seq.num_cached, seq.block_table))
            elif seq.seq_id in prev_tokens and seq.inflight > 0:
                # 命中 remap！直接从上一步“GPU 显存” (prev_tokens) 抓输入 token，写入位置后移 inflight 格！
                batch.append(([prev_tokens[seq.seq_id]], seq.num_cached + seq.inflight, seq.block_table))
            else:
                batch.append(([seq.tokens[-1]], seq.num_cached, seq.block_table))
        logits = self.model.paged_forward(self.kv, batch) if batch else None

        # 阶段 2：等待第 N 步采样完成并落账（对应 _post_process_step(self.prev_step_state)）
        prev, self.prev = self.prev, None
        self._post_process(prev)

        # 阶段 3：执行第 N+1 步采样
        sampled, off = {}, 0
        for seq, q, (qt, start, _) in zip(out.scheduled, out.seqlens_q, batch):
            sampled[seq.seq_id] = sample(logits[off + q - 1], seq, start + q)
            off += q

        # 阶段 4：Prefill 步当场落账；Decode 步记入 self.prev 留到下一步的阶段 2 处理！
        if out.prompt_run:
            toks = [[sampled[s.seq_id]] if s.num_cached + q >= s.get_len() else []
                    for s, q in zip(out.scheduled, out.seqlens_q)]
            self.scheduler.process_outputs(out, toks)
        elif out.scheduled:
            for s in out.scheduled:
                s.inflight += 1
            self.prev = (out, sampled)
        return out
```

请仔细观察阶段 1 里 `elif seq.seq_id in prev_tokens and seq.inflight > 0:` 那一行：
- 在同步引擎里，第 $N$ 步采出的 token 已经追加到了 `seq.tokens[-1]`，且 `seq.num_cached` 已经加了 1；
- 而在 `ZOSEngine` 里，因为阶段 2 的 `self._post_process(prev)` 排在阶段 1 的 `paged_forward` **后面**，所以在阶段 1 执行前向时，`seq.tokens` 里根本还没有第 $N$ 步的采样结果，`seq.num_cached` 也还停留在第 $N-1$ 步的位置！
- 因此，凡是上一步在途的请求（`seq.inflight > 0`），都必须通过 `prev_tokens[seq.seq_id]`（即 DeepInfer 里的 GPU 端 `remap`）直接拿到输入 token，并且把 KV 写入起点设为 **`seq.num_cached + seq.inflight`**！

---

## 13.2 真实踩坑录：当“在途请求”在第 N+1 步被抢占，会发生什么？

在写完第一版 `ZOSEngine` 后，我们在 `_post_process(prev)` 里很自然地写下了这样一行代码：

```python
# 第一版有 Bug 的写法：
for s in out.scheduled:
    s.inflight -= 1
```

看起来天衣无缝对不对？上一步结束时 `s.inflight += 1`，下一步结算时 `s.inflight -= 1`，一加一减正好归零。

然而，当我们跑 `test_zos_matches_sync` 的小块池随机测试时，程序突然报错：**某条请求的 `inflight` 变成了 `-1`，紧接着在下一步写 KV 时位置倒退了 1 格，输出彻底乱套！**

我们顺着日志把犯罪现场画出来，一下就看懂了 DeepInfer `scheduler.hpp` L917–L925 那段看似古怪的代码究竟在防什么：

```{.mermaid caption="图 13-1　在途请求（In-Flight Sequence）在下一轮调度时遭遇抢占的时序冲突与修复"}
sequenceDiagram
  participant S as Scheduler / ZOSEngine
  participant A as 请求 A (Sequence)
  participant P as 在途黑盒 (self.prev)
  S->>P: 1. 第 N 步 Decode 完成前向与采样，存入 self.prev
  S->>A: 2. 标记在途：A.inflight = 1（状态仍为 RUNNING）
  Note over S,A: 3. 进入第 N+1 步：先执行 schedule(N+1)，此时空闲块不足！
  S->>A: 4. 左指针选中抢占请求 A：调用 reset_for_recompute()<br/>将 A.inflight 清零为 0，状态改回 WAITING，释放全部物理块！
  P-->>S: 5. 阶段 2 打开 self.prev，结算第 N 步输出 _post_process(N)
  Note over S,A: 致命陷阱：<br/>陷阱 1：若无条件执行 A.inflight -= 1，计数器将从 0 变成 -1！<br/>陷阱 2：若照常把第 N 步 token 写入 A，会把已退回 WAITING 的 A 当成正常完成！
  S->>A: 正确防线：<br/>1. 仅当 A.inflight > 0 时才减 1；<br/>2. process_outputs 发现 decode 步但 A.status == WAITING，直接丢弃第 N 步输出！
```

看图 13-1 的第 4 步和第 5 步：
1. 在第 $N$ 步结束时，请求 A 处于在途状态（`A.inflight = 1`）。
2. 紧接着第 $N+1$ 步一上来先跑 `schedule(N+1)`。很不巧，这一步物理块不够用了，双指针抢占算法正好把请求 A 给抢占了！`_preempt(A)` 调用了 `A.reset_for_recompute()`，当场把 `A.inflight` 重置成了 `0`，把 A 的物理块全退了，状态改回 `Status.WAITING` 塞回了等待队列。
3. 随后到了第 $N+1$ 步的阶段 2，`_post_process(prev)` 拿着第 $N$ 步的旧名单 `out.scheduled`（里面还有请求 A！）来结账了：
   - 如果不检查 `if s.inflight > 0:` 就闭着眼睛减 1，`A.inflight` 就会从 `0` 变成 `-1`！
   - 如果 `process_outputs` 不检查 `if not out.prompt_run and seq.status == Status.WAITING: continue`，就会把第 $N$ 步算出来的那个已经作废的 token 硬塞进正在 `waiting` 里等重算的请求 A 肚子里，还会把 `A.num_cached` 改成 `len - 1`（而此时 A 的块表其实是空的！），下一步当场崩溃！

把这两道护栏（`minidi/zos.py` L30–L31 与 `minidi/scheduler.py` L177–L178）加上之后，再回头看第 5 章读过的 `csrc/engine/scheduler.hpp` L917–L925：

```cpp
if (!prompt_run && seq.status == SequenceStatus::WAITING) {
    // 注释：本步 decode 在途期间该请求已被抢占退回 waiting，输出直接丢弃
    continue;
}
```

那种“原来 DeepSeek 工程师写这行 `if` 是因为踩过一模一样的坑”的共鸣感，正是自己动手写一遍引擎最大的乐趣。

---

## 13.3 收尾冲洗（`flush_inflight_reqs`）与完整验证

当所有请求都跑完、`scheduler.has_unfinished()` 变成 `False` 退出主循环时，别忘了 `self.prev` 黑盒子里还装着**最后一步**尚未落账的采样结果！因此，`ZOSEngine.generate`（`minidi/zos.py` L74）在循环退出后必须最后调一次 `self._post_process(self.prev)`——这与 DeepInfer `LLM._run_engine` 在循环末尾调用 `flush_inflight_reqs()`（`llm.py` L184）完全对应。

在 `tests/test_engine.py::test_zos_matches_sync` 中，我们用 20 组不同的随机种子、紧凑的物理块池（14~39 块，高频触发抢占）和随机 EOS 跑对比测试：
1. 断言 `ZOSEngine` 与同步 `Engine` 生成的所有序列逐 token 100% 相等；
2. 断言运行结束后 `z.bm.num_free() == num_blocks - 1`（除保留的 0 号块外，没有任何一个物理块因异步抢占而泄漏）。
