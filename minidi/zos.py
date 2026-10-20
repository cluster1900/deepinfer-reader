"""异步调度（Zero-Overhead Scheduling, ZOS）。

对照 DeepInfer：
- deepinfer/worker/worker.py L89-L189 execute_model_step：1) forward 本步 2) post-process 上一步 3) sample 本步
  4) prompt 步当场 post-process，ZOS 的 decode 步留到下一步；
- worker.py L191 _correct_seqlens_qk 与 csrc/engine/zos.cpp prepare_zos_inputs：本步与上一步重叠的请求，
  输入 token 取自"上一步在 GPU 上采样出的结果"（remap[i] = 上一步下标），CPU 此刻还不知道它是什么；
- sequence.hpp ScheduleState.start_inflight_step / finish_inflight_step：在途 token 计数，
  让 get_num_blocks 提前为还没回来的 token 预留块。
ZOSEngine 在 CPU 上真实执行这套"先调度、后知道结果"的流程，测试保证输出与同步版逐 token 相同。
simulate_timeline 用来画 CPU/GPU 时间线（单位：抽象时间，不是实测）。
"""
from __future__ import annotations

from .engine import Engine, sample


class ZOSEngine(Engine):
    def __init__(self, *a, **kw):
        assert kw.get("nextn", 0) == 0, "教学版 ZOS 不与投机解码组合"
        super().__init__(*a, **kw)
        self.prev = None          # (SchedulerOutputs, {seq_id: 采样结果})，相当于"还在 GPU 上"的上一步
        self.wasted = 0           # 因为不知道上一步已结束而白算的 token 数

    def _post_process(self, prev):
        if prev is None:
            return
        out, sampled = prev
        for s in out.scheduled:
            if s.inflight > 0:        # 在途时被抢占的请求，_preempt 已经把计数清零
                s.inflight -= 1
        toks = [[sampled[s.seq_id]] for s in out.scheduled]
        self.wasted += sum(1 for s in out.scheduled if s.is_finished())
        self.scheduler.process_outputs(out, toks)

    def step(self):
        out = self.scheduler.schedule()              # 此时上一步的 token 还没回来
        self.steps.append((out.prompt_run, len(out.scheduled), out.num_batched_tokens))
        prev_tokens = self.prev[1] if self.prev else {}
        batch = []
        for seq, q in zip(out.scheduled, out.seqlens_q):
            if out.prompt_run:
                batch.append((seq.tokens[seq.num_cached: seq.num_cached + q], seq.num_cached, seq.block_table))
            elif seq.seq_id in prev_tokens and seq.inflight > 0:
                # remap 命中：输入 token 用上一步 GPU 上的采样结果，位置往后错一格
                batch.append(([prev_tokens[seq.seq_id]], seq.num_cached + seq.inflight, seq.block_table))
            else:
                batch.append(([seq.tokens[-1]], seq.num_cached, seq.block_table))
        logits = self.model.paged_forward(self.kv, batch) if batch else None   # 1. forward 本步
        prev, self.prev = self.prev, None
        self._post_process(prev)                                                 # 2. 处理上一步
        sampled, off = {}, 0
        for seq, q, (qt, start, _) in zip(out.scheduled, out.seqlens_q, batch):  # 3. sample 本步
            sampled[seq.seq_id] = sample(logits[off + q - 1], seq, start + q)
            off += q
        if out.prompt_run:                                                       # 4. prompt 步当场处理
            toks = [[sampled[s.seq_id]] if s.num_cached + q >= s.get_len() else []
                    for s, q in zip(out.scheduled, out.seqlens_q)]
            self.scheduler.process_outputs(out, toks)
        elif out.scheduled:
            for s in out.scheduled:
                s.inflight += 1
            self.prev = (out, sampled)                                           #    decode 步留到下一步
        return out

    def generate(self, prompts, max_steps=10_000, **kw):
        seqs = [self.add_request(p, **kw) for p in prompts]
        for _ in range(max_steps):
            if not self.scheduler.has_unfinished():
                break
            self.step()
        else:
            raise RuntimeError("超过 max_steps 仍未结束")
        self._post_process(self.prev)                 # flush_inflight_reqs
        self.prev = None
        for s in seqs:
            if s.block_table:
                self.bm.free_blocks(s.block_table)
                s.block_table = []
        return [s.output() for s in seqs]


def simulate_timeline(n_steps: int, *, sched=2.0, launch=1.0, fwd=6.0, samp=1.0, post=2.0, zos: bool):
    """返回 (事件列表, 总时长)。事件为 (资源, 名字, 开始, 结束)，资源是 "CPU" 或 "GPU"。

    同步：schedule(N) → launch → GPU forward+sample → CPU 等 GPU → post(N)，GPU 在 post 和下一次 schedule 期间空闲。
    ZOS：GPU 跑第 N 步时，CPU 依次做 schedule(N+1)、launch(N+1)、post(N-1)；只有 post 需要等第 N-1 步采样完成。
    """
    ev, cpu, gpu_free = [], 0.0, 0.0
    sample_done = {}
    for n in range(n_steps):
        ev.append(("CPU", f"schedule {n}", cpu, cpu + sched)); cpu += sched
        ev.append(("CPU", f"launch {n}", cpu, cpu + launch)); cpu += launch
        g0 = max(cpu, gpu_free)
        ev.append(("GPU", f"forward {n}", g0, g0 + fwd))
        ev.append(("GPU", f"sample {n}", g0 + fwd, g0 + fwd + samp))
        gpu_free = sample_done[n] = g0 + fwd + samp
        target = n if not zos else n - 1
        if target >= 0:
            cpu = max(cpu, sample_done[target])          # 等这一步采样完成（GPU→CPU 同步）
            ev.append(("CPU", f"post {target}", cpu, cpu + post)); cpu += post
    if zos:
        cpu = max(cpu, sample_done[n_steps - 1])
        ev.append(("CPU", f"post {n_steps - 1}", cpu, cpu + post)); cpu += post
    return ev, max(cpu, gpu_free)
