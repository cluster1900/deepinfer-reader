"""调度器：prefill 步与 decode 步分开，按间隔尝试 prefill，decode 时块不够就抢占。

对照 DeepInfer csrc/engine/scheduler.hpp（行号以 commit 8b16149 为准）：
- _schedule (L275)：NORMAL 引擎每步只做一种：prompt 步或 decode 步；
  decode_step 落在 prefill_attempt_interval 的倍数上才尝试 prefill，且 KV 使用率 <= 0.9（L80、L313）；
- _schedule_prefill (L351)：从 waiting 队头取，遇到 LATER 就停（不跳过），超长则 IGNORED；
  token 预算不够时切 chunk，NORMAL 引擎切了 chunk 后本轮不再继续排；
- _schedule_decode (L559)：running 按 output_len 降序排，从尾部（输出最短）开始排入，
  块不够时从头部（输出最长）开始抢占；
- process_outputs (L872)：prefill 完成的请求在这里才进入 running；chunk 没跑完的放回 waiting 队头；
- _check_stop (L1035)：EOS / 超过 max_model_len / 到达 max_tokens。
多 DP rank 投票（ALL/ANY 点）单独放在 vote.py。
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from .blocks import AllocStatus, BlockManager
from .sequence import Sequence, Status

PREFILL_WATERMARK = 0.9


@dataclass
class SchedulerOutputs:
    prompt_run: bool
    scheduled: list[Sequence] = field(default_factory=list)
    seqlens_q: list[int] = field(default_factory=list)
    ignored: list[Sequence] = field(default_factory=list)
    finished: list[Sequence] = field(default_factory=list)

    @property
    def num_batched_tokens(self) -> int:
        return sum(self.seqlens_q)

    def is_empty(self) -> bool:
        return not self.scheduled


class Scheduler:
    def __init__(self, bm: BlockManager, *, max_num_seqs=8, max_num_batched_tokens=64, max_model_len=128,
                 chunked_prefill=True, prefill_attempt_interval=1, stop_preempted_request=False, eos_id=None):
        self.bm = bm
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.max_model_len = max_model_len
        self.chunked_prefill = chunked_prefill
        self.prefill_attempt_interval = prefill_attempt_interval
        self.stop_preempted_request = stop_preempted_request
        self.eos_id = eos_id
        self.prompt_limit = max_model_len if chunked_prefill else min(max_model_len, max_num_batched_tokens)
        self.waiting: deque[Sequence] = deque()
        self.running: list[Sequence] = []
        self.decode_step = 0
        self.num_preempted = 0
        self.log: list[str] = []

    # ------------------------------------------------------------------
    def add(self, seq: Sequence) -> None:
        self.waiting.append(seq)

    def has_unfinished(self) -> bool:
        return bool(self.waiting) or bool(self.running)

    def _free(self, seq: Sequence) -> None:
        self.bm.free_blocks(seq.block_table)
        seq.block_table = []

    # ------------------------------------------------------------------
    def schedule(self) -> SchedulerOutputs:
        out = None
        if self.decode_step % self.prefill_attempt_interval == 0:
            want = self.bm.usage() <= PREFILL_WATERMARK and bool(self.waiting)
            if want:
                out = self._schedule_prefill()
        if out is None:
            out = self._schedule_decode()
        if out.prompt_run:
            self.decode_step = 0
        else:
            self.running = list(out.scheduled)
            self.decode_step += 1
        return out

    def _schedule_prefill(self) -> SchedulerOutputs | None:
        out = SchedulerOutputs(prompt_run=True)
        num_curr = len(self.running)
        budget = self.max_num_batched_tokens
        while self.waiting:
            seq = self.waiting[0]
            if seq.is_finished():                       # 入队前已判死（abort 等）
                out.ignored.append(self.waiting.popleft())
                continue
            todo = seq.get_len() - seq.num_cached
            if not seq.block_table:
                if seq.get_len() > self.prompt_limit:
                    seq.status = Status.FINISHED_IGNORED
                    out.ignored.append(self.waiting.popleft())
                    continue
                st = self.bm.can_allocate(seq.get_num_blocks())
                if st is AllocStatus.LATER:
                    break                                # 队头放不下就整体等待，保持 FCFS
                if st is AllocStatus.NEVER:
                    seq.status = Status.FINISHED_IGNORED
                    out.ignored.append(self.waiting.popleft())
                    continue
            if num_curr + 1 > self.max_num_seqs:
                break
            q = min(todo, budget - out.num_batched_tokens)
            is_chunked = q < todo
            if q <= 0 or (is_chunked and not self.chunked_prefill):
                break
            if not seq.block_table:
                seq.block_table = self.bm.allocate(seq.get_num_blocks())
            if not is_chunked:
                num_curr += 1
                seq.status = Status.RUNNING
            out.scheduled.append(self.waiting.popleft())
            out.seqlens_q.append(q)
            if is_chunked:
                break                                    # NORMAL 引擎：切了 chunk 就收手
        if out.scheduled or out.ignored:
            return out
        return None

    def _schedule_decode(self) -> SchedulerOutputs:
        out = SchedulerOutputs(prompt_run=False)
        out.finished = [s for s in self.running if s.is_finished()]
        for s in out.finished:                         # 对应 pre_sampler_sync 里的 finalize_seq → free_seq
            self._free(s)
        self.running = [s for s in self.running if not s.is_finished()]
        running = sorted(self.running, key=lambda s: s.output_len(), reverse=True)
        l, r = 0, len(running)
        while l < r and len(out.scheduled) < self.max_num_seqs:
            seq = running[r - 1]
            need = seq.get_num_blocks() - len(seq.block_table)
            while need > self.bm.num_free() and r - l > 1:
                self._preempt(running[l], out)
                l += 1
            if need <= self.bm.num_free():
                if need > 0:
                    seq.block_table += self.bm.allocate(need)
                elif need < 0:                     # 对应 _append_slot (L821-L825)：draft 被拒后回收多余尾块
                    self.bm.free_blocks(seq.block_table[need:])
                    del seq.block_table[need:]
                out.scheduled.append(seq)
            else:
                self._preempt(seq, out)
            r -= 1
        out.scheduled.sort(key=lambda s: s.get_len(), reverse=True)
        out.seqlens_q = [1 + len(s.draft_tokens) for s in out.scheduled]
        return out

    def _preempt(self, seq: Sequence, out: SchedulerOutputs) -> None:
        self.num_preempted += 1
        seq.num_preempted += 1
        self._free(seq)
        if self.stop_preempted_request:
            seq.status = Status.FINISHED_PREEMPTED
            out.ignored.append(seq)
            self.log.append(f"preempt-stop seq{seq.seq_id}")
        else:
            seq.reset_for_recompute()
            seq.status = Status.WAITING
            # 放回队头，但排在"已经分到块、prefill 做了一半"的请求后面。
            # 否则队头的重算请求要的块被半截 chunk 占着、半截 chunk 又排不到，两边互等（见第 12 章"坑"）。
            i = 0
            while i < len(self.waiting) and self.waiting[i].block_table:
                i += 1
            self.waiting.insert(i, seq)
            self.log.append(f"preempt-recompute seq{seq.seq_id}")

    # ------------------------------------------------------------------
    def process_outputs(self, out: SchedulerOutputs, new_tokens: list[list[int]]) -> None:
        """new_tokens[i]：第 i 个请求本步确定接受的 token（prefill 未完成的 chunk 为空列表）。"""
        for seq, q, toks in zip(out.scheduled, out.seqlens_q, new_tokens):
            if seq.is_finished():
                continue
            if not out.prompt_run and seq.status == Status.WAITING:
                continue                                  # 本步在途时被抢占：输出作废
            seq.num_cached += q                           # 乐观推进
            if seq.num_cached >= seq.get_len():
                if out.prompt_run:
                    self.running.append(seq)
                cnt = 0
                for t in toks:
                    seq.tokens.append(t)
                    cnt += 1
                    self._check_stop(seq, t)
                    if seq.is_finished():
                        break
                # 回填（commit_processed_tokens）：被拒绝的 draft 写过的 KV 位置不算数，
                # 最后一个新 token 还没进过模型，所以已缓存 = 长度 - 1
                seq.num_cached = seq.get_len() - 1
            else:
                seq.status = Status.WAITING
                self.waiting.appendleft(seq)              # chunk 没跑完，放回队头

    def _check_stop(self, seq: Sequence, token: int) -> None:
        if not seq.ignore_eos and self.eos_id is not None and token == self.eos_id:
            seq.status = Status.FINISHED_STOPPED
        elif seq.get_len() > self.max_model_len:
            seq.status = Status.FINISHED_LENGTH_CAPPED
        elif seq.output_len() == seq.max_tokens:
            seq.status = Status.FINISHED_LENGTH_CAPPED
