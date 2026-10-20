"""引擎：schedule → forward → sample → process_outputs 的 step 循环。

对照 DeepInfer：
- LLM.generate / _run_engine：deepinfer/entrypoints/llm.py L95-L222，循环调用 step()，
  没有未完成请求时调用 flush_inflight_reqs() 收尾；
- LLMEngine.step：deepinfer/engine/llm_engine.py L860；
- Worker.execute_model_step：deepinfer/worker/worker.py L89（forward → 处理上一步 → sample → 处理本步）。
本文件是同步版本（相当于 DEEPINFER_ZERO_OVERHEAD_SCHEDULING=0）；异步版见 zos.py。
可选的投机解码（draft 模型 + 贪心验证）用 nextn>0 打开。
"""
from __future__ import annotations

import numpy as np

from .blocks import BlockManager
from .model import PagedKV, TinyLM
from .scheduler import Scheduler, SchedulerOutputs
from .sequence import Sequence


def sample(logits_row: np.ndarray, seq: Sequence, position: int) -> int:
    """贪心或按 (seed, position) 确定的随机采样：同一请求同一位置，换什么批次都得到同一个 token。"""
    if seq.temperature <= 0:
        return int(np.argmax(logits_row))
    rng = np.random.default_rng([seq.seed, position])
    z = logits_row / seq.temperature
    p = np.exp(z - z.max())
    p /= p.sum()
    return int(rng.choice(len(p), p=p))


class Engine:
    def __init__(self, model: TinyLM | None = None, *, num_blocks=64, block_size=4, draft: TinyLM | None = None,
                 nextn: int = 0, **sched_kw):
        self.model = model or TinyLM()
        self.block_size = block_size
        self.kv = PagedKV(num_blocks, block_size, self.model.dim)
        self.bm = BlockManager(num_blocks)
        self.scheduler = Scheduler(self.bm, **sched_kw)
        # 对应 LLMEngine._init_cache（llm_engine.py L445-L451）：单个请求最多只能占 KV 容量的 80%。
        # 没有这道闸门，一个长请求的 chunked prefill 可能把使用率顶过 0.9 水位线，之后再也排不上 prefill。
        max_seq_len = int(block_size * num_blocks * 0.8)
        if self.scheduler.max_model_len > max_seq_len:
            raise ValueError(f"max_model_len={self.scheduler.max_model_len} 超过 KV 可容纳的 {max_seq_len}，"
                             "调大 num_blocks 或调小 max_model_len")
        self.draft, self.nextn = draft, nextn
        self.seq_counter = 0
        self.steps: list[tuple[bool, int, int]] = []   # (prompt_run, num_seqs, num_tokens)
        self.num_draft = self.num_accepted = 0

    def add_request(self, prompt: list[int], **kw) -> Sequence:
        if not prompt:
            raise ValueError("prompt must not be empty")
        seq = Sequence(self.seq_counter, list(prompt), block_size=self.block_size, **kw)
        self.seq_counter += 1
        self.scheduler.add(seq)
        return seq

    # ---------------------------------------------------------------
    def _propose(self, seq: Sequence) -> None:
        if self.draft is None or self.nextn == 0 or seq.is_finished():
            seq.draft_tokens = []
            return
        ctx = list(seq.tokens)
        drafts = []
        for _ in range(self.nextn):
            t = int(np.argmax(self.draft.reference_logits(ctx)[-1]))
            drafts.append(t)
            ctx.append(t)
        seq.draft_tokens = drafts

    def step(self) -> SchedulerOutputs:
        out = self.scheduler.schedule()
        self.steps.append((out.prompt_run, len(out.scheduled), out.num_batched_tokens))
        if out.is_empty():
            return out
        batch = []
        for seq, q in zip(out.scheduled, out.seqlens_q):
            start = seq.num_cached
            if out.prompt_run:
                q_tokens = seq.tokens[start:start + q]
            else:
                q_tokens = [seq.tokens[-1]] + seq.draft_tokens[: q - 1]
            batch.append((q_tokens, start, seq.block_table))
        logits = self.model.paged_forward(self.kv, batch)
        new_tokens, off = [], 0
        for seq, q, (q_tokens, start, _) in zip(out.scheduled, out.seqlens_q, batch):
            rows = logits[off: off + q]
            off += q
            if out.prompt_run:
                done = start + q >= seq.get_len()
                new_tokens.append([sample(rows[-1], seq, start + q)] if done else [])
                continue
            accepted = []
            for j in range(q):                  # 贪心验证：第 j 行预测位置 start+j+1
                t = sample(rows[j], seq, start + j + 1)
                accepted.append(t)
                if j < q - 1 and t != seq.draft_tokens[j]:
                    break
            self.num_draft += q - 1
            self.num_accepted += len(accepted) - 1
            new_tokens.append(accepted)
        self.scheduler.process_outputs(out, new_tokens)
        for seq in out.scheduled:
            if not seq.is_finished() and seq in self.scheduler.running:
                self._propose(seq)
        return out

    def generate(self, prompts: list[list[int]], max_steps: int = 10_000, **kw) -> list[list[int]]:
        seqs = [self.add_request(p, **kw) for p in prompts]
        for _ in range(max_steps):
            if not self.scheduler.has_unfinished():
                break
            self.step()
        else:
            raise RuntimeError("超过 max_steps 仍未结束")
        for s in seqs:
            if s.block_table:
                self.bm.free_blocks(s.block_table)
                s.block_table = []
        return [s.output() for s in seqs]


def reference_generate(model: TinyLM, prompt: list[int], max_tokens: int, eos_id=None) -> list[int]:
    """不带缓存、一次一个请求的参考生成，用来验证引擎输出。"""
    toks = list(prompt)
    out = []
    for _ in range(max_tokens):
        t = int(np.argmax(model.reference_logits(toks)[-1]))
        out.append(t)
        toks.append(t)
        if eos_id is not None and t == eos_id:
            break
    return out
