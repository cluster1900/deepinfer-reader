"""请求（Sequence）与状态机。

对照 DeepInfer csrc/engine/sequence.hpp：
- SequenceStatus 枚举顺序与 C++ 一致，is_finished = status >= FINISHED_STOPPED；
- num_cached 对应 cache.num_tokens：已经写进 KV 的 token 数；
- get_num_blocks 对应 Sequence::get_num_blocks：(len + inflight + draft 数 + bs - 1) // bs。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum


class Status(IntEnum):
    WAITING = 0
    RUNNING = 1
    SWAPPED = 2
    FINISHED_STOPPED = 3
    FINISHED_LENGTH_CAPPED = 4
    FINISHED_ABORTED = 5
    FINISHED_IGNORED = 6
    FINISHED_PREEMPTED = 7


FINISH_REASON = {
    Status.FINISHED_STOPPED: "stop",
    Status.FINISHED_LENGTH_CAPPED: "length",
    Status.FINISHED_ABORTED: "abort",
    Status.FINISHED_IGNORED: "prompt_length_exceeded",
    Status.FINISHED_PREEMPTED: "insufficient_system_resource",
}


@dataclass
class Sequence:
    seq_id: int
    prompt: list[int]
    max_tokens: int = 16
    ignore_eos: bool = False
    seed: int = 0
    temperature: float = 0.0
    block_size: int = 4
    tokens: list[int] = field(default_factory=list)
    status: Status = Status.WAITING
    num_cached: int = 0          # cache.num_tokens
    block_table: list[int] = field(default_factory=list)
    draft_tokens: list[int] = field(default_factory=list)
    accept_rates: list[float] = field(default_factory=list)
    inflight: int = 0            # schedule.total_inflight_tokens（异步调度用）
    num_preempted: int = 0

    def __post_init__(self):
        if not self.tokens:
            self.tokens = list(self.prompt)

    @property
    def prompt_len(self) -> int:
        return len(self.prompt)

    def get_len(self) -> int:
        return len(self.tokens)

    def output_len(self) -> int:
        return len(self.tokens) - len(self.prompt)

    def output(self) -> list[int]:
        return self.tokens[len(self.prompt):]

    def get_num_blocks(self) -> int:
        n = self.get_len() + self.inflight + len(self.draft_tokens)
        return (n + self.block_size - 1) // self.block_size

    def is_finished(self) -> bool:
        return self.status >= Status.FINISHED_STOPPED

    def reset_for_recompute(self) -> None:
        """对应 Sequence::reset_for_recompute：KV 作废，已生成的 token 保留，重新 prefill 整段。"""
        self.num_cached = 0
        self.inflight = 0
        self.draft_tokens = []
        self.accept_rates = []
