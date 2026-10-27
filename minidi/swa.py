"""滑动窗口（SWA）KV 的环形缓冲区，带投机解码回退余量。

对照 DeepInfer：
- deepinfer/worker/cache_engine.py L88 get_inference_swa_block_size：ring 长度 = window_size + DEEPINFER_MAX_BACKTRACKING_TOKENS；
- deepinfer/environ.py L139：MAX_BACKTRACKING_TOKENS 取 max(环境变量, nextn + 1)；
- 官方 Q&A：V4.1 推理视角 ring = 窗口 128 + 最大回退 6 = 134。
SWA KV 放在每个请求一个定长 slot 的 state cache 里（不走分页块），位置 p 写到 p % ring。

一次 decode forward 先把 k = 1 + 草稿数 个位置全部写进 ring，再并行做注意力。
第一个 query（位置 L）要看 [L-W+1, L]，而最后写入的位置 L+k-1 会覆盖 (L+k-1) - ring 号位置，
所以必须 ring >= W + k - 1，即余量 extra >= k - 1。被拒绝的草稿位置不用擦，下一步会被覆盖。
"""
from __future__ import annotations

import numpy as np


class SWARing:
    def __init__(self, window: int, extra: int, dim: int):
        self.window, self.extra = window, extra
        self.ring = window + extra
        self.buf = np.full((self.ring, dim), np.nan)
        self.pos_of_slot = np.full(self.ring, -1)
        self.committed = 0                 # 已提交（被接受）的长度

    def write(self, start: int, vecs: np.ndarray) -> None:
        assert start == self.committed, "一步只能从已提交长度处开始写"
        for i, v in enumerate(vecs):
            s = (start + i) % self.ring
            self.buf[s] = v
            self.pos_of_slot[s] = start + i

    def commit(self, new_len: int) -> None:
        assert new_len > self.committed
        self.committed = new_len

    def window_for(self, pos: int) -> np.ndarray:
        """位置 pos 的 query 能看到 [pos-window+1, pos]；若其中有位置已被覆盖则报错。"""
        lo = max(0, pos - self.window + 1)
        ps = np.arange(lo, pos + 1)
        slots = ps % self.ring
        if not (self.pos_of_slot[slots] == ps).all():
            raise RuntimeError("窗口内的位置已被覆盖")
        return self.buf[slots]
