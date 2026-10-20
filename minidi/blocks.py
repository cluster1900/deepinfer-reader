"""块管理器：引用计数 + 空闲双向链表。

对照 DeepInfer csrc/engine/block_manager.hpp：
- Blocks：块 0 保留（"0 代表空"），空闲块用 prev/next 串成链表，allocate 从表头取、free 到 ref_count==0 时挂回表尾；
- pin_block：给已分配块 +1 引用（fork 共享前缀时用）；
- BlockManager.can_allocate：返回 OK / LATER / NEVER 三态，"要的块数 >= 总块数"直接 NEVER。
Python 版用 dict 模拟链表，行为（分配顺序、三态判断）与 C++ 版一致。
"""
from __future__ import annotations

from collections import OrderedDict
from enum import Enum


class AllocStatus(Enum):
    OK = 0
    LATER = 1
    NEVER = 2


class BlockManager:
    def __init__(self, num_blocks: int):
        assert num_blocks > 1, "GPU blocks 太少了"
        self.num_blocks = num_blocks
        self.ref = [0] * num_blocks
        self.free = OrderedDict((i, None) for i in range(1, num_blocks))  # 有序 = 链表顺序

    def num_free(self) -> int:
        return len(self.free)

    def can_allocate(self, n: int) -> AllocStatus:
        if n >= self.num_blocks:
            return AllocStatus.NEVER
        return AllocStatus.OK if self.num_free() >= n else AllocStatus.LATER

    def allocate(self, n: int = 1) -> list[int]:
        out = []
        for _ in range(n):
            if not self.free:
                raise RuntimeError("list is empty, but pop_front is called")
            b, _ = self.free.popitem(last=False)  # pop_front
            assert self.ref[b] == 0
            self.ref[b] = 1
            out.append(b)
        return out

    def pin(self, blocks: list[int]) -> None:
        for b in blocks:
            assert self.ref[b] > 0, "try to pin free block"
            self.ref[b] += 1

    def free_blocks(self, blocks: list[int]) -> None:
        for b in reversed(blocks):  # C++ 版用 rbegin/rend 逆序释放
            assert 0 < b < self.num_blocks
            self.ref[b] -= 1
            assert self.ref[b] >= 0, "ref_count corrupted"
            if self.ref[b] == 0:
                self.free[b] = None  # push_back

    def usage(self) -> float:
        return 1.0 - self.num_free() / (self.num_blocks - 1)
