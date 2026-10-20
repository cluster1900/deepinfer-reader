import numpy as np
import pytest

from minidi.blocks import AllocStatus, BlockManager
from minidi.model import PagedKV, TinyLM


def test_paged_forward_matches_reference_under_any_chunking():
    m = TinyLM()
    rng = np.random.default_rng(0)
    toks = list(rng.integers(0, 64, 37))
    ref = m.reference_logits(toks)
    for bs in (1, 3, 4, 16):
        kv = PagedKV(64, bs, m.dim)
        table = list(rng.permutation(np.arange(1, 64))[: (len(toks) + bs - 1) // bs])  # 物理块乱序
        pos, rows = 0, []
        while pos < len(toks):
            q = int(rng.integers(1, 9))
            rows.append(m.paged_forward(kv, [(toks[pos:pos + q], pos, table)]))
            pos += q
        np.testing.assert_allclose(np.concatenate(rows)[: len(toks)], ref, rtol=0, atol=1e-10)


def test_block0_reserved_and_fifo_order():
    bm = BlockManager(5)
    assert bm.num_free() == 4
    assert bm.allocate(2) == [1, 2]
    bm.free_blocks([1])
    assert bm.allocate(3) == [3, 4, 1]       # 释放的块挂到表尾
    with pytest.raises(RuntimeError):
        bm.allocate(1)


def test_alloc_status_three_states():
    bm = BlockManager(5)
    assert bm.can_allocate(4) is AllocStatus.OK
    assert bm.can_allocate(5) is AllocStatus.NEVER   # >= num_blocks 永远不行（块 0 不可用）
    bm.allocate(2)
    assert bm.can_allocate(3) is AllocStatus.LATER


def test_pin_keeps_block_until_last_free():
    bm = BlockManager(4)
    [b] = bm.allocate(1)
    bm.pin([b])
    bm.free_blocks([b])
    assert b not in bm.free
    bm.free_blocks([b])
    assert b in bm.free
    with pytest.raises(AssertionError):
        bm.free_blocks([b])
