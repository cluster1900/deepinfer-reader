import itertools

import numpy as np
import pytest

from minidi import eplb, vote
from minidi.swa import SWARing
from minidi.varlen import cumulative, schedule_varlen, step_time

CURVE = [(8, 5.0), (16, 5.5), (24, 6.5), (32, 8.0), (48, 11.0)]


def brute(acc, curve):
    best = 0.0
    n = len(acc)
    for ks in itertools.product(*[range(len(a) + 1) for a in acc]):
        q = n + sum(ks)
        t = step_time(q, curve)
        if t is None:
            continue
        best = max(best, (n + sum(sum(a[:k]) for a, k in zip(acc, ks))) / t)
    return best


@pytest.mark.parametrize("seed", range(8))
def test_varlen_greedy_is_optimal_over_prefixes(seed):
    rng = np.random.default_rng(seed)
    acc = [cumulative(list(rng.uniform(0.2, 1.0, 3))) for _ in range(int(rng.integers(2, 5)))]
    seqlens, bsz, tp = schedule_varlen(acc, CURVE)
    assert sum(seqlens) == bsz and all(1 <= s <= 4 for s in seqlens)
    assert tp == pytest.approx(brute(acc, CURVE))


def test_varlen_cuts_low_confidence_tokens():
    acc = [cumulative([0.95] * 5), cumulative([0.3] * 5)]
    seqlens, _, _ = schedule_varlen(acc * 4, CURVE)
    assert seqlens[0] > seqlens[1]


def _run_ring(W, extra, k_max, steps=300, seed=0):
    rng = np.random.default_rng(seed)
    ring = SWARing(W, extra, 2)
    truth, length = {}, 0
    for _ in range(steps):
        k = int(rng.integers(1, k_max + 1))           # 1 个主 token + (k-1) 个草稿
        vecs = rng.normal(size=(k, 2))
        ring.write(length, vecs)
        for i in range(k):
            truth[length + i] = vecs[i]
        for p in range(length, length + k):           # 本步所有 query 并行读窗口
            lo = max(0, p - W + 1)
            np.testing.assert_array_equal(ring.window_for(p), np.stack([truth[i] for i in range(lo, p + 1)]))
        length += int(rng.integers(1, k + 1))         # 接受前若干个，其余回退
        ring.commit(length)


def test_swa_ring_survives_rollback_when_extra_ge_k_minus_1():
    for nextn in (0, 1, 3, 5):
        _run_ring(W=8, extra=nextn, k_max=1 + nextn)


def test_swa_ring_breaks_without_enough_extra():
    with pytest.raises(RuntimeError):
        _run_ring(W=8, extra=2, k_max=4, steps=200)   # extra = k-2，第一个 query 的窗口被覆盖


def test_eplb_improves_balance():
    rng = np.random.default_rng(0)
    E, R = 16, 4
    hot = rng.zipf(1.6, E).astype(float)
    tokens = rng.poisson(np.tile(hot * 50, (100, 1))).astype(float)
    p2l0, r0 = eplb.naive(E, R)
    base = eplb.moe_lb(tokens, p2l0, r0)
    p2l = eplb.replicate(tokens, E + R)           # 每卡 1 个冗余专家
    ranks = eplb.place(tokens, p2l, R)
    assert all(len(r) == (E + R) // R for r in ranks)
    assert eplb.moe_lb(tokens, p2l, ranks) < base


def test_vote_points():
    assert vote.decide(5, [True, True], [True, True])[0] == "decode"            # 不在尝试点
    assert vote.decide(70, [True, False], [True, True])[0] == "decode"          # ALL 点：要全员
    assert vote.decide(0, [True, True], [True, False]) == ("prefill", ["prefill", "empty-prefill"])
    assert vote.decide(100, [True, False], [True, True]) == ("prefill", ["prefill", "empty-prefill"])  # ANY 点
    assert vote.decide(0, [True, True], [False, False])[0] == "decode"          # 第二轮投票兜底
