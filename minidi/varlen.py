"""置信度驱动的变长验证（DSpark varlen verification）。

逐行对照 DeepInfer csrc/engine/mtp_varlen.hpp MTPVarlen::schedule（L150 起）：
1. 每个请求必验 1 个"主 token"，所以起点 total_q = num_seqs，期望 token 数 = num_seqs；
2. 把所有 draft token 的"累乘接受率"放进一个队列，按接受率降序排；
3. 逐个加入队列中的 token：total_q += 1，期望 += 接受率，用性能曲线估这一步耗时，
   吞吐 = 期望 token / 耗时，记下吞吐最大的 total_q（best_bsz）；
4. 再把前 best_bsz - num_seqs 个 token 按所属请求累加到 seqlens_q。
耗时 = 曲线上 >= total_q 的第一个 graph 的 baseline_ms + slope_ms * graph_size + sampler_ms。
（C++ 里算 best_bsz 与分配 seqlens_q 用的是相邻两轮的接受率，这里简化为同一组。）
"""
from __future__ import annotations

import bisect


def cumulative(rates: list[float]) -> list[float]:
    out, p = [], 1.0
    for r in rates:
        p *= r
        out.append(p)
    return out


def step_time(total_q: int, curve: list[tuple[int, float]], slope_ms=0.0, sampler_ms=0.0) -> float | None:
    sizes = [g for g, _ in curve]
    i = bisect.bisect_left(sizes, total_q)
    if i == len(curve):
        return None                                # 超过最大 CUDA graph，不可选
    g, base = curve[i]
    return base + max(0.0, slope_ms) * g + sampler_ms


def schedule_varlen(acc_rates: list[list[float]], curve, slope_ms=0.0, sampler_ms=0.0):
    """acc_rates[i] 是请求 i 每个 draft 的累乘接受率。返回 (seqlens_q, best_total_q, best_throughput)。"""
    n = len(acc_rates)
    queue = sorted(((r, i) for i, rs in enumerate(acc_rates) for r in rs), key=lambda x: (-x[0], x[1]))
    best_tp, best_bsz = 0.0, 0
    expected, bsz = float(n), n

    def update():
        nonlocal best_tp, best_bsz
        t = step_time(bsz, curve, slope_ms, sampler_ms)
        if t is not None and expected / t > best_tp:
            best_tp, best_bsz = expected / t, bsz

    update()
    for r, _ in queue:
        expected += r
        bsz += 1
        update()
    assert best_bsz >= n
    seqlens_q = [1] * n
    for r, i in queue[: best_bsz - n]:
        seqlens_q[i] += 1
    return seqlens_q, best_bsz, best_tp
