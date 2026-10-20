"""MoE 专家并行负载均衡（EPLB）：给热门专家加副本，再把副本贪心放到各 rank。

对照 DeepInfer deepinfer/model_executor/eplb/policy.py：
- rebalance_experts_global（L73）：每轮挑一个 logical expert 复制，挑法是"复制后所有 step 的最大每副本负载之和最小"；
- _batch_greedy_indices（L12）：副本按历史负载降序，逐个放到"放进去后 sum_step max_rank 负载最小"的未满 rank。
这里用 numpy 写单层版本，指标 moe_lb = 最忙 rank 负载 / 平均负载（与官方 Q&A 的 moe_lb 定义一致）。
"""
from __future__ import annotations

import numpy as np


def replicate(tokens: np.ndarray, num_physical: int) -> np.ndarray:
    """tokens: [steps, experts]。返回 physical_to_logical: [num_physical]。"""
    steps, E = tokens.shape
    count = np.ones(E)
    p2l = list(range(E))
    t = tokens + np.arange(E) * 1e-4                 # 同 C++：打破并列
    for _ in range(num_physical - E):
        best, best_score = None, None
        for j in range(E):
            c = count.copy()
            c[j] += 1
            score = (t / c).max(-1).sum()
            if best_score is None or score < best_score:
                best, best_score = j, score
        count[best] += 1
        p2l.append(best)
    return np.array(p2l)


def place(tokens: np.ndarray, p2l: np.ndarray, num_ranks: int) -> list[list[int]]:
    steps, E = tokens.shape
    count = np.bincount(p2l, minlength=E)
    per_copy = tokens / count                          # 每个副本分到的负载 [steps, E]
    cap = len(p2l) // num_ranks
    order = sorted(range(len(p2l)), key=lambda k: -per_copy[:, p2l[k]].sum())
    loads = np.zeros((num_ranks, steps))
    peaks = np.zeros(steps)
    ranks = [[] for _ in range(num_ranks)]
    for k in order:
        v = per_copy[:, p2l[k]]
        scores = [np.inf if len(ranks[r]) == cap else np.maximum(loads[r] + v, peaks).sum() for r in range(num_ranks)]
        r = int(np.argmin(scores))
        ranks[r].append(k)
        loads[r] += v
        peaks = np.maximum(peaks, loads[r])
    return ranks


def moe_lb(tokens: np.ndarray, p2l: np.ndarray, ranks: list[list[int]]) -> float:
    count = np.bincount(p2l, minlength=tokens.shape[1])
    per_copy = tokens / count
    load = np.array([[per_copy[:, p2l[k]] for k in r] for r in ranks]).sum(1)  # [ranks, steps]
    return float((load.max(0) / load.mean(0)).mean())


def naive(num_experts: int, num_ranks: int) -> tuple[np.ndarray, list[list[int]]]:
    p2l = np.arange(num_experts)
    cap = num_experts // num_ranks
    return p2l, [list(range(r * cap, (r + 1) * cap)) for r in range(num_ranks)]
