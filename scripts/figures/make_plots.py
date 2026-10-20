#!/usr/bin/env python3
"""生成书中的数据图（全部在 CPU 上由 minidi 实际运行得到，或读取 data/ 下的统计）。
同时支持生成中文版（assets/images/plots）和英文版（assets/images/plots_en）。

用法：
  python3 scripts/figures/make_plots.py --out assets/images/plots
  python3 scripts/figures/make_plots.py --out assets/images/plots_en --lang en
"""
import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from minidi import eplb  # noqa: E402
from minidi.engine import Engine  # noqa: E402
from minidi.model import TinyLM  # noqa: E402
from minidi.varlen import cumulative, schedule_varlen, step_time  # noqa: E402
from minidi.zos import simulate_timeline  # noqa: E402

from matplotlib import font_manager  # noqa: E402

for _d in ("/usr/share/fonts/opentype/noto", "/usr/share/fonts/truetype/noto"):
    if os.path.isdir(_d):
        for _f in sorted(os.listdir(_d)):
            if "CJK" in _f and "Regular" in _f:
                font_manager.fontManager.addfont(os.path.join(_d, _f))
plt.rcParams["font.sans-serif"] = [
    "Noto Sans CJK SC",
    "Noto Sans CJK JP",
    "PingFang SC",
    "PingFang HK",
    "Hiragino Sans GB",
    "Heiti TC",
    "STHeiti",
    "Arial Unicode MS",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["svg.hashsalt"] = "deepinfer-reader"

LOC_EN_MAP = {
    "TileLang 自定义算子 (layers/tilelang_ops)": "TileLang Custom Kernels (layers/tilelang_ops)",
    "模型结构 (model_executor/models)": "Model Definitions (model_executor/models)",
    "C++ 引擎核心 (csrc)": "C++ Engine Core (csrc)",
    "其余算子层 (layers 其他)": "Other Operator Layers (layers/*)",
    "Worker 与缓存 (worker)": "Worker & Cache Engine (worker)",
    "Python 引擎 (engine)": "Python Engine (engine)",
    "输入元数据 (input_metadata)": "Input Metadata (input_metadata)",
    "其余 deepinfer 包": "Other deepinfer Modules",
    "示例与基准 (examples)": "Examples & Benchmarks (examples)",
    "测试 (tests)": "Test Suite (tests)",
    "评测 (eval)": "Evaluation Harness (eval)",
    "deep_cluster": "deep_cluster",
    "其他（setup 等）": "Build & Setup (setup, etc.)",
}


def save(fig, out, name):
    fig.tight_layout()
    fig.savefig(os.path.join(out, name), dpi=150, metadata={"Software": None})
    plt.close(fig)


def loc(out, lang="zh"):
    d = json.load(open(os.path.join(ROOT, "data", "deepinfer_loc.json"), encoding="utf-8"))
    items = sorted(d["categories"].items(), key=lambda x: x[1])
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    labels = [LOC_EN_MAP.get(k, k) if lang == "en" else k for k, _ in items]
    ax.barh(labels, [v for _, v in items], color="#4C72B0")
    for i, (_, v) in enumerate(items):
        ax.text(v + 200, i, f"{v:,}", va="center", fontsize=8)
    if lang == "en":
        ax.set_xlabel("Lines of Code (.py/.cpp/.hpp, incl. comments & blank lines)")
        ax.set_title(f"DeepInfer {d['commit'][:7]} Codebase Breakdown ({d['total_lines']:,} lines total)")
    else:
        ax.set_xlabel("代码行数（.py/.cpp/.hpp，含注释空行）")
        ax.set_title(f"DeepInfer {d['commit'][:7]} 代码分布，共 {d['total_lines']:,} 行")
    save(fig, out, "ch01_loc.png")


def timeline(out, lang="zh"):
    fig, axes = plt.subplots(2, 1, figsize=(9, 3.8), sharex=True)
    colors = {"schedule": "#DD8452", "launch": "#C44E52", "post": "#8172B3", "forward": "#4C72B0", "sample": "#55A868"}
    for ax, zos in zip(axes, (False, True)):
        ev, total = simulate_timeline(5, zos=zos)
        for res, name, s, e in ev:
            y = 1 if res == "CPU" else 0
            kind, n = name.split()
            ax.broken_barh([(s, e - s)], (y - 0.35, 0.7), color=colors[kind], edgecolor="white")
            ax.text((s + e) / 2, y, f"{kind[0].upper()}{n}", ha="center", va="center", fontsize=7, color="white")
        ax.set_yticks([0, 1], ["GPU", "CPU"])
        if lang == "en":
            title_prefix = "Synchronous Scheduling" if not zos else "Async Scheduling (ZOS)"
            ax.set_title(f"{title_prefix}: 5 steps take {total:g} time units", fontsize=10)
        else:
            ax.set_title(("同步调度" if not zos else "异步调度（ZOS）") + f"：5 步共 {total:g} 个时间单位", fontsize=10)
    if lang == "en":
        axes[1].set_xlabel("Time (abstract units: schedule=2, launch=1, forward=6, sample=1, post=2)")
    else:
        axes[1].set_xlabel("时间（抽象单位：schedule=2, launch=1, forward=6, sample=1, post=2）")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors.values()]
    axes[0].legend(handles, list(colors), ncol=5, fontsize=7, loc="upper right")
    save(fig, out, "ch05_zos_timeline.png")


def varlen(out, lang="zh"):
    curve = [(8, 5.0), (16, 5.5), (24, 6.5), (32, 8.0), (48, 11.0), (64, 14.0)]
    rng = np.random.default_rng(0)
    acc = [cumulative(list(rng.uniform(0.4, 0.98, 5))) for _ in range(8)]
    n = len(acc)
    queue = sorted((r for a in acc for r in a), reverse=True)
    xs, ys, exp = [], [], float(n)
    for k in range(len(queue) + 1):
        if k:
            exp += queue[k - 1]
        t = step_time(n + k, curve)
        if t is not None:
            xs.append(n + k)
            ys.append(exp / t)
    seqlens, best, tp = schedule_varlen(acc, curve)
    fixed = (n + sum(sum(a) for a in acc)) / step_time(n * 6, curve)
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    if lang == "en":
        ax.plot(xs, ys, marker=".", color="#4C72B0", label="Greedy addition of drafts sorted by acceptance prob")
        ax.axvline(best, color="#C44E52", ls="--", label=f"Selected total_q={best} (per-seq {seqlens})")
        ax.axhline(fixed, color="#55A868", ls=":", label=f"Fixed verify all 5 drafts: total_q={n * 6}")
        ax.set_xlabel("Total verified tokens in step (total_q)")
        ax.set_ylabel("Expected accepted tokens / est. step time")
        ax.set_title("Variable-Length Verification: 8 requests x 5 draft tokens (synthetic)", fontsize=10)
    else:
        ax.plot(xs, ys, marker=".", color="#4C72B0", label="按接受率降序逐个加入 draft")
        ax.axvline(best, color="#C44E52", ls="--", label=f"选中 total_q={best}（每请求 {seqlens}）")
        ax.axhline(fixed, color="#55A868", ls=":", label=f"固定验证全部 5 个 draft：total_q={n * 6}")
        ax.set_xlabel("本步验证的 token 总数 total_q")
        ax.set_ylabel("期望接受 token / 估计耗时")
        ax.set_title("变长验证：8 个请求、每个 5 个 draft 的示例（合成数据）", fontsize=10)
    ax.legend(fontsize=8)
    save(fig, out, "ch06_varlen.png")


def eplb_fig(out, lang="zh"):
    rng = np.random.default_rng(0)
    E, R = 16, 4
    hot = rng.zipf(1.6, E).astype(float)
    tokens = rng.poisson(np.tile(hot * 50, (100, 1))).astype(float)
    p2l0, r0 = eplb.naive(E, R)
    p2l = eplb.replicate(tokens, E + R)
    r1 = eplb.place(tokens, p2l, R)

    def rank_loads(p2l, ranks):
        cnt = np.bincount(p2l, minlength=E)
        per = tokens.mean(0) / cnt
        return [sum(per[p2l[k]] for k in r) for r in ranks]

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2), sharey=True)
    if lang == "en":
        pairs = ((p2l0, r0, "Naive Placement: 4 experts/rank"), (p2l, r1, "EPLB: 16 experts + 4 replicas"))
    else:
        pairs = ((p2l0, r0, "朴素放置：每卡 4 个专家"), (p2l, r1, "EPLB：16 专家 + 4 冗余副本"))
    for ax, (p, rk, t) in zip(axes, pairs):
        ld = rank_loads(p, rk)
        ax.bar(range(R), ld, color="#4C72B0")
        ax.axhline(np.mean(ld), color="#C44E52", ls="--")
        ax.set_title(f"{t}\nmoe_lb={eplb.moe_lb(tokens, p, rk):.2f}", fontsize=9)
        ax.set_xlabel("rank")
    axes[0].set_ylabel("Avg tokens per step" if lang == "en" else "每步平均 token 数")
    save(fig, out, "ch07_eplb.png")


def trace(out, lang="zh"):
    m = TinyLM()
    rng = np.random.default_rng(7)
    prompts = [list(map(int, rng.integers(0, 64, int(n)))) for n in rng.integers(4, 30, 10)]
    e = Engine(m, num_blocks=20, max_num_seqs=6, max_num_batched_tokens=16, max_model_len=60)
    seqs = [e.add_request(p, max_tokens=20) for p in prompts]
    usage, kind, nseq = [], [], []
    while e.scheduler.has_unfinished():
        o = e.step()
        usage.append(e.bm.usage())
        kind.append(o.prompt_run)
        nseq.append(len(o.scheduled))
    fig, ax = plt.subplots(figsize=(8, 3.4))
    x = np.arange(len(usage))
    ax.bar(x, nseq, color=["#DD8452" if k else "#4C72B0" for k in kind], alpha=0.8)
    ax2 = ax.twinx()
    ax2.plot(x, usage, color="black", lw=1.2)
    ax2.axhline(0.9, color="#C44E52", ls=":", lw=1)
    ax2.set_ylim(0, 1.05)
    ax.set_xlabel("step")
    if lang == "en":
        ax.set_ylabel("Seqs in step (orange=prefill, blue=decode)")
        ax2.set_ylabel("KV block usage (red dotted = 0.9 watermark)")
        ax.set_title(f"minidi run: 10 requests, 19 usable blocks, {e.scheduler.num_preempted} preemptions", fontsize=10)
    else:
        ax.set_ylabel("本步请求数（橙=prefill 步，蓝=decode 步）")
        ax2.set_ylabel("KV 块使用率（红线 = 0.9 水位）")
        ax.set_title(f"minidi 实跑：10 个请求，19 个可用块，抢占 {e.scheduler.num_preempted} 次", fontsize=10)
    save(fig, out, "ch12_trace.png")
    assert all(s.is_finished() for s in seqs)


def swa_ring(out, lang="zh"):
    """W=4、k=3、已提交长度 L=10：用 minidi.swa.SWARing 实际写入，再画出每个槽里是哪个位置。"""
    from minidi.swa import SWARing
    W, k, L = 4, 3, 10
    fig, axes = plt.subplots(2, 1, figsize=(7.5, 3.6))
    for ax, extra in zip(axes, (k - 1, k - 2)):
        ring = SWARing(W, extra, 1)
        pos = 0
        while pos < L:
            ring.write(pos, np.zeros((1, 1))); pos += 1; ring.commit(pos)
        before = ring.pos_of_slot.copy()
        ring.write(L, np.zeros((k, 1)))
        need = set(range(L - W + 1, L + 1))
        for sl in range(ring.ring):
            p_now = int(ring.pos_of_slot[sl])
            overwritten = before[sl] != p_now and before[sl] in need
            color = "#C44E52" if overwritten else ("#DD8452" if p_now >= L else ("#4C72B0" if p_now in need else "#BBBBBB"))
            ax.add_patch(plt.Rectangle((sl, 0), 0.95, 1, color=color))
            if lang == "en":
                label = f"Slot {sl}\npos {p_now}" + (f"\n(overwrote {before[sl]})" if before[sl] != p_now and before[sl] >= 0 else "")
            else:
                label = f"槽{sl}\n位置 {p_now}" + (f"\n(覆盖 {before[sl]})" if before[sl] != p_now and before[sl] >= 0 else "")
            ax.text(sl + 0.47, 0.5, label, ha="center", va="center", fontsize=8, color="white")
        ok = all(int(ring.pos_of_slot[q % ring.ring]) == q for q in need)
        ax.set_xlim(0, 7); ax.set_ylim(0, 1); ax.axis("off")
        if lang == "en":
            status = "all preserved" if ok else "active window position overwritten!"
            ax.set_title(f"ring = W + {extra} = {ring.ring}: query at pos {L} needs {L-W+1}..{L} ({status})", fontsize=9)
        else:
            ax.set_title(f"ring = W + {extra} = {ring.ring}：位置 {L} 的 query 需要 {L-W+1}–{L}，" + ("全部还在" if ok else "其中有位置被覆盖"), fontsize=9)
    if lang == "en":
        fig.text(0.5, 0.01, "Orange = written this step   Blue = in-window history   Red = in-window overwritten   Gray = out-of-window", ha="center", fontsize=8)
    else:
        fig.text(0.5, 0.01, "橙=本步写入　蓝=窗口内旧位置　红=窗口内被覆盖　灰=窗口外", ha="center", fontsize=8)
    fig.subplots_adjust(bottom=0.1)
    save(fig, out, "ch15_swa_ring.png")


def block_table(out, lang="zh"):
    """两个请求的块表：minidi.blocks.BlockManager 实际分配（先分 A 再分 B，再释放/重分以打乱顺序）。"""
    from minidi.blocks import BlockManager
    bm = BlockManager(9)
    a = bm.allocate(2)
    b = bm.allocate(1)
    a += bm.allocate(1)
    b += bm.allocate(1)
    fig, ax = plt.subplots(figsize=(8, 2.8))
    owner = {0: ("Reserved" if lang == "en" else "保留", "#888888")}
    for name, table, n, c in (("A", a, 10, "#4C72B0"), ("B", b, 5, "#55A868")):
        for i, blk in enumerate(table):
            used = min(4, n - 4 * i)
            if lang == "en":
                pos_str = f"{name} pos {4*i}..{4*i+used-1}" if used > 1 else f"{name} pos {4*i}"
                rem_str = "" if used == 4 else f"\n{4-used} free"
            else:
                pos_str = f"{name} 位置 {4*i}–{4*i+used-1}" if used > 1 else f"{name} 位置 {4*i}"
                rem_str = "" if used == 4 else f"\n空 {4-used} 槽"
            owner[blk] = (pos_str + rem_str, c)
    for blk in range(9):
        lab, c = owner.get(blk, ("Free" if lang == "en" else "空闲", "#DDDDDD"))
        ax.add_patch(plt.Rectangle((blk, 0), 0.95, 1, color=c))
        blk_title = f"Block {blk}" if lang == "en" else f"块 {blk}"
        ax.text(blk + 0.47, 0.5, f"{blk_title}\n{lab}", ha="center", va="center", fontsize=7.5,
                color="white" if c not in ("#DDDDDD",) else "black")
    ax.set_xlim(0, 9); ax.set_ylim(0, 1); ax.axis("off")
    if lang == "en":
        ax.set_title(f"Physical Pool (block_size=4): Seq A block_table = {a}, Seq B block_table = {b}", fontsize=9)
    else:
        ax.set_title(f"物理池（block_size=4）：A 的块表 = {a}，B 的块表 = {b}", fontsize=9)
    save(fig, out, "ch11_block_table.png")


def generate_all(out, lang="zh"):
    os.makedirs(out, exist_ok=True)
    for f in (loc, timeline, varlen, eplb_fig, trace, swa_ring, block_table):
        f(out, lang=lang)
        print(f"+ [{lang}] {f.__name__}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "assets", "images", "plots"))
    ap.add_argument("--lang", choices=["zh", "en", "both"], default="both")
    a = ap.parse_args()
    if a.lang == "both":
        generate_all(a.out, lang="zh")
        out_en = os.path.join(ROOT, "assets", "images", "plots_en") if a.out == os.path.join(ROOT, "assets", "images", "plots") else a.out + "_en"
        generate_all(out_en, lang="en")
    else:
        generate_all(a.out, lang=a.lang)


if __name__ == "__main__":
    main()
