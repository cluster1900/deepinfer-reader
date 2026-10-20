"""多 DP rank 的步型投票：这一步全组做 prefill 还是 decode。

对照 DeepInfer csrc/engine/scheduler.hpp L296-L344：
- ALL 点：decode_step % prefill_attempt_interval == 0（默认 70，environ.py L174），全员 want 才做 prefill；
- ANY 点：decode_step != 0 且 decode_step % prefill_any_attempt_interval == 0（默认 100），任一 rank want 就做；重合时按 ANY；
- 第二轮投票：投出 prefill 后，若所有 rank 实际都没排出 prefill（slot 满、块不够），转回 decode，避免死循环；
- 没排出 prefill 的 rank 陪跑一个空 prompt 步，保证组内步型一致（MoE 的 EP 通信要求所有 rank 同步进入同一种 forward）。
"""
from __future__ import annotations


def decide(decode_step: int, wants: list[bool], can_schedule: list[bool], all_interval=70, any_interval=100) -> tuple[str, list[str]]:
    """返回 (全组步型, 每个 rank 的动作)。动作是 'prefill' / 'empty-prefill' / 'decode'。"""
    is_any = decode_step != 0 and decode_step % any_interval == 0
    is_all = decode_step % all_interval == 0
    n = len(wants)
    if not is_any and not is_all:
        return "decode", ["decode"] * n
    do_prefill = any(wants) if is_any else all(wants)
    if not do_prefill:
        return "decode", ["decode"] * n
    got = [w and c for w, c in zip(wants, can_schedule)]
    if not any(got):
        return "decode", ["decode"] * n
    return "prefill", ["prefill" if g else "empty-prefill" for g in got]
