# 第 15 章　SWA 环形缓冲、EPLB 与步型投票

本章的三个模块都是“算法”，不依赖引擎：`swa.py`、`eplb.py`、`vote.py`。每个都可以单独测。

## 15.1 SWA 环形缓冲

滑动窗口注意力只看最近 W 个位置，KV 不需要分页，每个请求一个定长的环形缓冲区就够了：位置 p 写到槽 `p % ring`。DeepInfer 正是把 SWA 状态放在每请求一个 slot 的 state cache 里，ring 长度是 `window + DEEPINFER_MAX_BACKTRACKING_TOKENS`（第 4 章）。

**为什么 ring 要比窗口长？** 投机解码。一次 decode forward 先把 `k = 1 + 草稿数` 个位置全部写进 ring，再并行做注意力。第一个 query 在位置 L，要看 `[L-W+1, L]`；最后写入的位置 `L+k-1` 会覆盖槽 `(L+k-1) % ring`，也就是位置 `L+k-1-ring` 原来的内容。要它不在第一个 query 的窗口里：

```
L + k - 1 - ring < L - W + 1   ⟺   ring > W + k - 2   ⟺   ring ≥ W + k - 1
```

即余量 `extra = ring - W ≥ k - 1 = nextn`。被拒绝的 draft 不用擦，下一步从已提交长度处重新写，会覆盖它们。

![图 15-1　W=4、k=3、已提交长度 10 时写入一步后的 ring（用 minidi.swa.SWARing 实际写入）](images/plots/ch15_swa_ring.png)

上图 ring=6（余量 2 = k-1）：位置 12 写到槽 0，覆盖的是位置 6，它已经不在任何 query 的窗口里。下图 ring=5（余量 1）：位置 12 写到槽 2，覆盖了位置 7，而位置 10 的 query 正需要它。

`SWARing.window_for(pos)` 在读取时检查窗口内每个槽记录的位置是否就是要的位置，不是就抛错。两个测试：

- `test_swa_ring_survives_rollback_when_extra_ge_k_minus_1`：`extra = nextn`（即 `k-1`），nextn 取 0、1、3、5，每步随机接受若干个、其余回退，窗口内容始终正确；
- `test_swa_ring_breaks_without_enough_extra`：`extra = k - 2`，一定会读到被覆盖的位置。

**与 DeepInfer 的差异**。`environ.py` L132 让 `DEEPINFER_MAX_BACKTRACKING_TOKENS` 至少是 `nextn + 1`，比我们推导的下界 `nextn` 多 1。官方 Q&A 的例子是“窗口 128 + 最大回退 6 = 134”（nextn=5）。多出的 1 个位置可能有本书没覆盖到的用途（比如某个 kernel 的读写顺序，或者主 token 之外还要多保留一个位置），**本书没有追到原因，不下结论**。

## 15.2 EPLB

`minidi/eplb.py` 是第 7 章两个策略函数的单层 NumPy 版：

- `replicate(tokens, num_physical)`：每轮枚举给哪个专家加副本，选“加完后每步最大每副本负载之和”最小的那个。C++ 版用 `scatter` 批量算出所有候选，这里直接循环，逻辑相同；`+ arange * 1e-4` 打破并列也照搬了。
- `place(tokens, p2l, num_ranks)`：副本按历史总负载降序，逐个放到“放进去后 Σ_step max_rank 负载”最小的未满 rank。
- `moe_lb`：每步最忙 rank / 平均，再对步取平均。

`test_eplb_improves_balance` 在 16 个专家、4 个 rank、Zipf 热度的合成数据上，断言每 rank 1 个冗余专家后，每个 rank 的副本数相等，且 moe_lb 低于朴素放置。图 7-2 就是这组数据。

目标函数为什么是 Σ_step max 而不是总负载？一步的耗时由最忙的 rank 决定，而不同步的热点专家可能不同。只看总负载，可能把两个“轮流热”的专家放在一起：总负载均衡了，但每一步都有一个 rank 特别忙。

## 15.3 步型投票

`minidi/vote.py` 的 `decide` 对照 `scheduler.hpp` L296–L344 的多 rank 投票：

```python
is_any = decode_step != 0 and decode_step % any_interval == 0
is_all = decode_step % all_interval == 0
...
do_prefill = any(wants) if is_any else all(wants)
...
got = [w and c for w, c in zip(wants, can_schedule)]
if not any(got):
    return "decode", ["decode"] * n
return "prefill", ["prefill" if g else "empty-prefill" for g in got]
```

- ALL 点（默认每 70 步）要所有 rank 都想 prefill；ANY 点（默认每 100 步）只要一个 rank 想；重合时按 ANY。
- 第二轮：投出 prefill 后，如果所有 rank 实际都排不出（并发满、块不够），整组回到 decode，避免空转。
- 排不出的 rank 跑一个空 prefill 步陪跑。MoE 的 EP 通信要求所有 rank 同时进入同一种 forward，一个 rank 做 decode、另一个做 prefill，dispatch/combine 就对不上。

`test_vote_points` 覆盖了非尝试点、ALL 点、ANY 点、第二轮回退和空 prefill 陪跑。为什么要两种间隔？ALL 点保证不会因为一个 rank 想 prefill 就打断所有 rank 的 decode；ANY 点保证一个 rank 积压的请求最终能被处理，不会因为别人总是不想 prefill 而饿死。
