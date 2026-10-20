# 第 14 章　投机解码与变长验证

## 14.1 草稿与验证

minidi 的草稿模型是同一个 `TinyLM` 加噪声：`TinyLM(noise=0.3)`。噪声为 0 时草稿就是目标模型，应该全部接受；噪声越大接受率越低。

`Engine._propose` 让草稿模型贪心地往后猜 `nextn` 个 token。decode 步的输入是 `[上一个 token] + drafts`，一次 forward 算出 `1 + nextn` 行 logits。贪心验证：

```python
accepted = []
for j in range(q):                  # 第 j 行预测位置 start+j+1
    t = sample(rows[j], seq, start + j + 1)
    accepted.append(t)
    if j < q - 1 and t != seq.draft_tokens[j]:
        break
```

第 j 行是“看完前 j 个 draft 之后”目标模型的选择。和 draft 一致就继续；不一致就用目标模型的选择替换并停下。所以每步至少产出 1 个 token（全部拒绝时），最多 `1 + nextn` 个。

```{.mermaid caption="图 14-1　一次验证：nextn=3，第 2 个 draft 被拒"}
flowchart LR
  I["输入：t, d1, d2, d3"] --> F["一次 forward<br/>4 行 logits"]
  F --> R0["第 0 行 → d1 ✔"]
  F --> R1["第 1 行 → x ≠ d2 ✘"]
  R0 --> OUT["本步产出：d1, x"]
  R1 --> OUT
  F -.-> R2["第 2、3 行丢弃<br/>写过的 KV 由回填作废"]
```这里是贪心（温度 0）版本；随机采样时要用接受/拒绝采样保持分布不变，DeepInfer 的 sampler 做的是后者（第 6 章）。

## 14.2 KV 回退

forward 时 `1 + nextn` 个位置的 KV 都写进了缓存，被拒绝的 draft 写过的位置怎么办？答案是**不用擦**。第 12 章的回填 `num_cached = get_len() - 1` 让已缓存长度只覆盖被接受的部分，下一步会从那里开始覆盖写。分页 KV 下这是免费的；第 15 章的 SWA 环形缓冲区就没这么简单了。

`get_num_blocks` 把 `len(draft_tokens)` 算进去，保证 draft 的写入位置有块可用。

## 14.3 无损性测试

```python
@pytest.mark.parametrize("noise", [0.0, 0.2, 0.6])
def test_speculative_decoding_is_lossless(noise):
    e = Engine(M, num_blocks=80, max_num_batched_tokens=32, draft=TinyLM(noise=noise), nextn=3)
    assert e.generate(ps, max_tokens=12) == ref(ps, 12)
    if noise == 0.0:
        assert e.num_accepted == e.num_draft
```

不管草稿多差，输出都必须与不用投机的参考一致；草稿完美时必须全部接受。后一条能抓到“验证位置错一格”这类错误：错一格时完美草稿也会被拒。

## 14.4 变长验证

`minidi/varlen.py` 逐行对照 `MTPVarlen::schedule`（第 6 章）：

```python
def schedule_varlen(acc_rates, curve, slope_ms=0.0, sampler_ms=0.0):
    n = len(acc_rates)
    queue = sorted(((r, i) for i, rs in enumerate(acc_rates) for r in rs), key=lambda x: (-x[0], x[1]))
    best_tp, best_bsz = 0.0, 0
    expected, bsz = float(n), n
    update()                          # 只验主 token
    for r, _ in queue:                # 按累乘接受率降序逐个加入
        expected += r; bsz += 1
        update()                      # 吞吐 = expected / step_time(bsz)
    seqlens_q = [1] * n
    for r, i in queue[: best_bsz - n]:
        seqlens_q[i] += 1
    return seqlens_q, best_bsz, best_tp
```

`step_time` 在阶梯形性能曲线上找第一个不小于 `bsz` 的档位，超过最大档返回 `None`（不可选），与 C++ 的 `cur_bsz_idx == num_bsz_selections` 时直接返回一致。

**为什么贪心是对的？** 只在合法前缀里选的前提下：同一请求的累乘接受率单调不增，按降序取前缀时同一请求的 draft 一定从前往后取；对每个固定的 `bsz`，期望 token 数最大的选法就是取接受率最大的 `bsz - n` 个。所以只需要在 `bsz` 这一维上扫一遍。`test_varlen_greedy_is_optimal_over_prefixes` 用暴力枚举所有“每个请求取前 k 个”的组合，验证贪心找到的吞吐等于最优值。`test_varlen_cuts_low_confidence_tokens` 把逐位置接受率 0.95 和 0.3 的两类请求混在一起，验证前者分到的验证长度多于后者。

这个论证依赖“同批所有 token 共享同一步耗时”，即 `step_time` 只依赖总数。DeepInfer 的 `slope_ms * graph_size` 修正也满足这一点。

minidi 的简化：DeepInfer 算总预算和分配 `seqlens_q` 用的是相邻两轮的接受率（为了配合 ZOS），这里用同一组。minidi 的引擎也没有接入 varlen：`Engine` 每步固定验证全部 draft，`varlen.py` 是一个独立可测的算法模块。把两者接起来是一个好练习：需要一个会输出接受率的草稿模型（比如用草稿 softmax 的最大概率当 confidence）。
