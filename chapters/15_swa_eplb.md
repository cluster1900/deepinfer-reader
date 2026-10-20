# 第 15 章　手写 SWA 环形缓冲区、EPLB 与多卡步型投票

在第 14 章里我们看到，当投机解码的草稿被拒绝时，分页 KV 只需要把 `num_cached` 指针往回拨，不需要擦除废数据。

但是，在 **滑动窗口注意力（SWA）** 和 **门控 KV 压缩状态** 使用的**定长环形缓冲区（Ring Buffer）**里，情况就完全不同了！因为在环形缓冲区中，逻辑位置 $p$ 是通过取模写进物理槽位 $p \bmod \text{ring}$ 的——如果环形缓冲区的长度 $\text{ring}$ 刚好等于窗口大小 $W$，那么本步写入的那几个候选草稿 token 就会**直接覆盖掉几格之前的有效历史窗口**！一旦草稿被拒绝，被覆盖掉的历史数据就再也找不回来了！

这一章我们打开 `minidi/swa.py`、`minidi/eplb.py` 和 `minidi/vote.py`，亲手实现这三个核心算法模块，并用代码实测验证环形缓冲区的回退余量公式。

---

## 15.1 SWA 环形缓冲区的回退余量推导与实证（`minidi/swa.py`）

我们先用最严格的不等式，推导一下单步前向与投机回退到底需要多大的环形缓冲区 `ring`：

- 假设滑动窗口大小是 $W$（每个位置 $q$ 的注意力需要读取 $[q - W + 1, \dots, q]$ 共 $W$ 个位置的 KV）。
- 在某一步 Decode 开始前，请求已经确认提交了 $L$ 个 token（位置 $0 \dots L-1$）。
- 本步投机验证一口气从位置 $L$ 开始往环形缓冲区里写入 $k = 1 + \text{nextn}$ 个新位置：$L, L+1, \dots, L+k-1$。
- 注意：**工业界的高性能注意力算子（如 FlashMLA）都是先调用 `store_kv_cache` 把本步全部 $k$ 个新位置的 KV 写进环形缓冲区，然后再启动注意力核并行计算这 $k$ 个位置的查询！**
- 这意味着，排在本步第 1 个的 query（位置 $L$）在算注意力时，它所需要的最老的那个历史位置 **`L - W + 1`**，绝对不能被本步最后写入的第 $k$ 个位置 **`L + k - 1`** 在取模时撞上并覆盖掉！
- 同时，万一本步验证后后面的草稿全部被拒绝（只保留位置 $L$），下一步从位置 $L+1$ 重新开始时，位置 $L+1$ 的窗口起点 **`L - W + 2`** 也同样包含在这段历史里。
- 要让最后写入的位置 $L + k - 1$ 绕一圈回来（即 $L + k - 1 - \text{ring}$）严格落在最老需求位置 $L - W + 1$ 的左边，必须满足：
  $$L + k - 1 - \text{ring} < L - W + 1 \iff \text{ring} > W + k - 2 \iff \text{ring} \ge W + k - 1 = W + \text{nextn}$$

也就是说，**单步前向与投机回退的最小安全回退余量是 $\text{extra} = \text{ring} - W = k - 1 = \text{nextn}$**！

我们在 `minidi/swa.py`（共 44 行）中写了 `SWARing` 类，它在每个槽位里同时记录当前存放的是哪个逻辑位置 `pos_of_slot[slot]`。一旦读取窗口时发现某个槽位里的逻辑位置已经被新数据覆盖，就立即抛出 `RuntimeError`。我们用 $W = 4, k = 3$（即 $1$ 个主 token + $2$ 个草稿，$\text{nextn} = 2$）、已提交长度 $L = 10$ 跑一次实测对比，结果如图 15-1 所示：

![图 15-1　SWA 环形缓冲区回退余量实测对比（W=4, k=3, L=10）：上图余量 k-1=2（ring=6）时窗口完整保留；下图余量 k-2=1（ring=5）时位置 7 被本步写入的位置 12 覆盖](images/plots/ch15_swa_ring.png)

看图 15-1：
- **上半图（`ring = W + (k - 1) = 4 + 2 = 6`）**：本步写入位置 `10, 11, 12`（橙色，分别落入槽 `4, 5, 0`）。其中位置 `12` 覆盖的是槽 `0` 里早已过期的旧位置 `6`；而位置 `10` 的 query 所需要的窗口 `7, 8, 9, 10`（蓝色槽 `1, 2, 3` 与橙色槽 `4`）完好无损！即便草稿 `11, 12` 随后被全部拒绝，下一步从位置 `11`（槽 `5`）重新写起，历史窗口也毫发无伤。
- **下半图（`ring = W + (k - 2) = 4 + 1 = 5`，仅少 1 个槽位）**：本步写入位置 `10, 11, 12` 时，位置 `12`（`12 % 5 == 2`）一头撞进了**槽 2**（红色），把原本住在槽 2 里的有效历史**位置 7** 给覆盖掉了！此时位置 `10` 要读窗口 `[7, 8, 9, 10]` 当场读出脏数据报错！

### 与 DeepInfer 源码的完整印证
这个推导与我们在第 4.3 节看到的 DeepInfer 真实源码严丝合缝地对上了：
1. **直接取下界 $W + \text{nextn}$ 的两处代码**：
   - DeepInfer 的压缩边界状态 Ring（`get_inference_compress_block_size`，`cache_engine.py` L131–L133 及 `compress.py` L98）计算公式正是 `compress_ratio + DEEPINFER_MAX_BACKTRACKING_TOKENS - 1 = ratio + nextn`；
   - DFlash 草稿模型自己的 SWA 层（`get_kv_index.py` L183）使用的有效窗口槽位数也是 `window_size + dflash_block_size = 128 + 5 = 133`。
2. **主模型 SWA Ring 取 $W + \text{nextn} + 1 = 128 + 6 = 134$ 的原因**：
   正如第 4.3 节所解析，主模型 SWA 除了满足单步前向与回退下界（133 槽）之外，还要支持跨 128 边界时的状态快照落盘（`do_copy_state` & `select_state_slice`）、Prefill 阶段加载严格先前的 128 个历史位置（`make_swa_block_load_plan`）以及 YoCo 预填充尾部保留（`keep_tokens = 134`），因此直接加上了完整的单步最大长度 `DEEPINFER_MAX_BACKTRACKING_TOKENS = nextn + 1 = 6`。

---

## 15.2 60 行手写 EPLB 专家复制与装箱算法（`minidi/eplb.py`）

接下来我们打开 `minidi/eplb.py`，它把第 7.4 节讲过的 `deepinfer/model_executor/eplb/policy.py` 浓缩成了两个极其干净的 NumPy 函数：

```{.mermaid caption="图 15-2　minidi/eplb.py 复刻的两阶段专家负载均衡算法流程"}
flowchart LR
  subgraph Stage1["第一阶段：replicate(tokens, num_physical)<br/>（对应 rebalance_experts_global）"]
    direction TB
    R1["输入历史步负载矩阵 tokens [T, E]<br/>初始每个逻辑专家 1 个副本 (count = 1)"] --> R2["循环 M = num_physical - E 轮：<br/>枚举给哪个逻辑专家 j 增加 1 个副本，<br/>能使 Σ_t max_e (tokens[t,e] / count[e]) 最小"]
    R2 --> R3["生成物理到逻辑映射表 p2l [E + M]<br/>并将各逻辑专家的负载均分给其所有副本"]
  end

  subgraph Stage2["第二阶段：place(tokens, p2l, num_ranks)<br/>（对应 _batch_greedy_indices）"]
    direction TB
    P1["将 E + M 个物理专家副本<br/>按窗口总负载从大到小降序排列"] --> P2["逐个放入尚未装满（容量 (E+M)//R）且<br/>放入后历史步峰值之和 Σ_t max_r load[t,r] 最小的卡"]
    P2 --> P3["输出每张 EP 卡持有的物理副本列表 ranks"]
  end

  Stage1 --> Stage2
```

看 `minidi/eplb.py` L16–L52 的核心代码，连打破平局的微小偏置 `+ np.arange(E) * 1e-4` 都与 DeepInfer 原版保持一致：

```python
def replicate(tokens: np.ndarray, num_physical: int) -> np.ndarray:
    t, e = tokens.shape
     biased = tokens.astype(float) + np.arange(e) * 1e-4
    cnt = np.ones(e, dtype=int)
    for _ in range(num_physical - e):
        # 试算给每个逻辑专家 k 增加 1 个副本后的步峰值之和，挑下降最多的那个！
        best_k, best_score = 0, np.inf
        for k in range(e):
            c = cnt.copy(); c[k] += 1
            score = (biased / c).max(axis=1).sum()
            if score < best_score:
                best_score, best_k = score, k
        cnt[best_k] += 1
    return np.repeat(np.arange(e), cnt)                   # physical_to_logical_map
```

在 `tests/test_algorithms.py::test_eplb_improves_balance` 中，我们对 16 个服从 Zipf 分布的逻辑专家、4 张卡、增加 4 个冗余副本运行测试，验证 EPLB 算出的跨卡不均衡度 `moe_lb` 显著优于朴素均匀放置（即第 7 章图 7-3 的数据来源）。

---

## 15.3 25 行手写多 DP 卡步型两轮投票（`minidi/vote.py`）

最后看 `minidi/vote.py`（共 25 行），它完整复刻了第 3.2 节图 3-1 中 `scheduler.hpp` L296–L344 的跨 DP 卡 `ALL(70)` / `ANY(100)` 两轮投票协议：

```python
def decide(decode_step: int, wants: list[bool], can_schedule: list[bool],
           all_interval: int = 70, any_interval: int = 100):
    n = len(wants)
    is_any = decode_step != 0 and decode_step % any_interval == 0
    is_all = decode_step % all_interval == 0
    if not (is_any or is_all):
        return "decode", ["decode"] * n
    # 第一轮投票：ANY 点只需任一卡想做 prefill，ALL 点需全员想做 prefill
    do_prefill = any(wants) if is_any else all(wants)
    if not do_prefill:
        return "decode", ["decode"] * n
    # 第二轮投票：尝试执行 _schedule_prefill 后，检查是否至少有一张卡真正排出了请求
    got = [w and c for w, c in zip(wants, can_schedule)]
    if not any(got):
        return "decode", ["decode"] * n                   # 全员空手而归，回退到 decode 防止死循环！
    return "prefill", ["prefill" if g else "empty-prefill" for g in got]
```

`tests/test_algorithms.py::test_vote_points` 逐一验证了五种关键场景：普通步直走 Decode、`ALL(70)` 点一票否决、`ANY(100)` 点一票放行且其他卡跑 `empty-prefill` 陪跑、以及第二轮投票全员排不出活时安全回退 Decode。
