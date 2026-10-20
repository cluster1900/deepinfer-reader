# 第 3 章　调度器：每一步跑谁、跑多少

DeepInfer 的调度器是一个 C++ 结构体 `Scheduler`，整个定义都在 `csrc/engine/scheduler.hpp`（1,329 行）里。Python 侧只调用它的 `schedule()` 和 `process_outputs()` 两个方法。本章按调用顺序拆开 `_schedule`、`_schedule_prefill`、`_schedule_decode` 和 `schedule`。

## 3.1 三个队列

调度器持有三个容器（L1124–L1126）：

| 容器 | 类型 | 放什么 |
| --- | --- | --- |
| `waiting` | `std::deque<SequencePtr>` | 等待 prefill 的请求，包括 chunk 没跑完的、被抢占后要重算的 |
| `running` | `std::vector<SequencePtr>` | 已完成 prefill、正在 decode 的请求 |
| `prompted` | `std::deque<tuple<SequencePtr,bool>>` | 已有 KV、可以直接进 decode 的请求；第二项表示是否 fork 出来的 |

`prompted` 在 PD 分离里是 CE 接收 PE 传来的请求的入口；在开源版的 `NORMAL` 引擎里，它主要被 `fork_seq`（L1062–L1122）使用：从一个已算好前缀的请求复制出新请求，共享前缀 KV 块（`pin_blocks` 增加引用计数），最后一个写了一半的块在被调度时复制一份（copy-on-write）。`examples/benchmark.py` 就靠它先算一个长前缀，再 fork 出几十个请求做 decode 压测。

## 3.2 每一步只做一种事

`_schedule`（L275–L345）按引擎类型分流：`PROMPT` 引擎只做 prefill，`COMPLETION` 引擎只做 decode，`NORMAL` 引擎每一步决定“全组做 prefill”还是“全组做 decode”。**同一步里不会既有 prefill 请求又有 decode 请求**，`SchedulerOutputs.prompt_run` 是一个布尔值，整步共享。

这一点和 vLLM 现在的默认做法相反（vLLM 的 chunked prefill 会把 prefill chunk 和 decode token 拼进同一个批次）。DeepInfer 这样做的原因可以从两处看出来：

1. 线上部署是 PD 分离的，PE 和 CE 本来就只做一种事，`NORMAL` 引擎主要用于离线推理和测试；
2. DP 组内各 rank 之间有 MoE 的 EP 通信，所有 rank 必须同时进入同一种 forward（prefill 和 decode 走不同的 kernel 和 CUDA Graph）。所以步型要在 DP 组内投票统一。

```{.mermaid caption="图 3-1　NORMAL 引擎的步型决策（scheduler.hpp L295–L344）"}
flowchart TB
  A["decode_step（各 rank 一致）"] --> B{"ANY 点？<br/>decode_step≠0 且 %100==0"}
  B -- 否 --> C{"ALL 点？<br/>decode_step % 70 == 0"}
  C -- 否 --> D["decode 步"]
  B -- 是 --> W["want = KV 使用率 ≤ 0.9<br/>且 waiting 队头可调度"]
  C -- 是 --> W
  W --> V{"第一轮 allreduce<br/>ANY：任一 rank want<br/>ALL：全员 want"}
  V -- 否 --> D
  V -- 是 --> P["want 的 rank 执行 _schedule_prefill"]
  P --> V2{"第二轮 allreduce：<br/>有 rank 真排出了 prefill？"}
  V2 -- 否 --> D
  V2 -- 是 --> Q{"本 rank 排出了？"}
  Q -- 是 --> R["prefill 步"]
  Q -- 否 --> S["空 prefill 步（陪跑）"]
```

两个间隔的默认值来自 `deepinfer/environ.py`：`DEEPINFER_PREFILL_ATTEMPT_INTERVAL=70`（L174）、`DEEPINFER_PREFILL_ANY_ATTEMPT_INTERVAL=100`（L177）。水位线是 `GPU_CACHE_PREFILL_WATERMARK = 0.9f`（L80）。

这组规则的效果需要算一下才看得清：

- **新请求最多等 70 个 decode 步**才有机会开始 prefill。因为 `decode_step` 只有落在 70 的倍数上才检查。好处是 decode 的每 token 延迟（TBT）不会被频繁插入的 prefill 打断，代价是首 token 延迟（TTFT）变长。
- **prefill 一旦开始会连着做**。prompt 步结束后 `decode_step` 清零（`schedule()` L676–L677），下一步 `decode_step==0` 又是 ALL 点，只要还有活、使用率不超过 0.9 就继续 prefill。
- **ANY 点是兜底**。假设 DP 组里只有一个 rank 收到了新请求，ALL 点永远凑不齐“全员想做”，它就会在第 100 个 decode 步被 ANY 点放行，其他 rank 跑空 prefill 步陪它。
- **第二轮投票防死循环**。注释（L328–L333）解释得很清楚：want 只看了 waiting 非空，但 `_schedule_prefill` 可能因为 slot 满或块不够什么也排不出来；如果此时硬做一个空 prompt 步，`decode_step` 被清零，下一步又投票、又空手，running 满且 waiting 非空时就永远卡住。

`minidi/vote.py` 用 30 行 Python 复现了这套逻辑，`tests/test_algorithms.py::test_vote_points` 覆盖了五种情形。

## 3.3 prefill 调度：严格先来先服务

`_schedule_prefill`（L351–L554）从 `waiting` 队头一个个往外取，任何一个约束不满足就 `break`，**不会跳过队头去找后面能放下的请求**。这一点很重要：它保证了公平，代价是一个大请求可能挡住后面的小请求（队头阻塞）。

```{.mermaid caption="图 3-2　_schedule_prefill 对队头请求的检查顺序"}
flowchart TB
  A["取 waiting 队头"] --> B{"已被判死？<br/>ABORTED / VE_FAILED / SO_FAILED"}
  B -- 是 --> B1["放进 ignored，继续下一个"]
  B -- 否 --> C{"schedulable？<br/>（多模态还没处理完为 false）"}
  C -- 否 --> Z["停止本轮"]
  C -- 是 --> D{"还没分配过块？"}
  D -- 是 --> D1{"长度 > prompt_limit？"}
  D1 -- 是 --> D2["FINISHED_IGNORED"]
  D1 -- 否 --> D3{"_can_allocate_with_state"}
  D3 -- LATER --> Z
  D3 -- NEVER --> D2
  D3 -- OK --> E
  D -- "否（fork 来的）" --> E["running 数 + 1 ≤ max_num_seqs？"]
  E -- 否 --> Z
  E -- 是 --> F{"sum seqlen_k 预算够？"}
  F -- 否 --> Z
  F -- 是 --> G["确定 seqlen_q：<br/>token 预算截断 → chunk checkpoint → 不切开图片"]
  G --> H{"切了 chunk 且未开 chunked_prefill？"}
  H -- 是 --> Z
  H -- 否 --> I["提交：分配块、置 RUNNING（未切 chunk 时）、<br/>pop 出 waiting、记录 seqlen_q"]
  I --> J{"NORMAL 引擎且切了 chunk？"}
  J -- 是 --> Z
  J -- 否 --> A
```

几处细节：

- **`prompt_limit`**：开了 chunked prefill 时是 `max_model_len`，否则是 `min(max_model_len, max_num_batched_tokens)`（L112）。不开 chunked prefill 时，一个 prompt 必须一步放得下。
- **块是一次性分配到 prompt 全长的**。`acquire.allocate_seq(*seq)` 在第一个 chunk 就把整个 prompt 需要的块要齐（L518–L520），后续 chunk 不再申请。所以 `_can_allocate_with_state(seq->get_num_blocks())` 检查的也是全长。
- **`prefill_max_sum_seqlen_k`**：一步 prefill 里所有请求的 `seqlen_k` 之和的上限（L358、L444–L449），控制注意力的总计算量。构造函数要求它不小于 `prompt_limit`（L120–L121）。
- **CP 对齐**：开 CP 时 token 预算按 `cp_align = cp_size*2` 向下取整（L470–L473），“*2 是 mirror cp”（L1175 的注释），即每个 CP rank 拿首尾对称的两段，平衡因果注意力的计算量。
- **图片不切开**：`_adjust_chunk_size`（L235–L260）保证 chunk 边界不落在一张图片的 `IMAGE_PAD` 序列中间。
- **NORMAL 引擎切了 chunk 就收手**（L543–L545）。被切的请求本步结束后在 `process_outputs` 里被放回 `waiting` 队头，下一步接着做。

## 3.4 decode 调度：从短的开始排，从长的开始杀

`_schedule_decode`（L559–L660）做四件事：

1. `free_finished_seqs`：把 `running` 里已结束的请求摘出来，放进 `finished_seqs`，块的释放推迟到 `pre_sampler_sync` 里的 `finalize_seq`；
2. 清理 `waiting` 里已被判死、但因为本 rank 一直没走 prefill 而滞留的请求（L564–L577 的注释讲了不清理会怎样拖死全组）；
3. 把 `prompted` 队头的请求移入 `running`，fork 来的要先复制 state 和最后一个块；
4. 选出本步 decode 的请求，块不够就抢占。

第 4 步的代码不长，原样翻成伪代码：

```text
sort(running, 按 output_len 降序)        # running[0] 输出最长
l, r = 0, len(running)
while l < r and len(scheduled) < max_num_seqs:
    seq = running[r-1]                    # 从输出最短的开始
    while 没有空闲块 and r - l > 1:
        _preempt(running[l]); l += 1      # 从输出最长的开始抢占
    if 有空闲块: _append_slot(seq); scheduled.append(seq)
    else:        _preempt(seq)
    r -= 1
```

```{.mermaid caption="图 3-3　decode 选择的双指针：右边排入，左边抢占"}
flowchart LR
  subgraph running["running（按 output_len 降序）"]
    direction LR
    a["out=900"] --- b["out=420"] --- c["out=130"] --- d["out=12"] --- e["out=3"]
  end
  L["l 指针：块不够时<br/>从这里开始 _preempt"] --> a
  R["r 指针：从这里开始<br/>_append_slot 并排入"] --> e
```

这里有一处**代码注释与代码行为不一致**，读的时候要以代码为准：函数头注释（L556–L557）写的是“block 不足时抢占 output 最短的 seq”，但比较函数是 `a->get_output_len() > b->get_output_len()`（L625–L626），排序后 `running[0]` 是输出**最长**的，而被抢占的是 `running[l]`，`l` 从 0 开始。所以实际被抢占的是输出最长的请求。`sequence.hpp` L56 对 `insufficient_system_resource` 的注释是“如果 kvcache 即将溢出，则占用 kvcache 最多的任务，会被我们强制终止”，和代码行为方向一致（注意按输出长度排序不完全等于按 KV 占用排序，prompt 长度没算进去）。`minidi` 照代码行为实现，`tests/test_scheduler.py::test_lcfs_preempts_longest_output_first` 验证了这一点。

代码注释称之为 `// LCFS`（L622）。抢占后的处理在 `_preempt`（L766–L790）：

- `stop_preempted_request=true`（服务默认）：请求直接结束，状态 `FINISHED_PREEMPTED`，释放块；
- `stop_preempted_request=false`（离线默认）：`reset_for_recompute()` 清空 KV 进度，放回 `waiting` **队头**，下次重新 prefill 整段（包括已经生成的 token）；
- 带结构化输出的请求不支持重算（grammar 状态无法回滚），一律直接结束（L769–L777）；
- `COMPLETION` 引擎没有 prefill 通道，不能重算，所以构造函数强制要求它 `stop_preempted_request=true`（L101–L104）。

为什么服务形态默认“抢占即终止”？抢占说明 KV 已经不够用；重算意味着把一个已经生成了几百上千个 token 的请求从头 prefill 一遍，在 KV 紧张时只会让情况更糟。在 PD 分离架构下，CE 没有 prefill 能力，更不可能重算。

选完之后，`scheduled` 再按总长度降序排一次（L648–L649），然后 `_mtp_varlen.schedule(scheduled, seqlens_q)` 决定每个请求本步验证几个草稿 token（第 6 章）。

## 3.5 `schedule()`：把结果变成元数据

`Scheduler::schedule`（L667–L735）是 Python 实际调用的入口。它在 `_schedule()` 之后为每个请求从 `SequenceMetadataPool` 取一个 `SequenceMetadata`，计算：

- `seqlen_k = cache.num_tokens + total_inflight_tokens + seqlen_q`：异步调度下要把在途 token 也算上（第 5 章）；
- `position_end` 不能超过 `runtime_max_model_len`（L698–L707）：RoPE 表等按位置索引的 buffer 只有这么多行；
- 打开统计时用 `FlopsEstimator` 估算本步注意力和总 TFLOPs，用于日志里的 MFU。

最后对每个请求调 `start_inflight_step(max_seqlen_q)`，把本步的 token 记为“在途”。

## 3.6 本章要点

| 问题 | DeepInfer 的选择 | 代价 |
| --- | --- | --- |
| prefill 与 decode 是否同批 | 不同批，整步统一 | 需要 DP 组投票；NORMAL 引擎的 TTFT 受 70 步间隔影响 |
| waiting 队列顺序 | 严格 FCFS，队头放不下就停 | 队头阻塞 |
| 块何时分配 | prefill 第一个 chunk 就要齐 prompt 全长；decode 每步按需加 | 长 prompt 早早占住块 |
| 抢占谁 | 输出最长的 | 被抢的往往是最“值钱”的请求 |
| 抢占后 | 服务：终止；离线：重算 | 服务端会返回 `insufficient_system_resource` |

第 12 章会在 `minidi` 里一行行实现这些规则，并用一次实际运行画出 prefill 步、decode 步和 KV 使用率随时间的变化。
