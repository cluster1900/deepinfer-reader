# 第 3 章　调度器：每一步跑谁、跑多少、不够时抢占谁

如果把推理引擎比作一家餐厅的后厨，GPU 是炉灶，KV Cache 是案板空间，那么**调度器（`Scheduler`）就是主厨**。每做完一锅菜，主厨都要在几微秒内拍板三个问题：

1. **这一锅是切新菜（Prefill）还是接着炒桌上的菜（Decode）？**
2. **如果是切新菜，一次切多少？案板摆不下怎么办？**
3. **如果是接着炒，炒到一半发现案板不够放新配料了，先把哪一桌的菜撤下去（抢占 Preemption）？**

DeepInfer 的调度器完全用 C++ 写在 `csrc/engine/scheduler.hpp`（1,329 行）里。这一章我们顺着它的核心入口 `schedule()`，把这三个问题背后的设计取舍和源码细节彻底讲透。

---

## 3.1 三个队列：请求在调度器里住哪儿？

打开 `scheduler.hpp` L1124–L1126，调度器手里攥着三个队列：

| 队列名 | C++ 容器类型 | 里面住着什么样的请求？ |
| --- | --- | --- |
| `waiting` | `std::deque<SequencePtr>` | 还没开始 prefill 的新请求、分块 prefill 只跑了一半的请求、以及 decode 时被抢占打回重算的请求 |
| `running` | `std::vector<SequencePtr>` | 已经跑完 prefill、正在一步步 decode 生成新 token 的活跃请求 |
| `prompted` | `std::deque<tuple<SequencePtr, bool>>` | **已经有现成前缀 KV**、无需本地 prefill 就能直接进 decode 的请求（布尔值标记是否为 `fork_seq` 克隆出来的请求） |

前两个队列大家都很熟悉，第三个 `prompted` 队列则是专门为两个场景准备的“绿色通道”：

- **线上 PD 分离**：当 Prefill 实例（PE）算完一个请求的 prompt 后，把 KV Cache 和首 token 传给 Decode 实例（CE），CE 直接把请求塞进 `prompted`，下一步就能进 `running` 开始解码。
- **本地前缀克隆（`fork_seq`，L1062–L1122）**：在压测或并行采样时，可以先让一条母请求跑完长 prompt，再调用 `fork_seq` 零拷贝克隆出几十条子请求。子请求通过 `pin_blocks` 共享母请求的前缀 KV 块（引用计数加 1），塞进 `prompted`；等到下一步真正进入 decode 时，只把最后一个未写满的块复制一份（Copy-on-Write）。

---

## 3.2 铁律一：Prefill 步与 Decode 步绝不混跑

当前主流的开源引擎（如 vLLM V1）默认采用“混合批处理”：把一个长 prompt 的切块（chunk）和几十个正在 decode 的单 token 请求拼进同一个批次里一起做前向。

但如果你看 DeepInfer 的 `_schedule()`（`scheduler.hpp` L275–L345），会发现它立下了一条截然不同的铁律：**每一步要么全组只做 Prefill（`prompt_run = true`），要么全组只做 Decode（`prompt_run = false`），绝不混在同一锅里！**

为什么 DeepInfer 要严格分开？原因有两个：

1. **算子与 CUDA Graph 形态完全不同**：Decode 步每个请求只有 $1 \sim 6$ 个 query token，跑的是高度定制的 CUDA Graph 与专门针对短 query 优化的 FlashMLA decode 核；而 Prefill 步有几千个 query token，跑的是计算密集型大矩阵乘与 YoCo 跨层缓存。分开跑能让两边的算子各自达到最高效率，互不拖累。
2. **多卡 DP 跑 MoE 必须步调一致**：在开启专家并行（EP）时，不同 DP rank 上的请求在经过每一层 MoE 时都要通过 `DeepEP` 做全网 ऑल-to-all（dispatch / combine）通信。如果 Rank 0 在跑耗时的 Prefill，而 Rank 1 在跑极快的 Decode 且走了 CUDA Graph，两边就会在集合通信处错位或严重空等。

因此，在离线推理使用的 `NORMAL` 引擎模式下，**所有 DP rank 必须每隔固定步数通过 `allreduce` 投票，共同决定下一步是集体做 Prefill 还是集体做 Decode**：

```{.mermaid caption="图 3-1　NORMAL 引擎的多卡步型决策与两轮投票（scheduler.hpp L295–L344）"}
flowchart TB
  subgraph Check["第一阶段：检查投票时机与本卡意愿"]
    direction LR
    S0["检查 decode_step 是否落在<br/>ALL(70) 或 ANY(100) 点"] --> W["计算本卡意愿 want：<br/>KV 使用率 ≤ 0.9 且 waiting 可调度"] --> V1{"第一轮 allreduce 投票<br/>ALL 需全员 want / ANY 需任一 want"}
  end

  subgraph Vote2["第二阶段：试排与第二轮确认（未通过或全空则走 Decode 步）"]
    direction LR
    SP["通过首轮后：want 为真的卡<br/>执行 _schedule_prefill() 试排"] --> V2{"第二轮 allreduce<br/>至少有一张卡排出请求？"}
    V2 -- "是" --> R_YES["全组进入 Prefill 步（无活卡跑空步陪跑）<br/>并将全组 decode_step 重置为 0"]
    V2 -- "否" --> D_RUN["全组回退进入 Decode 步<br/>执行 _schedule_decode()，decode_step += 1"]
  end

  Check --> Vote2
```

对照图 3-1，你会发现这套投票机制设计得非常周密（常量定义在 `deepinfer/environ.py` L174–L177）：

- **为什么设 70 步的 `ALL` 点和 100 步的 `ANY` 点？** 正在 decode 的请求对步间延迟（TBT）极其敏感。平时每跑 70 步 decode（`DEEPINFER_PREFILL_ATTEMPT_INTERVAL = 70`），如果所有 DP 卡手头都有新请求等着（全员 `want`），大家就一起切过去做 prefill。万一负载不均，只有某一张卡来了新请求，凑不齐“全员想做”，那么到了第 100 步（`DEEPINFER_PREFILL_ANY_ATTEMPT_INTERVAL = 100`），`ANY` 点就会强制放行，让其他卡跑一个空 prefill 步陪它把通信对齐。
- **为什么 Prefill 一旦开始就能连续做完？** 只要跑了一次 prefill 步，`schedule()`（L676–L677）就会把 `decode_step` 清零。下一步 `decode_step == 0` 恰好又命中 `0 % 70 == 0` 的 `ALL` 点！只要等待队列还没清空、KV 块使用率没超过 90% 水位线（`GPU_CACHE_PREFILL_WATERMARK = 0.9f`），引擎就会一鼓作气把积压的 prefill 连续跑完。
- **为什么还要第二轮投票？** 设想一种边界情况：某张卡 `waiting` 非空（所以第一轮投了赞成票 `want = true`），但进到 `_schedule_prefill` 里一查，发现 `running` 槽位已经满员（达到了 `max_num_seqs`），实际上一个请求也排不出来。如果没有第二轮投票确认“确实有卡排出了非空批次”，全组就会白跑一个空的 prefill 步，还把 `decode_step` 重置成 0，导致下一步又投票、又空跑，陷入死循环（详见 L328–L333 注释）。

---

## 3.3 Prefill 调度：严格先来先服务与分块截断

当投票决定做 Prefill 后，`_schedule_prefill`（`scheduler.hpp` L351–L554）开始从 `waiting` 队列头部取请求。它的核心原则只有四个字：**严格 FCFS（先来先服务）**——只要队头请求因为任何资源不足卡住了，循环立刻 `break`，**绝不越过队头去挑后面能放下的小请求**，从而彻底杜绝大请求被饿死。

对队头请求 `seq` 的完整准入与切块流程如下：

```{.mermaid caption="图 3-2　_schedule_prefill 对队头请求的四步准入与分块流程（scheduler.hpp L362–L550）"}
flowchart TB
  subgraph Row1["第一阶段：队头清理、就绪与显存容量检查（不满足 LATER 时立即 break 保持 FCFS）"]
    direction LR
    A1["1. 取 waiting 队头 seq<br/>若已取消或超长则移入 ignored"] --> A2["2. 检查多模态就绪 schedulable<br/>及 _can_allocate_with_state"] --> A3["3. 若返回 LATER 则 break 停下；<br/>若返回 OK 则进入预算切分"]
  end

  subgraph Row2["第二阶段：批次预算切分、块一次性分配与收尾"]
    direction LR
    B1["4. 校验并发数 ≤ max_num_seqs<br/>且累计 seqlen_k 未超上限"] --> B2["5. 计算本步 seqlen_q：<br/>按剩余预算、2×cp 对齐并避开图片中间"] --> B3["6. 首次调度一次性拿齐全部块；<br/>弹出队头，若切了 chunk 则收手"]
  end

  Row1 --> Row2
```

读这段代码时，有四个非常关键的工程细节：

1. **分块计算，但一次性拿齐全部显存块**：哪怕一个长度为 8,000 的长 prompt 被切成了好几个 chunk 分步计算，在它跑**第一个 chunk** 时，`acquire.allocate_seq(*seq)`（L518–L520）就会一次性把它整个 8,000 长度所需的全部 KV 块和 State 槽全部申请到手！后续跑第 2、3 个 chunk 时不再申请新块。这样做的好处是：**一个请求只要开始了第一个 chunk，就绝不会在后续 chunk 中因为显存块不够而半途卡死**。
2. **双预算上限控制计算耗时**：一步 prefill 不仅限制本步新算的 query token 总数 `max_num_batched_tokens`，还通过 `prefill_max_sum_seqlen_k`（L358、L444–L449）限制本步所有请求参与注意力的上下文总长 $\sum \text{seqlen\_k}$，防止长序列最后一个 chunk 的注意力计算时间失控。
3. **CP 镜像对齐与多模态防切裂**：开启上下文并行（CP）时，分块长度按 `2 * cp_size` 向下对齐（L470–L473），因为 DeepInfer 用的是首尾对称切分的 Mirror CP（第 7 章）；如果是多模态请求，`_adjust_chunk_size`（L235–L260）还会微调切分点，确保绝不把同一张图片的 `IMAGE_PAD` token 序列劈成两半。
4. **未跑完的 chunk 怎么续上？** 如果一个请求被切了 chunk（`is_chunked == true`），本步跑完后在 `process_outputs` 里发现 `cache.num_tokens < seq.get_len()`，就会把它重新塞回 `waiting` **队头**（`waiting.push_front`，L950），保证下一步 prefill 紧接着把剩下的半截跑完。

---

## 3.4 Decode 调度：为什么叫 LCFS？块不够时抢占谁？

当进入 `_schedule_decode`（`scheduler.hpp` L559–L660）时，调度器先把 `running` 里已经结束的请求摘出来，再把 `prompted` 队列里的新请求并入 `running`。

接下来就是最精彩的部分：此时 `running` 里的每一条请求都要往前走一步（生成 1 个主 token 外加验证最多 5 个草稿 token）。如果某条请求当前最后一个 KV 块已经写满了，它就需要向 `BlockManager` 再申请 1 个新块。**万一空闲块池已经见底，不够分给所有人，该保谁、抢占谁？**

我们直接看 `scheduler.hpp` L622–L647 的核心双指针算法：

```cpp
// LCFS (scheduler.hpp L622-L647 核心逻辑提炼)
std::stable_sort(running.begin(), running.end(),
                 [](const SequencePtr& a, const SequencePtr& b) {
                     return a->get_output_len() > b->get_output_len();
                 });
size_t l = 0, r = running.size();
while (l < r && scheduled.size() < max_num_seqs) {
    SequencePtr seq = running[r - 1];                  // 从右端（输出最短）开始取
    while (!_can_append_slot(*seq) && r - l > 1) {
        _preempt(running[l], ...);                     // 块不够时，从左端（输出最长）开始抢占！
        l++;
    }
    if (_can_append_slot(*seq)) {
        _append_slot(*seq, ...);
        scheduled.push_back(seq);
    } else {
        _preempt(seq, ...);
    }
    r--;
}
```

把这段代码画成双指针图，逻辑一目了然：

```{.mermaid caption="图 3-3　Decode 调度的双指针机制：右指针优先保新请求（LCFS），左指针优先抢占长输出大户"}
flowchart TB
  subgraph Sorted["1. running 队列按已生成长度 output_len 降序排列"]
    direction LR
    S0["running[0]：out=900<br/>（最早入 decode，占块最多）"] --- S1["running[1]<br/>out=420"] --- S2["running[2]<br/>out=130"] --- S3["running[3]<br/>out=12"] --- S4["running[4]：out=3<br/>（最晚入 decode，占块最少）"]
  end

  subgraph Ptrs["2. 左右双指针相向扫描（while l < r）"]
    direction LR
    LP["左指针 l（从 0 向右移）：<br/>当空闲块不够时，优先 _preempt(running[l])<br/>杀掉 1 个长输出大户即可腾出大量 KV 块"] --- RP["右指针 r-1（从末尾向左移）：<br/>每轮优先选 running[r-1] 执行 _append_slot<br/>即 LCFS（后进 decode 的短请求优先排入）"]
  end

  Sorted --> Ptrs
```

很多第一次读这段代码的同学都会问两个问题：

### 问题 1：为什么代码第 622 行注释写着 `// LCFS`（后到先服务）？
你看图 3-3 右边的指针 `r - 1`：`running` 按已生成的 token 数 `output_len` 从大到小排好后，谁的 `output_len` 最小？当然是**刚刚完成 prefill、最新进入 decode 阶段的请求**（比如 `output_len = 3`）！右指针 `r - 1` 从队列最右端开始往左取请求排入本步批次，意味着**最晚进入 decode 的请求最先获得块分配和调度机会**——这就是地地道道的 **LCFS（Last-Come, First-Served）**。

### 问题 2：为什么块不够时，左指针 `l` 从 `running[0]`（输出最长的请求）开始抢占？
注意这里函数头有一行过时的旧注释（L556 写着“抢占 output 最短的 seq”），但真实代码（L625 `a->get_output_len() > b->get_output_len()` 配合 `running[l]`）以及 `sequence.hpp` L56 对 `insufficient_system_resource` 的官方注释说得非常清楚：

> “如果 kvcache 即将溢出，则**占用 kvcache 最多的任务，会被我们强制终止**。”

为什么在线上服务里要优先抢占（甚至直接终止）输出最长的请求？算一笔账就明白了：如果去抢占右边刚生成了 3 个 token 的新请求，杀掉它只能腾出寥寥几个块，可能连杀十几个请求都凑不够本步所需的空闲块；而左边 `output_len = 900` 的请求自己就占着十几个块，**抢占它一个，释放出来的块就能救活右边几十个正在 decode 的短请求**，让系统的整体完成率与吞吐量最大化！

### 抢占之后怎么处理？（`_preempt`，L766–L790）
- **线上服务模式（`stop_preempted_request = true`）** 和 **所有带结构化输出（SO）的请求**：直接标为 `FINISHED_PREEMPTED`，调用 `_free_seq` 释放全部 KV 块与 State 槽，不再重算（因为线上 CE 实例根本没有 prefill 通道，且结构化输出的语法状态机不支持回滚）。
- **离线批处理模式（`stop_preempted_request = false`）**：释放该请求占用的全部 KV 块与 State 槽，调用 `reset_for_recompute()` 把 `cache.num_tokens` 清零（但保留已经生成在 `input.tokens` 里的所有 token），状态改回 `WAITING` 并塞回 `waiting` 队头，等下一轮 prefill 把整段序列重新算一遍。

---

## 3.5 `_append_slot` 的隐藏细节：草稿被拒后如何退还多余的块？

在 `_schedule_decode` 给请求分配新块的函数 `_append_slot`（`scheduler.hpp` L817–L828）里，藏着一段很容易被忽略、但对投机解码至关重要的三分支逻辑：

```cpp
void _append_slot(Sequence& seq, std::vector<std::pair<int, int>>& clone_blocks) {
    const size_t num_blocks = seq.get_num_blocks();
    auto& block_table = seq.cache.block_table;
    if (block_table.size() < num_blocks) {
        // 分支 1：块不够了，追加申请新块
        const size_t num_new_blocks = num_blocks - block_table.size();
        for (size_t i = 0; i < num_new_blocks; ++i)
            block_table.push_back(block_manager.allocate());
    } else if (block_table.size() > num_blocks) {
        // 分支 2：手里握着的块比实际需要的多！退还多余的尾块
        std::vector<int> redundant_blocks(block_table.begin() + num_blocks, block_table.end());
        block_table.resize(num_blocks);
        block_manager.free_blocks_nolock(redundant_blocks);
    }
}
```

请想一想：**一个正在往前生成的请求，序列只会越来越长，怎么会出现 `block_table.size() > num_blocks`（手里的块比需要的还多）呢？**

答案正是**投机解码 + 变长验证**！假设 `block_size = 64`：
1. 在第 $N$ 步调度时，某条请求已经有 62 个 token，外加 5 个候选草稿 token，总需求是 $62 + 5 = 67$ 个位置，跨过了 64 的边界，于是 `_append_slot` 为它分配了**第 2 个物理块**（`block_table.size() == 2`）。
2. 结果第 $N$ 步跑完验证，5 个草稿全部被拒绝！序列只增加了 1 个主采样 token，长度变成 $63$。
3. 到了第 $N+1$ 步，如果变长验证调度器决定本步不为它验证任何草稿（或者草稿数较少，总需求 $\le 64$），那么此刻 `seq.get_num_blocks()` 算出来只需要 **1 个块**！
4. 这时 `block_table.size() (2) > num_blocks (1)` 分支就会触发，把上一步为了草稿预留、但最终没用上的第 2 个物理块立刻退还给空闲池，绝不占着茅坑不拉屎。我们在 `minidi/scheduler.py`（L144–L146）中也完整复现了这一回收分支。

---

## 3.6 本章小结

把 DeepInfer 调度器的核心设计浓缩成一张对照表：

| 调度维度 | DeepInfer 的机制选择 | 工程收益与取舍 |
| --- | --- | --- |
| **步型组织** | Prefill 步与 Decode 步严格分离，多卡通过 `ALL(70)` / `ANY(100)` 两轮投票同步 | Decode 算子与 CUDA Graph 零干扰，MoE EP 通信严格对齐；代价是新请求在 `NORMAL` 下最多等 70 步 |
| **Prefill 准入** | 严格 FCFS（队头放不下就 `break`），首个 chunk 一次性分配全长 KV 块与 State 槽 | 彻底杜绝大请求饥饿与半截 chunk 显存死锁；代价是可能有队头阻塞 |
| **Decode 排入与抢占** | 按 `output_len` 降序排列，右端（最短输出，LCFS）优先拿块，左端（最长输出）优先被抢占 | 抢占一个长序列即可释放大量物理块救活大批短序列，最大化系统吞吐 |
| **块动态伸缩** | `_append_slot` 同时处理 `< num_blocks`（扩容）与 `> num_blocks`（缩容退块） | 精准回收投机草稿被拒后多预留的尾块，防止显存泄漏 |

搞清楚了主厨是怎么排单的，下一章我们就走进后厨的仓库，看看 **分页 KV 池** 和 **定长 State 池** 这两套显存到底是怎么算出来的。
