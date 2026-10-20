# 第 5 章　零开销异步调度（ZOS）：如何让 GPU 一刻不停？

在传统的同步推理引擎里，每一步 Decode 都是一场“你等我、我等你”的接力赛：

$$\text{CPU 调度第 } N \text{ 步} \longrightarrow \text{GPU 前向 + 采样第 } N \text{ 步} \longrightarrow \text{CPU 同步等待并处理第 } N \text{ 步输出} \longrightarrow \text{CPU 调度第 } N+1 \text{ 步}$$

当模型跑在 SM100 这样的大算力卡上、又开启了 CUDA Graph 时，GPU 跑完一步 Decode 往往只需要几毫秒到十几毫秒。可是在 GPU 算完第 $N$ 步之后、收到第 $N+1$ 步指令之前，它必须干等着 CPU 做完“拷回采样结果 → 更新块表与状态机 → 选出下一批请求 → 准备输入张量”这一整套动作。哪怕 CPU 只花 3 毫秒，GPU 也有 20%~30% 的时间在空转！

为了彻底消灭这段步间气泡，DeepInfer 实现了 **零开销异步调度（Zero-Overhead Scheduling, 简称 ZOS）**。这一章我们就来拆解：当 CPU 在**还不知道第 $N$ 步采样结果**的情况下提前去调度第 $N+1$ 步时，引起的三个连锁难题是怎么被逐一化解的。

---

## 5.1 直观对比：同步调度 vs 异步调度（ZOS）

我们先用 `minidi/zos.py` 里的流水线模拟器 `simulate_timeline`，画出同步模式与 ZOS 模式下 CPU 和 GPU 的工作时间线（设 `schedule=2, launch=1, forward=6, sample=1, post=2`）：

![图 5-1　同步调度与异步调度（ZOS）的 CPU / GPU 时间线对比（由 minidi/zos.py 生成）](images/plots/ch05_zos_timeline.png)

看图 5-1 的下半部分（异步调度 ZOS）：

- 当 GPU 正在埋头苦算 **第 $N$ 步的 `forward` 和 `sample`** 时，CPU 根本不闲着，立刻开始做 **第 $N+1$ 步的 `schedule(N+1)`**，并把 **第 $N+1$ 步的 `forward(N+1)`** 直接发射（launch）到 GPU 流上排队！
- 发射完 `forward(N+1)` 之后，CPU 这才回过头来等 **第 $N$ 步的采样完成事件**，执行 **第 $N$ 步的账本结算 `post(N)`**，最后再发射 **第 $N+1$ 步的 `sample(N+1)`**。
- 如此一来，GPU 流上的 `forward(0) → sample(0) → forward(1) → sample(1) ...` 像无缝焊接的火车车厢一样首尾相连，CPU 的调度与后处理耗时被 100% 藏在了 GPU 计算的阴影之下！

---

## 5.2 核心挑战：“盲人摸象”式的超前调度

把 `schedule(N+1)` 提前到 `sample(N)` 完成之前执行，听起来很美，但 CPU 在调度第 $N+1$ 步的那一刻，就像是在“盲人摸象”——因为第 $N$ 步的结果还在 GPU 肚子里面，CPU 面临三个棘手的未知数：

1. **未知数一（块表怎么预留？）**：第 $N$ 步还在 GPU 上跑，它的 KV 计数还没在 `process_outputs(N)` 里落账。如果 CPU 按旧长度算块数，第 $N+1$ 步就会少算块，导致 GPU 写越界！
2. **未知数二（输入 Token 与真实长度 `seqlens_k` 在哪儿？）**：第 $N+1$ 步每个请求的输入 token 正是第 $N$ 步刚刚采样出来的 token；如果开启了投机解码，第 $N$ 步到底接受了几个草稿 token（决定了第 $N+1$ 步真实的上下文长度 `seqlens_k`）也只有 GPU 知道！
3. **未知数三（万一第 $N$ 步已经采到了 EOS 怎么办？）**：如果某条请求在第 $N$ 步其实已经生成了结束符 `EOS`，但 CPU 在调度第 $N+1$ 步时还不知道，把它继续排进了第 $N+1$ 步，多算的这一步怎么收场？

DeepInfer 用一套跨越 C++ 调度器、Worker 流水线和 TileLang GPU 算子的组合拳，干净利落地解决了这三个问题。我们先看 `Worker.execute_model_step`（`deepinfer/worker/worker.py` L89–L189）的四阶段流水线全貌：

```{.mermaid caption="图 5-2　Worker.execute_model_step 的四阶段异步流水线（deepinfer/worker/worker.py L89–L189）"}
sequenceDiagram
  participant E as LLMEngine / Scheduler (CPU)
  participant W as Worker (CPU)
  participant G as GPU 计算流
  Note over E: 0. schedule(N+1)：带着第 N 步的在途计数超前调度
  E->>W: 进入 execute_model_step(N+1)
  W->>G: 阶段 1：_correct_seqlens_qk + launch forward(N+1)<br/>（GPU 算子直接从第 N 步输出显存读取输入 token 并修正 seqlens_k）
  Note over G: GPU 紧接着 sample(N) 立即执行 forward(N+1)
  W->>G: 阶段 2：等待 sample_end_evt(N) 完成
  G-->>W: 拷回第 N 步采样与草稿结果
  W->>E: 调用 post_process(N) 结算第 N 步账本、扣减在途计数、判断 EOS
  W->>G: 阶段 3：launch sample(N+1)，随后 CPU 执行 pre_sampler_sync()
  Note over W: 阶段 4：若是 Decode 步，将第 N+1 步状态存入 prev_step_state 留待下一步阶段 2 处理；<br/>若是 Prefill 步，当场执行 post_process(N+1)
```

> [!IMPORTANT]
> **铁律：`forbid_cuda_sync()` 守卫**
> 在图 5-2 的阶段 1（发射 `forward`）和阶段 3（发射 `sample`）中，`ModelRunner` 用上下文管理器 `with forbid_cuda_sync():`（`deepinfer/utils.py` L119–L132）包裹了整个调用链，将 PyTorch 的同步调试模式设为 `"error"`。只要任何一行代码不小心写了 `.item()`、`.cpu()` 或触发了隐式 CPU-GPU 同步，引擎就会当场抛异常报错！这保证了除了阶段 2 显式的 `sample_end_evt.synchronize()` 之外，CPU 发射流绝不被阻塞。

下面我们逐一拆解三大未知数的解法。

---

## 5.3 解法一：双变量在途计数（`total_inflight_tokens` 与 `last_inflight_tokens`）

为了解决“未知数一”（第 $N$ 步还没落账时怎么给第 $N+1$ 步算块数和长度），`Sequence::ScheduleState`（`csrc/engine/sequence.hpp` L204–L256）维护了两个在途计数器：

```cpp
// csrc/engine/sequence.hpp L241-L256
void start_inflight_step(int token_count) {     // 在每次 schedule() 结束时调用
    total_inflight_tokens += token_count;
    last_inflight_tokens = token_count;
}
void finish_inflight_step() {                   // 在每次 process_outputs() 开头调用
    if (total_inflight_tokens == last_inflight_tokens) {
        total_inflight_tokens = 0;
        last_inflight_tokens = 0;
    } else {
        total_inflight_tokens = last_inflight_tokens;
    }
}
```

很多读者看到这段代码会问：**为什么不能只用一个 `inflight_tokens` 变量，非要搞出 `total_inflight_tokens` 和 `last_inflight_tokens` 两个变量？**

跟着图 5-2 的时间线走一遍就明白了——因为在 `schedule(N+1)` 结束之后、`post_process(N)` 执行之前的那一小段时间里，**第 $N$ 步和第 $N+1$ 步同时处于“在途（in-flight）”状态**！

我们设单步最大前向 token 数为 $k = 1 + \text{nextn} = 6$，看这两个数字在每一步是怎么跳动的：

1. **第 $N$ 步 `schedule(N)` 结束时**：调用 `start_inflight_step(6)`。此时只有第 $N$ 步在途：`total = 6`，`last = 6`。
2. **第 $N+1$ 步 `schedule(N+1)` 期间**：此时第 $N$ 步还没做 `post_process(N)`！当 `_schedule_decode` 检查第 $N+1$ 步需要多少个物理块时，`seq.get_num_blocks()`（`sequence.hpp` L292）计算的是：
   $$\text{总预留位置数} = \text{get\_len()} + \text{total\_inflight\_tokens}(6) + \text{draft\_tokens}(5) = \text{get\_len()} + 11$$
   它把第 $N$ 步最多可能新增的 6 个在途 token 和第 $N+1$ 步的 5 个草稿全部算了进去，确保物理块表绝对够用！
3. **第 $N+1$ 步 `schedule(N+1)` 结束时**：再次调用 `start_inflight_step(6)`。此时第 $N$ 步和第 $N+1$ 步都在途：`total` 累加变成 $6 + 6 = 12$，而 `last` 记录最新这一步的 $6$。
4. **第 $N+1$ 步的阶段 2 执行 `post_process(N)` 时**：一进门先调 `finish_inflight_step()`。因为发现 `total (12) != last (6)`，说明后面还有第 $N+1$ 步在途，于是令 `total = last (6)`——正好把第 $N$ 步的在途账销掉，只留下第 $N+1$ 步的 6 个在途 token！

---

## 5.4 解法二：GPU 端直接接线（`remap`）与 `compute_seqlens_k_zos` 长度修正

解决了块表预留，接下来解决“未知数二”：发射 `forward(N+1)` 时，CPU 手里既没有第 $N$ 步刚采出来的输入 token，也不知道第 $N$ 步到底接受了几个草稿 token。怎么办？

答案是：**CPU 只负责搭桥（生成下标映射 `remap`），真正填输入 token 和修正 `seqlens_k` 的活全部交给 GPU 算子自己干！**

在图 5-2 的阶段 1，`Worker._correct_seqlens_qk`（`worker.py` L191–L246）调用 C++ 函数 `prepare_zos_inputs`（`csrc/engine/zos.cpp` L142 起），对比本步（第 $N+1$ 步）与上一步（第 $N$ 步）的请求列表，生成映射数组 `remap`：

- 如果本步第 $i$ 个请求在上一步也是活跃请求（下标为 $j$），则 `remap[i] = j`；
- 如果本步第 $i$ 个请求是刚从 `prompted` 进来的新请求（上一步没跑 decode），则 `remap[i] = -1`。

有了 `remap[i] = j`，GPU 在启动 `forward(N+1)` 前做两件事：

### 1. 输入 Token 直接在 GPU 显存内搬运
对于 `remap[i] == -1` 的新请求，CPU 直接把它的末位 token 填进 `input_ids`；而对于 `remap[i] = j >= 0` 的老请求，GPU 算子直接根据下标 $j$，去第 $N$ 步采样器留在显存里的输出张量 `prev_sampled_token_ids[j]` 和草稿张量 `prev_predict_token_ids[j]` 中把 token 抓过来填进第 $N+1$ 步的输入槽位，全程不经过 CPU！

### 2. 上下文长度 `seqlens_k` 的精确公式修正
这一步的数学关系非常漂亮。假设第 $N$ 步开始前，某条请求已确认写入 KV Cache 的长度是 $C = \text{cache.num\_tokens}$：

- 在第 $N$ 步里，它总是能接受 1 个主 token，外加实际接受了 $A_j = \text{prev\_num\_accepted\_tokens}[j]$ 个草稿 token（$0 \le A_j \le \text{nextn}$）。因此，第 $N$ 步真正有效写入 KV Cache 的 token 数是 $1 + A_j$。
- 到了第 $N+1$ 步，假设变长验证算子决定本步验证 $Q_i = \text{seqlens\_q}[i]$ 个 query token。那么第 $N+1$ 步注意力真正应该看到的上下文总长 $\text{seqlens\_k}[i]$ 应该是：
  $$\text{seqlens\_k}^{\text{真实}}[i] = C + (1 + A_j) + Q_i$$
- 可是 CPU 在调 `prepare_zos_inputs`（`zos.cpp` L188）时，既不知道 $A_j$，也不知道 GPU 重算后的 $Q_i$。于是 CPU 先把 `seqlens_k[i]` 设为一个假设“上一步全部 $\text{nextn}$ 个草稿都被接受、且扣掉本步临时 `seqlen_q`”的基准值：
  $$\text{seqlens\_k}^{\text{CPU初值}}[i] = C + (1 + \text{nextn})$$
- 紧接着，GPU 启动 TileLang 算子 `compute_seqlens_k_zos`（`layers/tilelang_ops/compute_seqlens_k_zos.py` L22–L23），在显存里就地执行一行修正：
  $$\text{seqlens\_k}[i] \gets \text{seqlens\_k}^{\text{CPU初值}}[i] + Q_i - (\text{nextn} - A_j)$$
  把 $\text{seqlens\_k}^{\text{CPU初值}}[i] = C + 1 + \text{nextn}$ 代入上式，$\text{nextn}$ 正好消掉，分毫不差地还原出：
  $$\text{seqlens\_k}[i] = C + (1 + A_j) + Q_i$$

---

## 5.5 解法三：事后作废——白算一步与在途抢占的优雅兜底

最后看“未知数三”：如果第 $N$ 步某条请求已经结束了，或者在第 $N+1$ 步调度时被抢占了，怎么保证状态不错乱？

1. **第 $N$ 步采到 EOS 时的“白算一步作废”**：
   假设请求 `seq` 在第 $N$ 步采样出了 `EOS`。CPU 在 `schedule(N+1)` 时还不知道，照常把它排进了第 $N+1$ 步并送上了 GPU。随后在图 5-2 的阶段 2，`post_process(N)` 拿到了第 $N$ 步的 `EOS`，把 `seq.status` 设为 `FINISHED_STOPPED`。等到了下一步结算第 $N+1$ 步的输出时，`process_outputs(N+1)` 一开头检查到 `seq.is_finished()` 为真（`scheduler.hpp` L905–L915），直接忽略第 $N+1$ 步为它算出的废 token，不会往输出列表里多塞任何一个字符！
2. **第 $N$ 步还在途，却在第 $N+1$ 步被抢占（Preempted In-Flight）**：
   假设请求 `seq` 参与了第 $N$ 步 decode，但在 `schedule(N+1)` 时因为显存块不够，被左指针选中执行了 `_preempt(seq)`！此时 `_preempt` 已经把它的物理块释放、状态改回 `WAITING` 并塞回了 `waiting` 队头。紧接着在阶段 2，第 $N$ 步的 GPU 结果回来了——此时绝对不能再把第 $N$ 步的 token 记到账上！因此，`process_outputs`（`scheduler.hpp` L917–L925）和 `process_MTPInfo`（L1214–L1219）都有一道专门的护栏：
   ```cpp
   // 如果是 decode 步，但请求当前状态已经变成 WAITING，说明它在途期间被抢占了，整步结果直接作废！
   if (!prompt_run && seq.status == SequenceStatus::WAITING) continue;
   ```

我们在第 13 章实现的 `minidi/zos.py`（`ZOSEngine`）完整包含了上述在途计数、`remap` 接线和事后作废逻辑，并通过 20 组随机并发测试验证了 ZOS 引擎与同步引擎生成的序列逐 token 100% 一致。

---

## 5.6 两个配套工程细节：CPU Gloo 通信组与 CUDA Graph 分档

为了让 ZOS 跑得丝滑，DeepInfer 还做了两项关键配套：

1. **开启 ZOS 时，多卡调度投票必须走 CPU Gloo 进程组**：
   还记得第 3 章说的 DP 跨卡 `allreduce` 投票吗？如果那个 `allreduce` 走 GPU 的 NCCL 组，它就必须在 GPU 流上排队，等前面的 `forward` 跑完才能拿到投票结果——这等于在 `schedule(N+1)` 里强行插入了一次 CPU-GPU 同步，异步重叠当场破功！因此，`llm_engine.py`（L313、L665–L679）在开启 ZOS 时，专门把调度器的通信组切换为 **CPU 上的 `gloo` 进程组**（`PG.cpu_data_parallel_group()`），让多卡 CPU 直接通过主机网络完成微秒级投票，完全不碰 GPU 流。
2. **CUDA Graph 离散档位（`build_graph_batch_sizes`）**：
   为了消除 Python 逐算子发射的开销，`cuda_graph_runner.py`（L41–L52）和 `dflash_cuda_graph_runner.py` 按本步总 query token 数 `total_q = sum(seqlens_q)` 预先捕获了一系列离散档位的计算图：先是小于 `interval`（默认 8）的 2 的幂 `[1, 2, 4]`，随后每隔 8 一个档位 `[8, 16, 24, 32, ..., max_size]`。运行时 `select(total_q)` 用二分查找挑出第一个 `>= total_q` 的档位图，多余槽位用 padding 填齐。

请注意这个“每隔 8 个 token 跳一档”的阶梯特性——从 16 个 token 增加到 17 个 token，GPU 耗时就会突然跳升一档！这直接引出了下一章的主题：**如何利用草稿置信度，把宝贵的验证预算刚好卡在最划算的 CUDA Graph 档位上？**
