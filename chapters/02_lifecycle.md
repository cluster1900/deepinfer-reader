# 第 2 章　一个请求的一生：从 `generate` 到 Token 落账

读一个复杂的系统，最怕一上来就陷进某个局部函数的细节里，忘了它是在什么时机、被谁调用的。

这一章我们不抠算子细节，而是像给包裹装上 GPS 追踪器一样，跟着一次 `llm.generate(["你好"])` 调用，完整走完**引擎启动 → 请求入队 → 单步调度与前向 → 采样与 Token 落账 → 延迟释放**的全生命周期。走完这一趟，后面每一章讨论的模块都能在这张全局路线图上找到自己的坐标。

---

## 2.1 阶段一：引擎启动（`LLM` 与 `LLMEngine.__init__`）

当你写下 `llm = LLM(model="deepseek-ai/DeepSeek-V4.1-Flash-1017")` 时，入口类 `LLM.__init__`（`deepinfer/entrypoints/llm.py` L54–L87）负责组装 `EngineArgs` 并创建核心对象 `LLMEngine`。这里有两个非常关键的默认值差异：

- **在线服务默认“抢占即终止”**：`EngineArgs.stop_preempted_request` 默认是 `True`（`engine/arg_utils.py` L40）。当显存块不够用时，被抢占的请求直接报错终止，返回 `finish_reason = "insufficient_system_resource"`。
- **离线 `LLM` 默认“抢占后重算”**：`LLM.__init__` 用 `kwargs.setdefault("stop_preempted_request", False)`（L68）覆盖了这一行为，让离线批处理在显存吃紧时把请求退回等待队列重新 prefill，保证任务不丢（第 3 章会详解原因）。

随后进入 `LLMEngine.__init__`（`deepinfer/engine/llm_engine.py` L155–L305），它按四个阶段完成初始化：

```{.mermaid caption="图 2-1　LLMEngine 启动的四个阶段（llm_engine.py L155–L305, L416–L458）"}
flowchart TB
  subgraph Row1["第一阶段：分词器、CP 子进程派生与显存容量探测"]
    direction LR
    A1["1. 加载 Tokenizer 与多模态处理器<br/>派生 CP 子进程（rank 0 为 driver）"] --> A2["2. 加载模型权重（load_model）<br/>跑 profile_run 实测前向峰值显存"] --> A3["3. 校验 80% 容量闸门：<br/>max_model_len ≤ 0.8 × 总块容量"]
  end

  subgraph Row2["第二阶段：双池显存分配、计算图预热与 C++ 调度器构建"]
    direction LR
    B1["4. init_cache_engine<br/>分配 GPU KV 池与 State 池"] --> B2["5. warm_up_model：捕获 CUDA Graph<br/>实测变长验证 decode 耗时曲线"] --> B3["6. 创建两套 C++ BlockManager、<br/>Scheduler 与 DeferredRelease"]
  end

  Row1 --> Row2
```

看着图 2-1，记住三个贯穿全书的关键设计：

1. **进程怎么分？** 每个 DP（数据并行）rank 是一个完全独立的 `LLMEngine` 进程（离线脚本用 `examples/_launch.py` 的 `launch_dp` 启动）。在单个 DP 进程内部，只有开启 CP（上下文并行）时才会再派生子进程：`cp_rank = 0` 的 `Worker` 直接驻留在主进程里（称为 `driver_worker`），`cp_rank >= 1` 的 `Worker` 作为子进程通过 `cp_broadcast_object` 接收主进程广播的方法名与参数（`worker/base.py` L290–L302）。
2. **为什么要两套 `BlockManager`？** `self.block_manager` 管理按 64 个 token 切块的分页 KV 池；而 `self.state_manager`（L265–L266）管理每条活跃请求独占一个槽位的定长状态池（SWA 滑动窗口、KV 压缩窗口等）。一条请求必须同时在两边拿到资源才能运行（详见第 4 章）。
3. **为什么要卡一道“80% 容量闸门”？** 在 `_init_cache`（L445–L451）中，如果单条请求的最大长度 `runtime_max_model_len` 超过了 GPU 分页 KV 总容量的 80%，引擎直接抛 `ValueError` 拒绝启动。这是为了给并发请求和在途预留留出缓冲，避免单条超长请求把显存池彻底锁死（详见第 4、12 章）。

---

## 2.2 阶段二：请求入队与状态机（`add_request`）

当调用 `llm.generate(prompts, sampling_params)` 时，每条 prompt 会经过分词后送入 `LLMEngine.add_request`（`llm_engine.py` L472–L631）。在真正进入 C++ 调度器之前，Python 层先做一轮安检与预处理：

- **合法性校验**：若 prompt 为空则抛 `ValueError`（L553）；若存在越界的 token ID，则打印错误日志并**直接丢弃该请求（返回 `None`）**，防止单个脏请求搞崩整个批次（L580–L589）。
- **YoCo 与 `prompt_logprobs` 互斥**：如果开启了 YoCo 预填充截断（第 7 章），前面的历史 token 在后半段层根本不会算 logits，因此强制关闭 `prompt_logprobs` 并告警（L529–L538）。
- **结构化输出（SO）异步编译**：如果请求带了 JSON Schema 或正则约束（`so_args`），它**不会立刻进入调度器**，而是先提交给后台线程池 `so_matcher_builder.submit(seq, so_args, _enqueue)` 编译语法状态机，编译完成后才通过回调 `_enqueue` 调用 `scheduler.add_sequence`（L606–L631）。这也是官方 Q&A 强调的“编译与推理解耦”。

一旦进入 C++ 侧，每条请求都被封装为一个 `Sequence` 对象（`csrc/engine/sequence.hpp`），其一生在以下状态之间流转：

```{.mermaid caption="图 2-2　Sequence 状态机（csrc/engine/sequence.hpp L30–L64）"}
flowchart LR
  subgraph Active["活跃状态（status < 3）"]
    direction TB
    W["WAITING (0)<br/>等待 prefill / 分块未跑完 / 被抢占重算"]
    R["RUNNING (1)<br/>已完成首轮或末块调度，正在执行"]
    W -- "末个 prefill chunk 被调度" --> R
    R -- "被抢占且允许重算" --> W
    W -- "中间 chunk 跑完放回队头" --> W
  end

  subgraph Term["终止状态（status ≥ 3，is_finished() 为真）"]
    direction TB
    F1["FINISHED_STOPPED (3) → 'stop'（采样到 EOS）"]
    F2["FINISHED_LENGTH_CAPPED (4) → 'length'（达 max_tokens 或 max_model_len）"]
    F3["FINISHED_ABORTED (5) / IGNORED (6) → 主动取消或 prompt 超长"]
    F4["FINISHED_PREEMPTED (7) → 'insufficient_system_resource'（抢占即终止）"]
    F5["FINISHED_SO_FAILED (8) / VE_FAILED (9) → 语法编译或视觉编码失败"]
  end

  Active --> Term
```

注意一个小细节：`SequenceStatus` 枚举里虽然定义了 `SWAPPED = 2`，但全仓没有任何代码会把状态设成 `SWAPPED`——DeepInfer 移除了把 KV 换出到 CPU 内存的慢路径，显存不足时要么直接终止（线上），要么丢弃 KV 重新 prefill（离线）。C++ 判断请求是否结束只需一行：`status >= SequenceStatus::FINISHED_STOPPED`（L44）。

---

## 2.3 阶段三：一步之内发生了什么（`LLMEngine.step`）

请求入队后，`LLM._run_engine`（`llm.py` L168–L194）在一个循环里不断调用 `LLMEngine.step()`，直到所有请求结束。一步之内的跨层调用顺序如下：

```{.mermaid caption="图 2-3　LLMEngine.step() 的单步调用时序（同步模式；异步重叠机制见第 5 章）"}
sequenceDiagram
  participant L as LLM._run_engine
  participant E as LLMEngine (Python)
  participant S as Scheduler (C++)
  participant W as Worker / ModelRunner
  L->>E: step()
  E->>S: schedule(sequence_metadata_pool)
  S-->>E: seq_metadata_list + SchedulerOutputs
  E->>W: execute_model_step(is_prompt, seq_metadata_list, ...)
  W->>W: 1. execute_main_model (GPU 跑模型前向 + DFlash 缓存更新)
  W->>W: 2. execute_sample (GPU 跑主采样 + DFlash 草稿生成与采样)
  W->>E: 3. pre_sampler_sync()（趁等 GPU 采样时，CPU 回收上一轮资源）
  W->>E: 4. post_process() → _process_model_outputs()
  E->>S: process_MTPInfo() + process_outputs()（更新 KV 计数与追加 token）
  E->>E: build_request_outputs()（组装输出放入对象池）
  E-->>L: 返回本步 RequestOutput 列表
```

这里有一个极其精妙的工程技巧——**趁 GPU 忙的时候干 CPU 杂活（`pre_sampler_sync` 与 `DeferredRelease`）**：

在图 2-3 的第 2 步，Worker 向 GPU 发射了采样 kernel，随后必须等 GPU 把采样出的 token ID 拷回 CPU 内存才能做第 4 步 `post_process`。在等待 GPU 的这段空档里，Worker 立刻回调引擎的 `pre_sampler_sync()`（`worker.py` L162–L165，`llm_engine.py` L971–L982），一口气干完三件 CPU 杂活：

1. 调用 `deferred_release.release()`：把**上一步**产出的 `SequenceMetadata`、`RequestOutput` 对象归还对象池，解除 fork 块的临时引用；
2. 对已结束的请求执行 `finalize_seq`（真正调用 `free_seq` 归还 KV 块与 State 槽）；
3. 调用 `so_matcher_builder.pump()`，把后台刚编译好语法自动机的请求推进调度器。

这样一来，内存回收与对象池管理的 CPU 开销被完全藏在了等待 GPU 采样的阴影里，不占任何关键路径时间。

---

## 2.4 核心不变量：`cache.num_tokens` 与两阶段提交

读第 3、5、6 章之前，必须先搞懂 `Sequence` 身上最重要的两个数字之间有什么关系：

- `seq.get_len()`（即 `seq.input.tokens.size()`）：当前序列里**已知的所有 token 总数**（包含 prompt 以及到目前为止已经生成并确认接受的所有输出 token）。
- `seq.cache.num_tokens`：**已经送进模型跑完前向、写入了 KV Cache 的 token 数量**。

请思考一个问题：**对于一个正在 `running` 队列里等待下一步 decode 的请求，`cache.num_tokens` 和 `get_len()` 相等吗？**

答案是**不相等，永远差 1**：

$$\text{cache.num\_tokens} = \text{seq.get\_len()} - 1$$

为什么？因为上一步刚在 GPU 上采样出来的那个最新 token（`last_output_token`），虽然已经被追加到了 `seq.input.tokens` 的末尾（所以 `get_len()` 增加了 1），但它**还没有作为输入送进模型跑过前向**，它的 K 和 V 还没有算出来！要等到下一步 decode 把它喂进模型，它的 KV 才会被写进缓存。

理解了这个“永远差 1”的不变量，再看 `Scheduler::process_outputs`（`scheduler.hpp` L872–L1022）是怎么更新账本的，就会豁然开朗：

```{.mermaid caption="图 2-4　process_outputs 对单条请求的账本更新流程（scheduler.hpp L898–L1016）"}
flowchart TB
  subgraph Row1["第一阶段：在途结算、异常拦截与乐观推进（advance_optimistically）"]
    direction LR
    P1["1. finish_inflight_step()<br/>扣减本步在途计数"] --> P2["2. 若已结束或在途被抢占<br/>释放槽位并作废本步输出"] --> Q1["3. advance_optimistically(q)<br/>暂存旧值并令 num_tokens += q"]
  end

  subgraph Row2["第二阶段：分块判断、追加采样 Token 与精确回填（commit_processed_tokens）"]
    direction LR
    Q2["4. 若 num_tokens < len（分块未完）<br/>设回 WAITING 塞回队头"] --> R1["5. 若已跑完：Prompt 步移入 running；<br/>逐个追加接受的草稿与主采样词并判停"] --> R3["6. Decode 步执行 commit_processed_tokens：<br/>num_tokens = 旧值 + 实际追加数（保持 = len - 1）"]
  end

  Row1 --> Row2
```

让我们拿一个具体的数字例子，把图 2-4 里的 **Prefill 步** 和 **Decode 步** 各算一遍：

1. **Prefill 跑完的那一步**：
   - 假设 prompt 长度是 $10$（`get_len() = 10`，`num_tokens = 0`），一步跑完（`seqlen_q = 10`）。
   - 执行 `advance_optimistically(10)` 后，`num_tokens` 变成 $10$。
   - 随后把模型采样出的第 1 个输出 token 追加到序列末尾，`get_len()` 变成 $11$。
   - 此时不需要额外修正，天然满足 $\text{num\_tokens} (10) = \text{get\_len()} (11) - 1$！同时请求在这时才正式加入 `running` 队列（L927）。
2. **带投机解码的 Decode 步**：
   - 进入本步前，`get_len() = 11`，`num_tokens = 10`（第 11 个 token 还没进过模型），外加 $5$ 个候选草稿 token。
   - 本步模型一共前向验证了 `seqlen_q = 1 + 5 = 6` 个 token（1 个主输入 + 5 个草稿）。
   - 执行 `advance_optimistically(6)`（`sequence.hpp` L216）：先把推进前的 `previous_num_tokens = 10` 存起来，再把 `num_tokens` 乐观地加到 $10 + 6 = 16$。
   - 假设验证结果是：**5 个草稿里接受了前 2 个，再加上模型顺带采出的 1 个新 token，一共往序列里追加了 `token_cnt = 3` 个新 token**。此时 `get_len()` 从 $11$ 涨到了 $11 + 3 = 14$。
   - 后面 3 个被拒绝的草稿虽然在刚才的前向里写进了 KV 块，但它们是废数据！于是最后调用 `commit_processed_tokens(3)`（`sequence.hpp` L220）：
     $$\text{num\_tokens} = \text{previous\_num\_tokens} (10) + \text{token\_cnt} (3) = 13$$
   - 看！$\text{num\_tokens} (13) = \text{get\_len()} (14) - 1$ 再次完美成立，被拒绝的那 3 个草稿写过的 KV 槽位不需要清零，下一步直接从位置 13 开始覆盖即可！

---

## 2.5 本章小结：Python 与 C++ 的分工边界

把这一章走过的路径浓缩成一张架构分工图：

```{.mermaid caption="图 2-5　DeepInfer 的四层分工与语言边界"}
flowchart TB
  subgraph L1["1. Python 入口与编排层（entrypoints / engine）"]
    direction LR
    LLM["LLM（离线批处理循环）"] --> Engine["LLMEngine（生命周期、异步 SO 编译、延迟释放）"]
  end

  subgraph L2["2. C++ 高速调度与元数据层（csrc/engine，pybind11 绑定）"]
    direction LR
    Sched["Scheduler（队列、步型投票、抢占、账本提交）"]
    BM["BlockManager × 2（KV 分页池与 State 定长池）"]
    Prep["zos.cpp / prepare_tensor_input_v4 / mtp_varlen"]
    Sched --> BM
    Sched --> Prep
  end

  subgraph L3["3. Python 执行器与模型层（worker / model_executor）"]
    direction LR
    Worker["Worker（进程协同、CUDA Graph、ZOS 流水线）"] --> Runner["ModelRunner → DeepseekForCausalLM + Sampler + DFlash"]
  end

  subgraph L4["4. 底层算子与通信库（vendor + tilelang_ops）"]
    direction LR
    Kernels["FlashMLA · DeepGEMM · DeepEP · DeepSelect · TileLang (CUDA / Ascend)"]
  end

  L1 --> L2
  L1 --> L3
  L3 --> L2
  L3 --> L4
```

记住这套分工：**C++ 管一切高频循环与状态账本（微秒级），Python 管流程编排与张量接线，底层 Kernel 管矩阵乘与通信**。下一章，我们就掀开 `csrc/engine/scheduler.hpp` 的盖子，看看 C++ 调度器在每一步到底是怎么选出那批幸运请求的。
