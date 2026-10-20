# 第 5 章　异步调度：让 GPU 一刻不停

## 5.1 问题：同步调度里的空泡

官方 Q&A 的描述很直接：decode 第 N 步的流程是 `schedule(N) → forward(N) → sample(N) → post-process(N)`，其中 schedule 和 post-process 是纯 CPU 逻辑；post-process(N) 之前必须做一次 GPU→CPU 同步拿到采样结果。于是在 post-process(N) 和 schedule(N+1) 期间，GPU 是空闲的。

decode 一步的 GPU 时间往往只有十几毫秒，CPU 那几毫秒的调度和后处理占比就不可忽视。`minidi/zos.py` 的 `simulate_timeline` 用一组抽象时长（schedule=2、launch=1、forward=6、sample=1、post=2）画出两种模式：

![图 5-1　同步与异步调度的 CPU/GPU 时间线（抽象时长，示意用，不是实测）](images/plots/ch05_zos_timeline.png)

同步模式下每步耗时 2+1+6+1+2 = 12 个单位，5 步 60；异步模式下稳态每步只受 GPU 的 7 个单位限制，5 步 40。`tests/test_engine.py::test_timeline_zos_hides_cpu` 固定了这组数字。真实收益取决于 CPU 与 GPU 时间之比，本书没有 DeepInfer 的实测数据。

## 5.2 思路：先调度、后知道结果

Q&A 说这是“参考已有实现（SGLang, Nanoflow）”：GPU 执行 forward(N) → sample(N) 时，CPU 同步执行 `schedule(N+1) → launch_forward(N+1) → post-process(N-1) → launch_sample(N+1)`。难点在于 schedule(N+1) 的时候，第 N 步采样出的 token 还在 GPU 上，CPU 不知道：

1. 第 N 步之后每个请求有多长？——不知道，只知道“比 CPU 记的长度多了在途的那些”；
2. 第 N+1 步的输入 token 是什么？——就是第 N 步的输出，但只在 GPU 上；
3. 第 N 步有没有请求已经结束？——不知道，可能白算一步。

DeepInfer 对这三个问题的回答分别是：在途计数、GPU 侧 remap、事后作废。

```{.mermaid caption="图 5-2　ZOS 下 Worker.execute_model_step 的四个阶段（worker.py L89–L189）"}
sequenceDiagram
  participant CPU
  participant GPU
  Note over CPU: schedule(N+1)：长度 = 已提交 + 在途
  CPU->>GPU: 1. launch forward(N+1)<br/>输入 token 由 remap 从 sample(N) 的输出取
  Note over GPU: forward(N) 可能还在跑
  CPU->>CPU: 2. post-process(N)：等 sample_end_evt(N)，<br/>token 落账、判停
  CPU->>GPU: 3. launch sample(N+1)
  CPU->>CPU: pre_sampler_sync：释放上一轮资源
  Note over CPU: 4. decode 步：把 N+1 记为 prev_step_state<br/>prompt 步：当场 post-process
```

## 5.3 代码：`execute_model_step`

所有步（prompt / decode、ZOS 开 / 关、占位步）都走同一个入口 `Worker.execute_model_step`（`deepinfer/worker/worker.py` L89–L189），docstring 列出了四个阶段：

1. **Forward 本步**（L113–L141）。driver worker 先调 `_correct_seqlens_qk`，看本步请求和上一步在途请求有没有交集；有交集就构造 `MetadataZOS`，然后 `model_runner.execute_main_model`。
2. **Post-process 上一步**（L143–L144）。`_post_process_step(self.prev_step_state)` 先 `sample_end_evt.synchronize()` 等上一步采样完成，再调引擎的 `post_process` 回调（第 2 章的 `process_outputs` 就在这里）。
3. **Sample 本步**（L146–L165）。开了变长验证时先等一个 D2H 事件，用 GPU 修正后的 `seqlens_q` 更新 CPU 侧元数据，然后 `execute_sample`，再调 `pre_sampler_sync`。
4. **决定本步何时处理**（L182–L187）：`if is_prompt or not DEEPINFER_ZERO_OVERHEAD_SCHEDULING` 就当场 post-process；否则存进 `self.prev_step_state`，留到下一步的第 2 阶段。

注意 prompt 步总是当场处理。prefill 完成的请求要在 `process_outputs` 里才进 `running`（第 2 章），如果推迟一步，下一步的 decode 调度就看不到它们。

`ModelRunner.execute_main_model` 的 docstring 写着“This function must NOT contain any cuda synchronize”，launch forward 和 launch sample 都包在 `forbid_cuda_sync()` 里（`model_runner.py` L321–L325、L356–L360）。这个上下文管理器（`deepinfer/utils.py` L119–L132）把 PyTorch 的 sync debug mode 设成 `error`：任何隐式的 CPU-GPU 同步（比如一个不小心的 `.item()`、阻塞拷贝）都会直接报错。docstring 特别说明 `torch.cuda.Event.synchronize` 不在检测范围内，所以 ZOS 有意设置的事件同步点不受影响。`tests/test_cuda_sync_guards.py` 专门测这个守卫的嵌套和多线程行为。异步调度最怕的就是某处悄悄同步一次，把重叠全抵消掉，这个守卫是把“不许同步”变成了可测试的约束。

## 5.4 在途计数

`Sequence::ScheduleState` 有两个计数（`csrc/engine/sequence.hpp` L241–L256）：

```cpp
void start_inflight_step(int token_count) {   // schedule() 末尾调用
    total_inflight_tokens += token_count;
    last_inflight_tokens = token_count;
}
void finish_inflight_step() {                 // process_outputs 开头调用
    if (total_inflight_tokens == last_inflight_tokens) { total = last = 0; }
    else { total_inflight_tokens = last_inflight_tokens; }
}
```

它们影响两处计算：

- `get_num_blocks = (len + total_inflight_tokens + 草稿数 + bs - 1) / bs`（L292）：给还没回来的 token 提前预留块；
- `seqlen_k = cache.num_tokens + total_inflight_tokens + seqlen_q`（`scheduler.hpp` L697）：本步注意力要看的长度包括在途 token。

## 5.5 remap：输入 token 不经过 CPU

`Worker._correct_seqlens_qk`（L191–L246）调 C++ 的 `prepare_zos_inputs`（`csrc/engine/zos.cpp` L142 起），对比本步和上一步的请求列表，生成 `remap[i] = j`（本步第 i 个请求对应上一步第 j 个；新请求为 -1）。`MetadataZOS` 的字段注释写得很清楚（`worker/zos_status.py` L25–L40）：

- `remap`：`[curr_n_seq_with_padding]`，命中为上一步下标，否则 -1；
- `prev_sampled_token_ids`：上一步采样出的 token，`[prev_n_seq]`；
- `prev_predict_token_ids`：上一步的草稿，`[prev_n_seq, nextn]`；
- `prev_num_accepted_tokens`：上一步每个请求接受了几个草稿。

`zos.cpp` 里的 `prepare_token_inputs`（L31 起）分两种情况（注释见 L84）：没有重叠时直接在 pinned 内存里拼好完整的 `input_ids`；有重叠时只填“fresh”行，命中 remap 的行“come from the previous GPU outputs”，由 GPU 上的算子从 `prev_sampled_token_ids` 里取。投机解码时还要用 `compute_seqlens_k_zos` 按上一步实际接受数在 GPU 上修正 `seqlens_k`（`worker.py` L221–L229）。

## 5.6 白算的一步怎么办

如果第 N 步某个请求采到了 EOS，CPU 在 schedule(N+1) 时还不知道，会照常把它排进第 N+1 步。等 post-process(N) 把它标成已结束，第 N+1 步的结果就作废：`process_outputs` 开头的“seq is already aborted”分支直接跳过已结束的请求（`scheduler.hpp` L905–L915）。

更微妙的情况是**在途时被抢占**：schedule(N+1) 的 decode 选择过程中把一个第 N 步还在途的请求抢占了，`_preempt` 已经把它的计数复位、放回 `waiting`。等第 N 步结果回来，`process_outputs` 看到“decode 步、状态却是 WAITING”，整步输出作废（L917–L925）。`process_MTPInfo` 里也有对称的判断（L1214–L1219），否则 `process_cnt` 会跑到 `schedule_cnt` 前面，触发 `MTPVarlen::schedule` 的配对断言。

`minidi` 第 13 章实现了这一整套：`ZOSEngine` 在 CPU 上真的先调度、后取结果，命中“remap”的请求，输入 token 直接取上一步的采样结果，位置往后错一格（`minidi/zos.py` `step`）。`tests/test_engine.py::test_zos_matches_sync` 在 20 组随机配置（含抢占和 EOS）上验证 ZOS 与同步版输出逐 token 相同，且结束后所有块都归还。

## 5.7 默认开关与 CPU 进程组

`EngineArgs.__post_init__`（`arg_utils.py` L48–L55）按引擎类型设置 `DEEPINFER_ZERO_OVERHEAD_SCHEDULING` 的默认值：`NORMAL` 和 `COMPLETION` 开，`PROMPT` 关（`engine_type.py` 的 `default_zero_overhead_scheduling`）。prefill 一步本身就长，CPU 开销占比小，而且 prompt 步本来就要当场处理，异步收益有限。

开 ZOS 时有一个容易忽略的细节：调度器 DP 投票（以及 `has_unfinished_requests` 的 allreduce）用的进程组是 **CPU 上的 gloo 组**（`PG.cpu_data_parallel_group()`），不开时才用 GPU 组（`llm_engine.py` L313 创建调度器时传入；`has_unfinished_requests` L665–L679 同理）。原因是 GPU 上的 allreduce 要等 GPU 流上前面的 kernel 跑完才能拿到结果，等于在调度阶段插入一次同步，异步就白做了。C++ 侧也据此判断 `use_cpu_dp_group = dp_group->getBackendName() == "gloo"`（`scheduler.hpp` L109）。

## 5.8 CUDA Graph

decode 一步要 launch 几百上千个 kernel，CPU launch 本身就是开销。DeepInfer 对 decode 主模型和 DFlash 草稿模型分别捕获 CUDA Graph（`worker/cuda_graph_runner.py`、`dflash_cuda_graph_runner.py`），在 `warm_up_model` 阶段完成（`worker/base.py` `warm_up_model`，L274 起）。

捕获哪些 batch size 由 `build_graph_batch_sizes(max_size, interval)` 决定（`cuda_graph_runner.py` L41–L52）：先是小于 `interval` 的 2 的幂，然后每隔 `interval` 一个，最后补上 `max_size`。`interval` 默认 8（`DEEPINFER_CUDAGRAPH_BATCH_SIZE_INTERVAL`，`environ.py` L155），`max_size = max_num_seqs * (1 + nextn)`（`model_runner.py` L76–L78），这里的“batch size”是本步 token 总数 `total_q`，不是请求数。例如 `max_num_seqs=4, nextn=5` 时 `max_size=24`，捕获 `[1, 2, 4, 8, 16, 24]`。运行时 `select(total_q)` 用 `bisect_left` 找第一个不小于 `total_q` 的图（L143–L148），多出来的位置用 padding 请求填满（`_correct_seqlens_qk` 里 `n_seq_with_padding` 的计算就是为此）。

图的离散档位直接影响第 6 章的变长验证：从 16 个 token 涨到 17 个，就要换到下一档图，耗时跳一级。`DEEPINFER_ENFORCE_EAGER=1` 可以关掉 CUDA Graph（`environ.py` L154）。
