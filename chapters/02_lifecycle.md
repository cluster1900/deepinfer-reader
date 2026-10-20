# 第 2 章　一个请求的一生

这一章跟着一次 `llm.generate(...)` 调用，从入口走到 token 返回。目标是建立整张地图：哪个对象在哪一层、谁调用谁、数据在 Python 与 C++ 之间怎么来回。细节留给后面各章。

## 2.1 启动：从 `LLM(...)` 到引擎就绪

`LLM.__init__`（`deepinfer/entrypoints/llm.py` L54–L87）做三件事：设置两个默认值，组装 `EngineArgs`，调用 `LLMEngine.from_engine_args`。两个默认值值得注意：

- `kwargs.setdefault("disable_log_stats", True)`：离线推理默认不打统计；
- `kwargs.setdefault("stop_preempted_request", False)`（L68）：离线推理里，被抢占的请求**重算**而不是直接结束。而 `EngineArgs` 自己的默认值是 `True`（`engine/arg_utils.py` L40），即服务形态下默认“抢占即终止”，并返回 finish reason `insufficient_system_resource`（`csrc/engine/sequence.hpp` L56）。第 3 章会解释为什么线上宁可杀掉请求也不重算。

`LLMEngine.__init__`（`llm_engine.py` L155–L305）的启动顺序如下：

```{.mermaid caption="图 2-1　LLMEngine 初始化顺序（llm_engine.py L155–L305，L416–L458）"}
flowchart TB
  A["get_hf_tokenizer"] --> B["若模型有 multi_modal_mapper：<br/>构造 MultimodalProcessor"]
  B --> C["_init_workers：<br/>CP rank 1..n-1 各起一个子进程<br/>rank 0 是本进程内的 driver Worker"]
  C --> D["load_model（dummy 格式可跳过）"]
  D --> E["_init_cache"]
  E --> E1["profile_num_available_blocks：<br/>跑一次 profile_run，按峰值显存反推块数"]
  E1 --> E2{"runtime_max_model_len<br/>≤ 0.8 × block_size × 块数？"}
  E2 -- 否 --> X["ValueError：调大 gpu_memory_utilization<br/>或调小 max_model_len"]
  E2 -- 是 --> E3["init_cache_engine：真正分配 KV 与 state"]
  E3 --> E4["warm_up_model：捕获 CUDA Graph，<br/>开启变长验证时测 decode 性能曲线"]
  E4 --> F["BlockManager(num_gpu_blocks)<br/>state_manager = BlockManager(num_state_seqs)"]
  F --> G["C++ Scheduler(...)"]
  G --> H["SequenceMetadataPool、DeferredRelease、StatsLogger"]
```

几个要点：

1. **进程模型**。每个 DP rank 是一个独立的 `LLMEngine` 进程（离线示例用 `examples/_launch.py` 的 `launch_dp` 为每个 DP rank 起一个进程，L15–L40）。在一个 DP rank 内部，只有上下文并行（CP）会再拆进程：`_init_workers` 为 `cp_rank = 1 .. cp_size-1` 各 spawn 一个 `Worker`，rank 0 的 `Worker` 直接在引擎进程内构造，称为 driver（L368–L414）。driver 调方法时，`run_workers` 先把方法名和参数通过 `cp_broadcast_object` 广播给其他 CP rank，再自己执行（`worker/base.py` L290–L302）。
2. **两套块管理器**。`block_manager` 管分页的 KV 块，`state_manager` 管每请求一个的定长 state 槽（只有模型有 state 时才建，L265–L266）。为什么要两套，见第 4 章。
3. **80% 闸门**。`_init_cache` 要求单个请求的最大长度不超过 KV 总容量的 80%（L445–L451）。第 12 章用 `minidi` 演示没有这道闸门会怎样卡死。

## 2.2 提交：`generate` 与 `add_request`

`LLM.generate`（L90–L142）把每个 prompt 交给 `_add_request`，后者先 `deepcopy` 采样参数，分配自增 `request_id`，用 tokenizer 编码，然后调用 `LLMEngine.add_request`（L144–L166）。注意这里的 `user_id="0"` 旁边有一行注释 `# HACK 随便加一个 user_id`，说明这个字段是给线上路由用的，离线时无意义。

`LLMEngine.add_request`（L472–L631）在把请求交给调度器之前做了这些检查和改写：

| 检查 / 改写 | 行为 | 位置 |
| --- | --- | --- |
| `sampling_params` 是 dict | 转成 `SamplingParams`，丢掉值为 None 的键 | L525–L526 |
| 写入 `eos_id` | 用 tokenizer 的 EOS 覆盖 | L527 |
| YoCo 截断开启且要 prompt logprobs | 强制关掉 prompt logprobs 并告警 | L529–L538 |
| 多模态 | 用 `overwrite_prompt` 把图片占位展开成图片 token | L543–L551 |
| prompt 为空 | 抛 `ValueError("prompt must not be empty")` | L553–L554 |
| token id 越界 | 打日志、**返回 None、请求被丢弃**，不抛异常 | L580–L589 |
| 结构化输出 | 异步提交给 `SOMatcherBuilder` 编译 grammar，编译完再入队 | L606–L631 |

最后一行值得多看一眼：带 `so_args` 的请求不会立刻进调度器，而是 `so_matcher_builder.submit(seq, so_args, _enqueue)`，grammar 编译完成后回调 `_enqueue` 才 `scheduler.add_sequence`。官方 Q&A 把这叫“编译与推理解耦，SO 编译异步进行”。所以 `get_num_unfinished_requests` 要把还在编译的请求数也加上（L658–L663），否则 `generate` 的循环会提前退出。

```{.mermaid caption="图 2-2　请求状态机（csrc/engine/sequence.hpp L30–L64）"}
stateDiagram-v2
  [*] --> WAITING: add_sequence
  WAITING --> RUNNING: prefill 最后一个 chunk 被调度
  RUNNING --> WAITING: 被抢占且 stop_preempted_request=false（重算）
  WAITING --> WAITING: chunk 未跑完，放回队头
  RUNNING --> FINISHED_STOPPED: 采到 EOS（"stop"）
  RUNNING --> FINISHED_LENGTH_CAPPED: 到 max_tokens 或超过 max_model_len（"length"）
  RUNNING --> FINISHED_PREEMPTED: 被抢占且 stop_preempted_request=true
  WAITING --> FINISHED_IGNORED: prompt 超长 / 块永远不够
  WAITING --> FINISHED_ABORTED: abort_request
  RUNNING --> FINISHED_ABORTED: abort_request
  RUNNING --> FINISHED_SO_FAILED: grammar 出错或超时
  WAITING --> FINISHED_VE_FAILED: 图片处理失败
```

枚举里还有一个 `SWAPPED`，但在 `csrc/` 中只出现在枚举定义里，没有任何地方把请求设成这个状态：开源版没有 KV 换出到 CPU 的路径。判断“已结束”用的是 `status >= FINISHED_STOPPED`（L44），所以枚举顺序不能随便改。

## 2.3 一步：`LLMEngine.step`

`LLM._run_engine`（L168–L194）是一个朴素的循环：有未完成请求就 `step()`，没有了就调一次 `flush_inflight_reqs()` 把在途的最后一步收回来，然后退出。每步返回的 `RequestOutput` 由 `RequestOutputPool` 管理，**下一次 `step()` 时就会被回收**（`step` 的 docstring，L866–L870），所以 `_run_engine` 每拿到一个输出都要 `RequestOutput.clone`。

一步内部的调用关系：

```{.mermaid caption="图 2-3　一次 step 的调用序列（同步模式；异步模式见第 5 章）"}
sequenceDiagram
  participant L as LLM._run_engine
  participant E as LLMEngine
  participant S as C++ Scheduler
  participant W as Worker（driver）
  participant M as ModelRunner / 模型
  L->>E: step()
  E->>S: schedule(sequence_metadata_pool)
  S-->>E: seq_metadata_list, SchedulerOutputs
  E->>W: clone_and_zero_states_and_blocks（fork 时）
  E->>W: execute_model_step(is_prompt, ...)
  W->>M: execute_main_model（forward）
  W->>M: execute_sample
  W->>E: pre_sampler_sync（释放上一轮资源、finalize 已结束请求）
  W->>W: _post_process_step
  W->>E: post_process(step_state)
  E->>S: process_MTPInfo、process_outputs
  E->>E: build_request_outputs
  E-->>L: request_outputs
```

对应代码：

1. **调度**（L881–L928）：`self.scheduler.schedule(...)` 是 C++，返回本步要跑的 `SequenceMetadata` 列表和一个 `SchedulerOutputs`。如果有 fork 出来的请求需要复制 state 或最后一个 KV 块，接着调 `clone_and_zero_states_and_blocks_on_default_stream`。prompt 步里还会检查结构化输出的簿记状态，发现异常就只判死这一个请求（`FINISHED_SO_FAILED`），不影响同批其他请求。
2. **执行**（L952，`_execute_model_async_step`）：调 Worker 的 `execute_model_step`。同步和异步两种模式都走这一个入口，差别在 Worker 内部（第 5 章）。
3. **回收与输出**（`post_process`，L984–L1015）：Worker 在采样完成后回调引擎的 `post_process`，里面调 `_process_model_outputs`（L687–L785），它依次做：
   - `process_MTPInfo`：把草稿 token 和置信度写回各请求（第 6 章）；
   - `scheduler.process_outputs`：把采样出的 token 追加进请求、判断停止、推进或回填 `cache.num_tokens`；
   - `build_request_outputs`：在 C++ 里组装 `RequestOutput`，放进对象池。

## 2.4 `process_outputs`：token 落账的地方

`Scheduler::process_outputs`（`scheduler.hpp` L872–L1022）是理解调度器状态推进的关键。对每个本步调度的请求：

```{.mermaid caption="图 2-4　process_outputs 对单个请求的处理分支"}
flowchart TB
  A["finish_inflight_step"] --> B{"已结束？<br/>（期间被 abort）"}
  B -- 是 --> B1["释放 MTP slot；prompt 步仍放进 running<br/>（让 decode 侧统一回收）"]
  B -- 否 --> C{"decode 步且状态是 WAITING？<br/>（在途时被抢占）"}
  C -- 是 --> C1["整步作废，什么都不记"]
  C -- 否 --> D["cache.advance_optimistically(seqlen_q)"]
  D --> E{"num_tokens ≥ len？"}
  E -- "否：chunk 没跑完" --> E1["放回 waiting 队头"]
  E -- 是 --> F["prompt 步：放进 running"]
  F --> G["取出被接受的 draft + 本步主 token"]
  G --> H["逐个 append_token_id 并 _check_stop，<br/>停了就不再追加"]
  H --> I{"decode 步？"}
  I -- 是 --> I1["commit_processed_tokens：<br/>num_tokens = 推进前 + 实际追加数"]
  I -- 否 --> J["记录首 token 延迟"]
  I1 --> J
```

有三点和直觉不同：

- **prefill 完成的请求在这里才进 `running`**，不是在调度时。调度 prefill 的那一刻，请求既不在 `waiting`（已经 pop 了）也不在 `running`，处于“在途”。这也是为什么 decode 的抢占永远碰不到正在 prefill 的请求（L917–L919 的注释）。
- `cache.num_tokens` 先**乐观推进** `seqlen_q`，再按实际接受的 token 数**回填**（`sequence.hpp` L216–L221）。投机解码时 `seqlen_q = 1 + 草稿数`，被拒绝的草稿写过的 KV 位置就这样“作废”，不用擦除，下次覆盖即可。
- `_check_stop`（`scheduler.hpp` L1035–L1051）只看三件事：EOS（且未设 `ignore_eos`）、总长度超过 `max_model_len`、输出数等于 `max_tokens`。开源版没有 stop string，停止字符串之类的逻辑在 API 层（官方 Q&A 提到的 DeepSeek Recipe）。

## 2.5 资源为什么延后一轮释放

`DeferredRelease`（`llm_engine.py` L67–L119）把本步产出的 `SequenceMetadata`、`RequestOutput`、fork 时 pin 住的块收集起来，在**下一步**的 `pre_sampler_sync` 里才释放。原因写在类注释里：“延后一轮释放 step 产出的资源，保证调用方读完”，并且“与 GPU 计算 overlap”。`pre_sampler_sync` 是 Worker 在 launch 了采样 kernel 之后、等待它完成之前调用的（`worker.py` L162–L165），这段时间 CPU 本来就在等 GPU，把释放、`finalize_seq`（真正 `free_seq` 归还块）、结构化输出编译队列的 `pump` 放在这里，相当于白捡的 CPU 时间。

这种“借用存储、下一步前必须消费完”的约定在 C++ 侧也到处都是：`SchedulerOutputs` 的 `clone_states/clone_blocks/zero_states` 三个数组借用调度器内部 vector 的存储（`scheduler.hpp` L279–L281），`process_outputs` 返回的 numpy 数组借用一块 static 存储（L879–L881）。好处是零拷贝、零分配，代价是对调用顺序有严格要求。自己写引擎时，在没有性能证据前不建议这样做，第 16 章会再提。

## 2.6 小结

```{.mermaid caption="图 2-5　分层与语言边界"}
flowchart LR
  subgraph Py1["Python：entrypoints / engine"]
    LLM --> LLMEngine
  end
  subgraph Cpp["C++（pybind11，deepinfer._C.engine）"]
    Scheduler
    BlockManager
    Sequence
    ZOS["zos.cpp / prepare_tensor_input"]
    MTP["mtp_varlen.hpp"]
  end
  subgraph Py2["Python：worker / model_executor"]
    Worker --> ModelRunner --> Model["DeepseekForCausalLM"]
    ModelRunner --> Sampler
  end
  subgraph K["vendor + tilelang_ops"]
    FlashMLA
    DeepGEMM
    DeepEP
    TileLang["TileLang kernels"]
  end
  LLMEngine --> Scheduler
  Scheduler --> BlockManager
  LLMEngine --> Worker
  Worker --> ZOS
  Model --> K
  Scheduler --> MTP
```

记住三句话就够了：调度和簿记在 C++，流程编排在 Python，计算在 vendor kernel 和 TileLang 算子里。接下来按这个顺序逐层往下读。
