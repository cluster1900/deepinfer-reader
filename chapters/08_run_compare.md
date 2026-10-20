# 第 8 章　配置、运行、评测与横向对比

## 8.1 环境要求与构建

README 的要求：Linux、GCC ≥ 13、Python 3.12、CUDA 13（NVIDIA）或 CANN 9.2（昇腾）。构建：

```bash
export DEEPINFER_ACCELERATOR=nvidia   # 昇腾用 ascend；运行时也要保持导出
./build.sh
```

硬件只支持 NVIDIA SM100 和昇腾 950。本书写作环境没有这两种硬件，**以下运行步骤都没有实际跑过**，只是对照源码和文档整理。

## 8.2 离线推理

```bash
python examples/offline_inference.py            # 默认 --ep 8
python examples/offline_inference.py --ep 2 --dp 8
python examples/offline_inference.py --ep 2 --cp 8
```

默认模型 `deepseek-ai/DeepSeek-V4.1-Flash-1017` 首次运行会下载到 HF 缓存；设 `DEEPINFER_HF_MODEL_DIR` 可指向本地目录（`examples/README.md`）。`offline_inference.py` 的关键设置（L24–L58）：

- `dp_size` 默认 `max(ep // cp, 1)`，并断言 `dp * cp` 能被 `ep` 整除；
- NVIDIA 上默认 `DEEPINFER_BATCH_INVARIANT=1`；
- `create_llm(..., max_num_batched_tokens=32768, max_model_len=32768, max_num_seqs=4, gpu_memory_utilization=0.85)`。

`examples/_config.py` 的 `setup_common_env` 在开 EPLB 时设 `DEEPINFER_OVERRIDE_NUM_EXTRA_EXPERTS` 默认 32、`DEEPINFER_EXPERT_REARRANGEMENT_STEP_INTERVAL=30`、`DEEPINFER_REARRANGE_AFTER_PREFILL_IN_NORMAL_ENGINE=1`（L75–L83）。也就是说，引擎本身默认不重排专家（第 7 章），示例脚本替你打开了。

Python 入口是 `deepinfer.entrypoints.llm.LLM`（L54 起），风格与 vLLM 的 `LLM` 类似：

```python
from deepinfer.entrypoints.llm import LLM
from deepinfer.sampling_params import SamplingParams

llm = LLM(model=path, expert_parallel_size=8, trust_remote_code=True)
outs = llm.generate(prompts, SamplingParams(temperature=0.6, max_tokens=256, seq_seed=42))
```

`LLM.__init__` 有两个与 `EngineArgs` 不同的默认值：`disable_log_stats=True` 和 `stop_preempted_request=False`（L66–L68）。后者决定被抢占的请求是重算还是直接终止（第 3 章）。

## 8.3 常用开关

DeepInfer 的大部分行为由 `deepinfer/environ.py` 里声明的环境变量控制。本书各章涉及的：

| 变量 | 默认 | 作用 | 章节 |
|---|---|---|---|
| `DEEPINFER_ZERO_OVERHEAD_SCHEDULING` | NORMAL/COMPLETION 开，PROMPT 关 | 异步调度 | 5 |
| `DEEPINFER_ENFORCE_EAGER` | 关 | 关闭 CUDA Graph | 5 |
| `DEEPINFER_CUDAGRAPH_BATCH_SIZE_INTERVAL` | 8 | CUDA Graph 档位间隔 | 5 |
| `DEEPINFER_DISABLE_SYNC_GUARD` | 关 | 关闭“禁止隐式同步”守卫 | 5 |
| `DEEPINFER_USE_NEXTN_DECODE` | 5 | 最大 draft 长度 | 6 |
| `DEEPINFER_ENABLE_MTP_VARLEN` | 关 | 置信度变长验证 | 6 |
| `DEEPINFER_MAX_BACKTRACKING_TOKENS` | 至少 nextn+1 | SWA 环形缓冲的回退余量 | 4 |
| `DEEPINFER_YOCO_PREFILL_TRUNCATION_MODE` / `_KEEP_TOKENS` | `off` / 0 | SWA bounded replay；README 建议 `compact` 和 `140` | 4 |
| `DEEPINFER_OVERRIDE_NUM_EXTRA_EXPERTS` | 0 | 冗余专家数下限 | 7 |
| `DEEPINFER_EXPERT_COUNT_WINDOW_SIZE` | 100 | EPLB 统计窗口 | 7 |
| `DEEPINFER_EXPERT_REARRANGEMENT_STEP_INTERVAL` | 0（不重排） | EPLB 重排间隔 | 7 |
| `DEEPINFER_BATCH_INVARIANT` | 关 | 批不变推理（仅 NVIDIA） | 8.5 |

“默认”一列取自 `environ.py` 的声明（`EnvVar.int()` 默认 0，`EnvVar.bool()` 默认 False）。ZOS 的默认值是 `EngineArgs.__post_init__` 用 `os.environ.setdefault` 写入的。

## 8.4 基准与评测

`examples/benchmark.py` 把 prefill 和 decode 两阶段解耦测量，默认 `decode_num_requests=64`、`context_length=32768`、`non_cached_tokens=4096`、`prefill_batch_size=32768`、nextn=5。`examples/README.md` 有几处容易误读的地方：

- **decode throughput 按“全部接受”记账**：`avg_SPS × decode_num_requests × (1 + nextn) / --cp`，不管真实接受率。它是硬件吞吐上限，不是有效 token 吞吐。
- profiler 开着时的步耗时最多会虚高近 50%（bsz=1），只能用来看 kernel 组成，不能用来算 SPS。
- 测量窗口会剔除撞上 MoE 负载统计 flush（EP all_gather）的区间。

`examples/sweep/run_sweep.sh` 扫一维参数，画吞吐-延迟 Pareto 曲线。README 的性能图是在 4 张 SM100（prefill）和 64 张（decode）上测的。

`eval/` 是精度评测：AIME 2026（30 题×32 次）、HLE（2500 题）、GPQA Diamond（198 题×8 次），指标是 Pass@1。生成和打分分两阶段，数据集需要自行下载转换。

## 8.5 批不变与确定性采样

**批不变（batch-invariant）**：同一请求不管和谁一起组 batch，输出 logits 都一样。浮点加法不满足结合律，batch 大小一变，GEMM 的分块和归约顺序就可能变，结果差几个 ulp，采样时就可能分岔。`DEEPINFER_BATCH_INVARIANT=1` 时：

- `deep_gemm` 和 `tile_kernels` 开启确定性算法（`worker/runtime_config.py` L67–L69）；
- 注意力 kernel 传 `batch_invariant=True`（`attn.py` L666、L1002）；
- 昇腾上直接断言不支持（L61–L63）。

**确定性采样**还需要随机数与 batch 无关。`SamplingParams.seq_seed` 是每个请求自己的 uint64 种子（`sampling_params.py` L44、L86–L91），通过 `prepare_sample.cpp` 逐采样位置传给 kernel。README 写明完整条件是 `DEEPINFER_BATCH_INVARIANT=1 DEEPINFER_ENABLE_MTP_VARLEN=0` 加固定 `seq_seed`。要关变长验证，是因为它让每个请求验证几个 token 取决于同批其他请求（第 6 章）。

`minidi` 在第 10 章用“种子 + 位置”派生随机数实现同样的性质，`test_seeded_sampling_is_batch_invariant` 验证单独跑和混在批里跑的输出一致。

## 8.6 与 vLLM、SGLang 的对比

README 的致谢写着 DeepInfer“originally built primarily on vLLM, with additional inspiration from SGLang and TensorRT-LLM”，`deepinfer/` 下有 21 个文件标注 “Derived from vLLM”。Q&A 说异步调度“参考已有实现（SGLang, Nanoflow）”。

下表的 DeepInfer 一列来自本书前几章对源码的核对；vLLM 和 SGLang 两列只写广为人知的设计，**没有在本书中对照它们的源码逐条核验**，版本不同可能有出入。

| 维度 | DeepInfer（commit 8b16149） | vLLM | SGLang |
|---|---|---|---|
| 定位 | 单一模型（DeepSeek-V4.1-Flash）的生产推理框架 | 通用、多模型 | 通用、多模型 |
| 硬件 | NVIDIA SM100、昇腾 950 | 多种 GPU/加速器 | 多种 GPU |
| 调度核心 | C++（`csrc/engine/scheduler.hpp`），pybind 暴露给 Python | Python | Python |
| prefill/decode | 同一步不混跑；跨 rank 投票决定步类型 | 支持 chunked prefill 与 decode 混批 | 支持混批 |
| KV 布局 | 分页，块大小固定 128，单 KV head，SWA 状态单独存放 | PagedAttention | 分页，RadixAttention |
| 前缀缓存 | 开源版没有前缀哈希缓存；Q&A 说外部 KV 存储基于 3FS，未开源 | 自动前缀缓存 | Radix 树前缀缓存 |
| 异步调度 | ZOS，NORMAL/COMPLETION 默认开 | 有异步调度选项 | overlap scheduler |
| 投机解码 | MTP/DFlash 草稿 + 置信度变长验证 | 多种草稿方式 | 多种草稿方式 |
| 并行 | DP 注意力 + EP MoE + CP，不用 TP | 以 TP/PP 为主，也支持 EP | TP/DP 注意力/EP |
| 批不变 | 开关控制（仅 NVIDIA） | — | — |

本书最有把握的结论是“取舍”的方向，而不是“谁更快”：DeepInfer 为一个模型、两种硬件、Wide-EP 部署深度定制，把调度器下沉到 C++，明确不混跑 prefill 和 decode，把前缀缓存和负载均衡留在框架之外。通用框架要支持大量模型和硬件，取舍自然不同。本书没有做任何性能对比实验。
