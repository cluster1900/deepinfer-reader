# DeepInfer 源码解读与从 0 到 1

[在线阅读](https://cluster1900.github.io/deepinfer-reader/) · [下载 EPUB](https://cluster1900.github.io/deepinfer-reader/deepinfer-reader.epub) · [下载 PDF](https://cluster1900.github.io/deepinfer-reader/deepinfer-reader.pdf) · [Releases（EPUB / PDF / 离线网页包）](https://github.com/cluster1900/deepinfer-reader/releases)

一本中文书，分两部分：

- **第一部分（第 1–8 章）完整解读 DeepSeek 开源的推理框架 [DeepInfer](https://github.com/deepseek-ai/DeepInfer)**：仓库地图、请求的一生、C++ 调度器、显存与 KV Cache、异步调度（ZOS）与 CUDA Graph、DSpark 置信度变长验证、模型/算子/并行/EPLB、配置运行与 vLLM/SGLang 对比。所有路径、行号、默认值都对照固定提交 `8b16149741a4935e1dec8cea27853f273b374c5b` 逐条核对；官方 Q&A 里的数字会单独标注。
- **第二部分（第 9–16 章）个人如何从 0 到 1**：用纯 Python + NumPy 写一个 CPU 上可跑的教学引擎 `minidi`（867 行），分阶段实现分页 KV、调度与抢占、异步调度、投机解码与变长验证、SWA 环形缓冲、EPLB 和多 rank 步型投票。每个阶段都有测试，并对照 DeepInfer 的真实代码，写明简化了什么、开发时踩过哪些坑。

全书 28 张图（21 张 mermaid 结构图、7 张由脚本从 `minidi` 实跑生成的数据图）。

## 章节

| | 章 | 内容 |
|---|---|---|
| | 前言 | 读的是哪份代码、怎么核对 |
| 一 | 1 定位与仓库地图 | 代码规模、模块划分、支持范围 |
| | 2 请求的一生 | `LLM.generate` → C++ 调度器 → Worker → 输出 |
| | 3 调度器 | prefill/decode 分离、跨 rank 投票、FCFS、抢占 |
| | 4 显存与 KV Cache | 块数计算、块管理、KV 格式、SWA 状态槽 |
| | 5 异步调度 | ZOS 四阶段、在途计数、remap、CUDA Graph |
| | 6 投机解码 | 置信度变长验证的贪心与两轮接受率 |
| | 7 模型、算子与并行 | 模块角色、算子库、DP/EP/CP、EPLB |
| | 8 配置、运行与对比 | 运行方式、常用开关、批不变、vLLM/SGLang |
| 二 | 9 路线图 | 阶段划分与原则 |
| | 10 最小引擎 | TinyLM、扁平变长批次、step 循环 |
| | 11 分页 KV | 块表、块管理、容量闸门 |
| | 12 调度与抢占 | FCFS、LCFS 抢占、队头互等的坑 |
| | 13 异步调度 | 先调度后知道结果、在途抢占的坑 |
| | 14 投机解码 | 验证、回退、变长验证的最优性 |
| | 15 SWA / EPLB / 投票 | ring 余量推导、负载均衡、步型投票 |
| | 16 走向 GPU | 差距清单与下一步 |

## 动手

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements-examples.txt   # numpy、matplotlib、pytest
python -m pytest                           # 63 个用例
python examples/quickstart.py              # 同步 / ZOS / 投机解码三种引擎对照
python scripts/figures/make_plots.py       # 重新生成数据图
```

## 构建

```bash
# 依赖：pandoc ≥ 3.0、Node ≥ 22、Chrome/Chromium、Noto CJK 字体、poppler-utils
PUPPETEER_SKIP_DOWNLOAD=1 npm ci
python3 scripts/check_book.py      # 门禁：章节结构、图片、疑似密钥、固定 commit
python3 build.py                   # dist/index.html、dist/deepinfer-reader.epub、dist/deepinfer-reader.pdf
python3 scripts/check_outputs.py dist --min-figures 28
```

推送到 `main` 后 GitHub Actions 构建并部署 GitHub Pages；推送 `vX.Y.Z` 标签会把 EPUB、PDF、离线网页包和 `SHA256SUMS.txt` 附到对应 Release。

## 已知的不确定之处

见书末“致谢与引用”一节：两处 DeepInfer 代码与注释不一致、SWA ring 余量多 1 的原因未追到、本书没有在 SM100/昇腾上实际运行 DeepInfer、对比表中 vLLM/SGLang 两列未逐条核验源码。

## 许可

`minidi/`、`tests/`、`examples/`、`scripts/` 中的代码以 MIT 协议发布（见 `LICENSE`）。书稿正文以 CC BY 4.0 发布。书中引用的 DeepInfer 代码片段版权归 DeepSeek-AI 所有（Apache License 2.0）。
