# DeepInfer 源码解读与从 0 到 1

<p align="center">
  <a href="README.md">🌐 <b>English</b></a> &nbsp;|&nbsp; 🇨🇳 <b>简体中文</b>
</p>

<p align="center">
  作者：<b>Hawk Wu</b><br/>
  🇨🇳 <b>中文版</b>：
  <a href="https://cluster1900.github.io/deepinfer-reader/index-zh.html"><b>📖 在线阅读（中文交互式网页版）</b></a> ·
  <a href="https://cluster1900.github.io/deepinfer-reader/deepinfer-reader.epub"><b>📱 下载中文 EPUB</b></a> ·
  <a href="https://cluster1900.github.io/deepinfer-reader/deepinfer-reader.pdf"><b>📄 下载中文 PDF</b></a>
  <br/>
  🌐 <b>English Edition (Default)</b>：
  <a href="https://cluster1900.github.io/deepinfer-reader/"><b>📖 Read Online (Interactive Web)</b></a> ·
  <a href="https://cluster1900.github.io/deepinfer-reader/deepinfer-reader-en.epub"><b>📱 Download EPUB (EN)</b></a> ·
  <a href="https://cluster1900.github.io/deepinfer-reader/deepinfer-reader-en.pdf"><b>📄 Download PDF (EN)</b></a> ·
  <a href="https://github.com/cluster1900/deepinfer-reader/releases"><b>📦 GitHub Releases</b></a>
</p>

这是一本深入剖析 DeepSeek 开源推理框架 **[DeepInfer](https://github.com/deepseek-ai/DeepInfer)**（固定提交 `8b16149741a4935e1dec8cea27853f273b374c5b`）、并带你用纯 Python + NumPy 从零手写同构教学引擎 `minidi` 的实战技术书（提供**英文版** `chapters_en/` 与**中文版** `chapters/` 完整双语版本）。

全书遵循 **“痛点直觉 → 小数字图解推演 → 真实 C++/Python 源码逐行对照 → 亲手写代码实验验证”** 的认知节奏，彻底讲透现代大模型生产级推理引擎的核心机制。

---

## 本书亮点

- **第一部分（第 1–8 章）：拆解真实的 DeepInfer 生产源码**
  - **全局定位与部署架构**：为什么 DeepInfer 放弃张量并行（TP），选择 DP/CP + Wide-EP + E/P/D 分离？6 万行代码里哪 1 万行是核心骨架？
  - **C++ 调度器与双指针抢占**：为什么 Prefill 步与 Decode 步绝不混跑？多 DP 卡怎么通过 `ALL(70)` / `ANY(100)` 两轮投票对齐步型？为什么 Decode 按 `output_len` 降序后右指针保新（LCFS）、左指针优先抢占长输出大户？
  - **双池显存账本**：为什么压缩 KV 走 128-token 分页块池，而滑动窗口（SWA）走定长环形状态池？为什么压缩状态 Ring 是 `ratio + 5`，而主模型 SWA Ring 是 `128 + 6 = 134`？
  - **零开销异步调度（ZOS）**：CPU 在还不知道第 $N$ 步采样结果时超前调度第 $N+1$ 步，如何通过双变量在途计数、GPU 端 `remap` 与 `compute_seqlens_k_zos` 公式修正在途长度？
  - **DFlash 投机解码与变长验证**：如何用一次并行前向提取 5 个草稿特征、再用马尔可夫头与置信度头逐步采样？如何结合 CUDA Graph 阶梯耗时曲线贪心分配每步验证预算？
  - **YoCo 双段架构与 EPLB 负载均衡**：为什么 Prefill 中间块可以直接跳过后半段 Consumer 层？如何通过历史步峰值贪心复制与放置冗余专家消除 MoE 木桶效应？
- **第二部分（第 9–16 章）：从 0 到 1 手写 CPU 教学引擎 `minidi`**
  - 用 **870 行纯 Python + NumPy**（零 GPU 依赖）分 7 个阶段完整复刻 `TinyLM`、`PagedKV`、`BlockManager`、`Scheduler`、`ZOSEngine`、投机解码与变长验证、`SWARing`、`EPLB` 与多卡投票。
  - 配套 **64 个 `pytest` 单元测试**（笔记本 CPU 1 秒内跑完），真实复盘开发过程中踩到的“缺少 80% 容量闸门死锁”“分块 Prefill 与抢占重算队头互等”“在途请求被抢占计数错乱”等工程陷阱。

---

## 全书目录（中文 & English）

| 部分 | 中文版章节 (`chapters/`) | English Edition (`chapters_en/`) | 核心主题与配套源码 |
| --- | --- | --- | --- |
| **序** | [前言](chapters/00_preface.md) | [Preface](chapters_en/00_preface.md) | 为什么要读 DeepInfer、全书双线结构与四步阅读法 |
| **第一部分**<br/>解读 DeepInfer | [第 1 章](chapters/01_overview.md) | [Chapter 1](chapters_en/01_overview.md) | **定位与仓库地图**：E/P/D 分离架构、子模块与 1 万行核心骨架 · `data/deepinfer_loc.json` |
| | [第 2 章](chapters/02_lifecycle.md) | [Chapter 2](chapters_en/02_lifecycle.md) | **一个请求的一生**：四阶段启动、状态机与 `num_tokens = len - 1` 不变量 |
| | [第 3 章](chapters/03_scheduler.md) | [Chapter 3](chapters_en/03_scheduler.md) | **C++ 调度器**：步型分离、两轮投票、FCFS 分块、LCFS 抢占与尾块缩容 · `csrc/engine/scheduler.hpp` |
| | [第 4 章](chapters/04_memory.md) | [Chapter 4](chapters_en/04_memory.md) | **显存与 KV Cache**：分页块池 vs 定长 State 池、`134` 与 `ratio+5` 推导 · `cache_engine.py` |
| | [第 5 章](chapters/05_zos.md) | [Chapter 5](chapters_en/05_zos.md) | **零开销异步调度（ZOS）**：四阶段流水线、双变量在途计数与 `seqlens_k` 修正 · `worker.py` · `zos.cpp` |
| | [第 6 章](chapters/06_varlen.md) | [Chapter 6](chapters_en/06_varlen.md) | **DFlash 投机与变长验证**：并行提特征 + 马尔可夫头 + 贪心卡档位 · `dflash.py` · `mtp_varlen.hpp` |
| | [第 7 章](chapters/07_model_parallel.md) | [Chapter 7](chapters_en/07_model_parallel.md) | **模型、算子与并行**：YoCo 双段截断、Wide-EP、Mega MoE 与 EPLB · `v4_modules/` · `eplb/policy.py` |
| | [第 8 章](chapters/08_run_compare.md) | [Chapter 8](chapters_en/08_run_compare.md) | **配置、确定性推理与横评**：12 个核心开关、批不变三条件、对比 vLLM/SGLang · `environ.py` |
| **第二部分**<br/>手写 `minidi` | [第 9 章](chapters/09_roadmap.md) | [Chapter 9](chapters_en/09_roadmap.md) | **从 0 到 1 路线图**：四条开发铁律与七阶段演进蓝图 · `examples/quickstart.py` |
| | [第 10 章](chapters/10_minimal_engine.md) | [Chapter 10](chapters_en/10_minimal_engine.md) | **最小引擎**：`TinyLM`、扁平变长批次、`Engine.step` 与按位置确定性采样 · `minidi/model.py` · `engine.py` |
| | [第 11 章](chapters/11_paged_kv.md) | [Chapter 11](chapters_en/11_paged_kv.md) | **手写分页 KV 与块管理**：块表寻址、FIFO 链表与“80% 容量闸门”死锁复盘 · `minidi/blocks.py` · `sequence.py` |
| | [第 12 章](chapters/12_scheduler.md) | [Chapter 12](chapters_en/12_scheduler.md) | **手写调度器与抢占**：双指针抢占、尾块回收与“半截 chunk 队头互等”死锁修复 · `minidi/scheduler.py` |
| | [第 13 章](chapters/13_zos.md) | [Chapter 13](chapters_en/13_zos.md) | **手写异步调度 `ZOSEngine`**：`self.prev` 信息隔离与在途抢占防线 · `minidi/zos.py` |
| | [第 14 章](chapters/14_speculative.md) | [Chapter 14](chapters_en/14_speculative.md) | **手写投机解码与变长验证**：单步并行验证回退与贪心前缀全局最优性证明 · `minidi/varlen.py` |
| | [第 15 章](chapters/15_swa_eplb.md) | [Chapter 15](chapters_en/15_swa_eplb.md) | **手写 SWA Ring、EPLB 与多卡投票**：$W+k-1$ 回退实测与贪心装箱 · `minidi/swa.py` · `eplb.py` · `vote.py` |
| | [第 16 章](chapters/16_toward_gpu.md) | [Chapter 16](chapters_en/16_toward_gpu.md) | **走向生产级 GPU**：差距清单与五步进阶改造路线 |
| **附录** | [致谢与勘误](chapters/99_credits.md) | [Credits & Notes](chapters_en/99_credits.md) | 致谢、引用来源及上游源码中两处“注释与实现不一致”的核对记录 |

---

## 三分钟快速上手

```bash
# 1. 创建虚拟环境并安装轻量依赖（仅需 numpy、matplotlib、pytest）
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-examples.txt

# 2. 运行全部 64 个单元测试（1 秒内完成）
python3 -m pytest

# 3. 运行快速演示：对比同步引擎、异步 ZOS 引擎与投机解码引擎
python3 examples/quickstart.py

# 4. 在 CPU 上实跑 minidi 生成中英双语全书数据图
python3 scripts/figures/make_plots.py
```

---

## 构建全书（中英双语 网页 / EPUB / PDF）

```bash
# 依赖：pandoc >= 3.0、Node >= 22、Chrome/Chromium、CJK 字体、poppler-utils
PUPPETEER_SKIP_DOWNLOAD=1 npm ci
python3 scripts/check_book.py      # 质量门禁：检查中文与英文全部章节结构、图表数量、固定 commit
python3 build.py                   # 生成中英双语 HTML、EPUB、PDF 到 dist/（默认首页为英文，可一键切换中文）
python3 scripts/check_outputs.py dist --min-figures 28
```

---

## 开源许可

- 仓库中的配套代码（`minidi/`、`tests/`、`examples/`、`scripts/`）采用 **MIT License** 开源（见 [`LICENSE`](LICENSE)）。
- 书稿正文与原创图表采用 **CC BY 4.0** 许可协议发布。
- 书中引用的 DeepInfer 源码片段版权归 DeepSeek-AI 所有（Apache License 2.0）。
