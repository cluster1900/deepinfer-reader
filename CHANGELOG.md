# 更新日志 (Changelog)

## v0.2.1（2026-10-24）

- **源码基线从 `8b16149` 更新到 `9bd9cf243fc53ef9afbf8f04d391d989117d6ab9`**（2026-10-21，上游 #157），中英两版同步：
  - `8b16149`：官方 Q&A 以 PDF 首发；`774ce24`：改为英文 `doc/inference_deployment_qa.md`、删除 PDF、新增两张配图。全书“官方 Q&A”出处改指 Markdown 版，行号引用按新文件重新对齐（第 4 章 L161–L191、L189–L191；第 7 章 L55–L59、L95–L103）。
  - `9bd9cf2`：`vendor/DeepEP` `93eb6eb` → `f6aa38f`，`build_vendor.sh` 去掉 SSH→HTTPS 重写；第 1 章子模块表与说明已更新。其余 5 个子模块锁定提交未变。
  - 第 7 章补充 EPLB 拓扑（EP4DP4 / EP2DP2）、冗余专家取整规则与 Wide-EP 访存说明。

## v0.2.0（2026-10-20）

- **新增完整英文版（English Edition，`chapters_en/`）**：
  - 将全部 18 个章节（前言、第 1–16 章、附录）与全部 35 张配图（28 张 Mermaid 架构图 + 7 张 `make_plots.py` 实跑数据图）完整翻译为地道英文版；
  - 构建并发布英文版交互式网页阅读器（`index-en.html` / `/en/`）、英文版标准电子书（`deepinfer-reader-en.epub`）与英文版高清 PDF（`deepinfer-reader-en.pdf`），支持顶栏一键切换 **`🌐 English` / `🌐 中文`**。
- **全书重写与知识勘误（第 0–16 章、附录）**：
  - 按“痛点直觉 → 小数字图解推演 → 真实源码逐行印证 → 实验与踩坑复盘”重构全部 18 个章节，提升可读性与叙事连贯性；
  - 完整解析 `DFlash`（DSpark）并行提特征 + 马尔可夫头（`markov_w1/w2`）+ `ConfidenceHead` 的三步投机采样架构（`mtp/dflash.py`）；
  - 彻底厘清主模型 SWA Ring 取 `128 + 6 = 134`（跨 128 边界状态快照 `do_copy_state`、Prefill 历史窗口加载与 YoCo 尾部保留）与压缩状态 Ring 取 `ratio + 5`（单步前向与回退下界 $W + k - 1$）的源码依据；
  - 补全 `cache.num_tokens == get_len() - 1` 核心不变量、ZOS 双变量在途计数（`total_inflight_tokens` / `last_inflight_tokens`）与 `compute_seqlens_k_zos` 修正公式、YoCo 双段截断以及 `_append_slot` 草稿被拒后的尾块缩容回收（同步补入 `minidi/scheduler.py` 与第 64 个单元测试）。
- **重绘全部 Mermaid 架构图（28 张 Mermaid + 7 张实跑数据图，共 35 张）**：
  - 消除原先高达 3341px（3x 下 10023px）或宽高比 18.65 的极端长图/宽图，全部重排为二维网格布局（1x 宽高比均在 `0.79–2.19` 之间）。
- **GitHub Pages 交互式阅读器升级（`build.py`、`assets/style.css`、`assets/print.css`）**：
  - 新增桌面端固定左侧目录、章节实时搜索过滤、“分章阅读 / 全文连读”切换、上一章/下一章底部导航、浅色/护眼/深色三套主题切换、中英双语切换、阅读进度条与图表点击放大灯箱（Lightbox）。

## v0.1.1（2026-10-20）

- 作者署名改为 Hawk Wu（书籍元数据、EPUB/PDF 扉页、网页、README、LICENSE）。
- 授权确认：代码 MIT，正文 CC BY 4.0。

## v0.1.0（2026-10-20）

首个完整版本。

- **第一部分：解读 DeepInfer**（第 1–8 章），对照 commit `8b16149741a4935e1dec8cea27853f273b374c5b` 逐条核对路径与行号：
  仓库地图与代码规模；请求生命周期；C++ 调度器（prefill/decode 分离、跨 rank ALL/ANY 投票、FCFS、按输出长度抢占）；
  显存与 KV Cache（块数计算、0.8 容量闸门、528 B/token 的 V41 格式、SWA 状态槽）；异步调度 ZOS 与 CUDA Graph；
  DSpark 置信度变长验证；模型模块角色、算子库、DP/EP/CP 并行与 EPLB；配置、运行、评测、批不变与 vLLM/SGLang 对比。
- **第二部分：个人从 0 到 1**（第 9–16 章）：CPU 教学引擎 `minidi`（867 行）与 63 个 pytest 用例，
  覆盖分页 KV、调度与抢占、异步调度、投机解码与变长验证、SWA 环形缓冲、EPLB、步型投票；记录了开发中遇到的三个真实死锁/计数错误及修法。
- 28 张图：21 张 mermaid 结构图，7 张由 `scripts/figures/make_plots.py` 从 `minidi` 实跑生成的数据图。
- 输出网页、EPUB、PDF；CI 门禁检查章节结构、图片、疑似密钥、固定 commit，并运行全部测试与示例。
- 已知的不确定之处列在书末“致谢与引用”。
