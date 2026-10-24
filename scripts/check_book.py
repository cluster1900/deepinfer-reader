#!/usr/bin/env python3
"""check_book.py：构建前的书稿门禁。CI 第一步运行。

  B1 章节文件名形如 NN_name.md，每章恰好一个一级标题，且标题中不含 $ 行内公式（避免 EPUB nav.xhtml 含 <math>）
  B2 正文引用的图片都存在；全书图（mermaid + 图片）数量 >= MIN_FIGURES
  B3 不出现疑似密钥（OpenAI / GitHub / AWS / Slack token 形态、私钥头）
  B4 commit SHA 只出现本书固定的那一个（防止混入别的版本的行号）
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CH = ROOT / "chapters"
CH_EN = ROOT / "chapters_en"
MIN_FIGURES = 28
SHA = "9bd9cf243fc53ef9afbf8f04d391d989117d6ab9"
SECRET = re.compile(r"(sk-[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|AKIA[0-9A-Z]{16}|xox[baprs]-[A-Za-z0-9-]{10,}|-----BEGIN [A-Z ]*PRIVATE KEY-----)")

errors = []


def check_dir(ch_dir: Path, label: str):
    files = sorted(ch_dir.glob("*.md"))
    if not files:
        errors.append(f"{label} 为空")
        return 0, []
    figures = 0
    for f in files:
        t = f.read_text(encoding="utf-8")
        if not re.fullmatch(r"\d\d_[a-z0-9_]+\.md", f.name):
            errors.append(f"{label}/{f.name}: 文件名不规范")
        prose = re.sub(r"^```.*?^```", "", t, flags=re.M | re.S)
        h1 = [l for l in prose.splitlines() if l.startswith("# ")]
        if len(h1) != 1:
            errors.append(f"{label}/{f.name}: 一级标题 {len(h1)} 个")
        for l in prose.splitlines():
            if l.startswith("#") and "$" in l:
                errors.append(f"{label}/{f.name}: 标题含 $ 公式（会导致 epubcheck OPF-014）: {l}")
        figures += len(re.findall(r"^```\{?\.?mermaid", t, re.M))
        for img in re.findall(r"!\[[^\]]*\]\(([^)]+)\)", t):
            figures += 1
            if not ((ROOT / "assets" / img).exists() or (ch_dir / img).exists()):
                errors.append(f"{label}/{f.name}: 图片不存在 {img}")
        for sha in re.findall(r"\b[0-9a-f]{40}\b", t):
            if sha != SHA:
                errors.append(f"{label}/{f.name}: 出现了别的 commit {sha}")
    if figures < MIN_FIGURES:
        errors.append(f"{label}: 图只有 {figures} 张，少于 {MIN_FIGURES}")
    return figures, files


figs_zh, files_zh = check_dir(CH, "chapters")
figs_en, files_en = (check_dir(CH_EN, "chapters_en") if CH_EN.is_dir() else (0, []))

for f in [*files_zh, *files_en, *ROOT.glob("*.md"), *ROOT.glob("minidi/*.py"), *ROOT.glob("tests/*.py"),
          *ROOT.glob("examples/*.py"), *ROOT.glob("scripts/**/*.py"), *ROOT.glob(".github/workflows/*.yml")]:
    if f.name == "check_book.py":
        continue
    if SECRET.search(f.read_text(encoding="utf-8", errors="ignore")):
        errors.append(f"{f.relative_to(ROOT)}: 疑似密钥")

if errors:
    print("check_book: FAIL\n  " + "\n  ".join(errors))
    sys.exit(1)
print(f"check_book: OK（中文 {len(files_zh)} 章 {figs_zh} 张图；英文 {len(files_en)} 章 {figs_en} 张图）")
