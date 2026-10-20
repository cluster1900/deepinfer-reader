#!/usr/bin/env python3
"""check_book.py：构建前的书稿门禁。CI 第一步运行。

  B1 章节文件名形如 NN_name.md，每章恰好一个一级标题
  B2 正文引用的图片都存在；全书图（mermaid + 图片）数量 >= MIN_FIGURES
  B3 不出现疑似密钥（OpenAI / GitHub / AWS / Slack token 形态、私钥头）
  B4 commit SHA 只出现本书固定的那一个（防止混入别的版本的行号）
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CH = ROOT / "chapters"
MIN_FIGURES = 28
SHA = "8b16149741a4935e1dec8cea27853f273b374c5b"
SECRET = re.compile(r"(sk-[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|AKIA[0-9A-Z]{16}|xox[baprs]-[A-Za-z0-9-]{10,}|-----BEGIN [A-Z ]*PRIVATE KEY-----)")

errors = []
files = sorted(CH.glob("*.md"))
if not files:
    errors.append("chapters/ 为空")
figures = 0
for f in files:
    t = f.read_text(encoding="utf-8")
    if not re.fullmatch(r"\d\d_[a-z0-9_]+\.md", f.name):
        errors.append(f"{f.name}: 文件名不规范")
    prose = re.sub(r"^```.*?^```", "", t, flags=re.M | re.S)
    h1 = [l for l in prose.splitlines() if l.startswith("# ")]
    if len(h1) != 1:
        errors.append(f"{f.name}: 一级标题 {len(h1)} 个")
    figures += len(re.findall(r"^```\{?\.?mermaid", t, re.M))
    for img in re.findall(r"!\[[^\]]*\]\(([^)]+)\)", t):
        figures += 1
        if not ((ROOT / "assets" / img).exists() or (CH / img).exists()):
            errors.append(f"{f.name}: 图片不存在 {img}")
    for sha in re.findall(r"\b[0-9a-f]{40}\b", t):
        if sha != SHA:
            errors.append(f"{f.name}: 出现了别的 commit {sha}")
for f in [*files, *ROOT.glob("*.md"), *ROOT.glob("minidi/*.py"), *ROOT.glob("tests/*.py"), *ROOT.glob("examples/*.py"),
          *ROOT.glob("scripts/**/*.py"), *ROOT.glob(".github/workflows/*.yml")]:
    if f.name == "check_book.py":
        continue
    if SECRET.search(f.read_text(encoding="utf-8", errors="ignore")):
        errors.append(f"{f.relative_to(ROOT)}: 疑似密钥")
if figures < MIN_FIGURES:
    errors.append(f"图只有 {figures} 张，少于 {MIN_FIGURES}")
if errors:
    print("check_book: FAIL\n  " + "\n  ".join(errors))
    sys.exit(1)
print(f"check_book: OK（{len(files)} 个章节文件，{figures} 张图）")
