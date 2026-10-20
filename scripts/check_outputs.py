#!/usr/bin/env python3
"""check_outputs.py：检查 build.py 的三种产物（网页、EPUB、PDF）是否完整可用。CI 在构建后运行。

  python3 scripts/check_outputs.py dist                          # 正式书稿
  python3 scripts/check_outputs.py build/fixture --min-figures 2 \\
      --expect 龘 --expect 损失 --expect 门禁通过                  # 构建样章

检查：
  O1 三个文件都在：index.html、deepinfer-reader.epub、deepinfer-reader.pdf
  O2 网页顶部有 EPUB / PDF 下载链接，没有残留的 mermaid 源码块
  O3 EPUB 里的图片数 ≥ --min-figures（mermaid 预渲染的图要真的打进 EPUB）
  O4 PDF：能被 pdfinfo 打开；所有字体都嵌入（pdffonts emb=yes）；至少有一个中文字体；
     有页码（页脚 "n / N"）；pdftotext 能抽出 --expect 给的每个字符串（缺字形时会抽不出或变成 .notdef）
依赖 poppler-utils（pdfinfo、pdffonts、pdftotext）。
"""
import argparse
import re
import subprocess
import sys
import zipfile
from pathlib import Path

SLUG = "deepinfer-reader"
CJK_FONT = re.compile(r"(CJK|SC|Hei|Song|Ming|Kai|PingFang|YaHei)", re.I)


def sh(*args):
    return subprocess.run(args, check=True, capture_output=True, text=True).stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dist")
    ap.add_argument("--min-figures", type=int, default=0)
    ap.add_argument("--expect", action="append", default=[])
    a = ap.parse_args()
    d = Path(a.dist)
    html, epub, pdf = d / "index.html", d / f"{SLUG}.epub", d / f"{SLUG}.pdf"
    errs = []
    for f in (html, epub, pdf):
        if not f.is_file() or f.stat().st_size == 0:
            errs.append(f"O1 缺少 {f.name}")
    if errs:
        sys.exit("\n".join(errs))

    page = html.read_text(encoding="utf-8")
    if f'href="{SLUG}.epub"' not in page or f'href="{SLUG}.pdf"' not in page:
        errs.append("O2 网页缺少 EPUB / PDF 下载链接")
    if re.search(r'class="[^"]*\bmermaid\b(?!-figure)', page) or "```mermaid" in page:
        errs.append("O2 网页里残留 mermaid 源码（没有预渲染）")

    with zipfile.ZipFile(epub) as z:
        imgs = [n for n in z.namelist() if n.lower().endswith((".png", ".svg", ".jpg", ".jpeg"))]
    if len(imgs) < a.min_figures:
        errs.append(f"O3 EPUB 只有 {len(imgs)} 张图，少于 {a.min_figures}")

    info = sh("pdfinfo", str(pdf))
    pages = int(re.search(r"^Pages:\s+(\d+)", info, re.M).group(1))
    fonts = sh("pdffonts", str(pdf)).splitlines()[2:]
    names = []
    for line in fonts:
        cols = line.split()
        # 列：name type... encoding emb sub uni objnum gen；emb 在倒数第 5 列
        if len(cols) >= 7:
            names.append(cols[0])
            if cols[-5] != "yes":
                errs.append(f"O4 PDF 字体未嵌入：{cols[0]}")
    if not any(CJK_FONT.search(n) for n in names):
        errs.append("O4 PDF 里没有中文字体")
    text = sh("pdftotext", "-layout", str(pdf), "-")
    if not re.search(rf"(?<!\d){pages}\s*/\s*{pages}(?!\d)", text):
        errs.append("O4 PDF 页脚没有页码")
    for s in a.expect:
        if s not in text:
            errs.append(f"O4 PDF 文本里找不到「{s}」（可能缺字形）")
    if errs:
        print("\n".join(errs))
        sys.exit(f"check_outputs: {len(errs)} 处不通过")
    print(f"check_outputs: OK（PDF {pages} 页，{len(set(names))} 种字体全部嵌入；EPUB {len(imgs)} 张图）")


if __name__ == "__main__":
    main()
