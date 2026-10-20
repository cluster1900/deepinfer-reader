#!/usr/bin/env python3
"""build.py：把 chapters/ 下的 Markdown 编译成三种格式。

  dist/index.html                 网页（GitHub Pages 首页，顶部有 EPUB / PDF 下载链接）
  dist/deepinfer-reader.epub   EPUB 3
  dist/deepinfer-reader.pdf    PDF（A4，目录、页码、书签，中文字体嵌入）

正文里的 ```mermaid 代码块在构建时预渲染成 PNG，三种格式用同一张图，
阅读器和 PDF 都不需要执行 JavaScript。写法：

    ```{.mermaid caption="图 6-1　Qwen3.5-2B 的层结构"}
    flowchart LR
      A --> B
    ```

（也可以只写 ```mermaid，不带图注。）

用法：
  python3 build.py                         # 构建全书
  python3 build.py --src tests/build_fixture --out build/fixture   # 构建测试样章
  python3 build.py --skip-pdf              # 只出网页和 EPUB（没有 Chrome 时）

依赖：pandoc ≥ 3.0；Node ≥ 22.13 并在仓库根目录执行过 `npm ci`
（@mermaid-js/mermaid-cli、puppeteer，用系统 Chrome，不下载浏览器）；
Chrome / Chromium（路径用 CHROME_PATH 指定，默认自动查找）；中文字体 Noto CJK（apt install fonts-noto-cjk）。
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

BASE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(BASE, "assets")
TITLE = "DeepInfer 源码解读与从 0 到 1"
AUTHOR = "Hawk Wu"
SLUG = "deepinfer-reader"
MD_FORMAT = "markdown-blank_before_blockquote+lists_without_preceding_blankline"
FENCE_OPEN = re.compile(r"^(?P<fence>`{3,}|~{3,})\s*(?P<info>mermaid|\{[^}]*\.mermaid\b[^}]*\})\s*$")
CAPTION = re.compile(r'caption="([^"]*)"')
FIG_ID = re.compile(r"#([A-Za-z][\w:-]*)")
MERMAID_CONFIG = {
    "theme": "neutral",
    "fontFamily": '"Noto Sans CJK SC", "Noto Sans SC", "PingFang SC", sans-serif',
    "flowchart": {"htmlLabels": True, "useMaxWidth": False},
}


def run(args, **kw):
    print("+", " ".join(args))
    subprocess.run(args, check=True, cwd=BASE, **kw)


def find_chrome():
    for cand in (os.environ.get("CHROME_PATH"), os.environ.get("PUPPETEER_EXECUTABLE_PATH")):
        if cand and os.path.exists(cand):
            return cand
    for name in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser"):
        path = shutil.which(name)
        if path:
            return path
    return None


def chapters(src):
    files = sorted(f for f in os.listdir(src) if f.endswith(".md"))
    if not files:
        sys.exit(f"{src} 里没有 .md 文件")
    return [os.path.join(src, f) for f in files]


class Mermaid:
    """把 mermaid 代码块渲染成 PNG（按内容哈希缓存），并把代码块换成图片引用。"""

    def __init__(self, work, chrome):
        self.out = os.path.join(work, "mermaid")
        os.makedirs(self.out, exist_ok=True)
        self.chrome = chrome
        self.count = 0
        self.cfg = os.path.join(work, "mermaid-config.json")
        self.pptr = os.path.join(work, "puppeteer-config.json")
        with open(self.cfg, "w", encoding="utf-8") as f:
            json.dump(MERMAID_CONFIG, f, ensure_ascii=False)
        with open(self.pptr, "w", encoding="utf-8") as f:
            json.dump({"executablePath": chrome, "args": ["--no-sandbox", "--disable-gpu"]}, f)
        self.mmdc = os.path.join(BASE, "node_modules", ".bin", "mmdc")

    def render(self, code, where):
        if not self.chrome or not os.path.exists(self.mmdc):
            sys.exit(f"{where}: 有 mermaid 图，但找不到 Chrome 或 mmdc（先在仓库根目录 npm ci，并设置 CHROME_PATH）")
        digest = hashlib.sha256((json.dumps(MERMAID_CONFIG, sort_keys=True) + code).encode()).hexdigest()[:16]
        png = os.path.join(self.out, f"{digest}.png")
        if not os.path.exists(png):
            src = os.path.join(self.out, f"{digest}.mmd")
            with open(src, "w", encoding="utf-8") as f:
                f.write(code)
            try:
                run([self.mmdc, "-q", "-i", src, "-o", png, "-s", "3", "-b", "white",
                     "-c", self.cfg, "-p", self.pptr])
            except subprocess.CalledProcessError:
                sys.exit(f"{where}: mermaid 图渲染失败，检查语法")
        self.count += 1
        with open(png, "rb") as f:
            head = f.read(24)
        width = int.from_bytes(head[16:20], "big") // 3  # -s 3 渲染，按 1x 宽度显示
        return os.path.relpath(png, os.path.dirname(self.out)), width

    def transform(self, path, dest):
        lines = open(path, encoding="utf-8").read().split("\n")
        out, i = [], 0
        while i < len(lines):
            m = FENCE_OPEN.match(lines[i])
            if not m:
                out.append(lines[i])
                i += 1
                continue
            fence, info, start = m.group("fence"), m.group("info"), i + 1
            j = start
            while j < len(lines) and not lines[j].startswith(fence):
                j += 1
            if j == len(lines):
                sys.exit(f"{path}:{i + 1}: mermaid 代码块没有结束标记")
            rel, width = self.render("\n".join(lines[start:j]) + "\n", f"{path}:{i + 1}")
            cap = CAPTION.search(info)
            fid = FIG_ID.search(info)
            alt = cap.group(1) if cap else ""
            attrs = f"#{fid.group(1)} " if fid else ""
            out.append(f"![{alt}]({rel}){{{attrs}.mermaid-figure style=\"width:min({width}px,100%)\"}}")
            i = j + 1
        with open(dest, "w", encoding="utf-8") as f:
            f.write("\n".join(out))


def downloads_snippet(path):
    html = (
        '<p class="downloads">下载：'
        f'<a href="{SLUG}.epub">EPUB</a> · '
        f'<a href="{SLUG}.pdf">PDF</a> · '
        '<a href="https://github.com/cluster1900/deepinfer-reader/releases">全部版本（Releases）</a></p>\n'
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(BASE, "chapters"))
    ap.add_argument("--out", default=os.path.join(BASE, "dist"))
    ap.add_argument("--skip-pdf", action="store_true")
    args = ap.parse_args()
    src_dir, dist = os.path.abspath(args.src), os.path.abspath(args.out)

    if shutil.which("pandoc") is None:
        sys.exit("找不到 pandoc：apt install pandoc 或 brew install pandoc")
    chrome = find_chrome()
    if not args.skip_pdf and not chrome:
        sys.exit("找不到 Chrome/Chromium（设置 CHROME_PATH），或加 --skip-pdf 只出网页和 EPUB")
    if os.path.isdir(dist):
        shutil.rmtree(dist)
    os.makedirs(dist)
    work = os.path.join(BASE, "build", "work")
    if os.path.isdir(os.path.join(work, "md")):
        shutil.rmtree(os.path.join(work, "md"))
    os.makedirs(os.path.join(work, "md"), exist_ok=True)

    mer = Mermaid(work, chrome)
    src = []
    for path in chapters(src_dir):
        dest = os.path.join(work, "md", os.path.basename(path))
        mer.transform(path, dest)
        src.append(dest)
    print(f"mermaid: {mer.count} 张图")

    resource = ":".join([work, src_dir, ASSETS, BASE])
    common = ["-f", MD_FORMAT, "--toc", "--toc-depth=2",
              "--metadata", f"title={TITLE}", "--metadata", f"author={AUTHOR}",
              "--metadata", "lang=zh-CN", "--resource-path", resource]
    css = os.path.join(ASSETS, "style.css")
    snippet = os.path.join(work, "downloads.html")
    downloads_snippet(snippet)

    run(["pandoc", *src, *common, "-s", "--mathml", "--embed-resources", "--css", css,
         "--include-before-body", snippet, "-o", os.path.join(dist, "index.html")])
    epub = os.path.join(dist, f"{SLUG}.epub")
    run(["pandoc", *src, *common, "--mathml", "--split-level=1", "--css", css, "-o", epub])
    if not args.skip_pdf:
        print_html = os.path.join(work, "print.html")
        run(["pandoc", *src, *common, "-s", "--mathml", "--embed-resources",
             "--css", css, "--css", os.path.join(ASSETS, "print.css"), "-o", print_html])
        run(["node", os.path.join(BASE, "scripts", "html2pdf.mjs"), print_html,
             os.path.join(dist, f"{SLUG}.pdf"), TITLE],
            env={**os.environ, "CHROME_PATH": chrome})
    open(os.path.join(dist, ".nojekyll"), "w").close()
    print("OK:", ", ".join(sorted(os.listdir(dist))))


if __name__ == "__main__":
    main()
