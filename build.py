#!/usr/bin/env python3
"""build.py：把 chapters/ 下的 Markdown 编译成三种格式。

  dist/index.html                 网页（GitHub Pages 交互式阅读器，含侧边栏目录、分章/全文切换、主题切换、EPUB/PDF 下载）
  dist/deepinfer-reader.epub      EPUB 3
  dist/deepinfer-reader.pdf       PDF（A4，目录、页码、书签，中文字体嵌入）

正文里的 ```mermaid 代码块在构建时预渲染成 PNG，三种格式用同一张图，
阅读器和 PDF 都不需要执行 JavaScript。写法：

    ```{.mermaid caption="图 6-1　DFlash 投机草稿流水线"}
    flowchart TB
      A --> B
    ```

用法：
  python3 build.py                         # 构建全书
  python3 build.py --src tests/build_fixture --out build/fixture   # 构建测试样章
  python3 build.py --skip-pdf              # 只出网页和 EPUB（没有 Chrome 时）
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys

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
    "fontFamily": '"Noto Sans CJK SC", "Noto Sans SC", "PingFang SC", "Hiragino Sans GB", sans-serif',
    "flowchart": {
        "htmlLabels": True,
        "useMaxWidth": False,
        "nodeSpacing": 28,
        "rankSpacing": 34,
        "padding": 12,
    },
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
    for mac_path in (
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Chromium.app/Contents/MacOS/Chromium",
    ):
        if os.path.exists(mac_path):
            return mac_path
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
        '<div class="downloads">'
        '<span>📚 <b>DeepInfer 源码解读与从 0 到 1</b>（基于 commit <code>8b16149</code> 逐行核对 · 配套 CPU 教学引擎 <code>minidi</code>）</span>'
        '<span>下载离线版：'
        f'<a href="{SLUG}.epub">📱 EPUB 电子书</a> · '
        f'<a href="{SLUG}.pdf">📄 PDF 排版版</a> · '
        '<a href="https://github.com/cluster1900/deepinfer-reader">⭐ GitHub 仓库</a> · '
        '<a href="https://github.com/cluster1900/deepinfer-reader/releases">📦 Releases</a>'
        '</span></div>\n'
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)


def reader_ui_snippet(path):
    """为 GitHub Pages (dist/index.html) 注入交互式阅读器增强脚本（侧边目录搜索、分章/全文切换、主题切换、图片灯箱）。"""
    html = f"""<div class="reader-progress-bar" id="readerProgressBar"></div>
<header class="reader-topbar" id="readerTopbar">
  <div class="reader-brand">
    <button class="reader-btn" id="tocToggleBtn" title="展开/收起目录">☰ 目录</button>
    <span>{TITLE}</span>
    <span class="reader-brand-badge">commit 8b16149</span>
  </div>
  <div class="reader-actions">
    <button class="reader-btn active" id="modeChapterBtn" title="每次只显示一章，底部带上一章/下一章导航">📖 分章阅读</button>
    <button class="reader-btn" id="modeScrollBtn" title="单页展开全书所有章节，连续滚动">📜 全文连读</button>
    <button class="reader-btn" id="themeBtn" title="切换浅色 / 护眼 / 深色主题">🌗 主题</button>
    <a class="reader-btn" href="{SLUG}.epub">📱 EPUB</a>
    <a class="reader-btn" href="{SLUG}.pdf">📄 PDF</a>
  </div>
</header>
<div class="img-lightbox" id="imgLightbox" title="点击任意处关闭">
  <img id="imgLightboxTarget" alt="放大预览" />
</div>
<script>
(function() {{
  const body = document.body;
  const toc = document.getElementById('TOC');
  if (!toc) return;
  body.classList.add('has-reader-ui');

  // 1. 将正文包裹进 <main class="book-main"> 并按 <h1> 切分章节 <section class="chapter-section">
  const main = document.createElement('main');
  main.className = 'book-main';
  const keepIds = new Set(['TOC', 'readerProgressBar', 'readerTopbar', 'imgLightbox']);
  const toMove = [];
  for (const node of Array.from(body.childNodes)) {{
    if (node.nodeType === 1 && (keepIds.has(node.id) || node.tagName === 'SCRIPT')) continue;
    toMove.push(node);
  }}
  body.appendChild(main);

  const chapters = [];
  let currentSection = null;
  for (const node of toMove) {{
    if (node.nodeType === 1 && node.tagName === 'H1') {{
      currentSection = document.createElement('section');
      currentSection.className = 'chapter-section';
      currentSection.dataset.chapterIndex = chapters.length;
      main.appendChild(currentSection);
      chapters.push({{ section: currentSection, h1: node, title: node.textContent.trim(), id: node.id }});
    }}
    if (currentSection) {{
      currentSection.appendChild(node);
    }} else {{
      main.appendChild(node);
    }}
  }}

  // 2. 为每个章节底部添加“上一章 / 下一章”导航栏
  const navFooters = [];
  chapters.forEach((ch, idx) => {{
    const footer = document.createElement('div');
    footer.className = 'chapter-nav-footer';
    if (idx > 0) {{
      const prev = chapters[idx - 1];
      const btn = document.createElement('button');
      btn.className = 'chapter-nav-btn prev';
      btn.innerHTML = '<span class="chapter-nav-label">← 上一章</span><span class="chapter-nav-title">' + prev.title + '</span>';
      btn.onclick = () => selectChapter(idx - 1, true);
      footer.appendChild(btn);
    }}
    if (idx < chapters.length - 1) {{
      const next = chapters[idx + 1];
      const btn = document.createElement('button');
      btn.className = 'chapter-nav-btn next';
      btn.innerHTML = '<span class="chapter-nav-label">下一章 →</span><span class="chapter-nav-title">' + next.title + '</span>';
      btn.onclick = () => selectChapter(idx + 1, true);
      footer.appendChild(btn);
    }}
    ch.section.appendChild(footer);
    navFooters.push(footer);
  }});

  // 3. 在 TOC 顶部插入章节搜索框
  const searchWrap = document.createElement('div');
  searchWrap.className = 'toc-search-wrap';
  searchWrap.innerHTML = '<input type="search" class="toc-search-input" placeholder="🔍 搜索章节或小节标题..." />';
  toc.insertBefore(searchWrap, toc.firstChild);
  const searchInput = searchWrap.querySelector('input');
  searchInput.addEventListener('input', () => {{
    const q = searchInput.value.trim().toLowerCase();
    toc.querySelectorAll('li').forEach(li => {{
      const text = li.textContent.toLowerCase();
      li.style.display = (!q || text.includes(q)) ? '' : 'none';
    }});
  }});

  // 4. 建立元素 ID 到章节索引的映射，支持点击目录自动切换到对应章
  const idToChapter = new Map();
  chapters.forEach((ch, idx) => {{
    if (ch.id) idToChapter.set(ch.id, idx);
    ch.section.querySelectorAll('[id]').forEach(el => idToChapter.set(el.id, idx));
  }});

  let mode = localStorage.getItem('deepinfer_reader_mode') || 'chapter';
  let activeChapter = 0;

  function updateTocHighlight(targetId) {{
    toc.querySelectorAll('a').forEach(a => {{
      const href = decodeURIComponent(a.getAttribute('href') || '');
      a.classList.toggle('active-toc', href === '#' + targetId);
    }});
  }}

  function selectChapter(idx, scrollToTop) {{
    if (idx < 0 || idx >= chapters.length) return;
    activeChapter = idx;
    if (mode === 'chapter') {{
      chapters.forEach((ch, i) => {{
        ch.section.style.display = (i === idx) ? 'block' : 'none';
      }});
      navFooters.forEach(f => f.style.display = 'flex');
      if (scrollToTop) window.scrollTo({{ top: 0, behavior: 'smooth' }});
      updateTocHighlight(chapters[idx].id);
      history.replaceState(null, '', '#' + chapters[idx].id);
    }}
  }}

  function applyMode(newMode) {{
    mode = newMode;
    localStorage.setItem('deepinfer_reader_mode', mode);
    document.getElementById('modeChapterBtn').classList.toggle('active', mode === 'chapter');
    document.getElementById('modeScrollBtn').classList.toggle('active', mode === 'scroll');
    if (mode === 'scroll') {{
      chapters.forEach(ch => ch.section.style.display = 'block');
      navFooters.forEach(f => f.style.display = 'none');
    }} else {{
      selectChapter(activeChapter, false);
    }}
  }}

  document.getElementById('modeChapterBtn').onclick = () => applyMode('chapter');
  document.getElementById('modeScrollBtn').onclick = () => applyMode('scroll');

  toc.addEventListener('click', (e) => {{
    const a = e.target.closest('a');
    if (!a) return;
    const rawHref = a.getAttribute('href') || '';
    if (!rawHref.startsWith('#')) return;
    const targetId = decodeURIComponent(rawHref.slice(1));
    const chIdx = idToChapter.get(targetId);
    body.classList.remove('toc-drawer-open');
    if (chIdx !== undefined && mode === 'chapter') {{
      e.preventDefault();
      const changed = (chIdx !== activeChapter);
      selectChapter(chIdx, false);
      const targetEl = document.getElementById(targetId);
      if (targetEl && targetId !== chapters[chIdx].id) {{
        setTimeout(() => targetEl.scrollIntoView({{ behavior: 'smooth', block: 'start' }}), changed ? 30 : 0);
      }} else {{
        window.scrollTo({{ top: 0, behavior: 'smooth' }});
      }}
      updateTocHighlight(targetId);
      history.replaceState(null, '', '#' + targetId);
    }}
  }});

  // 根据初始 URL Hash 定位章节
  const initHash = decodeURIComponent(location.hash.replace(/^#/, ''));
  if (initHash && idToChapter.has(initHash)) {{
    activeChapter = idToChapter.get(initHash);
  }}
  applyMode(mode);
  if (initHash) {{
    const el = document.getElementById(initHash);
    if (el) setTimeout(() => el.scrollIntoView(), 60);
  }}

  // 5. 主题切换（浅色 -> 护眼羊皮纸 -> 深色）
  const themes = ['light', 'sepia', 'dark'];
  const themeLabels = {{ light: '☀️ 浅色', sepia: '📜 护眼', dark: '🌙 深色' }};
  let curTheme = localStorage.getItem('deepinfer_reader_theme') || 'light';
  function applyTheme(t) {{
    curTheme = t;
    document.documentElement.setAttribute('data-theme', t);
    localStorage.setItem('deepinfer_reader_theme', t);
    document.getElementById('themeBtn').textContent = themeLabels[t] || '🌗 主题';
  }}
  applyTheme(curTheme);
  document.getElementById('themeBtn').onclick = () => {{
    const next = themes[(themes.indexOf(curTheme) + 1) % themes.length];
    applyTheme(next);
  }};

  // 6. 移动端目录抽屉切换
  document.getElementById('tocToggleBtn').onclick = () => {{
    if (window.innerWidth < 1024) {{
      body.classList.toggle('toc-drawer-open');
    }} else {{
      const hidden = toc.style.display === 'none';
      toc.style.display = hidden ? '' : 'none';
      body.style.paddingLeft = hidden ? '' : '3rem';
    }}
  }};

  // 7. 阅读进度条与滚动监听
  const bar = document.getElementById('readerProgressBar');
  window.addEventListener('scroll', () => {{
    const docH = document.documentElement.scrollHeight - window.innerHeight;
    const pct = docH > 0 ? Math.min(100, Math.max(0, (window.scrollY / docH) * 100)) : 0;
    bar.style.width = pct + '%';
  }}, {{ passive: true }});

  // 8. 点击任意图表放大预览（Lightbox）
  const lb = document.getElementById('imgLightbox');
  const lbImg = document.getElementById('imgLightboxTarget');
  main.querySelectorAll('figure img, img.mermaid-figure').forEach(img => {{
    img.addEventListener('click', () => {{
      lbImg.src = img.src;
      lb.classList.add('open');
    }});
  }});
  lb.addEventListener('click', () => lb.classList.remove('open'));
  window.addEventListener('keydown', (e) => {{
    if (e.key === 'Escape') lb.classList.remove('open');
  }});
}})();
</script>
"""
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
    ui_snippet = os.path.join(work, "reader_ui.html")
    downloads_snippet(snippet)
    reader_ui_snippet(ui_snippet)

    run(["pandoc", *src, *common, "-s", "--mathml", "--embed-resources", "--css", css,
         "--include-before-body", snippet, "--include-after-body", ui_snippet,
         "-o", os.path.join(dist, "index.html")])
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
