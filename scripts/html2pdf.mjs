// html2pdf.mjs：用系统 Chrome 把 Pandoc 生成的打印版 HTML 打成 PDF（A4、页脚页码、PDF 书签）。
// 用法：node scripts/html2pdf.mjs <print.html> <out.pdf> [书名]
import puppeteer from "puppeteer";
import { pathToFileURL } from "node:url";

const [input, output, title = ""] = process.argv.slice(2);
if (!input || !output) {
  console.error("用法：node scripts/html2pdf.mjs <print.html> <out.pdf> [书名]");
  process.exit(2);
}
const browser = await puppeteer.launch({
  executablePath: process.env.CHROME_PATH,
  args: ["--no-sandbox", "--disable-gpu", "--font-render-hinting=none"],
});
try {
  const page = await browser.newPage();
  await page.goto(pathToFileURL(input).href, { waitUntil: "networkidle0", timeout: 120000 });
  await page.evaluate(async () => { await document.fonts.ready; });
  const esc = (s) => s.replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
  await page.pdf({
    path: output,
    format: "A4",
    printBackground: true,
    outline: true,
    tagged: true,
    displayHeaderFooter: true,
    headerTemplate: `<div style="width:100%;font-size:8px;color:#888;text-align:right;padding:0 15mm;font-family:'Noto Sans CJK SC',sans-serif">${esc(title)}</div>`,
    footerTemplate: `<div style="width:100%;font-size:9px;color:#555;text-align:center;font-family:'Noto Sans CJK SC',sans-serif"><span class="pageNumber"></span> / <span class="totalPages"></span></div>`,
    margin: { top: "18mm", bottom: "18mm", left: "16mm", right: "16mm" },
  });
  console.log("PDF:", output);
} finally {
  await browser.close();
}
