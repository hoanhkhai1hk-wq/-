import fs from 'node:fs';
import { pathToFileURL } from 'node:url';
const report = new URL('./', import.meta.url);
const dependencyRoot = 'C:/Users/1234/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/pdfjs-dist/';
const pdfjs = await import(pathToFileURL(dependencyRoot + 'legacy/build/pdf.mjs').href);
const docxAudit = JSON.parse(fs.readFileSync(new URL('native-render-v11.json', report), 'utf8').replace(/^\uFEFF/, ''));
const pdf = await pdfjs.getDocument({
  data: new Uint8Array(fs.readFileSync(docxAudit.pdf)),
  standardFontDataUrl: pathToFileURL(dependencyRoot + 'standard_fonts/').href,
  useSystemFonts: false,
  disableFontFace: true,
}).promise;
const pages = [];
const outside = [];
for (let number = 1; number <= pdf.numPages; number++) {
  const page = await pdf.getPage(number);
  await page.getOperatorList();
  const content = await page.getTextContent();
  const viewport = page.getViewport({ scale: 1 });
  const texts = content.items.filter(item => typeof item.str === 'string');
  const fonts = new Set();
  for (const item of texts) {
    if (!item.str.trim()) continue;
    const [x, y] = [item.transform[4], item.transform[5]];
    if (x < -1 || y < -1 || x + item.width > viewport.width + 2 || y > viewport.height + 2) {
      outside.push({ page: number, text: item.str, x, y, width: item.width });
    }
    try { fonts.add(page.commonObjs.get(item.fontName).name); }
    catch { fonts.add(item.fontName); }
  }
  pages.push({ number, width: viewport.width, height: viewport.height,
    text: texts.map(item => item.str).join(''), fonts: [...fonts].sort() });
}
const joined = pages.map(page => page.text).join('\n');
const compact = joined.replace(/\s+/g, '');
const required = ['548张真实影像', '337张候选', '203张空标注', '8张几何/格式退出',
  '32.4%', '76.7%', '7.5%', '13.0%', '69条疑义待审登记',
  '本批不包装成正式独立测试集', '来源地块关系与标注语义未经证明',
  '固定选图与模型清单的现有标注一致性初测，非独立农业精度',
  '1／0／0', '722', '94项软件测试'];
const missing = required.filter(value => !compact.includes(value.replace(/\s+/g, '')));
if (pdf.numPages !== 12 || missing.length || outside.length || joined.includes('\uFFFD')) {
  throw new Error(JSON.stringify({ pageCount: pdf.numPages, missing, outside, replacementCharacters: joined.includes('\uFFFD') }));
}
const exactFontPages = pages.filter(page => page.fonts.some(font => /FangSong_GB2312/.test(font))).map(page => page.number);
if (!exactFontPages.includes(4) || !exactFontPages.includes(6)) throw new Error('Exact body font not observed in PDF');
const result = { passed: true, pages: pdf.numPages, A4: pages.every(page => Math.abs(page.width - 595.32) < 1 && Math.abs(page.height - 841.92) < 1),
  missingRequiredText: missing, textOutsidePage: outside, replacementCharacters: false, exactFontPages,
  fontNames: [...new Set(pages.flatMap(page => page.fonts))].sort(), pageDetails: pages };
fs.writeFileSync(new URL('pdf-verification.json', report), JSON.stringify(result, null, 2));
fs.writeFileSync(new URL('pdf-text-v11.txt', report), pages.map(page => `第${page.number}页\n${page.text}`).join('\n\n'));
console.log(JSON.stringify({ passed: result.passed, pages: result.pages, A4: result.A4, exactFontPages, textOutsidePage: outside.length }));
await pdf.destroy();
