"""Fill a copy of the verified official template through zagens-office.

No DOCX package is written by Python. ZIP/XML access below only reads addresses
and checks the generated document. Every edit is performed by the native CLI.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import xml.etree.ElementTree as ET
import zipfile
from xml.sax.saxutils import escape
from pathlib import Path

from PIL import Image

NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
SECTION_POSITIONS = {"background": 31, "features": 33, "solution": 35,
                     "innovation": 37, "market": 41, "business": 43,
                     "application": 45, "social": 47}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def body(path):
    with zipfile.ZipFile(path) as package:
        return ET.fromstring(package.read("word/document.xml")).find("w:body", NS)


def tail_address(path):
    counts = {}
    last = None
    for child in body(path):
        tag = child.tag.rsplit("}", 1)[-1]
        counts[tag] = counts.get(tag, 0) + 1
        if tag != "sectPr":
            last = f"/body/{tag}[{counts[tag]}]"
    if last is None:
        raise ValueError("Template has no body elements")
    return last


def build(args):
    template = Path(args.template).resolve()
    content_path = Path(args.content).resolve()
    engine = Path(args.engine).resolve()
    output = Path(args.output_dir).resolve()
    if not engine.is_file():
        raise FileNotFoundError("zagens-office engine is required")
    original_hash = sha(template)
    if original_hash != args.template_sha256.lower():
        raise ValueError("Official template SHA256 differs from the reviewed source")
    content = json.loads(content_path.read_text(encoding="utf-8-sig"))
    sections = {s["id"]: s for s in content["sections"]}
    if len(content["sections"]) != 8 or set(sections) != set(SECTION_POSITIONS):
        raise ValueError("Expected exactly eight official body sections")
    body_chars = sum(len(p) for section in sections.values() for p in section["paragraphs"])
    if body_chars > 30000:
        raise ValueError("Official template recommends at most 30000 body characters")
    if output.exists():
        raise FileExistsError("Output directory already exists; choose a new revision")
    output.mkdir(parents=True)
    assets_dir = output / "assets"
    assets_dir.mkdir()
    log_dir = output / "build-audit"
    log_dir.mkdir()
    document = output / "农智数湖-参赛方案审核稿.docx"
    shutil.copyfile(template, document)
    shutil.copyfile(content_path, output / "content-reviewed.json")
    sequence = 0
    audit = []

    def invoke(operation, payload, operations=1):
        nonlocal sequence
        sequence += 1
        request = log_dir / f"{sequence:03d}-request.json"
        request.write_text(json.dumps(payload,
                           ensure_ascii=False, indent=2), encoding="utf-8")
        run = subprocess.run([str(engine), "edit", "--path", str(document),
                              "--workspace", str(output), "--op", operation,
                              "--input-file", str(request)], capture_output=True)
        (log_dir / f"{sequence:03d}-response.json").write_bytes(run.stdout)
        if run.returncode:
            raise RuntimeError(f"Native editor failed at batch {sequence}: {run.stderr.decode(errors='replace')}")
        result = json.loads(run.stdout)
        if not result.get("ok") or result.get("result", {}).get("ok") is False:
            raise RuntimeError(f"Native editor rejected batch {sequence}: {result}")
        audit.append({"batch": sequence, "op": operation, "operations": operations, "ok": True})

    def edit(steps):
        invoke("batch", {"atomic": True, "ops": steps}, len(steps))

    def format_paragraph(target, *, heading=False, keep_next=False, keep_lines=False, size=10.5):
        return [
            {"op": "set_props", "target": target, "props": {
                "align": "left" if heading else "both", "keepNext": keep_next,
                "keepLines": keep_lines, "lineSpacing": 320, "spaceBefore": 80,
                "spaceAfter": 80, "indentFirstLine": 0 if heading else 420}},
            {"op": "set_props", "target": target + "/r[1]", "props": {
                "font": "黑体" if heading else args.body_font, "size": int(size * 2),
                "bold": heading, "color": "#000000"}},
        ]

    def set_text(target, text, **kwargs):
        # The native set_text operation requires a pre-existing text node.
        # Add one through the native XML editor for blank template paragraphs.
        parts = target.removeprefix("/body/").split("/")
        node = body(document)
        for part in parts:
            tag, index = part.rstrip("]").split("[")
            node = node.findall("w:" + tag, NS)[int(index) - 1]
        if node.find(".//w:t", NS) is None:
            action = {"op": "raw_set", "part": "/word/document.xml",
                      "xpath": "/w:document/w:body/" + "/".join("w:" + p for p in parts),
                      "action": "append", "xml": '<w:r xmlns:w="' + NS["w"] + '"><w:t>' + escape(text) + '</w:t></w:r>'}
        else:
            action = {"op": "set_text", "target": target, "text": text}
        return [action] + format_paragraph(target, **kwargs)

    info = content["basic_info"]
    edits = set_text("/body/p[7]", "（本地审核稿，未提交）", heading=True, size=14)
    edits += set_text("/body/p[10]", "本地审核稿 · 证据截至2026-10-21")
    for blank in (8, 9, 11, 12, 13):
        edits += [{"op": "set_props", "target": f"/body/p[{blank}]", "props": {
            "lineSpacing": 20, "spaceBefore": 0, "spaceAfter": 0}}]
    for row, text in enumerate((info["project_name"], info["theme"], info["track"],
                                info["team"], info["fill_date"]), 1):
        edits += set_text(f"/body/tbl[1]/tr[{row}]/tc[2]/p[1]", text, size=12)
    edits += set_text("/body/p[24]", "项目名称（与官网提交材料保持一致）：" + info["project_name"])
    edits += set_text("/body/p[25]", "应用行业及场景：" + info["industry"])
    edits += set_text("/body/p[28]", "；".join(info[k] for k in ("team", "captain", "organization", "phone")))
    edits += set_text("/body/p[50]", "附录A：验收摘要、实际界面与来源索引。合成资料、公开适配样例和用户图片分开说明，不附不存在的测评证书。")
    edits += set_text("/body/p[51]", "附录B：演示材料。配套讲稿与本机操作预览单独保存；【待补：经确认可公开访问的视频地址和二维码】。不以代码仓库或本地路径冒充视频链接。")
    edits += set_text("/body/p[52]", "附录C：提交前待核。团队和队长信息、农业资料及授权、来源独立完整标注评估、新机安装、正式在线问答、视频独立访问和最终字体检查尚未全部完成。本稿不能直接作为最终提交文件。")
    # Checkbox cells are identified from the actual template, not guessed row layouts.
    tables = body(document).findall("w:tbl", NS)
    for ti in (2, 3):
        for ri, row in enumerate(tables[ti - 1].findall("w:tr", NS), 1):
            for ci, cell in enumerate(row.findall("w:tc", NS), 1):
                text = "".join(cell.itertext())
                replacements = {"□计算机视觉": "☑计算机视觉", "□多模态": "☑多模态",
                                "□深度学习": "☑深度学习", "□方案验证阶段": "☑方案验证阶段"}
                selected = next((v for k, v in replacements.items() if k in text), None)
                if selected:
                    edits += set_text(f"/body/tbl[{ti}]/tr[{ri}]/tc[{ci}]/p[1]", selected)
    edit(edits)
    # The template's separate page-break paragraph can spill onto a blank page
    # after a full body page. Attach the break directly to the appendix heading.
    edit([{"op": "raw_set", "part": "/word/document.xml",
           "xpath": "/w:document/w:body/w:p[49]/w:pPr", "action": "append",
           "xml": '<w:pageBreakBefore xmlns:w="' + NS["w"] + '"/>'},
          {"op": "remove", "target": "/body/p[48]"}])
    # Reverse order preserves the original addresses of earlier sections.
    for section_id, pos in reversed(list(SECTION_POSITIONS.items())):
        paragraphs = sections[section_id]["paragraphs"]
        if not paragraphs:
            raise ValueError("Empty body section: " + section_id)
        target = f"/body/p[{pos}]"
        steps = set_text(target, paragraphs[0])
        steps += format_paragraph(f"/body/p[{pos - 1}]", heading=True, keep_next=True, size=12)
        for paragraph in reversed(paragraphs[1:]):
            steps += [{"op": "add", "target": target, "anchor": "after",
                       "element_type": "paragraph", "text": paragraph}]
            steps += format_paragraph(f"/body/p[{pos + 1}]")
        edit(steps)

    def append_paragraph(text, **kwargs):
        edit([{"op": "add", "target": tail_address(document), "anchor": "after",
               "element_type": "paragraph", "text": text}])
        edit(format_paragraph(tail_address(document), **kwargs))

    append_paragraph("验收摘要（2026-10-21，本机原型）", heading=True, keep_next=True, size=12)
    wanted = {"治理目录资产／版本／导入", "质量检查", "历史问答记录", "RGB 成功任务",
              "RGB 派生登记", "相关软件测试", "FTW 治理关联／真人复核／整改"}
    rows = [["项目", "实绩", "统计边界"]]
    rows += [[r["item"], r["value"], r["meaning"]]
             for r in content["evidence_table"] if r["item"] in wanted]
    edit([{"op": "add_table", "target": tail_address(document), "anchor": "after",
           "rows": rows, "props": {"header_row": True}}])
    table_index = len(body(document).findall("w:tbl", NS))
    for ri in range(1, len(rows) + 1):
        steps = []
        for ci in range(1, 4):
            target = f"/body/tbl[{table_index}]/tr[{ri}]/tc[{ci}]/p[1]"
            steps += format_paragraph(target, heading=ri == 1, size=10.5)
            steps += [{"op": "set_props", "target": target, "props": {
                "indentFirstLine": 0, "align": "left", "lineSpacing": 260,
                "spaceBefore": 40, "spaceAfter": 40}}]
        edit(steps)
    append_paragraph("第三方来源与可追溯版本", heading=True, keep_next=True, size=12)
    for reference in content["references"]:
        append_paragraph(f"[{reference['id']}] {reference['title']}。{reference['source']}")
    images = []
    if args.assets:
        plan = json.loads(Path(args.assets).read_text(encoding="utf-8-sig"))
        for i, item in enumerate(plan, 1):
            source = Path(item["source"]).resolve()
            dest = assets_dir / (f"figure-{i}" + source.suffix.lower())
            shutil.copyfile(source, dest)
            append_paragraph(f"图{i}　{item['caption']}", keep_next=True, keep_lines=True)
            with Image.open(dest) as im:
                width, height = im.size
            scale = min(5.5 * 914400 / width, 7.7 * 914400 / height)
            # The top-level image operation resolves a local file; batch expects
            # an already embedded OPC part and must not be used for imports.
            invoke("add_image", {"target": tail_address(document), "anchor": "after",
                   "image": str(dest.relative_to(output)).replace("\\", "/"),
                   "width_emu": int(width * scale), "height_emu": int(height * scale)})
            images.append({"source": str(source), "copied": str(dest.relative_to(output)),
                           "sha256": sha(dest), "caption": item["caption"]})
    # The source template uses a floating page-number box that clips page 10+.
    # Keep a native PAGE field in a normal right-aligned footer in this copy.
    invoke("raw_part_set", {"part": "/word/footer1.xml", "xml":
        '<w:ftr xmlns:w="' + NS["w"] + '"><w:p><w:pPr><w:jc w:val="right"/>'
        '</w:pPr><w:fldSimple w:instr=" PAGE "><w:r><w:rPr><w:sz w:val="18"/>'
        '</w:rPr><w:t>1</w:t></w:r></w:fldSimple></w:p></w:ftr>'})
    if sha(template) != original_hash:
        raise RuntimeError("Original template changed")
    # A success envelope alone is insufficient: verify actual saved paragraphs.
    saved = ["".join(p.itertext()) for p in body(document).findall("w:p", NS)]
    for section in sections.values():
        for paragraph in section["paragraphs"]:
            if saved.count(paragraph) != 1:
                raise RuntimeError("Saved body paragraph is missing or duplicated")
    if len(body(document).findall(".//w:drawing", NS)) < len(images):
        raise RuntimeError("An appendix image is missing")
    audit_path = output / "build-manifest.json"
    manifest = {"template_sha256": original_hash, "original_unchanged": True,
                "content_sha256": sha(content_path), "docx_sha256": sha(document),
                "body_font_requested": args.body_font, "body_size_pt": 10.5, "body_characters": body_chars,
                "draft_only": True, "saved_body_verified": True, "images": images, "batches": audit}
    audit_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"docx": str(document), "manifest": str(audit_path),
                      "batches": sequence, "images": len(images)}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", required=True)
    parser.add_argument("--template-sha256", required=True)
    parser.add_argument("--content", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--assets", help="JSON list of source paths and captions")
    parser.add_argument("--body-font", default="仿宋")
    parser.add_argument("--engine", default=str(Path.home() / ".zagens-pro/bin/zagens-office.exe"))
    build(parser.parse_args())


if __name__ == "__main__":
    main()
