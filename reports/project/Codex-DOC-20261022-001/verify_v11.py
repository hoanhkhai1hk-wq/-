"""Read-only verification of the four authorized V10 to V11 changes."""
from pathlib import Path
import copy
import hashlib
import json
import xml.etree.ElementTree as E
import zipfile

REPORT = Path(__file__).resolve().parent
ROOT = REPORT.parents[2]
OUT = ROOT / 'outputs/参赛审核稿-20261022-V11'
PLAN = json.loads((REPORT / 'revision-plan.json').read_text(encoding='utf-8'))
NS = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
W = '{' + NS['w'] + '}'


def text(element):
    return ''.join(n.text or '' for n in element.findall('.//w:t', NS))


def shape(element):
    if element is None:
        return None
    return (element.tag, tuple(sorted(element.attrib.items())),
            element.text if element.tag == W + 't' else (element.text or '').strip(),
            tuple(shape(c) for c in element))


def verify():
    with zipfile.ZipFile(PLAN['source']) as source, zipfile.ZipFile(OUT / '农智数湖-参赛方案审核稿.docx') as revised:
        old = E.fromstring(source.read('word/document.xml'))
        new = E.fromstring(revised.read('word/document.xml'))
        ob, nb = old.find('w:body', NS), new.find('w:body', NS)
        op, np = ob.findall('w:p', NS), nb.findall('w:p', NS)
        assert len(op) == len(np) == 80
        for i in range(80):
            expected = {56: PLAN['new57'], 62: PLAN['new63']}.get(i, text(op[i]))
            assert text(np[i]) == expected, f'Unexpected text at paragraph {i + 1}'
        for i in PLAN['font_positions']:
            for run in np[i - 1].findall('w:r', NS):
                if run.find('w:t', NS) is None:
                    continue
                fonts = run.find('w:rPr/w:rFonts', NS)
                assert fonts is not None and fonts.get(W + 'eastAsia') == '仿宋_GB2312', i
        for i in (57, 63):
            assert shape(op[i - 1].find('w:pPr', NS)) == shape(np[i - 1].find('w:pPr', NS))
        struck = [text(r) for r in np[62].findall('w:r', NS)
                  if r.find('w:rPr/w:strike', NS) is not None]
        assert struck == PLAN['appendix_struck_items']
        oldtables, newtables = ob.findall('w:tbl', NS), nb.findall('w:tbl', NS)
        assert len(oldtables) == len(newtables) == 4
        oldrows, newrows = oldtables[3].findall('w:tr', NS), newtables[3].findall('w:tr', NS)
        assert len(newrows) == len(oldrows) + 1
        for oldrow, newrow in zip(oldrows, newrows):
            assert shape(oldrow) == shape(newrow), 'An original summary row changed'
        assert [text(c) for c in newrows[-1].findall('w:tc', NS)] == PLAN['new_summary_row']
        newtables[3].remove(newrows[-1])
        # Remove exactly the authorized changes, then compare the entire document.
        for i in (57, 63):
            position = list(nb).index(np[i - 1])
            nb.remove(np[i - 1])
            nb.insert(position, copy.deepcopy(op[i - 1]))
        for body in (ob, nb):
            for i in PLAN['font_positions']:
                for run in body.findall('w:p', NS)[i - 1].findall('w:r', NS):
                    properties = run.find('w:rPr', NS)
                    if properties is None:
                        continue
                    fonts = properties.find('w:rFonts', NS)
                    if fonts is not None:
                        properties.remove(fonts)
                    # The native editor creates rPr for runs which previously inherited fonts.
                    if not len(properties) and not properties.attrib:
                        run.remove(properties)
        assert shape(old) == shape(new), 'Unapproved structure or formatting change'
        assert set(source.namelist()) == set(revised.namelist())
        substantive = []
        for name in source.namelist():
            if name == 'word/document.xml' or source.read(name) == revised.read(name):
                continue
            if name.endswith(('.xml', '.rels')):
                if shape(E.fromstring(source.read(name))) == shape(E.fromstring(revised.read(name))):
                    continue
            substantive.append(name)
        assert not substantive, substantive
        for name in revised.namelist():
            if name.endswith(('.xml', '.rels')):
                E.fromstring(revised.read(name))
    result = {
        'passed': True, 'only_paragraph_text_replacements': [57, 63],
        'original_paragraph_count': 80, 'new_table_rows': 1,
        'all_original_summary_rows_unchanged': len(oldrows),
        'font_paragraphs': len(PLAN['font_positions']), 'official_body_paragraphs_exact_font': 20,
        'appendix_struck_items': struck, 'all_other_document_structure_unchanged': True,
        'other_package_parts_semantically_changed': substantive,
        'existing_numbers_outside_authorized_text_unchanged': True,
        'cover_basic_info_video_placeholders_unchanged': True,
        'references_and_images_unchanged': True,
        'v10_source_sha_unchanged': hashlib.sha256(Path(PLAN['source']).read_bytes()).hexdigest() == PLAN['source_sha256'],
        'checker_normalization': 'Ignore font properties and resulting empty run-property containers only at the 34 authorized font paragraphs.'
    }
    assert result['v10_source_sha_unchanged']
    (REPORT / 'structural-verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    return result


def change_list():
    p = PLAN
    struck = p['appendix_struck_items']
    return f'''# V10→V11 改动清单

仅修订下列四处；V10 Word/PDF原件未改。V11仍为审核稿，未正式提交。

## 1. 应用情况第三段

原文：

> {p['old57']}

新文：

> {p['new57']}

数字依据：outputs/新增数据整理与初测-20261021.md、outputs/误分诊断-20261022.md；32.42%/76.74%/7.52%/13.01%按用户要求保留一位小数为32.4%/76.7%/7.5%/13.0%，未改变计分结果。规范与登记依据docs/RGB标签验收规范.md及outputs/标签疑义待审登记-20261022.md。

## 2. 验收摘要表新增一行

原文：无该行；原有7项摘要原样保留。

新文：

| 项目 | 实绩 | 统计边界 |
|---|---|---|
| {p['new_summary_row'][0]} | {p['new_summary_row'][1]} | {p['new_summary_row'][2]} |

FTW原行“FTW 治理关联／真人复核／整改｜1／0／0”及整行统计边界逐字、逐属性保留；软件测试94项等原数字不更新。

## 3. 附录C待核清单

原文：

> {p['old63']}

新文（删除线表示该项工作已完成，不表示对历史标签改判）：

> 附录C：提交前待核。已完成：~~{struck[0]}~~；~~{struck[1]}~~。原“来源独立完整标注评估”由冻结初测部分回应，仍待来源独立及完整标注证据，来源地块关系与标注语义未经证明。团队和队长信息、农业资料及授权、新机安装、正式在线问答、视频独立访问和最终字体检查尚未全部完成。本稿不能直接作为最终提交文件。

V10未将新增整理/诊断分别列为待办，因此本次在附录C明确新增两条已完成并加删除线；不把来源独立完整标注评估或农业资料授权误划为已完成。其余待核事项原样保留；用户列出的五项未完成事项全部保留。

## 4. 正文字体

原文/原属性：原有正文及附录说明、来源条目、图注使用“仿宋”。

新文/新属性：统一为“仿宋_GB2312”；共34个非标题文字段落（含20段正式正文）。字体名称来自真实字体文件内部记录，并非仅改显示名称。

标题、表格、封面与基础信息均保留V10原规格；正文原字号、段距与分页属性不另调整。所有文字除第1、3项及第2项新增表行外保持原样。封面团队/队长/单位/手机号/填报日期及视频地址与二维码占位均未填补；章节结构、参考文献、四张界面图和12/93/5/9/722等既有数字保留。

字体取自[武夷学院公开下载页](https://www.wuyiu.edu.cn/tyjxb/_t345/2024/1204/c1814a102386/page.htm)。仅为本机排版安装到用户字体目录，不把字体文件加入Git。PDF由Microsoft Word重新导出；最终页数及逐页检查结果见本目录验收记录。
'''


if __name__ == '__main__':
    result = verify()
    (OUT / '改动清单.md').write_text(change_list(), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False))
