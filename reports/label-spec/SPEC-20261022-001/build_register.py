"""Record Codex's developer-only visual first-pass questions; no label adjudication."""
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent

GROUPS = {
    '通道用途与材质待核': ('T01',
        '线状通道及两侧带状区的通行用途/材质不明，不能只凭浅色判道路，需分清水泥田埂、路面主体与沟渠。',
        '可核实通行且明显宽于相邻田埂的土路才建议道路；窄田界土带建议田埂；证据不足建议255未知，仅为待审建议。',
        'r36_c42 r36_c43 r37_c28 r37_c29 r37_c42 r39_c47 r40_c15 r40_c36 r42_c17 r43_c17 r44_c18 r45_c19 r46_c20 r47_c37 r48_c23 r49_c4 r50_c18 r51_c28 r53_c5 r55_c6 r57_c12'),
    '条带加宽或交汇处用途待核': ('T02',
        '既有田埂轮廓在交汇、加宽或不规则区域包络较宽；是否只是分隔田界，还是可通行土路/作业区，缺少可核用途证据。',
        '确认分隔田块的窄带才建议田埂；可通行且相对更宽的土路建议道路；非条带田面建议田块；不能确定建议255，均不改判。',
        'r41_c40 r42_c38 r45_c35 r45_c36 r45_c37 r47_c32 r48_c20 r50_c27 r51_c17 r52_c26 r55_c8 r56_c13'),
    '土色条带与机耕纹理待核': ('B01',
        '土色/浅绿条带与田内平行机耕纹理或田面色差相近，现有轮廓对应的独立田界作用需要复核；不是已认定错标。',
        '若只是田面纹理或田内非条带裸土建议田块；若有明确田界分隔作用的窄土条带建议田埂；证据不足建议255，仅登记。',
        'r37_c26 r38_c45 r39_c35 r40_c31 r41_c37 r42_c28 r42_c31 r42_c32 r43_c18 r46_c32 r46_c33 r47_c15 r47_c25 r47_c35 r48_c22 r48_c24 r48_c31 r49_c22 r49_c25 r50_c21 r50_c22 r50_c24 r51_c21 r51_c29'),
    '车辙与土路待核': ('T03',
        '田面内弯曲车辙连接路口，单幅无法证明它是固定交通土路还是田内作业转向；不能以车辙自动认路。',
        '田内作业纹理建议随田块；只有可核实的可通行且明显更宽土路才建议道路；证据不足建议255，本轮不写回。',
        'r52_c29'),
    '裁切边角的条带作用待核': ('B02',
        '条带主要位于切片边角，连续性及相邻田块/路肩关系受裁切限制，需要同场景上下文佐证，不把裁切本身当错误。',
        '取得同场景上下文后按田界窄带/田面/道路主体分别建议田埂/田块/道路；当前不能确定的区域建议255，不改旧标签。',
        'r36_c37 r37_c15 r38_c34 r39_c40 r40_c30 r44_c14 r52_c28 r57_c13'),
    '田界绿带与道路普通绿化待核': ('B03',
        '道路旁绿带与靠田侧窄带相邻，需核实被标部分是否确实分隔田块，不能把全部路旁绿化自动算田埂。',
        '有明确田界分隔作用才建议田埂；普通道路绿化不能由本规则自动归类，无法确定建议255，保留既有标注。',
        'r43_c7 r45_c4 r51_c6'),
}


if __name__ == '__main__':
    scope = json.loads((OUT / 'development-scope.json').read_text(encoding='utf-8'))
    facts = json.loads((OUT / 'screening-facts.json').read_text(encoding='utf-8'))['records']
    by_stem = {Path(rec['image_path']).stem: rec for rec in facts}
    selected, cases = set(), []
    for group, (code, question, suggestion, stems) in GROUPS.items():
        for stem in stems.split():
            rec = by_stem['tile_' + stem]
            assert rec['sample_id'] not in selected
            selected.add(rec['sample_id'])
            cases.append({'review_id': f'LS-V1-{len(cases) + 1:03d}', 'sample_id': rec['sample_id'],
                          'image_path': rec['image_path'], 'question_type': code, 'question_group': group,
                          'question': question, 'suggested_decision': suggestion, 'status': '待审',
                          'existing_labels': rec['labels'], 'contact_page': (rec['review_index'] - 1) // 36 + 1,
                          'contact_slot': (rec['review_index'] - 1) % 36 + 1,
                          'normative_version': 'RGB-LABEL-V1', 'existing_label_changed': False})
    frozen = set(scope['frozen_ids_excluded'])
    assert not (selected & frozen)
    coverage = [{'sample_id': rec['sample_id'], 'image_path': rec['image_path'],
                 'review_index': rec['review_index'], 'contact_page': (rec['review_index'] - 1) // 36 + 1,
                 'contact_slot': (rec['review_index'] - 1) % 36 + 1, 'first_pass_complete': True,
                 'screening_result': '疑义待审' if rec['sample_id'] in selected else '本轮未发现登记级疑义，不代表语义验收通过'} for rec in facts]
    result = {'version': 'RGB-LABEL-V1', 'scope_images': 313, 'screened': 313,
              'ambiguity_count': len(cases), 'distinct_images': len(selected),
              'without_registered_question': len(coverage) - len(cases),
              'method': 'Codex逐页目视9张开发侧总览，24张疑义焦点对照复看；依据位置、连续性与可见上下文登记，不用模型、不以颜色阈值自动改判。初筛不代表逐像素或实地语义验收。',
              'status': '全部待审；无已确认改判', 'groups': dict(Counter(c['question_group'] for c in cases)),
              'cases': cases, 'coverage': coverage, 'frozen_images_used': 0, 'labels_modified': 0}
    with (OUT / 'ambiguity-register.json').open('x', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    lines = ['# 标签疑义待审登记｜RGB-LABEL-V1', '',
             f'初筛范围：新批次313张开发图（337候选剔除冻结24），已逐页初筛313张；**登记{len(cases)}条，涉及{len(selected)}张不同图片，全部状态为“待审”**。剩余{313 - len(selected)}张仅为本轮未发现登记级疑义，不是标签已验收。', '',
             '本规范仅供后续新接入验收；此表用V1组织历史开发资料的审核问题，不追溯改判。建议判定是条件性待审意见，不是标签结论；没有把原标签改为255或其它类别。', '',
             '初筛方法：Codex逐页目视9张总览，24张疑义焦点对照复看；原图与已有轮廓只读。非真人实地审核、非逐像素全量语义核验。冻结24张一张未作为例图或登记输入。', '',
             '| 疑义组 | 条数 |', '|---|---:|']
    lines.extend(f'| {name} | {count} |' for name, count in result['groups'].items())
    lines += ['', '全部来源为`0413-r36c23/`，图片编号省略相同前缀；页/格指向[对照总览](../reports/label-spec/SPEC-20261022-001/)的`new313-contact-XX.jpg`，按行从左到右计格。', '',
              '| 登记编号 | 图片编号 | 疑点 | 建议判定（不改判） | 状态 | 页/格 |', '|---|---|---|---|---|---|']
    short = {
        'T01': ('线状通道的通行用途/材质待核，需区分路面、水泥田埂和沟渠', '可通行且明显更宽土路→道路；窄田界土带→田埂；证据不足→255建议'),
        'T02': ('田埂交汇/加宽区域是否为通行土路或作业区待核', '核用途与相对宽度；田界窄带→田埂，宽通行土路→道路，未明→255建议'),
        'B01': ('土色条带与田内机耕纹理/色差相近，独立田界作用待核', '田面纹理→田块；明确田界窄条→田埂；未明→255建议'),
        'T03': ('路口旁车辙是田内作业转向还是固定土路待核', '田内作业纹理→田块；满足通行与相对宽度条件才建议道路；未明→255建议'),
        'B02': ('条带仅在裁切边角，连续性和田块/路肩关系不足', '补同场景上下文后再提类别建议；当前未明区域→255建议'),
        'B03': ('道路绿化与靠田细带相邻，是否具有田界分隔作用待核', '明确田界分隔→田埂；普通绿化不自动并类；未明→255建议'),
    }
    for c in cases:
        question, suggestion = short[c['question_type']]
        lines.append(f'| {c["review_id"]} | {Path(c["image_path"]).name} | {question} | {suggestion} | 待审 | {c["contact_page"]:02d}/{c["contact_slot"]} |')
    lines += ['', '证据：逐图覆盖与完整条件性意见见`reports/label-spec/SPEC-20261022-001/ambiguity-register.json`；原图/原标签摘要见同目录开发范围清单及保护核验。后续处理须新建审核结果，不覆盖本登记或回写原标注。', '']
    report = ROOT / 'outputs/标签疑义待审登记-20261022.md'
    with report.open('x', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print('Registered', len(cases), 'questions on', len(selected), 'developer images; no labels changed')
