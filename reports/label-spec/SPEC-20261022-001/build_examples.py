"""Developer-only policy illustrations: original/crop displays, never new labels."""
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from prepare_review import ROOT, OUT, load, read_developer, write_new

# Manually reviewed cases. Positive means a policy illustration of the described
# visible region; it does not approve or rewrite the original full-image labels.
PLAN = {
    'C1': ('田面内非条带裸土', [
        ('P1', 'new_dev313', 'tile_r53_c18', [80, 80, 430, 430], '田块0', '裸土位于连续田面内部，未形成独立田界条带'),
        ('P2', 'new_dev313', 'tile_r54_c20', [80, 80, 430, 430], '田块0', '田面内部裸土与耕作纹理，不因土色改变类别'),
        ('E1', 'new_dev313', 'tile_r52_c29', [120, 60, 500, 470], '待审，不能自动判路', '路口旁车辙可能是田内转向，不证明是固定土路'),
    ]),
    'C2': ('沿田界窄土条带', [
        ('P1', 'new_dev313', 'tile_r50_c16', [90, 60, 420, 460], '田埂1', '沿田面之间连续窄带，关注土带而非两侧田面'),
        ('P2', 'new_dev313', 'tile_r54_c23', [45, 35, 470, 290], '田埂1', '两侧田面之间的土色窄分界，不能泛化为全部土色'),
        ('E1', 'new_dev313', 'tile_r42_c38', [25, 50, 490, 490], '待审', '不规则加宽绿土斑块，田界/作业区作用需核验'),
    ]),
    'C3': ('绿植分界', [
        ('P1', 'old_dev734', 'tile_r64_c38', [65, 45, 460, 470], '田埂1', '绿植窄带分隔两侧田面，依据作用而非深绿颜色'),
        ('P2', 'old_dev734', 'tile_r69_c36', [65, 45, 460, 470], '田埂1', '长草分界仍是田埂，不将田内作物当田埂'),
        ('E1', 'new_dev313', 'tile_r43_c7', [0, 0, 512, 512], '待审', '道路绿化与靠田细带需分开，普通路旁绿带不自动入类'),
    ]),
    'C4': ('土路/窄土带的相对宽度分支', [
        ('P1', 'new_dev313', 'tile_r50_c16', [90, 60, 420, 460], '窄土带分支：田埂1', '仅示范窄土带分支，不证明能通行，不充当宽土路道路正例'),
        ('P2', 'new_dev313', 'tile_r54_c23', [45, 35, 470, 290], '窄土带分支：田埂1', '仅示范田界窄土条带；宽土路分支仍缺可核通行证据'),
        ('E1', 'new_dev313', 'tile_r47_c32', [0, 0, 512, 512], '待审', '条带交汇处加宽，不能凭宽度或土色认定可通行道路'),
    ]),
    'C5': ('硬质路面主体', [
        ('P1', 'new_dev313', 'tile_r46_c3', [40, 40, 470, 470], '道路2', '有路面边界、标线及可见车辆的沥青通行主体'),
        ('P2', 'new_dev313', 'tile_r58_c11', [40, 40, 470, 470], '道路2', '连续浅灰硬化交叉路面，与路肩/田边分开'),
        ('E1', 'old_dev734', 'tile_r54_c46', [180, 180, 500, 500], '细边与路面主体分别看', 'E03历史已确认水泥细边可为田埂，不能所有水泥都判道路'),
    ]),
}


def font(size):
    return ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', size)


def wrap(draw, text, xy, fnt, width, fill=(30, 40, 45)):
    x, y = xy
    line = ''
    for char in text:
        if draw.textlength(line + char, font=fnt) > width:
            draw.text((x, y), line, font=fnt, fill=fill)
            y += int(fnt.size * 1.4)
            line = char
        else:
            line += char
    draw.text((x, y), line, font=fnt, fill=fill)


if __name__ == '__main__':
    scope = load(OUT / 'development-scope.json')
    examples = []
    for category, (title, plans) in PLAN.items():
        sheet = Image.new('RGB', (1350, 1050), (247, 248, 246))
        draw = ImageDraw.Draw(sheet)
        draw.text((18, 10), f'{category} {title}｜本项目约定 V1', font=font(27), fill=(22, 42, 45))
        draw.text((18, 52), '仅为验收说明图：上为只读原图，下为关注区域放大；青框不是新增标签。', font=font(18), fill=(60, 70, 70))
        for col, (kind, group, stem, roi, meaning, note) in enumerate(plans):
            rec = next(r for r in scope['allowed'][group] if Path(r['image_path']).stem == stem)
            original, annotation = read_developer(rec, scope)
            x = col * 450
            code = f'{category}-{kind}'
            draw.text((x + 16, 93), f'{code} {"正例" if kind.startswith("P") else "易混例"}', font=font(23), fill=(30, 80, 75) if kind.startswith('P') else (150, 80, 20))
            wrap(draw, rec['image_path'], (x + 16, 130), font(16), 415)
            context = original.copy()
            ImageDraw.Draw(context).rectangle(roi, outline=(0, 210, 230), width=3)
            context.thumbnail((410, 410))
            sheet.paste(context, (x + 16, 185))
            crop = original.crop(roi)
            crop.thumbnail((410, 260))
            sheet.paste(crop, (x + 16, 620))
            wrap(draw, meaning, (x + 16, 900), font(20), 414)
            wrap(draw, note, (x + 16, 940), font(17), 414)
            examples.append({'id': code, 'case': category, 'scenario': title,
                             'role': 'positive' if kind.startswith('P') else 'confusable',
                             'source_scope': group, 'sample_id': rec['sample_id'],
                             'image_path': rec['image_path'], 'image_sha256': rec['image_sha256'],
                             'existing_json_path': rec['json_path'], 'existing_json_sha256': rec['json_sha256'],
                             'roi_xyxy': roi, 'suggested_v1_meaning': meaning,
                             'description': note, 'label_written': False,
                             'wide_passable_dirt_road_positive_claimed': False,
                             'gallery': f'{category}-examples.jpg'})
        if category == 'C4':
            draw.text((18, 1020), '覆盖说明：本页2个正例仅属“窄土带→田埂”分支；“宽且可通行土路→道路”无可证实图。', font=font(17), fill=(140, 70, 15))
        sheet.save(OUT / f'{category}-examples.jpg', quality=95)
    write_new(OUT / 'examples-manifest.json', {'version': 'RGB-LABEL-V1', 'examples': examples,
               'frozen_images_used': 0, 'source_photos_unique': len({e['sample_id'] for e in examples}),
               'case_counts': {c: {'positive': 2, 'confusable': 1} for c in PLAN},
               'coverage_gap': 'C4 positives illustrate narrow-soil-strip branch only; no verified wide-passable-dirt-road Road positive; not fabricated',
               'historical_labels_approved_by_gallery': False})
    lines = ['# RGB标签验收V1对照例图集', '', '所有例图仅来自新313/旧734开发侧，冻结24张未使用。正例指所示区域可用于解释本项目约定，不代表整张历史标签获重新批准；青框仅为关注区域示意，不是新增标注。', '',
             '**证据缺口**：土路组C4的2正例均为“窄土带→田埂”分支，并复用C2同一来源。现有资料未能证明2张“可通行且明显更宽的土路→道路”，不以浅色硬化路面或机耕车辙冒充。', '',
             '| 例图编号 | 情形 | 来源图片 | 角色/说明 |', '|---|---|---|---|']
    for e in examples:
        lines.append(f'| [{e["id"]}]({e["gallery"]}) | {e["scenario"]} | {e["image_path"]} | {"正例" if e["role"] == "positive" else "易混例"}：{e["description"]} |')
    lines += ['', '机器可核来源、SHA256、关注区域及五类数量见examples-manifest.json；初筛覆盖与待审意见见ambiguity-register.json。', '']
    with (OUT / 'README.md').open('x', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print('Five scenario sheets and15 policy examples created, with C4 branch evidence gap explicit')
