"""Read-only evidence export for the R2 real-image/annotation grouping."""
from __future__ import annotations

import hashlib
import html
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from agrolake import db, rgb_governance, scenes
from agrolake.rgb_jobs import RgbCandidateJobs


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def table(title: str, rows: list[dict]) -> str:
    if not rows:
        return '<h2>' + html.escape(title) + '</h2><p>无记录</p>'
    columns = list(rows[0])
    return '<h2>' + html.escape(title) + '</h2><div class="scroll"><table><tr>' + ''.join('<th>' + html.escape(k) + '</th>' for k in columns) + '</tr>' + ''.join('<tr>' + ''.join('<td>' + html.escape(str(r.get(k, ''))) + '</td>' for k in columns) + '</tr>' for r in rows) + '</table></div>'


def main() -> None:
    state = json.loads((ROOT / 'work/real-scene-r2/journal.json').read_text(encoding='utf-8'))
    assert state['status'] == 'verified_grouped_incomplete_scene'
    database = ROOT / 'data/p0/agrolake.sqlite'
    before_sha = sha(database)
    job_id = state['rgb_job']['id']
    scene_id = state['registration']['registration']['scene_id']
    vids = [r['version_id'] for r in state['versions']]
    marks = ','.join('?' for _ in vids)
    with db.connect_readonly(database) as conn:
        def read(query, args=()):
            return [dict(r) for r in conn.execute(query, args)]
        group = scenes.scene_members(conn, scene_id)
        coverage = scenes.association_coverage(conn)
        assert coverage == state['after_coverage']
        assert group['missing_roles'] == ['attribute']
        backend = RgbCandidateJobs(project_root=ROOT, data_dir=ROOT / 'data/rgb_candidate', readonly=True)
        registered = rgb_governance.link_status(conn, job_id, backend=backend)
        assert registered['registered'] and registered['intact']
        evidence = {
            'task': 'R2-20261024', 'scene_id': scene_id,
            'facts': state['facts'], 'unknowns': state['unknowns'],
            'source_hashes': {k: v for k, v in state['protected_before'].items() if k.startswith('邓/')},
            'before_coverage': state['before_coverage'], 'after_coverage': coverage,
            'before_counts': state['before_counts'], 'after_counts': state['after_counts'],
            'scene_group': group, 'link_integrity': registered,
            'links': read('SELECT link_id,job_id,scene_id,image_asset_id,image_version_id,stats_asset_id,stats_version_id,source_image_sha256 FROM rgb_governance_links WHERE job_id=?', (job_id,)),
            'versions': read('SELECT version_id,asset_id,content_type,original_filename,source,intended_use,license_status,acquired_at,declared_scene_id,status,object_id FROM asset_versions WHERE version_id IN (' + marks + ') ORDER BY version_id', vids),
            'members': read('SELECT member_id,scene_id,version_id,role,basis,confirm_status FROM scene_members WHERE version_id IN (' + marks + ') ORDER BY member_id', vids),
            'dispositions': state['dispositions'], 'checks': state['checks'], 'events': state['events'],
        }
        assert all(not r['license_status'] and not r['acquired_at'] for r in evidence['versions'])
        with db.connect_readonly(ROOT / state['backup']) as previous:
            for t, key in [('asset_versions', 'version_id'), ('quality_checks', 'check_id'), ('dispositions', 'disp_id'), ('rgb_governance_links', 'link_id')]:
                current = {r[key]: dict(r) for r in conn.execute('SELECT * FROM ' + t)}
                assert all(current[r[key]] == dict(r) for r in previous.execute('SELECT * FROM ' + t))
    assert all(sha(ROOT / p) == value for p, value in state['protected_before'].items())
    assert sha(ROOT / state['backup']) == state['backup_sha256']
    assert sha(ROOT / state['rgb_backup']) == state['rgb_backup_sha256']
    assert sha(database) == before_sha
    evidence['database_sha256'] = before_sha
    evidence['previous_records_and_protected_files_unchanged'] = True
    dest = ROOT / 'outputs/真实场景成组证据-R2-20261024-证据'
    dest.mkdir(parents=True, exist_ok=True)
    (dest / '正式库只读证据.json').write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding='utf-8')
    groups = [{'role': k, 'version_ids': [r['version_id'] for r in v]} for k, v in group['roles'].items()]
    body = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>R2 真实素材成组证据</title><style>body{font:16px/1.6 "Microsoft YaHei",sans-serif;margin:30px;color:#20313a}h1,h2{color:#17685d}.scroll{overflow:auto;margin-bottom:24px}table{border-collapse:collapse}th,td{padding:9px;border:1px solid #ddd;text-align:left;vertical-align:top;min-width:90px;max-width:600px;overflow-wrap:anywhere}th{background:#eff8f3}</style><h1>R2 真实素材成组关联：只读审计快照</h1><p>原图为邓/hy0717_tile_r37_c15.tif，512×512 RGB三通道，每通道8位。同名JSON内嵌图像与原图逐像素一致；Road/Field/Field ridge各1个多边形，共13顶点。未确认地理坐标、原始采集来源、原始业务用途、授权、作物、采集日期、真实地块身份或实测面积。</p><p>既有RGB专用入口实际运行一次并登记派生PNG/统计JSON，再通过正规导入登记原始标注JSON。技术场景：' + html.escape(scene_id) + '。此编号不是实际农田身份。</p><p><strong>覆盖率按应用原函数计算：1/4=25% → 1/5=20%，未达到2/4。</strong>该指标要求影像、CSV属性、文本三类齐全；JSON标注计入text，仍缺attribute。新增场景扩大分母，不能借用其他场景或编造属性抬高分数。</p><p>系统登记的授权unknown占位已通过正规元数据操作清空；该操作自动复检，处置7—10如实留痕。未改格式门禁、判定规则或原始文件，未直接SQL造关联。本HTML是只读导出，不冒充应用原页面。</p>'
    for title, rows in [('应用实际场景角色', groups), ('RGB登记链接', evidence['links']), ('资产版本与空白元数据', evidence['versions']), ('场景成员记录', evidence['members']), ('许可空值修订与自动复检处置', evidence['dispositions']), ('全部本案质检记录', evidence['checks'])]:
        body += table(title, rows)
    (dest / '正式库只读证据.html').write_text(body + '</html>', encoding='utf-8')
    print(json.dumps({'coverage': coverage, 'link_intact': registered['intact'], 'members': evidence['members'], 'protected_files': len(state['protected_before'])}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
