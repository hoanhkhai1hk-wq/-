"""Read-only R1 verification and self-contained HTML/JSON evidence export."""
from __future__ import annotations

import hashlib
import html
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from agrolake import db


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def table(title: str, records: list[dict]) -> str:
    if not records:
        return f'<h2>{html.escape(title)}</h2><p>无记录</p>'
    columns = list(records[0])
    head = ''.join('<th>' + html.escape(c) + '</th>' for c in columns)
    body = ''.join('<tr>' + ''.join('<td>' + html.escape(str(r.get(c, '')))
                                  + '</td>' for c in columns) + '</tr>' for r in records)
    return f'<h2>{html.escape(title)}</h2><div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def main() -> None:
    journal = json.loads((ROOT / 'work/remediation-r1/journal.json').read_text(encoding='utf-8'))
    assert journal['status'] == 'verified'
    database = ROOT / 'data/p0/agrolake.sqlite'
    before_sha = sha(database)
    original_id = journal['problem_import']['version_id']
    fixed_id = journal['corrected_import']['version_id']
    old_check_id = journal['recheck_of_map']['scene_consistency']
    with db.connect_readonly(database) as conn:
        def read(query: str, params=()) -> list[dict]:
            return [dict(r) for r in conn.execute(query, params)]
        evidence = {
            'case': 'R1-20261024', 'nature': '合成演示资料',
            'database': 'data/p0/agrolake.sqlite', 'database_sha256': before_sha,
            'baseline': journal['baseline'], 'after': journal['after'],
            'versions': read('SELECT version_id,asset_id,version_no,original_filename,declared_scene_id,source,status,object_id FROM asset_versions WHERE version_id IN (?,?) ORDER BY version_id', (original_id, fixed_id)),
            'events': read('SELECT event_id,asset_id,version_id,event_type,original_filename,declared_scene_id,declared_source FROM import_events WHERE version_id IN (?,?) ORDER BY event_id', (original_id, fixed_id)),
            'checks': read('SELECT check_id,version_id,rule_id,status,reason,recheck_of_check_id,rule_version,measured_json,checked_at FROM quality_checks WHERE version_id IN (?,?) ORDER BY check_id', (original_id, fixed_id)),
            'dispositions': read('SELECT disp_id,version_id,action,operator,reason,before_json,after_json,created_at FROM dispositions WHERE version_id IN (?,?) ORDER BY disp_id', (original_id, fixed_id)),
            'recheck_join': read('SELECT old.check_id original_check_id,old.version_id original_version_id,old.rule_id,old.status original_status,new.check_id new_check_id,new.version_id new_version_id,new.status new_status,new.recheck_of_check_id FROM quality_checks new JOIN quality_checks old ON old.check_id=new.recheck_of_check_id WHERE old.check_id=?', (old_check_id,)),
        }
        assert len(evidence['recheck_join']) == 1
        relation = evidence['recheck_join'][0]
        assert relation['original_status'] == 'warn' and relation['new_status'] == 'pass'
        assert relation['original_version_id'] == original_id and relation['new_version_id'] == fixed_id
        assert evidence['versions'][0]['asset_id'] == evidence['versions'][1]['asset_id']
        assert evidence['versions'][0]['status'] == 'isolated'
        assert evidence['versions'][1]['status'] == 'active'
        totals = {t: conn.execute('SELECT COUNT(*) FROM ' + t).fetchone()[0]
                  for t in ('assets', 'asset_versions', 'quality_checks', 'dispositions')}
        totals['linked_rechecks'] = conn.execute('SELECT COUNT(*) FROM quality_checks WHERE recheck_of_check_id IS NOT NULL').fetchone()[0]
        assert totals == journal['after']
        with db.connect_readonly(ROOT / journal['backup']) as old:
            for name, key in [('asset_versions', 'version_id'), ('quality_checks', 'check_id'), ('dispositions', 'disp_id')]:
                current = {r[key]: dict(r) for r in conn.execute('SELECT * FROM ' + name)}
                assert all(current[r[key]] == dict(r) for r in old.execute('SELECT * FROM ' + name))
    evidence['unchanged_backup_records'] = True
    assert sha(ROOT / journal['backup']) == journal['backup_sha256']
    assert all(sha(ROOT / p) == value for p, value in journal['protected_before'].items())
    evidence['protected_code_rules_frozen_json_pdfs_unchanged'] = True
    assert sha(database) == before_sha
    dest = ROOT / 'outputs/整改复测案例-R1-20261024-证据'
    dest.mkdir(parents=True, exist_ok=True)
    (dest / '正式库只读证据.json').write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding='utf-8')
    sections = ''.join(table(title, evidence[key]) for title, key in [
        ('版本关系与最终状态', 'versions'), ('导入事件', 'events'),
        ('复检关联：原警告128 → 新通过142', 'recheck_join'),
        ('全部新增质检记录（含初检和复检）', 'checks'), ('问题处置记录', 'dispositions')])
    text = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>R1 正式库只读证据</title><style>body{font:16px/1.6 "Microsoft YaHei",sans-serif;margin:32px;color:#20313a}h1,h2{color:#17685d}.scroll{overflow:auto;border:1px solid #ddd;margin-bottom:28px}table{border-collapse:collapse;font-size:14px}th,td{padding:10px;border:1px solid #ddd;text-align:left;vertical-align:top;min-width:90px;max-width:460px;overflow-wrap:anywhere}th{background:#eff8f3}p{max-width:1000px}</style><h1>整改复测案例 R1：正式库只读证据</h1><p>性质：合成演示资料。以下是对正式数据库的 SELECT 结果导出，属于审计快照，不冒充应用原有页面或农业精度验证。应用原页面截图另附。</p><p>链路：版本17／检查128警告 → 黄伟浩（按用户明确授权，由Codex调用正规功能执行）／隔离处置6 → 同一资产14的新版本18 → 检查142通过，关联原检查128。原错误内容仍隔离保存。</p><p>基线：12资产、12版本、93检查、5处置、0关联复检。完成后：13资产、14版本、114检查、6处置、1关联复检。</p><p>修正版复测调用既有公开函数 agrolake.quality.run_checks，明确传入 recheck_of_map；未修改判定规则、原检查或原始内容。此函数不产生额外 dispositions 复检动作；跨版本复测的持久化证据是 quality_checks 关联，本案处置新增仅为隔离6。</p>' + sections + '</html>'
    (dest / '正式库只读证据.html').write_text(text, encoding='utf-8')
    print(json.dumps({'verified_relation': relation, 'totals': totals, 'database_unchanged_by_export': True, 'output': str(dest)}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
