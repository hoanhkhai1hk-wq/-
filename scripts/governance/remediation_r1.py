"""Execute the explicitly authorized R1 synthetic case once, through public APIs.

Application business writes are confined to ingest.import_bytes,
disposition.isolate and quality.run_checks. SQL in this script is SELECT-only.
Existing journals or backups prevent replay, including after partial failure.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from agrolake import db, disposition, ingest, quality

CASE = ROOT / 'work/remediation-r1'
DATABASE = ROOT / 'data/p0/agrolake.sqlite'
BACKUP = DATABASE.with_name('agrolake.sqlite.bak-20261024-pre-remediation')
EXPECTED = {'assets': 12, 'asset_versions': 12, 'quality_checks': 93,
            'dispositions': 5, 'linked_rechecks': 0}
SOURCE = '合成演示资料：场景失配样本'
OPERATOR = '黄伟浩'


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def counts(conn) -> dict:
    result = {t: conn.execute('SELECT COUNT(*) FROM ' + t).fetchone()[0]
              for t in ('assets', 'asset_versions', 'quality_checks', 'dispositions')}
    result['linked_rechecks'] = conn.execute(
        'SELECT COUNT(*) FROM quality_checks WHERE recheck_of_check_id IS NOT NULL'
    ).fetchone()[0]
    return result


def rows(conn, table: str, version_id: int) -> list[dict]:
    return [dict(r) for r in conn.execute(
        'SELECT * FROM ' + table + ' WHERE version_id = ?', (version_id,))]


def save(state: dict) -> None:
    (CASE / 'journal.json').write_text(
        json.dumps(state, ensure_ascii=False, indent=2), encoding='utf-8')


def main() -> None:
    if CASE.exists() or BACKUP.exists():
        raise RuntimeError('R1 directory or backup already exists: inspect it; never replay.')
    with db.connect_readonly(DATABASE) as ro:
        baseline = counts(ro)
        if baseline != EXPECTED:
            raise RuntimeError(f'Baseline mismatch; stopped without backup or business writes: {baseline}')
        before_versions = [dict(r) for r in ro.execute('SELECT * FROM asset_versions')]
        before_checks = [dict(r) for r in ro.execute('SELECT * FROM quality_checks')]
        before_dispositions = [dict(r) for r in ro.execute('SELECT * FROM dispositions')]
        image = dict(ro.execute('SELECT * FROM asset_versions WHERE version_id=1').fetchone())
    assert image['content_type'] == 'image' and image['declared_scene_id'] == 'S1'
    if DATABASE.with_name(DATABASE.name + '-wal').exists():
        raise RuntimeError('Active WAL detected; stopped before copying an incomplete database.')
    database_sha = digest(DATABASE)
    with DATABASE.open('rb') as src, BACKUP.open('xb') as dst:
        shutil.copyfileobj(src, dst)
    assert digest(BACKUP) == database_sha == digest(DATABASE)
    with db.connect_readonly(BACKUP) as copied:
        assert counts(copied) == EXPECTED

    CASE.mkdir(parents=True)
    protected = [ROOT / f for f in (
        'app.py', 'agrolake/ingest.py', 'agrolake/disposition.py',
        'agrolake/quality.py', 'agrolake/semantics.py', 'agrolake/scenes.py',
        'configs/quality_rules.json', 'configs/semantic_rules.json',
        'reports/data/Codex-DATA-20261021-001/frozen-evaluation.json')]
    protected += list((ROOT / 'outputs').rglob('*.pdf'))
    protected += list((ROOT / 'reports/rgb/NEW-EVAL-20261021-001').rglob('*.json'))
    protected_sha = {str(p.relative_to(ROOT)): digest(p) for p in protected}
    sample = ROOT / 'data/p0/samples_synth/s1_note_mismatch.json'
    original = sample.read_bytes()
    payload = json.loads(original)
    assert payload['scene_id'] == 'S9' and payload['synthetic'] is True
    corrected = original.replace(b'"S9"', b'"S1"', 1)
    expected_payload = dict(payload, scene_id='S1')
    assert json.loads(corrected) == expected_payload
    state = {'task': 'R1-20261024', 'nature': '合成演示资料', 'baseline': baseline,
             'backup': str(BACKUP.relative_to(ROOT)), 'backup_sha256': database_sha,
             'protected_before': protected_sha, 'image_version_id': 1,
             'image_scene': 'S1', 'source': SOURCE, 'operator': OPERATOR,
             'input_sha256': digest(sample), 'completed_step': 0}
    save(state)
    conn = db.connect(DATABASE)
    try:
        assert counts(conn) == EXPECTED, 'Database changed since baseline; stopped.'
        problem = ingest.import_bytes(conn, original, sample.name,
            store_root=ROOT / 'data/p0/store', source=SOURCE, scene_id='S1',
            intended_use='R1 合成场景失配整改闭环演示；非农业精度验证')
        checks = [dict(r) for r in quality.latest_checks(conn, problem['version_id'])]
        state.update(problem_import=problem, problem_checks=checks,
                     problem_events=rows(conn, 'import_events', problem['version_id']),
                     completed_step=1)
        save(state)
        mismatch = next((c for c in checks if c['rule_id'] == 'scene_consistency'
                         and c['status'] in ('warn', 'block')), None)
        if mismatch is None:
            raise RuntimeError('Actual rules did not produce scene mismatch; inspect recorded checks before continuing.')
        reason = ('R1：按用户授权确认合成场景失配事实；声明场景S1与影像版本#1一致，'
                  'JSON内容scene_id=S9；隔离原错误记录，修正版另行导入保留历史。')
        isolation = disposition.isolate(conn, problem['version_id'], OPERATOR, reason)
        state.update(isolation=isolation, completed_step=2)
        save(state)

        repaired_path = CASE / 's1_note_mismatch-corrected-S1.json'
        repaired_path.write_bytes(corrected)
        fixed = ingest.import_bytes(conn, corrected, sample.name,
            store_root=ROOT / 'data/p0/store', source=SOURCE, scene_id='S1',
            intended_use='R1 合成场景失配整改闭环演示；非农业精度验证')
        state.update(corrected_path=str(repaired_path.relative_to(ROOT)),
                     corrected_sha256=digest(repaired_path), corrected_import=fixed,
                     corrected_import_checks=[dict(r) for r in quality.latest_checks(conn, fixed['version_id'])],
                     corrected_events=rows(conn, 'import_events', fixed['version_id']),
                     completed_step=3)
        save(state)
        assert fixed['version_id'] != problem['version_id']
        assert fixed['asset_id'] == problem['asset_id']
        assert fixed['event_type'] == 'new_version_same_name'

        # Public quality executor supports explicit previous-check linkage.
        # Recheck the corrected immutable version, not the still-wrong old bytes.
        old_check_id = mismatch['check_id']
        latest = quality.run_checks(conn, fixed['version_id'], ROOT / 'data/p0/store',
                                   recheck_of_map={'scene_consistency': old_check_id})
        rechecked = [dict(r) for r in latest]
        linked = [r for r in rechecked if r['recheck_of_check_id'] == old_check_id]
        state.update(recheck_api='agrolake.quality.run_checks',
                     recheck_of_map={'scene_consistency': old_check_id},
                     recheck_checks=rechecked, linked_checks=linked, completed_step=4)
        save(state)
        assert len(linked) == 1 and linked[0]['status'] == 'pass'
        assert linked[0]['rule_id'] == mismatch['rule_id']
        assert quality.overall_status(latest) == 'pass'

        with db.connect_readonly(DATABASE) as ro:
            old = dict(ro.execute('SELECT * FROM asset_versions WHERE version_id=?',
                                  (problem['version_id'],)).fetchone())
            new = dict(ro.execute('SELECT * FROM asset_versions WHERE version_id=?',
                                  (fixed['version_id'],)).fetchone())
            assert old['status'] == 'isolated' and new['status'] == 'active'
            for table, previous, key in (
                ('asset_versions', before_versions, 'version_id'),
                ('quality_checks', before_checks, 'check_id'),
                ('dispositions', before_dispositions, 'disp_id')):
                actual = {r[key]: dict(r) for r in ro.execute('SELECT * FROM ' + table)}
                assert all(actual[r[key]] == r for r in previous), table + ': old record changed'
            state.update(after=counts(ro), problem_version=old, corrected_version=new,
                         problem_dispositions=rows(ro, 'dispositions', problem['version_id']),
                         corrected_dispositions=rows(ro, 'dispositions', fixed['version_id']))
        assert digest(sample) == state['input_sha256']
        assert digest(BACKUP) == database_sha
        assert {str(p.relative_to(ROOT)): digest(p) for p in protected} == protected_sha
        state.update(completed_step=5, protected_unchanged=True,
                     original_records_unchanged=True, status='verified')
        save(state)
        print(json.dumps({k: state[k] for k in ('status', 'problem_import', 'isolation',
              'corrected_import', 'linked_checks', 'baseline', 'after')}, ensure_ascii=False, indent=2))
    except Exception as exc:
        state.update(status='stopped', error=f'{type(exc).__name__}: {exc}')
        save(state)
        raise
    finally:
        conn.close()


if __name__ == '__main__':
    main()
