"""One authorized R2 grouping run, through existing RGB and ingestion APIs.

No rule changes, no direct business-write SQL, no invented agricultural fields.
The existing coverage requires image/attribute/text: an annotation JSON is text.
"""
from __future__ import annotations

import base64
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys
import uuid

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from agrolake import db, disposition, ingest, rgb_governance, scenes
from agrolake.rgb_jobs import RgbCandidateJobs

CASE = ROOT / 'work/real-scene-r2'
DATABASE = ROOT / 'data/p0/agrolake.sqlite'
BACKUP = DATABASE.with_name('agrolake.sqlite.bak-20261024-pre-r2')
IMAGE = ROOT / '邓/hy0717_tile_r37_c15.tif'
ANNOTATION = IMAGE.with_suffix('.json')


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(state: dict) -> None:
    (CASE / 'journal.json').write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding='utf-8')


def counts(conn) -> dict:
    return {t: conn.execute('SELECT COUNT(*) FROM ' + t).fetchone()[0]
            for t in ('assets', 'asset_versions', 'import_events', 'quality_checks', 'dispositions', 'rgb_governance_links')}


def short_scenes(conn) -> list[dict]:
    return [{'scene_id': s['scene_id'], 'missing_roles': s['missing_roles'],
             'roles': {k: [m['version_id'] for m in v] for k, v in s['roles'].items()}}
            for s in scenes.list_scenes(conn)]


def main() -> None:
    if CASE.exists():
        raise RuntimeError('R2 journal directory exists: inspect original task; never replay.')
    assert BACKUP.is_file(), 'Required pre-r2 backup must already exist.'
    assert sha(DATABASE) == sha(BACKUP), 'Formal database changed since verified backup; stop.'
    with db.connect_readonly(DATABASE) as ro:
        coverage = scenes.association_coverage(ro)
        assert (coverage['complete_scenes'], coverage['total_scenes'], coverage['coverage']) == (1, 4, 0.25)
        before_counts = counts(ro)
        before_scenes = short_scenes(ro)
        previous = {t: [dict(r) for r in ro.execute('SELECT * FROM ' + t)]
                    for t in ('asset_versions', 'quality_checks', 'dispositions', 'rgb_governance_links')}
    payload = json.loads(ANNOTATION.read_text(encoding='utf-8'))
    with Image.open(IMAGE) as image:
        assert image.mode == 'RGB' and image.size == (512, 512)
        embedded = Image.open(io.BytesIO(base64.b64decode(payload['imageData'])))
        assert image.mode == embedded.mode and image.size == embedded.size
        assert image.tobytes() == embedded.tobytes()
        image_facts = {'format': image.format, 'mode': image.mode,
                       'width': image.width, 'height': image.height,
                       'bands': list(image.getbands()),
                       'tiff_tags': {str(k): str(v) for k, v in image.tag_v2.items()},
                       'embedded_pixels_equal': True}
    shapes = payload.get('shapes', [])
    points = [p for s in shapes for p in s['points']]
    facts = {'image': image_facts,
             'annotation_header': {k: v for k, v in payload.items() if k not in ('shapes', 'imageData')},
             'shape_count': len(shapes), 'labels': dict(Counter(s['label'] for s in shapes)),
             'shape_types': dict(Counter(s['shape_type'] for s in shapes)),
             'point_count': len(points),
             'pixel_bounds': {'xmin': min(p[0] for p in points), 'xmax': max(p[0] for p in points),
                              'ymin': min(p[1] for p in points), 'ymax': max(p[1] for p in points)}}
    protected = [IMAGE, ANNOTATION, ROOT / 'app.py', ROOT / 'agrolake/ingest.py',
                 ROOT / 'agrolake/quality.py', ROOT / 'agrolake/scenes.py',
                 ROOT / 'agrolake/rgb_governance.py', ROOT / 'agrolake/rgb_jobs.py',
                 ROOT / 'configs/quality_rules.json', ROOT / 'configs/semantic_rules.json',
                 ROOT / 'outputs/RGB概率融合候选模型-20261020.json',
                 ROOT / 'reports/data/Codex-DATA-20261021-001/frozen-evaluation.json']
    protected += list((ROOT / 'outputs').rglob('*.pdf'))
    protected += list((ROOT / 'reports/rgb/NEW-EVAL-20261021-001').rglob('*.json'))
    before_sha = {str(p.relative_to(ROOT)): sha(p) for p in protected}
    rgb_db = ROOT / 'data/rgb_candidate/runs.sqlite3'
    rgb_backup = rgb_db.with_name('runs.sqlite3.bak-20261024-pre-r2')
    assert not rgb_backup.exists() and not rgb_db.with_name(rgb_db.name + '-wal').exists()
    with rgb_db.open('rb') as src, rgb_backup.open('xb') as dst:
        shutil.copyfileobj(src, dst)
    assert sha(rgb_backup) == sha(rgb_db)
    CASE.mkdir(parents=True)
    state = {'task': 'R2-20261024', 'before_coverage': coverage,
             'before_counts': before_counts, 'before_scenes': before_scenes,
             'backup': str(BACKUP.relative_to(ROOT)), 'backup_sha256': sha(BACKUP),
             'rgb_backup': str(rgb_backup.relative_to(ROOT)), 'rgb_backup_sha256': sha(rgb_backup),
             'protected_before': before_sha, 'facts': facts,
             'unknowns': {k: None for k in ('acquisition_source', 'original_business_use', 'license_status',
                         'crop', 'acquisition_date', 'real_field_id', 'location', 'crs', 'ground_resolution',
                         'real_area', 'annotation_author', 'semantic_completeness')},
             'completed_step': 1, 'request_id': uuid.uuid4().hex}
    save(state)
    backend = RgbCandidateJobs(project_root=ROOT, data_dir=ROOT / 'data/rgb_candidate')
    job = backend.submit(IMAGE.read_bytes(), IMAGE.name, state['request_id'])
    state.update(rgb_job=job)
    save(state)
    if job['status'] != 'succeeded':
        state.update(status='stopped', reason='Existing RGB route failed; no replay.')
        save(state)
        raise RuntimeError(job.get('error') or 'RGB job did not succeed')
    conn = db.connect(DATABASE)
    try:
        registration = rgb_governance.register(conn, job['id'],
            backend=RgbCandidateJobs(project_root=ROOT, data_dir=ROOT / 'data/rgb_candidate', readonly=True),
            store_root=ROOT / 'data/p0/store')
        state['registration'] = registration
        save(state)
        reg = registration['registration']
        assert reg['source_image_sha256'] == before_sha[str(IMAGE.relative_to(ROOT))]
        # The existing registration stores the unknown marker as 'unknown'.
        # Clear that placeholder through its regular metadata operation because
        # this request explicitly requires missing authorization to remain blank.
        clearings = []
        for vid in (reg['image_version_id'], reg['stats_version_id']):
            clearings.append(disposition.supplement_metadata(
                conn, vid, 'Codex-R2',
                'R2：原始素材授权未提供，按要求清空系统默认unknown占位；不声明获得许可。',
                {'license_status': ''}, ROOT / 'data/p0/store'))
        state['unknown_license_clearings'] = clearings
        save(state)
        note = ingest.import_bytes(conn, ANNOTATION.read_bytes(), ANNOTATION.name,
            store_root=ROOT / 'data/p0/store',
            source='用户指定本地标注文件：邓/hy0717_tile_r37_c15.json',
            scene_id=reg['scene_id'], intended_use=None, license_status=None,
            source_link=None, acquired_at=None)
        state.update(annotation_import=note, completed_step=2)
        save(state)
        with db.connect_readonly(DATABASE) as ro:
            state.update(after_coverage=scenes.association_coverage(ro),
                         after_counts=counts(ro), after_scenes=short_scenes(ro))
            group = scenes.scene_members(ro, reg['scene_id'])
            assert reg['image_version_id'] in [r['version_id'] for r in group['roles']['image']]
            assert note['version_id'] in [r['version_id'] for r in group['roles']['text']]
            assert group['missing_roles'] == ['attribute']
            state['group'] = group
            vids = (reg['image_version_id'], reg['stats_version_id'], note['version_id'])
            marks = ','.join('?' for _ in vids)
            state['versions'] = [dict(r) for r in ro.execute('SELECT * FROM asset_versions WHERE version_id IN (' + marks + ')', vids)]
            state['checks'] = [dict(r) for r in ro.execute('SELECT * FROM quality_checks WHERE version_id IN (' + marks + ') ORDER BY check_id', vids)]
            state['events'] = [dict(r) for r in ro.execute('SELECT * FROM import_events WHERE version_id IN (' + marks + ') ORDER BY event_id', vids)]
            state['members'] = [dict(r) for r in ro.execute('SELECT * FROM scene_members WHERE version_id IN (' + marks + ')', vids)]
            state['dispositions'] = [dict(r) for r in ro.execute('SELECT * FROM dispositions WHERE version_id IN (' + marks + ')', vids)]
            for t, key in [('asset_versions', 'version_id'), ('quality_checks', 'check_id'), ('dispositions', 'disp_id'), ('rgb_governance_links', 'link_id')]:
                current = {r[key]: dict(r) for r in ro.execute('SELECT * FROM ' + t)}
                assert all(current[r[key]] == r for r in previous[t]), t + ': previous record changed'
            assert all(not r['license_status'] and not r['acquired_at'] for r in state['versions'])
            assert state['after_coverage']['complete_scenes'] == 1
            assert state['after_coverage']['total_scenes'] == 5
        assert {str(p.relative_to(ROOT)): sha(p) for p in protected} == before_sha
        assert sha(BACKUP) == state['backup_sha256'] and sha(rgb_backup) == state['rgb_backup_sha256']
        png_sha = reg['png_sha256']
        png_path = ROOT / 'data/p0/store/objects' / png_sha[:2] / png_sha
        with Image.open(IMAGE) as supplied, Image.open(png_path) as derived:
            assert supplied.mode == derived.mode and supplied.size == derived.size
            assert supplied.tobytes() == derived.tobytes()
        state.update(status='verified_grouped_incomplete_scene', completed_step=3,
                     previous_records_unchanged=True, protected_unchanged=True)
        save(state)
        print(json.dumps({'status': state['status'], 'job_id': job['id'],
              'registration': reg, 'annotation_import': note,
              'before_coverage': coverage, 'after_coverage': state['after_coverage'],
              'missing_roles': group['missing_roles']}, ensure_ascii=False, indent=2))
    finally:
        conn.close()


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        journal = CASE / 'journal.json'
        if journal.is_file():
            stopped = json.loads(journal.read_text(encoding='utf-8'))
            if not str(stopped.get('status', '')).startswith('verified'):
                stopped.update(status='stopped', error=f'{type(exc).__name__}: {exc}')
                save(stopped)
        raise
