"""Freeze new-batch samples before inference; measure provisional annotation agreement.

This is not a certification of annotation semantics or source independence.
Only the fixed 20261017 RGB candidate is used; no fitting or model promotion.
"""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCHEMA = 'agrolake_new_batch_frozen_evaluation_v1'
SALT = 'DATA-20261021-001'
SEED = 20261017
SIZE = 24
MODEL_MANIFEST = ROOT / 'outputs/RGB概率融合候选模型-20261020.json'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def confined(rel, parent):
    path = (ROOT / rel).resolve()
    if not path.is_relative_to(Path(parent).resolve()) or not path.is_file():
        raise ValueError(f'Invalid source scope or missing file: {rel}')
    return path


def confusion(truth, prediction):
    """Truth rows and prediction columns; 255 truth is excluded, never background."""
    truth, prediction = np.asarray(truth), np.asarray(prediction)
    if truth.shape != prediction.shape or truth.ndim != 2:
        raise ValueError('truth and prediction must be equally sized 2D arrays')
    if not np.isin(truth, [0, 1, 2, 255]).all() or not np.isin(prediction, [0, 1, 2]).all():
        raise ValueError('Invalid truth/prediction class IDs')
    valid = truth != 255
    return np.bincount(truth[valid].astype(np.int64) * 3 + prediction[valid], minlength=9).reshape(3, 3)


def metrics(matrix):
    matrix = np.asarray(matrix)
    if matrix.shape != (3, 3) or not np.issubdtype(matrix.dtype, np.integer) or (matrix < 0).any():
        raise ValueError('Expected nonnegative integer 3x3 confusion')
    total = int(matrix.sum())
    divide = lambda a, b: float(a / b) if b else None
    classes = []
    for c, name in enumerate(['Field', 'Field ridge', 'Road']):
        tp = int(matrix[c, c])
        support, predicted = int(matrix[c].sum()), int(matrix[:, c].sum())
        classes.append({'id': c, 'name': name, 'support_pixels': support,
                        'predicted_pixels_on_labeled_area': predicted,
                        'precision': divide(tp, predicted), 'recall': divide(tp, support),
                        'iou': divide(tp, support + predicted - tp)})
    ious = [c['iou'] for c in classes if c['iou'] is not None]
    return {'confusion_truth_rows_prediction_columns': matrix.tolist(), 'valid_pixels': total,
            'pixel_accuracy': divide(int(np.trace(matrix)), total),
            'macro_iou_nonempty_union': float(np.mean(ious)) if ious else None,
            'classes': classes}


def load_mask(path):
    with Image.open(path) as im:
        if im.mode != 'L' or im.size != (512, 512):
            raise ValueError(f'Invalid mask mode/size: {path}')
        a = np.array(im)
    if not np.isin(a, [0, 1, 2, 255]).all():
        raise ValueError(f'Invalid mask values: {path}')
    return a


def freeze(candidate_dir, destination):
    candidate_dir, destination = Path(candidate_dir).resolve(), Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    if not candidate_dir.is_relative_to(ROOT / 'reports/labels'):
        raise ValueError('Candidate directory must be in project reports/labels')
    manifest_path = candidate_dir / 'manifest.json'
    manifest = read_json(manifest_path)
    eligible = []
    for rec in manifest['records']:
        if rec['dataset'] != '0413-r36c23' or rec['status'] != 'candidate':
            continue
        image_path = confined(rec['image_path'], ROOT / '0413-r36c23')
        json_path = confined(rec['json_path'], ROOT / '0413-r36c23')
        mask = (candidate_dir / rec['mask_path']).resolve()
        if not mask.is_relative_to(candidate_dir):
            raise ValueError('Mask path escape')
        if digest(image_path) != rec['image_sha256'] or digest(json_path) != rec['json_sha256'] or digest(mask) != rec['mask_sha256']:
            raise ValueError('Source/derived hash mismatch before freeze')
        truth = load_mask(mask)
        coverage = float(np.mean(truth != 255))
        if coverage < 0.95:
            continue
        matched = re.fullmatch(r'tile_r(\d+)_c(\d+)', image_path.stem)
        if not matched:
            raise ValueError('Unexpected tile coordinate name')
        row, col = map(int, matched.groups())
        eligible.append({'sample_id': rec['sample_id'], 'image_path': rec['image_path'],
                         'json_path': rec['json_path'], 'image_sha256': rec['image_sha256'],
                         'json_sha256': rec['json_sha256'], 'mask_path': mask.relative_to(ROOT).as_posix(),
                         'mask_sha256': rec['mask_sha256'], 'coverage': coverage,
                         'source_block_4x4': [row // 4, col // 4],
                         'order_key': hashlib.sha256((rec['image_sha256'] + SALT).encode()).hexdigest()})
    eligible.sort(key=lambda x: x['order_key'])
    selected, blocks = [], set()
    for rec in eligible:
        block = tuple(rec['source_block_4x4'])
        if block not in blocks:
            selected.append(rec)
            blocks.add(block)
        if len(selected) == SIZE:
            break
    selected_ids = {r['sample_id'] for r in selected}
    for rec in eligible:
        if len(selected) == SIZE:
            break
        if rec['sample_id'] not in selected_ids:
            selected.append(rec)
            selected_ids.add(rec['sample_id'])
    if len(selected) != SIZE:
        raise ValueError(f'Need {SIZE} eligible samples, got {len(selected)}')
    model = read_json(MODEL_MANIFEST)
    group = next(m for m in model['models'] if m['seed'] == SEED)
    pins = {p.relative_to(ROOT).as_posix(): digest(p) for p in [MODEL_MANIFEST, ROOT / 'recognition/rgb_candidate.py', Path(__file__), manifest_path]}
    for role in ['baseline_W5', 'retention_003', 'feedback_005']:
        path = confined(group[role]['path'], ROOT / 'reports/rgb')
        pins[path.relative_to(ROOT).as_posix()] = digest(path)
        if pins[path.relative_to(ROOT).as_posix()] != group[role]['sha256']:
            raise ValueError('Model weight hash mismatch')
    frozen = {'schema': SCHEMA, 'task': SALT, 'seed': SEED, 'size': SIZE,
              'eligible_count': len(eligible), 'selected_block_count': len({tuple(x['source_block_4x4']) for x in selected}),
              'independent_source_verified': False, 'semantic_reviewed': False,
              'purpose': '新增资料与既有标注的初步一致性测量，非独立农业准确率',
              'unknown_rule': 'unpainted and cross-class conflicts255 ignored; no inferred background',
              'model_manifest': MODEL_MANIFEST.relative_to(ROOT).as_posix(),
              'pinned_files': pins, 'records': selected}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('x', encoding='utf-8') as f:
        f.write(json.dumps(frozen, ensure_ascii=False, indent=2))
    print('Frozen', SIZE, 'samples; eligible', len(eligible), 'SHA256', digest(destination), flush=True)
    return frozen


def verify_frozen(path, expected_sha):
    if digest(path) != expected_sha:
        raise ValueError('Frozen list SHA256 mismatch')
    frozen = read_json(path)
    if frozen.get('schema') != SCHEMA or frozen.get('seed') != SEED or frozen.get('size') != SIZE or len(frozen['records']) != SIZE:
        raise ValueError('Unexpected frozen evaluation protocol')
    ids = [r['sample_id'] for r in frozen['records']]
    if len(set(ids)) != SIZE or any(not re.fullmatch(r'[A-Za-z0-9_-]{1,80}', x) for x in ids):
        raise ValueError('Invalid or duplicate sample IDs')
    for rel, expected in frozen['pinned_files'].items():
        if digest(confined(rel, ROOT)) != expected:
            raise ValueError(f'Pinned source changed: {rel}')
    for rec in frozen['records']:
        for kind, parent in [('image', ROOT / '0413-r36c23'), ('json', ROOT / '0413-r36c23'), ('mask', ROOT / 'reports/labels')]:
            if digest(confined(rec[kind + '_path'], parent)) != rec[kind + '_sha256']:
                raise ValueError(f'Frozen {kind} source changed: {rec["sample_id"]}')
        load_mask(ROOT / rec['mask_path'])
    return frozen


def color_mask(mask):
    palette = np.array([[60, 190, 70], [245, 200, 35], [155, 60, 215]], dtype=np.uint8)
    rgb = np.full((*mask.shape, 3), 135, dtype=np.uint8)
    for c in range(3):
        rgb[mask == c] = palette[c]
    return Image.fromarray(rgb)


def run(frozen_path, expected_sha, output_dir):
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if not output_dir.is_relative_to(ROOT / 'reports/rgb'):
        raise ValueError('Evaluation output must be in reports/rgb')
    frozen = verify_frozen(frozen_path, expected_sha)
    output_dir.mkdir(parents=True, exist_ok=False)
    write_json(output_dir / 'status.json', {'status': 'running', 'freeze_sha256': expected_sha})
    try:
        from recognition.rgb_candidate import CandidatePredictor, native_repeat2
        predictor = CandidatePredictor(ROOT / frozen['model_manifest'], seed=SEED)
        total = np.zeros((3, 3), dtype=np.int64)
        results = []
        sheets = [Image.new('RGB', (1800, 940), 'white') for _ in range(2)]
        for i, rec in enumerate(frozen['records']):
            prediction = native_repeat2(predictor.predict(ROOT / rec['image_path']))
            truth = load_mask(ROOT / rec['mask_path'])
            mat = confusion(truth, prediction)
            total += mat
            dest = output_dir / (rec['sample_id'] + '.png')
            Image.fromarray(prediction).save(dest)
            result = {'sample_id': rec['sample_id'], 'image_path': rec['image_path'],
                      'prediction_path': dest.name, 'prediction_sha256': digest(dest),
                      'ignore_pixels': int(np.count_nonzero(truth == 255)), **metrics(mat)}
            results.append(result)
            sheet = sheets[i // 12]
            x, y = (i % 3) * 600, ((i % 12) // 3) * 235
            draw = ImageDraw.Draw(sheet)
            draw.text((x + 3, y + 3), Path(rec['image_path']).stem, fill='black')
            with Image.open(ROOT / rec['image_path']) as im:
                original = im.convert('RGB')
            for j, (im, title) in enumerate([(original, 'Original'), (color_mask(truth), 'Existing labels'), (color_mask(prediction), 'Prediction')]):
                im.thumbnail((196, 196))
                sheet.paste(im, (x + j * 200, y + 36))
                draw.text((x + j * 200 + 2, y + 19), title, fill='black')
            write_json(output_dir / 'progress.json', {'completed': i + 1, 'total': SIZE, 'records': results})
            print(f'{i + 1}/{SIZE} {rec["sample_id"]} completed', flush=True)
        verify_frozen(frozen_path, expected_sha)
        for i, sheet in enumerate(sheets):
            sheet.save(output_dir / f'review-{i + 1}.jpg', quality=94)
        summary = {'status': 'complete', 'freeze_sha256': expected_sha, 'runtime': predictor.runtime,
                   'seed': SEED, 'samples': len(results), 'independent_source_verified': False,
                   'semantic_reviewed': False, 'purpose': frozen['purpose'], **metrics(total), 'records': results}
        write_json(output_dir / 'summary.json', summary)
        write_json(output_dir / 'status.json', {'status': 'complete', 'samples': len(results), 'freeze_sha256': expected_sha})
        return summary
    except BaseException as exc:
        write_json(output_dir / 'status.json', {'status': 'failed', 'error': str(exc), 'freeze_sha256': expected_sha})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('freeze')
    prep.add_argument('--candidate-dir', required=True)
    prep.add_argument('--output', required=True)
    execute = sub.add_parser('run')
    execute.add_argument('--frozen', required=True)
    execute.add_argument('--sha256', required=True)
    execute.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.command == 'freeze':
        freeze(args.candidate_dir, args.output)
    else:
        run(args.frozen, args.sha256, args.output)


if __name__ == '__main__':
    main()
