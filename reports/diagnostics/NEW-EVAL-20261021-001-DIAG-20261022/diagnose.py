"""Read saved arrays only. No torch import, forward call, fitting or label rewriting."""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
FROZEN = ROOT / 'reports/data/Codex-DATA-20261021-001/frozen-evaluation.json'
NEW = ROOT / 'reports/labels/RGB-candidates-DATA-20261021-001'
OLD = ROOT / 'reports/labels/RGB-candidates-DATA-20261016-005'
RUN = ROOT / 'reports/rgb/NEW-EVAL-20261021-001'


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, data):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def mask(path, expected=None):
    if expected and digest(path) != expected:
        raise ValueError(f'Hash mismatch: {path}')
    with Image.open(path) as im:
        if im.mode != 'L' or im.size != (512, 512):
            raise ValueError(f'Unexpected mask mode/size: {path}')
        data = np.array(im)
    if not np.isin(data, [0, 1, 2, 255]).all():
        raise ValueError('Invalid class ID')
    return data


def boundary_band(truth):
    """Known/known 4-neighbor interfaces, two pixel layers per side.

    The two endpoint pixels are the first layer; one 8-neighbor expansion makes
    the second. Image frame and known/255 interfaces are excluded.
    """
    truth = np.asarray(truth)
    if truth.ndim != 2 or not np.isin(truth, [0, 1, 2, 255]).all():
        raise ValueError('Invalid truth')
    known = truth != 255
    boundary = np.zeros(truth.shape, dtype=bool)
    dif = (truth[1:] != truth[:-1]) & known[1:] & known[:-1]
    boundary[1:] |= dif
    boundary[:-1] |= dif
    dif = (truth[:, 1:] != truth[:, :-1]) & known[:, 1:] & known[:, :-1]
    boundary[:, 1:] |= dif
    boundary[:, :-1] |= dif
    padded = np.pad(boundary, 1)
    band = np.zeros(truth.shape, dtype=bool)
    h, w = truth.shape
    for dy in range(3):
        for dx in range(3):
            band |= padded[dy:dy + h, dx:dx + w]
    return band & known


def classify(truth, prediction):
    truth, prediction = np.asarray(truth), np.asarray(prediction)
    if truth.ndim != 2 or truth.shape != prediction.shape or not np.isin(prediction, [0, 1, 2]).all():
        raise ValueError('Invalid prediction')
    band = boundary_band(truth)
    error = (truth != 255) & (truth != prediction)
    gross_a = (truth == 0) & (prediction == 1)
    gross_b = (truth == 0) & (prediction == 2)
    parts = {'a_field_to_ridge_outside_band': gross_a & ~band,
             'b_field_to_road_outside_band': gross_b & ~band,
             'c_boundary_band_error': error & band,
             'd_other_error_outside_band': error & ~band & ~gross_a & ~gross_b}
    counts = {k: int(v.sum()) for k, v in parts.items()}
    assert sum(counts.values()) == int(error.sum())
    return {'errors': int(error.sum()), 'gross_a_field_to_ridge': int(gross_a.sum()),
            'gross_b_field_to_road': int(gross_b.sum()),
            'a_boundary_subset': int((gross_a & band).sum()),
            'b_boundary_subset': int((gross_b & band).sum()),
            'all_errors_confined_to_band': bool(error.any() and not (error & ~band).any()),
            'valid_pixels': int((truth != 255).sum()), 'band_pixels': int(band.sum()),
            'known_predicted_road': int(((truth != 255) & (prediction == 2)).sum()),
            'all_predicted_road': int((prediction == 2).sum()),
            'road_true_positive': int(((truth == 2) & (prediction == 2)).sum()),
            'road_missed': int(((truth == 2) & (prediction != 2)).sum()),
            'road_false_positive_field': int(((truth == 0) & (prediction == 2)).sum()),
            'road_false_positive_ridge': int(((truth == 1) & (prediction == 2)).sum()),
            **counts}


def equivalent_width(binary):
    """2A/P4 on native raster; proxy affected by crop limits and diagonal orientation."""
    binary = np.asarray(binary, dtype=bool)
    area = int(binary.sum())
    if not area:
        return None
    padded = np.pad(binary, 1)
    perimeter = int(np.count_nonzero(padded[1:] != padded[:-1]) +
                    np.count_nonzero(padded[:, 1:] != padded[:, :-1]))
    return 2 * area / perimeter


def frozen_stats():
    frozen, saved = read_json(FROZEN), read_json(RUN / 'summary.json')
    if digest(FROZEN) != saved['freeze_sha256']:
        raise ValueError('Frozen manifest changed')
    rows = []
    for rec, measured in zip(frozen['records'], saved['records']):
        assert rec['sample_id'] == measured['sample_id']
        truth = mask(ROOT / rec['mask_path'], rec['mask_sha256'])
        pred = mask(RUN / measured['prediction_path'], measured['prediction_sha256'])
        if np.any(pred == 255):
            raise ValueError('Prediction contains255')
        valid = truth != 255
        mat = np.bincount(truth[valid].astype(np.int64) * 3 + pred[valid], minlength=9).reshape(3, 3)
        assert mat.tolist() == measured['confusion_truth_rows_prediction_columns']
        rows.append({'sample_id': rec['sample_id'], **classify(truth, pred)})
    total = {key: sum(int(row[key]) for row in rows) for key in rows[0] if key != 'sample_id'}
    image_counts = {key: sum(int(row[key] > 0) for row in rows) for key in total}
    total['boundary_error_fraction_of_all_errors'] = total['c_boundary_band_error'] / total['errors']
    total['pure_boundary_only_images'] = sum(row['all_errors_confined_to_band'] for row in rows)
    result = {'source': 'saved24 predictions + user-authorized saved derived truth masks only',
              'new_forwards': 0, 'totals': total, 'images': image_counts, 'per_image': rows,
              'interpretation': 'Boundary-associated is a geometric estimate, not proof of a <=2px shift or of label correctness'}
    write_new(OUT / 'frozen-statistics.json', result)
    return result


def developer_stats():
    frozen_ids = {r['sample_id'] for r in read_json(FROZEN)['records']}
    new = [r for r in read_json(NEW / 'manifest.json')['records'] if r['status'] == 'candidate' and r['sample_id'] not in frozen_ids]
    selection = read_json(ROOT / 'reports/rgb/TRN-20261017-003-reviewed-selection/selection.json')
    old_ids = {r['sample_id'] for r in selection['records'] if r['disposition'] == 'experiment_train'}
    old = [r for r in read_json(OLD / 'manifest.json')['records'] if r['sample_id'] in old_ids]
    assert len(new) == 313 and len(old) == 734
    groups = {}
    records = {}
    for name, recs, base in [('new_dev313', new, NEW), ('old_dev734', old, OLD)]:
        rows = []
        for rec in recs:
            assert rec['sample_id'] not in frozen_ids
            a = mask(base / rec['mask_path'], rec['mask_sha256'])
            width = {c: equivalent_width(a == c) for c in [1, 2]}
            rows.append({'sample_id': rec['sample_id'], 'image_path': rec['image_path'],
                         'json_path': rec['json_path'], 'image_sha256': rec['image_sha256'],
                         'json_sha256': rec['json_sha256'],
                         'class_pixels': {str(c): int(np.count_nonzero(a == c)) for c in [0, 1, 2, 255]},
                         'ridge_equivalent_width_pixels': width[1], 'road_equivalent_width_pixels': width[2]})
        stat = {'images': len(rows), 'class_pixels': {str(c): sum(row['class_pixels'][str(c)] for row in rows) for c in [0, 1, 2, 255]}}
        for cls, key in [('ridge', 'ridge_equivalent_width_pixels'), ('road', 'road_equivalent_width_pixels')]:
            widths = [row[key] for row in rows if row[key] is not None]
            stat[cls] = {'images_with_class': len(widths), 'width_q25_median_q75': np.quantile(widths, [.25, .5, .75]).tolist(),
                         'total_class_pixels': stat['class_pixels']['1' if cls == 'ridge' else '2']}
        groups[name] = stat
        records[name] = rows
    result = {'frozen_excluded': len(frozen_ids), 'no_new_inference': True,
              'scope': 'new313 exclude frozen24, legacy734 experiment_train; only saved labels, geometry stats',
              'width_caveat': '2A/P4 pixel proxy, not agronomic physical width; cannot establish semantics or source independence',
              'groups': groups, 'records': records}
    write_new(OUT / 'development-statistics.json', result)
    return result


def make_sheets(development):
    """Development-only label-width quantile examples, before any semantic judgment."""
    frozen = read_json(FROZEN)
    forbidden = {r[k] for r in frozen['records'] for k in ['image_path', 'json_path']}
    groups = development['records']
    selected = {}
    for name, rows in groups.items():
        selected[name] = {}
        for cls, width_key in [('ridge', 'ridge_equivalent_width_pixels'), ('road', 'road_equivalent_width_pixels')]:
            ranked = sorted([r for r in rows if r[width_key] is not None], key=lambda r: (r[width_key], r['sample_id']))
            picks = [ranked[int((len(ranked) - 1) * q)] for q in [.25, .5, .75]]
            selected[name][cls] = picks
            sheet = Image.new('RGB', (1200, 430), 'white')
            draw = ImageDraw.Draw(sheet)
            for i, rec in enumerate(picks):
                assert rec['image_path'] not in forbidden and rec['json_path'] not in forbidden
                ip, jp = ROOT / rec['image_path'], ROOT / rec['json_path']
                assert digest(ip) == rec['image_sha256'] and digest(jp) == rec['json_sha256']
                with Image.open(ip) as im:
                    original = im.convert('RGB')
                base = NEW if name == 'new_dev313' else OLD
                manifest = next(r for r in read_json(base / 'manifest.json')['records'] if r['sample_id'] == rec['sample_id'])
                labels = mask(base / manifest['mask_path'], manifest['mask_sha256'])
                overlay = np.array(original).astype(np.float32)
                for c, color in [(1, [255, 70, 40]), (2, [20, 100, 255])]:
                    overlay[labels == c] = .55 * overlay[labels == c] + .45 * np.array(color)
                # Only ridge/road outline polygons, preserve visibility of underlying appearance.
                marked = Image.fromarray(overlay.astype(np.uint8))
                shapes = read_json(jp)['shapes']
                poly = ImageDraw.Draw(marked)
                for shape in shapes:
                    if shape['label'] in ['Field ridge', 'Road']:
                        pts = [tuple(pt) for pt in shape['points']]
                        poly.line(pts + pts[:1], fill=(255, 50, 30) if shape['label'] == 'Field ridge' else (20, 100, 255), width=2)
                for j, image in enumerate([original, marked]):
                    image.thumbnail((195, 380))
                    sheet.paste(image, (400 * i + 200 * j, 45))
                draw.text((400 * i + 3, 3), rec['sample_id'], fill='black')
                draw.text((400 * i + 3, 18), Path(rec['image_path']).stem, fill='black')
                draw.text((400 * i + 3, 32), f'{cls} width proxy={rec[width_key]:.2f}px', fill='black')
            sheet.save(OUT / f'{name}-{cls}-examples.jpg', quality=96)
    write_new(OUT / 'development-examples.json', selected)


if __name__ == '__main__':
    stats = frozen_stats()
    dev = developer_stats()
    make_sheets(dev)
    print(json.dumps({'frozen_totals': stats['totals'], 'image_counts': stats['images'],
                      'development': dev['groups']}, ensure_ascii=False, indent=2))
