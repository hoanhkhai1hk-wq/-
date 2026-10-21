"""Read only developer images/labels, emit contact sheets and screening facts."""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent


def load(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def write_new(p, data):
    with Path(p).open('x', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read_developer(rec, scope):
    allowed = {r['sample_id']: r for group in scope['allowed'].values() for r in group}
    if rec['sample_id'] not in allowed or rec['sample_id'] in scope['frozen_ids_excluded']:
        raise ValueError('STOP: source is outside developer-only scope')
    assert allowed[rec['sample_id']]['image_path'] == rec['image_path']
    ip, jp = ROOT / rec['image_path'], ROOT / rec['json_path']
    assert digest(ip) == rec['image_sha256'] and digest(jp) == rec['json_sha256']
    with Image.open(ip) as im:
        original = im.convert('RGB')
    annotation = load(jp)
    return original, annotation


def panel(rec, scope, overlay=True):
    image, annotation = read_developer(rec, scope)
    if overlay:
        draw = ImageDraw.Draw(image)
        for s in annotation['shapes']:
            if s['label'] in ['Field ridge', 'Road']:
                pts = [tuple(v) for v in s['points']]
                draw.line(pts + pts[:1], fill=(255, 65, 40) if s['label'] == 'Field ridge' else (40, 110, 255), width=3)
    return image


if __name__ == '__main__':
    scope = load(OUT / 'development-scope.json')
    rows = scope['allowed']['new_dev313']
    manifest = {r['sample_id']: r for r in load(ROOT / 'reports/labels/RGB-candidates-DATA-20261021-001/manifest.json')['records']}
    facts = []
    for i, rec in enumerate(rows):
        image, annotation = read_developer(rec, scope)
        derived = manifest[rec['sample_id']]
        with Image.open(ROOT / 'reports/labels/RGB-candidates-DATA-20261021-001' / derived['mask_path']) as im:
            labels = np.array(im)
        shapes = []
        for si, s in enumerate(annotation['shapes']):
            xy = np.array(s['points'], dtype=float)
            shapes.append({'index': si, 'label': s['label'], 'bbox': [float(xy[:, 0].min()), float(xy[:, 1].min()), float(xy[:, 0].max()), float(xy[:, 1].max())],
                           'touches_crop_edge': bool((xy < 1).any() or (xy > 511).any())})
        facts.append({'review_index': i + 1, 'sample_id': rec['sample_id'], 'image_path': rec['image_path'],
                      'labels': dict(__import__('collections').Counter(s['label'] for s in annotation['shapes'])),
                      'class_pixels': rec['class_pixels'], 'ignore_pixels': int((labels == 255).sum()),
                      'conflict_pixels': derived['pixel_stats']['conflict_pixels'],
                      'ridge_width_proxy': rec['ridge_equivalent_width_pixels'], 'shapes': shapes,
                      'preliminary_status': '待逐图初筛，不是语义已验收'})
    for start in range(0, len(rows), 36):
        sheet = Image.new('RGB', (1200, 1380), 'white')
        draw = ImageDraw.Draw(sheet)
        for offset, rec in enumerate(rows[start:start + 36]):
            x, y = offset % 6 * 200, offset // 6 * 230
            im = panel(rec, scope)
            im.thumbnail((198, 198))
            sheet.paste(im, (x, y + 32))
            draw.text((x + 2, y + 2), f'{start + offset + 1:03d} {Path(rec["image_path"]).stem}', fill='black')
            counts = facts[start + offset]['labels']
            draw.text((x + 2, y + 17), f'F{counts.get("Field", 0)} B{counts.get("Field ridge", 0)} R{counts.get("Road", 0)}', fill='black')
        sheet.save(OUT / f'new313-contact-{start // 36 + 1:02d}.jpg', quality=94)
    write_new(OUT / 'screening-facts.json', {'images': len(facts), 'records': facts,
              'legend': 'red existing ridge outlines, blue existing road outlines; no new labels',
              'new_inference': 0, 'frozen_images_used': 0})
    print('313 developer pictures prepared in9 contact sheets; no label changes')
