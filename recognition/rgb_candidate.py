"""Candidate offline RGB single-image inference, candidate region table and quality export.

Task INF-20261020-002. Uses only the three verified candidate weight groups from
outputs/RGB概率融合候选模型-20261020.json (seeds 20261017/18/19). No training, no
network, no API model switch. Candidates are connected Field pixels only - they are
NOT verified real field parcels and pixel areas are NOT square meters.

torch is imported lazily: importing this module loads no model and writes no file.
"""
from pathlib import Path
import csv
import hashlib
import io
import json
import os
import platform
import sys

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / 'outputs' / 'RGB概率融合候选模型-20261020.json'
DEFAULT_SEED = 20261017
MANIFEST_SCHEMA = 'agrolake_verified_rgb_candidate_v1'
EXPECTED_TENSORS = 22
EXPECTED_PARAMS = 118307
# Fixed CPU affinity masks per seed, identical to settings.cpu_masks in the audited
# 003/005 task-inputs.json (Codex-TRN-20261020-003 / Codex-TRN-20261020-005).
CPU_MASKS = {20261017: 1, 20261018: 4, 20261019: 16}
THREAD_ENV = {'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
              'OMP_DYNAMIC': 'FALSE', 'MKL_DYNAMIC': 'FALSE', 'PYTHONHASHSEED': '0'}
CSV_HEADER = ['field_id', 'area_pixels', 'bbox_x0', 'bbox_y0', 'bbox_x1', 'bbox_y1',
              'centroid_x', 'centroid_y', 'edge_touch', 'small_patch']
CLASS_NAMES = {0: 'Field', 1: 'Field ridge', 2: 'Road'}
NOTICE = ('候选区域仅为class0四邻接连通块，非独立精度、非已核实真实地块；'
          '无比例尺或地理参考，像素面积不能换算为平方米。')


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def _apply_affinity(mask):
    if platform.system() != 'Windows':
        raise RuntimeError('CandidatePredictor requires the Windows affinity runtime.')
    import ctypes
    api = ctypes.WinDLL('kernel32', use_last_error=True)
    api.GetCurrentProcess.restype = ctypes.c_void_p
    api.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    api.GetProcessAffinityMask.argtypes = [ctypes.c_void_p,
                                           ctypes.POINTER(ctypes.c_size_t),
                                           ctypes.POINTER(ctypes.c_size_t)]
    handle = api.GetCurrentProcess()
    if not api.SetProcessAffinityMask(handle, mask):
        raise RuntimeError(f'SetProcessAffinityMask failed: {ctypes.get_last_error()}')
    actual, system = ctypes.c_size_t(), ctypes.c_size_t()
    if not api.GetProcessAffinityMask(handle, ctypes.byref(actual), ctypes.byref(system)):
        raise RuntimeError('GetProcessAffinityMask failed')
    if actual.value != mask:
        raise RuntimeError(f'CPU affinity mismatch: requested {mask}, actual {actual.value}')
    return actual.value


def _init_runtime(seed):
    """Fresh-process CPU runtime, mirroring the audited retention_run.runtime().

    Environment must already be exact or torch must not be imported yet; a loaded
    torch with mismatched thread settings is a hard error, never silently reused.
    """
    torch_loaded = 'torch' in sys.modules
    for name, value in THREAD_ENV.items():
        current = os.environ.get(name)
        if current == value:
            continue
        if torch_loaded:
            raise RuntimeError(f'{name}={current!r} but torch is already imported; '
                               'start a fresh process (env cannot be applied safely).')
        os.environ[name] = value
    if torch_loaded:
        import torch
        if torch.get_num_threads() != 1 or torch.get_num_interop_threads() != 1:
            raise RuntimeError('torch already imported with threads!=1; refusing to reuse.')
    affinity = _apply_affinity(CPU_MASKS[seed])
    import torch
    torch.set_num_threads(1)
    if not torch_loaded:
        torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.mkldnn.deterministic = True
    torch.backends.fp32_precision = 'ieee'
    torch.backends.mkldnn.conv.fp32_precision = 'ieee'
    torch.backends.mkldnn.matmul.fp32_precision = 'ieee'
    torch.set_flush_denormal(False)
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(seed)
    return {'seed': seed, 'pid': os.getpid(), 'cpu_mask': affinity,
            'threads': torch.get_num_threads(), 'interop': torch.get_num_interop_threads(),
            'torch_version': str(torch.__version__), 'fp32_precision': 'ieee'}


def build_reference_wide():
    """ReferenceWide exactly as audited in Codex-TRN-20261019-001/audit_common.py."""
    import torch
    from torch import nn
    from torch.nn import functional as F

    class ReferenceWide(nn.Module):
        def __init__(self):
            super().__init__()

            def block(a, b, d=1):
                return nn.Sequential(nn.Conv2d(a, b, 3, padding=d, dilation=d), nn.ReLU(),
                                     nn.Conv2d(b, b, 3, padding=d, dilation=d), nn.ReLU())
            self.enc1 = block(3, 16)
            self.enc2 = block(16, 32)
            self.middle = block(32, 64, 4)
            self.dec2 = block(96, 32)
            self.dec1 = block(48, 16)
            self.head = nn.Conv2d(16, 3, 1)

        def forward(self, x):
            a = self.enc1(x)
            b = self.enc2(F.max_pool2d(a, 2))
            c = self.middle(F.max_pool2d(b, 2))
            d = self.dec2(torch.cat((F.interpolate(c, size=b.shape[-2:], mode='bilinear',
                                                   align_corners=False), b), 1))
            return self.head(self.dec1(torch.cat((F.interpolate(d, size=a.shape[-2:], mode='bilinear',
                                                                align_corners=False), a), 1)))
    return ReferenceWide()


def load_rgb_512(image_path):
    """Accept only RGB uint8 mode images of exactly 512x512; reject everything else."""
    with Image.open(image_path) as im:
        if im.mode != 'RGB':
            raise ValueError(f'image mode is {im.mode!r}, only RGB uint8 is accepted')
        if im.size != (512, 512):
            raise ValueError(f'image size is {im.size}, only 512x512 is accepted')
        return np.array(im, dtype=np.uint8)


def network_input_256(rgb512):
    """PIL BILINEAR resize of the original RGB 512 image to the 256 network grid."""
    im = Image.fromarray(rgb512)
    return np.array(im.resize((256, 256), Image.Resampling.BILINEAR), dtype=np.uint8)


def fuse_rule(base, p003, p005):
    """Equal-mean softmax fusion with W5 Road support; pure numpy on new arrays."""
    base = np.asarray(base)
    if base.shape != p003.shape[1:] or base.shape != p005.shape[1:] \
            or p003.shape[0] != 3 or p005.shape[0] != 3:
        raise ValueError('probability/base shapes differ')
    if p003.dtype != np.float32 or p005.dtype != np.float32:
        raise ValueError('probabilities must be float32')
    mean = (p003 + p005) * np.float32(0.5)
    pred = np.argmax(mean[:2], axis=0).astype(np.uint8)
    pred[base == 2] = 2
    return pred


def native_repeat2(mask256):
    """Native 512 mask by nearest repeat2 on both axes, no probability interpolation."""
    return np.repeat(np.repeat(mask256, 2, axis=0), 2, axis=1)


def extract_candidate_fields(mask256):
    """class0 four-connected components on the 256 grid; all kept, no filtering.

    Returns rows with native-pixel geometry: area_pixels = grid*4, bbox native xyxy
    (right/bottom exclusive), centroid_xy = mean of contained native pixel centers
    (2*grid_mean + 0.5), edge_touch on the 256 border, small_patch native area < 200.
    """
    grid = np.asarray(mask256)
    if grid.shape != (256, 256):
        raise ValueError('candidate grid must be 256x256')
    field = grid == 0
    seen = np.zeros_like(field, dtype=bool)
    rows = []
    for r0 in range(256):
        for c0 in range(256):
            if not field[r0, c0] or seen[r0, c0]:
                continue
            stack = [(r0, c0)]
            seen[r0, c0] = True
            cells = []
            while stack:
                r, c = stack.pop()
                cells.append((r, c))
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= nr < 256 and 0 <= nc < 256 and field[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            rs = [r for r, _ in cells]
            cs = [c for _, c in cells]
            r1, r2, c1, c2 = min(rs), max(rs), min(cs), max(cs)
            area = len(cells) * 4
            edge = r1 == 0 or c1 == 0 or r2 == 255 or c2 == 255
            rows.append({
                'field_id': len(rows) + 1,
                'area_pixels': area,
                'bbox_x0': c1 * 2, 'bbox_y0': r1 * 2,
                'bbox_x1': (c2 + 1) * 2, 'bbox_y1': (r2 + 1) * 2,
                'centroid_x': float(2 * (sum(cs) / len(cells)) + 0.5),
                'centroid_y': float(2 * (sum(rs) / len(cells)) + 0.5),
                'edge_touch': edge,
                'small_patch': area < 200,
            })
    return rows


def candidate_csv(rows):
    buffer = io.StringIO(newline='')
    writer = csv.writer(buffer, lineterminator='\n')
    writer.writerow(CSV_HEADER)
    for row in rows:
        writer.writerow([row['field_id'], row['area_pixels'],
                         row['bbox_x0'], row['bbox_y0'], row['bbox_x1'], row['bbox_y1'],
                         row['centroid_x'], row['centroid_y'],
                         'true' if row['edge_touch'] else 'false',
                         'true' if row['small_patch'] else 'false'])
    return buffer.getvalue()


class CandidatePredictor:
    """Loads one verified candidate seed group once; predict() returns new arrays."""

    def __init__(self, manifest_path=DEFAULT_MANIFEST, seed=DEFAULT_SEED,
                 project_root=PROJECT_ROOT):
        self.project_root = Path(project_root).resolve()
        self.manifest_path = Path(manifest_path)
        self.manifest_sha256 = sha256_file(self.manifest_path)
        manifest = _read_json(self.manifest_path)
        if manifest.get('schema') != MANIFEST_SCHEMA:
            raise ValueError(f'unexpected manifest schema {manifest.get("schema")!r}')
        group = next((m for m in manifest['models'] if m['seed'] == seed), None)
        if group is None:
            raise ValueError(f'seed {seed} not in manifest; '
                             f'available: {[m["seed"] for m in manifest["models"]]}')
        self.seed = seed
        self.runtime = _init_runtime(seed)
        import torch
        base = (self.project_root / 'reports' / 'rgb').resolve()
        self.models = {}
        self.weights_sha256 = {}
        for role in ('baseline_W5', 'retention_003', 'feedback_005'):
            entry = group[role]
            path = (self.project_root / entry['path']).resolve()
            if not path.is_relative_to(base):
                raise ValueError(f'{role} path escapes project reports/rgb: {path}')
            if not path.is_file():
                raise FileNotFoundError(f'{role} weight missing: {path}')
            actual = sha256_file(path)
            if actual != entry['sha256']:
                raise ValueError(f'{role} sha256 mismatch: {actual} != {entry["sha256"]}')
            self.weights_sha256[role] = actual
            checkpoint = torch.load(path, map_location='cpu', weights_only=True)
            state = checkpoint['state_dict']
            tensors, params = len(state), sum(v.numel() for v in state.values())
            if tensors != EXPECTED_TENSORS or params != EXPECTED_PARAMS:
                raise ValueError(f'{role} has {tensors} tensors / {params} params, '
                                 f'expected {EXPECTED_TENSORS}/{EXPECTED_PARAMS}')
            model = build_reference_wide()
            model.load_state_dict(state, strict=True)
            model.eval().requires_grad_(False)
            self.models[role] = model

    def predict(self, image_path):
        """uint8 256x256 class mask; base W5 argmax, equal-mean 003/005, Road iff base 2."""
        import torch
        rgb = load_rgb_512(image_path)
        rgb256 = network_input_256(rgb)
        x = torch.from_numpy(rgb256.copy()).permute(2, 0, 1).float().div_(255).unsqueeze(0)
        with torch.no_grad():
            base = self.models['baseline_W5'](x).argmax(1)[0].numpy().astype(np.uint8)
            p003 = torch.softmax(self.models['retention_003'](x), dim=1)[0].numpy().astype(np.float32)
            p005 = torch.softmax(self.models['feedback_005'](x), dim=1)[0].numpy().astype(np.float32)
        return fuse_rule(base, p003, p005)


def export(predictor, image_path, output_dir):
    """Write mask256/mask512/overlay PNGs, candidate CSV, quality and metadata JSON.

    The output directory must not exist. All validation and computation happen
    before the directory is created, so input errors leave no success metadata.
    """
    import torch
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f'output directory already exists: {output_dir}')
    rgb = load_rgb_512(image_path)
    image_sha = sha256_file(image_path)
    mask256 = predictor.predict(image_path)
    mask512 = native_repeat2(mask256)
    grid_counts = {c: int((mask256 == c).sum()) for c in (0, 1, 2)}
    native_counts = {c: int((mask512 == c).sum()) for c in (0, 1, 2)}
    assert sum(grid_counts.values()) == 65536 and sum(native_counts.values()) == 262144
    rows = extract_candidate_fields(mask256)
    assert sum(r['area_pixels'] for r in rows) == native_counts[0]
    csv_text = candidate_csv(rows)
    colors = {0: (0, 255, 0), 1: (255, 255, 0), 2: (160, 32, 240)}
    tint = np.zeros((512, 512, 4), dtype=np.uint8)
    for cls, color in colors.items():
        region = mask512 == cls
        tint[region, :3] = color
        tint[region, 3] = 140
    combined = Image.alpha_composite(Image.fromarray(rgb).convert('RGBA'),
                                     Image.fromarray(tint)).convert('RGB')
    buffer = io.BytesIO()
    combined.save(buffer, format='PNG')
    overlay_bytes = buffer.getvalue()

    payloads = {}
    payloads['mask256.png'] = _png_bytes(Image.fromarray(mask256, mode='L'))
    payloads['mask512.png'] = _png_bytes(Image.fromarray(mask512, mode='L'))
    payloads['overlay.png'] = overlay_bytes
    payloads['candidate_fields.csv'] = csv_text.encode('utf-8')
    output_sha = {name: hashlib.sha256(data).hexdigest() for name, data in payloads.items()}
    edge_count = sum(1 for r in rows if r['edge_touch'])
    small_count = sum(1 for r in rows if r['small_patch'])
    quality = {
        'grid_class_counts': grid_counts,
        'native_class_counts': native_counts,
        'grid_total_65536': sum(grid_counts.values()) == 65536,
        'native_total_262144': sum(native_counts.values()) == 262144,
        'candidate_area_sum_equals_native_field_count':
            sum(r['area_pixels'] for r in rows) == native_counts[0],
        'connectivity_rule': '4-neighborhood (up/down/left/right) on the 256 grid',
        'candidate_count': len(rows),
        'edge_touch_count': edge_count,
        'small_patch_count': small_count,
        'small_patch_rule': 'native area_pixels < 200',
        'notes': NOTICE,
    }
    metadata = {
        'task': 'INF-20261020-002',
        'seed': predictor.seed,
        'manifest_path': predictor.manifest_path.name,
        'manifest_sha256': predictor.manifest_sha256,
        'weights_sha256': predictor.weights_sha256,
        'image_sha256': image_sha,
        'input': {'original_size': [512, 512], 'mode': 'RGB uint8',
                  'network_input': [256, 256], 'normalization': 'float32 /255'},
        'output': {'grid': [256, 256], 'native': 'nearest repeat2x, no probability interpolation'},
        'runtime': predictor.runtime,
        'versions': {'python': platform.python_version(), 'torch': torch.__version__,
                     'numpy': np.__version__, 'pillow': Image.__version__},
        'grid_class_counts': grid_counts,
        'native_class_counts': native_counts,
        'candidate_count': len(rows),
        'output_files_sha256': output_sha,
        'candidate_scope': 'class0 four-connected components on the 256 grid, all kept',
        'independent_test_available': False,
        'notes': NOTICE,
    }
    payloads['quality.json'] = json.dumps(quality, ensure_ascii=False, indent=2).encode('utf-8')
    payloads['metadata.json'] = json.dumps(metadata, ensure_ascii=False, indent=2).encode('utf-8')
    output_dir.mkdir(parents=True)
    for name, data in payloads.items():
        (output_dir / name).write_bytes(data)
    return metadata


def _png_bytes(image):
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    return buffer.getvalue()
