"""CPU RGB fitting experiment; no independent evaluation or FTW replacement.

Only original Field/Field ridge/Road labels are eligible. Unknown Fieldridge
annotations are deferred as whole images. Original data and historical flags
remain unchanged. Source relationships are unknown; predictions on fitting
images are demonstrations, never independent accuracy estimates.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location('rgb_fitting_input', ROOT/'scripts/rgb/train_smoke.py')
_input = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_input)
CLASSES = {'0': 'Field', '1': 'Field ridge', '2': 'Road', '255': 'ignore'}
COLORS = np.array([[40, 170, 70], [240, 190, 40], [220, 50, 50]], dtype=np.uint8)


class FittingError(ValueError):
    pass


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        raise FittingError(f'Cannot read JSON: {path}: {exc}') from exc


def resolve(path, root=ROOT):
    if not isinstance(path, (str, Path)) or not str(path).strip():
        raise FittingError('Path must not be empty')
    p = Path(path)
    return (p if p.is_absolute() else Path(root)/p).resolve()


def inside(path, parent):
    return os.path.commonpath([os.path.normcase(str(path)), os.path.normcase(str(parent))]) == os.path.normcase(str(parent))


def artifact(path, root=ROOT):
    p = resolve(path, root)
    base = (Path(root)/'reports/rgb').resolve()
    if p == base or not inside(p, base):
        raise FittingError('Artifact must be below reports/rgb')
    return p


def new_output(path, inputs, root=ROOT):
    p = artifact(path, root)
    if p.exists():
        raise FittingError('Output exists; refusing overwrite')
    for item in inputs:
        q = Path(item).resolve()
        if inside(p, q) or inside(q, p):
            raise FittingError('Output and input directories must be disjoint')
    return p


def relative(p, root=ROOT):
    return Path(p).resolve().relative_to(Path(root).resolve()).as_posix()


def bounded_int(value, name, lo, hi):
    if type(value) is not int or not lo <= value <= hi:
        raise FittingError(f'{name} must be an integer in {lo}..{hi}')


def verify_selection(selection, root=ROOT, size=128):
    """Reconstruct all decisions from hashed originals; never trust a split file."""
    root = Path(root).resolve()
    sp = artifact(selection, root)
    s = read_json(sp)
    if not isinstance(s, dict) or type(s.get('schema_version')) is not int or s['schema_version'] != 1:
        raise FittingError('Unknown selection schema')
    inp = resolve(s.get('input_dir'), root)
    ds = _input.RGBInspectionDataset(str(inp), root=str(root))
    if s.get('manifest_sha256') != sha(inp/'manifest.json') or s.get('policy_sha256') != sha(inp/'policy.json'):
        raise FittingError('Selection manifest/policy digest mismatch')
    manifest = read_json(inp/'manifest.json')
    chosen = s.get('records')
    if not isinstance(chosen, list) or len(chosen) != len(manifest['records']):
        raise FittingError('Selection must contain every original record exactly once')
    selected = {}
    for rec in chosen:
        if not isinstance(rec, dict) or not isinstance(rec.get('sample_id'), str) or rec['sample_id'].casefold() in selected:
            raise FittingError('Duplicate/invalid selection sample ID')
        selected[rec['sample_id'].casefold()] = rec
    raw = {r['sample_id']: r for r in manifest['records']}
    if {r.casefold() for r in raw} != set(selected) or len(raw) != len(manifest['records']):
        raise FittingError('Selection IDs do not match original manifest')
    eligible = set()
    decisions = {}
    for sid, rec in raw.items():
        row = selected[sid.casefold()]
        if row['sample_id'] != sid:
            raise FittingError('Selection ID spelling changed')
        for key in ('image_path', 'image_sha256', 'json_path', 'json_sha256', 'mask_path', 'mask_sha256'):
            if row.get(key) != rec.get(key):
                raise FittingError(f'{sid}: selection changed source field {key}')
        if row.get('semantic_reviewed') is not False:
            raise FittingError('Selection cannot assert semantic review')
        jp = Path(ds._resolve_source_rel(rec.get('json_path'), sid))
        ip = Path(ds._resolve_source_rel(rec.get('image_path'), sid))
        for path, prefix in ((jp, 'json'), (ip, 'image')):
            if path.stat().st_size != rec.get(prefix+'_bytes') or sha(path) != rec.get(prefix+'_sha256'):
                raise FittingError(f'{sid}: source bytes/digest mismatch')
        doc = read_json(jp)
        shapes = doc.get('shapes') if isinstance(doc, dict) else None
        if not isinstance(shapes, list) or not shapes:
            raise FittingError('Missing source shapes')
        labels = []
        for shape in shapes:
            if not isinstance(shape, dict) or shape.get('shape_type') != 'polygon' or shape.get('label') not in ('Field', 'Field ridge', 'Fieldridge', 'Road'):
                raise FittingError(f'{sid}: unknown source annotation')
            labels.append(shape['label'])
        expected = 'excluded_user_decision' if rec['status'] == 'review' else ('deferred_alias' if 'Fieldridge' in labels else 'experiment_train')
        if row.get('disposition') != expected:
            raise FittingError(f'{sid}: disposition contradicts original source')
        if expected == 'experiment_train':
            if row.get('experimental_training_allowed') is not True:
                raise FittingError('Experimental permission missing')
            eligible.add(sid)
        decisions[sid] = expected
    if not eligible:
        raise FittingError('No eligible experiment images')
    cache = []
    histogram = np.zeros(3, dtype=np.int64)
    ignore = 0
    for index, rec in enumerate(ds.records):
        # Validates every candidate, including deferred images, before output.
        item = ds[index]
        a = item['target'].numpy()
        stats = rec.get('pixel_stats')
        actual = {str(c): int((a == c).sum()) for c in range(3)}
        if not isinstance(stats, dict) or stats.get('class_pixels') != actual or stats.get('ignore_pixels') != int((a == 255).sum()) or stats.get('denominator') != a.size:
            raise FittingError('Candidate mask histogram mismatch')
        if rec['sample_id'] not in eligible:
            continue
        ip = Path(ds._resolve_source_rel(rec['image_path'], rec['sample_id']))
        with Image.open(ip) as im:
            rgb = np.array(im.resize((size, size), Image.Resampling.BILINEAR), dtype=np.uint8)
        target = np.array(Image.fromarray(a.astype(np.uint8)).resize((size, size), Image.Resampling.NEAREST))
        if not np.any(target != 255):
            raise FittingError('Resized sample has no valid pixels')
        histogram += np.array([np.count_nonzero(target == c) for c in range(3)])
        ignore += int((target == 255).sum())
        cache.append((rec['sample_id'], rgb, target))
    if len(cache) != len(eligible) or not (histogram > 0).all():
        raise FittingError('Require full coverage and all three training classes')
    return sp, s, ds, cache, histogram, ignore


class CachedImages(Dataset):
    def __init__(self, cache):
        self.cache = cache

    def __len__(self):
        return len(self.cache)

    def __getitem__(self, index):
        sid, rgb, mask = self.cache[index]
        return torch.from_numpy(rgb.copy()).permute(2, 0, 1).float()/255, torch.from_numpy(mask.astype(np.int64)), sid


def double_conv(a, b):
    return nn.Sequential(nn.Conv2d(a, b, 3, padding=1), nn.ReLU(), nn.Conv2d(b, b, 3, padding=1), nn.ReLU())


class SmallUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = double_conv(3, 8)
        self.enc2 = double_conv(8, 16)
        self.middle = double_conv(16, 32)
        self.dec2 = double_conv(48, 16)
        self.dec1 = double_conv(24, 8)
        self.head = nn.Conv2d(8, 3, 1)

    def forward(self, x):
        a = self.enc1(x)
        b = self.enc2(F.max_pool2d(a, 2))
        c = self.middle(F.max_pool2d(b, 2))
        d = self.dec2(torch.cat([F.interpolate(c, size=b.shape[-2:], mode='bilinear', align_corners=False), b], 1))
        e = self.dec1(torch.cat([F.interpolate(d, size=a.shape[-2:], mode='bilinear', align_corners=False), a], 1))
        return self.head(e)


def state_digest(state):
    h = hashlib.sha256()
    for name in sorted(state):
        t = state[name].detach().cpu().contiguous()
        h.update(name.encode('utf-8')); h.update(str(t.dtype).encode('ascii'))
        h.update(str(tuple(t.shape)).encode('ascii')); h.update(t.numpy().tobytes())
    return h.hexdigest()


def finite(value, description):
    if not bool(torch.isfinite(value).all()):
        raise FittingError(f'Non-finite {description}')


def class_weights_from_counts(histogram, strategy='sqrt_inverse'):
    counts = np.asarray(histogram)
    if counts.shape != (3,) or not np.issubdtype(counts.dtype, np.integer) or not (counts > 0).all():
        raise FittingError('Require three positive integer class counts')
    if strategy not in ('sqrt_inverse', 'inverse_frequency'):
        raise FittingError('Unknown class weighting strategy')
    inverse = counts.sum()/(3*counts.astype(np.float64))
    weights = np.sqrt(inverse) if strategy == 'sqrt_inverse' else inverse
    return (weights/weights.mean()).astype(np.float32)


def weighted_loss_components(logits, target, weights, normalization='batch_weight_sum', expected_weight=None):
    valid = target != 255
    if not bool(valid.any()):
        raise FittingError('No valid pixels for loss')
    if normalization == 'batch_weight_sum':
        denominator = weights[target[valid]].double().sum().item()
    elif normalization == 'global_expected_weight':
        if not isinstance(expected_weight, (int, float)) or isinstance(expected_weight, bool) or not math.isfinite(expected_weight) or expected_weight <= 0:
            raise FittingError('Expected global weight must be finite and positive')
        denominator = int(valid.sum().item()) * expected_weight
    else:
        raise FittingError('Unknown loss normalization')
    if not math.isfinite(denominator) or denominator <= 0:
        raise FittingError('Invalid loss denominator')
    loss_sum = F.cross_entropy(logits, target, weight=weights, ignore_index=255, reduction='sum')
    finite(loss_sum, 'weighted loss sum')
    return loss_sum, denominator


def train(selection, output, epochs=3, size=128, batch_size=8, seed=20261017, threads=4, root=ROOT, class_weighting='sqrt_inverse', loss_normalization='batch_weight_sum'):
    for value, name, lo, hi in ((epochs, 'epochs', 1, 20), (size, 'size', 32, 512), (batch_size, 'batch_size', 1, 32), (seed, 'seed', 0, 2**32-1), (threads, 'threads', 1, 8)):
        bounded_int(value, name, lo, hi)
    if size % 8:
        raise FittingError('size must be divisible by eight')
    if class_weighting not in ('sqrt_inverse', 'inverse_frequency'):
        raise FittingError('Unknown class weighting strategy')
    if loss_normalization not in ('batch_weight_sum', 'global_expected_weight'):
        raise FittingError('Unknown loss normalization')
    sp = artifact(selection, root)
    out = new_output(output, [sp.parent], root)
    sp, selection_data, ds, cache, histogram, ignore = verify_selection(sp, root, size)
    if inside(out, Path(ds.input_dir)) or inside(Path(ds.input_dir), out):
        raise FittingError('Output overlaps original input')
    weights = class_weights_from_counts(histogram, class_weighting)
    expected_weight = float(np.dot(histogram.astype(np.float64), weights.astype(np.float64)) / histogram.sum())
    dataset = CachedImages(cache)
    old_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(threads)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            model = SmallUNet()
            before = state_digest(model.state_dict())
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            w = torch.tensor(weights)
            refx, refy, refids = next(iter(DataLoader(dataset, batch_size=min(2, len(dataset)), shuffle=False)))
            model.eval()
            with torch.no_grad():
                initial = F.cross_entropy(model(refx), refy, weight=w, ignore_index=255).item()
            if not math.isfinite(initial):
                raise FittingError('Initial reference loss is non-finite')
            out.mkdir(parents=True, exist_ok=False)
            generator = torch.Generator().manual_seed(seed)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator, num_workers=0, drop_last=False)
            history = []
            step_count = 0
            for epoch in range(1, epochs+1):
                model.train()
                ids = []; numerator = 0.; denominator = 0.; batches = []
                for x, y, sample_ids in loader:
                    optimizer.zero_grad(set_to_none=True)
                    logits = model(x); finite(logits, 'logits')
                    loss_sum, mass = weighted_loss_components(logits, y, w, loss_normalization, expected_weight)
                    loss = loss_sum/mass; finite(loss, 'loss')
                    loss.backward()
                    for p in model.parameters():
                        if p.grad is None:
                            raise FittingError('Disconnected model parameter')
                        finite(p.grad, 'gradient')
                    optimizer.step()
                    for p in model.parameters(): finite(p, 'parameter')
                    numerator += loss_sum.item(); denominator += mass
                    ids.extend(sample_ids); batches.append(len(sample_ids)); step_count += 1
                if len(ids) != len(cache) or set(ids) != {c[0] for c in cache}:
                    raise FittingError('Incomplete epoch coverage')
                row = {'epoch':epoch, 'loss':numerator/denominator, 'loss_numerator':numerator,
                       'weight_denominator':denominator, 'sample_count':len(ids), 'batch_sizes':batches, 'sample_ids':ids}
                history.append(row)
                print(json.dumps({k:row[k] for k in ('epoch','loss','sample_count')}) , flush=True)
            model.eval()
            with torch.no_grad():
                reference_logits = model(refx)
                final = F.cross_entropy(reference_logits, refy, weight=w, ignore_index=255).item()
            after = state_digest(model.state_dict())
            if before == after or not math.isfinite(final):
                raise FittingError('No parameter update or invalid final loss')
            checkpoint = {'schema':'agrolake_rgb_internal_unet_v1', 'state_dict':model.state_dict(), 'size':size,
                          'selection_sha256':sha(sp),'seed':seed,'class_ids':CLASSES, 'class_weights':weights.tolist(),
                          'class_weighting':class_weighting,
                          'loss_normalization':loss_normalization, 'expected_weight_per_valid_pixel':expected_weight,
                          'trained_on_real_data':True,'semantic_reviewed':False,'independent_test_available':False}
            torch.save(checkpoint, out/'checkpoint.pt')
            loaded = torch.load(out/'checkpoint.pt', map_location='cpu', weights_only=True)
            clone = SmallUNet(); clone.load_state_dict(loaded['state_dict'], strict=True); clone.eval()
            with torch.no_grad(): equal = torch.equal(reference_logits, clone(refx))
            if not equal: raise FittingError('Checkpoint roundtrip logits differ')
            summary = {'kind':'real_rgb_internal_fitting','trained_on_real_data':True,'semantic_reviewed':False,
                       'independent_test_available':False,'split_created':False,'source_relationship':'unknown',
                       'selection_path':relative(sp,root),'selection_sha256':sha(sp),'input_manifest_sha256':selection_data['manifest_sha256'],
                       'sample_count':len(cache),'training_sample_ids':[r[0] for r in cache], 'epochs':epochs,'optimizer_steps':step_count,
                       'batch_size':batch_size,'size':size,'seed':seed,'threads':threads,'optimizer':'Adam','learning_rate':0.001,
                       'model':'SmallUNet width8/16/32; random initialization','parameter_count':sum(p.numel() for p in model.parameters()),
                       'training_class_pixels':{str(k):int(v) for k,v in enumerate(histogram)},'training_ignore_pixels':ignore,
                       'class_weights':weights.tolist(),'class_weighting':class_weighting,
                       'weight_formula':('sqrt(total_valid/(3*class_pixels))' if class_weighting == 'sqrt_inverse' else 'total_valid/(3*class_pixels)')+'; normalized to mean 1',
                       'loss_normalization':loss_normalization, 'expected_weight_per_valid_pixel':expected_weight,
                       'loss_aggregation':('sum(weighted CE)/sum(target class weights), ignore 255' if loss_normalization == 'batch_weight_sum' else 'sum(weighted CE)/(valid pixels * global expected target weight), ignore 255'),
                       'initial_state_sha256':before,'final_state_sha256':after,'checkpoint_roundtrip_equal':equal,
                       'checkpoint_sha256':sha(out/'checkpoint.pt'),'reference_sample_ids':list(refids),
                       'reference_initial_loss':initial,'reference_final_loss':final,'history':history,
                       'torch_version':str(torch.__version__),'numpy_version':str(np.__version__),
                       'limitations':['Fitting samples only; no independent accuracy estimate.','Fieldridge deferred as entire images.','No ditch labels or real area measurement.']}
            write_json(out/'summary.json', summary)
            with (out/'history.csv').open('w',encoding='utf-8-sig',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=['epoch','loss','sample_count','loss_numerator','weight_denominator']);writer.writeheader()
                writer.writerows({k:r[k] for k in writer.fieldnames} for r in history)
            (out/'README.md').write_text('真实RGB内部拟合实验。未创建独立验证/测试划分，不能据此报告泛化精度。\n'+
                f'样本{len(cache)}，轮数{epochs}，优化步数{step_count}。原始数据、FTW模型未修改。\n',encoding='utf-8')
            return summary
    finally:
        torch.set_num_threads(old_threads)


def predict(checkpoint, selection, sample_id, output, root=ROOT):
    cp=artifact(checkpoint,root);sp=artifact(selection,root)
    out=new_output(output,[cp.parent,sp.parent],root)
    payload=torch.load(cp,map_location='cpu',weights_only=True)
    if not isinstance(payload,dict) or payload.get('schema')!='agrolake_rgb_internal_unet_v1' or payload.get('selection_sha256')!=sha(sp) or payload.get('class_ids')!=CLASSES:
        raise FittingError('Checkpoint schema/selection/class binding mismatch')
    size=payload.get('size');bounded_int(size,'size',32,512)
    if size%8 or payload.get('semantic_reviewed') is not False or payload.get('independent_test_available') is not False:
        raise FittingError('Invalid checkpoint metadata')
    # Full source preflight also prevents a forged alias or exclusion becoming a demo.
    sp,selection_data,ds,cache,_,_=verify_selection(sp,root,size)
    if not isinstance(sample_id,str) or sample_id not in {c[0] for c in cache}:
        raise FittingError('Prediction demo must use an eligible fitting sample')
    rec=next(r for r in ds.records if r['sample_id']==sample_id)
    image_path=Path(ds._resolve_source_rel(rec['image_path'],sample_id));mask_path=Path(ds._resolve_derived_rel(rec['mask_path'],sample_id))
    with Image.open(image_path) as im: original=im.copy()
    with Image.open(mask_path) as im: annotation=np.array(im)
    small=next(c[1] for c in cache if c[0]==sample_id)
    x=torch.from_numpy(small.copy()).permute(2,0,1).float().unsqueeze(0)/255
    old_threads=torch.get_num_threads()
    try:
        torch.set_num_threads(4)
        with torch.random.fork_rng(devices=[]):
            model=SmallUNet();model.load_state_dict(payload['state_dict'],strict=True);model.eval()
            for p in model.parameters():finite(p,'checkpoint parameter')
            with torch.no_grad():
                logits=model(x);finite(logits,'prediction logits')
                prediction=F.interpolate(logits,size=(original.height,original.width),mode='bilinear',align_corners=False).argmax(1)[0].numpy().astype(np.uint8)
    finally:torch.set_num_threads(old_threads)
    counts={str(c):int((prediction==c).sum()) for c in range(3)}
    color=Image.fromarray(COLORS[prediction]);overlay=Image.blend(original,color,0.4)
    truth=np.full((*annotation.shape,3),110,dtype=np.uint8)
    for c in range(3):truth[annotation==c]=COLORS[c]
    width,height=original.size
    contact=Image.new('RGB',(width*3,height+32),'white');draw=ImageDraw.Draw(contact)
    for j,(label,im) in enumerate([('Original RGB',original),('Annotation (gray=ignore)',Image.fromarray(truth)),('Prediction: FITTING SAMPLE',overlay)]):
        draw.text((j*width+8,8),label,fill='black');contact.paste(im,(j*width,32))
    out.mkdir(parents=True,exist_ok=False)
    Image.fromarray(prediction).save(out/'prediction.png');overlay.save(out/'overlay.png');contact.save(out/'comparison.png')
    with (out/'pixels.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.writer(f);writer.writerow(['class_id','class_name','predicted_pixels','denominator'])
        for c in range(3):writer.writerow([c,CLASSES[str(c)],counts[str(c)],width*height])
    summary={'kind':'fitting_sample_prediction','sample_id':sample_id,'trained_on_real_data':True,'independent_test_available':False,
             'semantic_reviewed':False,'source_relationship':'unknown','width':width,'height':height,
             'class_ids':CLASSES,'predicted_pixel_counts':counts,'denominator':width*height,
             'image_path':rec['image_path'],'image_sha256':sha(image_path),'annotation_path':rec['mask_path'],'annotation_sha256':sha(mask_path),
             'checkpoint_path':relative(cp,root),'checkpoint_sha256':sha(cp),'selection_path':relative(sp,root),'selection_sha256':sha(sp),
             'artifacts':{p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file()},
             'note':'Training sample demonstration, not independent evaluation; counts are pixels, not real field count or square metres.'}
    write_json(out/'summary.json',summary)
    return summary


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    fit=commands.add_parser('train');fit.add_argument('--selection',required=True);fit.add_argument('--output',required=True)
    for flag,default in (('epochs',3),('size',128),('batch-size',8),('seed',20261017),('threads',4)):
        fit.add_argument('--'+flag,type=int,default=default)
    fit.add_argument('--class-weighting',choices=['sqrt_inverse','inverse_frequency'],default='sqrt_inverse')
    fit.add_argument('--loss-normalization',choices=['batch_weight_sum','global_expected_weight'],default='batch_weight_sum')
    pred=commands.add_parser('predict')
    for flag in ('checkpoint','selection','sample-id','output'):pred.add_argument('--'+flag,required=True)
    args=vars(parser.parse_args(argv));command=args.pop('command')
    try:
        result=train(**args) if command=='train' else predict(**args)
        print(json.dumps({k:v for k,v in result.items() if k not in ('history','training_sample_ids')},ensure_ascii=False,allow_nan=False))
        return 0
    except (FittingError,_input.RGBError,OSError,ValueError,RuntimeError,TypeError,KeyError) as exc:
        print('RGB fitting failed: '+str(exc));return 1


if __name__=='__main__':
    raise SystemExit(main())
