"""Whole fitting-set RGB diagnostics. These are NOT independent test scores."""
import argparse
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location('rgb_diagnostic_fitting', ROOT/'scripts/rgb/fit_baseline.py')
fit = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fit)
DiagnosticError = fit.FittingError


def confusion(target, prediction):
    """Rows=true class, columns=prediction; 255 is excluded, class 0 retained."""
    target, prediction = np.asarray(target), np.asarray(prediction)
    if target.shape != prediction.shape or not np.issubdtype(target.dtype, np.integer) or not np.issubdtype(prediction.dtype, np.integer):
        raise DiagnosticError('Require matching integer target/prediction arrays')
    if not np.isin(target, [0, 1, 2, 255]).all() or not np.isin(prediction, [0, 1, 2]).all():
        raise DiagnosticError('Unknown target/prediction class')
    valid = target != 255
    codes = target[valid].astype(np.int64)*3 + prediction[valid].astype(np.int64)
    return np.bincount(codes, minlength=9).reshape(3, 3)


def measures(matrix):
    matrix = np.asarray(matrix)
    if matrix.shape != (3, 3) or not np.issubdtype(matrix.dtype, np.integer) or np.any(matrix < 0):
        raise DiagnosticError('Require a nonnegative integer 3x3 confusion matrix')
    def ratio(a, b): return float(a/b) if b else None
    classes = {}
    for c in range(3):
        tp = int(matrix[c, c]); actual = int(matrix[c].sum()); predicted = int(matrix[:, c].sum())
        classes[str(c)] = {'name':fit.CLASSES[str(c)], 'true_pixels':actual, 'predicted_pixels':predicted,
                           'true_positive':tp, 'false_negative':actual-tp, 'false_positive':predicted-tp,
                           'precision':ratio(tp, predicted), 'recall':ratio(tp, actual), 'iou':ratio(tp, actual+predicted-tp)}
    present = [v['iou'] for v in classes.values() if v['iou'] is not None]
    total = int(matrix.sum())
    return {'confusion_true_rows_predicted_columns':matrix.tolist(), 'valid_pixels':total, 'classes':classes,
            'fitting_pixel_accuracy':ratio(int(np.trace(matrix)), total),
            'mean_iou_over_nonzero_unions':float(sum(present)/len(present)) if present else None}


def load_model(checkpoint, selection, root=ROOT):
    cp, sp = fit.artifact(checkpoint, root), fit.artifact(selection, root)
    payload = torch.load(cp, map_location='cpu', weights_only=True)
    if not isinstance(payload, dict) or payload.get('schema') != 'agrolake_rgb_internal_unet_v1' or payload.get('class_ids') != fit.CLASSES or payload.get('selection_sha256') != fit.sha(sp):
        raise DiagnosticError('Checkpoint schema/selection/class binding mismatch')
    size = payload.get('size'); fit.bounded_int(size, 'size', 32, 512)
    if size % 8 or payload.get('semantic_reviewed') is not False or payload.get('independent_test_available') is not False or payload.get('trained_on_real_data') is not True:
        raise DiagnosticError('Invalid internal-fitting checkpoint metadata')
    model = fit.SmallUNet(); model.load_state_dict(payload['state_dict'], strict=True); model.eval()
    for parameter in model.parameters(): fit.finite(parameter, 'checkpoint parameter')
    return cp, sp, payload, model


def diagnose(checkpoint, selection, output, batch_size=8, threads=4, root=ROOT):
    fit.bounded_int(batch_size, 'batch_size', 1, 32); fit.bounded_int(threads, 'threads', 1, 8)
    cp, sp = fit.artifact(checkpoint, root), fit.artifact(selection, root)
    out = fit.new_output(output, [cp.parent, sp.parent], root)
    old_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(threads)
        with torch.random.fork_rng(devices=[]):
            cp, sp, payload, model = load_model(cp, sp, root)
            _, selected, ds, cache, histogram, ignored = fit.verify_selection(sp, root, payload['size'])
            inp = Path(ds.input_dir).resolve()
            if fit.inside(out, inp) or fit.inside(inp, out):
                raise DiagnosticError('Diagnostic output overlaps input')
            by_id = {rec['sample_id']:rec for rec in ds.records}
            loader = DataLoader(fit.CachedImages(cache), batch_size=batch_size, shuffle=False, num_workers=0)
            out.mkdir(parents=True, exist_ok=False); (out/'predictions').mkdir()
            matrix = np.zeros((3, 3), dtype=np.int64)
            rows = []
            with torch.no_grad():
                for images, targets, ids in loader:
                    logits = model(images); fit.finite(logits, 'diagnostic logits')
                    predicted = logits.argmax(1).numpy().astype(np.uint8)
                    for target, pred, sid in zip(targets.numpy(), predicted, ids):
                        cm = confusion(target, pred); matrix += cm
                        metrics = measures(cm)
                        prediction = out/'predictions'/f'{sid}.png'; Image.fromarray(pred).save(prediction)
                        rec = by_id[sid]
                        rows.append({'sample_id':sid, 'image_path':rec['image_path'], 'image_sha256':rec['image_sha256'],
                            'mask_path':rec['mask_path'], 'mask_sha256':rec['mask_sha256'], 'ignored_pixels':int((target == 255).sum()),
                            'prediction_path':f'predictions/{sid}.png', 'prediction_sha256':fit.sha(prediction), **metrics})
            if len(rows) != len(cache) or [r['sample_id'] for r in rows] != [r[0] for r in cache]:
                raise DiagnosticError('Diagnostic did not cover all fitting samples exactly once')
            if matrix.sum() != histogram.sum() or sum(r['ignored_pixels'] for r in rows) != ignored:
                raise DiagnosticError('Diagnostic denominator differs from source masks')
            majority_class = int(histogram.argmax()); majority = np.zeros((3, 3), dtype=np.int64); majority[:, majority_class] = histogram
            summary = {'kind':'whole_fitting_set_diagnostic', 'evaluation_scope':'fitting_samples_only', 'semantic_reviewed':False,
                'independent_test_available':False, 'source_relationship':'unknown', 'split_created':False,
                'sample_count':len(rows), 'sample_ids':[r['sample_id'] for r in rows], 'prediction_resolution':[payload['size'],payload['size']],
                'target_resize':'Pillow NEAREST', 'image_resize':'Pillow BILINEAR', 'metrics_ignore_value':255,
                'checkpoint_path':fit.relative(cp,root), 'checkpoint_sha256':fit.sha(cp), 'checkpoint_parameter_sha256':fit.state_digest(model.state_dict()),
                'selection_path':fit.relative(sp,root), 'selection_sha256':fit.sha(sp), 'input_dir':fit.relative(inp,root),
                'input_manifest_sha256':selected['manifest_sha256'], 'input_policy_sha256':selected['policy_sha256'],
                'checkpoint_class_weights':payload['class_weights'], 'checkpoint_class_weighting':payload.get('class_weighting','legacy_unspecified'),
                'checkpoint_loss_normalization':payload.get('loss_normalization','legacy_batch_weight_sum'),
                'ignored_pixels':ignored, 'model':measures(matrix),
                'majority_constant_baseline':{'class_id':majority_class, **measures(majority)},
                'zero_denominator_policy':'null; macro IoU averages classes whose union is nonzero',
                'note':'All samples were used for fitting. No independent agricultural accuracy or generalization claim.'}
            fit.write_json(out/'records.json', rows)
            with (out/'records.csv').open('w',encoding='utf-8-sig',newline='') as stream:
                fields=['sample_id','valid_pixels','ignored_pixels','fitting_pixel_accuracy']+[f'{name}_{c}' for c in range(3) for name in ('precision','recall','iou')]
                writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
                for row in rows:
                    flat={k:row[k] for k in fields if k in row}
                    flat.update({f'{name}_{c}':row['classes'][str(c)][name] for c in range(3) for name in ('precision','recall','iou')})
                    writer.writerow(flat)
            summary['record_artifacts']={name:fit.sha(out/name) for name in ('records.json','records.csv')}
            fit.write_json(out/'summary.json',summary)
            (out/'README.md').write_text('全量训练集诊断，不是独立测试。\n'+
                '混淆矩阵：真实类别为行，预测类别为列；255不计入分母，0为田块。零分母用null。\n'+
                f'完整覆盖{len(rows)}个训练样本，诊断分辨率{payload["size"]}。predictions为该分辨率的L PNG。\n',encoding='utf-8')
            return summary
    finally:
        torch.set_num_threads(old_threads)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','selection','output'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--batch-size',type=int,default=8);parser.add_argument('--threads',type=int,default=4)
    try:
        summary=diagnose(**vars(parser.parse_args(argv)))
        print(json.dumps({k:summary[k] for k in ('sample_count','evaluation_scope','model','majority_constant_baseline')},ensure_ascii=False,allow_nan=False));return 0
    except (DiagnosticError,fit._input.RGBError,OSError,ValueError,RuntimeError,KeyError,TypeError) as exc:
        print('RGB diagnostic failed: '+str(exc));return 1


if __name__=='__main__':raise SystemExit(main())
