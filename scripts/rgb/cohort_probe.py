"""Bounded full-cohort memory probes, never independent evaluation or a model release."""
import argparse
import csv
import hashlib
import importlib.util
import json
import math
from contextlib import nullcontext
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location('cohort_diagnostic', ROOT/'scripts/rgb/diagnose.py')
diag = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(diag)
fit = diag.fit
ProbeError = fit.FittingError


class _WideProbeUNet(fit.SmallUNet):
    """Same forward structure; doubled channels are for memory diagnostics only."""
    def __init__(self):
        torch.nn.Module.__init__(self)
        self.enc1 = fit.double_conv(3, 16)
        self.enc2 = fit.double_conv(16, 32)
        self.middle = fit.double_conv(32, 64)
        self.dec2 = fit.double_conv(96, 32)
        self.dec1 = fit.double_conv(48, 16)
        self.head = torch.nn.Conv2d(16, 3, 1)


def build_model(base_width=8):
    if type(base_width) is not int or base_width not in (8, 16):
        raise ProbeError('base_width must be 8 or 16')
    return fit.SmallUNet() if base_width == 8 else _WideProbeUNet()


def validate_cohorts(config, selection_sha, eligible):
    if not isinstance(config, dict) or type(config.get('schema_version')) is not int or config['schema_version'] != 1:
        raise ProbeError('Unknown cohort schema')
    if config.get('selection_sha256') != selection_sha:
        raise ProbeError('Cohort configuration is not bound to this selection')
    groups = config.get('cohorts')
    if not isinstance(groups, list) or not 1 <= len(groups) <= 8:
        raise ProbeError('Require 1..8 cohorts')
    names = set()
    for group in groups:
        if not isinstance(group, dict):
            raise ProbeError('Invalid cohort entry')
        name = group.get('name')
        try:
            fit._input._validate_sample_id(name)
        except fit._input.RGBError as exc:
            raise ProbeError(str(exc)) from exc
        if len(name) > 40 or name.casefold() in names:
            raise ProbeError('Duplicate or oversized cohort name')
        names.add(name.casefold())
        ids = group.get('sample_ids')
        if not isinstance(ids, list) or not 1 <= len(ids) <= 16 or not all(isinstance(sid, str) for sid in ids):
            raise ProbeError('Each cohort requires 1..16 string IDs')
        if len({sid.casefold() for sid in ids}) != len(ids) or not set(ids).issubset(eligible):
            raise ProbeError('Duplicate or ineligible cohort sample ID')
        if 'reason' in group and not isinstance(group['reason'], str):
            raise ProbeError('Cohort reason must be text')
    return groups


def evaluate(model, x, y, ids, step, weights):
    model.eval()
    with torch.no_grad():
        logits = model(x); fit.finite(logits, 'cohort logits')
        numerator, denominator = fit.weighted_loss_components(logits, y, weights)
        loss = float((numerator/denominator).item())
        guesses = logits.argmax(1).numpy().astype(np.uint8)
    total = np.zeros((3, 3), dtype=np.int64)
    rows = []
    for sid, target, prediction in zip(ids, y.numpy(), guesses):
        matrix = diag.confusion(target, prediction); total += matrix
        rows.append({'sample_id':sid, 'ignored_pixels':int((target == 255).sum()), **diag.measures(matrix)})
    return {'step':step, 'weighted_ce':loss, 'per_image':rows, **diag.measures(total)}, guesses


def run_probe(selection, cohorts, output, steps=300, size=128, seed=20261017, threads=4, root=ROOT, learning_rate=.001, base_width=8, max_grad_norm=None):
    if max_grad_norm is not None:
        if isinstance(max_grad_norm,bool) or not isinstance(max_grad_norm,(int,float)) or not math.isfinite(max_grad_norm) or not 1e-6 <= max_grad_norm <= 1e6:
            raise ProbeError('max_grad_norm must be None or a finite number in 1e-6..1e6')
        max_grad_norm = float(max_grad_norm)
    if type(base_width) is not int or base_width not in (8, 16):
        raise ProbeError('base_width must be 8 or 16')
    if isinstance(learning_rate, bool) or not isinstance(learning_rate, (int, float)) or not math.isfinite(learning_rate) or not 1e-6 <= learning_rate <= 1e-2:
        raise ProbeError('learning_rate must be a finite number in 1e-6..1e-2')
    learning_rate = float(learning_rate)
    for value, name, lo, hi in ((steps,'steps',1,500),(size,'size',32,256),(seed,'seed',0,2**32-1),(threads,'threads',1,8)):
        fit.bounded_int(value,name,lo,hi)
    if size % 8:
        raise ProbeError('size must be divisible by eight')
    sp, cp = fit.artifact(selection,root), fit.artifact(cohorts,root)
    out = fit.new_output(output,[sp.parent,cp.parent],root)
    config_bytes = cp.read_bytes()
    config = json.loads(config_bytes.decode('utf-8'))
    config_sha = hashlib.sha256(config_bytes).hexdigest()
    selection_sha = fit.sha(sp)
    old_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(threads)
        _, _, ds, cache, histogram, ignored = fit.verify_selection(sp,root,size)
        if fit.sha(sp) != selection_sha:
            raise ProbeError('Selection changed during input verification')
        if fit.inside(out,Path(ds.input_dir)) or fit.inside(Path(ds.input_dir),out):
            raise ProbeError('Output overlaps candidate inputs')
        by_id = {sid:(rgb,target) for sid,rgb,target in cache}
        sources = {rec['sample_id']:rec for rec in ds.records}
        groups = validate_cohorts(config,selection_sha,set(by_id))
        weights_np = fit.class_weights_from_counts(histogram)
        weights = torch.tensor(weights_np)
        milestones = sorted({0,steps,*[n for n in (50,100,200,300) if n < steps]})
        out.mkdir(parents=True,exist_ok=False)
        (out/'cohorts.json').write_bytes(config_bytes)
        completed = []
        with torch.random.fork_rng(devices=[]):
            for group in groups:
                ids = group['sample_ids']; name = group['name']
                x = torch.stack([torch.from_numpy(by_id[sid][0].copy()).permute(2,0,1).float()/255 for sid in ids])
                y = torch.stack([torch.from_numpy(by_id[sid][1].astype(np.int64)) for sid in ids])
                torch.manual_seed(seed)
                model = build_model(base_width); initial_sha = fit.state_digest(model.state_dict())
                parameter_count = sum(p.numel() for p in model.parameters())
                optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
                first,_ = evaluate(model,x,y,ids,0,weights)
                history = [first]; exposures = {sid:0 for sid in ids}; updates = 0
                folder = out/name; folder.mkdir()
                clip_meta = {}; clip_rows = []; clip_snapshots = []
                snapshot_steps = {0,steps,steps-1,*[n for n in (300,400) if n < steps]}

                def save_clip_snapshot(step):
                    if max_grad_norm is None or step not in snapshot_steps: return
                    path = folder/f'clip-step-{step:04d}.pt'
                    torch.save({'schema':'agrolake_rgb_clipped_trace_v1','state_dict':model.state_dict(),
                        'optimizer_state_dict':optimizer.state_dict(),'optimizer_steps':step,'sample_ids':ids,
                        'size':size,'seed':seed,'threads':threads,'base_width':base_width,'learning_rate':learning_rate,
                        'max_grad_norm':max_grad_norm,'class_weights':weights_np.tolist(),
                        'selection_sha256':selection_sha,'cohorts_sha256':config_sha,
                        'not_candidate_model':True,'semantic_reviewed':False,'independent_test_available':False},path)
                    clip_snapshots.append({'step':step,'path':path.name,'sha256':fit.sha(path),
                        'parameter_sha256':fit.state_digest(model.state_dict())})

                save_clip_snapshot(0)
                trace_context = (folder/'gradient_trace.jsonl').open('w',encoding='utf-8',newline='\n') if max_grad_norm is not None else nullcontext(None)
                with trace_context as trace:
                    for step in range(1,steps+1):
                        model.train(); optimizer.zero_grad(set_to_none=True)
                        logits = model(x); fit.finite(logits,'cohort training logits')
                        numerator, denominator = fit.weighted_loss_components(logits,y,weights)
                        loss = numerator/denominator; fit.finite(loss,'cohort training loss'); loss.backward()
                        for parameter in model.parameters():
                            if parameter.grad is None:
                                raise ProbeError('Disconnected cohort parameter')
                            fit.finite(parameter.grad,'cohort gradient')
                        if max_grad_norm is not None:
                            norm = torch.nn.utils.clip_grad_norm_(model.parameters(),max_grad_norm,error_if_nonfinite=True,foreach=False)
                            after = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(p.grad) for p in model.parameters()]))
                            fit.finite(norm,'pre-clip norm');fit.finite(after,'post-clip norm')
                            if after.item() > max_grad_norm*(1+1e-5):raise ProbeError('Gradient clipping bound violated')
                            coefficient = torch.clamp(max_grad_norm/(norm+1e-6),max=1.0).item()
                            entry = {'step':step,'loss_before_update':loss.item(),'gradient_l2_before':norm.item(),
                                'gradient_l2_after':after.item(),'scale':coefficient,'clipped':coefficient < 1.0}
                            trace.write(json.dumps(entry,allow_nan=False)+'\n');trace.flush();clip_rows.append(entry)
                        optimizer.step()
                        for parameter in model.parameters(): fit.finite(parameter,'cohort parameter')
                        updates += 1
                        for sid in ids: exposures[sid] += 1
                        save_clip_snapshot(step)
                        if step in milestones:
                            row, predictions = evaluate(model,x,y,ids,step,weights); history.append(row)
                            print(json.dumps({'cohort':name,'step':step,'weighted_ce':row['weighted_ce'],'ridge_iou':row['classes']['1']['iou']}),flush=True)
                if max_grad_norm is not None:
                    clip_meta = {'max_grad_norm':max_grad_norm,'gradient_clipping':{
                        'mode':'global_l2_before_adam','foreach':False,'epsilon':1e-6,
                        'clipped_updates':sum(r['clipped'] for r in clip_rows),
                        'maximum_pre_clip_norm':max(r['gradient_l2_before'] for r in clip_rows),
                        'maximum_post_clip_norm':max(r['gradient_l2_after'] for r in clip_rows),
                        'trace_path':'gradient_trace.jsonl','trace_sha256':fit.sha(folder/'gradient_trace.jsonl'),
                        'snapshots':clip_snapshots}}
                if updates != steps or any(n != steps for n in exposures.values()):
                    raise ProbeError('Cohort exposure count mismatch')
                final_sha = fit.state_digest(model.state_dict())
                if initial_sha == final_sha:
                    raise ProbeError('Cohort parameters did not change')
                if fit.sha(cp) != config_sha or fit.sha(sp) != selection_sha:
                    raise ProbeError('Cohort config or selection changed during training')
                payload = {'schema':'agrolake_rgb_cohort_probe_v1','state_dict':model.state_dict(),'cohort_name':name,
                    'sample_ids':ids,'selection_sha256':selection_sha,'cohorts_sha256':config_sha,'size':size,'seed':seed,
                    'optimizer_steps':updates,'class_ids':fit.CLASSES,'class_weights':weights_np.tolist(),'learning_rate':learning_rate,
                    'base_width':base_width,'channel_widths':[base_width,base_width*2,base_width*4],'parameter_count':parameter_count,
                    'not_candidate_model':True,'semantic_reviewed':False,'independent_test_available':False,**clip_meta}
                torch.save(payload,folder/'probe_checkpoint.pt')
                clone = build_model(base_width); loaded = torch.load(folder/'probe_checkpoint.pt',map_location='cpu',weights_only=True)
                clone.load_state_dict(loaded['state_dict'],strict=True); clone.eval(); model.eval()
                with torch.no_grad(): roundtrip = torch.equal(clone(x),model(x))
                if not roundtrip: raise ProbeError('Cohort checkpoint roundtrip mismatch')
                artifacts = []
                for sid,prediction in zip(ids,predictions):
                    path = folder/f'{sid}.png'; Image.fromarray(prediction).save(path)
                    rgb,target = by_id[sid]
                    truth = np.full((size,size,3),110,dtype=np.uint8)
                    for c in range(3): truth[target == c] = fit.COLORS[c]
                    # Diagnostic-resolution pixels enlarged with NEAREST; no false native-resolution claim.
                    scale = 2; panel = Image.new('RGB',(size*scale*3,size*scale+32),'white'); draw = ImageDraw.Draw(panel)
                    for j,(label,im) in enumerate([('RGB',Image.fromarray(rgb)),('Existing labels',Image.fromarray(truth)),('COHORT MEMORY PROBE',Image.fromarray(fit.COLORS[prediction]))]):
                        draw.text((j*size*scale+3,8),label,fill='black');panel.paste(im.resize((size*scale,size*scale),Image.Resampling.NEAREST),(j*size*scale,32))
                    panel_path = folder/f'{sid}-comparison.png'; panel.save(panel_path)
                    rec = sources[sid]
                    artifacts.append({'sample_id':sid,'image_path':rec['image_path'],'image_sha256':rec['image_sha256'],
                        'mask_path':rec['mask_path'],'mask_sha256':rec['mask_sha256'],
                        'prediction_path':path.name,'prediction_sha256':fit.sha(path),'comparison_path':panel_path.name,'comparison_sha256':fit.sha(panel_path)})
                result = {'kind':'small_cohort_memory_check','cohort_name':name,'sample_count':len(ids),'sample_ids':ids,
                    'reason':group.get('reason',''),'optimizer_steps':updates,'samples_seen':updates*len(ids),'exposures_per_sample':exposures,
                    'size':size,'seed':seed,'threads':threads,'optimizer':'Adam','learning_rate':learning_rate,
                    'base_width':base_width,'channel_widths':[base_width,base_width*2,base_width*4],'parameter_count':parameter_count,
                    'class_weighting':'sqrt_inverse_from_full_eligible_pool','loss_normalization':'batch_weight_sum',
                    'global_eligible_pool_count':len(cache),'global_class_counts':{str(c):int(histogram[c]) for c in range(3)},
                    'global_ignored_pixels':ignored,'class_weights':weights_np.tolist(),
                    'selection_sha256':selection_sha,'cohorts_sha256':config_sha,'input_dir':fit.relative(ds.input_dir,root),
                    'initial_parameter_sha256':initial_sha,'final_parameter_sha256':final_sha,'checkpoint_sha256':fit.sha(folder/'probe_checkpoint.pt'),
                    'checkpoint_roundtrip_equal':roundtrip,'history':history,'artifacts':artifacts,
                    'not_candidate_model':True,'semantic_reviewed':False,'independent_test_available':False,'source_relationship':'unknown',**clip_meta}
                fit.write_json(folder/'summary.json',result); completed.append(result)
        overview = {'kind':'cohort_memory_probe_comparison','selection_sha256':selection_sha,'selection_path':fit.relative(sp,root),
            'cohorts_sha256':config_sha,'steps':steps,'size':size,'seed':seed,'learning_rate':learning_rate,'not_candidate_model':True,
            'base_width':base_width,'channel_widths':[base_width,base_width*2,base_width*4],'parameter_count':completed[0]['parameter_count'],
            'semantic_reviewed':False,'independent_test_available':False,'source_relationship':'unknown',
            'cohorts':[{'name':r['cohort_name'],'sample_count':r['sample_count'],'sample_ids':r['sample_ids'],
                'summary_path':r['cohort_name']+'/summary.json','summary_sha256':fit.sha(out/r['cohort_name']/'summary.json'),
                'final_ridge_metrics':r['history'][-1]['classes']['1']} for r in completed],
            'note':'Targeted fitting memory probes only. Different cohort composition, not independent model ranking or verified label semantics.',
            **({'max_grad_norm':max_grad_norm} if max_grad_norm is not None else {})}
        fit.write_json(out/'summary.json',overview)
        with (out/'final_per_image.csv').open('w',encoding='utf-8-sig',newline='') as stream:
            fields=['cohort','sample_id','ridge_true_pixels','ridge_precision','ridge_recall','ridge_iou']
            writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
            for result in completed:
                for row in result['history'][-1]['per_image']:
                    c=row['classes']['1'];writer.writerow({'cohort':result['cohort_name'],'sample_id':row['sample_id'],
                        'ridge_true_pixels':c['true_pixels'],'ridge_precision':c['precision'],'ridge_recall':c['recall'],'ridge_iou':c['iou']})
        (out/'README.md').write_text('# 固定小群组记忆检查\n\n仅使用既有允许训练图片，每步看完整指定群组。不是候选模型、独立评估或标签语义验证。\n'+
            'cohorts.json保留输入配置原字节；每组summary含全部里程碑/逐图指标、来源和CP摘要。final_per_image.csv为固定最后一步，不挑最好的一步。\n'+
            '预览使用诊断分辨率像素最近邻放大2倍。不同群组的平均指标不作为泛化比较；重点看固定共同问题图。\n',encoding='utf-8')
        return overview
    finally:
        torch.set_num_threads(old_threads)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for flag in ('selection','cohorts','output'):parser.add_argument('--'+flag,required=True)
    for flag,default in (('steps',300),('size',128),('seed',20261017),('threads',4)):parser.add_argument('--'+flag,type=int,default=default)
    parser.add_argument('--learning-rate',type=float,default=.001)
    parser.add_argument('--base-width',type=int,default=8)
    parser.add_argument('--max-grad-norm',type=float,default=None)
    try:
        result=run_probe(**vars(parser.parse_args(argv)));print(json.dumps(result,ensure_ascii=False,allow_nan=False));return 0
    except (ProbeError,fit._input.RGBError,OSError,ValueError,RuntimeError,TypeError,KeyError) as exc:
        print('RGB cohort probe failed: '+str(exc));return 1


if __name__=='__main__':raise SystemExit(main())
