"""Observe a prescribed cohort probe without changing its optimization trajectory."""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location('traced_cohort', ROOT/'scripts/rgb/cohort_probe.py')
cohort = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cohort)
fit = cohort.fit
TraceError = fit.FittingError


def l2(tensor):
    value = float(tensor.detach().double().square().sum().sqrt().item())
    if not math.isfinite(value):
        raise TraceError('Nonfinite diagnostic norm')
    return value


def run_trace(selection, cohorts, reference, output, steps=500, size=128,
              seed=20261018, threads=4, root=ROOT):
    """One width16/Adam.001 group; immutable reference must match the whole run."""
    for value, name, lo, hi in ((steps,'steps',1,500),(size,'size',32,256),
                                (seed,'seed',0,2**32-1),(threads,'threads',1,8)):
        fit.bounded_int(value,name,lo,hi)
    if size % 8:
        raise TraceError('size must be divisible by eight')
    sp, cp, rp = [fit.artifact(p,root) for p in (selection,cohorts,reference)]
    out = fit.new_output(output,[sp.parent,cp.parent,rp.parent],root)
    bindings = {fit.relative(p,root):fit.sha(p) for p in (sp,cp,rp)}
    config_bytes = cp.read_bytes()
    config = json.loads(config_bytes.decode('utf-8'))
    ref = fit.read_json(rp)
    selection_sha, config_sha = fit.sha(sp), hashlib.sha256(config_bytes).hexdigest()
    old_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(threads)
        _, _, ds, cache, histogram, ignored = fit.verify_selection(sp,root,size)
        by_id = {sid:(rgb,target) for sid,rgb,target in cache}
        groups = cohort.validate_cohorts(config,selection_sha,set(by_id))
        if len(groups) != 1:
            raise TraceError('Trace requires exactly one prescribed cohort')
        group = groups[0]; ids = group['sample_ids']
        expected = {'kind':'small_cohort_memory_check','sample_ids':ids,
            'cohort_name':group['name'],'optimizer_steps':steps,'size':size,'seed':seed,
            'threads':threads,'optimizer':'Adam','learning_rate':.001,'base_width':16,
            'selection_sha256':selection_sha,'cohorts_sha256':config_sha,
            'not_candidate_model':True,'semantic_reviewed':False,'independent_test_available':False}
        if any(ref.get(k) != v for k,v in expected.items()):
            raise TraceError('Reference does not match prescribed experiment')
        checkpoint = rp.parent/'probe_checkpoint.pt'
        if fit.sha(checkpoint) != ref['checkpoint_sha256']:
            raise TraceError('Reference checkpoint changed')
        bindings[fit.relative(checkpoint,root)] = fit.sha(checkpoint)
        reference_rows = {r['step']:r for r in ref['history']}
        milestones = {0,steps,*[n for n in (50,100,200,300) if n < steps]}
        if set(reference_rows) != milestones:
            raise TraceError('Reference milestone set mismatch')
        if fit.inside(out,Path(ds.input_dir)) or fit.inside(Path(ds.input_dir),out):
            raise TraceError('Output overlaps candidate inputs')
        weights_np = fit.class_weights_from_counts(histogram)
        if weights_np.tolist() != ref['class_weights']:
            raise TraceError('Reference weights mismatch')
        for path,digest in bindings.items():
            if fit.sha(Path(root)/path) != digest:
                raise TraceError('Bound input changed during verification')
        weights = torch.tensor(weights_np)
        x = torch.stack([torch.from_numpy(by_id[sid][0].copy()).permute(2,0,1).float()/255 for sid in ids])
        y = torch.stack([torch.from_numpy(by_id[sid][1].astype(np.int64)) for sid in ids])
        snapshots = sorted({0,steps,*range(25,steps+1,25)})
        out.mkdir(parents=True,exist_ok=False)
        (out/'cohorts.json').write_bytes(config_bytes)
        saved = []; matched = []; history = []
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            model = cohort.build_model(16)
            initial_sha = fit.state_digest(model.state_dict())
            if initial_sha != ref['initial_parameter_sha256']:
                raise TraceError('Reference initialization mismatch')
            optimizer = torch.optim.Adam(model.parameters(),lr=.001)

            def observe(step):
                evaluation,_ = cohort.evaluate(model,x,y,ids,step,weights)
                if step in reference_rows:
                    if evaluation != reference_rows[step]:
                        raise TraceError('Trajectory differs from reference at step '+str(step))
                    matched.append(step)
                if step in snapshots:
                    filename = f'step-{step:04d}.pt'
                    payload = {'schema':'agrolake_rgb_trace_probe_v1',
                        'state_dict':model.state_dict(),'optimizer_state_dict':optimizer.state_dict(),
                        'optimizer_steps':step,'seed':seed,'size':size,'threads':threads,
                        'base_width':16,'learning_rate':.001,'sample_ids':ids,
                        'selection_sha256':selection_sha,'cohorts_sha256':config_sha,
                        'class_weights':weights_np.tolist(),'not_candidate_model':True,
                        'semantic_reviewed':False,'independent_test_available':False}
                    torch.save(payload,out/filename)
                    saved.append({'step':step,'path':filename,'sha256':fit.sha(out/filename),
                                  'parameter_sha256':fit.state_digest(model.state_dict())})
                return evaluation

            first = observe(0); history.append(first)
            with (out/'trace.jsonl').open('w',encoding='utf-8',newline='\n') as stream:
                stream.write(json.dumps({'step':0,'evaluation':first},allow_nan=False)+'\n');stream.flush()
                for step in range(1,steps+1):
                    model.train(); optimizer.zero_grad(set_to_none=True)
                    logits = model(x);fit.finite(logits,'trace training logits')
                    numerator,denominator = fit.weighted_loss_components(logits,y,weights)
                    loss = numerator/denominator;fit.finite(loss,'trace training loss');loss.backward()
                    before = {}; norms = {}
                    for name,parameter in model.named_parameters():
                        if parameter.grad is None:
                            raise TraceError('Disconnected parameter')
                        fit.finite(parameter.grad,'trace gradient')
                        before[name] = parameter.detach().clone()
                        norms[name] = {'gradient_l2':l2(parameter.grad),
                                       'parameter_l2_before':l2(parameter)}
                    optimizer.step()
                    for name,parameter in model.named_parameters():
                        fit.finite(parameter,'trace parameter')
                        norms[name]['update_l2'] = l2(parameter.detach()-before[name])
                    evaluation = observe(step)
                    row = {'step':step,'loss_before_update':float(loss.item()),
                        **{key:math.sqrt(sum(n[key]**2 for n in norms.values()))
                           for key in ('gradient_l2','parameter_l2_before','update_l2')},
                        'parameters':norms,'evaluation':evaluation}
                    stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush()
                    if step in milestones: history.append(evaluation)
                    if step % 25 == 0 or step == steps:
                        print(json.dumps({k:row[k] for k in ('step','loss_before_update','gradient_l2','update_l2')}|
                            {'loss_after_update':evaluation['weighted_ce'],
                             'ridge_iou':evaluation['classes']['1']['iou']}),flush=True)
            final_sha = fit.state_digest(model.state_dict())
            if final_sha != ref['final_parameter_sha256'] or set(matched) != milestones:
                raise TraceError('Final trajectory binding failed')
            loaded = torch.load(out/saved[-1]['path'],weights_only=True,map_location='cpu')
            clone = cohort.build_model(16);clone.load_state_dict(loaded['state_dict'],strict=True);clone.eval()
            with torch.no_grad():
                if not torch.equal(clone(x),model(x)):
                    raise TraceError('Snapshot roundtrip mismatch')
        for path,digest in bindings.items():
            if fit.sha(Path(root)/path) != digest:
                raise TraceError('Bound input changed during trace')
        sources = {r['sample_id']:r for r in ds.records}
        result = {'kind':'prescribed_cohort_optimization_trace','sample_ids':ids,
            'optimizer_steps':steps,'samples_seen':steps*len(ids),
            'exposures_per_sample':{sid:steps for sid in ids},'size':size,'seed':seed,'threads':threads,
            'base_width':16,'parameter_count':sum(p.numel() for p in model.parameters()),
            'optimizer':'Adam','learning_rate':.001,'class_weights':weights_np.tolist(),
            'global_eligible_pool_count':len(cache),'global_ignored_pixels':ignored,
            'global_class_counts':histogram.tolist(),'inputs_sha256':bindings,
            'sources':[sources[sid] for sid in ids],
            'initial_parameter_sha256':initial_sha,'final_parameter_sha256':final_sha,
            'reference_trajectory_exact':True,'matched_reference_steps':sorted(matched),
            'checkpoint_roundtrip_equal':True,'snapshots':saved,'snapshot_schedule':snapshots,
            'trace_sha256':fit.sha(out/'trace.jsonl'),'history':history,
            'not_candidate_model':True,'semantic_reviewed':False,
            'independent_test_available':False,'source_relationship':'unknown'}
        fit.write_json(out/'summary.json',result)
        (out/'README.md').write_text('# 固定回退组逐步追踪\n\n仅已见小群组的优化诊断；非候选模型、独立评估或标签语义验证。\n'
            'trace.jsonl第0步仅评估；之后每行记录更新前损失/梯度/参数范数、实际参数移动范数、更新后逐图和总体指标。\n'
            'L2范数用float64累加；不裁剪梯度、不改优化器。0、每25步及最后一步固定保存模型和Adam状态，未挑选最佳模型。\n'
            'summary.json只有精确匹配原实验全部里程碑/初始化/末态且保存重载通过后才写入。\n',encoding='utf-8')
        return result
    finally:
        torch.set_num_threads(old_threads)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('selection','cohorts','reference','output'):parser.add_argument('--'+flag,required=True)
    for flag,default in (('steps',500),('size',128),('seed',20261018),('threads',4)):
        parser.add_argument('--'+flag,type=int,default=default)
    try:
        result = run_trace(**vars(parser.parse_args(argv)))
        print(json.dumps({'reference_trajectory_exact':result['reference_trajectory_exact'],
                          'optimizer_steps':result['optimizer_steps']}));return 0
    except (TraceError,fit._input.RGBError,OSError,ValueError,RuntimeError,TypeError,KeyError) as exc:
        print('RGB trace failed: '+str(exc));return 1


if __name__ == '__main__':raise SystemExit(main())
