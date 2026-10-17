"""Bounded whole-pool diagnostic training; not a deployed or independently tested model."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader

ROOT=Path(__file__).resolve().parents[2]
_spec=importlib.util.spec_from_file_location('pool_cohort',ROOT/'scripts/rgb/cohort_probe.py')
cohort=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(cohort)
fit=cohort.fit;diag=cohort.diag
SCHEMA='agrolake_rgb_pool_diagnostic_v1'


def evaluate(model,loader,weights,folder=None,sources=None):
    model.eval();matrix=np.zeros((3,3),dtype=np.int64);rows=[];numerator=0.;denominator=0.
    with torch.no_grad():
        for x,y,ids in loader:
            logits=model(x);fit.finite(logits,'pool evaluation logits')
            n,d=fit.weighted_loss_components(logits,y,weights);numerator+=n.item();denominator+=d
            guesses=logits.argmax(1).numpy().astype(np.uint8)
            for sid,target,pred in zip(ids,y.numpy(),guesses):
                cm=diag.confusion(target,pred);matrix+=cm
                row={'sample_id':sid,'ignored_pixels':int((target==255).sum()),**diag.measures(cm)}
                if folder is not None:
                    path=folder/f'{sid}.png';Image.fromarray(pred).save(path)
                    rec=sources[sid]
                    row.update({k:rec[k] for k in ('image_path','image_sha256','json_path','json_sha256','mask_path','mask_sha256')})
                    row.update(prediction_path='predictions/'+path.name,prediction_sha256=fit.sha(path))
                rows.append(row)
    return {'weighted_ce':numerator/denominator,'ignored_pixels':sum(r['ignored_pixels'] for r in rows),
            'sample_count':len(rows),**diag.measures(matrix)},rows


def train(selection,output,epochs=10,size=128,batch_size=8,seed=20261017,threads=4,root=ROOT):
    for v,n,lo,hi in ((epochs,'epochs',1,20),(size,'size',32,256),(batch_size,'batch_size',1,32),
                      (seed,'seed',0,2**32-1),(threads,'threads',1,8)):
        fit.bounded_int(v,n,lo,hi)
    if size%8:raise fit.FittingError('size must be divisible by eight')
    sp=fit.artifact(selection,root);out=fit.new_output(output,[sp.parent],root);selection_sha=fit.sha(sp)
    old_threads=torch.get_num_threads()
    try:
        torch.set_num_threads(threads)
        _,selected,ds,cache,hist,ignored=fit.verify_selection(sp,root,size)
        if fit.sha(sp)!=selection_sha:raise fit.FittingError('Selection changed')
        if fit.inside(out,Path(ds.input_dir)) or fit.inside(Path(ds.input_dir),out):raise fit.FittingError('Output overlaps input')
        weights_np=fit.class_weights_from_counts(hist);weights=torch.tensor(weights_np)
        ids=[r[0] for r in cache];sources={r['sample_id']:r for r in ds.records};dataset=fit.CachedImages(cache)
        generator=torch.Generator().manual_seed(seed)
        train_loader=DataLoader(dataset,batch_size=batch_size,shuffle=True,generator=generator,num_workers=0)
        eval_loader=DataLoader(dataset,batch_size=batch_size,shuffle=False,num_workers=0,
                               generator=torch.Generator().manual_seed((seed+1)%2**32))
        out.mkdir(parents=True,exist_ok=False);(out/'predictions').mkdir()
        milestones={0,epochs,epochs-1,*[n for n in (5,) if n<epochs]}
        saved=[];history=[];exposures=dict.fromkeys(ids,0);updates=0
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed);model=cohort.build_model(16);optimizer=torch.optim.Adam(model.parameters(),lr=.001)
            initial_sha=fit.state_digest(model.state_dict())
            metadata={'schema':SCHEMA,'size':size,'seed':seed,'threads':threads,'batch_size':batch_size,
                'base_width':16,'class_ids':fit.CLASSES,'class_weights':weights_np.tolist(),
                'learning_rate':.001,'max_grad_norm':5.0,'selection_sha256':selection_sha,'training_sample_ids':ids,
                'not_candidate_model':True,'semantic_reviewed':False,'independent_test_available':False,
                'source_relationship':'unknown','evaluation_scope':'fitting_samples_only'}

            def snapshot(epoch):
                if epoch not in milestones:return
                path=out/f'epoch-{epoch:02d}.pt'
                torch.save({**metadata,'epoch':epoch,'optimizer_steps':updates,'state_dict':model.state_dict(),
                    'optimizer_state_dict':optimizer.state_dict(),'shuffle_generator_state':generator.get_state()},path)
                saved.append({'epoch':epoch,'path':path.name,'sha256':fit.sha(path),
                    'parameter_sha256':fit.state_digest(model.state_dict())})

            evaluation,_=evaluate(model,eval_loader,weights)
            history.append({'epoch':0,'evaluation':evaluation});snapshot(0)
            with (out/'batches.jsonl').open('w',encoding='utf-8',newline='\n') as trace:
                for epoch in range(1,epochs+1):
                    model.train();seen=[];sizes=[];total_n=0.;total_d=0.;clipped=0
                    for x,y,batch_ids in train_loader:
                        optimizer.zero_grad(set_to_none=True);logits=model(x);fit.finite(logits,'pool training logits')
                        n,d=fit.weighted_loss_components(logits,y,weights);loss=n/d;loss.backward()
                        for p in model.parameters():
                            if p.grad is None:raise fit.FittingError('Disconnected parameter')
                            fit.finite(p.grad,'pool gradient')
                        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True,foreach=False)
                        after=torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(p.grad) for p in model.parameters()]))
                        fit.finite(after,'pool clipped norm')
                        if after.item()>5*(1+1e-5):raise fit.FittingError('Clipping bound violated')
                        scale=torch.clamp(5/(norm+1e-6),max=1.).item();optimizer.step()
                        for p in model.parameters():fit.finite(p,'pool parameter')
                        updates+=1;seen.extend(batch_ids);sizes.append(len(batch_ids));total_n+=n.item();total_d+=d;clipped+=scale<1
                        for sid in batch_ids:exposures[sid]+=1
                        trace.write(json.dumps({'step':updates,'epoch':epoch,'sample_ids':list(batch_ids),
                            'loss_numerator':n.item(),'weight_denominator':d,'loss_before_update':loss.item(),
                            'gradient_l2_before':norm.item(),'gradient_l2_after':after.item(),'scale':scale,'clipped':scale<1},allow_nan=False)+'\n')
                        trace.flush()
                    if len(seen)!=len(ids) or set(seen)!=set(ids):raise fit.FittingError('Incomplete pool epoch')
                    evaluation,_=evaluate(model,eval_loader,weights)
                    row={'epoch':epoch,'training_loss':total_n/total_d,'loss_numerator':total_n,'weight_denominator':total_d,
                        'sample_ids':seen,'batch_sizes':sizes,'clipped_updates':clipped,'evaluation':evaluation}
                    history.append(row);snapshot(epoch)
                    print(json.dumps({'seed':seed,'epoch':epoch,'training_loss':row['training_loss'],
                        'fitting_ce':evaluation['weighted_ce'],'ridge_iou':evaluation['classes']['1']['iou']}),flush=True)
            if updates!=epochs*((len(ids)+batch_size-1)//batch_size) or any(n!=epochs for n in exposures.values()):
                raise fit.FittingError('Pool update or exposure mismatch')
            if fit.sha(sp)!=selection_sha:raise fit.FittingError('Selection changed during training')
            final_sha=fit.state_digest(model.state_dict())
            if final_sha==initial_sha:raise fit.FittingError('No parameter updates')
            final_eval,records=evaluate(model,eval_loader,weights,out/'predictions',sources)
            if final_eval!=history[-1]['evaluation']:raise fit.FittingError('Final evaluation changed')
            loaded=torch.load(out/saved[-1]['path'],map_location='cpu',weights_only=True)
            clone=cohort.build_model(16);clone.load_state_dict(loaded['state_dict'],strict=True);clone.eval()
            refx=next(iter(eval_loader))[0]
            with torch.no_grad():equal=torch.equal(clone(refx),model(refx))
            if not equal:raise fit.FittingError('Pool checkpoint roundtrip failed')
        if [r['sample_id'] for r in records]!=ids:raise fit.FittingError('Prediction coverage mismatch')
        fit.write_json(out/'records.json',records)
        summary={**metadata,'kind':'whole_pool_clipped_diagnostic_training','selection_path':fit.relative(sp,root),
            'input_dir':fit.relative(ds.input_dir,root),'input_manifest_sha256':selected['manifest_sha256'],
            'input_policy_sha256':selected['policy_sha256'],'epochs':epochs,'sample_count':len(ids),
            'optimizer_steps':updates,'samples_seen':epochs*len(ids),'exposures_per_sample':exposures,
            'parameter_count':sum(p.numel() for p in model.parameters()),'global_class_counts':hist.tolist(),
            'global_ignored_pixels':ignored,'initial_parameter_sha256':initial_sha,'final_parameter_sha256':final_sha,
            'checkpoint_roundtrip_equal':equal,'snapshots':saved,'history':history,
            'records_sha256':fit.sha(out/'records.json'),'batches_sha256':fit.sha(out/'batches.jsonl'),
            'split_created':False,'torch_version':str(torch.__version__)}
        fit.write_json(out/'summary.json',summary)
        (out/'README.md').write_text('734允许池（合成测试可更少）的独立版本诊断训练。所有图片都参加拟合，无独立测试、未核标签语义、非候选。\n'
            '每轮evaluation为固定当轮末态完整同池评估，training_loss为更新中批次损失的加权合计，二者不可混用。\n'
            '固定末态及中间CP含Adam和洗牌状态，仅供复核；预测为诊断网格像素，不报真实面积。\n',encoding='utf-8')
        return summary
    finally:torch.set_num_threads(old_threads)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('selection','output'):p.add_argument('--'+key,required=True)
    for key,default in (('epochs',10),('size',128),('batch-size',8),('seed',20261017),('threads',4)):
        p.add_argument('--'+key,type=int,default=default)
    try:
        s=train(**vars(p.parse_args(argv)));print(json.dumps({'seed':s['seed'],'optimizer_steps':s['optimizer_steps']}));return 0
    except (fit.FittingError,fit._input.RGBError,OSError,RuntimeError,ValueError,TypeError,KeyError) as exc:
        print('Pool diagnostic failed: '+str(exc));return 1


if __name__=='__main__':raise SystemExit(main())
