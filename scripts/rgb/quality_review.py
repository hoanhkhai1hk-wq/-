"""Derive pixel regions and a targeted human label review from existing pool predictions."""
from collections import deque
import csv
import importlib.util
from pathlib import Path
import sqlite3

import numpy as np
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parents[2]
_spec=importlib.util.spec_from_file_location('quality_pool',ROOT/'scripts/rgb/full_pool.py')
pool=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(pool)
fit=pool.fit;diag=pool.diag


def regions(prediction):
    """All four-connected class-0 regions, including single pixels; grid coordinates only."""
    a=np.asarray(prediction)
    if a.ndim!=2 or not np.issubdtype(a.dtype,np.integer) or not np.isin(a,[0,1,2]).all():
        raise fit.FittingError('Require integer prediction grid with classes 0/1/2')
    h,w=a.shape;seen=np.zeros(a.shape,dtype=bool);result=[]
    for y in range(h):
        for x in range(w):
            if a[y,x]!=0 or seen[y,x]:continue
            seen[y,x]=True;queue=deque([(y,x)]);area=0;x0=x1=x;y0=y1=y
            while queue:
                yy,xx=queue.popleft();area+=1;x0=min(x0,xx);x1=max(x1,xx);y0=min(y0,yy);y1=max(y1,yy)
                for ny,nx in ((yy-1,xx),(yy+1,xx),(yy,xx-1),(yy,xx+1)):
                    if 0<=ny<h and 0<=nx<w and not seen[ny,nx] and a[ny,nx]==0:
                        seen[ny,nx]=True;queue.append((ny,nx))
            result.append({'region_index':len(result)+1,'area_pixels':area,'x_min':x0,'y_min':y0,
                'x_max_exclusive':x1+1,'y_max_exclusive':y1+1,'grid_width':w,'grid_height':h})
    return result


def choose_cases(rows):
    selected=[];used=set();limit=min(12,len(rows))
    def take(candidates,n,reason):
        for row in candidates:
            if len(selected)>=limit or n==0:return
            if row['sample_id'] in used:continue
            selected.append({'case_id':f'R{len(selected)+1:02d}','reason':reason,**row})
            used.add(row['sample_id']);n-=1
    ridge=[r for r in rows if r['ridge_true_pixels']>0]
    take(sorted(ridge,key=lambda r:(r['mean_ridge_iou'],r['sample_id'])),4,'low_mean_ridge_iou')
    take(sorted(ridge,key=lambda r:(-r['ridge_iou_spread'],r['sample_id'])),4,'large_seed_spread')
    road=[r for r in rows if r['road_true_pixels']>0]
    take(sorted(road,key=lambda r:(r['mean_road_iou'],r['sample_id'])),2,'low_mean_road_iou')
    remaining=sorted([r for r in rows if r['sample_id'] not in used],key=lambda r:r['sample_id'])
    positions=np.linspace(0,len(remaining)-1,limit-len(selected),dtype=int) if remaining else []
    take([remaining[i] for i in positions],limit-len(selected),'uniform_remainder')
    return selected


def build(selection,training_dirs,output,root=ROOT,baseline=None):
    if not isinstance(training_dirs,list) or len(training_dirs)!=3:raise fit.FittingError('Require three prescribed fits')
    sp=fit.artifact(selection,root);folders=[fit.artifact(p,root) for p in training_dirs]
    if len(set(folders))!=3:raise fit.FittingError('Duplicate training directories')
    baseline_folder=fit.artifact(baseline,root) if baseline is not None else None
    inputs=[sp.parent,*folders]+([baseline_folder] if baseline_folder else [])
    out=fit.new_output(output,inputs,root);bindings={fit.relative(sp,root):fit.sha(sp)}
    _,selected,ds,cache,_,_=fit.verify_selection(sp,root,128);ids=[r[0] for r in cache]
    source={r['sample_id']:r for r in ds.records};summaries=[];models=[];all_predictions=[]
    for folder in folders:
        s=fit.read_json(folder/'summary.json')
        if s.get('schema')!=pool.SCHEMA or s.get('size')!=128 or s.get('selection_sha256')!=fit.sha(sp) or s.get('training_sample_ids')!=ids:
            raise fit.FittingError('Training/selection/grid binding mismatch')
        if s.get('not_candidate_model') is not True or s.get('semantic_reviewed') is not False or s.get('independent_test_available') is not False:
            raise fit.FittingError('Invalid diagnostic flags')
        if fit.sha(folder/'records.json')!=s['records_sha256']:raise fit.FittingError('Records changed')
        records=fit.read_json(folder/'records.json')
        if [r['sample_id'] for r in records]!=ids:raise fit.FittingError('Record coverage mismatch')
        predictions={};cm=np.zeros((3,3),dtype=np.int64)
        for row,(_,_,target) in zip(records,cache):
            sid=row['sample_id'];rec=source[sid]
            if any(row.get(k)!=rec[k] for k in ('image_path','image_sha256','json_path','json_sha256','mask_path','mask_sha256')):
                raise fit.FittingError('Record source binding mismatch')
            expected='predictions/'+sid+'.png'
            if row['prediction_path']!=expected:raise fit.FittingError('Unexpected prediction path')
            path=folder/expected
            if fit.sha(path)!=row['prediction_sha256']:raise fit.FittingError('Prediction changed')
            with Image.open(path) as im:
                if im.mode!='L' or im.size!=(128,128):raise fit.FittingError('Wrong prediction grid')
                pred=np.array(im)
            matrix=diag.confusion(target,pred)
            if any(row[k]!=v for k,v in diag.measures(matrix).items()):raise fit.FittingError('Prediction metrics changed')
            cm+=matrix;predictions[sid]=pred;bindings[fit.relative(path,root)]=row['prediction_sha256']
        if any(s['history'][-1]['evaluation'][k]!=v for k,v in diag.measures(cm).items()):raise fit.FittingError('Pool metrics changed')
        summaries.append(s);models.append({r['sample_id']:r for r in records});all_predictions.append(predictions)
        for p in (folder/'summary.json',folder/'records.json'):bindings[fit.relative(p,root)]=fit.sha(p)
    if [s['seed'] for s in summaries]!=[20261017,20261018,20261019]:raise fit.FittingError('Require prescribed seed order')
    baseline_rows=None;baseline_metrics=None
    if baseline_folder is not None:
        bs=fit.read_json(baseline_folder/'summary.json');br=fit.read_json(baseline_folder/'records.json')
        if bs.get('kind')!='whole_fitting_set_diagnostic' or bs.get('selection_sha256')!=fit.sha(sp) or [r['sample_id'] for r in br]!=ids:
            raise fit.FittingError('Baseline binding mismatch')
        cm=np.zeros((3,3),dtype=np.int64)
        for r,(_,_,target) in zip(br,cache):
            sid=r['sample_id'];expected='predictions/'+sid+'.png'
            if r['prediction_path']!=expected or fit.sha(baseline_folder/expected)!=r['prediction_sha256']:raise fit.FittingError('Baseline prediction changed')
            with Image.open(baseline_folder/expected) as im:pred=np.array(im)
            matrix=diag.confusion(target,pred);cm+=matrix
            if any(r[k]!=v for k,v in diag.measures(matrix).items()):raise fit.FittingError('Baseline metrics changed')
            bindings[fit.relative(baseline_folder/expected,root)]=r['prediction_sha256']
        if any(bs['model'][k]!=v for k,v in diag.measures(cm).items()):raise fit.FittingError('Baseline aggregate mismatch')
        baseline_rows={r['sample_id']:r for r in br};baseline_metrics=bs['model']
        for p in (baseline_folder/'summary.json',baseline_folder/'records.json'):bindings[fit.relative(p,root)]=fit.sha(p)
    rows=[];region_rows=[]
    for sid in ids:
        ridge=[m[sid]['classes']['1']['iou'] for m in models];road=[m[sid]['classes']['2']['iou'] for m in models]
        rp=[v for v in ridge if v is not None];dp=[v for v in road if v is not None]
        row={'sample_id':sid,'ridge_true_pixels':models[0][sid]['classes']['1']['true_pixels'],
            'road_true_pixels':models[0][sid]['classes']['2']['true_pixels'],
            'mean_ridge_iou':sum(rp)/len(rp) if rp else None,'mean_road_iou':sum(dp)/len(dp) if dp else None,
            'ridge_iou_spread':max(rp)-min(rp) if rp else None,
            **{f'ridge_iou_seed{s["seed"]}':v for s,v in zip(summaries,ridge)}}
        rows.append(row)
        if baseline_rows is not None:
            previous=baseline_rows[sid]['classes']['1']['iou'];row['baseline_ridge_iou']=previous
            row['mean_minus_baseline_ridge_iou']=row['mean_ridge_iou']-previous if previous is not None and row['mean_ridge_iou'] is not None else None
        for r in regions(all_predictions[0][sid]):region_rows.append({'sample_id':sid,**r})
    cases=choose_cases(rows)
    for p,h in bindings.items():
        if fit.sha(Path(root)/p)!=h:raise fit.FittingError('Inputs changed before publication')
    out.mkdir(parents=True,exist_ok=False);(out/'cases').mkdir()
    fit.write_json(out/'quality.json',rows);fit.write_json(out/'regions.json',region_rows)
    fields=['sample_id','region_index','area_pixels','x_min','y_min','x_max_exclusive','y_max_exclusive','grid_width','grid_height']
    with (out/'regions.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(region_rows)
    with sqlite3.connect(out/'regions.sqlite') as connection:
        connection.execute('PRAGMA foreign_keys=ON')
        connection.execute('CREATE TABLE sample (sample_id TEXT PRIMARY KEY,image_path TEXT,image_sha256 TEXT,prediction_path TEXT,prediction_sha256 TEXT)')
        connection.executemany('INSERT INTO sample VALUES (?,?,?,?,?)',[(sid,models[0][sid]['image_path'],models[0][sid]['image_sha256'],
            fit.relative(folders[0]/models[0][sid]['prediction_path'],root),models[0][sid]['prediction_sha256']) for sid in ids])
        connection.execute('CREATE TABLE region (sample_id TEXT,region_index INTEGER,area_pixels INTEGER,x_min INTEGER,y_min INTEGER,x_max_exclusive INTEGER,y_max_exclusive INTEGER,grid_width INTEGER,grid_height INTEGER,PRIMARY KEY(sample_id,region_index),FOREIGN KEY(sample_id) REFERENCES sample(sample_id))')
        connection.executemany('INSERT INTO region VALUES (?,?,?,?,?,?,?,?,?)',[tuple(r[k] for k in fields) for r in region_rows])
        connection.execute('CREATE TABLE metadata (key TEXT PRIMARY KEY,value TEXT)')
        final_snapshot=summaries[0]['snapshots'][-1]
        connection.executemany('INSERT INTO metadata VALUES (?,?)',[('unit','diagnostic_grid_pixels'),('connectivity','4'),
            ('demonstration_seed','20261017'),('semantic_reviewed','false'),('physical_area','unknown'),
            ('selection_sha256',fit.sha(sp)),('checkpoint_path',fit.relative(folders[0]/final_snapshot['path'],root)),
            ('checkpoint_sha256',final_snapshot['sha256'])])
    connection.close()
    contact=Image.new('RGB',(1152,160*((len(cases)+2)//3)),'white');contact_draw=ImageDraw.Draw(contact)
    instructions=['# 现有标注定向审核（不是独立测试集）','',
        '请只审核现有标注是否代表你想识别的目标：黄色是否确为田埂；绿色田块、红色道路是否有明显错标/漏标。灰色为忽略区。',
        '可以按R编号回复“符合预期 / 有问题（描述位置） / 不清楚”。不需要重新修那48份，未知Fieldridge及两批来源也不重复询问。',
        '这12张按较低拟合/初始化分歧/道路问题定向选取，不能当随机样本或验证集。现有标注未自动修改。','']
    for j,case in enumerate(cases):
        sid=case['sample_id'];rec=source[sid];ip=Path(ds._resolve_source_rel(rec['image_path'],sid));mp=Path(ds._resolve_derived_rel(rec['mask_path'],sid))
        with Image.open(ip) as im:rgb=im.convert('RGB').copy()
        with Image.open(mp) as im:target=np.array(im)
        truth=np.full((*target.shape,3),110,dtype=np.uint8)
        for c in range(3):truth[target==c]=fit.COLORS[c]
        images=[('Original RGB',rgb),('Existing labels',Image.fromarray(truth))]
        for s,preds in zip(summaries,all_predictions):images.append((f'Seed {s["seed"]}: 128-grid prediction',Image.fromarray(fit.COLORS[preds[sid]])))
        width,height=rgb.size;panel=Image.new('RGB',(width*5,height+32),'white');draw=ImageDraw.Draw(panel)
        for k,(label,im) in enumerate(images):
            draw.text((k*width+5,8),label,fill='black')
            panel.paste(im.resize((width,height),Image.Resampling.NEAREST),(k*width,32))
        path=out/'cases'/f'{case["case_id"]}.png';panel.save(path)
        tx=(j%3)*384;ty=(j//3)*160;contact_draw.text((tx+3,ty+5),case['case_id']+' RGB / labels / seed17',fill='black')
        for k,im in enumerate((rgb,Image.fromarray(truth),images[2][1])):
            contact.paste(im.resize((128,128),Image.Resampling.NEAREST),(tx+k*128,ty+32))
        case.update(panel_path='cases/'+path.name,panel_sha256=fit.sha(path),image_path=rec['image_path'],
            mask_path=rec['mask_path'],json_path=rec['json_path'],review_status='unknown')
        instructions.append(f'- [{case["case_id"]}: {sid}](cases/{case["case_id"]}.png)；选择原因：{case["reason"]}；原JSON：{rec["json_path"]}')
    contact.save(out/'review-contact.png');fit.write_json(out/'case-index.json',cases)
    with (out/'review-decisions.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['case_id','sample_id','label_match','boundary_issues','note']);w.writeheader()
        for c in cases:w.writerow({'case_id':c['case_id'],'sample_id':c['sample_id'],'label_match':'unknown','boundary_issues':'unknown','note':''})
    (out/'review.md').write_text('\n'.join(instructions)+'\n',encoding='utf-8')
    summary={'kind':'pool_pixel_regions_and_targeted_human_review','sample_count':len(ids),'training_seeds':[s['seed'] for s in summaries],
        'fitting_metrics':[{'seed':s['seed'],**s['history'][-1]['evaluation']} for s in summaries],
        'baseline_fitting_metrics':baseline_metrics,
        'region_demonstration_seed':20261017,'region_count':len(region_rows),'region_area_pixels':sum(r['area_pixels'] for r in region_rows),
        'grid_size':[128,128],'connectivity':4,'area_unit':'diagnostic_grid_pixels','physical_area':None,
        'review_count':len(cases),'review_status':'awaiting_human_label_target_review','inputs_sha256':bindings,
        'semantic_reviewed':False,'independent_test_available':False,'source_relationship':'unknown','not_candidate_model':True,
        'artifacts':{p.relative_to(out).as_posix():fit.sha(p) for p in out.rglob('*') if p.is_file()}}
    fit.write_json(out/'summary.json',summary);return summary
