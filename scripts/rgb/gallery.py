"""Offline comparison of existing cohort probe artifacts; never runs inference."""
import argparse
import importlib.util
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch

ROOT=Path(__file__).resolve().parents[2]
_spec=importlib.util.spec_from_file_location('gallery_cohorts',ROOT/'scripts/rgb/cohort_probe.py')
cohort=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(cohort)
fit=cohort.fit;GalleryError=fit.FittingError


def child(parent,name):
    if not isinstance(name,str) or Path(name).name!=name or any(c in name for c in '/\\:'):
        raise GalleryError('Artifact filename must be a safe local component')
    if Path(name).suffix not in ('.png','.pt'):raise GalleryError('Unknown source artifact extension')
    try:fit._input._validate_sample_id(Path(name).stem)
    except fit._input.RGBError as exc:raise GalleryError(str(exc)) from exc
    p=(parent/name).resolve()
    if not fit.inside(p,parent.resolve()) or not p.is_file():raise GalleryError('Missing or escaped source artifact')
    return p


def metric(row):
    calculated=cohort.diag.measures(np.asarray(row['confusion_true_rows_predicted_columns']))
    if any(row.get(k)!=v for k,v in calculated.items()):raise GalleryError('Stored metric differs from confusion matrix')
    return calculated


def load_run(path,seed,width,selection_sha,allowed,steps,size):
    s=fit.read_json(path);folder=path.parent
    if not isinstance(s,dict) or s.get('seed')!=seed or s.get('optimizer_steps')!=steps or s.get('size')!=size:
        raise GalleryError('Run seed/step/resolution mismatch')
    for key,value in (('not_candidate_model',True),('semantic_reviewed',False),('independent_test_available',False)):
        if s.get(key) is not value:raise GalleryError('Require explicitly unreviewed, noncandidate fitting probe')
    if s.get('selection_sha256')!=selection_sha or s.get('source_relationship')!='unknown':raise GalleryError('Run selection/scope mismatch')
    ids=s.get('sample_ids')
    if not isinstance(ids,list) or not 1<=len(ids)<=16 or not all(isinstance(x,str) for x in ids) or len(set(ids))!=len(ids) or not set(ids)<=set(allowed):
        raise GalleryError('Duplicate or ineligible sample IDs')
    if s.get('sample_count')!=len(ids) or s.get('exposures_per_sample')!=dict.fromkeys(ids,steps) or s.get('samples_seen')!=steps*len(ids):
        raise GalleryError('Incomplete cohort exposure record')
    cp_path=child(folder,'probe_checkpoint.pt')
    if fit.sha(cp_path)!=s.get('checkpoint_sha256'):raise GalleryError('Checkpoint digest mismatch')
    cp=torch.load(cp_path,map_location='cpu',weights_only=True)
    if not isinstance(cp,dict) or cp.get('schema')!='agrolake_rgb_cohort_probe_v1' or cp.get('class_ids')!=fit.CLASSES or cp.get('sample_ids')!=ids or cp.get('selection_sha256')!=selection_sha:
        raise GalleryError('Unknown checkpoint binding')
    if cp.get('seed')!=seed or cp.get('optimizer_steps')!=steps or cp.get('size')!=size:raise GalleryError('Checkpoint metadata mismatch')
    for key,value in (('not_candidate_model',True),('semantic_reviewed',False),('independent_test_available',False)):
        if cp.get(key) is not value:raise GalleryError('Checkpoint scope mismatch')
    with torch.random.fork_rng(devices=[]):
        model=cohort.build_model(width);model.load_state_dict(cp['state_dict'],strict=True)
    count=sum(p.numel() for p in model.parameters())
    for obj in (s,cp):
        if obj.get('base_width',8)!=width or obj.get('parameter_count',count)!=count or obj.get('channel_widths',[8,16,32])!=[width,width*2,width*4]:
            raise GalleryError('Checkpoint architecture declaration mismatch')
    for p in model.parameters():fit.finite(p,'gallery checkpoint parameter')
    if fit.state_digest(cp['state_dict'])!=s.get('final_parameter_sha256'):raise GalleryError('Parameter digest mismatch')
    final=s['history'][-1]
    if final.get('step')!=steps or [r.get('sample_id') for r in final['per_image']]!=ids or [a.get('sample_id') for a in s['artifacts']]!=ids:
        raise GalleryError('Incomplete final predictions')
    group=metric(final);total=np.zeros((3,3),dtype=np.int64);samples={};inputs={str(path):fit.sha(path),str(cp_path):fit.sha(cp_path)}
    for sid,row,a in zip(ids,final['per_image'],s['artifacts']):
        rec=allowed[sid]
        for key in ('image_path','image_sha256','mask_path','mask_sha256'):
            if a.get(key)!=rec.get(key):raise GalleryError('Prediction source mismatch')
        per_image=metric(row);total+=np.asarray(row['confusion_true_rows_predicted_columns'])
        pred_path=child(folder,a['prediction_path']);panel_path=child(folder,a['comparison_path'])
        if fit.sha(pred_path)!=a['prediction_sha256'] or fit.sha(panel_path)!=a['comparison_sha256']:raise GalleryError('Prediction/preview digest mismatch')
        with Image.open(pred_path) as im:
            if im.mode!='L' or im.size!=(size,size):raise GalleryError('Invalid prediction mode/shape')
            pred=np.array(im)
        if not np.isin(pred,[0,1,2]).all():raise GalleryError('Unknown prediction class')
        with Image.open(panel_path) as im:
            if im.mode!='RGB' or im.size!=(size*6,size*2+32):raise GalleryError('Invalid diagnostic panel shape')
            panel=np.array(im)
        tiles=[panel[32:,j*size*2:(j+1)*size*2].copy() for j in range(3)]
        if not np.array_equal(tiles[2],fit.COLORS[pred].repeat(2,0).repeat(2,1)):raise GalleryError('Preview pixels disagree with prediction')
        truth_rgb=tiles[1][::2,::2];target=np.full((size,size),255,dtype=np.uint8)
        valid_color=np.all(truth_rgb==110,axis=2)
        for c,color in enumerate(fit.COLORS):
            matched=np.all(truth_rgb==color,axis=2);target[matched]=c;valid_color|=matched
        if not valid_color.all() or not np.array_equal(tiles[1],truth_rgb.repeat(2,0).repeat(2,1)):
            raise GalleryError('Unknown or resampled label-preview pixels')
        if cohort.diag.confusion(target,pred).tolist()!=row['confusion_true_rows_predicted_columns']:
            raise GalleryError('Displayed labels/prediction do not match metrics')
        samples[sid]={'metrics':per_image,'tiles':tiles};inputs[str(pred_path)]=fit.sha(pred_path);inputs[str(panel_path)]=fit.sha(panel_path)
    if total.tolist()!=group['confusion_true_rows_predicted_columns']:raise GalleryError('Per-image totals differ')
    return {'ids':ids,'group':group,'samples':samples,'parameter_count':count,'learning_rate':s['learning_rate'],'inputs':inputs}


def generate(config,output,root=ROOT):
    root=Path(root).resolve();cfg_path=fit.artifact(config,root);cfg=fit.read_json(cfg_path)
    if not isinstance(cfg,dict) or type(cfg.get('schema_version')) is not int or cfg['schema_version']!=1:raise GalleryError('Unknown gallery config')
    sp=fit.artifact(cfg.get('selection'),root);selection_sha=fit.sha(sp)
    if cfg.get('selection_sha256')!=selection_sha:raise GalleryError('Gallery selection digest mismatch')
    selection=fit.read_json(sp);records=selection.get('records') if isinstance(selection,dict) else None
    if not isinstance(records,list) or not all(isinstance(r,dict) and isinstance(r.get('sample_id'),str) for r in records) or len({r['sample_id'].casefold() for r in records})!=len(records):
        raise GalleryError('Invalid selection records')
    allowed={r['sample_id']:r for r in records if r.get('disposition')=='experiment_train' and r.get('experimental_training_allowed') is True and r.get('semantic_reviewed') is False}
    for sid in allowed:fit._input._validate_sample_id(sid)
    pairs=cfg.get('pairs');steps=cfg.get('steps');size=cfg.get('size');title=cfg.get('title','同一张图，比较两种网络')
    fit.bounded_int(steps,'steps',1,500);fit.bounded_int(size,'size',32,256)
    if size%8 or not isinstance(title,str) or not 1<=len(title)<=100 or not isinstance(pairs,list) or not 1<=len(pairs)<=3:
        raise GalleryError('Invalid gallery settings')
    paths=[];seen=set()
    for pair in pairs:
        seed=pair['seed'];fit.bounded_int(seed,'seed',0,2**32-1)
        if seed in seen:raise GalleryError('Duplicate gallery seed')
        seen.add(seed)
        paths.append((seed,fit.artifact(pair['reference_summary'],root),fit.artifact(pair['experiment_summary'],root)))
    out=fit.new_output(output,[cfg_path.parent,sp.parent,*[p.parent for _,a,b in paths for p in (a,b)]],root)
    views={};common={};inputs={str(cfg_path):fit.sha(cfg_path),str(sp):selection_sha};sample_ids=None;common_rate=None
    for seed,a,b in paths:
        ref=load_run(a,seed,8,selection_sha,allowed,steps,size);wide=load_run(b,seed,16,selection_sha,allowed,steps,size)
        if ref['ids']!=wide['ids'] or (sample_ids is not None and sample_ids!=ref['ids']) or ref['learning_rate']!=wide['learning_rate']:
            raise GalleryError('Runs must share sample order and learning rate')
        if common_rate is not None and ref['learning_rate']!=common_rate:raise GalleryError('Seed runs must share learning rate')
        common_rate=ref['learning_rate']
        sample_ids=ref['ids'];views[str(seed)]={'reference_group':ref['group'],'experiment_group':wide['group'],'samples':{}}
        for sid in sample_ids:
            rt=ref['samples'][sid]['tiles'];wt=wide['samples'][sid]['tiles']
            if not all(np.array_equal(rt[j],wt[j]) for j in (0,1)):raise GalleryError('Runs do not share original RGB/labels')
            if sid in common and not all(np.array_equal(common[sid][j],rt[j]) for j in (0,1)):raise GalleryError('Seed runs use different originals')
            common[sid]=rt[:2]
            views[str(seed)]['samples'][sid]={'reference':ref['samples'][sid]['metrics'],'experiment':wide['samples'][sid]['metrics'],
              'reference_image':f'assets/{seed}-reference-{sid}.png','experiment_image':f'assets/{seed}-experiment-{sid}.png',
              '_pixels':[rt[2],wt[2]]}
        inputs.update(ref['inputs']);inputs.update(wide['inputs'])
    if any(fit.sha(Path(path))!=digest for path,digest in inputs.items()):raise GalleryError('Source changed during gallery preflight')
    # Every source is checked before any output. No model forward/inference is called.
    out.mkdir(exist_ok=False);(out/'assets').mkdir()
    samples=[]
    for sid in sample_ids:
        for tag,pixels in zip(('rgb','labels'),common[sid]):Image.fromarray(pixels).save(out/f'assets/{tag}-{sid}.png')
        rec=allowed[sid];samples.append({'id':sid,'name':Path(rec['image_path']).name,'source_path':rec['image_path'],
          'image_sha256':rec['image_sha256'],'mask_sha256':rec['mask_sha256'],'rgb':f'assets/rgb-{sid}.png','labels':f'assets/labels-{sid}.png'})
    for seed,view in views.items():
        for sid,sample in view['samples'].items():
            for tag,pixels in zip(('reference','experiment'),sample.pop('_pixels')):Image.fromarray(pixels).save(out/sample[tag+'_image'])
    data={'title':title,'sample_count':len(sample_ids),'seeds':[seed for seed,_,_ in paths],'steps':steps,'size':size,
      'samples':samples,'views':views,'semantic_reviewed':False,'independent_test_available':False,'not_candidate_model':True,
      'selection_sha256':selection_sha,'source_relationship':'unknown','learning_rate':common_rate,
      'reference_base_width':8,'experiment_base_width':16,'reference_parameter_count':ref['parameter_count'],'experiment_parameter_count':wide['parameter_count']}
    encoded=json.dumps(data,ensure_ascii=False,allow_nan=False).replace('&','\\u0026').replace('<','\\u003c').replace('>','\\u003e')
    (out/'index.html').write_text(HTML.replace('__DATA__',encoded),encoding='utf-8');fit.write_json(out/'gallery.json',data)
    (out/'README.md').write_text('# RGB训练诊断对照页\n\n双击index.html，完整保留同目录assets。仅浏览已生成的训练图结果，不推理/上传/保存标注。\n'+
      '选择图片、初始化编号和类别，四图分别为原RGB/现有标注/原网络/宽网络。所有图均为诊断分辨率最近邻放大。\n'+
      '当前图片与全群组IoU不同；高重合度仅表示拟合现有标注，不是农业精度。语义/来源未知，未独立测试。\n'+
      '同seed编号不表示相同权重，网络宽度变化也改变算力。gallery.json含指标/来源，manifest.json含输入和派生文件摘要。\n',encoding='utf-8')
    manifest={'kind':'offline_rgb_cohort_gallery','selection_sha256':selection_sha,'sample_ids':sample_ids,'seeds':data['seeds'],
      'semantic_reviewed':False,'independent_test_available':False,'not_candidate_model':True,'inference_performed':False,
      'inputs':{fit.relative(Path(p),root):digest for p,digest in inputs.items()},
      'artifacts':{p.relative_to(out).as_posix():fit.sha(p) for p in sorted(out.rglob('*')) if p.is_file()}}
    fit.write_json(out/'manifest.json',manifest);return manifest


HTML=r'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src 'self' data:; style-src 'unsafe-inline'; script-src 'unsafe-inline'; connect-src 'none'; base-uri 'none'; form-action 'none'">
<link rel="icon" href="data:,"><title>农智数湖 · RGB训练对照</title><style>
:root{font-family:"Microsoft YaHei UI","Microsoft YaHei",sans-serif;color:#163b39;background:#e7eef2;font-size:15px}*{box-sizing:border-box}body{margin:0}header{padding:22px max(24px,calc((100% - 1280px)/2));display:flex;align-items:center;justify-content:space-between;background:#163b39;color:white}header strong{font-size:20px}header span{font-size:14px;color:#c3d6d2}main{max-width:1328px;padding:30px 24px 26px;margin:auto}h1{font-size:clamp(24px,3vw,36px);line-height:1.4;margin:0 0 10px;letter-spacing:-.4px}p{line-height:1.8;margin:8px 0}.scope{max-width:74ch;color:#405e5d}.controls{display:grid;grid-template-columns:2fr 1fr 1fr;gap:20px;margin:26px 0 20px}label{display:block;font-size:14px;font-weight:bold}select{display:block;width:100%;margin-top:8px;font:inherit;color:inherit;background:white;border:1px solid #96aeaa;border-radius:5px;padding:11px 12px;min-height:46px}button,a{font:inherit}button{padding:10px 18px;background:#15735f;border:1px solid #15735f;border-radius:5px;color:white;cursor:pointer}button:disabled{opacity:.5;cursor:default}a{color:#12644f;text-underline-offset:4px}select:focus-visible,button:focus-visible,a:focus-visible{outline:3px solid #d3a12f;outline-offset:3px}.images{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:16px}figure{margin:0;min-width:0;background:white;border:1px solid #c6d3d4;border-radius:5px;overflow:hidden}figcaption{padding:12px;font-weight:bold;font-size:14px}figure img{width:100%;height:auto;aspect-ratio:1;display:block;image-rendering:pixelated}.legend{display:flex;gap:20px;flex-wrap:wrap;margin:15px 0;color:#405e5d;font-size:13px}.legend i{display:inline-block;width:10px;height:10px;margin-right:6px;border-radius:2px}.legend span{white-space:nowrap}.note{font-size:13px;color:#526c6a}.metrics{background:white;padding:18px 22px;border-radius:5px;margin-top:22px}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}caption{text-align:left;font-weight:bold;font-size:16px;padding-bottom:12px}th,td{text-align:left;padding:12px 8px;border-bottom:1px solid #dee7e7}th{font-size:13px;color:#526c6a;font-weight:normal}td:nth-child(n+2){font-size:22px;font-weight:bold}tr:last-child td{border-bottom:0}.actions{display:flex;gap:12px;align-items:center;justify-content:space-between;margin:20px 0}.actions div{display:flex;gap:10px}.source{overflow-wrap:anywhere;font-size:13px;color:#526c6a}footer{border-top:1px solid #c6d3d4;padding-top:15px;max-width:78ch}.error{padding:12px;background:#fff0df;border:1px solid #b57926;color:#693e14}.error[hidden]{display:none}@media(max-width:700px){header{padding:18px 16px}main{padding:22px 16px}.controls{grid-template-columns:1fr 1fr;gap:14px}.controls label:first-child{grid-column:1/-1}.images{grid-template-columns:repeat(2,minmax(0,1fr));gap:10px}figcaption{padding:10px;font-size:13px}.metrics{padding:14px 10px}th,td{padding:10px 5px}td:nth-child(n+2){font-size:18px}.actions{align-items:flex-start;gap:14px;flex-wrap:wrap}.legend{gap:12px}}@media(prefers-reduced-motion:reduce){*{scroll-behavior:auto}}
</style></head><body><header><strong>农智数湖</strong><span>RGB训练诊断</span></header><main>
<h1 id="title"></h1><p class="scope" id="scope"></p><p class="scope">先选图片，再切换初始化编号。高重合度只表示拟合现有标注，不能当作新农田上的准确率。</p>
<div class="controls"><label>图片<select id="sample"></select></label><label>初始化编号<select id="seed"></select></label><label>查看类别<select id="class"><option value="1">田埂</option><option value="2">道路</option><option value="0">田块</option></select></label></div>
<p id="error" class="error" role="alert" hidden>图片读取失败，请完整保留对照页旁的assets目录，然后重新打开页面。</p>
<div class="images"><figure><figcaption>原图（诊断尺寸）</figcaption><img id="rgb" alt="所选农田图的RGB诊断像素"></figure><figure><figcaption>现有标注</figcaption><img id="labels" alt="现有田块、田埂和道路标注"></figure><figure><figcaption>原网络预测</figcaption><img id="reference" alt="原网络在训练图片上的预测"></figure><figure><figcaption>宽网络预测</figcaption><img id="experiment" alt="宽网络在训练图片上的预测"></figure></div>
<div class="legend"><span><i style="background:#28aa46"></i>田块</span><span><i style="background:#f0be28"></i>田埂</span><span><i style="background:#dc3232"></i>道路</span><span><i style="background:#6e6e6e"></i>忽略像素</span></div>
<p class="note" id="resolution"></p><div class="metrics"><table><caption id="metric-caption"></caption><thead><tr><th>网络</th><th>当前图片</th><th id="group-heading"></th></tr></thead><tbody><tr><td>原网络</td><td id="reference-sample"></td><td id="reference-group"></td></tr><tr><td>宽网络</td><td id="experiment-sample"></td><td id="experiment-group"></td></tr></tbody></table></div>
<div class="actions"><div><button id="prev" type="button">上一张</button><button id="next" type="button">下一张</button></div><a href="gallery.json" download>下载指标与来源记录</a></div><p class="source" id="source"></p>
<footer><p class="note">现有标注尚未完成语义复核，图片来源关系未知，未做独立测试。两种网络共享初始化编号，但权重形状和计算量不同。本页只查看已有结果，不上传图片、不运行模型、不修改标注。</p></footer>
</main><script id="gallery-data" type="application/json">__DATA__</script><script>
const data=JSON.parse(document.getElementById('gallery-data').textContent);const $=id=>document.getElementById(id);const names={'0':'田块','1':'田埂','2':'道路'};
$('title').textContent=data.title;$('scope').textContent=`全部${data.sample_count}张图片均参与训练，每次${data.steps}步。请选择同一张图查看两种网络。`;
$('resolution').textContent=`图像使用${data.size}×${data.size}诊断像素，按最近邻放大显示；灰色标注不计入IoU分母。`;$('group-heading').textContent=`${data.sample_count}图合计`;
for(const sample of data.samples){const option=document.createElement('option');option.value=sample.id;option.textContent=sample.name;$('sample').append(option)}
for(const seed of data.seeds){const option=document.createElement('option');option.value=String(seed);option.textContent=String(seed);$('seed').append(option)}
const percent=value=>value===null?'—':`${(value*100).toFixed(2)}%`;
function render(){const sample=data.samples.find(s=>s.id===$('sample').value);const view=data.views[$('seed').value];const row=view.samples[sample.id];const cls=$('class').value;$('error').hidden=true;
for(const [id,path] of Object.entries({rgb:sample.rgb,labels:sample.labels,reference:row.reference_image,experiment:row.experiment_image}))$(id).src=path;
for(const model of ['reference','experiment']){$(model+'-sample').textContent=percent(row[model].classes[cls].iou);$(model+'-group').textContent=percent(view[model+'_group'].classes[cls].iou)}
$('metric-caption').textContent=`${names[cls]} IoU：预测与现有标注的重合程度`;$('source').textContent=`来源：${sample.source_path}；样本编号：${sample.id}`;}
for(const id of ['sample','seed','class'])$(id).addEventListener('change',render);
for(const id of ['rgb','labels','reference','experiment'])$(id).addEventListener('error',()=>$('error').hidden=false);
for(const [id,delta] of [['prev',-1],['next',1]]){$(id).disabled=data.samples.length<2;$(id).addEventListener('click',()=>{$('sample').selectedIndex=($('sample').selectedIndex+delta+data.samples.length)%data.samples.length;render()})}render();
</script></body></html>'''


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True);parser.add_argument('--output',required=True)
    try:
        result=generate(**vars(parser.parse_args(argv)));print(json.dumps({'sample_count':len(result['sample_ids']),'seeds':result['seeds'],'artifact_count':len(result['artifacts'])+1},ensure_ascii=False));return 0
    except (GalleryError,fit._input.RGBError,OSError,ValueError,TypeError,KeyError,RuntimeError) as exc:
        print('RGB gallery failed: '+str(exc));return 1


if __name__=='__main__':raise SystemExit(main())
