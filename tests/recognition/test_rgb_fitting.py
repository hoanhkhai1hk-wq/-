"""Independent safety and numerical checks for internal fitting; retained fixtures."""
import importlib.util
import json
import hashlib
from pathlib import Path
import unittest
from unittest import mock
import uuid

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[2]

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

fit=load('fitting_tested',ROOT/'scripts/rgb/fit_baseline.py')
select=load('selection_tested',ROOT/'scripts/rgb/real_baseline.py')
class _Fixtures:
    CASE=ROOT/'reports/rgb'/('fitting-tests-'+uuid.uuid4().hex[:8])
    json=json

    @staticmethod
    def dump(p,value):
        p.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')

    @staticmethod
    def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

    @staticmethod
    def rel(p):return p.relative_to(ROOT).as_posix()

    @classmethod
    def fixture(cls,name):
        base=cls.CASE/name;base.mkdir(parents=True);(base/'src').mkdir();(base/'masks').mkdir()
        policy={'class_ids':{'0':'Field','1':'Field ridge','2':'Road','255':'ignore'},'no_background_class':True,
                'alias_assumption':{'map':{'Fieldridge':'Field ridge'},'status':'pending_semantic_review'},
                'training_policy':{'semantic_reviewed':False,'not_training_ready':True,'no_train_test_split':True}}
        records=[]
        for i in range(6):
            sid=f'sample_{i}';status='review' if i==5 else 'candidate'
            a=np.zeros((32,32,3),dtype=np.uint8);a[:,:11]=(190,30,40);a[:,11:21]=(30,190,50);a[:,21:]=(40,50,190)
            mask=np.zeros((32,32),dtype=np.uint8);mask[:,11:21]=1;mask[:,21:]=2;mask[0,:]=255
            ip=base/'src'/f'{sid}.png';jp=ip.with_suffix('.json');mp=base/'masks'/f'{sid}.png'
            Image.fromarray(a).save(ip);Image.fromarray(mask).save(mp)
            labels=['Field','Fieldridge' if i==4 else 'Field ridge','Road']
            shapes=[{'label':label,'shape_type':'polygon','points':[[j*10,1],[j*10+8,1],[j*10+8,30],[j*10,30]]} for j,label in enumerate(labels)]
            cls.dump(jp,{'imagePath':ip.name,'imageWidth':32,'imageHeight':32,'shapes':shapes})
            records.append({'sample_id':sid,'dataset':'synthetic','status':status,'width':32,'height':32,
                'image_path':cls.rel(ip),'json_path':cls.rel(jp),'image_sha256':cls.sha(ip),'json_sha256':cls.sha(jp),
                'image_bytes':ip.stat().st_size,'json_bytes':jp.stat().st_size,
                'mask_path':f'masks/{sid}.png' if status=='candidate' else None,'mask_sha256':cls.sha(mp) if status=='candidate' else None,
                'label_mapping':[{'shape_index':j,'original_label':v,'derived_class':j} for j,v in enumerate(labels)],
                'alias_normalized':[{'shape_index':1,'from':'Fieldridge','to':'Field ridge'}] if i==4 else [],
                'pixel_stats':{'denominator':1024,'ignore_pixels':32,'class_pixels':{str(v):int((mask==v).sum()) for v in range(3)}} if status=='candidate' else None,
                'semantic_reviewed':False,'not_training_ready':True,'review_reasons':[{'type':'synthetic_review'}] if status=='review' else []})
        cls.dump(base/'manifest.json',{'records':records});cls.dump(base/'policy.json',policy)
        return base

probe=_Fixtures


def fixture(name):
    base=probe.fixture(name)
    manifest=probe.json.loads((base/'manifest.json').read_text(encoding='utf-8'))
    records=[]
    for rec in manifest['records']:
        row=dict(rec)
        disposition='excluded_user_decision' if rec['status']=='review' else ('deferred_alias' if rec['alias_normalized'] else 'experiment_train')
        row.update(disposition=disposition,experimental_training_allowed=disposition=='experiment_train',semantic_reviewed=False)
        records.append(row)
    selected=probe.CASE/(name+'-selected');selected.mkdir()
    probe.dump(selected/'selection.json',{'schema_version':1,'input_dir':probe.rel(base),'manifest_sha256':probe.sha(base/'manifest.json'),
        'policy_sha256':probe.sha(base/'policy.json'),'records':records})
    return base,selected/'selection.json'


class FittingTests(unittest.TestCase):
    def test_global_normalization_batch_partition_gradient_property(self):
        logits=torch.tensor([[[[2.,-1.]],[[0.,1.]],[[1.,0.]]],[[[0.,1.]],[[2.,0.]],[[1.,2.]]]],requires_grad=True)
        targets=torch.tensor([[[0,255]],[[1,2]]]);w=torch.tensor([.1,1.,2.]);expected=.5
        total,mass=fit.weighted_loss_components(logits,targets,w,'global_expected_weight',expected)
        whole=total/mass;whole_gradient=torch.autograd.grad(whole,logits,retain_graph=True)[0]
        pieces=[]
        for i in range(2):
            numerator,denominator=fit.weighted_loss_components(logits[i:i+1],targets[i:i+1],w,'global_expected_weight',expected)
            pieces.append((numerator/denominator)*int((targets[i]!=255).sum())/3)
        partitioned=sum(pieces);partition_gradient=torch.autograd.grad(partitioned,logits,retain_graph=True)[0]
        self.assertAlmostEqual(whole.item(),partitioned.item(),places=6)
        torch.testing.assert_close(whole_gradient,partition_gradient)
        old,old_mass=fit.weighted_loss_components(logits,targets,w)
        torch.testing.assert_close(old/old_mass,F.cross_entropy(logits,targets,weight=w,ignore_index=255))
        self.assertEqual(float(whole_gradient[0,:,0,1].abs().sum()),0.)
        with self.assertRaises(fit.FittingError):fit.weighted_loss_components(logits,targets,w,'global_expected_weight',0)

    def test_global_normalization_training_metadata(self):
        _,selection=fixture('global-'+uuid.uuid4().hex[:6]);out=probe.CASE/('global-fit-'+uuid.uuid4().hex[:6])
        result=fit.train(selection,out,epochs=1,size=32,threads=1,seed=17,class_weighting='inverse_frequency',loss_normalization='global_expected_weight')
        self.assertEqual(result['loss_normalization'],'global_expected_weight')
        cp=torch.load(out/'checkpoint.pt',weights_only=True,map_location='cpu')
        self.assertEqual(cp['loss_normalization'],'global_expected_weight')
        self.assertGreater(cp['expected_weight_per_valid_pixel'],0)
    def test_class_weight_strategies_and_legacy_default(self):
        counts=np.array([80,15,5],dtype=np.int64)
        old=np.sqrt(counts.sum()/(3*counts.astype(np.float64)));old=(old/old.mean()).astype(np.float32)
        np.testing.assert_array_equal(fit.class_weights_from_counts(counts),old)
        w=fit.class_weights_from_counts(counts,'inverse_frequency')
        np.testing.assert_allclose(w*counts,np.full(3,80*w[0]),rtol=1e-6)
        for bad in ([80,0,5],[1.,2.,3.],[True,True,True],[1,2]):
            with self.assertRaises(fit.FittingError):fit.class_weights_from_counts(bad)
        with self.assertRaises(fit.FittingError):fit.class_weights_from_counts(counts,'bad')

    def test_inverse_weighting_runs_and_is_recorded(self):
        _,selection=fixture('inverse-'+uuid.uuid4().hex[:6]);out=probe.CASE/('inverse-fit-'+uuid.uuid4().hex[:6])
        result=fit.train(selection,out,epochs=1,size=32,batch_size=3,threads=1,seed=17,class_weighting='inverse_frequency')
        self.assertEqual(result['class_weighting'],'inverse_frequency');self.assertEqual(result['optimizer_steps'],2)
        cp=torch.load(out/'checkpoint.pt',weights_only=True,map_location='cpu')
        self.assertEqual(cp['class_weighting'],'inverse_frequency');self.assertEqual(cp['class_weights'],result['class_weights'])
    def test_selection_all_dispositions_and_rerun_refusal(self):
        base=probe.fixture('selection-'+uuid.uuid4().hex[:6]);out=probe.CASE/('selection-out-'+uuid.uuid4().hex[:6])
        result=select.select_candidates(str(base),str(out))
        self.assertEqual(result['dispositions'],{'experiment_train':4,'deferred_alias':1,'excluded_user_decision':1})
        fit.verify_selection(out/'selection.json',size=32)
        with self.assertRaises(select.BaselineError):select.select_candidates(str(base),str(out))

    def test_selection_preflight_rejections(self):
        for kind in ('mapping','histogram','duplicate','unknown_label'):
            base=probe.fixture(kind+'-select-'+uuid.uuid4().hex[:6]);path=base/'manifest.json'
            m=probe.json.loads(path.read_text(encoding='utf-8'))
            if kind=='mapping':m['records'][4]['label_mapping'][1]['original_label']='Field ridge'
            elif kind=='histogram':m['records'][0]['pixel_stats']['class_pixels']['0']=1
            elif kind=='duplicate':m['records'][1]['sample_id']='sample_0'
            else:
                p=base/'src/sample_0.json';j=probe.json.loads(p.read_text(encoding='utf-8'));j['shapes'][0]['label']='Drain';probe.dump(p,j)
                m['records'][0].update(json_sha256=probe.sha(p),json_bytes=p.stat().st_size)
            probe.dump(path,m);out=probe.CASE/('bad-selection-'+uuid.uuid4().hex[:6])
            with self.subTest(kind=kind),self.assertRaises((select.BaselineError,select.ts.RGBError)):
                select.select_candidates(str(base),str(out))
            self.assertFalse(out.exists())

    def test_real_steps_coverage_roundtrip_and_independent_reference(self):
        base,selection=fixture('train-'+uuid.uuid4().hex[:6]);out=probe.CASE/('fit-'+uuid.uuid4().hex[:6])
        rng=torch.random.get_rng_state().clone();threads=torch.get_num_threads()
        result=fit.train(selection,out,epochs=2,size=32,batch_size=3,threads=1,seed=17)
        self.assertTrue(torch.equal(rng,torch.random.get_rng_state()));self.assertEqual(threads,torch.get_num_threads())
        self.assertEqual(result['sample_count'],4);self.assertEqual(result['optimizer_steps'],4)
        self.assertNotEqual(result['initial_state_sha256'],result['final_state_sha256'])
        self.assertFalse(result['independent_test_available']);self.assertFalse(result['semantic_reviewed'])
        for row in result['history']:
            self.assertEqual(row['batch_sizes'],[3,1]);self.assertEqual(len(row['sample_ids']),4)
            self.assertEqual(set(row['sample_ids']),{f'sample_{i}' for i in range(4)})
            self.assertAlmostEqual(row['loss'],row['loss_numerator']/row['weight_denominator'],places=9)
        checkpoint=torch.load(out/'checkpoint.pt',weights_only=True,map_location='cpu')
        model=fit.SmallUNet();model.load_state_dict(checkpoint['state_dict'],strict=True);model.eval()
        _,_,_,cache,_,_=fit.verify_selection(selection,size=32)
        xs=[];ys=[]
        for sid in result['reference_sample_ids']:
            _,rgb,target=next(r for r in cache if r[0]==sid)
            xs.append(torch.from_numpy(rgb.copy()).permute(2,0,1).float()/255);ys.append(torch.from_numpy(target.astype(np.int64)))
        torch.set_num_threads(1)
        try:
            with torch.no_grad():v=F.cross_entropy(model(torch.stack(xs)),torch.stack(ys),weight=torch.tensor(result['class_weights']),ignore_index=255).item()
        finally:torch.set_num_threads(threads)
        self.assertAlmostEqual(v,result['reference_final_loss'],places=6)
        predicted=probe.CASE/('predict-'+uuid.uuid4().hex[:6])
        p=fit.predict(out/'checkpoint.pt',selection,'sample_0',predicted)
        with Image.open(predicted/'prediction.png') as image:a=np.array(image)
        self.assertEqual(a.shape,(32,32))
        self.assertEqual(sum(p['predicted_pixel_counts'].values()),1024)
        self.assertEqual(p['predicted_pixel_counts'],{str(i):int((a==i).sum()) for i in range(3)})
        with Image.open(predicted/'comparison.png') as image:self.assertEqual(image.size,(96,64))
        with self.assertRaises(fit.FittingError):fit.predict(out/'checkpoint.pt',selection,'sample_4',probe.CASE/'alias-prediction')
        with self.assertRaises(fit.FittingError):fit.train(selection,out,size=32,threads=1)

    def test_invalid_parameters_no_output(self):
        _,selection=fixture('params-'+uuid.uuid4().hex[:6])
        for kwargs in ({'epochs':True},{'epochs':0},{'size':33},{'batch_size':33},{'threads':0},{'seed':-1},{'seed':2**32},{'class_weighting':'bad'}):
            out=probe.CASE/('invalid-'+uuid.uuid4().hex[:6])
            with self.subTest(kwargs=kwargs),self.assertRaises(fit.FittingError):fit.train(selection,out,**kwargs)
            self.assertFalse(out.exists())

    def test_alias_cannot_be_promoted_by_selection(self):
        _,selection=fixture('promote-'+uuid.uuid4().hex[:6]);s=probe.json.loads(selection.read_text(encoding='utf-8'))
        s['records'][4].update(disposition='experiment_train',experimental_training_allowed=True);probe.dump(selection,s)
        out=probe.CASE/'promoted-output'
        with self.assertRaises(fit.FittingError):fit.train(selection,out,size=32,threads=1)
        self.assertFalse(out.exists())

    def test_selection_cannot_drop_excluded_rows(self):
        _,selection=fixture('drop-'+uuid.uuid4().hex[:6]);s=probe.json.loads(selection.read_text(encoding='utf-8'))
        s['records'].pop();probe.dump(selection,s)
        with self.assertRaises(fit.FittingError):fit.verify_selection(selection,size=32)

    def test_source_and_mask_tampering(self):
        for kind in ('source','mask'):
            base,selection=fixture(kind+'-'+uuid.uuid4().hex[:6])
            p=base/('src/sample_0.json' if kind=='source' else 'masks/sample_0.png')
            p.write_bytes(b'altered synthetic fixture')
            with self.subTest(kind=kind),self.assertRaises((fit.FittingError,fit._input.RGBError)):
                fit.verify_selection(selection,size=32)

    def test_paths_and_empty_paths(self):
        for path in ('','reports/rgb','reports/labels/new','../outside'):
            with self.subTest(path=path),self.assertRaises(fit.FittingError):fit.new_output(path,[])
        with self.assertRaises(fit.FittingError):fit.new_output('reports/rgb/a/child',[ROOT/'reports/rgb/a'])

    def test_ignore_gradient_and_nonfinite_guard(self):
        logits=torch.randn((1,3,2,2),requires_grad=True)
        y=torch.tensor([[[0,1],[2,255]]]);loss=F.cross_entropy(logits,y,ignore_index=255);loss.backward()
        self.assertEqual(float(logits.grad[0,:,1,1].abs().sum()),0.)
        with self.assertRaises(fit.FittingError):fit.finite(torch.tensor(float('nan')),'probe')

    def test_nonfinite_optimizer_result_never_reports_success(self):
        _,selection=fixture('nonfinite-'+uuid.uuid4().hex[:6]);out=probe.CASE/('failed-fit-'+uuid.uuid4().hex[:6])
        def corrupt_step(optimizer,*args,**kwargs):
            with torch.no_grad():optimizer.param_groups[0]['params'][0].fill_(float('inf'))
        with mock.patch.object(torch.optim.Adam,'step',new=corrupt_step),self.assertRaises(fit.FittingError):
            fit.train(selection,out,epochs=1,size=32,threads=1)
        self.assertTrue(out.exists());self.assertFalse((out/'summary.json').exists());self.assertFalse((out/'checkpoint.pt').exists())


if __name__=='__main__':unittest.main()
