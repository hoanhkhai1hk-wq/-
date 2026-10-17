"""Small synthetic cohort checks; retained fixtures, no real photographs."""
import importlib.util
from pathlib import Path
import unittest
from unittest import mock
import uuid
import numpy as np
from PIL import Image
import torch
from tests.recognition.test_rgb_fitting import fixture, probe

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('cohort_tested',ROOT/'scripts/rgb/cohort_probe.py')
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)


def prepare(name):
    base,selection=fixture(name)
    # Make sample_0 field-only so subgroup and whole-pool histograms differ.
    m=probe.json.loads((base/'manifest.json').read_text(encoding='utf-8'))
    a=np.zeros((32,32),dtype=np.uint8);a[0]=255;path=base/'masks/sample_0.png';Image.fromarray(a).save(path)
    rec=m['records'][0];rec['mask_sha256']=probe.sha(path)
    rec['pixel_stats']['class_pixels']={str(c):int((a==c).sum()) for c in range(3)}
    probe.dump(base/'manifest.json',m)
    s=probe.json.loads(selection.read_text(encoding='utf-8'));s['manifest_sha256']=probe.sha(base/'manifest.json');s['records'][0]['mask_sha256']=probe.sha(path);probe.dump(selection,s)
    folder=probe.CASE/(name+'-config');folder.mkdir();config=folder/'cohorts.json'
    probe.dump(config,{'schema_version':1,'selection_sha256':probe.sha(selection),'cohorts':[
        {'name':'small2','sample_ids':['sample_0','sample_1']},{'name':'mixed4','sample_ids':['sample_0','sample_1','sample_2','sample_3']}]})
    return selection,config


class CohortProbeTests(unittest.TestCase):
    def test_full_cohort_updates_same_initialization_and_global_weights(self):
        selection,config=prepare('cohort-'+uuid.uuid4().hex[:6]);out=probe.CASE/('cohort-out-'+uuid.uuid4().hex[:6])
        rng=torch.random.get_rng_state().clone();threads=torch.get_num_threads()
        result=mod.run_probe(selection,config,out,steps=3,size=32,seed=17,threads=1)
        self.assertTrue(torch.equal(rng,torch.random.get_rng_state()));self.assertEqual(threads,torch.get_num_threads())
        self.assertTrue(result['not_candidate_model']);self.assertFalse(result['independent_test_available'])
        initial=[]
        _,_,_,cache,hist,_=mod.fit.verify_selection(selection,size=32)
        expected=np.sqrt(hist.sum()/(3*hist.astype(float)));expected=(expected/expected.mean()).astype(np.float32)
        byid={sid:(rgb,target) for sid,rgb,target in cache}
        for group in result['cohorts']:
            folder=out/group['name'];s=mod.fit.read_json(folder/'summary.json');initial.append(s['initial_parameter_sha256'])
            self.assertEqual(s['optimizer_steps'],3);self.assertEqual(s['samples_seen'],3*s['sample_count'])
            self.assertTrue(all(n==3 for n in s['exposures_per_sample'].values()));self.assertEqual([r['step'] for r in s['history']],[0,3])
            self.assertEqual(s['global_eligible_pool_count'],4);np.testing.assert_array_equal(s['class_weights'],expected)
            self.assertNotEqual(s['initial_parameter_sha256'],s['final_parameter_sha256'])
            cp=torch.load(folder/'probe_checkpoint.pt',weights_only=True,map_location='cpu')
            model=mod.fit.SmallUNet();model.load_state_dict(cp['state_dict'],strict=True);model.eval()
            xs=torch.stack([torch.from_numpy(byid[sid][0].copy()).permute(2,0,1).float()/255 for sid in s['sample_ids']])
            with torch.no_grad():guess=model(xs).argmax(1).numpy()
            total=np.zeros((3,3),dtype=np.int64)
            for sid,pred,artifact in zip(s['sample_ids'],guess,s['artifacts']):
                with Image.open(folder/artifact['prediction_path']) as im:np.testing.assert_array_equal(np.array(im),pred)
                matrix=np.zeros((3,3),dtype=np.int64)
                for true,g in zip(byid[sid][1].ravel(),pred.ravel()):
                    if true!=255:matrix[true,g]+=1
                total+=matrix
            self.assertEqual(total.tolist(),s['history'][-1]['confusion_true_rows_predicted_columns'])
            with self.assertRaises(mod.fit.FittingError):mod.fit.predict(folder/'probe_checkpoint.pt',selection,'sample_0',probe.CASE/('candidate-'+uuid.uuid4().hex[:6]))
        self.assertEqual(initial[0],initial[1])
        with self.assertRaises(mod.ProbeError):mod.run_probe(selection,config,out)

    def test_bad_configuration_rejected_before_output(self):
        selection,config=prepare('reject-'+uuid.uuid4().hex[:6]);original=mod.fit.read_json(config)
        cases=[]
        for ids in (['sample_4'],['sample_5'],['sample_0','sample_0'],['../escape']):
            bad=probe.json.loads(probe.json.dumps(original));bad['cohorts'][0]['sample_ids']=ids;cases.append(bad)
        bad=probe.json.loads(probe.json.dumps(original));bad['cohorts'][0]['name']='../escape';cases.append(bad)
        bad=probe.json.loads(probe.json.dumps(original));bad['selection_sha256']='0'*64;cases.append(bad)
        for value in cases:
            probe.dump(config,value);out=probe.CASE/('rejected-'+uuid.uuid4().hex[:6])
            with self.assertRaises(mod.ProbeError):mod.run_probe(selection,config,out,steps=1,size=32,threads=1)
            self.assertFalse(out.exists())

    def test_invalid_parameters_no_output(self):
        selection,config=prepare('params-cohort-'+uuid.uuid4().hex[:6])
        invalid_lr = [True, 0, -.001, float('nan'), float('inf'), 1e-7, .1, '0.0003', None]
        for kwargs in [{'steps':True},{'steps':0},{'steps':501},{'size':33},{'seed':-1},{'threads':0},
                       *[{'learning_rate':value} for value in invalid_lr],
                       *[{'max_grad_norm':value} for value in (True,0,-1,float('nan'),float('inf'),1e-7,1e7,'5')],
                       *[{'base_width':value} for value in (True,0,9,32,8.0,'16',None)]]:
            out=probe.CASE/('invalid-cohort-'+uuid.uuid4().hex[:6])
            with self.subTest(kwargs=kwargs),self.assertRaises(mod.ProbeError):mod.run_probe(selection,config,out,**kwargs)
            self.assertFalse(out.exists())

    def test_default_model_matches_existing_initialization_and_forward(self):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261017);old=mod.fit.SmallUNet().eval()
            torch.manual_seed(20261017);default=mod.build_model().eval()
            self.assertEqual(type(default),type(old))
            self.assertEqual(mod.fit.state_digest(old.state_dict()),mod.fit.state_digest(default.state_dict()))
            x=torch.linspace(0,1,3*32*32).reshape(1,3,32,32)
            with torch.no_grad():self.assertTrue(torch.equal(old(x),default(x)))
            self.assertEqual(sum(p.numel() for p in default.parameters()),29779)

    def test_wide_model_updates_saves_and_reloads_with_declared_shape(self):
        selection,config=prepare('wide-cohort-'+uuid.uuid4().hex[:6]);out=probe.CASE/('wide-out-'+uuid.uuid4().hex[:6])
        result=mod.run_probe(selection,config,out,steps=2,size=32,seed=17,threads=1,base_width=16)
        summary=mod.fit.read_json(out/'small2/summary.json')
        cp=torch.load(out/'small2/probe_checkpoint.pt',weights_only=True,map_location='cpu')
        for obj in (result,summary,cp):
            self.assertEqual(obj['base_width'],16);self.assertEqual(obj['channel_widths'],[16,32,64]);self.assertEqual(obj['parameter_count'],118307)
        self.assertNotEqual(summary['initial_parameter_sha256'],summary['final_parameter_sha256'])
        wide=mod.build_model(16);wide.load_state_dict(cp['state_dict'],strict=True);wide.eval()
        self.assertEqual(tuple(cp['state_dict']['enc1.0.weight'].shape),(16,3,3,3))
        self.assertEqual(tuple(cp['state_dict']['head.weight'].shape),(3,16,1,1))
        for parameter in wide.parameters():self.assertTrue(torch.isfinite(parameter).all())
        with self.assertRaises(RuntimeError):mod.fit.SmallUNet().load_state_dict(cp['state_dict'],strict=True)
        _,_,_,cache,_,_=mod.fit.verify_selection(selection,size=32);byid={sid:rgb for sid,rgb,_ in cache}
        xs=torch.stack([torch.from_numpy(byid[sid].copy()).permute(2,0,1).float()/255 for sid in summary['sample_ids']])
        with torch.no_grad():pred=wide(xs).argmax(1).numpy()
        for sid,expected in zip(summary['sample_ids'],pred):
            with Image.open(out/'small2'/f'{sid}.png') as im:np.testing.assert_array_equal(np.array(im),expected)
        with self.assertRaises(mod.fit.FittingError):mod.fit.predict(out/'small2/probe_checkpoint.pt',selection,'sample_0',probe.CASE/('wide-reject-'+uuid.uuid4().hex[:6]))

    def test_learning_rate_scales_first_update_and_is_recorded(self):
        selection,config=prepare('lr-cohort-'+uuid.uuid4().hex[:6]);states=[]
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(17);initial=mod.fit.SmallUNet().state_dict()
        for rate in (.001,.0003):
            out=probe.CASE/('lr-out-'+uuid.uuid4().hex[:6])
            kwargs={} if rate==.001 else {'learning_rate':rate}
            result=mod.run_probe(selection,config,out,steps=1,size=32,seed=17,threads=1,**kwargs)
            self.assertEqual(result['learning_rate'],rate)
            summary=mod.fit.read_json(out/'small2/summary.json')
            cp=torch.load(out/'small2/probe_checkpoint.pt',weights_only=True,map_location='cpu')
            self.assertEqual(summary['learning_rate'],rate);self.assertEqual(cp['learning_rate'],rate)
            states.append(cp['state_dict'])
        movement=0.
        for name,start in initial.items():
            old_delta=states[0][name]-start;new_delta=states[1][name]-start
            self.assertTrue(torch.allclose(new_delta,old_delta*.3,rtol=1e-4,atol=2e-8),name)
            movement+=float(new_delta.abs().sum())
        self.assertGreater(movement,0)

    def test_input_change_during_run_cannot_publish_success(self):
        selection,config=prepare('changed-cohort-'+uuid.uuid4().hex[:6]);out=probe.CASE/('changed-out-'+uuid.uuid4().hex[:6])
        original_step=torch.optim.Adam.step
        def changed(optimizer,*args,**kwargs):
            result=original_step(optimizer,*args,**kwargs);config.write_text('{}',encoding='utf-8');return result
        with mock.patch.object(torch.optim.Adam,'step',new=changed),self.assertRaises(mod.ProbeError):
            mod.run_probe(selection,config,out,steps=1,size=32,threads=1)
        self.assertFalse((out/'summary.json').exists());self.assertFalse((out/'small2/summary.json').exists())

    def test_clipping_preserves_direction_enforces_bound_and_records_adam(self):
        selection,config=prepare('clip-'+uuid.uuid4().hex[:6]);out=probe.CASE/('clip-out-'+uuid.uuid4().hex[:6])
        original=torch.nn.utils.clip_grad_norm_;observed=[]
        def inspect(parameters,max_norm,**kwargs):
            parameters=list(parameters);before=[p.grad.detach().clone() for p in parameters]
            norm=original(parameters,max_norm,**kwargs)
            coefficient=torch.clamp(max_norm/(norm+1e-6),max=1.0)
            self.assertTrue(kwargs['error_if_nonfinite']);self.assertFalse(kwargs['foreach'])
            for start,p in zip(before,parameters):self.assertTrue(torch.equal(p.grad,start*coefficient))
            actual=sum(float(p.grad.double().square().sum()) for p in parameters)**.5
            self.assertLessEqual(actual,max_norm*(1+1e-5));observed.append(norm.item())
            return norm
        with mock.patch.object(torch.nn.utils,'clip_grad_norm_',new=inspect):
            result=mod.run_probe(selection,config,out,steps=2,size=32,threads=1,base_width=16,max_grad_norm=.01)
        self.assertEqual(len(observed),4);self.assertEqual(result['max_grad_norm'],.01)
        for group in result['cohorts']:
            folder=out/group['name'];summary=mod.fit.read_json(folder/'summary.json');meta=summary['gradient_clipping']
            rows=[probe.json.loads(line) for line in (folder/meta['trace_path']).read_text().splitlines()]
            self.assertEqual([r['step'] for r in rows],[1,2]);self.assertTrue(all(r['clipped'] for r in rows))
            self.assertEqual(meta['clipped_updates'],2);self.assertEqual(meta['trace_sha256'],mod.fit.sha(folder/meta['trace_path']))
            self.assertEqual([a['step'] for a in meta['snapshots']],[0,1,2])
            for a in meta['snapshots']:
                cp=torch.load(folder/a['path'],weights_only=True,map_location='cpu')
                self.assertEqual(cp['max_grad_norm'],.01);self.assertTrue(cp['not_candidate_model'])
                self.assertEqual(mod.fit.state_digest(cp['state_dict']),a['parameter_sha256'])
                self.assertTrue(all(s['step'].item()==a['step'] for s in cp['optimizer_state_dict']['state'].values()))
            final=torch.load(folder/'probe_checkpoint.pt',weights_only=True,map_location='cpu')
            self.assertEqual(final['gradient_clipping'],meta)

    def test_default_and_explicit_none_skip_clipping_and_keep_identical_trajectory(self):
        selection,config=prepare('noclip-'+uuid.uuid4().hex[:6]);states=[]
        for kwargs in ({},{'max_grad_norm':None}):
            out=probe.CASE/('noclip-out-'+uuid.uuid4().hex[:6])
            with mock.patch.object(torch.nn.utils,'clip_grad_norm_',side_effect=AssertionError('Default must not clip')):
                mod.run_probe(selection,config,out,steps=2,size=32,threads=1,**kwargs)
            summary=mod.fit.read_json(out/'small2/summary.json');states.append(summary)
            self.assertNotIn('max_grad_norm',summary);self.assertFalse((out/'small2/gradient_trace.jsonl').exists())
            self.assertEqual(list((out/'small2').glob('clip-step-*.pt')),[])
        self.assertEqual(states[0]['history'],states[1]['history'])
        self.assertEqual(states[0]['initial_parameter_sha256'],states[1]['initial_parameter_sha256'])
        self.assertEqual(states[0]['final_parameter_sha256'],states[1]['final_parameter_sha256'])

    def test_nonfinite_global_norm_cannot_step_or_publish_success(self):
        selection,config=prepare('norm-overflow-'+uuid.uuid4().hex[:6]);out=probe.CASE/('norm-overflow-out-'+uuid.uuid4().hex[:6])
        original=torch.nn.utils.clip_grad_norm_
        def overflow(parameters,*args,**kwargs):
            parameters=list(parameters)
            for p in parameters:p.grad.fill_(1e30)
            return original(parameters,*args,**kwargs)
        with mock.patch.object(torch.nn.utils,'clip_grad_norm_',new=overflow),mock.patch.object(torch.optim.Adam,'step') as step:
            with self.assertRaises(RuntimeError):mod.run_probe(selection,config,out,steps=1,size=32,threads=1,max_grad_norm=5)
            step.assert_not_called()
        self.assertFalse((out/'summary.json').exists());self.assertFalse((out/'small2/summary.json').exists())


if __name__=='__main__':unittest.main()
