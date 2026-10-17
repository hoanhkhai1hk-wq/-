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
                       *[{'learning_rate':value} for value in invalid_lr]]:
            out=probe.CASE/('invalid-cohort-'+uuid.uuid4().hex[:6])
            with self.subTest(kwargs=kwargs),self.assertRaises(mod.ProbeError):mod.run_probe(selection,config,out,**kwargs)
            self.assertFalse(out.exists())

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


if __name__=='__main__':unittest.main()
