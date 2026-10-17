"""Whole-pool exposure, saved shuffle state, and independent last-epoch replay."""
import importlib.util
import json
from pathlib import Path
import unittest
import uuid

import numpy as np
import torch
from torch.utils.data import DataLoader
from tests.recognition.test_rgb_fitting import fixture,probe

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('pool_tested',ROOT/'scripts/rgb/full_pool.py')
pool=importlib.util.module_from_spec(spec);spec.loader.exec_module(pool)


class FullPoolTests(unittest.TestCase):
    def test_partial_batches_and_saved_generator_reproduce_last_epoch(self):
        _,selection=fixture('pool-'+uuid.uuid4().hex[:8]);out=probe.CASE/('pool-out-'+uuid.uuid4().hex[:8])
        rng=torch.random.get_rng_state().clone();threads=torch.get_num_threads()
        s=pool.train(selection,out,epochs=2,size=32,batch_size=3,seed=18,threads=1)
        self.assertTrue(torch.equal(rng,torch.random.get_rng_state()));self.assertEqual(threads,torch.get_num_threads())
        self.assertEqual(s['optimizer_steps'],4);self.assertEqual(s['samples_seen'],8)
        self.assertTrue(all(n==2 for n in s['exposures_per_sample'].values()))
        self.assertEqual([r['batch_sizes'] for r in s['history'][1:]],[[3,1],[3,1]])
        self.assertEqual([a['epoch'] for a in s['snapshots']],[0,1,2])
        _,_,_,cache,_,_=pool.fit.verify_selection(selection,size=32)
        cp=torch.load(out/'epoch-01.pt',map_location='cpu',weights_only=True)
        model=pool.cohort.build_model(16);model.load_state_dict(cp['state_dict'],strict=True)
        opt=torch.optim.Adam(model.parameters(),lr=.001);opt.load_state_dict(cp['optimizer_state_dict'])
        gen=torch.Generator();gen.set_state(cp['shuffle_generator_state'])
        loader=DataLoader(pool.fit.CachedImages(cache),batch_size=3,shuffle=True,generator=gen)
        old=torch.get_num_threads();torch.set_num_threads(1);seen=[]
        try:
            for x,y,ids in loader:
                model.train();opt.zero_grad(set_to_none=True)
                w=torch.tensor(s['class_weights']);loss=torch.nn.functional.cross_entropy(model(x),y,weight=w,ignore_index=255,reduction='sum')/w[y[y!=255]].double().sum().item()
                loss.backward();total=torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(p.grad) for p in model.parameters()]))
                scale=torch.clamp(5/(total+1e-6),max=1.)
                for p in model.parameters():p.grad.mul_(scale)
                opt.step();seen.extend(ids)
        finally:torch.set_num_threads(old)
        self.assertEqual(seen,s['history'][-1]['sample_ids'])
        self.assertEqual(pool.fit.state_digest(model.state_dict()),s['final_parameter_sha256'])
        final=torch.load(out/'epoch-02.pt',map_location='cpu',weights_only=True)
        self.assertTrue(torch.equal(gen.get_state(),final['shuffle_generator_state']))
        for key,state in opt.state_dict()['state'].items():
            for name,value in state.items():self.assertTrue(torch.equal(value,final['optimizer_state_dict']['state'][key][name]))
        rows=pool.fit.read_json(out/'records.json');matrix=np.zeros((3,3),dtype=np.int64)
        byid={sid:target for sid,_,target in cache}
        from PIL import Image
        for row in rows:
            with Image.open(out/row['prediction_path']) as im:pred=np.array(im)
            matrix+=np.array([[int(((byid[row['sample_id']]==a)&(pred==b)).sum()) for b in range(3)] for a in range(3)])
        self.assertEqual(matrix.tolist(),s['history'][-1]['evaluation']['confusion_true_rows_predicted_columns'])
        with self.assertRaises(pool.fit.FittingError):pool.fit.predict(out/'epoch-02.pt',selection,'sample_0',probe.CASE/('pool-reject-'+uuid.uuid4().hex[:8]))
        with self.assertRaises(pool.fit.FittingError):pool.train(selection,out)

    def test_invalid_inputs_and_exclusion_changes_do_not_create_output(self):
        _,selection=fixture('pool-invalid-'+uuid.uuid4().hex[:8])
        for args in ({'epochs':True},{'epochs':0},{'epochs':21},{'size':33},{'batch_size':0},{'threads':9},{'seed':-1}):
            out=probe.CASE/('pool-bad-'+uuid.uuid4().hex[:8])
            with self.assertRaises(pool.fit.FittingError):pool.train(selection,out,**args)
            self.assertFalse(out.exists())
        data=json.loads(selection.read_text());data['records'][4]['disposition']='experiment_train';data['records'][4]['experimental_training_allowed']=True;probe.dump(selection,data)
        out=probe.CASE/('pool-alias-'+uuid.uuid4().hex[:8])
        with self.assertRaises(pool.fit.FittingError):pool.train(selection,out,size=32,threads=1)
        self.assertFalse(out.exists())


if __name__=='__main__':unittest.main()
