"""Check loss tradeoffs, default compatibility, and saved-state replay after reweighting."""
import uuid
import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader
from tests.recognition.test_rgb_full_pool import pool
from tests.recognition.test_rgb_fitting import fixture,probe


class PoolReweightTests(unittest.TestCase):
    def test_ridge_only_weights_and_invalid_values(self):
        hist=np.array([1000,50,80])
        base=pool.fit.class_weights_from_counts(hist)
        self.assertTrue(np.array_equal(base,pool.training_weights(hist)))
        actual=pool.training_weights(hist,.5)
        self.assertEqual(actual[0],base[0]);self.assertEqual(actual[2],base[2]);self.assertEqual(actual[1],base[1]*.5)
        for bad in (True,False,0,-1,1.1,float('nan'),float('inf'),'0.5',None):
            with self.assertRaises(pool.fit.FittingError):pool.training_weights(hist,bad)
        # Labels remain fixed; reducing the rare-class weight changes the tradeoff,
        # rather than inserting a brightness threshold or a new target.
        logits=torch.tensor([[[[0.,0.]],[[0.,0.]],[[0.,0.]]]],requires_grad=True)
        target=torch.tensor([[[0,1]]])
        grads=[]
        for weights in (base,actual):
            n,d=pool.fit.weighted_loss_components(logits,target,torch.tensor(weights))
            grads.append(torch.autograd.grad(n/d,logits,retain_graph=True)[0])
        self.assertGreater(abs(grads[1][0,1,0,0]),abs(grads[0][0,1,0,0]))
        self.assertLess(abs(grads[1][0,1,0,1]),abs(grads[0][0,1,0,1]))

    def test_paired_initialization_shuffle_and_manual_replay(self):
        _,selection=fixture('weights-'+uuid.uuid4().hex[:8])
        a=probe.CASE/('weight-default-'+uuid.uuid4().hex[:8]);b=probe.CASE/('weight-half-'+uuid.uuid4().hex[:8])
        before=torch.random.get_rng_state().clone();threads=torch.get_num_threads()
        sa=pool.train(selection,a,epochs=2,size=32,batch_size=3,seed=19,threads=1)
        sb=pool.train(selection,b,epochs=2,size=32,batch_size=3,seed=19,threads=1,ridge_weight_scale=.5)
        self.assertTrue(torch.equal(before,torch.random.get_rng_state()));self.assertEqual(threads,torch.get_num_threads())
        self.assertEqual(sa['schema'],pool.SCHEMA);self.assertNotIn('ridge_weight_scale',sa)
        self.assertEqual(sb['schema'],pool.REWEIGHTED_SCHEMA);self.assertEqual(sb['ridge_weight_scale'],.5)
        self.assertEqual(sa['initial_parameter_sha256'],sb['initial_parameter_sha256'])
        for x,y in zip(sa['history'][1:],sb['history'][1:]):
            self.assertEqual(x['sample_ids'],y['sample_ids']);self.assertEqual(x['batch_sizes'],y['batch_sizes'])
        _,_,_,cache,_,_=pool.fit.verify_selection(selection,size=32)
        cp=torch.load(b/'epoch-01.pt',map_location='cpu',weights_only=True)
        model=pool.cohort.build_model(16);model.load_state_dict(cp['state_dict'],strict=True)
        opt=torch.optim.Adam(model.parameters(),lr=.001);opt.load_state_dict(cp['optimizer_state_dict'])
        gen=torch.Generator();gen.set_state(cp['shuffle_generator_state'])
        loader=DataLoader(pool.fit.CachedImages(cache),batch_size=3,shuffle=True,generator=gen)
        torch.set_num_threads(1)
        try:
            for x,y,ids in loader:
                model.train();opt.zero_grad(set_to_none=True);w=torch.tensor(sb['class_weights'])
                loss=torch.nn.functional.cross_entropy(model(x),y,weight=w,ignore_index=255,reduction='sum')/w[y[y!=255]].double().sum().item()
                loss.backward();norm=torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(p.grad) for p in model.parameters()]))
                scale=torch.clamp(5/(norm+1e-6),max=1.)
                for p in model.parameters():p.grad.mul_(scale)
                opt.step()
        finally:torch.set_num_threads(threads)
        final=torch.load(b/'epoch-02.pt',map_location='cpu',weights_only=True)
        self.assertEqual(pool.fit.state_digest(model.state_dict()),sb['final_parameter_sha256'])
        self.assertTrue(torch.equal(gen.get_state(),final['shuffle_generator_state']))
        for key,state in opt.state_dict()['state'].items():
            for name,v in state.items():self.assertTrue(torch.equal(v,final['optimizer_state_dict']['state'][key][name]))
        with self.assertRaises(pool.fit.FittingError):pool.fit.predict(b/'epoch-02.pt',selection,'sample_0',probe.CASE/('reject-half-'+uuid.uuid4().hex[:8]))
        with self.assertRaises(pool.diag.DiagnosticError):pool.diag.load_model(b/'epoch-02.pt',selection)
        for value in (0,True,float('nan')):
            out=probe.CASE/('bad-half-'+uuid.uuid4().hex[:8])
            with self.assertRaises(pool.fit.FittingError):pool.train(selection,out,ridge_weight_scale=value)
            self.assertFalse(out.exists())


if __name__=='__main__':unittest.main()
