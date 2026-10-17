"""Instrumentation must preserve the original optimizer trajectory."""
import importlib.util
import json
from pathlib import Path
import unittest
import uuid

import numpy as np
import torch

from tests.recognition.test_rgb_cohort_probe import prepare, mod, probe

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('trace_tested',ROOT/'scripts/rgb/trace_probe.py')
trace = importlib.util.module_from_spec(spec);spec.loader.exec_module(trace)


def inputs():
    selection,config = prepare('trace-'+uuid.uuid4().hex[:8])
    data = mod.fit.read_json(config);data['cohorts'] = data['cohorts'][:1];probe.dump(config,data)
    reference = probe.CASE/('trace-ref-'+uuid.uuid4().hex[:8])
    mod.run_probe(selection,config,reference,steps=3,size=32,seed=18,threads=1,base_width=16)
    return selection,config,reference/'small2/summary.json'


class TraceProbeTests(unittest.TestCase):
    def test_observation_preserves_trajectory_and_adam_update(self):
        selection,config,reference = inputs();out = probe.CASE/('trace-out-'+uuid.uuid4().hex[:8])
        rng = torch.random.get_rng_state().clone();threads = torch.get_num_threads()
        result = trace.run_trace(selection,config,reference,out,steps=3,size=32,seed=18,threads=1)
        self.assertTrue(torch.equal(rng,torch.random.get_rng_state()));self.assertEqual(threads,torch.get_num_threads())
        original = mod.fit.read_json(reference)
        self.assertEqual(result['history'],original['history'])
        self.assertEqual(result['final_parameter_sha256'],original['final_parameter_sha256'])
        self.assertEqual(result['snapshot_schedule'],[0,3]);self.assertTrue(result['not_candidate_model'])
        rows = [json.loads(line) for line in (out/'trace.jsonl').read_text().splitlines()]
        self.assertEqual([r['step'] for r in rows],[0,1,2,3])
        state = torch.load(out/'step-0000.pt',weights_only=True,map_location='cpu')
        model = mod.build_model(16);model.load_state_dict(state['state_dict'],strict=True)
        optimizer = torch.optim.Adam(model.parameters(),lr=.001);optimizer.load_state_dict(state['optimizer_state_dict'])
        _,_,_,cache,_,_ = mod.fit.verify_selection(selection,size=32);byid = {sid:(rgb,y) for sid,rgb,y in cache}
        x = torch.stack([torch.from_numpy(byid[sid][0].copy()).permute(2,0,1).float()/255 for sid in state['sample_ids']])
        y = torch.stack([torch.from_numpy(byid[sid][1].astype(np.int64)) for sid in state['sample_ids']])
        old_threads = torch.get_num_threads();torch.set_num_threads(1)
        try:
            weights = torch.tensor(state['class_weights']);valid = y != 255
            loss = torch.nn.functional.cross_entropy(model(x),y,weight=weights,ignore_index=255,reduction='sum')/weights[y[valid]].double().sum().item()
            loss.backward();self.assertEqual(loss.item(),rows[1]['loss_before_update'])
            before = {n:p.detach().numpy().copy() for n,p in model.named_parameters()}
            grad_square = 0
            for name,p in model.named_parameters():
                square = float(np.sum(p.grad.numpy().astype(np.float64)**2));grad_square += square
                self.assertAlmostEqual(square**.5,rows[1]['parameters'][name]['gradient_l2'],places=12)
            self.assertAlmostEqual(grad_square**.5,rows[1]['gradient_l2'],places=12)
            optimizer.step();movement_square = 0
            for name,p in model.named_parameters():
                movement_square += float(np.sum((p.detach().numpy()-before[name]).astype(np.float64)**2))
            self.assertAlmostEqual(movement_square**.5,rows[1]['update_l2'],places=12)
        finally:
            torch.set_num_threads(old_threads)
        final = torch.load(out/'step-0003.pt',weights_only=True,map_location='cpu')
        self.assertTrue(all(s['step'].item()==3 for s in final['optimizer_state_dict']['state'].values()))
        with self.assertRaises(mod.fit.FittingError):
            mod.fit.predict(out/'step-0003.pt',selection,'sample_0',probe.CASE/('trace-candidate-'+uuid.uuid4().hex[:8]))
        with self.assertRaises(trace.TraceError):trace.run_trace(selection,config,reference,out)

    def test_invalid_scope_and_arguments_fail_before_output(self):
        selection,config,reference = inputs()
        for kwargs in ({'steps':True},{'steps':501},{'size':33},{'seed':-1},{'threads':0},{'seed':19}):
            out = probe.CASE/('trace-bad-'+uuid.uuid4().hex[:8])
            args = {'steps':3,'size':32,'seed':18,'threads':1}|kwargs
            with self.subTest(kwargs=kwargs),self.assertRaises(trace.TraceError):
                trace.run_trace(selection,config,reference,out,**args)
            self.assertFalse(out.exists())
        original = mod.fit.read_json(config);original['cohorts'][0]['sample_ids'] = ['sample_4'];probe.dump(config,original)
        out = probe.CASE/('trace-excluded-'+uuid.uuid4().hex[:8])
        with self.assertRaises(trace.TraceError):trace.run_trace(selection,config,reference,out,steps=3,size=32,seed=18,threads=1)
        self.assertFalse(out.exists())

    def test_reference_mismatch_never_publishes_success(self):
        selection,config,reference = inputs();data = mod.fit.read_json(reference)
        data['final_parameter_sha256'] = '0'*64;probe.dump(reference,data)
        out = probe.CASE/('trace-drift-'+uuid.uuid4().hex[:8])
        with self.assertRaises(trace.TraceError):trace.run_trace(selection,config,reference,out,steps=3,size=32,seed=18,threads=1)
        self.assertTrue((out/'trace.jsonl').exists());self.assertFalse((out/'summary.json').exists())


if __name__ == '__main__':unittest.main()
