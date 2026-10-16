"""Hand-counted metrics and synthetic integration; no real data required."""
import importlib.util
from pathlib import Path
import unittest
from unittest import mock
import uuid
import numpy as np
import torch
from PIL import Image
from tests.recognition.test_rgb_fitting import fixture, probe

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('diagnostic_tested',ROOT/'scripts/rgb/diagnose.py')
diag=importlib.util.module_from_spec(spec);spec.loader.exec_module(diag)


class DiagnosticTests(unittest.TestCase):
    def test_hand_count_orientation_ignore_and_class_zero(self):
        target=np.array([0,0,1,1,2,255]);pred=np.array([0,1,0,1,2,2])
        np.testing.assert_array_equal(diag.confusion(target,pred),[[1,1,0],[1,1,0],[0,0,1]])
        m=diag.measures(np.array([[8,2,0],[3,1,1],[0,1,4]]))
        self.assertEqual(m['valid_pixels'],20);self.assertEqual(m['fitting_pixel_accuracy'],.65)
        self.assertEqual(m['classes']['1']['precision'],.25);self.assertEqual(m['classes']['1']['recall'],.2)
        self.assertEqual(m['classes']['1']['iou'],.125)

    def test_absent_classes_have_null_denominators(self):
        m=diag.measures(np.zeros((3,3),dtype=np.int64))
        self.assertIsNone(m['mean_iou_over_nonzero_unions']);self.assertIsNone(m['fitting_pixel_accuracy'])
        for c in m['classes'].values():
            self.assertIsNone(c['recall']);self.assertIsNone(c['precision']);self.assertIsNone(c['iou'])
        m=diag.measures(diag.confusion(np.array([0,1,2]),np.zeros(3,dtype=int)))
        self.assertEqual(m['classes']['1']['recall'],0.);self.assertIsNone(m['classes']['1']['precision'])

    def test_illegal_values_and_types(self):
        for target,pred in (([3],[0]),([0],[255]),([0.],[0]),([True],[0]),([0,1],[0])):
            with self.subTest(target=target,pred=pred),self.assertRaises(diag.DiagnosticError):diag.confusion(target,pred)
        for matrix in (np.zeros((2,2),dtype=int),np.ones((3,3)),-np.ones((3,3),dtype=int)):
            with self.assertRaises(diag.DiagnosticError):diag.measures(matrix)

    def test_full_diagnostic_has_no_training_and_independent_png_counts(self):
        _,selection=fixture('diagnose-'+uuid.uuid4().hex[:6]);out=probe.CASE/('diagnosis-'+uuid.uuid4().hex[:6])
        cpdir=probe.CASE/('cp-'+uuid.uuid4().hex[:6]);cpdir.mkdir()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(17);model=diag.fit.SmallUNet()
        payload={'schema':'agrolake_rgb_internal_unet_v1','state_dict':model.state_dict(),'size':32,'selection_sha256':diag.fit.sha(selection),
            'seed':17,'class_ids':diag.fit.CLASSES,'class_weights':[1.,1.,1.],'trained_on_real_data':True,'semantic_reviewed':False,'independent_test_available':False}
        torch.save(payload,cpdir/'checkpoint.pt')
        rng=torch.random.get_rng_state().clone();threads=torch.get_num_threads()
        with mock.patch.object(torch.optim.Adam,'step',side_effect=AssertionError('training forbidden')),mock.patch.object(torch.Tensor,'backward',side_effect=AssertionError('backward forbidden')):
            summary=diag.diagnose(cpdir/'checkpoint.pt',selection,out,batch_size=3,threads=1)
        self.assertEqual(summary['sample_count'],4);self.assertTrue(torch.equal(rng,torch.random.get_rng_state()));self.assertEqual(threads,torch.get_num_threads())
        rows=diag.fit.read_json(out/'records.json');total=np.zeros((3,3),dtype=np.int64)
        _,_,_,cache,_,_=diag.fit.verify_selection(selection,size=32);targets={sid:target for sid,_,target in cache}
        for row in rows:
            with Image.open(out/row['prediction_path']) as im:pred=np.array(im);self.assertEqual(im.mode,'L')
            expected=np.zeros((3,3),dtype=np.int64)
            for true,guess in zip(targets[row['sample_id']].ravel(),pred.ravel()):
                if true!=255:expected[true,guess]+=1
            np.testing.assert_array_equal(expected,row['confusion_true_rows_predicted_columns']);total+=expected
        np.testing.assert_array_equal(total,summary['model']['confusion_true_rows_predicted_columns'])
        self.assertFalse(summary['independent_test_available']);self.assertEqual(summary['evaluation_scope'],'fitting_samples_only')
        with self.assertRaises(diag.DiagnosticError):diag.diagnose(cpdir/'checkpoint.pt',selection,out)
        bad=diag.fit.read_json(selection);bad['records'].pop();probe.dump(selection,bad)
        other=probe.CASE/('bad-diagnostic-'+uuid.uuid4().hex[:6])
        with self.assertRaises(diag.DiagnosticError):diag.diagnose(cpdir/'checkpoint.pt',selection,other)
        self.assertFalse(other.exists())


if __name__=='__main__':unittest.main()
