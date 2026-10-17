import importlib.util
from pathlib import Path
import unittest
import uuid
import sqlite3
import shutil

import numpy as np
from tests.recognition.test_rgb_fitting import fixture,probe

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('quality_tested',ROOT/'scripts/rgb/quality_review.py')
q=importlib.util.module_from_spec(spec);spec.loader.exec_module(q)


class QualityReviewTests(unittest.TestCase):
    def test_four_connectivity_single_pixels_and_all_pixel_accounting(self):
        a=np.array([[0,1,0],[1,0,0],[0,1,1]],dtype=np.uint8)
        rows=q.regions(a)
        self.assertEqual([r['area_pixels'] for r in rows],[1,3,1])
        self.assertEqual([(r['x_min'],r['y_min'],r['x_max_exclusive'],r['y_max_exclusive']) for r in rows],[(0,0,1,1),(1,0,3,2),(0,2,1,3)])
        self.assertEqual(q.regions(np.ones((2,2),dtype=np.uint8)),[])
        for bad in (np.array([[255]],dtype=np.uint8),np.zeros((2,2,2)),np.zeros((2,2),dtype=float)):
            with self.assertRaises(q.fit.FittingError):q.regions(bad)

    def test_real_artifact_binding_sql_and_unknown_decisions(self):
        _,selection=fixture('quality-'+uuid.uuid4().hex[:8]);folders=[]
        for seed in (20261017,20261018,20261019):
            folder=probe.CASE/('quality-pool-'+uuid.uuid4().hex[:8])
            q.pool.train(selection,folder,epochs=1,size=128,batch_size=3,seed=seed,threads=1);folders.append(folder)
        baseline=probe.CASE/('quality-baseline-'+uuid.uuid4().hex[:8]);baseline.mkdir();(baseline/'predictions').mkdir()
        shutil.copyfile(folders[0]/'records.json',baseline/'records.json')
        for p in (folders[0]/'predictions').glob('*.png'):shutil.copyfile(p,baseline/'predictions'/p.name)
        reference=q.fit.read_json(folders[0]/'summary.json')
        q.fit.write_json(baseline/'summary.json',{'kind':'whole_fitting_set_diagnostic','selection_sha256':q.fit.sha(selection),'model':reference['history'][-1]['evaluation']})
        out=probe.CASE/('quality-out-'+uuid.uuid4().hex[:8]);result=q.build(selection,folders,out,baseline=baseline)
        self.assertEqual(result['baseline_fitting_metrics'],reference['history'][-1]['evaluation'])
        self.assertEqual(result['sample_count'],4);self.assertEqual(result['review_count'],4)
        self.assertEqual(result['review_status'],'awaiting_human_label_target_review')
        self.assertFalse(result['semantic_reviewed']);self.assertIsNone(result['physical_area'])
        regions=q.fit.read_json(out/'regions.json')
        with sqlite3.connect(out/'regions.sqlite') as db:
            self.assertEqual(db.execute('SELECT count(*),COALESCE(sum(area_pixels),0) FROM region').fetchone(),(len(regions),sum(r['area_pixels'] for r in regions)))
            self.assertEqual(db.execute("SELECT value FROM metadata WHERE key='unit'").fetchone()[0],'diagnostic_grid_pixels')
            self.assertEqual(db.execute('SELECT count(*) FROM sample').fetchone()[0],4)
            self.assertEqual(db.execute("SELECT value FROM metadata WHERE key='checkpoint_sha256'").fetchone()[0],reference['snapshots'][-1]['sha256'])
        self.assertTrue(all(c['review_status']=='unknown' for c in q.fit.read_json(out/'case-index.json')))
        with self.assertRaises(q.fit.FittingError):q.build(selection,folders,out)
        first=q.fit.read_json(folders[0]/'records.json');(folders[0]/first[0]['prediction_path']).write_bytes(b'changed')
        rejected=probe.CASE/('quality-bad-'+uuid.uuid4().hex[:8])
        with self.assertRaises(q.fit.FittingError):q.build(selection,folders,rejected)
        self.assertFalse(rejected.exists())


if __name__=='__main__':unittest.main()
