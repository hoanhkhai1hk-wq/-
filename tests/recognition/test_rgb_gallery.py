"""Retained synthetic probes only; gallery generation must not run inference."""
import importlib.util
import json
from pathlib import Path
import shutil
import unittest
from unittest import mock
import uuid
import numpy as np
from PIL import Image
from tests.recognition.test_rgb_cohort_probe import prepare,mod,probe

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('tested_gallery',ROOT/'scripts/rgb/gallery.py')
gallery=importlib.util.module_from_spec(spec);spec.loader.exec_module(gallery)


class GalleryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.selection,cohorts=prepare('gallery-fixture-'+uuid.uuid4().hex[:6]);cls.sources=[]
        for width in (8,16):
            out=probe.CASE/('gallery-source-'+uuid.uuid4().hex[:6])
            mod.run_probe(cls.selection,cohorts,out,steps=1,size=32,seed=17,threads=1,base_width=width)
            cls.sources.append(out/'small2')
        cls.config={'schema_version':1,'selection':cls.selection.relative_to(ROOT).as_posix(),
            'selection_sha256':mod.fit.sha(cls.selection),'steps':1,'size':32,
            'pairs':[{'seed':17,'reference_summary':(cls.sources[0]/'summary.json').relative_to(ROOT).as_posix(),
            'experiment_summary':(cls.sources[1]/'summary.json').relative_to(ROOT).as_posix()}]}

    def config_file(self,value):
        folder=probe.CASE/('gallery-config-'+uuid.uuid4().hex[:6]);folder.mkdir();path=folder/'config.json'
        mod.fit.write_json(path,value);return path

    def test_gallery_source_pixels_metrics_and_safe_inline_data_without_inference(self):
        cfg=json.loads(json.dumps(self.config));cfg['title']='</script><img src=x onerror=alert(1)>'
        config=self.config_file(cfg);out=probe.CASE/('gallery-out-'+uuid.uuid4().hex[:6])
        with mock.patch.object(gallery.cohort.fit.SmallUNet,'forward',side_effect=AssertionError('Gallery ran inference')):
            result=gallery.generate(config,out)
        self.assertFalse(result['inference_performed']);self.assertFalse(result['independent_test_available'])
        self.assertEqual(len(list(out.rglob('*.png'))),8)
        text=(out/'index.html').read_text(encoding='utf-8');self.assertNotIn(cfg['title'],text);self.assertIn('\\u003c/script\\u003e',text)
        self.assertNotIn('fetch(',text);self.assertNotIn('https://',text);self.assertIn("connect-src 'none'",text)
        data=mod.fit.read_json(out/'gallery.json');self.assertEqual(data['sample_count'],2);self.assertEqual(data['title'],cfg['title'])
        for name,digest in result['artifacts'].items():self.assertEqual(mod.fit.sha(out/name),digest)
        for source,tag in zip(self.sources,('reference','experiment')):
            s=mod.fit.read_json(source/'summary.json')
            for a,row in zip(s['artifacts'],s['history'][-1]['per_image']):
                sid=a['sample_id'];view=data['views']['17']['samples'][sid]
                self.assertEqual(view[tag]['classes'],row['classes'])
                with Image.open(source/a['comparison_path']) as im:panel=np.array(im)
                with Image.open(out/view[tag+'_image']) as im:np.testing.assert_array_equal(np.array(im),panel[32:,128:192])
        with self.assertRaises(gallery.GalleryError):gallery.generate(config,out)

    def test_tampered_artifact_scope_or_metric_rejected_before_output(self):
        for kind in ('prediction','scope','matrix','path','ineligible'):
            clone=probe.CASE/('gallery-tamper-'+uuid.uuid4().hex[:6]);shutil.copytree(self.sources[1],clone)
            summary=mod.fit.read_json(clone/'summary.json')
            if kind=='prediction':
                p=clone/summary['artifacts'][0]['prediction_path'];p.write_bytes(p.read_bytes()+b'changed')
            elif kind=='scope':summary['semantic_reviewed']=True
            elif kind=='path':summary['artifacts'][0]['prediction_path']='../escape.png'
            elif kind=='ineligible':summary['sample_ids'][0]='sample_4'
            else:
                row=summary['history'][-1]['per_image'][0];cm=np.array(row['confusion_true_rows_predicted_columns'])
                cm[0,0]+=1;row.update(gallery.cohort.diag.measures(cm))
            mod.fit.write_json(clone/'summary.json',summary)
            cfg=json.loads(json.dumps(self.config));cfg['pairs'][0]['experiment_summary']=(clone/'summary.json').relative_to(ROOT).as_posix()
            config=self.config_file(cfg);out=probe.CASE/('gallery-bad-'+uuid.uuid4().hex[:6])
            with self.subTest(kind=kind),self.assertRaises(gallery.GalleryError):gallery.generate(config,out)
            self.assertFalse(out.exists())

    def test_bad_pair_selection_and_order_rejected(self):
        cases=[[]]
        cfg=json.loads(json.dumps(self.config));cfg['selection_sha256']='0'*64;cases.append(cfg)
        cfg=json.loads(json.dumps(self.config));cfg['pairs']*=2;cases.append(cfg)
        cfg=json.loads(json.dumps(self.config));cfg['pairs'][0]['seed']=18;cases.append(cfg)
        clone=probe.CASE/('gallery-order-'+uuid.uuid4().hex[:6]);shutil.copytree(self.sources[1],clone)
        s=mod.fit.read_json(clone/'summary.json');s['sample_ids']=list(reversed(s['sample_ids']));mod.fit.write_json(clone/'summary.json',s)
        cfg=json.loads(json.dumps(self.config));cfg['pairs'][0]['experiment_summary']=(clone/'summary.json').relative_to(ROOT).as_posix();cases.append(cfg)
        for cfg in cases:
            out=probe.CASE/('gallery-invalid-'+uuid.uuid4().hex[:6])
            with self.assertRaises(gallery.GalleryError):gallery.generate(self.config_file(cfg),out)
            self.assertFalse(out.exists())


if __name__=='__main__':unittest.main()
