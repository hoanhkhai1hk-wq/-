"""Independent integrity regressions: offline synthetic jobs, isolated stores only."""
from __future__ import annotations
import json
import sqlite3
import sys
import unittest
from contextlib import contextmanager
from pathlib import Path

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from test_rgb_governance import GovernBase, rg, rf

class IntegrityTest(GovernBase):
    @contextmanager
    def rgb_write(self):
        c=sqlite3.connect(self.rgb_dir/'runs.sqlite3')
        try:
            yield c
            c.commit()
        finally:c.close()

    def assert_no_statistics(self,bundle):
        self.assertNotIn('native_class_counts',bundle.snapshot)
        self.assertNotIn('candidate_count',bundle.snapshot)
        self.assertNotIn('三类像素统计',bundle.text)
        self.assertTrue(bundle.gaps)

    def test_sql_count_mismatch_does_not_publish_success_statistics(self):
        with self.rgb_write() as c:
            c.execute('UPDATE class_counts SET pixels=pixels+1 WHERE job_id=? AND class_id=0',(self._job['id'],))
        bundle=rf.build(self.conn,self._job['id'],data_dir=self.rgb_dir)
        self.assert_no_statistics(bundle)
        self.assertFalse(bundle.snapshot.get('sql_counts_verified'))

    def test_sql_region_mismatch_does_not_publish_success_statistics(self):
        with self.rgb_write() as c:
            c.execute('UPDATE regions SET bbox_x1=bbox_x1-1 WHERE job_id=?',(self._job['id'],))
        self.assert_no_statistics(rf.build(self.conn,self._job['id'],data_dir=self.rgb_dir))

    def test_missing_saved_manifest_is_a_gap_not_an_exception(self):
        p=self.rgb_dir/'jobs'/self._job['id']/'model-manifest.json'
        p.rename(p.with_suffix('.saved'))
        self.assert_no_statistics(rf.build(self.conn,self._job['id'],data_dir=self.rgb_dir))

    def test_invalid_job_path_is_not_copied_to_snapshot(self):
        value='C:/private-personal/farm.tif'
        b=rf.build(self.conn,value,data_dir=self.rgb_dir)
        self.assertNotIn(value,json.dumps(b.as_dict(),ensure_ascii=False))

    def test_source_name_path_is_not_sent_to_the_model(self):
        with self.rgb_write() as c:
            c.execute('UPDATE runs SET source_name=? WHERE job_id=?',('C:/private-personal/farm.tif',self._job['id']))
        b=rf.build(self.conn,self._job['id'],data_dir=self.rgb_dir)
        self.assertNotIn('private-personal',json.dumps(b.as_dict(),ensure_ascii=False))

    def test_wrong_version_is_not_an_intact_link(self):
        reg=self.register()['registration']
        self.conn.execute('UPDATE rgb_governance_links SET image_version_id=? WHERE job_id=?',(reg['stats_version_id'],self._job['id']))
        self.conn.commit()
        self.assertFalse(rg.link_status(self.conn,self._job['id'],backend=self.jobs_ro)['intact'])
        with self.assertRaises(rg.RgbGovernanceError):self.register()

    def test_wrong_version_object_is_not_an_intact_link(self):
        reg=self.register()['registration']
        self.conn.execute('UPDATE asset_versions SET object_id=? WHERE version_id=?',(reg['json_sha256'],reg['image_version_id']))
        self.conn.commit()
        self.assertFalse(rg.link_status(self.conn,self._job['id'],backend=self.jobs_ro)['intact'])

    def test_wrong_asset_id_is_not_an_intact_link(self):
        reg=self.register()['registration']
        self.conn.execute('UPDATE rgb_governance_links SET image_asset_id=? WHERE job_id=?',(reg['stats_asset_id'],self._job['id']))
        self.conn.commit()
        self.assertFalse(rg.link_status(self.conn,self._job['id'],backend=self.jobs_ro)['intact'])

    def test_object_path_escape_is_rejected_even_with_matching_bytes(self):
        reg=self.register()['registration']
        obj=self.conn.execute('SELECT * FROM objects WHERE object_id=?',(reg['png_sha256'],)).fetchone()
        from agrolake import config
        outside=Path(self.tmp.name)/'outside-store.png'
        outside.write_bytes((config.storage_root()/obj['relpath']).read_bytes())
        self.conn.execute('UPDATE objects SET relpath=? WHERE object_id=?',(str(outside),reg['png_sha256']))
        self.conn.commit()
        self.assertFalse(rg.link_status(self.conn,self._job['id'],backend=self.jobs_ro)['intact'])

    def test_broken_registered_object_is_not_reported_as_verified(self):
        reg=self.register()['registration']
        obj=self.conn.execute('SELECT * FROM objects WHERE object_id=?',(reg['json_sha256'],)).fetchone()
        from agrolake import config
        p=config.storage_root()/obj['relpath'];p.write_bytes(p.read_bytes()+b' ')
        bundle=rf.build(self.conn,self._job['id'],data_dir=self.rgb_dir)
        info=bundle.snapshot['registration']
        self.assertFalse(info['intact'])
        self.assertIn('核验未通过',bundle.text)

    def test_isolated_asset_is_shown_as_isolated(self):
        reg=self.register()['registration']
        self.conn.execute("UPDATE asset_versions SET status='isolated' WHERE version_id=?",(reg['image_version_id'],))
        self.conn.commit()
        bundle=rf.build(self.conn,self._job['id'],data_dir=self.rgb_dir)
        self.assertEqual(bundle.snapshot['registration']['image_version_status'],'isolated')

    def test_invalid_registration_path_is_not_copied_to_facts(self):
        self.register()
        self.conn.execute('UPDATE rgb_governance_links SET scene_id=? WHERE job_id=?',
                          ('C:/private-personal/farm.tif',self._job['id']))
        self.conn.commit()
        b=rf.build(self.conn,self._job['id'],data_dir=self.rgb_dir)
        self.assertFalse(b.snapshot['registration']['intact'])
        self.assertNotIn('private-personal',json.dumps(b.as_dict(),ensure_ascii=False))

    def test_general_scene_does_not_call_rgb_stats_human_review_or_missing_recognition(self):
        from agrolake import qa_facts
        reg=self.register()['registration']
        b=qa_facts.build(self.conn,'scene',scene_id=reg['scene_id'])
        self.assertEqual(b.snapshot['rgb_job_ids'],[self._job['id']])
        self.assertIn('存在RGB登记记录',b.text)
        self.assertIn('不代表已有真人复核',b.text)
        self.assertNotIn('该场景尚无识别结果落库',b.text)

    def test_corrupt_artifact_index_is_a_gap(self):
        with self.rgb_write() as c:
            c.execute('UPDATE runs SET artifact_sha256=? WHERE job_id=?',('{broken',self._job['id']))
        b=rf.build(self.conn,self._job['id'],data_dir=self.rgb_dir)
        self.assert_no_statistics(b)
        self.assertFalse(rf.list_jobs(data_dir=self.rgb_dir)['available'])

if __name__=='__main__': unittest.main(verbosity=2)
