"""用实际服务生成的合成结果独立核验对照客户端，不代表农业真值。"""
import copy,json,sys,unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(Path(__file__).parent)]
import test_codex_regressions as fixture
from agrolake_service import corrections
from agrolake_web.client import RecognitionClient,ApiError

class IntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.case=fixture.CorrectionRegressions('test_equal_pixel_totals_do_not_verify_pixel_alignment')
        cls.case.setUp()
        row=cls.case.submit();cls.case.wait(row['child_job_id'])
        cls.detail=corrections.get_correction(cls.case.manager,row['correction_id'])
        cls.client=RecognitionClient('http://127.0.0.1:1')
    @classmethod
    def tearDownClass(cls):cls.case.tearDown()
    def check(self,edit):
        value=copy.deepcopy(self.detail);edit(value)
        with patch.object(self.client,'_request_json',return_value=value):
            with self.assertRaises(ApiError) as raised:self.client.get_correction(value['correction']['correction_id'])
        self.assertEqual(raised.exception.category,'protocol')
    def test_actual_response_accepted(self):
        with patch.object(self.client,'_request_json',return_value=self.detail):
            self.assertEqual(self.client.get_correction(self.detail['correction']['correction_id']),self.detail)
    def test_snapshot_matches_parent(self):
        self.check(lambda d:d['snapshot']['parent_facts'].__setitem__('input_sha256','b'*64))
    def test_child_matches_revised_input(self):
        self.check(lambda d:d['child'].__setitem__('input_sha256','b'*64))
    def test_missing_artifact_cannot_be_complete(self):
        self.check(lambda d:d['child']['artifacts'].pop())
    def test_total_matches_classes(self):
        self.check(lambda d:d['comparison']['class_pixel_comparison']['child'].__setitem__('total',99999))
    def test_delta_matches_counts(self):
        self.check(lambda d:d['comparison']['class_pixel_comparison']['delta'].__setitem__('1',999))
    def test_nodata_denominator_matches_counts(self):
        self.check(lambda d:d['comparison']['nodata']['child'].__setitem__('denominator',99999))
    def test_nodata_ratio_matches_counts(self):
        self.check(lambda d:d['comparison']['nodata']['child'].__setitem__('ratio',.9))
    def test_problem_opinion_facts_match(self):
        self.check(lambda d:d['snapshot']['problem_review'].__setitem__('input_sha256','b'*64))
    def test_comparison_opinion_matches_snapshot(self):
        self.check(lambda d:d['comparison']['parent_problem_review'].__setitem__('decision','accepted'))
    def test_synthetic_flag_cannot_disappear(self):
        self.check(lambda d:d['correction'].__setitem__('is_synthetic_case',False))
    def test_missing_nullable_revised_field_is_protocol_error(self):
        self.check(lambda d:d['snapshot']['revised'].pop('source_link'))
    def test_missing_nullable_child_opinion_is_protocol_error(self):
        self.check(lambda d:d['child'].pop('latest_task_review'))
    def test_artifact_sha_must_be_a_string(self):
        self.check(lambda d:d['child']['artifacts'][0].__setitem__('sha256',int('1'*64)))

if __name__=='__main__':
    out=fixture.OUT/'client-integrity';out.mkdir()
    with (out/'tests.log').open('x',encoding='utf-8') as log:
        result=unittest.TextTestRunner(stream=log,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(IntegrityTests))
    summary={'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'successful':result.wasSuccessful(),'directory':str(out)}
    (out/'result.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary));print((out/'tests.log').read_text(encoding='utf-8'))
    raise SystemExit(0 if result.wasSuccessful() else 1)
