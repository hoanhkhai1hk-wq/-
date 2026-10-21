"""Acceptance contract tests: scope, evidence traceability, and no label adjudication."""
import importlib.util
import json
import unittest
from collections import Counter
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


class AcceptanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scope = load(OUT / 'development-scope.json')
        cls.examples = load(OUT / 'examples-manifest.json')
        cls.register = load(OUT / 'ambiguity-register.json')
        cls.allowed = {r['sample_id']: r for group in cls.scope['allowed'].values() for r in group}
        spec = importlib.util.spec_from_file_location('prepare_spec_review', OUT / 'prepare_review.py')
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    def test_developer_scope_has_exact_counts_and_excludes_frozen(self):
        self.assertEqual(len(self.scope['allowed']['new_dev313']), 313)
        self.assertEqual(len(self.scope['allowed']['old_dev734']), 734)
        self.assertEqual(len(self.allowed), 1047)
        self.assertFalse(set(self.allowed) & set(self.scope['frozen_ids_excluded']))

    def test_reader_refuses_frozen_before_file_access(self):
        fake = {'sample_id': self.scope['frozen_ids_excluded'][0], 'image_path': 'must-not-open'}
        with self.assertRaisesRegex(ValueError, 'outside developer-only'):
            self.module.read_developer(fake, self.scope)

    def test_reader_refuses_path_swap_under_valid_id(self):
        rec = dict(self.scope['allowed']['new_dev313'][0])
        rec['image_path'] = 'must-not-open'
        with self.assertRaises(AssertionError):
            self.module.read_developer(rec, self.scope)

    def test_five_scenarios_each_have_two_positive_one_confusable(self):
        counts = Counter((e['case'], e['role']) for e in self.examples['examples'])
        self.assertEqual({e['case'] for e in self.examples['examples']}, {'C1', 'C2', 'C3', 'C4', 'C5'})
        for case in ['C1', 'C2', 'C3', 'C4', 'C5']:
            self.assertEqual(counts[(case, 'positive')], 2)
            self.assertEqual(counts[(case, 'confusable')], 1)
            ids = [e['sample_id'] for e in self.examples['examples'] if e['case'] == case and e['role'] == 'positive']
            self.assertEqual(len(set(ids)), 2)

    def test_examples_source_and_sha_match_developer_manifest(self):
        for e in self.examples['examples']:
            self.assertIn(e['sample_id'], self.allowed)
            self.assertNotIn(e['sample_id'], self.scope['frozen_ids_excluded'])
            rec = self.allowed[e['sample_id']]
            self.assertEqual(e['image_path'], rec['image_path'])
            self.assertEqual(e['image_sha256'], rec['image_sha256'])
            self.assertEqual(e['existing_json_sha256'], rec['json_sha256'])
            self.assertFalse(e['label_written'])
            self.assertFalse(e['wide_passable_dirt_road_positive_claimed'])
            self.assertEqual(self.module.digest(ROOT / e['image_path']), e['image_sha256'])
            self.assertEqual(self.module.digest(ROOT / e['existing_json_path']), e['existing_json_sha256'])

    def test_crops_are_valid_and_gallery_files_decode(self):
        for e in self.examples['examples']:
            x0, y0, x1, y1 = e['roi_xyxy']
            self.assertTrue(0 <= x0 < x1 <= 512 and 0 <= y0 < y1 <= 512)
        for gallery in {e['gallery'] for e in self.examples['examples']}:
            with Image.open(OUT / gallery) as im:
                im.load()
                self.assertEqual(im.size, (1350, 1050))

    def test_soil_route_branch_gap_is_not_hidden(self):
        self.assertIn('narrow-soil-strip branch only', self.examples['coverage_gap'])
        positives = [e for e in self.examples['examples'] if e['case'] == 'C4' and e['role'] == 'positive']
        self.assertTrue(all('窄土带分支' in e['suggested_v1_meaning'] for e in positives))
        document = (ROOT / 'docs/RGB标签验收规范.md').read_text(encoding='utf-8')
        self.assertIn('宽土路道路分支暂无可证正例', document)
        self.assertIn('不追溯改判', document)
        self.assertIn('本项目约定，不是行业标准', document)

    def test_register_counts_are_unique_and_all_pending(self):
        cases = self.register['cases']
        self.assertEqual(len(cases), 69)
        self.assertEqual(len({c['review_id'] for c in cases}), 69)
        self.assertEqual(len({c['sample_id'] for c in cases}), 69)
        new_ids = {r['sample_id'] for r in self.scope['allowed']['new_dev313']}
        for case in cases:
            self.assertIn(case['sample_id'], new_ids)
            self.assertEqual(case['status'], '待审')
            self.assertFalse(case['existing_label_changed'])
            self.assertTrue(case['question'] and case['suggested_decision'])

    def test_every_developer_image_has_exactly_one_first_pass_record(self):
        coverage = self.register['coverage']
        expected = {r['sample_id'] for r in self.scope['allowed']['new_dev313']}
        self.assertEqual(len(coverage), 313)
        self.assertEqual({r['sample_id'] for r in coverage}, expected)
        self.assertTrue(all(r['first_pass_complete'] for r in coverage))
        self.assertEqual(sum(r['screening_result'] == '疑义待审' for r in coverage), 69)
        self.assertEqual(self.register['without_registered_question'], 244)

    def test_markdown_register_has_all_rows_and_pending_status(self):
        text = (ROOT / 'outputs/标签疑义待审登记-20261022.md').read_text(encoding='utf-8')
        rows = [line for line in text.splitlines() if line.startswith('| LS-V1-')]
        self.assertEqual(len(rows), 69)
        self.assertTrue(all('| 待审 |' in row for row in rows))
        for case in self.register['cases']:
            self.assertTrue(any(case['review_id'] in row and Path(case['image_path']).name in row for row in rows))


if __name__ == '__main__':
    unittest.main(verbosity=2)
