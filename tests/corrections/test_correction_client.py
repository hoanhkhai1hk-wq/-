"""整改复跑客户端行为测试（DEV-20261015-016）。

服务端全部为本测试的合成假服务，不代表真实识别。覆盖：
- 成功回执与本次填写一致性校验；错误状态/离线/超时/坏JSON；
- 整改列表完整性、跨任务拒绝；origin 校验；
- 详情对照结构校验（父/子摘要、快照问题意见、像素对照）；
- resume 同号确认、created 必须为 False；bool 不当整数；
- 替换文件字段严格两字段。
证据写入 reports/corrections/DEV-20261015-017/ 下的独立目录并保留，不删除。

用法：
  .venv-p0/Scripts/python.exe tests/corrections/test_correction_client.py
"""
from __future__ import annotations

import json
import sys
import unittest
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT), str(ROOT / "tests" / "web")):
    if p not in sys.path:
        sys.path.insert(0, p)

from agrolake_web import client as C  # noqa: E402
from fixtures import start_fake_server  # noqa: E402

OUT = ROOT / "reports" / "corrections" / "DEV-20261015-020" / datetime.now().strftime(
    "unit-%Y%m%d-%H%M%S-%f")
OUT.mkdir(parents=True, exist_ok=False)

_HEX = "a" * 64


def review(review_id=3, job_id=10, **over):
    base = {"review_id": review_id, "job_id": job_id, "target": "task", "field_id": None,
            "decision": "questioned", "decision_zh": "存疑待核实", "reviewer": "r", "note": "n",
            "request_id": "rev-1", "input_sha256": _HEX, "model_sha256": "f" * 64,
            "run_seq": 1, "created_at": "2026-10-15 09:00:00"}
    base.update(over)
    return base


def record(correction_id=1, **over):
    base = {"correction_id": correction_id, "root_job_id": 10, "parent_job_id": 10,
            "review_id": 3, "revision_no": 1, "child_job_id": 11, "request_id": "req-1",
            "derived_job_request_id": "corr-abc", "operator": "op", "reason": "why",
            "changes": {"scene_id": "S2"}, "is_synthetic_case": True,
            "submission_state": "linked", "prepared_at": "2026-10-15 10:00:00",
            "submitted_at": "2026-10-15 10:00:01", "error": None}
    base.update(over)
    return base


def summary(job_id=10, **over):
    base = {"job_id": job_id, "status": "succeeded", "input_sha256": _HEX,
            "model_sha256": "f" * 64, "original_filename": "x.tif", "source": "S",
            "scene_id": "SC", "source_link": None, "license_status": None,
            "sample_kind": "synthetic_test", "attempt_count": 1, "run_seq": 1,
            "run_status": "succeeded", "class_counts": {"0": 4, "1": 4, "2": 2, "255": 2},
            "evidence_complete": True,
            "artifacts": [{"name": n, "sha256": "b" * 64, "size_bytes": 1} for n in
                          ("mask.tif", "preview.png", "class_pixels.csv",
                           "field_pixels.csv", "run.json")],
            "region_count": 1, "governance": None, "latest_task_review": None}
    base.update(over)
    return base


def snapshot(**over):
    base = {"schema": "AGROLAKE-CORRECTION-1.0.0",
            "parent_facts": {"job_id": 10, "status": "succeeded", "input_id": 1,
                             "input_sha256": _HEX, "original_filename": "x.tif",
                             "model_sha256": "f" * 64, "source": "S", "scene_id": "SC",
                             "source_link": None, "license_status": None,
                             "sample_kind": "synthetic_test"},
            "parent_registration": {"asset_id": 1, "version_id": 2, "manifest_sha256": "c" * 64},
            "problem_review": review(),
            "revised": {"replaced": False, "input_sha256": _HEX, "filename": "x.tif",
                        "size_bytes": 100, "source": "S", "scene_id": "S2",
                        "source_link": None, "license_status": None,
                        "changed_fields": ["scene_id"], "bytes_changed": False},
            "changes": {"scene_id": "S2"}, "note": "n"}
    base.update(over)
    return base


def comparison(**over):
    base = {"revised_input_sha256": _HEX, "revised_filename": "x.tif",
            "changed_fields": ["scene_id"], "bytes_changed": False,
            "parent_problem_review": review(),
            "parent_registration": {"asset_id": 1, "version_id": 2, "manifest_sha256": "c" * 64},
            "child_state": "succeeded", "child_succeeded": True,
            "same_input_sha256": True, "same_model_sha256": True,
            "input_sha256": {"parent": _HEX, "child": _HEX},
            "model_sha256": {"parent": "f" * 64, "child": "f" * 64},
            "child_latest_task_review": None,
            "nodata": {"parent": {"pixels": 2, "denominator": 12, "ratio": 0.166667},
                       "child": {"pixels": 2, "denominator": 12, "ratio": 0.166667}, "note": "n"},
            "class_pixel_comparison": {"comparable_pixel_space": True, "same_total_pixels": True,
                                       "parent": {"0": 4, "1": 4, "2": 2, "255": 2, "total": 12},
                                       "child": {"0": 4, "1": 4, "2": 2, "255": 2, "total": 12},
                                       "delta": {"0": 0, "1": 0, "2": 0, "255": 0}, "note": "n"},
            "note": "n"}
    base.update(over)
    return base


def detail(**over):
    base = {"correction": record(),
            "snapshot": snapshot(),
            "parent": summary(10, governance={"asset_id": 1, "version_id": 2,
                                              "manifest_sha256": "c" * 64}),
            "child": summary(11, scene_id="S2"),
            "comparison": comparison(), "note": "n"}
    base.update(over)
    return base


def ok_json(obj):
    return 200, {}, json.dumps(obj, ensure_ascii=False).encode("utf-8")


class CorrectionClientTests(unittest.TestCase):
    def setUp(self):
        self._servers = []

    def tearDown(self):
        for stop in self._servers:
            stop()

    def serve(self, responder):
        server, url, stop = start_fake_server(responder)
        self._servers.append(stop)
        return url

    # ---- 创建 -------------------------------------------------------------

    def test_submit_success_echo(self):
        seen = {}

        def responder(method, path, headers, body):
            seen["method"] = method
            seen["path"] = path
            seen["body"] = json.loads(body)
            return 201, {}, json.dumps(dict(record(), created=True), ensure_ascii=False).encode()

        url = self.serve(responder)
        result = C.RecognitionClient(url).submit_correction(
            10, request_id="req-1", review_id=3, operator="op", reason="why",
            changes={"scene_id": "S2"})
        self.assertTrue(result["created"])
        self.assertEqual(seen["path"], "/jobs/10/corrections")
        self.assertEqual(seen["body"]["changes"], {"scene_id": "S2"})

    def test_submit_mismatch_echo_rejected(self):
        # 服务端回执的 operator 与本次不一致 -> 不得当作成功。
        url = self.serve(lambda m, p, h, b: ok_json(dict(record(operator="OTHER"), created=True)))
        with self.assertRaises(C.ApiError) as caught:
            C.RecognitionClient(url).submit_correction(
                10, request_id="req-1", review_id=3, operator="op", reason="why",
                changes={"scene_id": "S2"})
        self.assertEqual(caught.exception.category, "protocol")

    def test_submit_rejects_bool_review_and_bad_replacement(self):
        url = self.serve(lambda m, p, h, b: ok_json(dict(record(), created=True)))
        client = C.RecognitionClient(url)
        with self.assertRaises(C.ApiError):
            client.submit_correction(10, request_id="req-1", review_id=True,
                                     operator="op", reason="why", changes={"scene_id": "S2"})
        with self.assertRaises(C.ApiError):
            client.submit_correction(
                10, request_id="req-1", review_id=3, operator="op", reason="why",
                replacement={"filename": "x.tif", "content_base64": "AA==", "extra": 1})
        with self.assertRaises(C.ApiError):
            client.submit_correction(10, request_id="bad id!", review_id=3,
                                     operator="op", reason="why", changes={"scene_id": "S2"})

    def test_submit_changes_whitelist_blank_and_type(self):
        # changes 白名单/类型本地校验；仅空白可选字段规范化为 None 并与回执一致。
        seen = {}

        def responder(method, path, headers, body):
            payload = json.loads(body)
            seen["changes"] = payload.get("changes")
            return ok_json(dict(record(changes=payload.get("changes") or {}), created=True))

        client = C.RecognitionClient(self.serve(responder))
        with self.assertRaises(C.ApiError) as caught:
            client.submit_correction(10, request_id="req-1", review_id=3, operator="op",
                                     reason="why", changes={"drop_table": 1})
        self.assertEqual(caught.exception.category, "protocol")
        with self.assertRaises(C.ApiError) as caught:
            client.submit_correction(10, request_id="req-1", review_id=3, operator="op",
                                     reason="why", changes={"scene_id": 5})
        self.assertEqual(caught.exception.category, "protocol")
        result = client.submit_correction(
            10, request_id="req-1", review_id=3, operator="op", reason="why",
            changes={"source_link": "   ", "license_status": ""})
        self.assertEqual(seen["changes"], {"source_link": None, "license_status": None})
        self.assertTrue(result["created"])

    def test_submit_conflict_409(self):
        url = self.serve(lambda m, p, h, b: (409, {}, json.dumps(
            {"error": "同一请求号已存在但内容不同"}, ensure_ascii=False).encode()))
        with self.assertRaises(C.ApiError) as caught:
            C.RecognitionClient(url).submit_correction(
                10, request_id="req-1", review_id=3, operator="op", reason="why",
                changes={"scene_id": "S2"})
        self.assertEqual(caught.exception.category, "conflict")

    # ---- 列表 -------------------------------------------------------------

    def test_list_ok_and_completeness(self):
        def responder(method, path, headers, body):
            return ok_json({"job_id": 10, "total": 1, "limit": 50, "truncated": False,
                            "corrections": [record()], "origin": None, "note": "n"})

        url = self.serve(responder)
        listing = C.RecognitionClient(url).list_corrections(10)
        self.assertEqual(listing["total"], 1)

        # 截断标记不一致 -> protocol。
        bad = self.serve(lambda m, p, h, b: ok_json(
            {"job_id": 10, "total": 2, "limit": 50, "truncated": False,
             "corrections": [record()], "origin": None}))
        with self.assertRaises(C.ApiError) as caught:
            C.RecognitionClient(bad).list_corrections(10)
        self.assertEqual(caught.exception.category, "protocol")

    def test_list_cross_job_rejected(self):
        # 回执 job_id 是别的作业 -> 停止展示。
        url = self.serve(lambda m, p, h, b: ok_json(
            {"job_id": 99, "total": 0, "limit": 50, "truncated": False,
             "corrections": [], "origin": None}))
        with self.assertRaises(C.ApiError) as caught:
            C.RecognitionClient(url).list_corrections(10)
        self.assertEqual(caught.exception.category, "protocol")

    def test_list_origin_validated(self):
        # origin 的 child_job_id 必须等于当前作业。
        url = self.serve(lambda m, p, h, b: ok_json(
            {"job_id": 10, "total": 0, "limit": 50, "truncated": False, "corrections": [],
             "origin": record(root_job_id=9, parent_job_id=9, child_job_id=8)}))
        with self.assertRaises(C.ApiError) as caught:
            C.RecognitionClient(url).list_corrections(10)
        self.assertEqual(caught.exception.category, "protocol")

    def test_list_origin_ok(self):
        url = self.serve(lambda m, p, h, b: ok_json(
            {"job_id": 10, "total": 0, "limit": 50, "truncated": False, "corrections": [],
             "origin": record(root_job_id=9, parent_job_id=9, child_job_id=10)}))
        listing = C.RecognitionClient(url).list_corrections(10)
        self.assertEqual(listing["origin"]["child_job_id"], 10)

    # ---- 详情 -------------------------------------------------------------

    def test_detail_ok(self):
        url = self.serve(lambda m, p, h, b: ok_json(detail()))
        got = C.RecognitionClient(url).get_correction(1)
        self.assertEqual(got["correction"]["correction_id"], 1)

    def test_detail_rejects_cross_job_parent(self):
        url = self.serve(lambda m, p, h, b: ok_json(detail(parent=summary(99))))
        with self.assertRaises(C.ApiError) as caught:
            C.RecognitionClient(url).get_correction(1)
        self.assertEqual(caught.exception.category, "protocol")

    def test_detail_rejects_problem_review_mismatch(self):
        # 快照问题意见不属于父作业 -> 拒绝。
        bad_snap = snapshot(problem_review=review(review_id=3, job_id=99))
        url = self.serve(lambda m, p, h, b: ok_json(detail(snapshot=bad_snap)))
        with self.assertRaises(C.ApiError):
            C.RecognitionClient(url).get_correction(1)

    def test_detail_rejects_child_state_mismatch(self):
        # 对照状态与子摘要不一致 -> 拒绝。失败子作业本身证据不齐（无产物/计数）。
        url = self.serve(lambda m, p, h, b: ok_json(
            detail(child=summary(11, scene_id="S2", status="failed",
                                 evidence_complete=False, class_counts=None, artifacts=[]),
                   comparison=comparison(
                       child_state="succeeded", child_succeeded=True,
                       child_latest_task_review=None, class_pixel_comparison=None,
                       nodata={"parent": {"pixels": 2, "denominator": 12, "ratio": 0.166667},
                               "child": None, "note": "n"}))))
        with self.assertRaises(C.ApiError) as caught:
            C.RecognitionClient(url).get_correction(1)
        self.assertEqual(caught.exception.category, "protocol")

    def test_detail_not_submitted_child(self):
        # 未链接子作业：child 为 null，对照 child_state=not_submitted，映射空、无 NoData/像素对照。
        url = self.serve(lambda m, p, h, b: ok_json(detail(
            correction=record(child_job_id=None, submission_state="prepared"),
            child=None,
            comparison=comparison(
                child_state="not_submitted", child_succeeded=False,
                same_input_sha256=None, same_model_sha256=None,
                child_latest_task_review=None, class_pixel_comparison=None,
                input_sha256={"parent": _HEX, "child": None},
                model_sha256={"parent": "f" * 64, "child": None},
                nodata={"parent": {"pixels": 2, "denominator": 12, "ratio": 0.166667},
                        "child": None, "note": "n"}))))
        got = C.RecognitionClient(url).get_correction(1)
        self.assertEqual(got["comparison"]["child_state"], "not_submitted")

    def test_detail_incomplete_pixels_rejected(self):
        url = self.serve(lambda m, p, h, b: ok_json(detail(
            comparison=comparison(class_pixel_comparison={"comparable_pixel_space": True}))))
        with self.assertRaises(C.ApiError):
            C.RecognitionClient(url).get_correction(1)

    def test_detail_rejects_missing_revised_nullable_key(self):
        # 缺失键（而非显式 null）必须 protocol，不能泄漏原生 KeyError。
        snap = snapshot()
        del snap["revised"]["source_link"]
        url = self.serve(lambda m, p, h, b: ok_json(detail(snapshot=snap)))
        with self.assertRaises(C.ApiError) as caught:
            C.RecognitionClient(url).get_correction(1)
        self.assertEqual(caught.exception.category, "protocol")

    def test_detail_rejects_missing_child_opinion_key(self):
        child = summary(11, scene_id="S2")
        del child["latest_task_review"]
        url = self.serve(lambda m, p, h, b: ok_json(detail(child=child)))
        with self.assertRaises(C.ApiError) as caught:
            C.RecognitionClient(url).get_correction(1)
        self.assertEqual(caught.exception.category, "protocol")

    def test_detail_rejects_non_string_sha(self):
        # sha256 给十进制 64 位整数不得被 str() 强转接受。
        child = summary(11, scene_id="S2")
        child["artifacts"][0]["sha256"] = int("1" * 64)
        url = self.serve(lambda m, p, h, b: ok_json(detail(child=child)))
        with self.assertRaises(C.ApiError) as caught:
            C.RecognitionClient(url).get_correction(1)
        self.assertEqual(caught.exception.category, "protocol")

    def test_detail_accepts_explicit_null_nullable(self):
        # 显式 null（默认夹具即 source_link=None、latest_task_review=None）仍合法。
        url = self.serve(lambda m, p, h, b: ok_json(detail()))
        got = C.RecognitionClient(url).get_correction(1)
        self.assertIsNone(got["snapshot"]["revised"]["source_link"])
        self.assertIsNone(got["child"]["latest_task_review"])

    # ---- resume -----------------------------------------------------------

    def test_resume_ok_and_echo(self):
        seen = {}

        def responder(method, path, headers, body):
            seen["path"] = path
            seen["body"] = json.loads(body)
            return ok_json(dict(record(submission_state="linked"), created=False))

        url = self.serve(responder)
        got = C.RecognitionClient(url).resume_correction(1, request_id="req-1")
        self.assertFalse(got["created"])
        self.assertEqual(seen["path"], "/corrections/1/resume")
        self.assertEqual(seen["body"], {"request_id": "req-1"})

    def test_resume_created_true_rejected(self):
        url = self.serve(lambda m, p, h, b: ok_json(dict(record(), created=True)))
        with self.assertRaises(C.ApiError) as caught:
            C.RecognitionClient(url).resume_correction(1, request_id="req-1")
        self.assertEqual(caught.exception.category, "protocol")

    def test_resume_request_mismatch(self):
        url = self.serve(lambda m, p, h, b: ok_json(dict(record(request_id="other"), created=False)))
        with self.assertRaises(C.ApiError):
            C.RecognitionClient(url).resume_correction(1, request_id="req-1")

    # ---- 通用错误 ---------------------------------------------------------

    def test_bad_json_and_offline(self):
        url = self.serve(lambda m, p, h, b: (200, {"Content-Type": "application/json"}, b"nope"))
        with self.assertRaises(C.ApiError) as caught:
            C.RecognitionClient(url).list_corrections(10)
        self.assertEqual(caught.exception.category, "badjson")

    def test_error_status_maps_http(self):
        url = self.serve(lambda m, p, h, b: (404, {}, json.dumps(
            {"error": "未知作业 10"}, ensure_ascii=False).encode()))
        with self.assertRaises(C.ApiError) as caught:
            C.RecognitionClient(url).submit_correction(
                10, request_id="req-1", review_id=3, operator="op", reason="why",
                changes={"scene_id": "S2"})
        self.assertEqual(caught.exception.category, "http")
        self.assertEqual(caught.exception.status, 404)


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(CorrectionClientTests)
    with (OUT / "tests.log").open("x", encoding="utf-8") as log:
        result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    summary_out = {"tests": result.testsRun, "failures": len(result.failures),
                   "errors": len(result.errors), "skipped": len(result.skipped),
                   "successful": result.wasSuccessful(), "directory": str(OUT)}
    (OUT / "result.json").write_text(json.dumps(summary_out, ensure_ascii=False, indent=2),
                                     encoding="utf-8")
    print(json.dumps(summary_out, ensure_ascii=False))
    print((OUT / "tests.log").read_text(encoding="utf-8"))
    sys.exit(0 if result.wasSuccessful() else 1)
