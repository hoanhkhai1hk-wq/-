# -*- coding: utf-8 -*-
"""LAB-20261016-001 修正标注接收与复检的测试（由 Codex 运行；WorkBuddy 只编写）。

测试只写 ``reports/labels/`` 下**全新 fixture 目录**，不使用自动删除的临时目录，
保留全部 fixture；不触碰真实 903 对数据、005 产物、855 候选，不改原产物、不安装、不联网。

事实（不泛称"本轮未运行"）：002 已由 Codex 实际运行并通过——31 模块 / 23 独立 / 真实 14
核对全过，真实输出 ``reports/labels/LAB-20261016-002-intake``；48 份仍 ``pending``、0 提交，
没有人工修标、没有真实训练。004（LAB-20261016-004）仅补默认输出路径边界：
``_resolve_output(None,...)`` 不再绕过输入/修正目录父子校验，并新增 2 个聚焦回归
（``TestDefaultOutputPath``）——**新例待 Codex 复验，不提前称 004 通过**。计划命令：

    & .venv-p0/Scripts/python.exe -B -m unittest tests.labels.test_corrections -v
"""

import copy
import hashlib
import importlib.util
import json
import os
import unittest
import uuid

from unittest import mock

from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
LABELS = os.path.join(ROOT, "reports", "labels")
SCRIPT = os.path.join(ROOT, "scripts", "labels", "intake_corrections.py")

# 与 005 真实 policy 等价的**必要**策略子集（不复制整份文件、不硬编码其 SHA）。
DEFAULT_POLICY = {
    "class_ids": {
        "0": "Field 田块", "1": "Field ridge 田埂",
        "2": "Road 道路", "255": "ignore 忽略/未画/冲突/未确认",
    },
    "no_background_class": True,
    "alias_assumption": {
        "map": {"Fieldridge": "Field ridge"},
        "status": "pending_semantic_review",
    },
    "training_policy": {
        "semantic_reviewed": False, "not_training_ready": True, "no_train_test_split": True,
    },
}


def _load_module():
    spec = importlib.util.spec_from_file_location("intake_corrections_mod", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


MOD = _load_module()


def _rel(p):
    return os.path.relpath(p, ROOT).replace("\\", "/")


def _sha256_file(p):
    with open(p, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _new_dir(parent, prefix):
    while True:
        p = os.path.join(parent, "%s-%s" % (prefix, uuid.uuid4().hex[:12]))
        if not os.path.exists(p):
            os.makedirs(p)
            return p


def _new_path(parent, prefix):
    """返回一个**尚不存在**的全新路径（供 prepare/intake 输出，二者都要求目标不存在）。"""
    while True:
        p = os.path.join(parent, "%s-%s" % (prefix, uuid.uuid4().hex[:12]))
        if not os.path.exists(p):
            return p


def _mask_image(size, values):
    img = Image.new("L", size, 0)
    img.putdata(list(values))
    return img


class CorrFixture:
    """构造 005 风格输入：<fx>/src 原始文件 + <fx>/input/{manifest,policy,masks,review}。"""

    def __init__(self, prefix):
        self.fx = _new_dir(LABELS, prefix)
        self.src = os.path.join(self.fx, "src")
        self.input = os.path.join(self.fx, "input")
        os.makedirs(self.src)
        os.makedirs(self.input)
        os.makedirs(os.path.join(self.input, "masks"))
        os.makedirs(os.path.join(self.input, "review"))
        self.records = []
        self.counter = 0
        self.policy = copy.deepcopy(DEFAULT_POLICY)

    def _emit_src(self, size, color=(10, 20, 30), stem=None):
        self.counter += 1
        stem = stem or ("img_%02d" % self.counter)
        tif = os.path.join(self.src, stem + ".tif")
        Image.new("RGB", size, color).save(tif, format="TIFF")
        return stem, tif, os.path.join(self.src, stem + ".json")

    def _write_src_json(self, jpath, size, shapes):
        stem = os.path.basename(jpath)[:-5]
        data = {"version": "5.1.1", "imagePath": stem + ".tif",
                "imageWidth": size[0], "imageHeight": size[1], "imageData": None,
                "shapes": shapes or [{"label": "Field", "shape_type": "polygon",
                                      "points": [[0, 0], [size[0], 0], [size[0], size[1]]]}]}
        with open(jpath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        return data

    def add_candidate(self, size, mask_values, dataset="邓", sample_id=None):
        stem, tif, jpath = self._emit_src(size)
        self._write_src_json(jpath, size, None)
        sid = sample_id or uuid.uuid4().hex[:16]
        with open(tif, "rb") as f:
            tif_sha = hashlib.sha256(f.read()).hexdigest()
        mimg = _mask_image(size, mask_values)
        mpath = os.path.join(self.input, "masks", sid + ".png")
        mimg.save(mpath, format="PNG")
        hist = mimg.histogram()
        ps = {"denominator": size[0] * size[1], "unpainted_pixels": hist[255],
              "conflict_pixels": 0, "ignore_pixels": hist[255],
              "class_covered_pixels": {"0": hist[0], "1": hist[1], "2": hist[2]},
              "class_pixels": {"0": hist[0], "1": hist[1], "2": hist[2]}}
        rec = {
            "sample_id": sid, "dataset": dataset,
            "json_path": _rel(jpath), "json_sha256": _sha256_file(jpath),
            "json_bytes": os.path.getsize(jpath),
            "image_path": _rel(tif), "image_sha256": tif_sha,
            "image_bytes": os.path.getsize(tif), "width": size[0], "height": size[1],
            "status": "candidate", "label_mapping": [], "alias_normalized": [],
            "unsupported_reasons": [], "review_reasons": [], "derived_json_changes": [],
            "mask_path": "masks/%s.png" % sid, "mask_sha256": _sha256_file(mpath),
            "review_png_path": None, "review_png_sha256": None,
            "review_json_path": None, "review_json_sha256": None,
            "pixel_stats": ps, "semantic_reviewed": False, "not_training_ready": True,
        }
        self.records.append(rec)
        return rec

    def add_review(self, size, reasons, shapes=None, dataset="邓"):
        stem, tif, jpath = self._emit_src(size, color=(40, 50, 60))
        sid = uuid.uuid4().hex[:16]
        data = self._write_src_json(jpath, size, shapes)
        rpng = os.path.join(self.input, "review", sid + ".png")
        with Image.open(tif) as im:
            im.load()
            im.save(rpng, format="PNG")
        rjson = os.path.join(self.input, "review", sid + ".json")
        derived = dict(data, imagePath=sid + ".png")
        with open(rjson, "w", encoding="utf-8") as f:
            json.dump(derived, f, ensure_ascii=False)
        rec = {
            "sample_id": sid, "dataset": dataset,
            "json_path": _rel(jpath), "json_sha256": _sha256_file(jpath),
            "json_bytes": os.path.getsize(jpath),
            "image_path": _rel(tif), "image_sha256": _sha256_file(tif),
            "image_bytes": os.path.getsize(tif), "width": size[0], "height": size[1],
            "status": "review", "label_mapping": [], "alias_normalized": [],
            "unsupported_reasons": [], "review_reasons": reasons, "derived_json_changes": [],
            "mask_path": None, "mask_sha256": None,
            "review_png_path": "review/%s.png" % sid,
            "review_png_sha256": _sha256_file(rpng),
            "review_json_path": "review/%s.json" % sid,
            "review_json_sha256": _sha256_file(rjson),
            "pixel_stats": None, "semantic_reviewed": False, "not_training_ready": True,
        }
        self.records.append(rec)
        return rec

    def write_manifest(self):
        with open(os.path.join(self.input, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump({"task_id": "DATA-20261016-005", "records": self.records},
                      f, ensure_ascii=False)
        with open(os.path.join(self.input, "policy.json"), "w", encoding="utf-8") as f:
            json.dump(self.policy, f, ensure_ascii=False)

    def out_dir(self, prefix="out"):
        return os.path.join(LABELS, "%s-%s" % (prefix, uuid.uuid4().hex[:12]))

    def corrections_dir(self, prefix="corr"):
        return _new_dir(self.fx, prefix)

    def review_copy(self, sid):
        p = os.path.join(self.input, "review", sid + ".json")
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)

    def submit(self, corr_dir, sid, mutator=None, name=None):
        doc = self.review_copy(sid)
        if mutator:
            mutator(doc)
        path = os.path.join(corr_dir, (name or sid) + ".json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False)
        return path


def _read_json(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------------------
BOWTIE = [[0, 0], [7, 7], [7, 0], [0, 7]]          # 自交
QUAD = [[0, 0], [7, 0], [7, 7], [0, 7]]            # 合法四边形


def _bowtie_reason(si=0, label="Field"):
    return [{"type": "self_intersection", "shape_index": si, "label": label}]


class TestBasics(unittest.TestCase):
    def test_empty_submission_all_pending(self):
        b = CorrFixture("lab001-empty")
        r1 = b.add_review((8, 8), _bowtie_reason(), shapes=[
            {"label": "Field", "shape_type": "polygon", "points": BOWTIE}])
        b.add_review((8, 8), _bowtie_reason(), shapes=[
            {"label": "Field", "shape_type": "polygon", "points": BOWTIE}])
        b.write_manifest()
        out = b.out_dir()
        s = MOD.intake(b.input, None, out, root=ROOT)
        self.assertEqual(s["counts"]["input_total"], 2)
        self.assertEqual(s["counts"]["input_review"], 2)
        self.assertEqual(s["counts"]["corrections_files"], 0)
        self.assertEqual(s["counts"]["results"]["pending"], 2)
        self.assertEqual(s["counts"]["results"]["candidate"], 0)
        recs = _read_json(os.path.join(out, "records.json"))["records"]
        self.assertEqual(len(recs), 2)
        for e in recs:
            self.assertEqual(e["status"], "pending")
            self.assertIsNone(e["correction"])
            self.assertIsNone(e["geometry_check_passed"])
        self.assertIsNone(s["corrections_dir"])

    def test_empty_corrections_dir_all_pending(self):
        b = CorrFixture("lab001-emptycorr")
        b.add_review((8, 8), _bowtie_reason(), shapes=[
            {"label": "Field", "shape_type": "polygon", "points": BOWTIE}])
        b.write_manifest()
        corr = b.corrections_dir()
        out = b.out_dir()
        s = MOD.intake(b.input, corr, out, root=ROOT)
        self.assertEqual(s["counts"]["corrections_files"], 0)
        self.assertEqual(s["counts"]["results"]["pending"], 1)


class TestStatuses(unittest.TestCase):
    def test_unchanged_not_claimed_fixed(self):
        b = CorrFixture("lab001-unchanged")
        rec = b.add_review((8, 8), _bowtie_reason(), shapes=[
            {"label": "Field", "shape_type": "polygon", "points": BOWTIE}])
        b.write_manifest()
        corr = b.corrections_dir()
        b.submit(corr, rec["sample_id"])            # 原样提交
        out = b.out_dir()
        s = MOD.intake(b.input, corr, out, root=ROOT)
        self.assertEqual(s["counts"]["results"]["unchanged"], 1)
        e = _read_json(os.path.join(out, "records.json"))["records"][0]
        self.assertEqual(e["status"], "unchanged")
        self.assertEqual(e["changed_shape_indices"], [])
        self.assertFalse(e["geometry_check_passed"])   # 点未变，仍自交
        self.assertIsNone(e["outputs"]["mask_path"])
        # 提交原字节保存在 accepted/
        aj = os.path.join(out, e["outputs"]["accepted_json_path"])
        cj = os.path.join(corr, rec["sample_id"] + ".json")
        with open(aj, "rb") as f1, open(cj, "rb") as f2:
            self.assertEqual(f1.read(), f2.read())
        self.assertTrue(os.path.exists(os.path.join(out, e["outputs"]["accepted_png_path"])))

    def test_bowtie_fixed_to_quad_candidate_and_mask(self):
        b = CorrFixture("lab001-bowtie")
        rec = b.add_review((4, 4), _bowtie_reason(), shapes=[
            {"label": "Field", "shape_type": "polygon",
             "points": [[0, 0], [3, 3], [3, 0], [0, 3]]}])
        b.write_manifest()
        corr = b.corrections_dir()

        def mut(doc):
            doc["shapes"][0]["points"] = [[0, 0], [4, 0], [4, 2], [0, 2]]

        b.submit(corr, rec["sample_id"], mutator=mut)
        out = b.out_dir()
        s = MOD.intake(b.input, corr, out, root=ROOT)
        self.assertEqual(s["counts"]["results"]["candidate"], 1)
        e = _read_json(os.path.join(out, "records.json"))["records"][0]
        self.assertEqual(e["status"], "candidate")
        self.assertTrue(e["geometry_check_passed"])
        self.assertEqual(e["changed_shape_indices"], [0])
        # 4x4 画布：Pillow 闭区间填充 x0..3、y0..2 → 类 0 手算 12 像素，余 4 忽略（不是 8/8）。
        self.assertEqual(e["class_pixels"]["0"], 12)
        self.assertEqual(e["valid_pixels"], 12)
        self.assertEqual(e["ignore_pixels"], 4)
        self.assertEqual(sum(e["class_pixels"].values()), 12)
        # 掩膜 16 像素完整手算：x0..3 且 y0..2 → 值 0（Field）；y=3 整行未画 → 255。
        mask_path = os.path.join(out, e["outputs"]["mask_path"])
        with Image.open(mask_path) as mk:
            mk.load()
            self.assertEqual(mk.mode, "L")
            self.assertEqual(mk.size, (4, 4))
            data = list(mk.getdata())
            hist = mk.histogram()
        expected = [0 if (0 <= x <= 3 and 0 <= y <= 2) else 255
                    for y in range(4) for x in range(4)]
        self.assertEqual(data, expected)
        self.assertEqual(hist[0], 12)
        self.assertEqual(hist[255], 4)
        self.assertEqual(sum(hist[v] for v in (1, 2)), 0)

    def test_out_of_bounds_still_review(self):
        b = CorrFixture("lab001-oob")
        rec = b.add_review((8, 8), [{"type": "canvas_oob", "shape_index": 0, "label": "Field"}],
                           shapes=[{"label": "Field", "shape_type": "polygon",
                                    "points": [[0, 0], [100, 0], [0, 100]]}])
        b.write_manifest()
        corr = b.corrections_dir()

        def mut(doc):
            doc["shapes"][0]["points"] = [[0, 0], [99, 0], [0, 99]]

        b.submit(corr, rec["sample_id"], mutator=mut)
        out = b.out_dir()
        s = MOD.intake(b.input, corr, out, root=ROOT)
        self.assertEqual(s["counts"]["results"]["review"], 1)
        e = _read_json(os.path.join(out, "records.json"))["records"][0]
        self.assertEqual(e["status"], "review")
        self.assertFalse(e["geometry_check_passed"])
        types = {p["type"] for p in e["after_problems"]}
        self.assertIn("canvas_oob", types)
        self.assertIsNone(e["outputs"]["mask_path"])

    def test_collinear_zero_area_review(self):
        b = CorrFixture("lab001-zeroarea")
        rec = b.add_review((8, 8), _bowtie_reason(), shapes=[
            {"label": "Field", "shape_type": "polygon", "points": BOWTIE}])
        b.write_manifest()
        corr = b.corrections_dir()

        def mut(doc):
            doc["shapes"][0]["points"] = [[0, 0], [2, 0], [4, 0]]   # 3 点互异但共线

        b.submit(corr, rec["sample_id"], mutator=mut)
        out = b.out_dir()
        s = MOD.intake(b.input, corr, out, root=ROOT)
        self.assertEqual(s["counts"]["results"]["review"], 1)
        e = _read_json(os.path.join(out, "records.json"))["records"][0]
        types = {p["type"] for p in e["after_problems"]}
        self.assertIn("zero_area", types)
        self.assertEqual(e["status"], "review")

    def test_empty_result_cross_class_overlap(self):
        b = CorrFixture("lab001-emptyres")
        shapes = [
            {"label": "Field", "shape_type": "polygon", "points": [[0, 0], [3, 3], [3, 0], [0, 3]]},
            {"label": "Road", "shape_type": "polygon", "points": [[0, 0], [3, 3], [3, 0], [0, 3]]},
        ]
        reasons = [{"type": "self_intersection", "shape_index": 0, "label": "Field"},
                   {"type": "self_intersection", "shape_index": 1, "label": "Road"}]
        rec = b.add_review((4, 4), reasons, shapes=shapes)
        b.write_manifest()
        corr = b.corrections_dir()

        def mut(doc):
            for sh in doc["shapes"]:
                sh["points"] = [[0, 0], [4, 0], [4, 4], [0, 4]]

        b.submit(corr, rec["sample_id"], mutator=mut)
        out = b.out_dir()
        s = MOD.intake(b.input, corr, out, root=ROOT)
        self.assertEqual(s["counts"]["results"]["empty"], 1)
        e = _read_json(os.path.join(out, "records.json"))["records"][0]
        self.assertEqual(e["status"], "empty")
        self.assertEqual(e["valid_pixels"], 0)
        self.assertIsNone(e["outputs"]["mask_path"])

    def test_points_int_float_equivalent_unchanged(self):
        # points 只换数值表示（int→float）且数值等价，应记 unchanged，不误称已修。
        b = CorrFixture("lab001-ptsfloat")
        rec = b.add_review((8, 8), _bowtie_reason(), shapes=[
            {"label": "Field", "shape_type": "polygon", "points": BOWTIE}])
        b.write_manifest()
        corr = b.corrections_dir()

        def mut(doc):
            doc["shapes"][0]["points"] = [[float(x), float(y)] for x, y in BOWTIE]

        b.submit(corr, rec["sample_id"], mutator=mut)
        out = b.out_dir()
        s = MOD.intake(b.input, corr, out, root=ROOT)
        self.assertEqual(s["counts"]["results"]["unchanged"], 1)
        e = _read_json(os.path.join(out, "records.json"))["records"][0]
        self.assertEqual(e["status"], "unchanged")
        self.assertEqual(e["changed_shape_indices"], [])
        self.assertIsNone(e["outputs"]["mask_path"])


class TestRejections(unittest.TestCase):
    def _expect_reject(self, b, corr, out):
        with self.assertRaises(MOD.CorrectionError):
            MOD.intake(b.input, corr, out, root=ROOT)
        self.assertFalse(os.path.exists(out))

    def _one_review(self, prefix):
        b = CorrFixture(prefix)
        rec = b.add_review((8, 8), _bowtie_reason(), shapes=[
            {"label": "Field", "shape_type": "polygon", "points": BOWTIE}])
        b.write_manifest()
        return b, rec

    def test_nonfinite_rejected(self):
        b, rec = self._one_review("lab001-nan")
        corr = b.corrections_dir()

        def mut(doc):
            doc["shapes"][0]["points"] = [[0, 0], [float("nan"), 0], [7, 7]]

        b.submit(corr, rec["sample_id"], mutator=mut)
        self._expect_reject(b, corr, b.out_dir())

    def test_bool_rejected(self):
        b, rec = self._one_review("lab001-bool")
        corr = b.corrections_dir()

        def mut(doc):
            doc["shapes"][0]["points"] = [[True, 0], [0, 1], [1, 1]]

        b.submit(corr, rec["sample_id"], mutator=mut)
        self._expect_reject(b, corr, b.out_dir())

    def test_class_change_rejected(self):
        b, rec = self._one_review("lab001-class")
        corr = b.corrections_dir()

        def mut(doc):
            doc["shapes"][0]["label"] = "Road"

        b.submit(corr, rec["sample_id"], mutator=mut)
        self._expect_reject(b, corr, b.out_dir())

    def test_dimension_change_rejected(self):
        b, rec = self._one_review("lab001-dim")
        corr = b.corrections_dir()

        def mut(doc):
            doc["imageWidth"] = 999

        b.submit(corr, rec["sample_id"], mutator=mut)
        self._expect_reject(b, corr, b.out_dir())

    def test_same_value_float_width_rejected(self):
        # 8 → 8.0：同值但 int→float 的类型变化，非 points 深度比较必须拒绝。
        b, rec = self._one_review("lab001-floatdim")
        corr = b.corrections_dir()

        def mut(doc):
            doc["imageWidth"] = 8.0

        b.submit(corr, rec["sample_id"], mutator=mut)
        self._expect_reject(b, corr, b.out_dir())

    def test_nested_number_type_change_rejected(self):
        # 嵌套非 points 数字 1 → 1.0：同值换类型也必须拒绝。
        b = CorrFixture("lab001-nestednum")
        rec = b.add_review((8, 8), _bowtie_reason(), shapes=[
            {"label": "Field", "shape_type": "polygon", "group_id": 1, "flags": {},
             "points": BOWTIE}])
        b.write_manifest()
        corr = b.corrections_dir()

        def mut(doc):
            doc["shapes"][0]["group_id"] = 1.0

        b.submit(corr, rec["sample_id"], mutator=mut)
        self._expect_reject(b, corr, b.out_dir())

    def test_group_id_bool_equal_to_int_rejected(self):
        # group_id 1 → True：True == 1，但 bool 与 int 类型不同，必须拒绝。
        b = CorrFixture("lab001-groupbool")
        rec = b.add_review((8, 8), _bowtie_reason(), shapes=[
            {"label": "Field", "shape_type": "polygon", "group_id": 1, "flags": {},
             "points": BOWTIE}])
        b.write_manifest()
        corr = b.corrections_dir()

        def mut(doc):
            doc["shapes"][0]["group_id"] = True

        b.submit(corr, rec["sample_id"], mutator=mut)
        self._expect_reject(b, corr, b.out_dir())

    def test_toplevel_change_rejected(self):
        b, rec = self._one_review("lab001-top")
        corr = b.corrections_dir()

        def mut(doc):
            doc["version"] = "9.9.9"

        b.submit(corr, rec["sample_id"], mutator=mut)
        self._expect_reject(b, corr, b.out_dir())

    def test_training_ready_field_rejected(self):
        b, rec = self._one_review("lab001-ready")
        corr = b.corrections_dir()

        def mut(doc):
            doc["training_ready"] = True

        b.submit(corr, rec["sample_id"], mutator=mut)
        self._expect_reject(b, corr, b.out_dir())

    def test_add_shape_rejected(self):
        b, rec = self._one_review("lab001-add")
        corr = b.corrections_dir()

        def mut(doc):
            doc["shapes"].append({"label": "Field", "shape_type": "polygon",
                                  "points": [[1, 1], [2, 1], [2, 2]]})

        b.submit(corr, rec["sample_id"], mutator=mut)
        self._expect_reject(b, corr, b.out_dir())

    def test_remove_shape_rejected(self):
        b = CorrFixture("lab001-rm")
        shapes = [{"label": "Field", "shape_type": "polygon", "points": BOWTIE},
                  {"label": "Road", "shape_type": "polygon", "points": BOWTIE}]
        reasons = [_bowtie_reason(0, "Field")[0], _bowtie_reason(1, "Road")[0]]
        rec = b.add_review((8, 8), reasons, shapes=shapes)
        b.write_manifest()
        corr = b.corrections_dir()

        def mut(doc):
            doc["shapes"].pop()

        b.submit(corr, rec["sample_id"], mutator=mut)
        self._expect_reject(b, corr, b.out_dir())

    def test_candidate_id_rejected(self):
        b = CorrFixture("lab001-candid")
        cand = b.add_candidate((4, 4), [0] * 16)
        b.add_review((8, 8), _bowtie_reason(), shapes=[
            {"label": "Field", "shape_type": "polygon", "points": BOWTIE}])
        b.write_manifest()
        corr = b.corrections_dir()
        # 以候选 id 名提交（内容无所谓，成员资格先拒绝）
        with open(os.path.join(corr, cand["sample_id"] + ".json"), "w", encoding="utf-8") as f:
            json.dump({"shapes": []}, f)
        self._expect_reject(b, corr, b.out_dir())

    def test_unknown_id_rejected(self):
        b, rec = self._one_review("lab001-unknown")
        corr = b.corrections_dir()
        b.submit(corr, rec["sample_id"])                      # 合法一份
        with open(os.path.join(corr, "deadbeefdeadbeef.json"), "w", encoding="utf-8") as f:
            json.dump({"shapes": []}, f)
        self._expect_reject(b, corr, b.out_dir())

    def test_subdir_rejected(self):
        b, rec = self._one_review("lab001-subdir")
        corr = b.corrections_dir()
        os.makedirs(os.path.join(corr, "nested"))
        self._expect_reject(b, corr, b.out_dir())

    def test_other_extension_rejected(self):
        b, rec = self._one_review("lab001-ext")
        corr = b.corrections_dir()
        with open(os.path.join(corr, "notes.txt"), "w", encoding="utf-8") as f:
            f.write("x")
        self._expect_reject(b, corr, b.out_dir())

    def test_last_bad_no_output(self):
        b = CorrFixture("lab001-lastbad")
        good = b.add_review((8, 8), _bowtie_reason(), shapes=[
            {"label": "Field", "shape_type": "polygon", "points": BOWTIE}])
        bad = b.add_review((8, 8), _bowtie_reason(), shapes=[
            {"label": "Field", "shape_type": "polygon", "points": BOWTIE}])
        b.write_manifest()
        corr = b.corrections_dir()
        b.submit(corr, good["sample_id"])

        def mut(doc):
            doc["shapes"][0]["label"] = "Road"

        b.submit(corr, bad["sample_id"], mutator=mut)
        out = b.out_dir()
        with self.assertRaises(MOD.CorrectionError):
            MOD.intake(b.input, corr, out, root=ROOT)
        self.assertFalse(os.path.exists(out))


class TestScopeAndSafety(unittest.TestCase):
    def test_rerun_existing_output_refused(self):
        b = CorrFixture("lab001-rerun")
        b.add_review((8, 8), _bowtie_reason(), shapes=[
            {"label": "Field", "shape_type": "polygon", "points": BOWTIE}])
        b.write_manifest()
        out = b.out_dir()
        MOD.intake(b.input, None, out, root=ROOT)
        with self.assertRaises(MOD.CorrectionError):
            MOD.intake(b.input, None, out, root=ROOT)

    def test_output_inside_input_refused(self):
        b = CorrFixture("lab001-inside")
        b.add_review((8, 8), _bowtie_reason(), shapes=[
            {"label": "Field", "shape_type": "polygon", "points": BOWTIE}])
        b.write_manifest()
        with self.assertRaises(MOD.CorrectionError):
            MOD.intake(b.input, None, os.path.join(b.input, "intake"), root=ROOT)

    def test_corrections_inside_input_refused(self):
        b = CorrFixture("lab001-corrinside")
        b.add_review((8, 8), _bowtie_reason(), shapes=[
            {"label": "Field", "shape_type": "polygon", "points": BOWTIE}])
        b.write_manifest()
        with self.assertRaises(MOD.CorrectionError):
            MOD.intake(b.input, os.path.join(b.input, "corr"), b.out_dir(), root=ROOT)

    def test_original_files_unchanged(self):
        b = CorrFixture("lab001-intact")
        rec = b.add_review((8, 8), _bowtie_reason(), shapes=[
            {"label": "Field", "shape_type": "polygon", "points": BOWTIE}])
        b.write_manifest()
        corr = b.corrections_dir()

        def mut(doc):
            doc["shapes"][0]["points"] = QUAD

        b.submit(corr, rec["sample_id"], mutator=mut)
        before_j = _sha256_file(os.path.join(ROOT, rec["json_path"]))
        before_i = _sha256_file(os.path.join(ROOT, rec["image_path"]))
        before_copy = _sha256_file(os.path.join(b.input, rec["review_json_path"]))
        MOD.intake(b.input, corr, b.out_dir(), root=ROOT)
        self.assertEqual(before_j, _sha256_file(os.path.join(ROOT, rec["json_path"])))
        self.assertEqual(before_i, _sha256_file(os.path.join(ROOT, rec["image_path"])))
        self.assertEqual(before_copy, _sha256_file(os.path.join(b.input, rec["review_json_path"])))

    def test_non_project_cwd_and_utf8(self):
        b = CorrFixture("lab001-cwd")
        rec = b.add_review((8, 8), _bowtie_reason(), shapes=[
            {"label": "Field", "shape_type": "polygon", "points": BOWTIE}], dataset="邓")
        b.write_manifest()
        corr = b.corrections_dir()

        def mut(doc):
            doc["shapes"][0]["points"] = QUAD

        b.submit(corr, rec["sample_id"], mutator=mut)
        out = b.out_dir()
        old = os.getcwd()
        try:
            os.chdir(LABELS)                        # 非项目 cwd
            s = MOD.intake(_rel(b.input), _rel(corr), _rel(out), root=ROOT)
        finally:
            os.chdir(old)
        self.assertEqual(s["counts"]["results"]["candidate"], 1)
        with open(os.path.join(out, "records.csv"), "rb") as f:
            raw = f.read()
        self.assertTrue(raw.startswith(b"\xef\xbb\xbf"))    # UTF-8 BOM
        self.assertIn("邓".encode("utf-8"), raw)

    def test_resolved_submission_escape_rejected_mocked_link(self):
        # 不依赖 Windows 具备建 symlink 权限：mock realpath 让提交文件解析到修正目录之外。
        b = CorrFixture("lab001-link")
        rec = b.add_review((8, 8), _bowtie_reason(), shapes=[
            {"label": "Field", "shape_type": "polygon", "points": BOWTIE}])
        b.write_manifest()
        corr = b.corrections_dir()
        b.submit(corr, rec["sample_id"])
        out = b.out_dir()
        outside = os.path.join(b.src, rec["sample_id"] + ".json")
        real = MOD.os.path.realpath
        # 只重定向“修正目录内那份提交文件”，不影响 _scan_record 对 review 副本的解析。
        linked = {os.path.normcase(os.path.join(corr, rec["sample_id"] + ".json")),
                  os.path.normcase(os.path.join(os.path.realpath(corr),
                                                rec["sample_id"] + ".json"))}

        def fake(p, *args, **kwargs):
            if os.path.normcase(str(p)) in linked:
                return outside
            return real(p, *args, **kwargs)

        with mock.patch.object(MOD.os.path, "realpath", side_effect=fake):
            with self.assertRaises(MOD.CorrectionError):
                MOD.intake(b.input, corr, out, root=ROOT)
        self.assertFalse(os.path.exists(out))


class TestDefaultOutputPath(unittest.TestCase):
    """LAB-20261016-004：默认输出路径不再绕过输入/修正目录父子范围校验。"""

    def test_default_output_rejected_when_input_is_labels_root(self):
        # 输入为 reports/labels 根时，默认生成目录必位于输入内 → 必须 CorrectionError。
        with self.assertRaises(MOD.CorrectionError):
            MOD._resolve_output(None, ROOT, LABELS, None)

    def test_default_output_generated_for_normal_fixture(self):
        # 正常输入（labels 下独立 fixture）时省略 --output：默认生成全新同级目录，可用。
        b = CorrFixture("lab004-default")
        b.add_review((8, 8), _bowtie_reason(), shapes=[
            {"label": "Field", "shape_type": "polygon", "points": BOWTIE}])
        b.write_manifest()
        s = MOD.intake(b.input, None, None, root=ROOT)
        out_real = os.path.realpath(s["output_dir"]) if os.path.isabs(s["output_dir"]) \
            else os.path.realpath(os.path.join(ROOT, s["output_dir"]))
        labels_real = os.path.realpath(LABELS)
        input_real = os.path.realpath(b.input)
        # 默认输出位于 reports/labels 内，但不在输入目录之内（同级新目录），且真实落盘。
        self.assertTrue(MOD._within(out_real, labels_real))
        self.assertFalse(MOD._within(out_real, input_real))
        self.assertTrue(os.path.isdir(out_real))
        self.assertTrue(os.path.isfile(os.path.join(out_real, "summary.json")))


class TestPrepareIntegration(unittest.TestCase):
    """用既有 prepare_candidate.prepare 生成真实 005 风格输入，再走 intake。

    避免手写 fixture 掩盖接口不符；仅合成 8x8，不触碰真实数据。
    """

    def test_prepare_then_intake(self):
        prep = MOD._PREP
        fx = _new_dir(LABELS, "lab002-integration")
        src = os.path.join(fx, "src")
        os.makedirs(src)

        def emit(stem, points, color):
            tif = os.path.join(src, stem + ".tif")
            Image.new("RGB", (8, 8), color).save(tif, format="TIFF")
            jpath = os.path.join(src, stem + ".json")
            doc = {"version": "5.1.1", "imagePath": stem + ".tif",
                   "imageWidth": 8, "imageHeight": 8, "imageData": None,
                   "shapes": [{"label": "Field", "shape_type": "polygon", "points": points}]}
            with open(jpath, "w", encoding="utf-8") as f:
                json.dump(doc, f, ensure_ascii=False)
            return {
                "sample_id": stem, "dataset": "邓",
                "json_path": _rel(jpath), "json_sha256": _sha256_file(jpath),
                "json_bytes": os.path.getsize(jpath),
                "image_path": _rel(tif), "image_sha256": _sha256_file(tif),
                "image_bytes": os.path.getsize(tif), "width": 8, "height": 8,
            }

        sid_rev = "lab002rev" + uuid.uuid4().hex[:6]
        sid_cnd = "lab002cnd" + uuid.uuid4().hex[:6]
        rec_rev = emit(sid_rev, [[0, 0], [7, 7], [7, 0], [0, 7]], (20, 30, 40))   # 自交
        rec_cnd = emit(sid_cnd, [[1, 1], [6, 1], [6, 6], [1, 6]], (40, 50, 60))   # 合法

        idx = os.path.join(fx, "index.json")
        with open(idx, "w", encoding="utf-8") as f:
            json.dump({"schema_version": "candidate-input/1",
                       "records": [rec_rev, rec_cnd]}, f, ensure_ascii=False)

        prepared = _new_path(LABELS, "lab002-prep")
        prep.prepare(idx, prepared, root=ROOT)
        # prepare 真实产物：一份 review 副本 + 一份候选。
        self.assertTrue(os.path.isfile(os.path.join(prepared, "review", sid_rev + ".json")))
        self.assertTrue(os.path.isfile(os.path.join(prepared, "masks", sid_cnd + ".png")))
        by_status = {r["sample_id"]: r["status"]
                     for r in _read_json(os.path.join(prepared, "manifest.json"))["records"]}
        self.assertEqual(by_status[sid_rev], "review")
        self.assertEqual(by_status[sid_cnd], "candidate")

        # 只改 points 修正 review，再由 intake 复检。
        corr = _new_dir(fx, "corr")
        rdoc = _read_json(os.path.join(prepared, "review", sid_rev + ".json"))
        rdoc["shapes"][0]["points"] = [[0, 0], [7, 0], [7, 7], [0, 7]]
        with open(os.path.join(corr, sid_rev + ".json"), "w", encoding="utf-8") as f:
            json.dump(rdoc, f, ensure_ascii=False)

        out = _new_path(LABELS, "lab002-intake")
        s = MOD.intake(prepared, corr, out, root=ROOT)
        self.assertEqual(s["counts"]["input_total"], 2)
        self.assertEqual(s["counts"]["input_review"], 1)
        self.assertEqual(s["counts"]["input_candidate"], 1)
        self.assertEqual(s["counts"]["corrections_files"], 1)
        self.assertEqual(s["counts"]["results"]["candidate"], 1)
        self.assertEqual(s["counts"]["results"]["pending"], 0)
        recs = {r["sample_id"]: r for r in _read_json(os.path.join(out, "records.json"))["records"]}
        self.assertEqual(set(recs), {sid_rev})            # 仅 review 记录进入 results
        self.assertEqual(recs[sid_rev]["status"], "candidate")
        self.assertEqual(recs[sid_rev]["changed_shape_indices"], [0])
        self.assertTrue(recs[sid_rev]["geometry_check_passed"])
        # accepted 原字节 == 提交字节。
        aj = os.path.join(out, recs[sid_rev]["outputs"]["accepted_json_path"])
        with open(aj, "rb") as f1, open(os.path.join(corr, sid_rev + ".json"), "rb") as f2:
            self.assertEqual(f1.read(), f2.read())


if __name__ == "__main__":
    unittest.main(verbosity=2)
