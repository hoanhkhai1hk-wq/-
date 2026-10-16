# -*- coding: utf-8 -*-
"""DATA-20261016-009 训练前复核工具最终小修的测试（由 Codex 运行；WorkBuddy 只编写）。

测试只写 `reports/labels/` 下**全新 fixture 目录**，不使用自动删除的临时目录，
保留所有 fixture；不触碰真实 903 对数据、不改原产物、不安装、不联网。

    & .venv-p0/Scripts/python.exe -B -m unittest tests.labels.test_review_pack -v
"""

import copy
import hashlib
import importlib.util
import json
import os
import unittest
import uuid

from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
LABELS = os.path.join(ROOT, "reports", "labels")
SCRIPT = os.path.join(ROOT, "scripts", "labels", "build_review_pack.py")

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
    spec = importlib.util.spec_from_file_location("build_review_pack_mod", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


MOD = _load_module()


def _rel(p):
    return os.path.relpath(p, ROOT).replace("\\", "/")


def _sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def _new_dir(parent, prefix):
    while True:
        p = os.path.join(parent, "%s-%s" % (prefix, uuid.uuid4().hex[:12]))
        if not os.path.exists(p):
            os.makedirs(p)
            return p


def _mask_image(size, values):
    """按 0/1/2/255 值序列构造 L 模式 mask。"""
    img = Image.new("L", size, 0)
    img.putdata(list(values))
    return img


class PackFixture:
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
        if stem is None:
            stem = "img_%02d" % self.counter
        tif = os.path.join(self.src, stem + ".tif")
        Image.new("RGB", size, color).save(tif, format="TIFF")
        jpath = os.path.join(self.src, stem + ".json")
        return stem, tif, jpath

    def _write_src_json(self, jpath, size, shapes):
        data = {"version": "5.1.1", "imagePath": os.path.basename(jpath)[:-5] + ".tif",
                "imageWidth": size[0], "imageHeight": size[1], "imageData": None,
                "shapes": shapes or [{"label": "Field", "shape_type": "polygon",
                                      "points": [[0, 0], [size[0], 0], [size[0], size[1]]]}]}
        with open(jpath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        return data

    def add_candidate(self, size, mask_values, dataset="邓", tile=None,
                      pixel_stats_override=None, mask_values_illegal=None,
                      image_sha_override=None, status="candidate", semantic_reviewed=False,
                      not_training_ready=True, shapes=None, sample_id=None):
        # tile 作为文件名前缀，先写文件再使用，避免提前重命名不存在的 JSON。
        stem, tif, jpath = self._emit_src(size, stem=tile)
        data = self._write_src_json(jpath, size, shapes)
        assert data["imagePath"] == os.path.basename(tif)
        sid = sample_id or uuid.uuid4().hex[:16]
        tif_bytes = os.path.getsize(tif)
        with open(tif, "rb") as f:
            tif_sha = hashlib.sha256(f.read()).hexdigest()
        # 候选 mask
        mv = mask_values_illegal if mask_values_illegal is not None else mask_values
        mimg = _mask_image(size, mv)
        if mask_values_illegal is not None:
            mimg.putdata(list(mv))
        mpath = os.path.join(self.input, "masks", sid + ".png")
        mimg.save(mpath, format="PNG")
        hist = mimg.histogram()
        ps = {"denominator": size[0] * size[1], "unpainted_pixels": hist[255],
              "conflict_pixels": 0, "ignore_pixels": hist[255],
              "class_covered_pixels": {"0": hist[0], "1": hist[1], "2": hist[2]},
              "class_pixels": {"0": hist[0], "1": hist[1], "2": hist[2]}}
        if pixel_stats_override:
            ps.update(pixel_stats_override)
        rec = {
            "sample_id": sid, "dataset": dataset,
            "json_path": _rel(jpath), "json_sha256": _sha256_file(jpath),
            "json_bytes": os.path.getsize(jpath),
            "image_path": _rel(tif),
            "image_sha256": image_sha_override or tif_sha,
            "image_bytes": tif_bytes,
            "width": size[0], "height": size[1], "status": status,
            "label_mapping": [], "alias_normalized": [], "unsupported_reasons": [],
            "review_reasons": [], "derived_json_changes": [],
            "mask_path": "masks/%s.png" % sid, "mask_sha256": _sha256_file(mpath),
            "review_png_path": None, "review_png_sha256": None,
            "review_json_path": None, "review_json_sha256": None,
            "pixel_stats": ps, "semantic_reviewed": semantic_reviewed,
            "not_training_ready": not_training_ready,
        }
        self.records.append(rec)
        return rec

    def add_review(self, size, reasons, shapes=None, dataset="邓",
                   copy_mismatch=False, copy_mutator=None, status="review"):
        stem, tif, jpath = self._emit_src(size, color=(40, 50, 60))
        sid = uuid.uuid4().hex[:16]
        data = self._write_src_json(jpath, size, shapes)
        rpng = os.path.join(self.input, "review", sid + ".png")
        with Image.open(tif) as im:
            im.load()
            if copy_mismatch:
                im2 = im.copy()
                im2.putpixel((0, 0), (255, 255, 255))
                im2.save(rpng, format="PNG")
            else:
                im.save(rpng, format="PNG")
        rjson = os.path.join(self.input, "review", sid + ".json")
        derived = dict(data, imagePath=sid + ".png")
        if copy_mutator:
            copy_mutator(derived)
        with open(rjson, "w", encoding="utf-8") as f:
            json.dump(derived, f, ensure_ascii=False)
        rec = {
            "sample_id": sid, "dataset": dataset,
            "json_path": _rel(jpath), "json_sha256": _sha256_file(jpath),
            "json_bytes": os.path.getsize(jpath),
            "image_path": _rel(tif), "image_sha256": _sha256_file(tif),
            "image_bytes": os.path.getsize(tif),
            "width": size[0], "height": size[1], "status": status,
            "label_mapping": [], "alias_normalized": [], "unsupported_reasons": [],
            "review_reasons": reasons, "derived_json_changes": [],
            "mask_path": None, "mask_sha256": None,
            "review_png_path": "review/%s.png" % sid,
            "review_png_sha256": _sha256_file(rpng),
            "review_json_path": "review/%s.json" % sid,
            "review_json_sha256": _sha256_file(rjson),
            "pixel_stats": None, "semantic_reviewed": False, "not_training_ready": True,
        }
        self.records.append(rec)
        return rec

    def write_manifest(self, write_policy=True):
        path = os.path.join(self.input, "manifest.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"task_id": "DATA-20261016-005", "records": self.records},
                      f, ensure_ascii=False)
        if write_policy:
            ppath = os.path.join(self.input, "policy.json")
            with open(ppath, "w", encoding="utf-8") as f:
                json.dump(self.policy, f, ensure_ascii=False)
        return path

    def out_dir(self, prefix="out"):
        return os.path.join(LABELS, "%s-%s" % (prefix, uuid.uuid4().hex[:12]))


def _read_json(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------------------
class TestAuditStats(unittest.TestCase):
    def test_counts_and_class_image_counts_not_hardcoded(self):
        b = PackFixture("fx-007-stats")
        # 16 像素：0×10,1×3,2×1,255×2
        vals = [0] * 10 + [1] * 3 + [2] * 1 + [255] * 2
        b.add_candidate((4, 4), vals)
        b.add_candidate((4, 4), [0] * 16)          # 仅类 0
        b.write_manifest()
        out = b.out_dir()
        res = MOD.build_pack(b.input, out, root=ROOT)
        self.assertEqual(res["counts"]["input"], 2)
        self.assertEqual(res["counts"]["candidate"], 2)
        audit = _read_json(os.path.join(out, "audit.json"))
        self.assertFalse(audit["training_ready"])
        self.assertFalse(audit["semantic_reviewed"])
        self.assertFalse(audit["split_created"])
        # 类 0 两张都有像素；类 1/2 各 1 张
        self.assertEqual(audit["candidate_class_image_counts"]["0"], 2)
        self.assertEqual(audit["candidate_class_image_counts"]["1"], 1)
        self.assertEqual(audit["candidate_class_image_counts"]["2"], 1)
        self.assertEqual(audit["candidate_pixel_totals"]["class_pixels"]["0"], 26)
        self.assertEqual(audit["candidate_pixel_totals"]["ignore_pixels"], 2)

    def test_hand_computed_pixels_in_csv(self):
        b = PackFixture("fx-007-csv")
        vals = [0] * 10 + [1] * 3 + [2] * 1 + [255] * 2
        rec = b.add_candidate((4, 4), vals)
        b.write_manifest()
        out = b.out_dir()
        MOD.build_pack(b.input, out, root=ROOT)
        with open(os.path.join(out, "records.csv"), "r", encoding="utf-8-sig") as f:
            text = f.read()
        self.assertIn(rec["sample_id"], text)
        audit = _read_json(os.path.join(out, "audit.json"))
        self.assertEqual(audit["candidate_pixel_totals"]["class_pixels"]["1"], 3)
        self.assertEqual(audit["candidate_pixel_totals"]["valid_pixels"], 14)
        # ignore 比例 2/16
        self.assertEqual(audit["candidate_pixel_totals"]["ignore_pixels"], 2)
        self.assertEqual(audit["candidate_pixel_totals"]["denominator_sum"], 16)

    def test_wrong_histogram_rejected(self):
        b = PackFixture("fx-007-hist")
        vals = [0] * 16
        b.add_candidate((4, 4), vals,
                        pixel_stats_override={"class_pixels": {"0": 15, "1": 0, "2": 0}})
        b.write_manifest()
        out = b.out_dir()
        with self.assertRaises(MOD.BuildError):
            MOD.build_pack(b.input, out, root=ROOT)
        self.assertFalse(os.path.exists(out))

    def test_illegal_mask_value_rejected(self):
        b = PackFixture("fx-007-maskval")
        b.add_candidate((4, 4), [0] * 16, mask_values_illegal=[0] * 15 + [7])
        b.write_manifest()
        out = b.out_dir()
        with self.assertRaises(MOD.BuildError):
            MOD.build_pack(b.input, out, root=ROOT)
        self.assertFalse(os.path.exists(out))

    def test_candidate_without_valid_pixels_rejected(self):
        b = PackFixture("fx-007-novalid")
        b.add_candidate((4, 4), [255] * 16)
        b.write_manifest()
        out = b.out_dir()
        with self.assertRaises(MOD.BuildError):
            MOD.build_pack(b.input, out, root=ROOT)
        self.assertFalse(os.path.exists(out))


class TestReviewPreview(unittest.TestCase):
    def test_problem_shape_index_preserved_and_preview(self):
        b = PackFixture("fx-007-review")
        reasons = [{"type": "canvas_oob", "shape_index": 0, "label": "Field",
                    "points": [{"point_index": 1, "x": -0.15, "y": 512.3, "excess": 0.31}]}]
        rec = b.add_review((8, 8), reasons)
        b.write_manifest()
        out = b.out_dir()
        MOD.build_pack(b.input, out, root=ROOT)
        gal = _read_json(os.path.join(out, "gallery.json"))
        rev = [s for s in gal["samples"] if s["sample_id"] == rec["sample_id"]][0]
        self.assertEqual(rev["kind"], "review")
        self.assertEqual(rev["problems"][0]["shape_index"], 0)
        self.assertEqual(rev["problems"][0]["type"], "canvas_oob")
        # 越界坐标完整保留于 detail 文本
        self.assertIn("x", rev["problems"][0]["detail"]["points"][0])
        self.assertTrue(os.path.exists(os.path.join(out, rev["preview_path"])))
        self.assertEqual(rev["preview_basis"], "relative to output directory")
        # 派生副本引用基准
        bases = {r["basis"] for r in rev["derived_refs"]}
        self.assertIn("relative to input directory", bases)

    def test_review_copy_pixel_mismatch_rejected(self):
        b = PackFixture("fx-007-copymis")
        b.add_review((8, 8), [{"type": "self_intersection", "shape_index": 0, "label": "Field"}],
                     copy_mismatch=True)
        b.write_manifest()
        out = b.out_dir()
        with self.assertRaises(MOD.BuildError):
            MOD.build_pack(b.input, out, root=ROOT)
        self.assertFalse(os.path.exists(out))


class TestSelection(unittest.TestCase):
    def test_select_deterministic_limit_and_priority(self):
        cands = []
        for i in range(20):
            cands.append({"sample_id": "s%02d" % i, "dataset": "邓" if i % 2 else "hy0713 (1)",
                          "class_pixels": {"0": 1, "1": 0, "2": 0}})
        cands.append({"sample_id": "road1", "dataset": "邓",
                      "class_pixels": {"0": 1, "1": 0, "2": 1}})
        cands.append({"sample_id": "ridge1", "dataset": "hy0713 (1)",
                      "class_pixels": {"0": 1, "1": 1, "2": 0}})
        a = MOD.select_samples(cands, limit=12)
        c = MOD.select_samples(list(cands), limit=12)
        self.assertEqual([x["sample_id"] for x in a], [x["sample_id"] for x in c])
        self.assertLessEqual(len(a), 12)
        ids = [x["sample_id"] for x in a]
        self.assertIn("road1", ids)       # 道路优先
        self.assertIn("ridge1", ids)      # 田埂优先
        self.assertEqual(len(ids), len(set(ids)))

    def test_select_no_more_than_available(self):
        cands = [{"sample_id": "a", "dataset": "邓", "class_pixels": {"0": 1, "1": 0, "2": 0}}]
        self.assertEqual(len(MOD.select_samples(cands, limit=12)), 1)


class TestParsingAndLinks(unittest.TestCase):
    def test_parse_tile_name(self):
        p = MOD.parse_tile_name("邓/hy0717_tile_r35_c49.tif")
        self.assertEqual(p["prefix"], "hy0717")
        self.assertEqual(p["row"], "35")
        self.assertEqual(p["col"], "49")
        empty = MOD.parse_tile_name("no_tile_pattern.tif")
        self.assertEqual(empty["prefix"], "")

    def test_parse_tile_name_empty_prefix(self):
        # `tile_r54_c24` 无 prefix，但仍应保留 row/col。
        p = MOD.parse_tile_name("tile_r54_c24.tif")
        self.assertEqual(p["prefix"], "")
        self.assertEqual(p["row"], "54")
        self.assertEqual(p["col"], "24")

    def test_href_encodes_spaces(self):
        self.assertEqual(MOD._href("review/a b.png"), "review/a%20b.png")

    def test_html_escapes_dataset(self):
        b = PackFixture("fx-007-esc")
        b.add_candidate((4, 4), [0] * 16, dataset="A<B>&C")
        b.write_manifest()
        out = b.out_dir()
        MOD.build_pack(b.input, out, root=ROOT)
        with open(os.path.join(out, "index.html"), "r", encoding="utf-8") as f:
            text = f.read()
        self.assertIn("A&lt;B&gt;&amp;C", text)
        self.assertNotIn("A<B>&C", text)


class TestValidationAndScope(unittest.TestCase):
    def test_last_record_bad_source_no_output(self):
        b = PackFixture("fx-007-lastbad")
        b.add_candidate((4, 4), [0] * 16)
        b.add_candidate((4, 4), [0] * 16, image_sha_override="0" * 64)
        b.write_manifest()
        out = b.out_dir()
        with self.assertRaises(MOD.BuildError):
            MOD.build_pack(b.input, out, root=ROOT)
        self.assertFalse(os.path.exists(out))

    def test_escape_path_rejected(self):
        b = PackFixture("fx-007-escape")
        rec = b.add_candidate((4, 4), [0] * 16)
        rec["json_path"] = "../evil.json"
        b.write_manifest()
        out = b.out_dir()
        with self.assertRaises(MOD.BuildError):
            MOD.build_pack(b.input, out, root=ROOT)
        self.assertFalse(os.path.exists(out))

    def test_unsupported_status_rejected(self):
        b = PackFixture("fx-007-status")
        b.add_candidate((4, 4), [0] * 16, status="empty")
        b.write_manifest()
        out = b.out_dir()
        with self.assertRaises(MOD.BuildError):
            MOD.build_pack(b.input, out, root=ROOT)

    def test_training_ready_input_rejected(self):
        b = PackFixture("fx-007-ready")
        b.add_candidate((4, 4), [0] * 16, not_training_ready=False)
        b.write_manifest()
        out = b.out_dir()
        with self.assertRaises(MOD.BuildError):
            MOD.build_pack(b.input, out, root=ROOT)

    def test_rerun_existing_output_refused(self):
        b = PackFixture("fx-007-rerun")
        b.add_candidate((4, 4), [0] * 16)
        b.write_manifest()
        out = b.out_dir()
        MOD.build_pack(b.input, out, root=ROOT)
        with self.assertRaises(MOD.BuildError):
            MOD.build_pack(b.input, out, root=ROOT)

    def test_output_inside_input_refused(self):
        b = PackFixture("fx-007-inside")
        b.add_candidate((4, 4), [0] * 16)
        b.write_manifest()
        out = os.path.join(b.input, "pack")
        with self.assertRaises(MOD.BuildError):
            MOD.build_pack(b.input, out, root=ROOT)

    def test_original_files_unchanged(self):
        b = PackFixture("fx-007-intact")
        rec = b.add_candidate((4, 4), [0] * 16)
        b.write_manifest()
        before = _sha256_file(os.path.join(ROOT, rec["image_path"]))
        out = b.out_dir()
        MOD.build_pack(b.input, out, root=ROOT)
        self.assertEqual(before, _sha256_file(os.path.join(ROOT, rec["image_path"])))
        self.assertEqual(_sha256_file(os.path.join(ROOT, rec["json_path"])), rec["json_sha256"])


class TestFix008Boundaries(unittest.TestCase):
    """DATA-20261016-008 定向修复对应的独立失败案例。"""

    def _expect_reject(self, b):
        b.write_manifest()
        out = b.out_dir()
        with self.assertRaises(MOD.BuildError):
            MOD.build_pack(b.input, out, root=ROOT)
        self.assertFalse(os.path.exists(out))

    # ---- 1. 原 JSON 前置结构校验（含末条坏 JSON 不留半成品）----
    def test_last_invalid_source_json_no_output(self):
        b = PackFixture("fx-008-lastjson")
        b.add_candidate((4, 4), [0] * 16)
        b.add_review((8, 8), [{"type": "canvas_oob", "shape_index": 0, "label": "Field"}])
        r = b.records[-1]
        jp = os.path.join(ROOT, r["json_path"])
        with open(jp, "w", encoding="utf-8") as f:
            f.write("{bad")
        r["json_sha256"] = _sha256_file(jp)
        r["json_bytes"] = os.path.getsize(jp)
        self._expect_reject(b)

    def test_original_json_imagepath_mismatch_rejected(self):
        b = PackFixture("fx-008-jsipath")
        rec = b.add_candidate((4, 4), [0] * 16)
        jp = os.path.join(ROOT, rec["json_path"])
        data = _read_json(jp)
        data["imagePath"] = "other.tif"
        with open(jp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        rec["json_sha256"] = _sha256_file(jp)
        rec["json_bytes"] = os.path.getsize(jp)
        self._expect_reject(b)

    def test_original_json_dim_mismatch_rejected(self):
        b = PackFixture("fx-008-jsidim")
        rec = b.add_candidate((4, 4), [0] * 16)
        jp = os.path.join(ROOT, rec["json_path"])
        data = _read_json(jp)
        data["imageWidth"] = 999
        with open(jp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        rec["json_sha256"] = _sha256_file(jp)
        rec["json_bytes"] = os.path.getsize(jp)
        self._expect_reject(b)

    def test_original_json_unknown_label_rejected(self):
        b = PackFixture("fx-008-jsilabel")
        rec = b.add_candidate((4, 4), [0] * 16, shapes=[
            {"label": "Ditch", "shape_type": "polygon",
             "points": [[0, 0], [1, 0], [1, 1]]}])
        # 重写一次源 JSON 以确保结构即上述内容（fixture 已按 shapes 写入）
        self._expect_reject(b)

    # ---- 2. review 副本 JSON 前置校验 ----
    def test_invalid_review_copy_json_no_output(self):
        b = PackFixture("fx-008-copyjson")
        b.add_review((8, 8), [{"type": "canvas_oob", "shape_index": 0, "label": "Field"}])
        r = b.records[-1]
        rp = os.path.join(b.input, r["review_json_path"])
        with open(rp, "w", encoding="utf-8") as f:
            f.write("{bad")
        r["review_json_sha256"] = _sha256_file(rp)
        self._expect_reject(b)

    def test_review_copy_bad_image_path_rejected(self):
        b = PackFixture("fx-008-copyimg")
        b.add_review((8, 8), [{"type": "canvas_oob", "shape_index": 0, "label": "Field"}],
                     copy_mutator=lambda d: d.__setitem__("imagePath", "review/x.png"))
        self._expect_reject(b)

    def test_review_copy_changed_points_rejected(self):
        b = PackFixture("fx-008-copypoints")

        def mut(d):
            d["shapes"][0]["points"] = [[0, 0], [1, 1], [2, 2]]

        b.add_review((8, 8), [{"type": "self_intersection", "shape_index": 0, "label": "Field"}],
                     copy_mutator=mut)
        self._expect_reject(b)

    # ---- 3. policy 校验 ----
    def test_policy_missing_rejected(self):
        b = PackFixture("fx-008-nopolicy")
        b.add_candidate((4, 4), [0] * 16)
        b.write_manifest(write_policy=False)
        out = b.out_dir()
        with self.assertRaises(MOD.BuildError):
            MOD.build_pack(b.input, out, root=ROOT)
        self.assertFalse(os.path.exists(out))

    def test_policy_class0_background_rejected(self):
        b = PackFixture("fx-008-pol0")
        b.add_candidate((4, 4), [0] * 16)
        b.policy["class_ids"]["0"] = "Background"
        self._expect_reject(b)

    def test_policy_255_training_class_rejected(self):
        b = PackFixture("fx-008-pol255")
        b.add_candidate((4, 4), [0] * 16)
        b.policy["class_ids"]["255"] = "Field"
        self._expect_reject(b)

    def test_policy_training_ready_rejected(self):
        b = PackFixture("fx-008-polready")
        b.add_candidate((4, 4), [0] * 16)
        b.policy["training_policy"]["not_training_ready"] = False
        self._expect_reject(b)

    # ---- 4. review_reasons 校验 ----
    def test_review_reason_bad_indices_rejected(self):
        for bad in (5, -1, "0", 1.0, True):
            b = PackFixture("fx-008-idx")
            b.add_review((8, 8), [{"type": "canvas_oob", "shape_index": bad, "label": "Field"}])
            self._expect_reject(b)

    def test_review_reason_unknown_type_rejected(self):
        b = PackFixture("fx-008-rtype")
        b.add_review((8, 8), [{"type": "weird", "shape_index": 0, "label": "Field"}])
        self._expect_reject(b)

    def test_review_reason_label_mismatch_rejected(self):
        b = PackFixture("fx-008-rlabel")
        b.add_review((8, 8), [{"type": "canvas_oob", "shape_index": 0, "label": "Road"}])
        self._expect_reject(b)

    def test_review_empty_reasons_rejected(self):
        b = PackFixture("fx-008-rempty")
        b.add_review((8, 8), [])
        self._expect_reject(b)

    # ---- 6. pixel_stats 类型校验 ----
    def test_pixel_stats_bool_rejected(self):
        b = PackFixture("fx-008-pbool")
        b.add_candidate((4, 4), [0] * 16,
                        pixel_stats_override={"class_pixels": {"0": True, "1": 0, "2": 0}})
        self._expect_reject(b)

    def test_pixel_stats_float_rejected(self):
        b = PackFixture("fx-008-pfloat")
        b.add_candidate((4, 4), [0] * 16,
                        pixel_stats_override={"class_pixels": {"0": 16.0, "1": 0, "2": 0}})
        self._expect_reject(b)

    # ---- 5. 同源/派生去重 ----
    def test_duplicate_source_reference_rejected(self):
        b = PackFixture("fx-008-dup")
        rec = b.add_candidate((4, 4), [0] * 16)
        dup = copy.deepcopy(rec)
        dup["sample_id"] = "dup_source_id"
        b.records.append(dup)
        self._expect_reject(b)

    # ---- 非项目 cwd ----
    def test_build_works_from_non_project_cwd(self):
        b = PackFixture("fx-008-cwd")
        b.add_candidate((4, 4), [0] * 16)
        b.write_manifest()
        out = b.out_dir()
        old = os.getcwd()
        try:
            os.chdir(LABELS)
            res = MOD.build_pack(b.input, out, root=ROOT)
        finally:
            os.chdir(old)
        self.assertEqual(res["counts"]["candidate"], 1)
        self.assertTrue(os.path.exists(os.path.join(out, "audit.json")))

    # ---- 可视：候选 255 透明 + 左原图/右叠加 ----
    def test_candidate_overlay_255_transparent(self):
        orig = Image.new("RGB", (4, 4), (10, 20, 30))
        mask = _mask_image((4, 4), [255, 255, 0, 0] + [1] * 4 + [2] * 4 + [0] * 4)
        over = MOD.render_candidate_overlay(orig, mask)
        self.assertEqual(over.getpixel((0, 0)), (10, 20, 30))   # 255 透明=原图
        self.assertNotEqual(over.getpixel((0, 2)), (10, 20, 30))  # 类 0 显示

    def test_candidate_preview_side_by_side(self):
        orig = Image.new("RGB", (4, 4), (10, 20, 30))
        mask = _mask_image((4, 4), [0] * 16)
        prev = MOD.render_candidate_preview(orig, mask)
        self.assertEqual(prev.size, (4 * 2 + 8, 4))

    # ---- review CSV/信息统计留空 ----
    def test_review_csv_stats_blank(self):
        import csv as _csv
        b = PackFixture("fx-008-csvblank")
        b.add_review((8, 8), [{"type": "canvas_oob", "shape_index": 0, "label": "Field"}])
        b.write_manifest()
        out = b.out_dir()
        MOD.build_pack(b.input, out, root=ROOT)
        with open(os.path.join(out, "records.csv"), "r", encoding="utf-8-sig", newline="") as f:
            rows = list(_csv.DictReader(f))
        rev = [r for r in rows if r["status"] == "review"][0]
        for key in ("class0_pixels", "class1_pixels", "class2_pixels",
                    "valid_pixels", "ignore_pixels", "ignore_ratio", "denominator"):
            self.assertEqual(rev[key], "", "review 行 %s 应为空" % key)


class TestFix009Boundaries(unittest.TestCase):
    """DATA-20261016-009 最终小修对应的独立拒绝/通过案例。"""

    def _expect_reject(self, b):
        b.write_manifest()
        out = b.out_dir()
        with self.assertRaises(MOD.BuildError):
            MOD.build_pack(b.input, out, root=ROOT)
        self.assertFalse(os.path.exists(out))

    # ---- 1. parse_tile_name：无 prefix 时 prefix 为空串且保留 row/col ----
    def test_parse_tile_name_empty_prefix_fields(self):
        p = MOD.parse_tile_name("tile_r54_c24.tif")
        self.assertEqual(p["prefix"], "")
        self.assertEqual(p["row"], "54")
        self.assertEqual(p["col"], "24")

    # ---- 2. policy 四类身份必须严格对应 ----
    def test_policy_class1_identity_wrong_rejected(self):
        b = PackFixture("fx-009-pol1")
        b.add_candidate((4, 4), [0] * 16)
        b.policy["class_ids"]["1"] = "Road 道路"          # 1 被改成 Road
        self._expect_reject(b)

    def test_policy_class2_identity_wrong_rejected(self):
        b = PackFixture("fx-009-pol2")
        b.add_candidate((4, 4), [0] * 16)
        b.policy["class_ids"]["2"] = "Field ridge 田埂"    # 2 被改成 Field ridge
        self._expect_reject(b)

    def test_policy_class1_concatenated_conflict_rejected(self):
        b = PackFixture("fx-009-polcat")
        b.add_candidate((4, 4), [0] * 16)
        b.policy["class_ids"]["1"] = "Field ridge Road 田埂道路"   # 拼接矛盾描述
        self._expect_reject(b)

    def test_policy_extra_training_class_rejected(self):
        b = PackFixture("fx-009-polextra")
        b.add_candidate((4, 4), [0] * 16)
        b.policy["class_ids"]["3"] = "Water 水体"           # 额外训练类
        self._expect_reject(b)

    def test_policy_valid_identity_accepted(self):
        # DEFAULT_POLICY（含中文说明）必须被接受。
        b = PackFixture("fx-009-polok")
        b.add_candidate((4, 4), [0] * 16)
        b.write_manifest()
        out = b.out_dir()
        MOD.build_pack(b.input, out, root=ROOT)
        self.assertTrue(os.path.exists(os.path.join(out, "audit.json")))

    def test_policy_english_only_identity_accepted(self):
        b = PackFixture("fx-009-poleng")
        b.add_candidate((4, 4), [0] * 16)
        b.policy["class_ids"] = {"0": "Field", "1": "Field ridge",
                                 "2": "Road", "255": "ignore"}
        b.write_manifest()
        out = b.out_dir()
        MOD.build_pack(b.input, out, root=ROOT)
        self.assertTrue(os.path.exists(os.path.join(out, "audit.json")))

    # ---- 3. 原 JSON imagePath 必须严格等于 TIFF basename ----
    def _mutate_source_imagepath(self, b, rec, value):
        jp = os.path.join(ROOT, rec["json_path"])
        data = _read_json(jp)
        data["imagePath"] = value
        with open(jp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        rec["json_sha256"] = _sha256_file(jp)
        rec["json_bytes"] = os.path.getsize(jp)

    def test_source_imagepath_foreign_prefix_rejected(self):
        b = PackFixture("fx-009-foreignpath")
        rec = b.add_candidate((4, 4), [0] * 16)
        base = os.path.basename(rec["image_path"])
        self._mutate_source_imagepath(b, rec, "foreign/" + base)
        self._expect_reject(b)

    def test_source_imagepath_backslash_prefix_rejected(self):
        b = PackFixture("fx-009-bspath")
        rec = b.add_candidate((4, 4), [0] * 16)
        base = os.path.basename(rec["image_path"])
        self._mutate_source_imagepath(b, rec, "sub\\" + base)
        self._expect_reject(b)

    def test_source_imagepath_absolute_rejected(self):
        b = PackFixture("fx-009-abspath")
        rec = b.add_candidate((4, 4), [0] * 16)
        self._mutate_source_imagepath(
            b, rec, os.path.abspath(os.path.join(ROOT, rec["image_path"])))
        self._expect_reject(b)

    # ---- 4. 候选 mask / review 副本必须是真正的 PNG ----
    def test_disguised_tiff_mask_rejected(self):
        b = PackFixture("fx-009-masktiff")
        rec = b.add_candidate((4, 4), [0] * 16)
        mp = os.path.join(b.input, rec["mask_path"])
        with Image.open(mp) as im:
            im.load()
            data = im.copy()
        data.save(mp, format="TIFF")        # 同一 .png 路径实为 TIFF
        rec["mask_sha256"] = _sha256_file(mp)
        self._expect_reject(b)

    def test_disguised_tiff_review_copy_rejected(self):
        b = PackFixture("fx-009-revtiff")
        rec = b.add_review((8, 8),
                           [{"type": "canvas_oob", "shape_index": 0, "label": "Field"}])
        rp = os.path.join(b.input, rec["review_png_path"])
        with Image.open(rp) as im:
            im.load()
            data = im.copy()
        data.save(rp, format="TIFF")
        rec["review_png_sha256"] = _sha256_file(rp)
        self._expect_reject(b)

    # ---- 5. 别名只能正向：原稿单向派生，副本不反向归一 ----
    def test_forward_alias_normalized_accepted(self):
        b = PackFixture("fx-009-fwdalias")
        shapes = [{"label": "Fieldridge", "shape_type": "polygon",
                   "points": [[0, 0], [7, 0], [7, 7]]}]

        def mut(d):
            d["shapes"][0]["label"] = "Field ridge"

        rec = b.add_review(
            (8, 8),
            [{"type": "self_intersection", "shape_index": 0, "label": "Fieldridge"}],
            shapes=shapes, copy_mutator=mut)
        b.write_manifest()
        out = b.out_dir()
        MOD.build_pack(b.input, out, root=ROOT)      # 正向映射应通过
        self.assertTrue(os.path.exists(os.path.join(out, "audit.json")))
        gal = _read_json(os.path.join(out, "gallery.json"))
        rev = [s for s in gal["samples"] if s["sample_id"] == rec["sample_id"]][0]
        self.assertEqual(rev["problems"][0]["shape_index"], 0)

    def test_reverse_alias_rewrite_rejected(self):
        b = PackFixture("fx-009-revalias")
        shapes = [{"label": "Field ridge", "shape_type": "polygon",
                   "points": [[0, 0], [7, 0], [7, 7]]}]

        def mut(d):
            d["shapes"][0]["label"] = "Fieldridge"    # 逆向改写

        b.add_review(
            (8, 8),
            [{"type": "canvas_oob", "shape_index": 0, "label": "Field ridge"}],
            shapes=shapes, copy_mutator=mut)
        self._expect_reject(b)

    def test_copy_non_label_change_rejected(self):
        b = PackFixture("fx-009-copyother")

        def mut(d):
            d["version"] = "9.9.9"                    # 非 label 字段被改

        b.add_review((8, 8),
                     [{"type": "canvas_oob", "shape_index": 0, "label": "Field"}],
                     copy_mutator=mut)
        self._expect_reject(b)

    # ---- _draw_contours：文字不受画布边缘截断且完整问题索引仍在 HTML ----
    def test_highlight_text_clamped_small_canvas(self):
        orig = Image.new("RGB", (6, 5), (10, 20, 30))
        shapes = [{"label": "Field", "shape_type": "polygon",
                   "points": [[5, 4], [3, 0], [5, 4]]}]
        prev = MOD.render_review_preview(orig, shapes, [{"shape_index": 0}])
        self.assertEqual(prev.size[1], 5)             # 画布高未改
        self.assertGreater(prev.size[0], 6)           # 并排仍成立

    def test_index_html_lists_problem_indices(self):
        b = PackFixture("fx-009-indexidx")
        shapes = [{"label": "Field", "shape_type": "polygon",
                   "points": [[0, 0], [7, 0], [7, 7]]},
                  {"label": "Field", "shape_type": "polygon",
                   "points": [[1, 1], [6, 1], [6, 6]]}]
        reasons = [{"type": "canvas_oob", "shape_index": 0, "label": "Field"},
                   {"type": "self_intersection", "shape_index": 1, "label": "Field"}]
        b.add_review((8, 8), reasons, shapes=shapes)
        b.write_manifest()
        out = b.out_dir()
        MOD.build_pack(b.input, out, root=ROOT)
        with open(os.path.join(out, "index.html"), "r", encoding="utf-8") as f:
            text = f.read()
        self.assertIn("shape_index=<code>0</code>", text)
        self.assertIn("shape_index=<code>1</code>", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
