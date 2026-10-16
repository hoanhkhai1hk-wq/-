# -*- coding: utf-8 -*-
"""DATA-20261016-004 候选准备工具的测试（由 Codex 运行；WorkBuddy 不执行）。

测试只写 `reports/labels/` 下的**全新 fixture 目录**与全新输出目录，不使用
`TemporaryDirectory` 自动删除；子进程显式 UTF-8、`-B`、`PYTHONDONTWRITEBYTECODE=1`，
不吞错误、不假装通过。fixture 图片用 Pillow 现场生成（小尺寸），不依赖原始 903 对数据。
"""

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import unittest
import uuid

from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
LABELS = os.path.join(ROOT, "reports", "labels")
SCRIPT = os.path.join(ROOT, "scripts", "labels", "prepare_candidate.py")


def _load_module():
    spec = importlib.util.spec_from_file_location("prepare_candidate_mod", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


MOD = _load_module()


def _rel(p):
    return os.path.relpath(p, ROOT).replace("\\", "/")


def _sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _new_dir(parent, prefix):
    while True:
        p = os.path.join(parent, "%s-%s" % (prefix, uuid.uuid4().hex[:12]))
        if not os.path.exists(p):
            os.makedirs(p)
            return p


def _shape(label, points, shape_type="polygon", group_id=None, flags=None):
    return {"label": label, "points": points, "shape_type": shape_type,
            "group_id": group_id, "flags": flags if flags is not None else {}}


class FixtureBuilder:
    """在 reports/labels 下创建全新 fixture 目录并返回索引路径。"""

    def __init__(self, name_prefix):
        self.dir = _new_dir(LABELS, name_prefix)
        self.records = []
        self.counter = 0

    def add(self, dataset, size, shapes, image_data="DUMMY", image_path_name=None,
            width=None, height=None, sha_override=None, bytes_override=None):
        self.counter += 1
        sid = uuid.uuid4().hex[:16]
        stem = "img_%02d" % self.counter
        tif = os.path.join(self.dir, stem + ".tif")
        im = Image.new("RGB", size, (12, 34, 56))
        im.save(tif, format="TIFF")

        jpath = os.path.join(self.dir, stem + ".json")
        data = {
            "version": "5.1.1", "flags": {}, "imagePath": image_path_name or (stem + ".tif"),
            "imageWidth": width if width is not None else size[0],
            "imageHeight": height if height is not None else size[1],
            "imageData": image_data, "shapes": shapes,
        }
        with open(jpath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

        rec = {
            "sample_id": sid, "dataset": dataset,
            "json_path": _rel(jpath), "json_sha256": sha_override or _sha256_file(jpath),
            "json_bytes": bytes_override or os.path.getsize(jpath),
            "image_path": _rel(tif), "image_sha256": _sha256_file(tif),
            "image_bytes": os.path.getsize(tif),
            "width": size[0], "height": size[1],
        }
        self.records.append(rec)
        return rec

    def write_index(self, records=None, extra=None):
        records = self.records if records is None else records
        idx = os.path.join(self.dir, "index.json")
        obj = {"schema_version": "candidate-input/1", "records": records}
        if extra:
            obj.update(extra)
        with open(idx, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False)
        return idx

    def out_dir(self, prefix="out"):
        return os.path.join(LABELS, "%s-%s" % (prefix, uuid.uuid4().hex[:12]))


def _run_cli(index, out, cwd):
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    return subprocess.run(
        [sys.executable, "-B", SCRIPT, "--input", index, "--output", out],
        cwd=cwd, env=env, capture_output=True, text=True, encoding="utf-8")


def _read_json(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _entry(manifest, sid):
    for e in manifest["records"]:
        if e["sample_id"] == sid:
            return e
    raise AssertionError("sample not found: %s" % sid)


class TestMasks(unittest.TestCase):
    def test_field_full_canvas_all_zero(self):
        b = FixtureBuilder("fx-field")
        rec = b.add("fixture", (4, 4),
                    [_shape("Field", [[0, 0], [4, 0], [4, 4], [0, 4]])])
        idx = b.write_index()
        out = b.out_dir()
        s = MOD.prepare(idx, out, root=ROOT)
        self.assertEqual(s["counts"]["candidate"], 1)
        m = Image.open(os.path.join(out, "masks", rec["sample_id"] + ".png"))
        vals = set(m.getdata())
        self.assertTrue(vals.issubset({0, 1, 2, 255}))
        self.assertEqual(vals, {0})
        e = _entry(_read_json(os.path.join(out, "manifest.json")), rec["sample_id"])
        st = e["pixel_stats"]
        self.assertEqual(st["class_pixels"]["0"], 16)
        self.assertEqual(st["unpainted_pixels"], 0)
        self.assertEqual(st["conflict_pixels"], 0)
        # 派生产物路径相对输出目录
        self.assertEqual(e["mask_path"], "masks/%s.png" % rec["sample_id"])
        self.assertTrue(os.path.exists(os.path.join(out, e["mask_path"])))

    def test_field_road_and_alias(self):
        b = FixtureBuilder("fx-mix")
        rec = b.add("fixture", (8, 8), [
            _shape("Field", [[0, 0], [4, 0], [4, 4], [0, 4]]),
            _shape("Road", [[4, 0], [8, 0], [8, 4], [4, 4]]),
            _shape("Fieldridge", [[0, 4], [8, 4], [8, 8], [0, 8]]),
        ])
        idx = b.write_index()
        out = b.out_dir()
        s = MOD.prepare(idx, out, root=ROOT)
        self.assertEqual(s["counts"]["candidate"], 1)
        self.assertEqual(s["alias_normalized_count"], 1)
        m = Image.open(os.path.join(out, "masks", rec["sample_id"] + ".png"))
        vals = set(m.getdata())
        self.assertTrue(vals.issubset({0, 1, 2, 255}))
        self.assertIn(0, vals)
        self.assertIn(1, vals)
        self.assertIn(2, vals)
        e = _entry(_read_json(os.path.join(out, "manifest.json")), rec["sample_id"])
        # Fieldridge 派生为 1，且记录别名归一
        self.assertTrue(any(a["from"] == "Fieldridge" and a["to"] == "Field ridge"
                            for a in e["alias_normalized"]))
        self.assertTrue(any(mp["original_label"] == "Fieldridge" and mp["derived_class"] == 1
                            for mp in e["label_mapping"]))

    def test_unpainted_is_255(self):
        b = FixtureBuilder("fx-unpainted")
        rec = b.add("fixture", (8, 8),
                    [_shape("Field", [[0, 0], [4, 0], [4, 4], [0, 4]])])
        idx = b.write_index()
        out = b.out_dir()
        MOD.prepare(idx, out, root=ROOT)
        e = _entry(_read_json(os.path.join(out, "manifest.json")), rec["sample_id"])
        st = e["pixel_stats"]
        self.assertGreater(st["unpainted_pixels"], 0)
        self.assertEqual(st["conflict_pixels"], 0)
        m = Image.open(os.path.join(out, "masks", rec["sample_id"] + ".png"))
        data = list(m.getdata())
        self.assertEqual(data.count(255), st["unpainted_pixels"])
        self.assertEqual(st["unpainted_pixels"] + st["class_pixels"]["0"], 64)

    def test_same_class_overlap_is_union(self):
        b = FixtureBuilder("fx-union")
        # 两个重叠的 Field 方块：同类取并集，冲突为 0。
        rec = b.add("fixture", (8, 8), [
            _shape("Field", [[0, 0], [5, 0], [5, 5], [0, 5]]),
            _shape("Field", [[3, 3], [8, 3], [8, 8], [3, 8]]),
        ])
        idx = b.write_index()
        out = b.out_dir()
        MOD.prepare(idx, out, root=ROOT)
        e = _entry(_read_json(os.path.join(out, "manifest.json")), rec["sample_id"])
        st = e["pixel_stats"]
        self.assertEqual(st["conflict_pixels"], 0)
        self.assertEqual(list(st["class_covered_pixels"].keys()), ["0", "1", "2"])
        m = Image.open(os.path.join(out, "masks", rec["sample_id"] + ".png"))
        self.assertTrue(set(m.getdata()).issubset({0, 255}))
        # 独立手算（Pillow 含边界、按 8×8 画布裁剪的像素口径）：
        #   A：行列 0..5 -> 6×6 = 36
        #   B：行列 3..8，右下边到 8 被画布裁剪为 3..7 -> 5×5 = 25
        #   重叠：行列 3..5 -> 3×3 = 9
        #   同类并集 = 36 + 25 - 9 = 52
        self.assertEqual(st["class_covered_pixels"]["0"], 52)

    def test_cross_class_overlap_255_order_independent(self):
        shapes_a = [
            _shape("Field", [[0, 0], [6, 0], [6, 6], [0, 6]]),
            _shape("Road", [[2, 2], [8, 2], [8, 8], [2, 8]]),
        ]
        shapes_b = [
            _shape("Road", [[2, 2], [8, 2], [8, 8], [2, 8]]),
            _shape("Field", [[0, 0], [6, 0], [6, 6], [0, 6]]),
        ]
        outs = []
        for i, shapes in enumerate((shapes_a, shapes_b)):
            b = FixtureBuilder("fx-order%d" % i)
            rec = b.add("fixture", (8, 8), shapes)
            idx = b.write_index()
            out = b.out_dir()
            MOD.prepare(idx, out, root=ROOT)
            e = _entry(_read_json(os.path.join(out, "manifest.json")), rec["sample_id"])
            self.assertEqual(e["status"], "candidate")
            self.assertGreater(e["pixel_stats"]["conflict_pixels"], 0)
            with open(os.path.join(out, "masks", rec["sample_id"] + ".png"), "rb") as f:
                outs.append(f.read())
        self.assertEqual(outs[0], outs[1])

    def test_full_canvas_two_class_overlap_is_empty_not_candidate(self):
        # 满幅两个类别完全叠加 -> 每个像素跨类冲突 -> 全 255；有效(互斥)像素为 0。
        b = FixtureBuilder("fx-allignore")
        rec = b.add("fixture", (8, 8), [
            _shape("Field", [[0, 0], [8, 0], [8, 8], [0, 8]]),
            _shape("Road", [[0, 0], [8, 0], [8, 8], [0, 8]]),
        ])
        idx = b.write_index()
        out = b.out_dir()
        s = MOD.prepare(idx, out, root=ROOT)
        self.assertEqual(s["counts"]["candidate"], 0)
        self.assertEqual(s["counts"]["empty"], 1)
        e = _entry(_read_json(os.path.join(out, "manifest.json")), rec["sample_id"])
        self.assertEqual(e["status"], "empty")
        self.assertTrue(e["not_training_ready"])
        st = e["pixel_stats"]
        self.assertEqual(sum(st["class_pixels"].values()), 0)
        self.assertEqual(st["conflict_pixels"], 64)
        # 诊断 mask 仍保留，且整幅为 255
        self.assertEqual(e["mask_path"], "masks/%s.png" % rec["sample_id"])
        m = Image.open(os.path.join(out, e["mask_path"]))
        self.assertEqual(set(m.getdata()), {255})

    def test_derived_paths_relative_to_output_dir(self):
        b = FixtureBuilder("fx-pathbase")
        rec = b.add("fixture", (8, 8),
                    [_shape("Field", [[0, 0], [6, 6], [6, 0], [0, 6]])])  # bowtie -> review
        idx = b.write_index()
        out = b.out_dir()
        MOD.prepare(idx, out, root=ROOT)
        man = _read_json(os.path.join(out, "manifest.json"))
        self.assertEqual(man["path_bases"]["source_paths"], "relative to project root")
        self.assertEqual(man["path_bases"]["derived_paths"], "relative to this output directory")
        pol = _read_json(os.path.join(out, "policy.json"))
        self.assertEqual(pol["path_bases"]["derived_paths"], "relative to this output directory")
        e = _entry(man, rec["sample_id"])
        self.assertEqual(e["status"], "review")
        # 源路径仍相对项目根
        self.assertEqual(e["json_path"], rec["json_path"])
        # 派生产物路径相对输出目录
        self.assertEqual(e["review_png_path"], "review/%s.png" % rec["sample_id"])
        self.assertEqual(e["review_json_path"], "review/%s.json" % rec["sample_id"])
        self.assertTrue(os.path.exists(os.path.join(out, e["review_png_path"])))
        self.assertTrue(os.path.exists(os.path.join(out, e["review_json_path"])))
        with open(os.path.join(out, "review.csv"), "r", encoding="utf-8-sig") as f:
            csv_text = f.read()
        self.assertIn("review/%s.png" % rec["sample_id"], csv_text)
        self.assertIn("review/%s.json" % rec["sample_id"], csv_text)

    def test_policy_closure_wording_unambiguous(self):
        b = FixtureBuilder("fx-closure")
        b.add("fixture", (8, 8), [_shape("Field", [[0, 0], [4, 0], [4, 4], [0, 4]])])
        idx = b.write_index()
        out = b.out_dir()
        MOD.prepare(idx, out, root=ROOT)
        pol = _read_json(os.path.join(out, "policy.json"))
        ch = pol["rasterization"]["coordinate_handling"]
        self.assertIn("不追加", ch)
        self.assertIn("隐式闭合", ch)
        self.assertNotIn("不闭合", ch)


class TestGeometry(unittest.TestCase):
    def _status(self, size, shapes):
        b = FixtureBuilder("fx-geom")
        rec = b.add("fixture", size, shapes)
        idx = b.write_index()
        out = b.out_dir()
        MOD.prepare(idx, out, root=ROOT)
        e = _entry(_read_json(os.path.join(out, "manifest.json")), rec["sample_id"])
        return e, out, rec

    def test_bowtie_both_directions_review(self):
        b1 = FixtureBuilder("fx-bow1")
        rec1 = b1.add("fixture", (8, 8),
                      [_shape("Field", [[0, 0], [6, 6], [6, 0], [0, 6]])])
        idx1 = b1.write_index()
        out1 = b1.out_dir()
        MOD.prepare(idx1, out1, root=ROOT)
        e1 = _entry(_read_json(os.path.join(out1, "manifest.json")), rec1["sample_id"])
        self.assertEqual(e1["status"], "review")
        self.assertTrue(any(r["type"] == "self_intersection" for r in e1["review_reasons"]))
        # 反向 bowtie
        b2 = FixtureBuilder("fx-bow2")
        rec2 = b2.add("fixture", (8, 8),
                      [_shape("Field", [[0, 6], [6, 0], [6, 6], [0, 0]])])
        idx2 = b2.write_index()
        out2 = b2.out_dir()
        MOD.prepare(idx2, out2, root=ROOT)
        e2 = _entry(_read_json(os.path.join(out2, "manifest.json")), rec2["sample_id"])
        self.assertEqual(e2["status"], "review")

    def test_simple_polygon_not_review(self):
        e, out, rec = self._status((8, 8), [_shape("Field", [[0, 0], [6, 0], [6, 6], [0, 6]])])
        self.assertEqual(e["status"], "candidate")
        self.assertEqual(e["review_reasons"], [])

    def test_canvas_oob_review(self):
        e, out, rec = self._status((8, 8),
                                   [_shape("Field", [[0, 0], [6, 0], [6, 8.5], [0, 6]])])
        self.assertEqual(e["status"], "review")
        self.assertTrue(any(r["type"] == "canvas_oob" for r in e["review_reasons"]))

    def test_float_noise_within_eps_not_oob(self):
        # 越界 1e-7 在 EPS 容差内，不应判越界
        e, out, rec = self._status((8, 8),
                                   [_shape("Field", [[0, 0], [6, 0], [6, 8 + 1e-7], [0, 6]])])
        self.assertEqual(e["status"], "candidate")

    def test_fewer_than_3_distinct_review(self):
        e, out, rec = self._status((8, 8), [_shape("Field", [[1, 1], [1, 1], [3, 3]])])
        self.assertEqual(e["status"], "review")
        self.assertTrue(any(r["type"] == "fewer_than_3_distinct_points"
                            for r in e["review_reasons"]))

    def test_review_whole_file_has_no_mask(self):
        b = FixtureBuilder("fx-whole")
        rec = b.add("fixture", (8, 8), [
            _shape("Field", [[0, 0], [6, 0], [6, 6], [0, 6]]),   # 正常
            _shape("Road", [[0, 0], [6, 6], [6, 0], [0, 6]]),    # bowtie
        ])
        idx = b.write_index()
        out = b.out_dir()
        MOD.prepare(idx, out, root=ROOT)
        e = _entry(_read_json(os.path.join(out, "manifest.json")), rec["sample_id"])
        self.assertEqual(e["status"], "review")
        self.assertIsNone(e["mask_path"])
        self.assertIsNotNone(e["review_png_path"])
        self.assertIsNotNone(e["review_json_path"])
        self.assertFalse(os.path.exists(os.path.join(out, "masks", rec["sample_id"] + ".png")))
        self.assertTrue(os.path.exists(os.path.join(out, "review", rec["sample_id"] + ".png")))
        self.assertTrue(os.path.exists(os.path.join(out, "review", rec["sample_id"] + ".json")))


class TestReviewCopies(unittest.TestCase):
    def test_png_fidelity_json_points_and_original_unchanged(self):
        b = FixtureBuilder("fx-copy")
        pts = [[0, 0], [6, 6], [6, 0], [0, 6]]
        rec = b.add("fixture", (8, 8), [
            _shape("Field", pts),
            _shape("Fieldridge", [[0, 0], [1, 0], [1, 1], [0, 1]], group_id=7,
                   flags={"occluded": True}),
        ])
        idx = b.write_index()
        out = b.out_dir()

        j_sha_before = _sha256_file(os.path.join(ROOT, rec["json_path"]))
        t_sha_before = _sha256_file(os.path.join(ROOT, rec["image_path"]))

        MOD.prepare(idx, out, root=ROOT)

        # 原文件未被改动
        self.assertEqual(j_sha_before, _sha256_file(os.path.join(ROOT, rec["json_path"])))
        self.assertEqual(t_sha_before, _sha256_file(os.path.join(ROOT, rec["image_path"])))

        rpng = os.path.join(out, "review", rec["sample_id"] + ".png")
        rjson = os.path.join(out, "review", rec["sample_id"] + ".json")
        # PNG 与原 TIFF 逐像素一致
        with Image.open(os.path.join(ROOT, rec["image_path"])) as src:
            src.load()
            src_bytes = src.tobytes()
        with Image.open(rpng) as cp:
            cp.load()
            self.assertEqual(cp.mode, "RGB")
            self.assertEqual(cp.tobytes(), src_bytes)

        copy = _read_json(rjson)
        self.assertEqual(copy["imagePath"], rec["sample_id"] + ".png")
        self.assertIsNone(copy["imageData"])
        self.assertEqual(copy["version"], "5.1.1")
        # shape 顺序、points、group_id、flags 均保留
        self.assertEqual([s["points"] for s in copy["shapes"]],
                         [s["points"] for s in _read_json(os.path.join(ROOT, rec["json_path"]))["shapes"]])
        self.assertEqual(copy["shapes"][1]["group_id"], 7)
        self.assertEqual(copy["shapes"][1]["flags"], {"occluded": True})
        # Fieldridge 拼写归一并逐 shape 记录
        self.assertEqual(copy["shapes"][1]["label"], "Field ridge")
        e = _entry(_read_json(os.path.join(out, "manifest.json")), rec["sample_id"])
        self.assertTrue(any(c["shape_index"] == 1 and c["from"] == "Fieldridge"
                            and c["to"] == "Field ridge" for c in e["derived_json_changes"]))

    def test_review_csv_bom_and_rows(self):
        b = FixtureBuilder("fx-csv")
        rec = b.add("fixture", (8, 8), [_shape("Field", [[0, 0], [6, 6], [6, 0], [0, 6]])])
        idx = b.write_index()
        out = b.out_dir()
        MOD.prepare(idx, out, root=ROOT)
        with open(os.path.join(out, "review.csv"), "rb") as f:
            head = f.read(3)
        self.assertEqual(head, b"\xef\xbb\xbf")
        with open(os.path.join(out, "review.csv"), "r", encoding="utf-8-sig") as f:
            text = f.read()
        self.assertIn("source_id", text)
        self.assertIn("pending_semantic_review", text)
        self.assertIn(rec["sample_id"], text)


class TestUnsupported(unittest.TestCase):
    def test_unknown_label_excluded_not_background(self):
        b = FixtureBuilder("fx-unknown")
        rec = b.add("fixture", (8, 8), [
            _shape("Field", [[0, 0], [4, 0], [4, 4], [0, 4]]),
            _shape("Ditch", [[4, 4], [8, 4], [8, 8], [4, 8]]),
        ])
        idx = b.write_index()
        out = b.out_dir()
        s = MOD.prepare(idx, out, root=ROOT)
        self.assertEqual(s["counts"]["unsupported"], 1)
        self.assertEqual(s["counts"]["candidate"], 0)
        e = _entry(_read_json(os.path.join(out, "manifest.json")), rec["sample_id"])
        self.assertEqual(e["status"], "unsupported")
        self.assertIsNone(e["mask_path"])
        self.assertFalse(os.path.exists(os.path.join(out, "masks", rec["sample_id"] + ".png")))
        self.assertEqual(s["unknown_label_count"], 1)

    def test_non_polygon_unsupported(self):
        b = FixtureBuilder("fx-nonpoly")
        rec = b.add("fixture", (8, 8), [_shape("Field", [[0, 0], [4, 4]], shape_type="rectangle")])
        idx = b.write_index()
        out = b.out_dir()
        MOD.prepare(idx, out, root=ROOT)
        e = _entry(_read_json(os.path.join(out, "manifest.json")), rec["sample_id"])
        self.assertEqual(e["status"], "unsupported")


class TestInputValidation(unittest.TestCase):
    def test_bad_sha_rejected(self):
        b = FixtureBuilder("fx-badsha")
        b.add("fixture", (8, 8), [_shape("Field", [[0, 0], [4, 0], [4, 4], [0, 4]])],
              sha_override="0" * 64)
        idx = b.write_index()
        out = b.out_dir()
        with self.assertRaises(MOD.PrepareError):
            MOD.prepare(idx, out, root=ROOT)

    def test_escape_path_rejected(self):
        b = FixtureBuilder("fx-escape")
        rec = b.add("fixture", (8, 8), [_shape("Field", [[0, 0], [4, 0], [4, 4], [0, 4]])])
        rec["json_path"] = "../evil.json"
        idx = b.write_index()
        out = b.out_dir()
        with self.assertRaises(MOD.PrepareError):
            MOD.prepare(idx, out, root=ROOT)

    def test_absolute_path_rejected(self):
        b = FixtureBuilder("fx-abs")
        rec = b.add("fixture", (8, 8), [_shape("Field", [[0, 0], [4, 0], [4, 4], [0, 4]])])
        rec["image_path"] = os.path.join(ROOT, "邓", "x.tif")
        idx = b.write_index()
        out = b.out_dir()
        with self.assertRaises(MOD.PrepareError):
            MOD.prepare(idx, out, root=ROOT)

    def test_duplicate_sample_id_rejected(self):
        b = FixtureBuilder("fx-dupid")
        r1 = b.add("fixture", (8, 8), [_shape("Field", [[0, 0], [4, 0], [4, 4], [0, 4]])])
        r2 = b.add("fixture", (8, 8), [_shape("Field", [[0, 0], [4, 0], [4, 4], [0, 4]])])
        r2["sample_id"] = r1["sample_id"]
        idx = b.write_index()
        out = b.out_dir()
        with self.assertRaises(MOD.PrepareError):
            MOD.prepare(idx, out, root=ROOT)

    def test_bool_coordinate_rejected(self):
        b = FixtureBuilder("fx-bool")
        b.add("fixture", (8, 8), [_shape("Field", [[True, 0], [4, 0], [4, 4]])])
        idx = b.write_index()
        out = b.out_dir()
        with self.assertRaises(MOD.PrepareError):
            MOD.prepare(idx, out, root=ROOT)

    def test_non_finite_coordinate_rejected(self):
        b = FixtureBuilder("fx-nonfinite")
        b.add("fixture", (8, 8), [_shape("Field", [[0, 0], [4, 0], [float("nan"), 4]])])
        idx = b.write_index()
        out = b.out_dir()
        with self.assertRaises(MOD.PrepareError):
            MOD.prepare(idx, out, root=ROOT)

    def test_imagepath_stem_mismatch_rejected(self):
        b = FixtureBuilder("fx-stem")
        b.add("fixture", (8, 8), [_shape("Field", [[0, 0], [4, 0], [4, 4], [0, 4]])],
              image_path_name="other.tif")
        idx = b.write_index()
        out = b.out_dir()
        with self.assertRaises(MOD.PrepareError):
            MOD.prepare(idx, out, root=ROOT)

    def test_bad_schema_rejected(self):
        b = FixtureBuilder("fx-schema")
        b.add("fixture", (8, 8), [_shape("Field", [[0, 0], [4, 0], [4, 4], [0, 4]])])
        idx = b.write_index(extra={"schema_version": "other"})
        out = b.out_dir()
        with self.assertRaises(MOD.PrepareError):
            MOD.prepare(idx, out, root=ROOT)

    def test_existing_output_refused_and_originals_intact(self):
        b = FixtureBuilder("fx-exist")
        rec = b.add("fixture", (8, 8), [_shape("Field", [[0, 0], [4, 0], [4, 4], [0, 4]])])
        idx = b.write_index()
        out = b.out_dir()
        os.makedirs(out)
        sentinel = os.path.join(out, "keep.txt")
        with open(sentinel, "w", encoding="utf-8") as f:
            f.write("keep")
        with self.assertRaises(MOD.PrepareError):
            MOD.prepare(idx, out, root=ROOT)
        self.assertTrue(os.path.exists(sentinel))
        with open(os.path.join(ROOT, rec["json_path"]), "rb") as f:
            self.assertTrue(f.read())
        self.assertTrue(os.path.exists(os.path.join(ROOT, rec["image_path"])))

    def test_sample_id_validator_pure(self):
        # 纯校验：危险 id 直接拒绝，不触盘。
        for bad in ["", "../../escape", "a/b", "a\\b", "a.b", "con", "NUL", "PRN",
                    "COM1", "lpt9", "x" * 81, " lead", "trail ", "a b"]:
            with self.assertRaises(MOD.PrepareError):
                MOD._validate_sample_id(bad)
        for good in ["a", "A1_-", "0" * 80, uuid.uuid4().hex]:
            MOD._validate_sample_id(good)  # 不应抛错

    def test_unsafe_sample_id_rejected_no_escape(self):
        b = FixtureBuilder("fx-unsafeid")
        rec = b.add("fixture", (8, 8), [_shape("Field", [[0, 0], [4, 0], [4, 4], [0, 4]])])
        rec["sample_id"] = "../../escape"
        idx = b.write_index()
        out = b.out_dir()
        with self.assertRaises(MOD.PrepareError):
            MOD.prepare(idx, out, root=ROOT)
        # 拒绝发生在创建输出目录之前，且未向 fixture 外逃逸写文件。
        self.assertFalse(os.path.exists(out))
        self.assertFalse(os.path.exists(os.path.join(LABELS, "escape.png")))
        self.assertFalse(os.path.exists(os.path.join(os.path.dirname(LABELS), "escape.json")))

    def test_windows_reserved_sample_id_rejected(self):
        b = FixtureBuilder("fx-reserved")
        rec = b.add("fixture", (8, 8), [_shape("Field", [[0, 0], [4, 0], [4, 4], [0, 4]])])
        rec["sample_id"] = "con"
        idx = b.write_index()
        out = b.out_dir()
        with self.assertRaises(MOD.PrepareError):
            MOD.prepare(idx, out, root=ROOT)
        self.assertFalse(os.path.exists(out))

    def test_case_insensitive_duplicate_id_rejected(self):
        b = FixtureBuilder("fx-dupcase")
        r1 = b.add("fixture", (8, 8), [_shape("Field", [[0, 0], [4, 0], [4, 4], [0, 4]])])
        r2 = b.add("fixture", (8, 8), [_shape("Field", [[0, 0], [4, 0], [4, 4], [0, 4]])])
        r1["sample_id"] = "Abc123"
        r2["sample_id"] = "abc123"
        idx = b.write_index()
        out = b.out_dir()
        with self.assertRaises(MOD.PrepareError):
            MOD.prepare(idx, out, root=ROOT)
        self.assertFalse(os.path.exists(out))

    def test_duplicate_source_reference_rejected(self):
        b = FixtureBuilder("fx-dupsrc")
        r1 = b.add("fixture", (8, 8), [_shape("Field", [[0, 0], [4, 0], [4, 4], [0, 4]])])
        r2 = b.add("fixture", (8, 8), [_shape("Field", [[0, 0], [4, 0], [4, 4], [0, 4]])])
        # 不同写法、resolve+normcase 后指向同一 JSON。
        r2["json_path"] = "./" + r1["json_path"]
        idx = b.write_index()
        out = b.out_dir()
        with self.assertRaises(MOD.PrepareError):
            MOD.prepare(idx, out, root=ROOT)
        self.assertFalse(os.path.exists(out))

    def test_imagepath_same_stem_diff_ext_rejected(self):
        b = FixtureBuilder("fx-stemext")
        b.add("fixture", (8, 8), [_shape("Field", [[0, 0], [4, 0], [4, 4], [0, 4]])],
              image_path_name="img_01.png")  # 与 img_01.tif 同 stem 异后缀
        idx = b.write_index()
        out = b.out_dir()
        with self.assertRaises(MOD.PrepareError):
            MOD.prepare(idx, out, root=ROOT)

    def test_imagepath_foreign_prefix_rejected(self):
        b = FixtureBuilder("fx-stemprefix")
        b.add("fixture", (8, 8), [_shape("Field", [[0, 0], [4, 0], [4, 4], [0, 4]])],
              image_path_name="sub/img_01.tif")  # basename 相同但带外部路径前缀
        idx = b.write_index()
        out = b.out_dir()
        with self.assertRaises(MOD.PrepareError):
            MOD.prepare(idx, out, root=ROOT)

    def test_json_width_bool_rejected(self):
        b = FixtureBuilder("fx-wbool")
        b.add("fixture", (8, 8), [_shape("Field", [[0, 0], [4, 0], [4, 4], [0, 4]])],
              width=True)
        idx = b.write_index()
        out = b.out_dir()
        with self.assertRaises(MOD.PrepareError):
            MOD.prepare(idx, out, root=ROOT)

    def test_json_width_nonpositive_rejected(self):
        b = FixtureBuilder("fx-wzero")
        b.add("fixture", (8, 8), [_shape("Field", [[0, 0], [4, 0], [4, 4], [0, 4]])],
              width=0)
        idx = b.write_index()
        out = b.out_dir()
        with self.assertRaises(MOD.PrepareError):
            MOD.prepare(idx, out, root=ROOT)

    def test_bad_last_source_no_output_dir(self):
        # 完整校验必须在创建输出目录之前完成：最后一个来源坏 SHA 时不得留下半成品目录。
        b = FixtureBuilder("fx-lastbad")
        b.add("fixture", (8, 8), [_shape("Field", [[0, 0], [4, 0], [4, 4], [0, 4]])])
        b.add("fixture", (8, 8), [_shape("Field", [[0, 0], [4, 0], [4, 4], [0, 4]])],
              sha_override="2" * 64)
        idx = b.write_index()
        out = b.out_dir()
        with self.assertRaises(MOD.PrepareError):
            MOD.prepare(idx, out, root=ROOT)
        self.assertFalse(os.path.exists(out))


class TestCliAndCwd(unittest.TestCase):
    def _fixture(self):
        b = FixtureBuilder("fx-cli")
        b.add("fixture", (8, 8), [_shape("Field", [[0, 0], [4, 0], [4, 4], [0, 4]])])
        return b, b.write_index()

    def test_cli_success_from_non_project_cwd(self):
        b, idx = self._fixture()
        out = b.out_dir()
        # cwd 既不是项目根，也不是脚本目录
        res = _run_cli(idx, out, cwd=b.dir)
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertTrue(os.path.exists(os.path.join(out, "summary.json")))
        self.assertTrue(os.path.exists(os.path.join(out, "manifest.json")))
        self.assertTrue(os.path.exists(os.path.join(out, "policy.json")))

    def test_cli_nonzero_on_bad_hash(self):
        b = FixtureBuilder("fx-cli-bad")
        b.add("fixture", (8, 8), [_shape("Field", [[0, 0], [4, 0], [4, 4], [0, 4]])],
              sha_override="1" * 64)
        idx = b.write_index()
        out = b.out_dir()
        res = _run_cli(idx, out, cwd=b.dir)
        self.assertNotEqual(res.returncode, 0)

    def test_cli_nonzero_on_existing_output(self):
        b, idx = self._fixture()
        out = b.out_dir()
        os.makedirs(out)
        res = _run_cli(idx, out, cwd=b.dir)
        self.assertNotEqual(res.returncode, 0)

    def test_cli_nonzero_on_escape(self):
        b = FixtureBuilder("fx-cli-esc")
        rec = b.add("fixture", (8, 8), [_shape("Field", [[0, 0], [4, 0], [4, 4], [0, 4]])])
        rec["json_path"] = "../evil.json"
        idx = b.write_index()
        out = b.out_dir()
        res = _run_cli(idx, out, cwd=b.dir)
        self.assertNotEqual(res.returncode, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
