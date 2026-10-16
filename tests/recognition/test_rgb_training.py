"""Tests for the RGB input adaptation and synthetic training drill.

Run with the existing CPU interpreter::

    .venv-recognition/Scripts/python.exe -B -m unittest tests.recognition.test_rgb_training -v

Fixtures are written into fresh, kept subdirectories under ``reports/rgb`` only;
no real data is touched.  These are software-rule tests: they never claim that
synthetic results represent real agricultural accuracy.
"""

import hashlib
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
import unittest
import uuid

import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
SCRIPT = os.path.join(ROOT, "scripts", "rgb", "train_smoke.py")
FIXTURE_ROOT = os.path.join(ROOT, "reports", "rgb", "TRN-20261016-001-tests")

_spec = importlib.util.spec_from_file_location("train_smoke_under_test", SCRIPT)
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)

RGBError = mod.RGBError


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _rel(path, base=ROOT):
    return os.path.relpath(path, base).replace("\\", "/")


def _cli_env():
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def _run_cli(args, cwd):
    """Run the CLI with explicit UTF-8 strict decoding (no GBK fallback)."""
    return subprocess.run(
        [sys.executable, "-B", SCRIPT, *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="strict",
        cwd=cwd,
        env=_cli_env(),
    )


def _new_dir(name):
    os.makedirs(FIXTURE_ROOT, exist_ok=True)
    d = os.path.join(FIXTURE_ROOT, f"{name}-{uuid.uuid4().hex[:8]}")
    os.makedirs(d)
    return d


def _default_policy():
    return {
        "task_id": "TEST",
        "class_ids": {
            "0": "Field 田块",
            "1": "Field ridge 田埂",
            "2": "Road 道路",
            "255": "ignore 忽略/未画/冲突/未确认",
        },
        "no_background_class": True,
        "alias_assumption": {
            "map": {"Fieldridge": "Field ridge"},
            "status": "pending_semantic_review",
            "note": "pending",
        },
        "training_policy": {
            "semantic_reviewed": False,
            "not_training_ready": True,
            "no_train_test_split": True,
        },
    }


def _default_rgb(h=8, w=8, shift=0):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[..., 0] = 10 + shift
    img[..., 1] = 20 + shift
    img[..., 2] = 30 + shift
    img[:, : w // 2] = (200, 50, 50)
    return img


def _default_mask(h=8, w=8):
    mask = np.full((h, w), 255, dtype=np.uint8)
    mask[1:4, 1:4] = 0
    mask[4:7, 1:4] = 1
    mask[1:4, 4:7] = 2
    return mask


def make_package(name, specs, policy=None):
    """Build a DATA-005-style candidate package under reports/rgb."""
    base = _new_dir(name)
    src = os.path.join(base, "src")
    masks = os.path.join(base, "masks")
    os.makedirs(src)
    os.makedirs(masks)
    records = []
    for spec in specs:
        sid = spec["id"]
        status = spec.get("status", "candidate")
        rec = {"sample_id": sid, "status": status}
        if status != "candidate":
            records.append(rec)
            continue
        rgb = spec.get("rgb", _default_rgb())
        h, w = rgb.shape[0], rgb.shape[1]
        mask = spec.get("mask", _default_mask(h, w))
        ip = os.path.join(src, f"{sid}.png")
        mp = os.path.join(masks, f"{sid}.png")
        jp = os.path.join(src, f"{sid}.json")
        Image.fromarray(rgb).save(ip, format="PNG")
        Image.fromarray(mask).save(mp, format="PNG")
        with open(jp, "w", encoding="utf-8") as f:
            json.dump({"imagePath": f"{sid}.png"}, f)
        rec.update({
            "json_path": _rel(jp),
            "image_path": _rel(ip),
            "json_sha256": _sha(jp),
            "json_bytes": os.path.getsize(jp),
            "image_sha256": _sha(ip),
            "image_bytes": os.path.getsize(ip),
            "width": w,
            "height": h,
            "mask_path": _rel(mp, base),
            "mask_sha256": _sha(mp),
            "review_png_path": None,
            "semantic_reviewed": False,
            "not_training_ready": True,
        })
        rec.update(spec.get("overrides", {}))
        records.append(rec)

    with open(os.path.join(base, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump({
            "task_id": "TEST",
            "input_index": "test",
            "created": "2026-10-16T00:00:00",
            "path_bases": {"source_paths": "relative to project root",
                           "derived_paths": "relative to this output directory"},
            "records": records,
        }, f, ensure_ascii=False, indent=2)
    with open(os.path.join(base, "policy.json"), "w", encoding="utf-8") as f:
        json.dump(policy or _default_policy(), f, ensure_ascii=False, indent=2)
    return base


# --------------------------------------------------------------------------- #
class TestDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = make_package("dataset", [
            {"id": "aaa111", "rgb": _default_rgb(8, 8, 0)},
            {"id": "bbb222", "rgb": _default_rgb(8, 8, 5)},
            {"id": "rev333", "status": "review"},
        ])

    def test_len_excludes_review_not_hardcoded(self):
        ds = mod.RGBInspectionDataset(input_dir=self.base, root=ROOT)
        self.assertEqual(len(ds), 2)
        self.assertEqual(ds.excluded_review_count, 1)
        self.assertEqual({r["status"] for r in ds.records}, {"candidate"})

    def test_item_shapes_dtypes_values(self):
        ds = mod.RGBInspectionDataset(input_dir=self.base, root=ROOT)
        item = ds[0]
        self.assertEqual(item["image"].shape, (3, 8, 8))
        self.assertEqual(item["image"].dtype, torch.float32)
        self.assertEqual(item["target"].shape, (8, 8))
        self.assertEqual(item["target"].dtype, torch.int64)
        self.assertTrue(set(int(v) for v in torch.unique(item["target"]).tolist()).issubset({0, 1, 2, 255}))
        self.assertEqual(item["status"], "candidate")
        self.assertIs(item["semantic_reviewed"], False)
        self.assertIs(item["training_ready"], False)

    def test_channel_and_value_fidelity(self):
        ds = mod.RGBInspectionDataset(input_dir=self.base, root=ROOT)
        item = ds[0]
        image = item["image"]
        # left half is exactly (200, 50, 50) -> /255 on the correct channels
        self.assertAlmostEqual(float(image[0, 0, 0]), 200 / 255, places=6)
        self.assertAlmostEqual(float(image[1, 0, 0]), 50 / 255, places=6)
        self.assertAlmostEqual(float(image[2, 0, 0]), 50 / 255, places=6)
        # right half keeps the base colour (10, 20, 30)
        self.assertAlmostEqual(float(image[0, 0, 7]), 10 / 255, places=6)
        self.assertAlmostEqual(float(image[1, 0, 7]), 20 / 255, places=6)
        self.assertAlmostEqual(float(image[2, 0, 7]), 30 / 255, places=6)

    def test_zero_and_255_preserved(self):
        ds = mod.RGBInspectionDataset(input_dir=self.base, root=ROOT)
        target = ds[0]["target"]
        self.assertTrue(bool((target == 0).any()), "class 0 must be preserved")
        self.assertTrue(bool((target == 255).any()), "ignore 255 must be preserved")


# --------------------------------------------------------------------------- #
class TestLoss(unittest.TestCase):
    def test_equal_logits_ce_is_log_three(self):
        logits = torch.zeros(1, 3, 4, 4)
        target = torch.zeros(1, 4, 4, dtype=torch.int64)
        loss = mod.segmentation_loss(logits, target)
        self.assertAlmostEqual(float(loss), math.log(3.0), places=6)

    def test_ignore_pixel_zero_gradient(self):
        logits = torch.zeros(1, 3, 4, 4, requires_grad=True)
        target = torch.zeros(1, 4, 4, dtype=torch.int64)
        target[0, 0, 0] = 255
        loss = mod.segmentation_loss(logits, target)
        loss.backward()
        grad = logits.grad
        self.assertTrue(bool(torch.allclose(grad[0, :, 0, 0], torch.zeros(3))),
                        "ignored pixel must have zero gradient")
        self.assertFalse(bool(torch.allclose(grad[0, :, 1, 1], torch.zeros(3))),
                         "valid pixel must have non-zero gradient")

    def test_all_ignore_raises(self):
        logits = torch.zeros(1, 3, 4, 4)
        target = torch.full((1, 4, 4), 255, dtype=torch.int64)
        with self.assertRaises(RGBError):
            mod.segmentation_loss(logits, target)

    def test_illegal_values_raise(self):
        logits = torch.zeros(1, 3, 4, 4)
        target = torch.zeros(1, 4, 4, dtype=torch.int64)
        target[0, 0, 0] = 3
        with self.assertRaises(RGBError):
            mod.segmentation_loss(logits, target)

    def test_shape_and_dtype_raise(self):
        logits = torch.zeros(1, 3, 4, 4)
        with self.assertRaises(RGBError):
            mod.segmentation_loss(logits, torch.zeros(1, 4, 4, dtype=torch.int32))
        with self.assertRaises(RGBError):
            mod.segmentation_loss(torch.zeros(1, 2, 4, 4), torch.zeros(1, 4, 4, dtype=torch.int64))
        with self.assertRaises(RGBError):
            mod.segmentation_loss(torch.zeros(1, 3, 4, 4), torch.zeros(1, 4, 3, dtype=torch.int64))


# --------------------------------------------------------------------------- #
class TestRejections(unittest.TestCase):
    def test_bad_mask_sha(self):
        base = make_package("badmask", [{"id": "c1", "overrides": {"mask_sha256": "0" * 64}}])
        ds = mod.RGBInspectionDataset(input_dir=base, root=ROOT)
        with self.assertRaises(RGBError):
            ds[0]

    def test_bad_image_sha(self):
        base = make_package("badimg", [{"id": "c1", "overrides": {"image_sha256": "0" * 64}}])
        ds = mod.RGBInspectionDataset(input_dir=base, root=ROOT)
        with self.assertRaises(RGBError):
            ds[0]

    def test_size_mismatch(self):
        base = make_package("badsize", [{"id": "c1", "overrides": {"width": 99}}])
        ds = mod.RGBInspectionDataset(input_dir=base, root=ROOT)
        with self.assertRaises(RGBError):
            ds[0]

    def test_path_escape_rejected(self):
        base = make_package("escape", [{"id": "c1", "overrides": {"image_path": "../evil.png"}}])
        with self.assertRaises(RGBError):
            mod.RGBInspectionDataset(input_dir=base, root=ROOT)

    def test_duplicate_id_rejected(self):
        base = make_package("dup", [{"id": "same"}, {"id": "same"}])
        with self.assertRaises(RGBError):
            mod.RGBInspectionDataset(input_dir=base, root=ROOT)

    def test_bad_status_rejected(self):
        base = make_package("badstatus", [{"id": "c1", "status": "empty"}])
        with self.assertRaises(RGBError):
            mod.RGBInspectionDataset(input_dir=base, root=ROOT)

    def test_all_ignore_mask_rejected(self):
        mask = np.full((8, 8), 255, dtype=np.uint8)
        base = make_package("allignore", [{"id": "c1", "mask": mask}])
        ds = mod.RGBInspectionDataset(input_dir=base, root=ROOT)
        with self.assertRaises(RGBError):
            ds[0]

    def test_float_width_rejected(self):
        base = make_package("floatwidth", [{"id": "c1", "overrides": {"width": 3.0}}])
        with self.assertRaises(RGBError):
            mod.RGBInspectionDataset(input_dir=base, root=ROOT)

    def test_bool_dimension_rejected(self):
        base = make_package("booldim", [{"id": "c1", "overrides": {"height": True}}])
        with self.assertRaises(RGBError):
            mod.RGBInspectionDataset(input_dir=base, root=ROOT)

    def test_bool_bytes_rejected(self):
        base = make_package("boolbytes", [{"id": "c1", "overrides": {"image_bytes": True}}])
        with self.assertRaises(RGBError):
            mod.RGBInspectionDataset(input_dir=base, root=ROOT)

    def test_bad_sha_format_rejected(self):
        base = make_package("badshafmt", [{"id": "c1", "overrides": {"image_sha256": "nope"}}])
        with self.assertRaises(RGBError):
            mod.RGBInspectionDataset(input_dir=base, root=ROOT)

    def test_duplicate_source_json_rejected(self):
        base = make_package("dupjson", [{"id": "c1"}])
        mpath = os.path.join(base, "manifest.json")
        with open(mpath, encoding="utf-8") as f:
            manifest = json.load(f)
        rec0 = manifest["records"][0]
        dup = dict(rec0)
        dup["sample_id"] = "c2"
        # distinct image and mask files, but the *same* source JSON reference
        new_img = os.path.join(os.path.dirname(os.path.join(ROOT, rec0["image_path"])), "c2.png")
        shutil.copyfile(os.path.join(ROOT, rec0["image_path"]), new_img)
        dup["image_path"] = _rel(new_img)
        new_mask = os.path.join(base, "masks", "c2.png")
        shutil.copyfile(os.path.join(base, rec0["mask_path"]), new_mask)
        dup["mask_path"] = _rel(new_mask, base)
        manifest["records"].append(dup)
        with open(mpath, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        with self.assertRaises(RGBError):
            mod.RGBInspectionDataset(input_dir=base, root=ROOT)

    def test_non_object_alias_policy_rejected(self):
        policy = _default_policy()
        policy["alias_assumption"] = ["bad"]
        base = make_package("badalias", [{"id": "c1"}], policy=policy)
        with self.assertRaises(RGBError):
            mod.RGBInspectionDataset(input_dir=base, root=ROOT)

    def test_non_object_training_policy_rejected(self):
        policy = _default_policy()
        policy["training_policy"] = "bad"
        base = make_package("badtp", [{"id": "c1"}], policy=policy)
        with self.assertRaises(RGBError):
            mod.RGBInspectionDataset(input_dir=base, root=ROOT)

    def test_output_inside_input_rejected(self):
        base = make_package("outin", [{"id": "c1"}])
        out = os.path.join(base, "out")
        with self.assertRaises(RGBError):
            mod.inspect_real(input_dir=base, output_dir=out, root=ROOT)
        self.assertFalse(os.path.exists(out), "rejected run must not create output")


# --------------------------------------------------------------------------- #
class TestSmokeValidation(unittest.TestCase):
    def test_seed_negative_rejected(self):
        out = os.path.join(_new_dir("seed-neg"), "run")
        with self.assertRaises(RGBError):
            mod.run_smoke(output_dir=out, root=ROOT, steps=3, seed=-1)
        self.assertFalse(os.path.exists(out))

    def test_seed_out_of_range_rejected(self):
        out = os.path.join(_new_dir("seed-big"), "run")
        with self.assertRaises(RGBError):
            mod.run_smoke(output_dir=out, root=ROOT, steps=3, seed=2 ** 32)
        self.assertFalse(os.path.exists(out))

    def test_seed_bool_rejected(self):
        out = os.path.join(_new_dir("seed-bool"), "run")
        with self.assertRaises(RGBError):
            mod.run_smoke(output_dir=out, root=ROOT, steps=3, seed=True)
        self.assertFalse(os.path.exists(out))

    def test_fixed_reference_batch_final_loss(self):
        # final_loss must be the loss of the *saved* model on the fixed reference
        # batch (first two unshuffled synthetic samples), recomputable offline.
        out = os.path.join(_new_dir("fixedloss"), "run")
        summary = mod.run_smoke(output_dir=out, root=ROOT, steps=3, seed=17)
        self.assertEqual(summary["reference_batch_indices"], [0, 1])
        self.assertEqual(summary["loss_reference"], "same_fixed_synthetic_batch_before_after")
        samples = [
            (os.path.join(out, "synthetic", "images", f"syn_{i}.png"),
             os.path.join(out, "synthetic", "masks", f"syn_{i}.png"))
            for i in range(4)
        ]
        ds = mod._SyntheticPNGDataset(samples)
        loader = torch.utils.data.DataLoader(ds, batch_size=2, shuffle=False)
        ref = next(iter(loader))
        net = mod.SmokeNet()
        net.load_state_dict(torch.load(os.path.join(out, "synthetic_state.pt"),
                                       weights_only=True, map_location="cpu"))
        net.eval()
        with torch.no_grad():
            expected = float(mod.segmentation_loss(net(ref["image"]), ref["target"]).item())
        self.assertAlmostEqual(summary["final_loss"], expected, places=6)
        self.assertTrue(math.isfinite(summary["initial_loss"]))
        self.assertTrue(math.isfinite(summary["last_step_loss_before_update"]))

    def test_nonfinite_gradient_is_explicit_failure(self):
        out = os.path.join(_new_dir("badgrad"), "run")
        real_cls = mod.SmokeNet

        class BadGradNet(real_cls):
            def __init__(self, *a, **k):
                super().__init__(*a, **k)
                next(self.parameters()).register_hook(lambda g: torch.full_like(g, float("inf")))

        mod.SmokeNet = BadGradNet
        try:
            with self.assertRaises(RGBError):
                mod.run_smoke(output_dir=out, root=ROOT, steps=1, seed=17)
        finally:
            mod.SmokeNet = real_cls


# --------------------------------------------------------------------------- #
class TestSmokeAndInspect(unittest.TestCase):
    def test_run_smoke_updates_and_roundtrip(self):
        out = os.path.join(_new_dir("smoke"), "run")
        summary = mod.run_smoke(output_dir=out, root=ROOT, steps=6, seed=5)
        self.assertEqual(summary["kind"], "synthetic_smoke")
        self.assertEqual(summary["optimizer_steps"], 6)
        self.assertTrue(summary["parameter_changed"])
        self.assertTrue(summary["checkpoint_roundtrip_equal"])
        self.assertTrue(math.isfinite(summary["initial_loss"]))
        self.assertTrue(math.isfinite(summary["final_loss"]))
        self.assertFalse(summary["trained_on_real_data"])
        self.assertFalse(summary["production_model"])
        # checkpoint reloads strictly into a fresh network
        net = mod.SmokeNet()
        state = torch.load(os.path.join(out, "synthetic_state.pt"), weights_only=True, map_location="cpu")
        net.load_state_dict(state, strict=True)
        self.assertIn("optimizer_steps", summary)

    def test_run_smoke_rerun_rejected(self):
        out = os.path.join(_new_dir("smoke2"), "run")
        mod.run_smoke(output_dir=out, root=ROOT, steps=3, seed=1)
        with self.assertRaises(RGBError):
            mod.run_smoke(output_dir=out, root=ROOT, steps=3, seed=1)

    def test_run_smoke_bad_steps(self):
        out = os.path.join(_new_dir("smoke3"), "run")
        with self.assertRaises(RGBError):
            mod.run_smoke(output_dir=out, root=ROOT, steps=0, seed=1)

    def test_inspect_real_no_model(self):
        base = make_package("inspect", [
            {"id": "c1"}, {"id": "c2"}, {"id": "rev1", "status": "review"},
        ])
        out = os.path.join(_new_dir("inspect-out"), "run")
        summary = mod.inspect_real(input_dir=base, output_dir=out, root=ROOT)
        self.assertEqual(summary["kind"], "real_input_inspection")
        self.assertEqual(summary["optimizer_steps"], 0)
        self.assertEqual(summary["candidate_count"], 2)
        self.assertEqual(summary["excluded_review_count"], 1)
        self.assertFalse(summary["trained_on_real_data"])
        self.assertFalse(summary["training_ready"])
        self.assertFalse(summary["split_created"])
        self.assertFalse(any(n.endswith(".pt") for n in os.listdir(out)), "inspection must not save a model")
        self.assertTrue(os.path.isfile(os.path.join(out, "records.json")))


# --------------------------------------------------------------------------- #
class TestCLI(unittest.TestCase):
    def test_no_args_shows_help(self):
        result = _run_cli([], cwd=ROOT)
        self.assertEqual(result.returncode, 0)
        self.assertIn("usage", (result.stdout + result.stderr).lower())
        self.assertIn("smoke", result.stdout)
        self.assertIn("inspect", result.stdout)

    def test_bad_steps_nonzero_chinese_error(self):
        out = os.path.join(_new_dir("cli-bad"), "run")
        result = _run_cli(["smoke", "--steps", "0", "--output", out], cwd=ROOT)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("错误", result.stderr)

    def test_inspect_from_non_project_cwd(self):
        base = make_package("cli-inspect", [{"id": "c1"}])
        cwd = _new_dir("cli-cwd")
        out = os.path.join(_new_dir("cli-out"), "run")
        result = _run_cli(
            ["inspect", "--input", base, "--output", out, "--root", ROOT], cwd=cwd
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertTrue(os.path.isfile(os.path.join(out, "summary.json")))

    def test_inspect_relative_paths_from_foreign_cwd(self):
        # A truly foreign cwd (outside the project) with relative --input/--output:
        # both must resolve against the project root, not os.getcwd().
        base = make_package("cli-rel-inspect", [{"id": "c1"}])
        out = os.path.join(_new_dir("cli-rel-inspect-out"), "run")
        rel_in = _rel(base)
        rel_out = _rel(out)
        cwd = os.path.dirname(ROOT)
        result = _run_cli(
            ["inspect", "--input", rel_in, "--output", rel_out, "--root", ROOT], cwd=cwd
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertTrue(os.path.isfile(os.path.join(out, "summary.json")))

    def test_smoke_relative_output_from_foreign_cwd(self):
        out = os.path.join(_new_dir("cli-rel-smoke"), "run")
        rel_out = _rel(out)
        cwd = os.path.dirname(ROOT)
        result = _run_cli(
            ["smoke", "--steps", "3", "--seed", "4", "--output", rel_out, "--root", ROOT],
            cwd=cwd,
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertTrue(os.path.isfile(os.path.join(out, "summary.json")))


if __name__ == "__main__":
    unittest.main()
