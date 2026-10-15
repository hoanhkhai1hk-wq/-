#!/usr/bin/env python3
"""意义测试：scripts/data/prepare_annotation_pack.py（任务 DATA-20261015-003）。

覆盖：选图确定性/中心与远端、组不足取全部、PNG 与 TIFF 逐像素一致、来源路径越界与
坏 SHA/字节拒绝、被篡改原图拒绝、已有输出不覆盖、原图不变、未标注语义、CLI 边界。

测试只读写 reports/data/DATA-20261015-003/ 下的全新自测子目录，不触碰真实原图与旧索引。
运行：".venv-p0/Scripts/python.exe" -B -m unittest tests.data.test_annotation_pack -v
"""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import unittest
import uuid
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = PROJECT_ROOT / "scripts" / "data"
SCRIPT = SCRIPT_DIR / "prepare_annotation_pack.py"
TEST_ROOT = PROJECT_ROOT / "reports" / "data" / "DATA-20261015-004"
TEST_ROOT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(SCRIPT_DIR))
import prepare_annotation_pack as pap  # noqa: E402

TILE = 16


def _uniq(tag: str) -> str:
    return f"{tag}-{uuid.uuid4().hex[:8]}"


def new_subdir(tag: str) -> Path:
    """在本次自测根下新建唯一子目录（含 --help 用的空目录），本任务不删除。"""
    p = TEST_ROOT / _uniq(tag)
    p.mkdir(parents=True, exist_ok=False)
    return p


def entry(r: int, c: int, group: str = "qie0716rgb") -> dict:
    """选图函数所需的最小条目（不落盘），用于端到端 select_items 验证。"""
    return {
        "group": group,
        "row": r,
        "col": c,
        "path": f"{group}/tiles/tile_r{r}_c{c}.tif",
    }



def make_tile(path: Path, seed: int, size: int = TILE):
    path.parent.mkdir(parents=True, exist_ok=True)
    buf = bytearray()
    for r in range(size):
        for c in range(size):
            v = (seed * 31 + r * 7 + c * 3) % 256
            buf += bytes((v, (v * 2) % 256, (v * 3) % 256))
    Image.frombytes("RGB", (size, size), bytes(buf)).save(path, format="TIFF")


def make_group(root: Path, group: str, coords):
    """coords: iterable of (row, col)。返回该组索引条目列表。"""
    entries = []
    for i, (r, c) in enumerate(coords):
        rel = f"{group}/tiles/tile_r{r}_c{c}.tif"
        p = root / group / "tiles" / f"tile_r{r}_c{c}.tif"
        make_tile(p, seed=(hash((group, r, c)) % 1000) + i)
        entries.append(
            {
                "path": rel,
                "bytes": p.stat().st_size,
                "sha256": pap.sha256_file(p),
                "shape": [TILE, TILE, 3],
                "dtype": "uint8",
            }
        )
    # 写入 meta.json（脚本会读取作来源信息）
    (root / group).mkdir(parents=True, exist_ok=True)
    (root / group / "meta.json").write_text(
        json.dumps({"source_path": f"C:/orig/{group}_result.tif", "tile_size": TILE, "rows": 9, "cols": 9}),
        encoding="utf-8",
    )
    return entries


def write_manifest(path: Path, entries):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entries, ensure_ascii=False, indent=1), encoding="utf-8")


def fresh_fixture(tag: str):
    base = TEST_ROOT / _uniq(tag)
    root = base / "src"
    manifest = base / "manifest.json"
    return base, root, manifest


def run_build(base: Path, root: Path, manifest: Path, per_group=6, tile_size=TILE, tag="out"):
    out = TEST_ROOT / _uniq(f"{tag}-out")
    out = pap.validate_output_dir(out, PROJECT_ROOT)
    return pap.build_annotation_pack(root, manifest, out, per_group, tile_size), out


class SelectionTests(unittest.TestCase):
    def test_determinism_center_and_far(self):
        base, root, manifest = fresh_fixture("sel")
        entries = make_group(root, "qie0716rgb", [(r, c) for r in range(3) for c in range(3)])
        write_manifest(manifest, entries)
        validated = pap.load_manifest(manifest, root)

        chosen1, order1 = pap.select_for_group(validated, 6)
        chosen2, order2 = pap.select_for_group(validated, 6)
        # 确定性：两次结果与次序一致
        self.assertEqual([e["path"] for e in chosen1], [e["path"] for e in chosen2])
        self.assertEqual(order1, order2)

        chosen_paths = {e["path"] for e in chosen1}
        self.assertEqual(len(chosen1), 6)
        # 选择次序第 1 张应最接近包围盒中心 (r1,c1)
        first_path = min(order1, key=lambda p: order1[p])
        self.assertTrue(first_path.endswith("tile_r1_c1.tif"), first_path)
        self.assertEqual(order1["qie0716rgb/tiles/tile_r1_c1.tif"], 1)
        # 角点 (0,0) 应作为“远端”被选中
        self.assertIn("qie0716rgb/tiles/tile_r0_c0.tif", chosen_paths)

    def test_group_fewer_than_six_takes_all(self):
        base, root, manifest = fresh_fixture("few")
        e1 = make_group(root, "qie0716rgb", [(0, 0), (0, 1), (1, 0), (1, 1)])
        write_manifest(manifest, e1)
        validated = pap.load_manifest(manifest, root)
        chosen, order = pap.select_for_group(validated, 6)
        self.assertEqual(len(chosen), 4)
        # 短组分支也必须返回 dict（path -> 1-based 排名），与长组分支类型一致
        self.assertIsInstance(order, dict)
        self.assertEqual(sorted(order.values()), [1, 2, 3, 4])

    # ---- 端到端（经 select_items），而不只是局部长度 ----
    def test_select_items_fewer_than_six(self):
        chosen = pap.select_items([entry(0, 0), entry(1, 1)], 6)
        self.assertEqual(len(chosen), 2)
        self.assertEqual([v["selection_rank"] for v in chosen], [1, 2])

    def test_select_items_exactly_six(self):
        coords = [(r, c) for r in range(3) for c in range(3)][:6]
        chosen = pap.select_items([entry(r, c) for r, c in coords], 6)
        self.assertEqual(len(chosen), 6)
        self.assertEqual(sorted(v["selection_rank"] for v in chosen), [1, 2, 3, 4, 5, 6])

    def test_select_items_empty_other_group(self):
        chosen = pap.select_items([entry(0, 0), entry(1, 1), entry(2, 2)], 6)
        self.assertEqual(len(chosen), 3)
        self.assertTrue(all(v["group"] == "qie0716rgb" for v in chosen))

    def test_exact_equal_center_distance_uses_numeric_tiebreak(self):
        # r1c1 与 r1c2 到包围盒中心的几何距离完全相同；float 比较会误选 r1c2，
        # Fraction 精确距离应让 numeric 同分规则选中 r1c1。
        rows = [entry(r, c) for r, c in [(0, 0), (0, 3), (1, 1), (1, 2), (2, 0), (2, 3), (0, 1)]]
        _, order = pap.select_for_group(rows, 6)
        first = min(order, key=order.get)
        self.assertEqual(first, "qie0716rgb/tiles/tile_r1_c1.tif")
        self.assertEqual(order[first], 1)
        # 若用浮点 == 比较，会误选几何距离相同的 r1c2；这里显式确认没有发生。
        self.assertNotEqual(first, "qie0716rgb/tiles/tile_r1_c2.tif")


class BuildOutputTests(unittest.TestCase):
    def _fixture_nine(self, tag):
        base, root, manifest = fresh_fixture(tag)
        entries = make_group(root, "qie0716rgb", [(r, c) for r in range(3) for c in range(3)])
        write_manifest(manifest, entries)
        return base, root, manifest

    def test_png_pixel_exact_and_unlabeled(self):
        base, root, manifest = self._fixture_nine("pix")
        result, out = run_build(base, root, manifest, per_group=6)
        m = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(m["target_status"], "pending")
        self.assertIsNone(m["target_classes"])
        self.assertEqual(m["total_input"], 9)
        self.assertEqual(m["groups"]["qie0716rgb"]["selected_count"], 6)
        self.assertEqual(len(m["items"]), 6)
        for item in m["items"]:
            self.assertEqual(item["annotation_status"], "unlabeled")
            self.assertIsNone(item["label_path"])
            self.assertEqual(item["png_size"], [TILE, TILE])
            png = out / item["png_relative_path"]
            src = root / item["source_relative_path"]
            with Image.open(png) as a, Image.open(src) as b:
                a.load(); b.load()
                self.assertEqual(a.mode, "RGB")
                self.assertEqual(a.tobytes(), b.tobytes())
            self.assertEqual(pap.sha256_file(png), item["png_sha256"])
        # 不含任何掩膜/标签文件
        names = [p.name.lower() for p in out.rglob("*")]
        self.assertFalse(any("mask" in n or n.endswith("_label.png") for n in names))
        # 联系图存在
        self.assertTrue((out / "contact_sheet.png").is_file())
        # CSV 为 UTF-8 BOM 且状态列未标注
        raw = (out / "annotation_todo.csv").read_bytes()
        self.assertTrue(raw.startswith(b"\xef\xbb\xbf"))
        with open(out / "annotation_todo.csv", encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 6)
        for row in rows:
            self.assertEqual(row["标注状态"], "unlabeled")
            self.assertEqual(row["待填目标类别"], "")

    def test_source_unchanged(self):
        base, root, manifest = self._fixture_nine("unchanged")
        before = {p: pap.sha256_file(p) for p in root.rglob("*.tif")}
        run_build(base, root, manifest, per_group=6)
        after = {p: pap.sha256_file(p) for p in root.rglob("*.tif")}
        self.assertEqual(before, after)
        self.assertTrue(len(before) == 9 and all(before.values()))


class RejectionTests(unittest.TestCase):
    def _valid_nine(self, tag):
        base, root, manifest = fresh_fixture(tag)
        entries = make_group(root, "qie0716rgb", [(r, c) for r in range(3) for c in range(3)])
        write_manifest(manifest, entries)
        return base, root, manifest, entries

    def test_path_escape_rejected(self):
        base, root, manifest = fresh_fixture("escape")
        make_group(root, "qie0716rgb", [(0, 0)])
        for bad in [
            "qie0716rgb/tiles/../../evil.tif",
            "../qie0716rgb/tiles/tile_r0_c0.tif",
            "C:/abs/qie0716rgb/tiles/tile_r0_c0.tif",
            "/abs/qie0716rgb/tiles/tile_r0_c0.tif",
            "qie0716rgb\\tiles\\tile_r0_c0.tif",
        ]:
            write_manifest(manifest, [{"path": bad, "bytes": 1, "sha256": "0" * 64}])
            with self.assertRaises(pap.PackError):
                pap.load_manifest(manifest, root)

    def test_bad_sha_and_bytes_rejected(self):
        base, root, manifest = fresh_fixture("badsha")
        entries = make_group(root, "qie0716rgb", [(0, 0)])
        for field, value in [("sha256", "xyz"), ("sha256", "A" * 64), ("bytes", 0), ("bytes", -3), ("bytes", "12"), ("bytes", True)]:
            bad = dict(entries[0]); bad[field] = value
            write_manifest(manifest, [bad])
            with self.assertRaises(pap.PackError):
                pap.load_manifest(manifest, root)

    def test_duplicate_path_and_coord_rejected(self):
        base, root, manifest = fresh_fixture("dup")
        entries = make_group(root, "qie0716rgb", [(0, 0)])
        write_manifest(manifest, entries + entries)
        with self.assertRaises(pap.PackError):
            pap.load_manifest(manifest, root)

    def test_tampered_source_rejected(self):
        base, root, manifest, entries = self._valid_nine("tamper")
        # 先成功构建一次
        run_build(base, root, manifest, per_group=3)
        # 篡改其中一张原图内容
        victim = root / entries[0]["path"]
        make_tile(victim, seed=999, size=TILE)
        with self.assertRaises(pap.PackError):
            run_build(base, root, manifest, per_group=3)

    def test_existing_output_not_overwritten(self):
        base, root, manifest, entries = self._valid_nine("exist")
        pre = TEST_ROOT / _uniq("preout")
        pre.mkdir(parents=True, exist_ok=True)
        marker = pre / "keep.txt"
        marker.write_text("keep", encoding="utf-8")
        with self.assertRaises(pap.PackError):
            pap.validate_output_dir(pre, PROJECT_ROOT)
        self.assertTrue(marker.is_file())  # 未被删除

    def test_output_outside_reports_data_rejected(self):
        outside = SCRIPT_DIR / _uniq("outside")
        with self.assertRaises(pap.PackError):
            pap.validate_output_dir(outside, PROJECT_ROOT)

    def test_validate_source_root_accepts_fixture_and_rejects_other(self):
        base, root, manifest = fresh_fixture("srcok")
        make_group(root, "qie0716rgb", [(0, 0)])
        self.assertEqual(pap.validate_source_root(root, PROJECT_ROOT), root.resolve())
        with self.assertRaises(pap.PackError):
            pap.validate_source_root(PROJECT_ROOT, PROJECT_ROOT)

    def test_validate_input_manifest_rejects_outside_reports_data(self):
        with self.assertRaises(pap.PackError):
            pap.validate_input_manifest(PROJECT_ROOT / "AGENTS.md", PROJECT_ROOT)

    def test_build_entry_enforces_input_and_output_range(self):
        base, root, manifest = fresh_fixture("entryguard")
        entries = make_group(root, "qie0716rgb", [(0, 0)])
        write_manifest(manifest, entries)
        # 输出在 reports/data 之外 -> build 入口直接拒绝（不依赖 CLI 预校验）
        with self.assertRaises(pap.PackError):
            pap.build_annotation_pack(root, manifest, SCRIPT_DIR / _uniq("out-guard"), 6, TILE)
        # source_root 非法（项目根不是归档根，也不在 reports/data 内）-> 拒绝
        with self.assertRaises(pap.PackError):
            pap.build_annotation_pack(
                PROJECT_ROOT, manifest, TEST_ROOT / _uniq("src-guard"), 6, TILE
            )


class CliTests(unittest.TestCase):
    def _env(self):
        # 子进程显式 UTF-8；只设置本测试的子环境，不改系统环境或全局变量。
        env = dict(os.environ)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        return env

    def test_help_from_other_cwd(self):
        # 用 reports 内新建的空目录作为 cwd，不用 TemporaryDirectory（本任务不删除）。
        other_cwd = new_subdir("helpcwd")
        res = subprocess.run(
            [sys.executable, "-B", str(SCRIPT), "--help"],
            cwd=str(other_cwd), capture_output=True, text=True, encoding="utf-8", env=self._env(),
        )
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertIn("usage", res.stdout.lower())

    def test_cli_success_and_failure_exitcodes(self):
        base, root, manifest = fresh_fixture("cli")
        entries = make_group(root, "qie0716rgb", [(r, c) for r in range(3) for c in range(3)])
        write_manifest(manifest, entries)
        out = TEST_ROOT / _uniq("cli-out")
        ok = subprocess.run(
            [sys.executable, "-B", str(SCRIPT), "--source-root", str(root), "--manifest", str(manifest),
             "--output", str(out), "--per-group", "6", "--tile-size", str(TILE)],
            capture_output=True, text=True, encoding="utf-8", env=self._env(),
        )
        self.assertEqual(ok.returncode, 0, ok.stderr)
        self.assertTrue((out / "manifest.json").is_file())
        # 同一输出目录再跑一次应非零退出（不覆盖）
        again = subprocess.run(
            [sys.executable, "-B", str(SCRIPT), "--source-root", str(root), "--manifest", str(manifest),
             "--output", str(out), "--per-group", "6", "--tile-size", str(TILE)],
            capture_output=True, text=True, encoding="utf-8", env=self._env(),
        )
        self.assertNotEqual(again.returncode, 0)
        # 坏索引应非零退出
        bad_manifest = base / "bad.json"
        write_manifest(bad_manifest, [{"path": "qie0716rgb/tiles/tile_r0_c0.tif", "bytes": 1, "sha256": "0" * 64}])
        bad = subprocess.run(
            [sys.executable, "-B", str(SCRIPT), "--source-root", str(root), "--manifest", str(bad_manifest),
             "--output", str(TEST_ROOT / _uniq("bad-out"))],
            capture_output=True, text=True, encoding="utf-8", env=self._env(),
        )
        self.assertNotEqual(bad.returncode, 0)

    def test_relative_params_anchor_to_project_root(self):
        # 相对参数必须锚定项目根，而不是 Host 默认 cwd。
        self.assertEqual(
            pap.anchor_path("scripts/data", PROJECT_ROOT), PROJECT_ROOT / "scripts" / "data"
        )
        abs_p = TEST_ROOT / "abs-check"
        self.assertEqual(pap.anchor_path(str(abs_p), PROJECT_ROOT), abs_p)


if __name__ == "__main__":
    unittest.main(verbosity=2)
