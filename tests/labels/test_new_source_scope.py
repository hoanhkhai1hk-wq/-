# -*- coding: utf-8 -*-
"""DATA-20261021-001 新增来源 0413-r36c23 允许列表范围测试（由 Codex 运行；WorkBuddy 不执行）。

仅标准库 unittest，导入既有脚本 `scripts/labels/prepare_candidate.py`，
只测试 `_resolve_source` 的路径范围行为。使用 TemporaryDirectory 创建**合成项目根**
并在其中写临时假文件，不读取、不依赖真实 0413-r36c23 / 邓 / hy0713 (1) 数据，
不在真实新增来源写任何 fixture。不测栅格化算法（旧 test_candidate.py 已覆盖）。
"""

import importlib.util
import os
import tempfile
import unittest

SCRIPT = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..",
    "scripts", "labels", "prepare_candidate.py"))


def _load_module():
    spec = importlib.util.spec_from_file_location("prepare_candidate_scope_mod", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


MOD = _load_module()


def _same_path(a, b):
    return os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))


class TestNewSourceScope(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def _mk(self, rel_dir, name):
        d = os.path.join(self.root, rel_dir)
        os.makedirs(d, exist_ok=True)
        p = os.path.join(d, name)
        with open(p, "wb") as f:
            f.write(b"SYNTHETIC-FIXTURE")
        return p

    def test_new_dir_real_file_accepted(self):
        """新允许目录 0413-r36c23 内真实存在的文件被接受，返回 resolve 后路径。"""
        real = self._mk("0413-r36c23", "img_0001.tif")
        got = MOD._resolve_source(os.path.join("0413-r36c23", "img_0001.tif"),
                                  self.root, "test")
        self.assertTrue(_same_path(got, real),
                        "应返回 resolve 后的真实路径：%r != %r" % (got, real))
        self.assertEqual(os.path.normcase(os.path.realpath(got)),
                         os.path.normcase(os.path.realpath(real)))

    def test_old_two_dirs_still_accepted(self):
        """旧两目录 邓、hy0713 (1) 继续被接受，返回 resolve 后路径。"""
        for d, name in (("邓", "a.tif"), ("hy0713 (1)", "b.json")):
            real = self._mk(d, name)
            got = MOD._resolve_source(os.path.join(d, name), self.root, "test")
            self.assertTrue(_same_path(got, real),
                            "%s 应继续接受且返回真实路径：%r != %r" % (d, got, real))

    def test_prefix_lookalike_dir_rejected(self):
        """近似同名前缀目录（0413-r36c23-copy、邓x）不在允许列表，拒绝。"""
        for d, name in (("0413-r36c23-copy", "img.tif"), ("邓x", "a.tif")):
            self._mk(d, name)
            rel = os.path.join(d, name)
            with self.assertRaises(MOD.PrepareError,
                                   msg="近似前缀目录应拒绝：%r" % rel):
                MOD._resolve_source(rel, self.root, "test")

    def test_dotdot_escape_rejected(self):
        """含 '..' 的路径一律拒绝，即使目标恰好存在。"""
        real = self._mk("0413-r36c23", "img.tif")
        # 从 allowed 子目录语义上逃逸的写法（rel 相对 root，'..' 指向 root 之外）。
        with self.assertRaises(MOD.PrepareError, msg="含 '..' 的路径应拒绝"):
            MOD._resolve_source(os.path.join("..", "outside.tif"), self.root, "test")
        # 确认该写法不是被当作允许目录内的合法文件放行。
        with self.assertRaises(MOD.PrepareError):
            MOD._resolve_source(os.path.join("0413-r36c23", "..", "img.tif"),
                                self.root, "test")
        self.assertTrue(os.path.isfile(real))  # 原文件未被触碰

    def test_outside_root_or_absolute_rejected(self):
        """项目根外的文件与绝对路径均拒绝。"""
        # 合成根之外的相对写法必然含 '..'，先被 '..' 规则拒绝（已测）；
        # 这里补：根内但不在任何允许目录的文件、以及绝对路径（含根内/根外）。
        self._mk("", "loose.json")
        with self.assertRaises(MOD.PrepareError,
                               msg="根内非允许目录文件应拒绝"):
            MOD._resolve_source("loose.json", self.root, "test")
        with self.assertRaises(MOD.PrepareError, msg="绝对路径应拒绝"):
            MOD._resolve_source(os.path.join(self.root, "0413-r36c23", "img.tif"),
                                self.root, "test")
        with tempfile.TemporaryDirectory() as outside:
            p = self._mk("", "keepme.txt")  # 根内哨兵，证明没有写根外
            with self.assertRaises(MOD.PrepareError, msg="根外绝对路径应拒绝"):
                MOD._resolve_source(os.path.join(outside, "evil.tif"),
                                    self.root, "test")
            self.assertFalse(os.path.exists(os.path.join(outside, "evil.tif")))
            self.assertTrue(os.path.isfile(p))


if __name__ == "__main__":
    unittest.main(verbosity=2)
