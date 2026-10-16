# -*- coding: utf-8 -*-
"""LAB-20261016-001：修正标注接收与复检（标准库 + 现有 Pillow）。

独立 CLI/API：接收人工**另存**的几何修正 JSON，保留原始文件、005 副本与 855 候选，
生成独立复检清单，并对几何通过的修正派生 L 掩膜。当前**没有**人工修正，
本工具不自动修 48 份、不伪造人审结论、不做真实训练。

设计要点：

* 复用 ``build_review_pack`` 的 ``_load_manifest`` / ``_load_policy`` /
  ``_validate_record_fields`` / ``_scan_record`` 与重复 id/来源检查，对**整个 005 输入**
  只读审计；复用 ``prepare_candidate`` 的点校验/几何/栅格化。不复制或重写既有审计算法。
* ``corrections`` 目录必须已存在且位于 ``reports/labels`` 内，不能是 005 输入本身/子目录，
  也不能与输出互为祖先；只接受平铺 ``<sample_id>.json`` 子集，拒绝子目录/其它扩展/
  未知 id/候选 id/大小写碰撞/路径逃逸，且每个文件在读取前须先经 ``realpath`` 校验仍在
  修正目录内（拒绝符号链接/联接/解析后逃逸或重复来源）。省略 ``corrections`` 即“空提交”。
* 只允许改 ``points``：与 **005 review 原始副本**比对，顶层（除 shapes）与非 points 字段必须
  深相等（``imageData=null``、``imagePath`` 保持同目录 PNG basename 不改）；禁止增删 shape、
  改类别、改图/宽高、加入 ``training_ready``。坐标必须有限数且非 bool；点数不足、零面积、
  自交、越界仍记 review。点完全没变记 ``unchanged``，不声称已修。
* 逐条结果为 ``pending``（未提交）/``unchanged``/``review``（提交但仍有几何问题）/
  ``candidate``（几何通过且有效互斥像素 >0）/``empty``（几何通过但有效像素 0）。
  ``candidate`` 生成 L PNG（0/1/2/255）掩膜；所有有效结构的提交原字节存入 ``accepted/``，
  并复制同目录配图 PNG 使 ``imagePath`` 仍可打开。
* 所有输入与修正内容完整预检后才创建输出目录；最后一份坏也不留输出。模块 import
  不执行、不写目录；不新增依赖；不训练/切分/模型/网页/Labelme。

用法（由 Codex 运行；WorkBuddy 只编写）：

    python -B scripts/labels/intake_corrections.py \
        [--input <005目录>] [--corrections <reports/labels内已存在目录>] [--output <全新目录>]

类别沿用 005：``0=Field 田块``、``1=Field ridge 田埂``（含 Fieldridge 待复核拼写归一）、
``2=Road 道路``、``255=忽略/未画/冲突/未确认``；**0 不是背景**。
"""

import argparse
import csv
import importlib.util
import json
import os
import random
import sys
import time

TASK_ID = "LAB-20261016-001"

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))


class CorrectionError(Exception):
    """范围/输入/修正校验失败：非零退出，绝不产生半成品输出。"""


def _load_sibling(mod_name, filename):
    spec = importlib.util.spec_from_file_location(mod_name, os.path.join(_HERE, filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)   # 仅定义函数/常量，无副作用、不写目录
    return mod


_PREP = _load_sibling("lab001_prepare_candidate", "prepare_candidate.py")
_PACK = _load_sibling("lab001_build_review_pack", "build_review_pack.py")

DEFAULT_INPUT_REL = _PACK.DEFAULT_INPUT_REL
EPS = _PREP.EPS
LABEL_TO_CLASS = _PREP.LABEL_TO_CLASS

_RESULT_STATUSES = ("pending", "unchanged", "review", "candidate", "empty")


# --------------------------------------------------------------------------
# 基础工具（复用 005 工具与 build_review_pack 的既有实现）
# --------------------------------------------------------------------------
def _within(child, parent):
    return _PACK._within(child, parent)


def _rel(path, base):
    return _PACK._rel(path, base)


def _sha256_of(path):
    return _PACK._sha256_of(path)


def _read_bytes(path):
    return _PACK._read_bytes(path)


def _write_json(path, obj):
    _PACK._write_json(path, obj)


def _strict_equal(a, b):
    """类型感知深比较：区分 ``bool``/``int``/``float``（``True != 1``、``8 != 8.0``）。

    非 points 字段要求类型一致：把 ``8`` 改成 ``8.0``（同值但换类型）也判为不等，
    从而拒绝“仅表示类型变化”的元数据改动（宽高/版本/flags/类别字段等）。
    """
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            return False
        return all(_strict_equal(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return False
        return all(_strict_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, str) or isinstance(b, str):
        return isinstance(a, str) and isinstance(b, str) and a == b
    if a is None or b is None:
        return a is None and b is None
    return type(a) is type(b) and a == b


def _points_equal(a, b):
    """points 数值等价比较：允许 ``int``/``float`` 互相等价。

    只把坐标的表示类型从 ``[[0,0],...]`` 改成 ``[[0.0,0.0],...]`` 不应被误判为“已修正”；
    结构/数值合法性由调用前的 ``_PREP._clean_points`` 负责，这里只判是否真的改变。
    """
    if not (isinstance(a, list) and isinstance(b, list)):
        return a == b
    if len(a) != len(b):
        return False
    for pa, pb in zip(a, b):
        if not (isinstance(pa, (list, tuple)) and isinstance(pb, (list, tuple))):
            return False
        if len(pa) != 2 or len(pb) != 2:
            return False
        for va, vb in zip(pa, pb):
            if isinstance(va, bool) or isinstance(vb, bool):
                return False
            if not (isinstance(va, (int, float)) and isinstance(vb, (int, float))):
                return False
            if va != vb:
                return False
    return True


def _shoelace_area(pts):
    """多边形面积（px²，绝对值）；用于识别共线零面积（不足 3 点由调用方先判）。"""
    n = len(pts)
    if n < 3:
        return 0.0
    s = 0.0
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return abs(s) / 2.0


def _geometry_problems(pts, w, h, label, si):
    """逐 shape 几何问题（复用 005 的点校验/自交/越界，另加零面积）。

    返回问题列表；为空表示该 shape 几何通过。不足 3 个不同点直接短路（后续无意义）。
    """
    probs = []
    if len(set(pts)) < 3:
        probs.append({"type": "fewer_than_3_distinct_points", "shape_index": si,
                      "label": label, "n_distinct": len(set(pts))})
        return probs
    sc = _PREP._self_crossings(pts)
    if sc:
        probs.append({"type": "self_intersection", "shape_index": si,
                      "label": label, "crossing_pairs": sc})
    viol = _PREP._canvas_violations(pts, w, h)
    if viol:
        probs.append({"type": "canvas_oob", "shape_index": si,
                      "label": label, "points": viol})
    if _shoelace_area(pts) <= EPS:
        probs.append({"type": "zero_area", "shape_index": si, "label": label})
    return probs


# --------------------------------------------------------------------------
# 范围解析
# --------------------------------------------------------------------------
def _labels_dir(root):
    return os.path.join(root, "reports", "labels")


def _resolve_input(input_dir, root):
    real = _PACK._resolve_input(input_dir, root)   # BuildError -> 调用方转换
    return real


def _resolve_corrections(corrections_dir, root, in_dir):
    """解析修正目录（**不创建**）。省略即空提交（返回 None）。"""
    if corrections_dir is None:
        return None
    p = corrections_dir if os.path.isabs(corrections_dir) else os.path.join(root, corrections_dir)
    real = os.path.realpath(p)
    if not _within(real, os.path.realpath(_labels_dir(root))):
        raise CorrectionError("修正目录必须位于 reports/labels 内：%s" % corrections_dir)
    if not os.path.isdir(real):
        raise CorrectionError("修正目录不存在或不是目录：%s" % real)
    in_real = os.path.realpath(in_dir)
    if real == in_real or _within(real, in_real) or _within(in_real, real):
        raise CorrectionError("修正目录不能是 005 输入本身/子目录，也不能包含输入：%s"
                              % corrections_dir)
    return real


def _resolve_output(output_dir, root, in_dir, corr_dir):
    """解析输出目录（**不创建**）。已存在、位于/包含输入或修正目录即拒绝。

    ``output_dir=None`` 时在 ``reports/labels`` 下生成随机默认目录，随后**不再直接返回**，
    而是赋回 ``output_dir`` 走与显式路径**完全相同**的校验（LAB-20261016-004 修补：
    此前默认路径提前 return，绕过输入/修正目录父子范围验证）。
    """
    labels = os.path.realpath(_labels_dir(root))
    if output_dir is None:
        while True:
            name = "%s-intake-%s-%s" % (
                TASK_ID, time.strftime("%Y%m%d-%H%M%S"), "%06x" % random.randrange(0xFFFFFF))
            candidate = os.path.join(labels, name)
            if not os.path.exists(candidate):
                output_dir = candidate
                break
    p = output_dir if os.path.isabs(output_dir) else os.path.join(root, output_dir)
    real = os.path.realpath(os.path.abspath(p))
    parent = os.path.realpath(os.path.dirname(os.path.abspath(p)))
    if not _within(parent, labels):
        raise CorrectionError("输出目录必须位于 reports/labels 内：%s" % output_dir)
    if real == labels:
        raise CorrectionError("输出目录不能是 reports/labels 本身")
    if os.path.exists(p):
        raise CorrectionError("输出目录已存在，拒绝覆盖/删除：%s" % p)
    for other, tag in ((os.path.realpath(in_dir), "输入目录"),
                       ((os.path.realpath(corr_dir) if corr_dir else None), "修正目录")):
        if other is None:
            continue
        if real == other or _within(real, other) or _within(other, real):
            raise CorrectionError("输出目录不得与%s相同、位于其下或包含它：%s" % (tag, output_dir))
    return os.path.abspath(p)


# --------------------------------------------------------------------------
# 修正目录枚举与单份校验
# --------------------------------------------------------------------------
def _iter_corrections(corr_dir, review_ids, all_ids):
    """枚举平铺 ``<sample_id>.json``；任何越界/未知/候选/碰撞直接拒绝。返回 {id: 路径}。"""
    if corr_dir is None:
        return {}
    corr_real = os.path.realpath(corr_dir)
    # 第一遍：逐文件解析真实路径（**在读取其 JSON 之前**），再判类型/扩展/安全名/碰撞。
    found, seen_cf, seen_real = [], {}, {}
    for name in sorted(os.listdir(corr_dir)):
        full = os.path.join(corr_dir, name)
        # 每个文件 resolve 后必须仍在修正目录内：拒绝符号链接/联接/任何解析后逃逸；
        # 解析到同一真实路径的重复来源亦拒绝（不依赖 Windows 具备建链接权限）。
        real = os.path.realpath(full)
        if not _within(real, corr_real):
            raise CorrectionError("修正文件解析到修正目录之外（疑似链接/逃逸）：%s" % name)
        rkey = os.path.normcase(real)
        if rkey in seen_real:
            raise CorrectionError("修正文件解析后重复来源：%s 与 %s" % (name, seen_real[rkey]))
        seen_real[rkey] = name
        if os.path.isdir(full):
            raise CorrectionError("修正目录不接受子目录：%s" % name)
        if not os.path.isfile(full):
            raise CorrectionError("修正目录不接受非普通文件：%s" % name)
        if not name.endswith(".json"):
            raise CorrectionError("修正目录只接受 .json 文件（其它扩展拒绝）：%s" % name)
        stem = name[:-len(".json")]
        try:
            _PACK._validate_sample_id(stem)
        except _PACK.BuildError:
            raise CorrectionError("修正文件名不是安全的 <sample_id>.json（拒绝路径逃逸）：%r"
                                  % (name,))
        cf = stem.casefold()
        if cf in seen_cf:
            raise CorrectionError("修正文件大小写碰撞：%s 与 %s" % (name, seen_cf[cf]))
        seen_cf[cf] = name
        found.append((name, stem, full))
    # 第二遍：成员资格（须为 005 review；候选/未知均拒绝）。
    out = {}
    for name, stem, full in found:
        if stem not in review_ids:
            if stem in all_ids:
                raise CorrectionError("候选 id 不接受修正（仅可修正 005 review 记录）：%s" % name)
            raise CorrectionError("未知 sample_id（不在 005 记录内）：%s" % name)
        out[stem] = full
    return out


def _validate_correction(sid, sub_path, base_doc, w, h):
    """校验单份修正：仅允许改 points。返回 (原始字节, 清洗后点列表, 变化的 shape 索引)。"""
    try:
        raw = _read_bytes(sub_path)
    except OSError as e:
        raise CorrectionError("%s：修正文件读取失败 %r" % (sid, e))
    try:
        sub = json.loads(raw.decode("utf-8"))
    except Exception as e:  # noqa: BLE001
        raise CorrectionError("%s：修正 JSON 解析失败 %r" % (sid, e))
    if not isinstance(sub, dict):
        raise CorrectionError("%s：修正 JSON 根不是对象" % sid)

    base_wo = {k: v for k, v in base_doc.items() if k != "shapes"}
    sub_wo = {k: v for k, v in sub.items() if k != "shapes"}
    if not _strict_equal(sub_wo, base_wo):
        raise CorrectionError(
            "%s：顶层字段（除 shapes）必须与 005 副本完全一致（仅允许改 points；"
            "不得改 imageData/imagePath/宽高，也不得加入 training_ready）" % sid)

    base_shapes = base_doc.get("shapes")
    sub_shapes = sub.get("shapes")
    if not isinstance(sub_shapes, list):
        raise CorrectionError("%s：shapes 非列表" % sid)
    if not isinstance(base_shapes, list):
        raise CorrectionError("%s：005 副本 shapes 非列表（输入异常）" % sid)
    if len(sub_shapes) != len(base_shapes):
        raise CorrectionError("%s：禁止增删 shape（%d != %d）"
                              % (sid, len(sub_shapes), len(base_shapes)))

    cleaned, changed = [], []
    for i, (bs, ss) in enumerate(zip(base_shapes, sub_shapes)):
        if not isinstance(ss, dict):
            raise CorrectionError("%s：shape[%d] 不是对象" % (sid, i))
        if isinstance(bs, dict):
            bs_wo = {k: v for k, v in bs.items() if k != "points"}
            ss_wo = {k: v for k, v in ss.items() if k != "points"}
            if not _strict_equal(ss_wo, bs_wo):
                raise CorrectionError(
                    "%s：shape[%d] 非 points 字段被修改（不得改类别/shape_type/其它字段）"
                    % (sid, i))
        try:
            pts = _PREP._clean_points(ss, "%s shape[%d]" % (sid, i))
        except _PREP.PrepareError as e:
            raise CorrectionError("%s：shape[%d] 坐标非法（须有限数且非 bool）：%s"
                                  % (sid, i, e))
        cleaned.append(pts)
        base_pts = bs.get("points") if isinstance(bs, dict) else None
        if not _points_equal(base_pts, ss.get("points")):
            changed.append(i)
    return raw, cleaned, changed


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------
def intake(input_dir=None, corrections_dir=None, output_dir=None, root=None):
    """接收另存修正、复检并输出独立目录。返回 summary 字典。"""
    root = os.path.abspath(root) if root else _DEFAULT_ROOT
    try:
        in_dir = _resolve_input(input_dir, root)
        corr_dir = _resolve_corrections(corrections_dir, root, in_dir)
        out_dir = _resolve_output(output_dir, root, in_dir, corr_dir)

        manifest, records = _PACK._load_manifest(in_dir)
        policy = _PACK._load_policy(in_dir)

        seen_ids, dedup, infos = {}, {}, []
        for idx, rec in enumerate(records):
            _PACK._validate_record_fields(rec, idx)
            sid = rec["sample_id"]
            key = sid.casefold()
            if key in seen_ids:
                raise CorrectionError("sample_id 大小写折叠后重复：%s（首见 record[%d]）"
                                      % (sid, seen_ids[key]))
            seen_ids[key] = idx
            infos.append(_PACK._scan_record(rec, root, in_dir, dedup))

        by_id = {it["sample_id"]: it for it in infos}
        all_ids = set(by_id)
        review_ids = {it["sample_id"] for it in infos if it["status"] == "review"}
        corr_files = _iter_corrections(corr_dir, review_ids, all_ids)

        # ---- 完成全部校验/几何/栅格化（内存中）后才创建输出 ----
        entries, accepted, masks = [], [], []
        result_counts = {s: 0 for s in _RESULT_STATUSES}
        for sid in sorted(review_ids):
            it = by_id[sid]
            w, h = it["width"], it["height"]
            base_path = os.path.join(in_dir, it["review_json_path"].replace("/", os.sep))
            try:
                base_doc = json.loads(_read_bytes(base_path).decode("utf-8"))
            except Exception as e:  # noqa: BLE001
                raise CorrectionError("%s：005 副本 JSON 解析失败 %r" % (sid, e))

            entry = {
                "sample_id": sid, "dataset": it["dataset"], "input_status": it["status"],
                "source": {
                    "json_path": it["json_path"], "image_path": it["image_path"],
                    "json_sha256": it["json_sha256"], "image_sha256": it["image_sha256"],
                    "basis": "relative to project root",
                },
                "five_copy": {
                    "review_json_path": it["review_json_path"],
                    "review_png_path": it["review_png_path"],
                    "review_json_sha256": it["review_json_sha256"],
                    "review_png_sha256": it["review_png_sha256"],
                    "basis": "relative to input directory",
                },
                "before_problems": it["review_reasons"],
                "correction": None, "changed_shape_indices": None,
                "geometry_check_passed": None, "after_problems": None,
                "class_pixels": None, "valid_pixels": None, "ignore_pixels": None,
                "outputs": {"accepted_json_path": None, "accepted_png_path": None,
                            "mask_path": None},
                "accepted_json_sha256": None, "accepted_png_sha256": None, "mask_sha256": None,
                "status": "pending",
                "flags": {"semantic_reviewed": False, "training_ready": False,
                          "not_training_ready": True, "split_created": False,
                          "human_verified": False, "submitter": "unknown"},
            }

            if sid in corr_files:
                sub_path = corr_files[sid]
                raw, cleaned, changed = _validate_correction(sid, sub_path, base_doc, w, h)
                sub_shapes = json.loads(raw.decode("utf-8"))["shapes"]
                after = []
                for i, pts in enumerate(cleaned):
                    after.extend(_geometry_problems(pts, w, h, sub_shapes[i]["label"], i))
                if not changed:
                    status = "unchanged"
                elif after:
                    status = "review"
                else:
                    sr = [{"shape_index": i, "original_label": sub_shapes[i]["label"],
                           "derived_class": LABEL_TO_CLASS[sub_shapes[i]["label"]],
                           "points": cleaned[i]} for i in range(len(cleaned))]
                    mask_img, stats = _PREP._rasterize(sr, w, h)
                    valid = _PREP._valid_pixel_total(stats)
                    status = "candidate" if valid > 0 else "empty"
                    entry["class_pixels"] = stats["class_pixels"]
                    entry["valid_pixels"] = valid
                    entry["ignore_pixels"] = stats["ignore_pixels"]
                    if status == "candidate":
                        masks.append((sid, mask_img))

                entry["status"] = status
                entry["correction"] = {
                    "path": _rel(sub_path, root), "sha256": _PACK._sha256_of(sub_path),
                    "bytes": len(raw), "basis": "relative to project root",
                }
                entry["changed_shape_indices"] = changed
                entry["geometry_check_passed"] = (len(after) == 0)
                entry["after_problems"] = after
                accepted.append((sid, raw, base_path, it))

            result_counts[entry["status"]] += 1
            entries.append(entry)

        created = time.strftime("%Y-%m-%dT%H:%M:%S")
        # 统计 005 输入批次（仅计数，不合并/不重建）。
        input_counts = {"total": len(infos), "candidate": 0, "review": 0}
        for it in infos:
            input_counts[it["status"]] += 1

        # ---- 写出阶段：此时才创建输出目录 ----
        try:
            os.makedirs(out_dir, exist_ok=False)
        except FileExistsError:
            raise CorrectionError("输出目录已存在，拒绝覆盖：%s" % out_dir)
        accepted_dir = os.path.join(out_dir, "accepted")
        masks_dir = os.path.join(out_dir, "masks")
        os.makedirs(accepted_dir)
        os.makedirs(masks_dir)

        for sid, raw, base_path, it in accepted:
            jpath = os.path.join(accepted_dir, sid + ".json")
            with open(jpath, "wb") as f:
                f.write(raw)                      # 提交原字节，逐字节保存
            png_src = os.path.join(in_dir, it["review_png_path"].replace("/", os.sep))
            png_dst = os.path.join(accepted_dir, sid + ".png")
            with open(png_dst, "wb") as f:
                f.write(_read_bytes(png_src))     # 同目录配图，使 imagePath 仍可打开
            for e in entries:
                if e["sample_id"] == sid:
                    e["outputs"]["accepted_json_path"] = _rel(jpath, out_dir)
                    e["outputs"]["accepted_png_path"] = _rel(png_dst, out_dir)
                    e["accepted_json_sha256"] = _sha256_of(jpath)
                    e["accepted_png_sha256"] = _sha256_of(png_dst)
                    break

        for sid, img in masks:
            mpath = os.path.join(masks_dir, sid + ".png")
            img.save(mpath, format="PNG")
            for e in entries:
                if e["sample_id"] == sid:
                    e["outputs"]["mask_path"] = _rel(mpath, out_dir)
                    e["mask_sha256"] = _sha256_of(mpath)
                    break

        path_bases = {
            "source_paths": "relative to project root",
            "input_derived_paths": "relative to input directory",
            "output_paths": "relative to this output directory",
            "note": "correction.path 相对项目根；five_copy.* 相对 005 输入目录；"
                    "outputs.* 相对本输出目录（如 accepted/<id>.json、masks/<id>.png）。",
        }
        flags = {"semantic_reviewed": False, "training_ready": False,
                 "not_training_ready": True, "split_created": False, "human_verified": False}

        summary = {
            "task_id": TASK_ID, "created": created,
            "input_dir": _rel(in_dir, root),
            "corrections_dir": (_rel(corr_dir, root) if corr_dir else None),
            "output_dir": _rel(out_dir, root),
            "source_manifest": {
                "path": _rel(os.path.join(in_dir, "manifest.json"), root),
                "sha256": _sha256_of(os.path.join(in_dir, "manifest.json")),
            },
            "source_policy": {
                "path": _rel(os.path.join(in_dir, "policy.json"), root),
                "sha256": _sha256_of(os.path.join(in_dir, "policy.json")),
            },
            "counts": {
                "input_total": input_counts["total"],
                "input_candidate": input_counts["candidate"],
                "input_review": input_counts["review"],
                "corrections_files": len(corr_files),
                "results": dict(result_counts),
            },
            "original_candidates": {
                "count": input_counts["candidate"],
                "note": "005 候选仅计数，不合并、不重建整批、不视为训练就绪。",
            },
            "flags": flags,
            "path_bases": path_bases,
            "notes": [
                "当前没有人工修正时 corrections_files=0，全部 review 记为 pending。",
                "geometry_check_passed 只是几何复检事实，不代表人审语义通过（human_verified 恒为 false）。",
                "提交者身份不明；不记录 human_verified=true，不训练、不划分、不报精度。",
                "复用 005 的类别与别名（Fieldridge→Field ridge）待复核假设；0 不是背景。",
            ],
        }

        records_doc = {"task_id": TASK_ID, "created": created,
                       "input_dir": summary["input_dir"],
                       "corrections_dir": summary["corrections_dir"],
                       "counts": summary["counts"], "flags": flags,
                       "path_bases": path_bases, "records": entries}
        _write_json(os.path.join(out_dir, "summary.json"), summary)
        _write_json(os.path.join(out_dir, "records.json"), records_doc)
        _write_csv(os.path.join(out_dir, "records.csv"), entries)
        _write_readme(os.path.join(out_dir, "README.md"), summary, entries)
        return summary

    except CorrectionError:
        raise
    except (_PACK.BuildError, _PREP.PrepareError) as e:
        raise CorrectionError(str(e))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as e:
        raise CorrectionError("运行失败：%r" % e)


def _write_csv(path, entries):
    header = ["sample_id", "dataset", "status", "submitted", "changed_shape_count",
              "before_problem_count", "after_problem_count", "geometry_check_passed",
              "class0_pixels", "class1_pixels", "class2_pixels", "valid_pixels",
              "correction_json_path", "correction_sha256",
              "accepted_json_path", "accepted_png_path", "mask_path"]
    rows = []
    for e in entries:
        cp = e.get("class_pixels") or {}
        rows.append({
            "sample_id": e["sample_id"], "dataset": e["dataset"], "status": e["status"],
            "submitted": "true" if e["correction"] else "false",
            "changed_shape_count": ("" if e["changed_shape_indices"] is None
                                    else len(e["changed_shape_indices"])),
            "before_problem_count": len(e["before_problems"] or []),
            "after_problem_count": ("" if e["after_problems"] is None
                                    else len(e["after_problems"])),
            "geometry_check_passed": ("" if e["geometry_check_passed"] is None
                                      else str(e["geometry_check_passed"]).lower()),
            "class0_pixels": "" if not cp else cp.get("0", 0),
            "class1_pixels": "" if not cp else cp.get("1", 0),
            "class2_pixels": "" if not cp else cp.get("2", 0),
            "valid_pixels": "" if e["valid_pixels"] is None else e["valid_pixels"],
            "correction_json_path": e["correction"]["path"] if e["correction"] else "",
            "correction_sha256": e["correction"]["sha256"] if e["correction"] else "",
            "accepted_json_path": e["outputs"]["accepted_json_path"] or "",
            "accepted_png_path": e["outputs"]["accepted_png_path"] or "",
            "mask_path": e["outputs"]["mask_path"] or "",
        })
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _write_readme(path, summary, entries):
    c = summary["counts"]
    r = c["results"]
    lines = [
        "# 修正标注接收与复检（零基础说明）",
        "",
        "本目录由 `scripts/labels/intake_corrections.py` 生成，是**派生结果**；"
        "原始标注、005 候选/副本、855 候选的 mask 均**未被修改、移动或删除**。",
        "",
        "## 1. 它做什么",
        "",
        "- 接收人工**另存**的修正 JSON（只允许改 `points`），与 005 review 原始副本逐字段比对；",
        "- 复检几何（点数不足 / 零面积 / 自交 / 越界）与有效像素，逐条给出结果状态；",
        "- 把有效结构的提交**原字节**存入 `accepted/`，并为几何通过者派生 L 掩膜。",
        "",
        "## 2. 本次实际计数（不硬编码）",
        "",
        "| 项目 | 数量 |",
        "|---|---:|",
        "| 005 输入记录 | %d |" % c["input_total"],
        "| 005 候选（仅计数） | %d |" % c["input_candidate"],
        "| 005 待复核 review | %d |" % c["input_review"],
        "| 提交的修正文件 | %d |" % c["corrections_files"],
        "| 结果 pending（未提交） | %d |" % r["pending"],
        "| 结果 unchanged（点未变） | %d |" % r["unchanged"],
        "| 结果 review（仍有几何问题） | %d |" % r["review"],
        "| 结果 candidate（几何通过且有像素） | %d |" % r["candidate"],
        "| 结果 empty（几何通过但无有效像素） | %d |" % r["empty"],
        "",
        "## 3. 结果状态",
        "",
        "- `pending`：未提交修正，不重新猜测坐标；",
        "- `unchanged`：提交了但与 005 副本 `points` 完全一致，不声称已修；",
        "- `review`：提交了但仍存在几何问题（点数不足/零面积/自交/越界）；",
        "- `candidate`：几何通过且有效互斥像素 >0；",
        "- `empty`：几何通过但有效互斥像素为 0。",
        "",
        "## 4. 两目录工作流程",
        "",
        "1. 把 005 目录 `review/<id>.json` 与 `review/<id>.png` 复制到 `reports/labels` 下的"
        "另存工作目录，用标注工具只改 `points`；",
        "2. **只把改好的 `<id>.json` 交回**一个 `reports/labels` 内的接收目录（PNG 放展示/临时工作区即可）；",
        "3. 运行本工具，输出目录 `accepted/` 自带同目录配图（`<id>.png`），"
        "`imagePath` 仍可打开。",
        "",
        "## 5. 边界与状态",
        "",
        "- `semantic_reviewed=false`、`training_ready=false`、`not_training_ready=true`、"
        "`split_created=false`；`human_verified` 恒为 false，提交者身份不明。",
        "- `geometry_check_passed` 只是几何复检事实，不等于人审语义通过；",
        "- 只允许改 `points`：若标注工具改动版本/flags/类别/宽高等会被拒绝，请核对另存内容；",
        "- 不训练、不划分、不报精度/真实地块数/平方米；0 不是背景。",
        "",
        "## 6. 文件",
        "",
        "- `summary.json`：计数、来源 manifest/policy SHA、路径基准、状态。",
        "- `records.json`：逐 review 记录（状态、变化 shape 索引、前后问题、像素、摘要）。",
        "- `records.csv`：UTF-8 BOM 逐图清单。",
        "- `accepted/`：提交原字节 `<id>.json` 与同目录配图 `<id>.png`。",
        "- `masks/`：几何通过候选的 L 掩膜（0/1/2/255）。",
        "",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main(argv=None):
    parser = argparse.ArgumentParser(description="修正标注接收与复检（仅坐标修正）")
    parser.add_argument("--input", default=None, help="005 候选目录（须在 reports/labels 内）")
    parser.add_argument("--corrections", default=None, help="另存修正目录（须在 reports/labels 内）")
    parser.add_argument("--output", default=None, help="全新输出目录（须在 reports/labels 内）")
    args = parser.parse_args(argv)
    try:
        summary = intake(args.input, args.corrections, args.output)
    except CorrectionError as e:
        sys.stderr.write("[CorrectionError] %s\n" % e)
        return 2
    c = summary["counts"]
    r = c["results"]
    print(summary["output_dir"])
    print("input=%d candidate=%d review=%d submitted=%d pending=%d unchanged=%d "
          "review_out=%d candidate_out=%d empty=%d"
          % (c["input_total"], c["input_candidate"], c["input_review"], c["corrections_files"],
             r["pending"], r["unchanged"], r["review"], r["candidate"], r["empty"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
