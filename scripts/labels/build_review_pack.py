# -*- coding: utf-8 -*-
"""DATA-20261016-007：RGB 训练前审计与离线可视复核包（标准库 + 现有 Pillow）。

只读 DATA-20261016-005 候选目录，做**完整训练前审计**并生成**离线**可视复核包：

  * 逐记录校验原始 JSON/TIFF、候选 mask、review 副本的 SHA/bytes/模式/尺寸/像素一致性；
  * 实算 mask 直方图并与 ``manifest.pixel_stats`` 逐项核对（denominator / class_pixels /
    ignore_pixels），候选必须有有效互斥像素；
  * 输出 ``audit.json``（``training_ready=false`` / ``semantic_reviewed=false`` /
    ``split_created=false``）、逐图 CSV、review 轮廓预览、最多 12 张候选彩色叠加、
    ``index.html`` / ``gallery.json`` / ``README.md``。

本工具**不训练、不切分、不修改任何原始标注/图像/旧产物/服务/网页/模型/配置**，
不新建 train/val/test 划分，不报精度、真实地块数或平方米。WorkBuddy 只编写，由 Codex 运行：

    python -B scripts/labels/build_review_pack.py --input <005目录> --output <全新目录>

默认输入 ``<根>/reports/labels/RGB-candidates-DATA-20261016-005``；输出只能写到
``<根>/reports/labels`` 内一个**全新**目录（已存在即拒绝，不覆盖/不删除）。

当前工具只适用于 005 的 ``candidate`` / ``review`` 批次；``empty`` / ``unsupported``
状态明确不支持并拒绝（见 ``docs/RGB训练前复核.md``）。
"""

import argparse
import csv
import hashlib
import html
import json
import math
import os
import random
import re
import sys
import time
import urllib.parse

from PIL import Image, ImageDraw, ImageFont

TASK_ID = "DATA-20261016-007"
SOURCE_TASK_ID = "DATA-20261016-005"
DEFAULT_INPUT_REL = "reports/labels/RGB-candidates-DATA-20261016-005"

ALLOWED_SOURCE_DIRS = ("邓", "hy0713 (1)")
SUPPORTED_STATUSES = ("candidate", "review")
CLASS_KEYS = ("0", "1", "2")
MASK_VALUES = frozenset((0, 1, 2, 255))
MAX_CANDIDATE_SAMPLES = 12

# 已知原稿 label；Fieldridge 为待确认拼写（其归一假设见 ALIAS_MAP）。
KNOWN_LABELS = frozenset(("Field", "Field ridge", "Fieldridge", "Road"))
# 005 既定别名映射：仅此一条，status 必须为 pending_semantic_review。
ALIAS_MAP = {"Fieldridge": "Field ridge"}
# 当前支持的 review 问题类型（005 实际出现 canvas_oob / self_intersection）。
SUPPORTED_REASON_TYPES = frozenset(
    ("canvas_oob", "self_intersection", "too_few_distinct_points"))
POLICY_CLASS_KEYS = ("0", "1", "2", "255")
# 类别身份必须严格对应（取类名开头英文 token，允许既有中文说明）：
# 0=Field、1=Field ridge、2=Road、255=ignore。不得缺类/额外训练类/调换/拼接矛盾描述。
EXPECTED_CLASS_TOKENS = {"0": "field", "1": "field ridge", "2": "road", "255": "ignore"}

SHA_RE = re.compile(r"[0-9a-f]{64}")
SAFE_ID_RE = re.compile(r"[A-Za-z0-9_-]{1,80}")
# prefix 可空：同时接受 `hy0717_tile_r35_c49` 与 `tile_r54_c24`（后者 prefix 空、row/col 保留）。
TILE_RE = re.compile(r"^(?:(?P<prefix>.+?)_)?tile_r(?P<row>\d+)_c(?P<col>\d+)$")
WIN_RESERVED = ({"CON", "PRN", "AUX", "NUL"}
                | {"COM%d" % i for i in range(1, 10)}
                | {"LPT%d" % i for i in range(1, 10)})

# 轮廓颜色（按原 label）；未知 label 用品红，不静默丢点。
LABEL_COLORS = {
    "Field": (0, 200, 0),
    "Field ridge": (240, 200, 0),
    "Fieldridge": (240, 200, 0),
    "Road": (230, 40, 40),
}
_UNKNOWN_COLOR = (200, 0, 200)

# 候选叠加颜色（类 0 也必须显示，255 保持原图可见即透明）。
MASK_OVERLAY = {
    "0": (0, 180, 0, 110),
    "1": (230, 200, 0, 110),
    "2": (220, 60, 60, 110),
}

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))


class BuildError(Exception):
    """输入/范围/校验失败：非零退出，绝不产生“成功”复核包。"""


# --------------------------------------------------------------------------
# 基础工具
# --------------------------------------------------------------------------
def _is_int(v):
    return isinstance(v, int) and not isinstance(v, bool)


def _is_sha256(s):
    return isinstance(s, str) and bool(SHA_RE.fullmatch(s))


def _is_finite_number(v):
    return (isinstance(v, (int, float)) and not isinstance(v, bool)
            and math.isfinite(v))


def _within(child, parent):
    child = os.path.abspath(child)
    parent = os.path.abspath(parent)
    return child == parent or child.startswith(parent + os.sep)


def _rel(path, base):
    return os.path.relpath(path, base).replace("\\", "/")


def _href(rel_path):
    """相对链接转 URL（保留 '/'，编码空格与中文）。"""
    return urllib.parse.quote(rel_path, safe="/")


def _sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_bytes(path):
    with open(path, "rb") as f:
        return f.read()


def _write_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _validate_sample_id(sid):
    if not isinstance(sid, str) or not SAFE_ID_RE.fullmatch(sid):
        raise BuildError("sample_id 非法（须匹配 [A-Za-z0-9_-]{1,80}）：%r" % (sid,))
    if sid.upper() in WIN_RESERVED:
        raise BuildError("sample_id 命中 Windows 保留名：%r" % sid)


def _resolve_source(rel, root, ctx):
    """源路径：必须是相对路径，resolve 后落在允许目录内。"""
    if not isinstance(rel, str) or not rel:
        raise BuildError("%s：路径字段非字符串或为空" % ctx)
    if os.path.isabs(rel):
        raise BuildError("%s：拒绝绝对路径 %r" % (ctx, rel))
    norm = rel.replace("\\", "/")
    if any(part == ".." for part in norm.split("/")):
        raise BuildError("%s：拒绝含 '..' 的路径 %r" % (ctx, rel))
    real = os.path.realpath(os.path.join(root, norm))
    allowed = [os.path.realpath(os.path.join(root, d)) for d in ALLOWED_SOURCE_DIRS]
    allowed.append(os.path.realpath(os.path.join(root, "reports", "labels")))
    if not any(_within(real, a) for a in allowed):
        raise BuildError("%s：路径越界（仅允许 邓、hy0713 (1) 或 reports/labels 内）：%r"
                         % (ctx, rel))
    if not os.path.isfile(real):
        raise BuildError("%s：原始文件不存在：%s" % (ctx, real))
    return real


def _resolve_derived(rel, input_dir, ctx):
    """派生产物路径：必须落在 input 目录内（相对 input 目录）。"""
    if not isinstance(rel, str) or not rel:
        raise BuildError("%s：派生产物路径非字符串或为空" % ctx)
    if os.path.isabs(rel):
        raise BuildError("%s：拒绝绝对派生产物路径 %r" % (ctx, rel))
    norm = rel.replace("\\", "/")
    if any(part == ".." for part in norm.split("/")):
        raise BuildError("%s：派生产物路径含 '..'：%r" % (ctx, rel))
    real = os.path.realpath(os.path.join(input_dir, norm))
    if not _within(real, os.path.realpath(input_dir)):
        raise BuildError("%s：派生产物路径越界（须在 input 目录内）：%r" % (ctx, rel))
    if not os.path.isfile(real):
        raise BuildError("%s：派生产物文件不存在：%s" % (ctx, real))
    return real


# --------------------------------------------------------------------------
# 范围解析
# --------------------------------------------------------------------------
def _labels_dir(root):
    return os.path.join(root, "reports", "labels")


def _resolve_input(input_dir, root):
    if input_dir is None:
        input_dir = DEFAULT_INPUT_REL
    p = input_dir if os.path.isabs(input_dir) else os.path.join(root, input_dir)
    real = os.path.realpath(p)
    if not _within(real, os.path.realpath(_labels_dir(root))):
        raise BuildError("输入目录必须位于 reports/labels 内：%s" % input_dir)
    if not os.path.isdir(real):
        raise BuildError("输入目录不存在或不是目录：%s" % real)
    return real


def _resolve_output(output_dir, root, input_dir):
    """解析目标输出目录（**不创建**）。已存在、位于输入下或等于 labels 本身即拒绝。"""
    labels = os.path.realpath(_labels_dir(root))
    if output_dir is None:
        while True:
            name = "%s-review-pack-%s-%s" % (
                TASK_ID, time.strftime("%Y%m%d-%H%M%S"), "%06x" % random.randrange(0xFFFFFF))
            candidate = os.path.join(labels, name)
            if not os.path.exists(candidate):
                return candidate
    p = output_dir if os.path.isabs(output_dir) else os.path.join(root, output_dir)
    real = os.path.realpath(os.path.abspath(p))
    parent = os.path.realpath(os.path.dirname(os.path.abspath(p)))
    if not _within(parent, labels):
        raise BuildError("输出目录必须位于 reports/labels 内：%s" % output_dir)
    if real == labels:
        raise BuildError("输出目录不能是 reports/labels 本身")
    if os.path.exists(p):
        raise BuildError("输出目录已存在，拒绝覆盖/删除：%s" % p)
    in_real = os.path.realpath(input_dir)
    if real == in_real or _within(real, in_real) or _within(in_real, real):
        raise BuildError("输出目录不得与输入目录相同、位于输入下或包含输入：%s" % output_dir)
    return os.path.abspath(p)


# --------------------------------------------------------------------------
# 索引与记录校验
# --------------------------------------------------------------------------
def _load_manifest(input_dir):
    path = os.path.join(input_dir, "manifest.json")
    if not os.path.isfile(path):
        raise BuildError("输入目录缺少 manifest.json：%s" % path)
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:  # noqa: BLE001
        raise BuildError("manifest.json 解析失败：%r" % e)
    if not isinstance(data, dict):
        raise BuildError("manifest 根不是对象")
    records = data.get("records")
    if not isinstance(records, list) or not records:
        raise BuildError("manifest.records 缺失或为空")
    return data, records


def _class_token(value):
    """取类名字符串开头的英文类别 token（到首个非 ASCII 字符为止），压缩空白并小写。

    例：``"Field ridge 田埂"`` → ``"field ridge"``；``"ignore 忽略/未画..."`` → ``"ignore"``。
    非字符串或空返回 None，交由调用方按“身份不符/缺类”拒绝。
    """
    if not isinstance(value, str) or not value:
        return None
    eng = []
    for ch in value:
        if ord(ch) > 127:
            break
        eng.append(ch)
    token = " ".join("".join(eng).split())
    return token.casefold() or None


def _load_policy(input_dir):
    """读取并校验 005 真实 policy.json（必须在创建输出目录之前）。

    只校验**必要策略语义**，不硬编码整份 policy 的 SHA，也不忽略真实 005 既有字段：
    四类身份必须严格对应（0=Field、1=Field ridge、2=Road、255=ignore，允许既有中文
    说明，但不得缺类/额外训练类/调换/拼接矛盾描述）、255 必须是忽略而非训练类别、
    不得擅自声明训练就绪、别名映射必须恰为 Fieldridge→Field ridge 且
    status=pending_semantic_review。
    """
    path = os.path.join(input_dir, "policy.json")
    if not os.path.isfile(path):
        raise BuildError("输入目录缺少 policy.json（005 策略文件）：%s" % path)
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:  # noqa: BLE001
        raise BuildError("policy.json 解析失败：%r" % e)
    if not isinstance(data, dict):
        raise BuildError("policy.json 根不是对象")
    cids = data.get("class_ids")
    if not isinstance(cids, dict):
        raise BuildError("policy.class_ids 缺失或非对象")
    for k in POLICY_CLASS_KEYS:
        v = cids.get(k)
        if not isinstance(v, str) or not v.strip():
            raise BuildError("policy.class_ids[%s] 缺失/非字符串或为空" % k)
    extra = [k for k in cids if k not in POLICY_CLASS_KEYS]
    if extra:
        raise BuildError("policy.class_ids 含额外类别键（疑似额外训练类）：%r" % (extra,))
    for k in POLICY_CLASS_KEYS:
        got = _class_token(cids[k])
        if got != EXPECTED_CLASS_TOKENS[k]:
            raise BuildError(
                "policy.class_ids[%s] 类别身份不符（须为 %r，实际英文 token=%r）：%r"
                % (k, EXPECTED_CLASS_TOKENS[k], got, cids[k]))
    if data.get("no_background_class") is not True:
        raise BuildError("policy.no_background_class 必须为 true")
    alias = data.get("alias_assumption")
    if not isinstance(alias, dict):
        raise BuildError("policy.alias_assumption 缺失或非对象")
    if alias.get("map") != ALIAS_MAP:
        raise BuildError("policy.alias_assumption.map 必须恰为 {Fieldridge: Field ridge}：%r"
                         % (alias.get("map"),))
    if alias.get("status") != "pending_semantic_review":
        raise BuildError("policy.alias_assumption.status 必须为 pending_semantic_review")
    tp = data.get("training_policy")
    if not isinstance(tp, dict):
        raise BuildError("policy.training_policy 缺失或非对象")
    if tp.get("semantic_reviewed") is not False:
        raise BuildError("policy.training_policy.semantic_reviewed 必须为 false")
    if tp.get("not_training_ready") is not True:
        raise BuildError("policy.training_policy.not_training_ready 必须为 true")
    if tp.get("no_train_test_split") is not True:
        raise BuildError("policy.training_policy.no_train_test_split 必须为 true")
    return data


REQUIRED_FIELDS = ("sample_id", "dataset", "json_path", "image_path",
                   "json_sha256", "json_bytes", "image_sha256", "image_bytes",
                   "width", "height", "status")


def _validate_record_fields(rec, idx):
    if not isinstance(rec, dict):
        raise BuildError("record[%d] 不是对象" % idx)
    for key in REQUIRED_FIELDS:
        if key not in rec:
            raise BuildError("record[%d] 缺字段 %s" % (idx, key))
    _validate_sample_id(rec["sample_id"])
    if not isinstance(rec["dataset"], str) or not rec["dataset"]:
        raise BuildError("record[%d].dataset 非法" % idx)
    for key in ("json_path", "image_path"):
        if not isinstance(rec[key], str) or not rec[key]:
            raise BuildError("record[%d].%s 非法" % (idx, key))
    for key in ("json_sha256", "image_sha256"):
        if not _is_sha256(rec[key]):
            raise BuildError("record[%d].%s 非法 SHA256" % (idx, key))
    for key in ("json_bytes", "image_bytes", "width", "height"):
        if not _is_int(rec[key]) or rec[key] <= 0:
            raise BuildError("record[%d].%s 非法整数（含 bool/负数/零）" % (idx, key))
    if rec["status"] not in SUPPORTED_STATUSES:
        raise BuildError(
            "record[%d].status=%r 不支持；本工具只适用于 candidate/review 批次"
            "（empty/unsupported 明确拒绝）" % (idx, rec["status"]))
    # 不接受把源候选擅自改 training_ready 的输入。
    if rec.get("semantic_reviewed") is not False:
        raise BuildError("record[%d].semantic_reviewed 必须为 false" % idx)
    if rec.get("not_training_ready") is not True:
        raise BuildError("record[%d].not_training_ready 必须为 true" % idx)


def _mask_histogram(mask_img, ctx):
    if mask_img.mode != "L":
        raise BuildError("%s：mask 非 L 模式（mode=%s）" % (ctx, mask_img.mode))
    hist = mask_img.histogram()
    if len(hist) != 256:
        raise BuildError("%s：mask 直方图非 256 档" % ctx)
    for v, n in enumerate(hist):
        if n and v not in MASK_VALUES:
            raise BuildError("%s：mask 含非法像素值 %d" % (ctx, v))
    return hist


def parse_tile_name(name):
    """从文件名解析 ``*_tile_r<row>_c<col>``；不可解析返回空串字段。"""
    stem = os.path.splitext(str(name).replace("\\", "/").rsplit("/", 1)[-1])[0]
    m = TILE_RE.match(stem)
    if not m:
        return {"prefix": "", "row": "", "col": ""}
    return {"prefix": m.group("prefix") or "", "row": m.group("row"), "col": m.group("col")}


def _register_once(dedup, real, sid, kind, ctx):
    """按 resolve+normcase 去重：同一物理文件不得被多条记录复用。"""
    key = os.path.normcase(os.path.realpath(real))
    if key in dedup:
        raise BuildError("%s：%s 与已有记录（%s）复用同一文件：%s"
                         % (ctx, kind, dedup[key], real))
    dedup[key] = sid


def _validate_shape_structure(sh, ctx):
    """校验单个 shape 的可渲染结构；坏 shape 必须显式拒绝，不得静默跳过。"""
    if not isinstance(sh, dict):
        raise BuildError("%s 不是对象" % ctx)
    label = sh.get("label")
    if not isinstance(label, str) or label not in KNOWN_LABELS:
        raise BuildError("%s.label 为未知类别：%r" % (ctx, label))
    if sh.get("shape_type") != "polygon":
        raise BuildError("%s.shape_type 非 polygon：%r" % (ctx, sh.get("shape_type")))
    pts = sh.get("points")
    if not isinstance(pts, list) or not pts:
        raise BuildError("%s.points 非列表或为空" % ctx)
    for pi, p in enumerate(pts):
        if not (isinstance(p, (list, tuple)) and len(p) == 2):
            raise BuildError("%s.points[%d] 不是 [x, y]" % (ctx, pi))
        x, y = p
        if not _is_finite_number(x) or not _is_finite_number(y):
            raise BuildError("%s.points[%d] 坐标须为有限数且非 bool：%r" % (ctx, pi, p))


def _canon_label(label):
    return ALIAS_MAP.get(label, label) if isinstance(label, str) else label


def _expected_copy_view(jdoc, png_basename):
    """以**原稿**为基准构造 review 副本的期望视图（只做单向派生，不反向归一）。

    - imagePath 变为副本 PNG basename；
    - imageData 变为 null；
    - shape.label 仅按既定正向映射 Fieldridge→Field ridge（其余原 label 保留）；
    - 其余原字段、points、顺序均保持不变。

    副本 JSON 必须与该期望视图**直接深比较相等**；副本侧不做任何归一，因此
    “原 Field ridge → 副本 Fieldridge”这类逆向改写会被判为不一致。
    """
    d = dict(jdoc)
    d["imagePath"] = png_basename
    d["imageData"] = None
    shapes = d.get("shapes")
    if isinstance(shapes, list):
        out = []
        for sh in shapes:
            if isinstance(sh, dict):
                s2 = dict(sh)
                if "label" in s2:
                    s2["label"] = _canon_label(s2["label"])
                out.append(s2)
            else:
                out.append(sh)
        d["shapes"] = out
    return d


def _scan_record(rec, root, input_dir, dedup):
    """单记录只读完整校验；任何不符即 BuildError。返回规范化 info。

    所有读取/解析/结构错误均包装为 BuildError，且都在创建输出目录之前完成，
    因此末条记录失败也不会留下半成品输出目录。
    """
    sid = rec["sample_id"]
    jreal = _resolve_source(rec["json_path"], root, "record[%s].json_path" % sid)
    ireal = _resolve_source(rec["image_path"], root, "record[%s].image_path" % sid)
    _register_once(dedup, jreal, sid, "原 JSON", sid)
    _register_once(dedup, ireal, sid, "原 TIFF", sid)

    try:
        jbytes = _read_bytes(jreal)
        ibytes = _read_bytes(ireal)
    except OSError as e:
        raise BuildError("%s：源文件读取失败 %r" % (sid, e))
    if len(jbytes) != rec["json_bytes"] or _sha256_of(jreal) != rec["json_sha256"]:
        raise BuildError("%s：原 JSON 摘要/大小与 manifest 不符" % sid)
    if len(ibytes) != rec["image_bytes"] or _sha256_of(ireal) != rec["image_sha256"]:
        raise BuildError("%s：原图像摘要/大小与 manifest 不符" % sid)

    # ---- 原 JSON 必须解析并验证核心结构（不只看 SHA）----
    try:
        with open(jreal, "r", encoding="utf-8") as f:
            jdoc = json.load(f)
    except Exception as e:  # noqa: BLE001
        raise BuildError("%s：原 JSON 解析失败 %r" % (sid, e))
    if not isinstance(jdoc, dict):
        raise BuildError("%s：原 JSON 根不是对象" % sid)
    ipath = jdoc.get("imagePath")
    if not isinstance(ipath, str) or not ipath:
        raise BuildError("%s：原 JSON imagePath 非法：%r" % (sid, ipath))
    img_basename = os.path.basename(rec["image_path"].replace("\\", "/"))
    # 字段字符串必须严格等于实际 TIFF basename；不剥目录迁就 foreign/ 前缀、斜杠/反斜杠
    # 目录前缀或绝对路径（原稿规则保持严格）。
    if ipath != img_basename:
        raise BuildError(
            "%s：原 JSON imagePath 必须严格等于原 TIFF basename（不得含目录/绝对路径）：%r != %r"
            % (sid, ipath, img_basename))
    for key, want in (("imageWidth", rec["width"]), ("imageHeight", rec["height"])):
        val = jdoc.get(key)
        if not _is_int(val) or val <= 0 or val != want:
            raise BuildError("%s：原 JSON %s 非法或与记录不符：%r" % (sid, key, val))
    shapes = jdoc.get("shapes")
    if not isinstance(shapes, list):
        raise BuildError("%s：原 JSON shapes 非列表" % sid)
    if rec["status"] == "candidate" and not shapes:
        raise BuildError("%s：candidate 原 JSON shapes 为空" % sid)
    for si, sh in enumerate(shapes):
        _validate_shape_structure(sh, "%s.shapes[%d]" % (sid, si))

    try:
        img = Image.open(ireal)
        img.load()
    except Exception as e:  # noqa: BLE001
        raise BuildError("%s：图像解码失败 %r" % (sid, e))
    if img.mode != "RGB":
        raise BuildError("%s：图像非 RGB（mode=%s）" % (sid, img.mode))
    if img.size != (rec["width"], rec["height"]):
        raise BuildError("%s：图像尺寸与 manifest 不符 %s" % (sid, list(img.size)))

    info = {
        "sample_id": sid, "dataset": rec["dataset"], "status": rec["status"],
        "json_path": rec["json_path"], "image_path": rec["image_path"],
        "json_sha256": rec["json_sha256"], "json_bytes": rec["json_bytes"],
        "image_sha256": rec["image_sha256"], "image_bytes": rec["image_bytes"],
        "width": rec["width"], "height": rec["height"],
        "alias_count": len(rec.get("alias_normalized") or []),
        "problem_count": 0, "review_reasons": [], "shapes": shapes,
        "mask_path": None, "mask_sha256": None,
        "review_png_path": None, "review_json_path": None,
        # 统计仅对候选有意义；review 留空表示“未生成”，不是 0% 已知有效。
        "class_pixels": None, "valid_pixels": None,
        "ignore_pixels": None, "denominator": None, "ignore_ratio": None,
    }
    info.update(parse_tile_name(rec["image_path"]))

    if rec["status"] == "candidate":
        if (rec.get("review_png_path") is not None
                or rec.get("review_json_path") is not None
                or rec.get("review_png_sha256") is not None
                or rec.get("review_json_sha256") is not None):
            raise BuildError("%s：candidate 不应携带 review 副本" % sid)
        if rec.get("review_reasons"):
            raise BuildError("%s：candidate 不应携带 review 理由" % sid)
        if not isinstance(rec.get("mask_path"), str) or not rec["mask_path"]:
            raise BuildError("%s：candidate 缺 mask_path" % sid)
        if not _is_sha256(rec.get("mask_sha256")):
            raise BuildError("%s：candidate mask_sha256 非法" % sid)
        mreal = _resolve_derived(rec["mask_path"], input_dir,
                                 "record[%s].mask_path" % sid)
        _register_once(dedup, mreal, sid, "候选 mask", sid)
        if _sha256_of(mreal) != rec["mask_sha256"]:
            raise BuildError("%s：候选 mask SHA 与 manifest 不符" % sid)
        try:
            mask = Image.open(mreal)
            mask.load()
        except Exception as e:  # noqa: BLE001
            raise BuildError("%s：mask 解码失败 %r" % (sid, e))
        if mask.format != "PNG":
            raise BuildError("%s：候选 mask 必须是真正的 PNG（实际 format=%r）"
                             % (sid, mask.format))
        if mask.mode != "L":
            raise BuildError("%s：mask 非 L 模式（mode=%s）" % (sid, mask.mode))
        if mask.size != (rec["width"], rec["height"]):
            raise BuildError("%s：mask 尺寸与 manifest 不符 %s" % (sid, list(mask.size)))
        hist = _mask_histogram(mask, sid)
        class_pixels = {"0": hist[0], "1": hist[1], "2": hist[2]}
        ignore_pixels = hist[255]
        valid = sum(class_pixels.values())
        if valid <= 0:
            raise BuildError("%s：候选 mask 无有效互斥像素（应有 >0）" % sid)
        ps = rec.get("pixel_stats")
        if not isinstance(ps, dict):
            raise BuildError("%s：candidate 缺 pixel_stats" % sid)
        want_cp = ps.get("class_pixels")
        if not isinstance(want_cp, dict):
            raise BuildError("%s：pixel_stats.class_pixels 非法" % sid)
        for k in CLASS_KEYS:
            v = want_cp.get(k)
            if not _is_int(v) or v < 0:
                raise BuildError("%s：pixel_stats.class_pixels[%s] 须为非负整数（禁止 float/bool）：%r"
                                 % (sid, k, v))
            if v != class_pixels[k]:
                raise BuildError("%s：class_pixels[%s] 实算 %d != manifest %r"
                                 % (sid, k, class_pixels[k], v))
        ig = ps.get("ignore_pixels")
        if not _is_int(ig) or ig < 0:
            raise BuildError("%s：pixel_stats.ignore_pixels 须为非负整数：%r" % (sid, ig))
        if ig != ignore_pixels:
            raise BuildError("%s：ignore_pixels 实算 %d != manifest %r"
                             % (sid, ignore_pixels, ig))
        den = ps.get("denominator")
        if not _is_int(den) or den <= 0:
            raise BuildError("%s：pixel_stats.denominator 须为正整数：%r" % (sid, den))
        if den != rec["width"] * rec["height"]:
            raise BuildError("%s：denominator 实算 %d != manifest %r"
                             % (sid, rec["width"] * rec["height"], den))
        info["class_pixels"] = class_pixels
        info["valid_pixels"] = valid
        info["ignore_pixels"] = ignore_pixels
        info["denominator"] = rec["width"] * rec["height"]
        info["ignore_ratio"] = ignore_pixels / float(rec["width"] * rec["height"])
        info["mask_path"] = rec["mask_path"]
        info["mask_sha256"] = rec["mask_sha256"]
        img.close()
        mask.close()
    else:  # review
        if rec.get("mask_path") is not None or rec.get("mask_sha256") is not None:
            raise BuildError("%s：review 记录不应有 mask" % sid)
        if not (isinstance(rec.get("review_png_path"), str)
                and isinstance(rec.get("review_json_path"), str)):
            raise BuildError("%s：review 缺副本路径" % sid)
        if not (_is_sha256(rec.get("review_png_sha256"))
                and _is_sha256(rec.get("review_json_sha256"))):
            raise BuildError("%s：review 副本 SHA 非法" % sid)
        rpng = _resolve_derived(rec["review_png_path"], input_dir,
                                "record[%s].review_png_path" % sid)
        rjson = _resolve_derived(rec["review_json_path"], input_dir,
                                 "record[%s].review_json_path" % sid)
        _register_once(dedup, rpng, sid, "review PNG 副本", sid)
        _register_once(dedup, rjson, sid, "review JSON 副本", sid)
        if _sha256_of(rpng) != rec["review_png_sha256"]:
            raise BuildError("%s：review PNG SHA 与 manifest 不符" % sid)
        if _sha256_of(rjson) != rec["review_json_sha256"]:
            raise BuildError("%s：review JSON SHA 与 manifest 不符" % sid)

        # ---- review 副本 JSON 必须解析并验证结构，且与原稿深比较一致 ----
        try:
            with open(rjson, "r", encoding="utf-8") as f:
                rdoc = json.load(f)
        except Exception as e:  # noqa: BLE001
            raise BuildError("%s：review 副本 JSON 解析失败 %r" % (sid, e))
        if not isinstance(rdoc, dict):
            raise BuildError("%s：review 副本 JSON 根不是对象" % sid)
        dpath = rdoc.get("imagePath")
        if not isinstance(dpath, str) or not dpath:
            raise BuildError("%s：review 副本 imagePath 非法：%r" % (sid, dpath))
        dnorm = dpath.replace("\\", "/")
        if "/" in dnorm:
            raise BuildError("%s：review 副本 imagePath 不应含目录：%r" % (sid, dpath))
        if dnorm != os.path.basename(rec["review_png_path"].replace("\\", "/")):
            raise BuildError("%s：review 副本 imagePath 与副本 PNG basename 不符：%r"
                             % (sid, dpath))
        if rdoc.get("imageData") is not None:
            raise BuildError("%s：review 副本 imageData 必须为 null" % sid)
        for key, want in (("imageWidth", rec["width"]), ("imageHeight", rec["height"])):
            val = rdoc.get(key)
            if not _is_int(val) or val <= 0 or val != want:
                raise BuildError("%s：review 副本 %s 非法或与记录不符：%r" % (sid, key, val))
        rshapes = rdoc.get("shapes")
        if not isinstance(rshapes, list):
            raise BuildError("%s：review 副本 shapes 非列表" % sid)
        if len(rshapes) != len(shapes):
            raise BuildError("%s：review 副本 shapes 数量与原稿不符" % sid)
        for si, sh in enumerate(rshapes):
            _validate_shape_structure(sh, "%s.副本.shapes[%d]" % (sid, si))
        want = _expected_copy_view(
            jdoc, os.path.basename(rec["review_png_path"].replace("\\", "/")))
        if want != rdoc:
            raise BuildError(
                "%s：review 副本与原稿（经单向派生 imagePath/imageData/Fieldridge→Field ridge）"
                "不一致；副本侧不做反向归一" % sid)

        try:
            cp = Image.open(rpng)
            cp.load()
        except Exception as e:  # noqa: BLE001
            raise BuildError("%s：review PNG 解码失败 %r" % (sid, e))
        if cp.format != "PNG":
            raise BuildError("%s：review 副本必须是真正的 PNG（实际 format=%r）"
                             % (sid, cp.format))
        if cp.mode != "RGB":
            raise BuildError("%s：review PNG 非 RGB（mode=%s）" % (sid, cp.mode))
        if cp.size != (rec["width"], rec["height"]):
            raise BuildError("%s：review PNG 尺寸与 manifest 不符 %s" % (sid, list(cp.size)))
        if cp.tobytes() != img.tobytes():
            raise BuildError("%s：review PNG 与原 TIFF 逐像素不一致" % sid)

        # ---- review_reasons 校验：非空、type 已知、shape_index 0 起有效、label 与对应原 shape 一致
        recs = rec.get("review_reasons")
        if not isinstance(recs, list):
            raise BuildError("%s：review_reasons 非列表" % sid)
        if not recs:
            raise BuildError("%s：review_reasons 为空（review 至少须一条问题）" % sid)
        for ri, r in enumerate(recs):
            if not isinstance(r, dict):
                raise BuildError("%s：review_reasons[%d] 不是对象" % (sid, ri))
            t = r.get("type")
            if t not in SUPPORTED_REASON_TYPES:
                raise BuildError("%s：review_reasons[%d].type 未知：%r" % (sid, ri, t))
            si = r.get("shape_index")
            if not _is_int(si) or si < 0 or si >= len(shapes):
                raise BuildError("%s：review_reasons[%d].shape_index 非法/越界：%r"
                                 % (sid, ri, si))
            if r.get("label") != shapes[si].get("label"):
                raise BuildError("%s：review_reasons[%d].label 与对应原 shape 不符：%r != %r"
                                 % (sid, ri, r.get("label"), shapes[si].get("label")))

        info["review_reasons"] = recs
        info["problem_count"] = len(recs)
        info["review_png_path"] = rec["review_png_path"]
        info["review_json_path"] = rec["review_json_path"]
        info["review_png_sha256"] = rec["review_png_sha256"]
        info["review_json_sha256"] = rec["review_json_sha256"]
        img.close()
        cp.close()

    return info


# --------------------------------------------------------------------------
# 确定性选样（最多 12 张）
# --------------------------------------------------------------------------
def _interleave_by_dataset(pool):
    groups = {}
    for r in pool:
        groups.setdefault(r["dataset"], []).append(r)
    for g in groups.values():
        g.sort(key=lambda r: r["sample_id"])
    out, i = [], 0
    names = sorted(groups)
    while True:
        added = False
        for n in names:
            if i < len(groups[n]):
                out.append(groups[n][i])
                added = True
        if not added:
            return out
        i += 1


def select_samples(cands, limit=MAX_CANDIDATE_SAMPLES):
    """确定性选样：优先覆盖道路/田埂并跨批次，样本按 sample_id 排序。"""
    road = [c for c in cands if c.get("class_pixels", {}).get("2", 0) > 0]
    ridge = [c for c in cands if c.get("class_pixels", {}).get("1", 0) > 0
             and c.get("class_pixels", {}).get("2", 0) == 0]
    other = [c for c in cands if c.get("class_pixels", {}).get("1", 0) == 0
             and c.get("class_pixels", {}).get("2", 0) == 0]
    pools = [_interleave_by_dataset(road), _interleave_by_dataset(ridge),
             _interleave_by_dataset(other)]
    chosen, seen = [], set()
    while len(chosen) < limit:
        progressed = False
        for p in pools:
            if p:
                r = p.pop(0)
                if r["sample_id"] not in seen:
                    seen.add(r["sample_id"])
                    chosen.append(r)
                progressed = True
                if len(chosen) >= limit:
                    break
        if not progressed:
            break
    return chosen


# --------------------------------------------------------------------------
# 绘制
# --------------------------------------------------------------------------
def _font():
    try:
        return ImageFont.load_default()
    except Exception:  # noqa: BLE001
        return None


def _seq_points(points):
    out = []
    if not isinstance(points, list):
        return out
    for p in points:
        if isinstance(p, (list, tuple)) and len(p) == 2:
            x, y = p
            if isinstance(x, (int, float)) and isinstance(y, (int, float)) \
                    and not isinstance(x, bool) and not isinstance(y, bool):
                if math.isfinite(x) and math.isfinite(y):
                    out.append((float(x), float(y)))
    return out


def _draw_contours(img, shapes, highlight_ids, base_width=2, highlight_width=4):
    """在原图上按原 JSON 绘制轮廓；高亮 shape 额外加粗并标 0 起的 shape_index。

    绘制可被画布裁切，但**不修改任何源坐标**。
    """
    draw = ImageDraw.Draw(img)
    font = _font()
    w, h = img.size
    for si, sh in enumerate(shapes):
        if not isinstance(sh, dict):
            continue
        seq = _seq_points(sh.get("points"))
        if not seq:
            continue
        color = LABEL_COLORS.get(sh.get("label"), _UNKNOWN_COLOR)
        hi = si in highlight_ids
        width = highlight_width if hi else base_width
        if len(seq) >= 2:
            draw.line(seq + [seq[0]], fill=color, width=width)
        else:
            x, y = seq[0]
            draw.ellipse([x - 2, y - 2, x + 2, y + 2], fill=color)
        if hi:
            label = "s%d" % si
            lx = min(max(seq[0][0], 0), w - 1)
            ly = min(max(seq[0][1], 0), h - 1)
            if font is not None:
                try:
                    box = draw.textbbox((0, 0), label, font=font)
                except Exception:  # noqa: BLE001
                    box = None
                if box is not None:
                    tw = box[2] - box[0]
                    th = box[3] - box[1]
                    # 按 textbbox 把文字起点限制到画布内，避免钉在 w-1/h-1 边缘被截掉
                    # （小 fixture 不足字体大小时缩短起点距离，但不改任何源坐标）。
                    lx = min(lx, max(0, w - tw))
                    ly = min(ly, max(0, h - th))
            draw.text((lx, ly), label, fill=(255, 255, 255), font=font)
    return img


def _side_by_side(left, right, gap=8, bg=(18, 18, 18)):
    w = left.size[0] + gap + right.size[0]
    h = max(left.size[1], right.size[1])
    canvas = Image.new("RGB", (w, h), bg)
    canvas.paste(left, (0, 0))
    canvas.paste(right, (left.size[0] + gap, 0))
    return canvas


def render_review_preview(orig_rgb, shapes, review_reasons):
    """review 预览：左原图 / 右原图+轮廓（问题 shape 高亮并标索引）。"""
    highlight = set()
    for r in review_reasons or []:
        if isinstance(r, dict) and _is_int(r.get("shape_index")):
            highlight.add(r["shape_index"])
    left = orig_rgb.copy()
    right = orig_rgb.copy()
    _draw_contours(right, shapes, highlight)
    return _side_by_side(left, right)


def render_candidate_overlay(orig_rgb, mask_l):
    """候选叠加：彩色 mask 覆盖原图；255 保持原图可见；类 0 也显示。"""
    base = orig_rgb.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    for k in CLASS_KEYS:
        color = MASK_OVERLAY[k]
        sel = mask_l.point(lambda v, val=int(k): 255 if v == val else 0)
        overlay = Image.composite(Image.new("RGBA", base.size, color), overlay, sel)
    return Image.alpha_composite(base, overlay).convert("RGB")


def render_candidate_preview(orig_rgb, mask_l):
    """候选预览：左原图 / 右彩色叠加并排，便于直接比对（255 透明、类 0 显示）。"""
    return _side_by_side(orig_rgb.copy(), render_candidate_overlay(orig_rgb, mask_l))


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------
def build_pack(input_dir=None, output_dir=None, root=None):
    root = os.path.abspath(root) if root else _DEFAULT_ROOT
    in_dir = _resolve_input(input_dir, root)
    out_dir = _resolve_output(output_dir, root, in_dir)  # 仅解析，不创建

    manifest, records = _load_manifest(in_dir)
    policy = _load_policy(in_dir)  # 读取并校验 005 真实 policy（在创建输出之前）

    # ---- 校验阶段：任何失败都在创建输出目录之前抛出 ----
    seen_ids = {}
    dedup = {}
    infos = []
    for idx, rec in enumerate(records):
        _validate_record_fields(rec, idx)
        sid = rec["sample_id"]
        key = sid.casefold()
        if key in seen_ids:
            raise BuildError("sample_id 大小写折叠后重复：%s（首见 record[%d]）"
                             % (sid, seen_ids[key]))
        seen_ids[key] = idx
        infos.append(_scan_record(rec, root, in_dir, dedup))

    # ---- 汇总（全部按实际，不硬编码）----
    counts = {"input": len(infos), "candidate": 0, "review": 0}
    batch_counts = {}
    pixel_totals = {"class_pixels": {"0": 0, "1": 0, "2": 0}, "valid_pixels": 0,
                    "ignore_pixels": 0, "denominator_sum": 0}
    class_image_counts = {"0": 0, "1": 0, "2": 0}
    problem_type_counts = {}
    for it in infos:
        counts[it["status"]] += 1
        bc = batch_counts.setdefault(it["dataset"], {"candidate": 0, "review": 0})
        bc[it["status"]] += 1
        if it["status"] == "candidate":
            for k in CLASS_KEYS:
                pixel_totals["class_pixels"][k] += it["class_pixels"][k]
                if it["class_pixels"][k] > 0:
                    class_image_counts[k] += 1
            pixel_totals["valid_pixels"] += it["valid_pixels"]
            pixel_totals["ignore_pixels"] += it["ignore_pixels"]
            pixel_totals["denominator_sum"] += it["denominator"]
        else:
            for r in it["review_reasons"]:
                t = r.get("type") if isinstance(r, dict) else None
                if t:
                    problem_type_counts[t] = problem_type_counts.get(t, 0) + 1

    created = time.strftime("%Y-%m-%dT%H:%M:%S")
    rel_in = _rel(in_dir, root)

    # ---- 写出阶段：此时才创建输出目录 ----
    try:
        os.makedirs(out_dir, exist_ok=False)
    except FileExistsError:
        raise BuildError("输出目录已存在，拒绝覆盖：%s" % out_dir)
    previews_dir = os.path.join(out_dir, "previews")
    os.makedirs(previews_dir)

    # 逐图 CSV：review 的像素统计留空（未生成），不写 0% 已知有效。
    csv_rows = []
    for it in infos:
        cp = it["class_pixels"]
        csv_rows.append({
            "sample_id": it["sample_id"], "dataset": it["dataset"], "status": it["status"],
            "image_path": it["image_path"], "image_sha256": it["image_sha256"],
            "json_path": it["json_path"], "json_sha256": it["json_sha256"],
            "width": it["width"], "height": it["height"],
            "class0_pixels": "" if cp is None else cp["0"],
            "class1_pixels": "" if cp is None else cp["1"],
            "class2_pixels": "" if cp is None else cp["2"],
            "valid_pixels": "" if it["valid_pixels"] is None else it["valid_pixels"],
            "ignore_pixels": "" if it["ignore_pixels"] is None else it["ignore_pixels"],
            "ignore_ratio": "" if it["ignore_ratio"] is None else "%.6f" % it["ignore_ratio"],
            "denominator": "" if it["denominator"] is None else it["denominator"],
            "alias_count": it["alias_count"], "problem_count": it["problem_count"],
            "tile_prefix": it["prefix"], "tile_row": it["row"], "tile_col": it["col"],
            "mask_path": it["mask_path"] or "", "mask_sha256": it["mask_sha256"] or "",
            "review_png_path": it["review_png_path"] or "",
            "review_json_path": it["review_json_path"] or "",
        })
    _write_csv(os.path.join(out_dir, "records.csv"), csv_rows)

    # ---- 可视包 ----
    gallery = []
    chosen = select_samples([i for i in infos if i["status"] == "candidate"])
    chosen_ids = {c["sample_id"] for c in chosen}

    for it in infos:
        sid = it["sample_id"]
        if it["status"] == "review":
            with Image.open(_resolve_source(it["image_path"], root, "img")) as im:
                im.load()
                preview = render_review_preview(im, it["shapes"], it["review_reasons"])
            pname = "previews/%s.png" % sid
            preview.save(os.path.join(out_dir, "previews", sid + ".png"), format="PNG")
            gallery.append({
                "kind": "review", "sample_id": sid, "dataset": it["dataset"],
                "status": it["status"], "preview_path": pname,
                "preview_basis": "relative to output directory",
                "source": {"json_path": it["json_path"], "image_path": it["image_path"],
                           "json_sha256": it["json_sha256"], "image_sha256": it["image_sha256"],
                           "basis": "relative to project root"},
                "derived_refs": [
                    {"path": it["review_png_path"], "basis": "relative to input directory",
                     "note": "review 可编辑 PNG 副本（与原图逐像素一致）"},
                    {"path": it["review_json_path"], "basis": "relative to input directory",
                     "note": "review 可编辑 JSON 副本（points 原样）"},
                ],
                "problems": [
                    {"type": r.get("type"), "shape_index": r.get("shape_index"),
                     "label": r.get("label"),
                     "detail": {k: v for k, v in r.items() if k != "type"}}
                    for r in it["review_reasons"] if isinstance(r, dict)
                ],
                "note": "预览左=原图 / 右=原图+轮廓，粗线为问题 shape 并标 0 起 shape_index；"
                        "越界坐标保留于 problems.detail。尚待复核。",
            })
        elif sid in chosen_ids:
            with Image.open(_resolve_source(it["image_path"], root, "img")) as im:
                im.load()
                base = im.copy()
            with Image.open(_resolve_derived(it["mask_path"], in_dir, "mask")) as mk:
                mk.load()
                overlay = render_candidate_preview(base, mk)
            pname = "previews/%s.png" % sid
            overlay.save(os.path.join(out_dir, "previews", sid + ".png"), format="PNG")
            gallery.append({
                "kind": "candidate", "sample_id": sid, "dataset": it["dataset"],
                "status": it["status"], "preview_path": pname,
                "preview_basis": "relative to output directory",
                "source": {"json_path": it["json_path"], "image_path": it["image_path"],
                           "json_sha256": it["json_sha256"], "image_sha256": it["image_sha256"],
                           "basis": "relative to project root"},
                "derived_refs": [
                    {"path": it["mask_path"], "basis": "relative to input directory",
                     "note": "候选 mask（值域 0/1/2/255）"},
                ],
                "class_pixels": it["class_pixels"], "valid_pixels": it["valid_pixels"],
                "ignore_pixels": it["ignore_pixels"], "ignore_ratio": it["ignore_ratio"],
                "note": "预览左=原图 / 右=派生候选彩色掩膜叠加（255 保持原图可见、类 0 也显示）；"
                        "非模型预测，尚待复核。",
            })

    _write_json(os.path.join(out_dir, "gallery.json"), {
        "task_id": TASK_ID, "source_task_id": SOURCE_TASK_ID, "input_dir": rel_in,
        "created": created, "candidate_sample_limit": MAX_CANDIDATE_SAMPLES,
        "samples": gallery,
    })

    audit = {
        "task_id": TASK_ID, "source_task_id": SOURCE_TASK_ID,
        "input_dir": rel_in, "created": created,
        "training_ready": False, "semantic_reviewed": False, "split_created": False,
        "counts": counts, "batch_counts": batch_counts,
        "candidate_pixel_totals": pixel_totals,
        "candidate_class_image_counts": class_image_counts,
        "review_problem_type_counts": dict(sorted(problem_type_counts.items())),
        "policy_checked": {
            "class_ids": list(policy["class_ids"].keys()),
            "no_background_class": policy.get("no_background_class"),
            "alias_map": policy["alias_assumption"].get("map"),
            "alias_status": policy["alias_assumption"].get("status"),
            "not_training_ready": policy["training_policy"].get("not_training_ready"),
        },
        "todos": [
            "修正 %d 份 review 文件中的几何问题（数量按实际）后重新质检。" % counts["review"],
            "对全部候选做人工语义确认（当前 semantic_reviewed=false）。",
            "确认 Fieldridge\uff08无空格\uff09→Field ridge 的别名归一假设是否成立。",
            "核实两批来源图像的空间/时相关系，并设计训练/验证/测试划分（当前未划分）。",
            "现有 8 波段 FTW 权重与 RGB 输入不兼容，需另行确定 RGB 方案。",
            "数据集无独立沟渠类别，不得用田埂替代沟渠。",
        ],
        "notes": [
            "候选 mask 是机械派生结果，不等于语义真值；不报精度、真实地块数或平方米。",
            "本工具只读输入，不训练、不切分、不修改任何原始标注/图像/旧产物。",
            "可视叠加为派生 mask，不是模型预测，也不代表确认/修标成功。",
        ],
        "path_bases": {
            "source_paths": "relative to project root",
            "derived_input_paths": "relative to input directory",
            "preview_paths": "relative to output directory",
        },
    }
    _write_json(os.path.join(out_dir, "audit.json"), audit)

    _write_index_html(os.path.join(out_dir, "index.html"), out_dir, in_dir, gallery, audit)
    _write_readme(os.path.join(out_dir, "README.md"), audit, gallery)

    return {
        "output_dir": _rel(out_dir, root), "counts": counts,
        "candidate_visual_samples": len([s for s in gallery if s["kind"] == "candidate"]),
        "review_visual_samples": len([s for s in gallery if s["kind"] == "review"]),
    }


def _write_csv(path, rows):
    header = ["sample_id", "dataset", "status", "image_path", "image_sha256",
              "json_path", "json_sha256", "width", "height",
              "class0_pixels", "class1_pixels", "class2_pixels", "valid_pixels",
              "ignore_pixels", "ignore_ratio", "denominator", "alias_count",
              "problem_count", "tile_prefix", "tile_row", "tile_col",
              "mask_path", "mask_sha256", "review_png_path", "review_json_path"]
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _copy_link(out_dir, in_dir, derived_rel):
    """派生副本（相对 input 目录）→ 相对本输出目录的相邻链接，不泄漏绝对路径。"""
    target = os.path.join(in_dir, derived_rel.replace("/", os.sep))
    return _rel(target, out_dir)


def _write_index_html(path, out_dir, in_dir, gallery, audit):
    c = audit["counts"]
    e = html.escape
    lines = [
        "<!DOCTYPE html>",
        '<html lang="zh-CN"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        "<title>RGB 训练前审计与离线可视复核包（%s）</title>" % TASK_ID,
        "<style>body{font-family:sans-serif;margin:16px;line-height:1.6;max-width:1100px;",
        "overflow-wrap:anywhere;word-break:break-word}",
        "img{max-width:100%;height:auto;border:1px solid #999}",
        ".card{border:1px solid #ccc;border-radius:6px;padding:8px;margin:12px 0;",
        "overflow-wrap:anywhere}",
        ".muted{color:#666}code{background:#f4f4f4;padding:1px 4px;overflow-wrap:anywhere}",
        "table{border-collapse:collapse;max-width:100%}",
        "td,th{border:1px solid #ccc;padding:4px;overflow-wrap:anywhere}</style>",
        "</head><body>",
        "<h1>RGB 训练前审计与离线可视复核包</h1>",
        "<p>来源任务 <code>%s</code>；本页为<strong>离线</strong>静态页，不联网、无 CDN、无外部服务。"
        "所有样本仅标“待复核”，本页<strong>不保存</strong>修正或确认。</p>" % SOURCE_TASK_ID,
        "<p class='muted'>本页显示的是派生候选 mask 叠加与标注轮廓，<strong>不是模型预测</strong>，"
        "也不代表修标/确认成功。</p>",
        "<h2>审计计数</h2>",
        "<table><tr><th>项目</th><th>数量</th></tr>",
        "<tr><td>输入记录</td><td>%d</td></tr>" % c["input"],
        "<tr><td>候选 candidate</td><td>%d</td></tr>" % c["candidate"],
        "<tr><td>待复核 review</td><td>%d</td></tr>" % c["review"],
        "</table>",
        "<p><b>training_ready=%s；semantic_reviewed=%s；split_created=%s</b></p>"
        % (audit["training_ready"], audit["semantic_reviewed"], audit["split_created"]),
        "<h2>图例</h2>",
        "<ul>",
        "<li><span style='color:#00c800'>■</span> 0 = Field 田块（也是候选叠加色）</li>",
        "<li><span style='color:#f0c800'>■</span> 1 = Field ridge 田埂（含派生归一 Fieldridge）</li>",
        "<li><span style='color:#e62828'>■</span> 2 = Road 道路</li>",
        "<li>255 = 忽略（未画/跨类重叠/未确认）：叠加时保持原图可见。</li>",
        "<li>0 不是背景。</li>",
        "</ul>",
    ]

    reviews = [s for s in gallery if s["kind"] == "review"]
    cands = [s for s in gallery if s["kind"] == "candidate"]
    lines.append("<h2>待复核样本（%d）</h2>" % len(reviews))
    for s in reviews:
        lines.append("<div class='card'>")
        lines.append("<h3>%s <span class='muted'>(%s)</span></h3>"
                     % (e(s["sample_id"]), e(s["dataset"])))
        lines.append("<img src='%s' alt='%s'>" % (e(_href(s["preview_path"])), e(s["sample_id"])))
        lines.append("<p class='muted'>左=原图 / 右=原图+轮廓，粗线为问题 shape（标 0 起索引），"
                     "尚待复核。</p>")
        lines.append("<p class='muted'>source json=<code>%s</code> image=<code>%s</code></p>"
                     % (e(s["source"]["json_path"]), e(s["source"]["image_path"])))
        lines.append("<ul>")
        for r in s["derived_refs"]:
            lines.append("<li><a href='%s'>%s</a>（%s）</li>"
                         % (e(_href(_copy_link(out_dir, in_dir, r["path"]))),
                            e(r["path"]), e(r["basis"])))
        lines.append("</ul>")
        lines.append("<p>问题（shape_index 从 0 起）：</p><ul>")
        for p in s["problems"]:
            lines.append("<li>type=<code>%s</code>, shape_index=<code>%s</code>, "
                         "label=<code>%s</code>, detail=<code>%s</code></li>"
                         % (e(str(p["type"])), e(str(p["shape_index"])),
                            e(str(p["label"])), e(json.dumps(p["detail"], ensure_ascii=False))))
        lines.append("</ul></div>")

    lines.append("<h2>候选叠加样例（最多 %d，实际 %d）</h2>"
                 % (MAX_CANDIDATE_SAMPLES, len(cands)))
    for s in cands:
        lines.append("<div class='card'>")
        lines.append("<h3>%s <span class='muted'>(%s)</span></h3>"
                     % (e(s["sample_id"]), e(s["dataset"])))
        lines.append("<img src='%s' alt='%s'>" % (e(_href(s["preview_path"])), e(s["sample_id"])))
        lines.append("<p class='muted'>左=原图 / 右=候选彩色掩膜叠加（255 透明、类 0 显示），"
                     "尚待复核。</p>")
        lines.append("<p class='muted'>source json=<code>%s</code> image=<code>%s</code></p>"
                     % (e(s["source"]["json_path"]), e(s["source"]["image_path"])))
        cp = s["class_pixels"]
        lines.append("<p>class0=%d class1=%d class2=%d valid=%d ignore=%d ratio=%.4f</p>"
                     % (cp["0"], cp["1"], cp["2"], s["valid_pixels"],
                        s["ignore_pixels"], s["ignore_ratio"]))
        for r in s["derived_refs"]:
            lines.append("<p><a href='%s'>%s</a>（%s）</p>"
                         % (e(_href(_copy_link(out_dir, in_dir, r["path"]))),
                            e(r["path"]), e(r["basis"])))
        lines.append("</div>")

    lines.append("<h2>下一步</h2><ol>")
    for t in audit["todos"]:
        lines.append("<li>%s</li>" % e(t))
    lines.append("</ol>")
    lines.append("<p class='muted'>打开本页查看问题；在标注软件中另存修正副本后需<b>重新质检</b>；"
                 "本页不保存修正/确认。</p>")
    lines.append("</body></html>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _write_readme(path, audit, gallery):
    c = audit["counts"]
    lines = [
        "# RGB 训练前审计与离线可视复核包（零基础说明）",
        "",
        "本目录由 `scripts/labels/build_review_pack.py` 生成，是**派生结果**，"
        "原图/标注/旧产物均未修改。",
        "",
        "## 1. 输入与前置",
        "",
        "- 输入：`%s`（005 候选目录，含 `manifest.json`、`masks/`、`review/`）。" % audit["input_dir"],
        "- 本工具只适用于 `candidate` / `review` 批次；`empty` / `unsupported` 明确拒绝。",
        "- 全部来源校验在创建输出目录前完成；任一来源失败即整体失败，不产生半成品目录。",
        "",
        "## 2. 操作与预期",
        "",
        "1. 用浏览器打开 `index.html`：左侧为待复核样本（原图+标注轮廓，问题 shape 高亮并标索引），",
        "   下方为最多 %d 张候选叠加样例（原图+彩色候选 mask）。" % MAX_CANDIDATE_SAMPLES,
        "2. 查看 `audit.json`：`training_ready=false` / `semantic_reviewed=false` / `split_created=false`。",
        "3. 查看 `records.csv`：逐图来源 SHA、状态、像素数、ignore 比例、别名/问题条数。",
        "4. 机器查找样本对应关系请看 `gallery.json`，不要靠解析 HTML 猜。",
        "",
        "## 3. 计数",
        "",
        "| 项目 | 数量 |",
        "|---|---:|",
        "| 输入记录 | %d |" % c["input"],
        "| 候选 | %d |" % c["candidate"],
        "| 待复核 | %d |" % c["review"],
        "",
        "## 4. 下一步",
        "",
        "1. 打开 `index.html` 看问题；在标注软件中**另存**修正副本后**重新质检**。",
        "2. 本页**不保存**修正或确认，仅显示“待复核”。",
        "3. 全体语义确认、别名口径、来源空间/时相关系与训练划分仍待定。",
        "",
        "## 5. 边界",
        "",
        "- 候选 mask 是机械派生，不代表真值；不报精度、真实地块数或平方米。",
        "- 本页叠加是派生 mask，不是模型预测；**未实际打开 Labelme GUI**。",
        "- 未划分 train/val/test，不训练、不切分。",
        "",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main(argv=None):
    parser = argparse.ArgumentParser(description="RGB 训练前审计与离线可视复核包")
    parser.add_argument("--input", default=None, help="005 候选目录（须在 reports/labels 内）")
    parser.add_argument("--output", default=None, help="全新输出目录（须在 reports/labels 内）")
    args = parser.parse_args(argv)
    root = _DEFAULT_ROOT
    try:
        res = build_pack(args.input, args.output, root)
    except BuildError as ex:
        sys.stderr.write("[BuildError] %s\n" % ex)
        return 2
    print(res["output_dir"])
    c = res["counts"]
    print("input=%d candidate=%d review=%d visual_candidate=%d visual_review=%d"
          % (c["input"], c["candidate"], c["review"],
             res["candidate_visual_samples"], res["review_visual_samples"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
