# -*- coding: utf-8 -*-
"""DATA-20261016-005：RGB 候选掩膜与待复核副本准备工具（标准库 + 现有 Pillow）。

在 DATA-20261016-004 初版基础上，按 Codex 初审修复边界问题（不重放 004）：
  1. sample_id 仅接受便携安全组件 `[A-Za-z0-9_-]{1,80}` 且拒绝 Windows 保留名、
     大小写折叠重复 id；JSON/图像来源按 resolve+normcase 去重。
  2. 完整索引与**所有**来源的 SHA/bytes/解析/属性/坐标检查，全部在**创建输出目录之前**
     完成；任一（含最后一个）来源出错即抛 `PrepareError`，输出目录不产生。允许两遍读取，
     但不会把所有 903 条内嵌 `imageData` 常驻内存。
  3. 有效像素判定只看**互斥** `class_pixels` 之和；满幅跨类叠加全 255 记为 empty，非 candidate。
  4. JSON `imagePath` 必须**严格等于**实际 TIFF basename；JSON 宽高必须为正整数（拒绝 bool）。
  5. 派生产物路径（mask/review 副本）记录为**相对输出目录**；源路径仍相对项目根。
  6. 有效像素口径回归测试改为独立手算（勿改正确的栅格化实现）。
  7. policy 中“不闭合”歧义改为：不改写点序/不追加源坐标，Pillow 填充时隐式闭合多边形。

本工具**只读**原始标注与图像，生成**派生**候选索引图与待人工复核副本，
绝不修改、移动或删除任何原始文件。所有候选均为“待语义复核、未训练就绪”。

用法（由 Codex 运行；WorkBuddy 不执行）：
    python -B scripts/labels/prepare_candidate.py [--input <索引>] [--output <全新目录>]

- 默认输入 `<根>/reports/labels/DATA-20261016-004-input.json`（005 修复沿用 004 索引）。
- 自定义输入必须位于 `<根>/reports/labels` 内；其引用的原始文件必须落在
  `<根>/邓`、`<根>/hy0713 (1)` 或 `<根>/reports/labels`（fixture）内。
- 输出只能写到 `<根>/reports/labels` 内一个**全新**目录；已存在即拒绝，不覆盖/不删除。

类别：0=Field 田块，1=Field ridge 田埂，2=Road 道路，255=忽略/未画/冲突/未确认。
无背景类别。`Fieldridge`→`Field ridge` 仅为**待复核的拼写归一假设**，不是已确认真值。
"""

import argparse
import copy
import csv
import hashlib
import io
import json
import math
import os
import random
import re
import sys
import time

from PIL import Image, ImageChops, ImageDraw
import PIL

TASK_ID = "DATA-20261016-005"
# 005 复用 004 的输入索引（不新建索引文件）。
DEFAULT_INPUT_REL = "reports/labels/DATA-20261016-004-input.json"
EPS = 1e-6

# 类别：0 不是背景；255 为忽略/未画/冲突/未确认。
CLASS_FIELD = 0
CLASS_RIDGE = 1
CLASS_ROAD = 2
CLASS_IGNORE = 255

LABEL_TO_CLASS = {
    "Field": CLASS_FIELD,
    "Field ridge": CLASS_RIDGE,
    "Fieldridge": CLASS_RIDGE,   # 待复核的拼写归一假设
    "Road": CLASS_ROAD,
}
# 仅记录“原 label -> 派生 label”的拼写归一，用于日志与策略说明。
ALIAS_MAP = {"Fieldridge": "Field ridge"}

CLASS_NAMES = {
    "0": "Field 田块",
    "1": "Field ridge 田埂",
    "2": "Road 道路",
    "255": "ignore 忽略/未画/冲突/未确认",
}

ALLOWED_SOURCE_DIRS = ("邓", "hy0713 (1)")
REQUIRED_FIELDS = (
    "sample_id", "dataset", "json_path", "json_sha256", "json_bytes",
    "image_path", "image_sha256", "image_bytes", "width", "height",
)
SHA_RE = re.compile(r"[0-9a-f]{64}")
# 便携安全文件名组件：字母/数字/下划线/连字符，长度 1..80。
SAFE_ID_RE = re.compile(r"[A-Za-z0-9_-]{1,80}")
# Windows 设备保留名（大小写不敏感）。
WIN_RESERVED = ({"CON", "PRN", "AUX", "NUL"}
                | {"COM%d" % i for i in range(1, 10)}
                | {"LPT%d" % i for i in range(1, 10)})

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))


class PrepareError(Exception):
    """输入/范围/输出校验失败：非零退出，绝不产生“成功”候选。"""


# --------------------------------------------------------------------------
# 基础工具
# --------------------------------------------------------------------------
def _find_root():
    return _DEFAULT_ROOT


def _is_number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _is_int(v):
    return isinstance(v, int) and not isinstance(v, bool)


def _is_sha256(s):
    return isinstance(s, str) and bool(SHA_RE.fullmatch(s))


def _within(child, parent):
    child = os.path.abspath(child)
    parent = os.path.abspath(parent)
    return child == parent or child.startswith(parent + os.sep)


def _basename(p):
    """取路径末段，兼容 '/' 与 '\\'。"""
    return p.replace("\\", "/").rsplit("/", 1)[-1]


def _sha256_of_bytes(b):
    return hashlib.sha256(b).hexdigest()


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


def _rel(root, path):
    return os.path.relpath(path, root).replace("\\", "/")


def _validate_sample_id(sid):
    """sample_id 必须是便携安全组件，且不能命中 Windows 保留名。"""
    if not isinstance(sid, str) or not SAFE_ID_RE.fullmatch(sid):
        raise PrepareError(
            "sample_id 非法（须匹配 [A-Za-z0-9_-]{1,80} 的便携安全组件）：%r" % (sid,))
    if sid.upper() in WIN_RESERVED:
        raise PrepareError("sample_id 命中 Windows 保留名：%r" % sid)


def _source_key(rel, root):
    """来源去重键：resolve（真实路径）+ normcase（大小写折叠）。"""
    joined = rel if os.path.isabs(rel) else os.path.join(root, rel)
    return os.path.normcase(os.path.realpath(joined))


# --------------------------------------------------------------------------
# 范围校验
# --------------------------------------------------------------------------
def _labels_dir(root):
    return os.path.join(root, "reports", "labels")


def _resolve_input(input_path, root):
    if input_path is None:
        input_path = DEFAULT_INPUT_REL
    p = input_path if os.path.isabs(input_path) else os.path.join(root, input_path)
    real = os.path.realpath(p)
    if not _within(real, os.path.realpath(_labels_dir(root))):
        raise PrepareError("输入索引必须位于 reports/labels 内：%s" % input_path)
    if not os.path.isfile(real):
        raise PrepareError("输入索引不存在：%s" % real)
    return real


def _resolve_source(rel, root, ctx):
    if not isinstance(rel, str) or not rel:
        raise PrepareError("%s：路径字段非字符串或为空" % ctx)
    if os.path.isabs(rel):
        raise PrepareError("%s：拒绝绝对路径 %r" % (ctx, rel))
    norm = rel.replace("\\", "/")
    if any(part == ".." for part in norm.split("/")):
        raise PrepareError("%s：拒绝含 '..' 的路径 %r" % (ctx, rel))
    real = os.path.realpath(os.path.join(root, norm))
    allowed = [os.path.realpath(os.path.join(root, d)) for d in ALLOWED_SOURCE_DIRS]
    allowed.append(os.path.realpath(_labels_dir(root)))
    if not any(_within(real, a) for a in allowed):
        raise PrepareError("%s：路径越界（仅允许 邓、hy0713 (1) 或 reports/labels 内）：%r"
                           % (ctx, rel))
    if not os.path.isfile(real):
        raise PrepareError("%s：原始文件不存在：%s" % (ctx, real))
    return real


def _resolve_output(output_dir, root):
    """解析目标输出目录路径（**不创建**）。已存在即拒绝。"""
    labels = os.path.realpath(_labels_dir(root))
    if output_dir is None:
        while True:
            name = "%s-%s-%s" % (TASK_ID, time.strftime("%Y%m%d-%H%M%S"),
                                 "%06x" % random.randrange(0xFFFFFF))
            candidate = os.path.join(labels, name)
            if not os.path.exists(candidate):
                return candidate
    p = output_dir if os.path.isabs(output_dir) else os.path.join(root, output_dir)
    parent = os.path.realpath(os.path.dirname(os.path.abspath(p)))
    if not _within(parent, labels):
        raise PrepareError("输出目录必须位于 reports/labels 内：%s" % output_dir)
    if os.path.exists(p):
        raise PrepareError("输出目录已存在，拒绝覆盖/删除：%s" % p)
    return os.path.abspath(p)


# --------------------------------------------------------------------------
# 索引与记录校验
# --------------------------------------------------------------------------
def _load_index(input_path):
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:  # noqa: BLE001 - 明确转成 PrepareError
        raise PrepareError("索引 JSON 解析失败：%r" % e)
    if not isinstance(data, dict):
        raise PrepareError("索引根不是对象")
    if data.get("schema_version") != "candidate-input/1":
        raise PrepareError("未知 schema_version：%r" % data.get("schema_version"))
    records = data.get("records")
    if not isinstance(records, list) or not records:
        raise PrepareError("索引 records 缺失或为空")
    return records


def _validate_record_fields(rec, idx):
    if not isinstance(rec, dict):
        raise PrepareError("record[%d] 不是对象" % idx)
    for key in REQUIRED_FIELDS:
        if key not in rec:
            raise PrepareError("record[%d] 缺字段 %s" % (idx, key))
    _validate_sample_id(rec["sample_id"])
    if not isinstance(rec["dataset"], str):
        raise PrepareError("record[%d].dataset 非法" % idx)
    for key in ("json_path", "image_path"):
        if not isinstance(rec[key], str) or not rec[key]:
            raise PrepareError("record[%d].%s 非法" % (idx, key))
    if os.path.splitext(rec["json_path"])[1].lower() != ".json":
        raise PrepareError("record[%d].json_path 非 .json" % idx)
    if os.path.splitext(rec["image_path"])[1].lower() not in (".tif", ".tiff"):
        raise PrepareError("record[%d].image_path 非 .tif/.tiff" % idx)
    for key in ("json_sha256", "image_sha256"):
        if not _is_sha256(rec[key]):
            raise PrepareError("record[%d].%s 非法 SHA256" % (idx, key))
    for key in ("json_bytes", "image_bytes", "width", "height"):
        if not _is_int(rec[key]) or rec[key] <= 0:
            raise PrepareError("record[%d].%s 非法整数（含 bool/负数/零）" % (idx, key))


def _clean_points(shape, ctx):
    """类型级校验：非列表/非二元/布尔/非有限坐标一律拒绝（整体失败）。"""
    pts = shape.get("points")
    if not isinstance(pts, list):
        raise PrepareError("%s：points 不是列表" % ctx)
    out = []
    for k, pt in enumerate(pts):
        if (not isinstance(pt, (list, tuple))) or len(pt) != 2:
            raise PrepareError("%s：第 %d 点结构非法" % (ctx, k))
        x, y = pt[0], pt[1]
        if isinstance(x, bool) or isinstance(y, bool):
            raise PrepareError("%s：第 %d 点含 bool" % (ctx, k))
        if not (_is_number(x) and _is_number(y)):
            raise PrepareError("%s：第 %d 点非数值" % (ctx, k))
        if not (math.isfinite(x) and math.isfinite(y)):
            raise PrepareError("%s：第 %d 点含非有限坐标" % (ctx, k))
        out.append((float(x), float(y)))
    return out


# --------------------------------------------------------------------------
# 几何（参考 WB002 已验证算法：orientation 阈值 1e-6 px²，双方向一致）
# --------------------------------------------------------------------------
def _orient(ax, ay, bx, by, cx, cy):
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)


def _sgn(v):
    if v > EPS:
        return 1
    if v < -EPS:
        return -1
    return 0


def _proper_cross(p1, p2, p3, p4):
    d1 = _sgn(_orient(p3[0], p3[1], p4[0], p4[1], p1[0], p1[1]))
    d2 = _sgn(_orient(p3[0], p3[1], p4[0], p4[1], p2[0], p2[1]))
    d3 = _sgn(_orient(p1[0], p1[1], p2[0], p2[1], p3[0], p3[1]))
    d4 = _sgn(_orient(p1[0], p1[1], p2[0], p2[1], p4[0], p4[1]))
    return d1 * d2 < 0 and d3 * d4 < 0


def _self_crossings(pts):
    n = len(pts)
    if n < 4:
        return 0
    edges = []
    for i in range(n):
        a = pts[i]
        b = pts[(i + 1) % n]
        if a == b:
            continue  # 零长/连续重复边不参与
        edges.append((i, a, b))
    cnt = 0
    m = len(edges)
    for x in range(m):
        i, a, b = edges[x]
        for y in range(x + 1, m):
            j, c, d = edges[y]
            if j == i + 1 or (i == 0 and j == n - 1):
                continue  # 相邻边/首尾边共享顶点
            if _proper_cross(a, b, c, d):
                cnt += 1
    return cnt


def _canvas_violations(pts, w, h):
    out = []
    for i, (x, y) in enumerate(pts):
        if x < -EPS or x > w + EPS or y < -EPS or y > h + EPS:
            out.append({"point_index": i, "x": x, "y": y,
                        "excess": max(-x, x - w, -y, y - h)})
    return out


# --------------------------------------------------------------------------
# 掩膜栅格化
# --------------------------------------------------------------------------
def _rasterize(shape_records, w, h):
    masks = {c: Image.new("L", (w, h), 0) for c in (CLASS_FIELD, CLASS_RIDGE, CLASS_ROAD)}
    for sr in shape_records:
        cls = sr["derived_class"]
        if cls is None:
            continue
        pts = sr["points"]
        if len(set(pts)) < 3:
            continue
        ImageDraw.Draw(masks[cls]).polygon(pts, fill=255)

    b = {c: masks[c].point(lambda v: 1 if v else 0)
         for c in (CLASS_FIELD, CLASS_RIDGE, CLASS_ROAD)}
    cov = ImageChops.add(ImageChops.add(b[CLASS_FIELD], b[CLASS_RIDGE]), b[CLASS_ROAD])
    idmap = ImageChops.add(
        ImageChops.add(
            b[CLASS_FIELD].point(lambda v: 0),
            b[CLASS_RIDGE].point(lambda v: 1 if v else 0)),
        b[CLASS_ROAD].point(lambda v: 2 if v else 0))
    single = cov.point(lambda v: 255 if v == 1 else 0)
    not_single = cov.point(lambda v: 255 if v != 1 else 0)
    final = ImageChops.add(ImageChops.multiply(idmap, single), not_single)

    hist = cov.histogram()
    unpainted = hist[0]
    conflict = sum(hist[2:])
    covered = {c: masks[c].histogram()[255] for c in (CLASS_FIELD, CLASS_RIDGE, CLASS_ROAD)}
    class_pixels = {c: ImageChops.multiply(single, masks[c]).histogram()[255]
                    for c in (CLASS_FIELD, CLASS_RIDGE, CLASS_ROAD)}
    stats = {
        "denominator": w * h,
        "unpainted_pixels": unpainted,
        "conflict_pixels": conflict,
        "ignore_pixels": unpainted + conflict,
        "class_covered_pixels": {str(c): covered[c] for c in covered},
        "class_pixels": {str(c): class_pixels[c] for c in class_pixels},
    }
    return final, stats


def _valid_pixel_total(pixel_stats):
    """有效像素 = 互斥类别像素（class_pixels）之和；不含跨类冲突/未画。"""
    return sum(pixel_stats["class_pixels"].values())


# --------------------------------------------------------------------------
# 单记录校验（在创建输出目录之前调用，只读、不写盘）
# --------------------------------------------------------------------------
def _scan_record(rec, root):
    sid = rec["sample_id"]
    jpath = _resolve_source(rec["json_path"], root, "record[%s].json_path" % sid)
    ipath = _resolve_source(rec["image_path"], root, "record[%s].image_path" % sid)

    # 原始文件 SHA/bytes 校验（索引被改即拒绝）。
    jbytes = _read_bytes(jpath)
    ibytes = _read_bytes(ipath)
    if len(jbytes) != rec["json_bytes"] or _sha256_of_bytes(jbytes) != rec["json_sha256"]:
        raise PrepareError("%s：JSON 摘要/大小与索引不符" % sid)
    if len(ibytes) != rec["image_bytes"] or _sha256_of_bytes(ibytes) != rec["image_sha256"]:
        raise PrepareError("%s：图像摘要/大小与索引不符" % sid)

    try:
        data = json.loads(jbytes.decode("utf-8"))
    except Exception as e:  # noqa: BLE001
        raise PrepareError("%s：JSON 解析失败 %r" % (sid, e))
    if not isinstance(data, dict):
        raise PrepareError("%s：JSON 根不是对象" % sid)
    shapes = data.get("shapes", [])
    if not isinstance(shapes, list):
        raise PrepareError("%s：shapes 不是列表" % sid)

    # imagePath 必须严格等于实际 TIFF basename（拒绝同 stem 异后缀或外来路径前缀）。
    ip_field = data.get("imagePath")
    img_base = _basename(rec["image_path"])
    if not isinstance(ip_field, str) or ip_field != img_base:
        raise PrepareError("%s：JSON imagePath 必须严格等于图像 basename（%r != %r）"
                           % (sid, ip_field, img_base))

    # JSON 声明尺寸必须为正整数（拒绝 bool/非正）且与索引一致。
    jw, jh = data.get("imageWidth"), data.get("imageHeight")
    if not (_is_int(jw) and _is_int(jh) and jw > 0 and jh > 0):
        raise PrepareError("%s：JSON 声明宽高必须为正整数（拒绝 bool/非正）" % sid)
    if jw != rec["width"] or jh != rec["height"]:
        raise PrepareError("%s：JSON 声明尺寸与索引不符" % sid)

    # 图像尺寸/RGB。
    try:
        img = Image.open(io.BytesIO(ibytes))
        img.load()
    except Exception as e:  # noqa: BLE001
        raise PrepareError("%s：图像解码失败 %r" % (sid, e))
    if img.mode != "RGB":
        raise PrepareError("%s：图像非 RGB（mode=%s）" % (sid, img.mode))
    if img.size != (rec["width"], rec["height"]):
        raise PrepareError("%s：图像尺寸与索引/JSON 不符 %s" % (sid, list(img.size)))
    w, h = rec["width"], rec["height"]
    img.close()

    # 逐 shape 几何分析。
    reasons, unsupported, shape_records, label_map = [], [], [], []
    label_counts, alias_entries, unknown_entries = {}, [], []
    for si, sh in enumerate(shapes):
        if not isinstance(sh, dict):
            raise PrepareError("%s：shape[%d] 不是对象" % (sid, si))
        label = sh.get("label")
        if not isinstance(label, str):
            raise PrepareError("%s：shape[%d].label 非字符串" % (sid, si))
        label_counts[label] = label_counts.get(label, 0) + 1
        stype = sh.get("shape_type")
        if stype != "polygon":
            unsupported.append({"reason": "shape_type_not_polygon",
                                "shape_index": si, "shape_type": stype})
            continue
        pts = _clean_points(sh, "%s shape[%d]" % (sid, si))
        if label in LABEL_TO_CLASS:
            cls = LABEL_TO_CLASS[label]
        else:
            cls = None
            unsupported.append({"reason": "unknown_label",
                                "shape_index": si, "label": label})
            unknown_entries.append({"sample_id": sid, "shape_index": si, "label": label})

        alias = None
        if label in ALIAS_MAP:
            alias = {"shape_index": si, "from": label, "to": ALIAS_MAP[label]}
            alias_entries.append({"sample_id": sid, "shape_index": si,
                                  "from": label, "to": ALIAS_MAP[label]})

        distinct = set(pts)
        if len(distinct) < 3:
            reasons.append({"type": "fewer_than_3_distinct_points", "shape_index": si,
                            "label": label, "n_distinct": len(distinct)})
        else:
            sc = _self_crossings(pts)
            if sc:
                reasons.append({"type": "self_intersection", "shape_index": si,
                                "label": label, "crossing_pairs": sc})
            viol = _canvas_violations(pts, w, h)
            if viol:
                reasons.append({"type": "canvas_oob", "shape_index": si,
                                "label": label, "points": viol})
        label_map.append({"shape_index": si, "original_label": label,
                          "derived_class": cls})
        shape_records.append({"shape_index": si, "original_label": label,
                              "derived_class": cls, "alias": alias, "points": pts})

    if unsupported:
        status = "unsupported"
    elif reasons:
        status = "review"
    else:
        status = None  # candidate/empty 在写出阶段按互斥像素判定

    return {
        "rec": rec, "sid": sid, "w": w, "h": h, "status": status,
        "shape_records": shape_records, "label_map": label_map,
        "reasons": reasons, "unsupported": unsupported,
        "alias_entries": alias_entries, "unknown_entries": unknown_entries,
        "label_counts": label_counts,
    }


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------
def prepare(input_path=None, output_dir=None, root=None):
    root = os.path.abspath(root) if root else _find_root()
    index_path = _resolve_input(input_path, root)
    out_dir = _resolve_output(output_dir, root)   # 仅解析，不创建

    records = _load_index(index_path)

    # ---- 校验阶段：任何失败都在创建输出目录之前抛出 ----
    seen_ids, seen_json, seen_image = {}, {}, {}
    prepared = []
    for idx, rec in enumerate(records):
        _validate_record_fields(rec, idx)
        sid = rec["sample_id"]
        id_key = sid.casefold()  # 大小写折叠，拒绝仅大小写不同的重复 id
        if id_key in seen_ids:
            raise PrepareError("sample_id 大小写折叠后重复：%s（首见 record[%d]）"
                               % (sid, seen_ids[id_key]))
        seen_ids[id_key] = idx
        for fld, seen in (("json_path", seen_json), ("image_path", seen_image)):
            key = _source_key(rec[fld], root)  # resolve + normcase
            if key in seen:
                raise PrepareError("%s 重复来源引用（resolve+normcase 相同）：%s"
                                   % (fld, rec[fld]))
            seen[key] = idx
        # 单记录完整校验（SHA/bytes/解析/ imagePath /宽高/RGB/坐标/几何）。
        prepared.append(_scan_record(rec, root))

    # 至此完整索引与所有来源均已核验；此时才创建输出目录。
    try:
        os.makedirs(out_dir, exist_ok=False)
    except FileExistsError:
        raise PrepareError("输出目录已存在，拒绝覆盖：%s" % out_dir)
    masks_dir = os.path.join(out_dir, "masks")
    review_dir = os.path.join(out_dir, "review")
    os.makedirs(masks_dir)
    os.makedirs(review_dir)

    counts = {"input": 0, "candidate": 0, "review": 0, "empty": 0, "unsupported": 0}
    label_shape_counts = {}
    alias_log = []
    unknown_label_log = []
    review_rows = []
    manifest = []
    pixel_totals = {"class_pixels": {"0": 0, "1": 0, "2": 0},
                    "class_covered_pixels": {"0": 0, "1": 0, "2": 0},
                    "ignore_pixels": 0, "conflict_pixels": 0, "unpainted_pixels": 0}

    # ---- 写出阶段 ----
    for p in prepared:
        rec = p["rec"]
        sid = p["sid"]
        w, h = p["w"], p["h"]
        counts["input"] += 1
        for lab, n in p["label_counts"].items():
            label_shape_counts[lab] = label_shape_counts.get(lab, 0) + n
        alias_log.extend(p["alias_entries"])
        unknown_label_log.extend(p["unknown_entries"])

        mask_path = mask_sha = None
        rpng_path = rjson_path = rpng_sha = rjson_sha = None
        pixel_stats = None
        derived_record = []

        if p["status"] is None:
            final, pixel_stats = _rasterize(p["shape_records"], w, h)
            # 有效像素只看互斥 class_pixels；满幅跨类叠加全 255 视为 empty。
            status = "candidate" if _valid_pixel_total(pixel_stats) > 0 else "empty"
            mpath = os.path.join(masks_dir, sid + ".png")
            final.save(mpath, format="PNG")
            # 派生产物路径：相对输出目录。
            mask_path = "masks/%s.png" % sid
            mask_sha = _sha256_of(mpath)
        else:
            status = p["status"]

        if status == "review":
            ipath = _resolve_source(rec["image_path"], root, "record[%s].image_path" % sid)
            rpng = os.path.join(review_dir, sid + ".png")
            with Image.open(ipath) as im:
                im.load()
                im.save(rpng, format="PNG")
            rpng_path = "review/%s.png" % sid
            rpng_sha = _sha256_of(rpng)

            jpath = _resolve_source(rec["json_path"], root, "record[%s].json_path" % sid)
            data = json.loads(_read_bytes(jpath).decode("utf-8"))
            newdata = copy.deepcopy(data)
            newdata["imagePath"] = sid + ".png"
            newdata["imageData"] = None
            derived_changes = []
            for si2, sh2 in enumerate(newdata.get("shapes", [])):
                lab = sh2.get("label")
                if lab in ALIAS_MAP:
                    sh2["label"] = ALIAS_MAP[lab]
                    derived_changes.append({"shape_index": si2, "from": lab,
                                            "to": ALIAS_MAP[lab]})
            rjson = os.path.join(review_dir, sid + ".json")
            _write_json(rjson, newdata)
            rjson_path = "review/%s.json" % sid
            rjson_sha = _sha256_of(rjson)
            derived_record = derived_changes

            for r in p["reasons"]:
                detail = {k: v for k, v in r.items() if k != "type"}
                review_rows.append({
                    "source_id": sid, "dataset": rec["dataset"],
                    "source_file": os.path.basename(rec["json_path"]),
                    "shape_index": r.get("shape_index"),
                    "problem_type": r["type"],
                    "detail": json.dumps(detail, ensure_ascii=False, sort_keys=True),
                    "review_png": rpng_path, "review_json": rjson_path,
                    "review_status": "pending_semantic_review",
                })

        counts[status] += 1
        entry = {
            "sample_id": sid, "dataset": rec["dataset"],
            "json_path": rec["json_path"], "image_path": rec["image_path"],
            "json_sha256": rec["json_sha256"], "json_bytes": rec["json_bytes"],
            "image_sha256": rec["image_sha256"], "image_bytes": rec["image_bytes"],
            "width": w, "height": h,
            "status": status,
            "label_mapping": p["label_map"],
            "alias_normalized": p["alias_entries"],
            "unsupported_reasons": p["unsupported"],
            "review_reasons": p["reasons"],
            "derived_json_changes": derived_record,
            "mask_path": mask_path, "mask_sha256": mask_sha,
            "review_png_path": rpng_path, "review_png_sha256": rpng_sha,
            "review_json_path": rjson_path, "review_json_sha256": rjson_sha,
            "pixel_stats": pixel_stats,
            "semantic_reviewed": False,
            "not_training_ready": True,
        }
        manifest.append(entry)

        if pixel_stats:
            for c in ("0", "1", "2"):
                pixel_totals["class_pixels"][c] += pixel_stats["class_pixels"][c]
                pixel_totals["class_covered_pixels"][c] += pixel_stats["class_covered_pixels"][c]
            pixel_totals["ignore_pixels"] += pixel_stats["ignore_pixels"]
            pixel_totals["conflict_pixels"] += pixel_stats["conflict_pixels"]
            pixel_totals["unpainted_pixels"] += pixel_stats["unpainted_pixels"]

    created = time.strftime("%Y-%m-%dT%H:%M:%S")

    path_bases = {
        "source_paths": "relative to project root",
        "derived_paths": "relative to this output directory",
        "note": "manifest/review.csv 中 json_path/image_path 相对项目根；mask_path 与 review "
                "副本路径相对本输出目录（如 masks/<id>.png、review/<id>.png）。拷贝产物目录后，"
                "源路径需回到项目根解释，派生产物路径相对该目录解释。",
    }

    policy = {
        "task_id": TASK_ID, "created": created,
        "pillow_version": PIL.__version__,
        "class_ids": CLASS_NAMES,
        "no_background_class": True,
        "class_id_note": "0 是田块，不是背景；未画/冲突/未确认一律 255。",
        "alias_assumption": {
            "map": {"Fieldridge": "Field ridge"},
            "status": "pending_semantic_review",
            "note": "Fieldridge→Field ridge 仅为派生层拼写归一假设，保留原 label 与修改记录，"
                    "不视为所有真值已确认。无沟渠类别。",
        },
        "rasterization": {
            "library": "Pillow ImageDraw.polygon",
            "pillow_version": PIL.__version__,
            "fill_value": 255,
            "coordinate_handling": "不重排、不改写点序，也不追加或回写任何源坐标；"
                                   "Pillow 填充时对多边形做隐式闭合（自动连接末点与首点）。"
                                   "输入坐标原样使用（含 float），不四舍五入。",
            "rule": "同类多边形取并集；不同类别覆盖>1 一律 255；未画区域 255；"
                    "结果与绘制顺序无关。不宣称精确面积或真值。",
            "valid_pixel_rule": "有效像素只取互斥类别像素（class_pixels）之和；"
                                "满幅跨类叠加全 255 记为 empty，不是 candidate。",
        },
        "overflow_overlap_policy": "same-class union; cross-class overlap -> 255; unpainted -> 255",
        "geometry_policy": {
            "epsilon": EPS,
            "canvas_bounds": "[0,W]/[0,H]（画布边界口径）",
            "orientation_threshold_px2": EPS,
            "self_intersection": "非相邻边严格内部穿越，双向 orientation 判号一致（参考 WB002）。",
            "review_if": "任一 shape 自交、画布越界或不足3个不同点 -> 整份进入 review，不生成训练 mask。",
            "duplicate_or_closed_points": "只重复点/显式闭合点不自动判错，不重排/不回写坐标；"
                                          "Pillow 填充时隐式闭合多边形。",
        },
        "sample_id_policy": {
            "pattern": "[A-Za-z0-9_-]{1,80}",
            "reject_windows_reserved": sorted(WIN_RESERVED),
            "reject_casefold_duplicates": True,
            "reject_duplicate_source_refs": "resolve + normcase 相同即拒绝",
        },
        "pre_output_validation": "完整索引与所有来源的 SHA/bytes/解析/属性/坐标检查均在创建输出目录前完成；"
                                 "任一来源失败即 PrepareError 且不产生输出目录。",
        "path_bases": path_bases,
        "unsupported_policy": "存在未知 label 或非 polygon shape -> 整份排除并记录，不默当背景。",
        "output_schema": {
            "policy.json": "策略与 schema",
            "manifest.json": "全部记录：源路径/SHA/bytes、原→派生类别、状态、掩膜/副本路径与 SHA、像素统计、复核原因",
            "summary.json": "实际计数与像素/类别统计",
            "masks/": "候选索引图（<sample_id>.png，值域 0/1/2/255，含 empty 诊断图）",
            "review/": "问题文件的可编辑 PNG+JSON 副本（<sample_id>.png/.json）",
            "review.csv": "UTF-8 BOM，逐问题记录",
            "README.md": "零基础说明",
        },
        "training_policy": {
            "semantic_reviewed": False,
            "not_training_ready": True,
            "no_train_test_split": True,
            "note": "候选不等于语义已审核；不划分 train/test，不自动训练。",
        },
    }
    _write_json(os.path.join(out_dir, "policy.json"), policy)

    _write_json(os.path.join(out_dir, "manifest.json"), {
        "task_id": TASK_ID, "input_index": _rel(root, index_path),
        "created": created, "path_bases": path_bases, "records": manifest,
    })

    _write_review_csv(os.path.join(out_dir, "review.csv"), review_rows)

    summary = {
        "task_id": TASK_ID, "created": created,
        "input_index": _rel(root, index_path), "output_dir": _rel(root, out_dir),
        "path_bases": path_bases,
        "counts": counts,
        "label_shape_counts": dict(sorted(label_shape_counts.items())),
        "alias_normalized_count": len(alias_log),
        "alias_normalized": alias_log,
        "unknown_label_count": len(unknown_label_log),
        "unknown_labels": unknown_label_log,
        "pixel_totals": pixel_totals,
        "review_record_count": len(review_rows),
        "semantic_reviewed": False,
        "not_training_ready": True,
        "no_train_test_split": True,
        "head_note": "候选/副本均为派生；不改原图/JSON/旧报告/生产库/模型/依赖/.env/任务/记忆。",
        "expected_reference": "最终审查预期最多 855 候选 / 48 review，实际以本结果为准（不硬编码）。",
    }
    _write_json(os.path.join(out_dir, "summary.json"), summary)

    _write_readme(os.path.join(out_dir, "README.md"), summary)

    return summary


def _write_review_csv(path, rows):
    header = ["source_id", "dataset", "source_file", "shape_index", "problem_type",
              "detail", "review_png", "review_json", "review_status"]
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _write_readme(path, summary):
    c = summary["counts"]
    lines = [
        "# RGB 候选掩膜与待复核副本（零基础说明）",
        "",
        "本目录由 `scripts/labels/prepare_candidate.py` 生成，是**派生结果**；原始图片与标注未被修改、"
        "移动或删除。所有候选均为**待语义复核、未训练就绪**。",
        "",
        "## 1. 本次实际计数",
        "",
        "| 项目 | 数量 |",
        "|---|---:|",
        "| 输入记录 | %d |" % c["input"],
        "| 候选（candidate） | %d |" % c["candidate"],
        "| 待复核（review） | %d |" % c["review"],
        "| 零有效像素（empty） | %d |" % c["empty"],
        "| 排除（unsupported） | %d |" % c["unsupported"],
        "",
        "## 2. 像素值含义",
        "",
        "- `0` = Field 田块；`1` = Field ridge 田埂；`2` = Road 道路；`255` = 忽略。",
        "- `255` 同时代表：未画区域、不同类别重叠>1 的像素、以及**未确认即视为不可用**的区域。",
        "- 满幅不同类别叠加（整图 255）记为 **empty 诊断图**，不是候选。",
        "- **0 不是背景**，本项目不设背景类别。",
        "",
        "## 3. 路径基准",
        "",
        "- `manifest.json` 中 `json_path`/`image_path` 是**相对项目根**的源路径。",
        "- `mask_path` 与 `review` 副本路径是**相对本输出目录**的派生产物路径"
        "（如 `masks/<id>.png`、`review/<id>.png`）。",
        "- 把本目录整体拷贝走后，源路径仍需回到项目根解释，派生产物路径相对本目录解释。",
        "",
        "## 4. 怎样打开与修改副本",
        "",
        "1. 在 `review/` 目录用图片编辑器打开 `<sample_id>.png`（与原图逐像素一致）。",
        "2. 用标注工具打开 `<sample_id>.json`：`points` 未被改动，`imageData` 已置空，"
        "`imagePath` 指向同目录 PNG。",
        "3. 修改后请**另存**，不要覆盖本目录副本；**修改副本后仍需重新质检**，"
        "导入或另存都不等于自动验收通过。",
        "",
        "## 5. 来源追溯与内链",
        "",
        "- 策略与 schema：见 [policy.json](policy.json)",
        "- 逐记录清单：见 [manifest.json](manifest.json)",
        "- 机器汇总：见 [summary.json](summary.json)",
        "- 待复核清单：见 [review.csv](review.csv)",
        "- 候选索引图：见 [masks/](masks/)；待复核副本：见 [review/](review/)",
        "",
        "## 6. 未确认与训练划分",
        "",
        "- 未确认区域/重叠区域/背景规则仍待人工复核；本目录**不代表真值**。",
        "- 尚未划分 train/test，不得据此自动训练或报告精度/面积。",
        "",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main(argv=None):
    parser = argparse.ArgumentParser(description="RGB 候选掩膜与待复核副本准备工具")
    parser.add_argument("--input", default=None, help="索引 JSON（须在 reports/labels 内）")
    parser.add_argument("--output", default=None, help="全新输出目录（须在 reports/labels 内）")
    args = parser.parse_args(argv)
    try:
        summary = prepare(args.input, args.output)
    except PrepareError as e:
        sys.stderr.write("[PrepareError] %s\n" % e)
        return 2
    c = summary["counts"]
    print(summary["output_dir"])
    print("input=%d candidate=%d review=%d empty=%d unsupported=%d"
          % (c["input"], c["candidate"], c["review"], c["empty"], c["unsupported"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
