#!/usr/bin/env python3
"""准备 RGB 人工标注包（任务 DATA-20261015-003）。

用途：从 DATA-20261015-002 可信索引（manifest.json）出发，按切片行列空间覆盖，
为每个批次挑选固定数量的切片，导出逐像素一致的 512x512 RGB PNG，并生成联系图、
来源/待标注清单与零基础中文说明，供人工查看与标注。

重要边界（本次只做准备，不产出标注）：
- 380 张真实 RGB 无配套标签，且与现有 8 波段 FTW 模型不兼容；本脚本不训练、不推理、
  不猜测任何类别或分割结果。
- 所有样品的 annotation_status 固定为 "unlabeled"、label_path 固定为 null；
  空标注不等于负样本，也不代表已标注。
- 选图只按行列做空间覆盖建议，不宣称视觉/统计代表性；两批次是否同一源地块未知，
  不配成时相，不做 train/test 划分。
- 原图无地理参考，不报面积/位置。

只依赖标准库与 Pillow。所有路径相对脚本 __file__ 定位项目根，不依赖 Host 默认 cwd。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import secrets
import sys
from datetime import datetime
from fractions import Fraction
from pathlib import Path, PurePosixPath

from PIL import Image, ImageDraw, ImageFont

SCHEMA_VERSION = "annotation-pack/1"
KNOWN_GROUPS = ("qie0716rgb", "qie0718rgb")
TILE_RE = re.compile(
    r"^(?P<group>qie0716rgb|qie0718rgb)/tiles/tile_r(?P<row>\d+)_c(?P<col>\d+)\.tif$"
)
SHA_RE = re.compile(r"^[0-9a-f]{64}$")
ABS_DRIVE_RE = re.compile(r"^[A-Za-z]:")

DEFAULT_TILE_SIZE = 512
DEFAULT_PER_GROUP = 6

SELECTION_POLICY = {
    "per_group": DEFAULT_PER_GROUP,
    "method": "空间覆盖：先取最接近行列包围盒中心的切片，再逐次取到已选集最小距离最大的未选图",
    "normalization": "对批次内 row/col 做 min-max 归一化（Fraction 精确分数）；跨度<=0 时按 1 处理",
    "distance": "归一化(row,col)上的平方欧氏距离，用 Fraction 精确表达，不依赖浮点比较",
    "tie_break": "numeric(row, col, path) 升序",
    "caveat": "仅为按行列的空间覆盖建议，不保证视觉或统计代表性，不代表真实地块分布",
}


class PackError(Exception):
    """输入校验或输出约束失败。"""


# --------------------------------------------------------------------------
# 基础工具
# --------------------------------------------------------------------------
def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _default_font(size: int = 14):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # 兼容不支持 size 参数的旧 Pillow
        return ImageFont.load_default()


# --------------------------------------------------------------------------
# 路径与输入校验
# --------------------------------------------------------------------------
def _is_within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def resolve_source_path(source_root: Path, rel: str) -> Path:
    """把索引中的相对路径安全解析到原图根内，拒绝绝对/../越界与 symlink 逃逸。"""
    if not isinstance(rel, str) or not rel or rel != rel.strip():
        raise PackError(f"非法相对路径：{rel!r}")
    if "\\" in rel:
        raise PackError(f"路径必须使用正斜杠：{rel!r}")
    if rel.startswith("/") or ABS_DRIVE_RE.match(rel):
        raise PackError(f"拒绝绝对路径：{rel!r}")
    parts = PurePosixPath(rel).parts
    if any(part in ("..", "", ".") for part in parts):
        raise PackError(f"拒绝包含 .. / 空段的路径：{rel!r}")
    if not TILE_RE.match(rel):
        raise PackError(f"路径不符合已知批次/tiles/tile_rN_cN.tif 形式：{rel!r}")

    root_resolved = source_root.resolve()
    candidate = (root_resolved / Path(*parts)).resolve()
    if not _is_within(candidate, root_resolved):
        raise PackError(f"解析后越出原图根（含 symlink 逃逸）：{rel!r} -> {candidate}")
    return candidate


def validate_entries(raw_entries, source_root: Path):
    """校验索引结构、路径规范、SHA/字节类型、重复项，并对照磁盘实际文件。"""
    if not isinstance(raw_entries, list):
        raise PackError("manifest 必须是 JSON 列表")

    seen_paths = set()
    seen_coords = set()
    validated = []
    for idx, entry in enumerate(raw_entries):
        if not isinstance(entry, dict):
            raise PackError(f"第 {idx} 条不是对象")

        rel = entry.get("path")
        if not isinstance(rel, str):
            raise PackError(f"第 {idx} 条缺少字符串 path")

        resolved = resolve_source_path(source_root, rel)
        m = TILE_RE.match(rel)
        group = m.group("group")
        row = int(m.group("row"))
        col = int(m.group("col"))

        if rel in seen_paths:
            raise PackError(f"重复路径：{rel}")
        if (group, row, col) in seen_coords:
            raise PackError(f"重复行列坐标：{group} r{row}c{col}")
        seen_paths.add(rel)
        seen_coords.add((group, row, col))

        sha = entry.get("sha256")
        if not isinstance(sha, str) or not SHA_RE.match(sha):
            raise PackError(f"{rel} 的 sha256 不是 64 位小写十六进制字符串")
        nbytes = entry.get("bytes")
        if isinstance(nbytes, bool) or not isinstance(nbytes, int) or nbytes <= 0:
            raise PackError(f"{rel} 的 bytes 不是正整数")

        if not resolved.is_file():
            raise PackError(f"索引文件不存在：{rel} -> {resolved}")
        actual_bytes = resolved.stat().st_size
        if actual_bytes != nbytes:
            raise PackError(f"{rel} 字节数不符：索引 {nbytes}，实际 {actual_bytes}")
        actual_sha = sha256_file(resolved)
        if actual_sha != sha:
            raise PackError(f"{rel} SHA256 不符（文件可能被篡改）：索引 {sha}，实际 {actual_sha}")

        validated.append(
            {
                "group": group,
                "row": row,
                "col": col,
                "path": rel,
                "resolved": resolved,
                "sha256": sha,
                "bytes": nbytes,
                "shape": entry.get("shape"),
            }
        )
    if not validated:
        raise PackError("索引为空")
    return validated


def load_manifest(manifest_path: Path, source_root: Path):
    if not manifest_path.is_file():
        raise PackError(f"索引文件不存在：{manifest_path}")
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PackError(f"索引 JSON 解析失败：{exc}") from exc
    return validate_entries(raw, source_root)


# --------------------------------------------------------------------------
# 输入范围校验（CLI 与公开 build 入口统一执行，直接调用不能绕过）
# --------------------------------------------------------------------------
def archive_source_root(project_root: Path) -> Path:
    return (Path(project_root) / "data" / "raw" / "user_uploads" / "20261015").resolve()


def reports_data_root(project_root: Path) -> Path:
    return (Path(project_root) / "reports" / "data").resolve()


def anchor_path(path_str, project_root: Path) -> Path:
    """相对参数明确锚定项目根，避免依赖 Host 默认 cwd；绝对路径原样使用。"""
    p = Path(path_str)
    return p if p.is_absolute() else (Path(project_root) / p)


def validate_source_root(source_root: Path, project_root: Path) -> Path:
    """source_root 必须是既有归档根，或项目 reports/data 内的 fixture 根。

    不允许借 source-root 读取其它项目、凭据或任意磁盘位置。
    """
    resolved = Path(source_root).resolve()
    if not resolved.is_dir():
        raise PackError(f"source_root 不存在或不是目录：{resolved}")
    archive = archive_source_root(project_root)
    reports_data = reports_data_root(project_root)
    if resolved == archive or _is_within(resolved, reports_data):
        return resolved
    raise PackError(
        "source_root 必须是既有归档根 data/raw/user_uploads/20261015，"
        f"或项目 reports/data 内的 fixture 根；拒绝其它来源：{resolved}"
    )


def validate_input_manifest(manifest_path: Path, project_root: Path) -> Path:
    """索引解析后必须位于项目 reports/data 内且存在。"""
    resolved = Path(manifest_path).resolve()
    if not resolved.is_file():
        raise PackError(f"索引文件不存在：{resolved}")
    reports_data = reports_data_root(project_root)
    if not _is_within(resolved, reports_data):
        raise PackError(f"索引必须位于项目 reports/data 内：{resolved}")
    return resolved


# --------------------------------------------------------------------------
# 选图（按行列空间覆盖，确定性）
# --------------------------------------------------------------------------
def _norm(value: int, lo: int, hi: int) -> Fraction:
    """行/列 min-max 归一化，用 Fraction 精确表达；跨度<=0 按 1 处理。"""
    span = hi - lo
    if span <= 0:
        span = 1
    return Fraction(value - lo, span)


def _sqdist(a, b) -> Fraction:
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def select_for_group(entries, count: int):
    """对单批次按行列空间覆盖选 count 张；不足则全取。

    返回 (选中列表, order)。**所有分支**（含空组、不足 count）的 order 都是
    {path: 1-based 排名} 的 dict，避免调用方对类型做兼容猜测。
    """
    items = sorted(entries, key=lambda e: (e["row"], e["col"], e["path"]))
    if count <= 0:
        raise PackError("每批数量必须为正整数")
    if len(items) <= count:
        order = {e["path"]: i + 1 for i, e in enumerate(items)}
        return items, order

    rows = [e["row"] for e in items]
    cols = [e["col"] for e in items]
    rlo, rhi = min(rows), max(rows)
    clo, chi = min(cols), max(cols)
    coords = {e["path"]: (_norm(e["row"], rlo, rhi), _norm(e["col"], clo, chi)) for e in items}

    # 包围盒真正中点：归一化后恒为 (1/2, 1/2)，用 Fraction 精确表示，不依赖浮点。
    center = (Fraction(1, 2), Fraction(1, 2))
    first = min(
        items,
        key=lambda e: (_sqdist(coords[e["path"]], center), e["row"], e["col"], e["path"]),
    )
    selected = [first]
    selected_paths = {first["path"]}

    while len(selected) < count:
        candidates = [e for e in items if e["path"] not in selected_paths]

        def min_dist(e):
            p = coords[e["path"]]
            return min(_sqdist(p, coords[s["path"]]) for s in selected)

        best = max(min_dist(e) for e in candidates)
        winners = [e for e in candidates if min_dist(e) == best]
        nxt = min(winners, key=lambda e: (e["row"], e["col"], e["path"]))
        selected.append(nxt)
        selected_paths.add(nxt["path"])

    order = {e["path"]: i + 1 for i, e in enumerate(selected)}
    ordered = sorted(selected, key=lambda e: (e["row"], e["col"], e["path"]))
    return ordered, order



def select_items(validated, per_group: int):
    """按 KNOWN_GROUPS 顺序选图，并给出稳定 uid / selection_rank。"""
    by_group = {g: [] for g in KNOWN_GROUPS}
    for e in validated:
        by_group[e["group"]].append(e)

    result = []
    ranks = {}
    for group in KNOWN_GROUPS:
        chosen, order = select_for_group(by_group[group], per_group)
        for e in chosen:
            e = dict(e)
            e["_group_order"] = KNOWN_GROUPS.index(group)
            result.append(e)
            ranks[e["path"]] = order.get(e["path"], 0)

    result.sort(key=lambda e: (e["_group_order"], e["row"], e["col"], e["path"]))
    for i, e in enumerate(result, start=1):
        e["uid"] = f"AP-{i:02d}"
        e["selection_rank"] = ranks[e["path"]]
    return result


# --------------------------------------------------------------------------
# 输出约束
# --------------------------------------------------------------------------
def validate_output_dir(output_dir: Path, project_root: Path) -> Path:
    """输出必须是项目 reports/data 下的全新目录；拒绝覆盖/逃逸/指向其它区域。"""
    reports_data = (project_root / "reports" / "data").resolve()
    if not reports_data.is_dir():
        raise PackError(f"项目 reports/data 不存在：{reports_data}")
    resolved = output_dir.resolve()
    if not _is_within(resolved, reports_data):
        raise PackError(f"输出目录必须位于项目 reports/data 内：{resolved}")
    if resolved == reports_data:
        raise PackError("输出目录不能就是 reports/data 本身")
    if resolved.exists():
        raise PackError(f"输出目录已存在，拒绝覆盖：{resolved}")
    return resolved


def make_default_output(project_root: Path) -> Path:
    suffix = f"{_now_stamp()}-{secrets.token_hex(3)}"
    return project_root / "reports" / "data" / f"annotation-pack-{suffix}"


# --------------------------------------------------------------------------
# 产物生成
# --------------------------------------------------------------------------
def build_contact_sheet(items, out_path: Path, thumb: int = 256, cols: int = 4):
    n = len(items)
    if n == 0:
        raise PackError("没有可用于联系图的样品")
    rows = (n + cols - 1) // cols
    pad = 8
    label_h = 18
    cell_w = thumb + pad * 2
    cell_h = thumb + label_h + pad * 2
    sheet = Image.new("RGB", (cols * cell_w, rows * cell_h), (245, 245, 245))
    draw = ImageDraw.Draw(sheet)
    font = _default_font(14)
    for i, it in enumerate(items):
        r, c = divmod(i, cols)
        x = c * cell_w + pad
        y = r * cell_h + pad
        with Image.open(it["_png_abs"]) as src:
            src = src.convert("RGB")
            src.thumbnail((thumb, thumb))
            sheet.paste(src, (x, y))
        label = f"{it['uid']} {it['group']} r{it['row']}c{it['col']}"
        draw.text((x, y + thumb + 2), label, fill=(0, 0, 0), font=font)
    sheet.save(out_path, format="PNG")


def _export_png(src_path: Path, dst_path: Path, tile_size: int) -> str:
    with Image.open(src_path) as img:
        if img.mode != "RGB":
            raise PackError(f"期望 RGB 模式，实际 {img.mode}：{src_path.name}")
        if img.size != (tile_size, tile_size):
            raise PackError(f"期望 {tile_size}x{tile_size}，实际 {img.size}：{src_path.name}")
        img.load()
        raw = img.tobytes()
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(dst_path, format="PNG")
    with Image.open(dst_path) as back:
        back.load()
        if back.mode != "RGB" or back.size != (tile_size, tile_size):
            raise PackError(f"PNG 回读尺寸/模式异常：{dst_path.name}")
        if back.tobytes() != raw:
            raise PackError(f"PNG 与 TIFF 逐像素不一致：{dst_path.name}")
    return sha256_file(dst_path)


def _meta_source_info(meta_path: Path):
    if not meta_path.is_file():
        return {"meta_file": str(meta_path.name), "status": "待核实"}
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"meta_file": str(meta_path.name), "status": "待核实（meta 解析失败）"}
    return {
        "meta_file": meta_path.name,
        "original_source_path": meta.get("source_path"),
        "tile_size": meta.get("tile_size"),
        "rows": meta.get("rows"),
        "cols": meta.get("cols"),
        "status": "待核实：文件名/旧路径不能证明拍摄方式、日期、正射处理或作物身份",
    }


def _write_pack_readme(path: Path, source_manifest: Path, item_count: int):
    text = f"""# RGB 待标注包使用说明

本目录由 `scripts/data/prepare_annotation_pack.py` 生成，用于**人工查看与标注**第一批真实 RGB 切片。
本轮只准备输入，**不含任何标签**：所有样品 `annotation_status = unlabeled`、`label_path = null`。

来源索引：`{source_manifest.as_posix()}`
样品数量：{item_count}

## 目录内容

- `png/`：与原始切片逐像素一致的 512x512 RGB PNG（不拉伸、不增强、不重新标色）。
- `contact_sheet.png`：全部样品的联系缩略图，带编号（AP-01…）与批次/行列。
- `manifest.json`：来源、SHA256、字节数、批次/行列、PNG 路径/尺寸、待标注状态。
- `annotation_todo.csv`：可直接用 Excel 打开的待填清单（UTF-8 BOM）。
- 本说明。

## 建议步骤（零基础）

1. 先打开 `contact_sheet.png` 总览，记住每张的编号 AP-xx。
2. 用系统图片查看器打开 `png/` 中对应图片，对照原切片确认画面内容。
3. 打开 `annotation_todo.csv`，在“待填目标类别”和“人工记录”列填写你的判断与备注。
4. 目标类别定义（打算分几类、每类判定标准）确定后，再开始真正的标注工作。

## 必须注意

- 这些图片**没有标签**，也**不能直接用于训练**；空着不等于“负样本”。
- 选图只按行列做空间覆盖建议，**不代表**视觉或统计代表性，也不代表真实地块分布。
- 两批次是否同一源地块未知，不能当作同一地块的时相配对；不做 train/test 划分。
- 原图无地理参考，面积只能按像素计，不能换算平方米。
- 现有 8 波段 FTW 权重与本批 3 通道 RGB 不兼容，禁止复制通道凑 8 通道。
"""
    path.write_text(text, encoding="utf-8")


def build_annotation_pack(
    source_root: Path,
    manifest_path: Path,
    output_dir: Path,
    per_group: int = DEFAULT_PER_GROUP,
    tile_size: int = DEFAULT_TILE_SIZE,
    project_root: Path | None = None,
) -> dict:
    """完整生成流程。

    输入范围与输出约束都在本入口统一校验（直接调用也不能绕过），
    不依赖调用方先自行校验，也不依赖 Host 默认 cwd。
    """
    project_root = Path(project_root) if project_root is not None else project_root_from_file()
    source_root = validate_source_root(Path(source_root), project_root)
    manifest_path = validate_input_manifest(Path(manifest_path), project_root)
    output_dir = validate_output_dir(Path(output_dir), project_root)

    manifest_sha = sha256_file(manifest_path)
    validated = load_manifest(manifest_path, source_root)
    selected = select_items(validated, per_group)

    output_dir.mkdir(parents=True, exist_ok=False)
    png_dir = output_dir / "png"
    png_dir.mkdir()

    items = []
    for it in selected:
        png_name = f"{it['group']}__tile_r{it['row']}_c{it['col']}.png"
        png_abs = png_dir / png_name
        png_sha = _export_png(it["resolved"], png_abs, tile_size)
        record = {
            "uid": it["uid"],
            "group": it["group"],
            "row": it["row"],
            "col": it["col"],
            "source_relative_path": it["path"],
            "source_sha256": it["sha256"],
            "source_bytes": it["bytes"],
            "png_relative_path": f"png/{png_name}",
            "png_sha256": png_sha,
            "png_size": [tile_size, tile_size],
            "annotation_status": "unlabeled",
            "label_path": None,
            "selection_rank": it["selection_rank"],
        }
        record["_png_abs"] = png_abs
        items.append(record)

    build_contact_sheet(items, output_dir / "contact_sheet.png")

    group_stats = {}
    for g in KNOWN_GROUPS:
        group_stats[g] = {
            "input_count": sum(1 for e in validated if e["group"] == g),
            "selected_count": sum(1 for e in items if e["group"] == g),
        }

    out_manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_manifest_path": manifest_path.as_posix(),
        "source_manifest_sha256": manifest_sha,
        "source_root": source_root.as_posix(),
        "selection_policy": dict(SELECTION_POLICY, per_group=per_group),
        "total_input": len(validated),
        "groups": group_stats,
        "target_status": "pending",
        "target_classes": None,
        "target_note": "目标类别尚未确定；本轮所有样品保持 pending/unlabeled，label_path=null",
        "source_info": {
            g: _meta_source_info(source_root / g / "meta.json") for g in KNOWN_GROUPS
        },
        "items": [{k: v for k, v in it.items() if not k.startswith("_")} for it in items],
        "notes": [
            "本包只含待标注输入，不含任何分割掩膜或类别标签。",
            "空标注不是负样本，也不代表已标注。",
            "选图为按行列的空间覆盖建议，不构成视觉/统计代表性保证。",
            "两批次来源关系未知，不配时相、不划分 train/test。",
            "原图无地理参考，不报面积或位置。",
        ],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(out_manifest, ensure_ascii=False, indent=1), encoding="utf-8"
    )

    with open(output_dir / "annotation_todo.csv", "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["编号", "批次", "行", "列", "原图相对路径", "PNG相对路径", "标注状态", "待填目标类别", "来源信息", "人工记录"]
        )
        for it in items:
            writer.writerow(
                [
                    it["uid"],
                    it["group"],
                    it["row"],
                    it["col"],
                    it["source_relative_path"],
                    it["png_relative_path"],
                    it["annotation_status"],
                    "",
                    out_manifest["source_info"][it["group"]].get("original_source_path", "待核实"),
                    "",
                ]
            )

    _write_pack_readme(output_dir / "README.md", manifest_path, len(items))

    return {
        "output_dir": output_dir.as_posix(),
        "total_input": len(validated),
        "item_count": len(items),
        "groups": group_stats,
        "items": [{k: v for k, v in it.items() if not k.startswith("_")} for it in items],
    }


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def project_root_from_file() -> Path:
    return Path(__file__).resolve().parents[2]


def build_parser(project_root: Path) -> argparse.ArgumentParser:
    default_source = project_root / "data" / "raw" / "user_uploads" / "20261015"
    default_manifest = project_root / "reports" / "data" / "DATA-20261015-002" / "manifest.json"
    parser = argparse.ArgumentParser(
        description="准备 RGB 人工标注包：按行列空间覆盖选图并导出待标注输入（不含标签）。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source-root", default=str(default_source), help="原图根目录（默认 DATA-001/002 归档根）")
    parser.add_argument("--manifest", default=str(default_manifest), help="可信索引 manifest.json（JSON 列表）")
    parser.add_argument("--output", default=None, help="输出目录；必须位于项目 reports/data 内且不存在。缺省自动生成唯一目录")
    parser.add_argument("--per-group", type=int, default=DEFAULT_PER_GROUP, help="每批挑选数量")
    parser.add_argument("--tile-size", type=int, default=DEFAULT_TILE_SIZE, help="切片边长（用于校验）")
    return parser


def main(argv=None) -> int:
    project_root = project_root_from_file()
    args = build_parser(project_root).parse_args(argv)

    if args.per_group <= 0:
        print("error: --per-group 必须为正整数", file=sys.stderr)
        return 2
    if args.tile_size <= 0:
        print("error: --tile-size 必须为正整数", file=sys.stderr)
        return 2

    source_root = anchor_path(args.source_root, project_root)
    manifest_path = anchor_path(args.manifest, project_root)
    output_dir = (
        anchor_path(args.output, project_root) if args.output else make_default_output(project_root)
    )

    try:
        result = build_annotation_pack(
            source_root, manifest_path, output_dir, args.per_group, args.tile_size, project_root
        )
    except PackError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"OK 输出目录：{result['output_dir']}")
    print(f"输入总数：{result['total_input']}；导出样品：{result['item_count']}")
    for g, s in result["groups"].items():
        print(f"  {g}: 输入 {s['input_count']}，选中 {s['selected_count']}")
    print("所有样品 annotation_status=unlabeled，label_path=null（未标注，不训练）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
