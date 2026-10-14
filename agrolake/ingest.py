"""统一接入：影像 JPG/PNG、属性 CSV、记录 JSON/TXT。

- 对象按内容 SHA256 存储；同名不同内容不覆盖；精确重复仍记录本次导入事件。
- 原始文件名只作为元数据，不参与目标路径拼接。
- 所有写路径限制在存储根目录内。
"""
from __future__ import annotations

import csv
import io
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from . import config, db, quality, semantics, storage

REQUIRED_CSV_COLUMNS = ["scene_id", "field_id", "crop", "area", "area_unit"]


class UnsupportedFormat(Exception):
    """扩展名不在 P0 接入范围。"""


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _get_or_create_asset(conn: sqlite3.Connection, key: str, filename: str) -> int:
    row = conn.execute("SELECT asset_id FROM assets WHERE asset_key = ?", (key,)).fetchone()
    if row:
        return row["asset_id"]
    cur = conn.execute(
        "INSERT INTO assets(asset_key, original_filename, created_at) VALUES (?,?,?)",
        (key, filename, _now()),
    )
    return cur.lastrowid


def _register_object(conn: sqlite3.Connection, sha: str, relpath: str, size: int) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO objects(object_id, relpath, size_bytes, created_at)"
        " VALUES (?,?,?,?)",
        (sha, relpath, size, _now()),
    )


def _parse_image(path: Path) -> dict:
    from PIL import Image

    with Image.open(path) as im:
        mode = im.mode
        bands = len(im.getbands())
        w, h = im.size
    bitdepth = 16 if "16" in mode else 8
    if mode in ("1",):
        bitdepth = 1
    if mode in ("I", "F"):
        bitdepth = 32
    return {"width": w, "height": h, "bands": bands, "bitdepth": bitdepth}


REQUIRED_ROW_VALUES = ["scene_id", "field_id", "crop"]


def _decode_text(data: bytes) -> tuple[str | None, str | None]:
    """严格按 UTF-8（含 BOM）解码。失败时不使用 replace 偷偷损坏字段，返回错误说明。"""
    try:
        return data.decode("utf-8-sig"), None
    except UnicodeDecodeError as exc:
        return None, f"文本编码错误（非 UTF-8）：{exc}"


def _parse_csv(conn: sqlite3.Connection, version_id: int, text: str) -> None:
    reader = csv.reader(io.StringIO(text))
    header = next(reader, None)
    if header is None:
        db.set_meta(conn, f"csv_parse_issue:{version_id}", "CSV 无表头")
        return
    mapping, _unknown = semantics.resolve_columns(header)
    col_index: dict[str, int | None] = {}
    for canonical, raw in mapping.items():
        try:
            col_index[canonical] = header.index(raw)
        except ValueError:  # 理论上不会发生；防御性处理
            col_index[canonical] = None
    missing = [c for c in REQUIRED_CSV_COLUMNS if c not in mapping]
    if missing:
        db.set_meta(
            conn, f"csv_parse_issue:{version_id}",
            f"缺少必填列：{missing}（识别到 {list(mapping)}）",
        )

    def cell(row, canonical):
        idx = col_index.get(canonical)
        if idx is None:
            return None
        return row[idx] if idx < len(row) else None

    empty_required: list[dict] = []
    has_category_col = col_index.get("category") is not None
    for i, row in enumerate(reader):
        if not row or all((c or "").strip() == "" for c in row):
            continue
        scene_id = cell(row, "scene_id")
        field_id = cell(row, "field_id")
        crop = cell(row, "crop")
        area_raw = cell(row, "area")
        unit_raw = cell(row, "area_unit")
        area_m2, status, reason = semantics.convert_area(area_raw, unit_raw)
        issues = None if status == "ok" else reason

        blanks = [name for name, val in
                  (("scene_id", scene_id), ("field_id", field_id), ("crop", crop))
                  if not (val and str(val).strip())]
        if blanks:
            empty_required.append({"row": i, "fields": blanks})

        category_raw = cell(row, "category") if has_category_col else None
        category_std = None
        if has_category_col:
            category_std, cat_issue = semantics.normalize_category(category_raw)
            if cat_issue:
                issues = f"{issues}; {cat_issue}" if issues else cat_issue

        conn.execute(
            "INSERT INTO attribute_records(version_id, scene_id, field_id, crop,"
            " category_raw, category_std, area_raw, area_unit_raw, area_m2, row_index, issues)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (
                version_id,
                scene_id,
                field_id,
                crop,
                category_raw,
                category_std,
                area_raw,
                unit_raw,
                area_m2,
                i,
                issues,
            ),
        )
    if empty_required:
        db.set_meta(
            conn, f"csv_value_issue:{version_id}",
            f"{len(empty_required)} 行必填值缺失：{empty_required[:5]}",
        )


def _parse_text(conn: sqlite3.Connection, version_id: int, text: str | None, fmt: str,
                declared_scene: str | None, source: str | None,
                decode_error: str | None = None) -> None:
    scene_id = declared_scene
    payload_json = None
    issues = None
    if decode_error:
        # 编码错误：登记可解释记录，不写入任何半截字段值
        issues = decode_error
    elif fmt == "json":
        try:
            payload = json.loads(text)
            payload_json = json.dumps(payload, ensure_ascii=False)
            payload_scene = None
            if isinstance(payload, dict):
                payload_scene = payload.get("scene_id") or payload.get("scene")
            if payload_scene:
                scene_id = str(payload_scene)
                if declared_scene and str(declared_scene) != str(payload_scene):
                    issues = f"scene_id 失配：声明={declared_scene} 内容={payload_scene}"
        except json.JSONDecodeError as exc:
            issues = f"JSON 解析失败：{exc}"
            payload_json = None
    else:
        payload_json = json.dumps({"text": text}, ensure_ascii=False)

    conn.execute(
        "INSERT INTO text_records(version_id, scene_id, source, fmt, payload_json, issues)"
        " VALUES (?,?,?,?,?,?)",
        (version_id, scene_id, source, fmt, payload_json, issues),
    )


def _add_scene_member(conn: sqlite3.Connection, scene_id: str | None, version_id: int,
                      role: str, basis: str) -> None:
    if not scene_id:
        return
    conn.execute(
        "INSERT OR IGNORE INTO scenes(scene_id, label, created_at) VALUES (?,?,?)",
        (scene_id, None, _now()),
    )
    conn.execute(
        "INSERT INTO scene_members(scene_id, version_id, role, basis, time_condition,"
        " confirm_status, created_at) VALUES (?,?,?,?,?,?,?)",
        (scene_id, version_id, role, basis, None, "auto", _now()),
    )


def import_bytes(
    conn: sqlite3.Connection,
    data: bytes,
    original_filename: str,
    *,
    store_root: Path | None = None,
    source: str | None = None,
    scene_id: str | None = None,
    intended_use: str | None = None,
    source_link: str | None = None,
    license_status: str | None = None,
    acquired_at: str | None = None,
) -> dict:
    store_root = store_root or config.storage_root()
    filename = storage.safe_name(original_filename)
    ext = Path(filename).suffix.lower()
    ctype = config.classify_extension(ext)
    if ctype is None:
        if ext in (".tif", ".tiff", ".geotiff"):
            raise UnsupportedFormat(
                "TIFF/GeoTIFF 本轮待接入：未验证栅格读写与空间信息，"
                "不使用 Pillow 把多波段栅格当作普通照片处理。"
            )
        raise UnsupportedFormat(f"暂不支持的格式：{ext or '(无扩展名)'}")

    sha, obj_path, _created = storage.store_bytes(store_root, data)
    relpath = storage.relative_to_root(store_root, obj_path)
    _register_object(conn, sha, relpath, len(data))

    asset_key, _identity = semantics.asset_identity(filename, scene_id, source)
    asset_id = _get_or_create_asset(conn, asset_key, filename)

    prev = conn.execute(
        "SELECT v.version_id, v.version_no, v.object_id FROM asset_versions v"
        " WHERE v.asset_id = ? ORDER BY v.version_no DESC LIMIT 1",
        (asset_id,),
    ).fetchone()
    # 跨文件名/跨资产的内容重复识别：同一内容只保留唯一有效版本
    global_dup = conn.execute(
        "SELECT version_id FROM asset_versions WHERE object_id = ? ORDER BY version_id LIMIT 1",
        (sha,),
    ).fetchone()

    if prev is None:
        version_no = 1
        if global_dup is not None:
            event_type = "duplicate_content"
            status = "duplicate"
            duplicate_of = global_dup["version_id"]
        else:
            event_type = "new_asset"
            status = "active"
            duplicate_of = None
    else:
        version_no = prev["version_no"] + 1
        if prev["object_id"] == sha:
            event_type = "duplicate_content"
            status = "duplicate"
            duplicate_of = global_dup["version_id"] if global_dup else prev["version_id"]
        elif global_dup is not None:
            event_type = "duplicate_content"
            status = "duplicate"
            duplicate_of = global_dup["version_id"]
        else:
            event_type = "new_version_same_name"
            status = "active"
            duplicate_of = None

    meta = {"width": None, "height": None, "bands": None, "bitdepth": None}
    now = _now()
    cur = conn.execute(
        "INSERT INTO asset_versions(asset_id, version_no, content_type, object_id,"
        " original_filename, source, source_link, license_status, acquired_at,"
        " declared_scene_id, intended_use, width, height, bands, bitdepth, status,"
        " duplicate_of_version_id, created_at, updated_at)"
        " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            asset_id, version_no, ctype, sha, filename, source, source_link,
            license_status, acquired_at, scene_id, intended_use,
            meta["width"], meta["height"], meta["bands"], meta["bitdepth"],
            status, duplicate_of, now, now,
        ),
    )
    version_id = cur.lastrowid

    # 按类型解析并落明细
    if ctype == "image":
        try:
            info = _parse_image(obj_path)
            meta.update(info)
            conn.execute(
                "UPDATE asset_versions SET width=?, height=?, bands=?, bitdepth=?"
                " WHERE version_id=?",
                (info["width"], info["height"], info["bands"], info["bitdepth"], version_id),
            )
        except Exception:
            # 解码失败交由质检规则给出阻断结论
            pass
    elif ctype == "attribute":
        text, decode_error = _decode_text(data)
        if decode_error:
            db.set_meta(conn, f"csv_parse_issue:{version_id}", decode_error)
        else:
            _parse_csv(conn, version_id, text)
    elif ctype == "text":
        text, decode_error = _decode_text(data)
        _parse_text(conn, version_id, text, ext.lstrip("."), scene_id, source, decode_error)

    _add_scene_member(conn, scene_id, version_id, ctype, "declared")

    conn.execute(
        "INSERT INTO import_events(imported_at, asset_id, version_id, original_filename,"
        " declared_source, declared_scene_id, content_sha256, object_relpath, size_bytes,"
        " event_type, notes) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (
            now, asset_id, version_id, filename, source, scene_id, sha, relpath,
            len(data), event_type, None,
        ),
    )
    conn.commit()

    quality.run_checks(conn, version_id, store_root)
    return {
        "version_id": version_id,
        "asset_id": asset_id,
        "sha256": sha,
        "event_type": event_type,
        "object_relpath": relpath,
    }


def import_path(conn: sqlite3.Connection, path: Path, **kwargs) -> dict:
    path = Path(path)
    data = path.read_bytes()
    return import_bytes(conn, data, path.name, **kwargs)
