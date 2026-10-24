"""R2b: derive pixel-only attributes and ingest once through the public API.

Run stages separately: prepare, import, export. Never replay the import stage.
No business-write SQL, no agricultural guesses, no changes to application rules.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import html
import json
import math
from pathlib import Path
import sys

import PIL
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from agrolake import config, db, ingest, quality, scenes, semantics

DATABASE = ROOT / "data/p0/agrolake.sqlite"
BACKUP = DATABASE.with_name("agrolake.sqlite.bak-20261024-pre-r2b")
ANNOTATION = ROOT / "邓/hy0717_tile_r37_c15.json"
CASE = ROOT / "work/remediation-r2b"
JOURNAL = CASE / "journal.json"
CSV_PATH = CASE / "hy0717_tile_r37_c15-derived-pixel-attributes.csv"
EVIDENCE = ROOT / "outputs/真实场景成组证据-R2-20261024-证据"
SCENE = "RGB-a82c31ebe8064742b5a9445b909df7d0"
SOURCE = "由原始标注 hy0717_tile_r37_c15.json 计算派生"
LABELS = (("Field", "田块"), ("Field ridge", "田埂"), ("Road", "道路"))
HEADER = ["scene_id", "类别", "多边形数", "像素面积", "像素面积占比"]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def counts(conn) -> dict:
    return {table: conn.execute("SELECT COUNT(*) FROM " + table).fetchone()[0]
            for table in ("assets", "asset_versions", "import_events", "quality_checks",
                          "dispositions", "scene_members", "attribute_records")}


def derive(payload: dict) -> tuple[list[dict], dict]:
    """Union per-class Pillow masks; count coverage on the original pixel canvas.

    Cross-class overlap is retained in each class and reported separately. This
    measures annotated polygon coverage, not exclusive training-label counts.
    """
    width, height = payload["imageWidth"], payload["imageHeight"]
    if (width, height) != (512, 512):
        raise ValueError("Expected the specified 512 x 512 annotation canvas")
    by_label = {label: [] for label, _ in LABELS}
    for shape in payload["shapes"]:
        if shape["shape_type"] != "polygon" or shape["label"] not in by_label:
            raise ValueError("Unsupported source shape or label; stop without guessing")
        points = shape["points"]
        if len(points) < 3 or any(len(p) != 2 or any(not math.isfinite(v) for v in p)
                                  for p in points):
            raise ValueError("Invalid polygon coordinates")
        by_label[shape["label"]].append([tuple(p) for p in points])
    masks, rows = [], []
    for label, chinese in LABELS:
        mask = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(mask)
        for points in by_label[label]:
            draw.polygon(points, fill=255)
        # Histogram count and byte count must agree, including zero coverage.
        area = mask.histogram()[255]
        raw = mask.tobytes()
        assert area == raw.count(255) and set(raw) <= {0, 255}
        masks.append(raw)
        rows.append({"scene_id": SCENE, "类别": chinese,
                     "多边形数": len(by_label[label]), "像素面积": area,
                     "像素面积占比": format(area / (width * height), ".12f")})
    depth = Counter(sum(value != 0 for value in pixel) for pixel in zip(*masks))
    assert sum(depth.values()) == width * height
    assert sum(row["像素面积"] for row in rows) == sum(k * v for k, v in depth.items())
    return rows, {"library": "Pillow ImageDraw.polygon", "pillow_version": PIL.__version__,
                  "canvas_width": width, "canvas_height": height,
                  "denominator_pixels": width * height,
                  "coordinate_handling": "原始浮点坐标原样传入；不修改源坐标；隐式闭合，画布外裁剪；边界按Pillow填充规则",
                  "class_handling": "同类取并集；跨类重叠分别计入各类，不决定类别优先级",
                  "coverage_depth_histogram": dict(sorted(depth.items())),
                  "unannotated_pixels": depth[0],
                  "cross_class_overlap_pixels": sum(v for k, v in depth.items() if k > 1),
                  "label_mapping": dict(LABELS)}


def read_state() -> dict:
    return json.loads(JOURNAL.read_text(encoding="utf-8"))


def prepare() -> None:
    if CASE.exists() or BACKUP.exists():
        raise RuntimeError("R2b artifacts/backup already exist; inspect original run, do not overwrite")
    if DATABASE.with_name(DATABASE.name + "-wal").exists():
        raise RuntimeError("WAL present; plain-file backup unsafe, stop")
    before_sha = sha(DATABASE)
    with db.connect_readonly(DATABASE) as conn:
        coverage = scenes.association_coverage(conn)
        assert (coverage["complete_scenes"], coverage["total_scenes"], coverage["coverage"]) == (1, 5, 0.2)
        members = scenes.scene_members(conn, SCENE)
        assert not members["roles"]["attribute"] and members["roles"]["image"]
        before_counts = counts(conn)
    with DATABASE.open("rb") as src, BACKUP.open("xb") as dst:
        while block := src.read(1024 * 1024):
            dst.write(block)
    assert sha(BACKUP) == before_sha == sha(DATABASE)
    payload = json.loads(ANNOTATION.read_text(encoding="utf-8"))
    assert sha(ANNOTATION) == "8890793a40d8a1562e7a3628cef1db98be43ada3188ab4d4428c581ee291e908"
    rows, method = derive(payload)
    CASE.mkdir(parents=True)
    with CSV_PATH.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=HEADER)
        writer.writeheader()
        writer.writerows(rows)
    with CSV_PATH.open(encoding="utf-8", newline="") as handle:
        serialized = list(csv.DictReader(handle))
    assert serialized == [{k: str(v) for k, v in row.items()} for row in rows]
    mapping, unknown = semantics.resolve_columns(HEADER)
    protected = [ANNOTATION, ANNOTATION.with_suffix(".tif"), ROOT / "app.py"]
    protected += list((ROOT / "agrolake").glob("*.py"))
    protected += list((ROOT / "configs").glob("*.json"))
    protected += list((ROOT / "outputs").rglob("*.pdf"))
    protected += [ROOT / "outputs/RGB概率融合候选模型-20261020.json",
                  ROOT / "reports/data/Codex-DATA-20261021-001/frozen-evaluation.json"]
    protected += list((ROOT / "reports/rgb/NEW-EVAL-20261021-001").rglob("*.json"))
    state = {"stage": "prepared", "scene_id": SCENE, "source": SOURCE,
             "backup": str(BACKUP.relative_to(ROOT)), "backup_sha256": before_sha,
             "annotation_sha256": sha(ANNOTATION), "csv_sha256": sha(CSV_PATH),
             "csv_path": str(CSV_PATH.relative_to(ROOT)), "rows": rows, "rasterization": method,
             "before_coverage": coverage, "before_members": members, "before_counts": before_counts,
             "preflight": {"recognized_columns": mapping, "unrecognized_columns": unknown,
                           "missing_business_columns": [c for c in ingest.REQUIRED_CSV_COLUMNS if c not in mapping],
                           "policy": "按原样正规导入，保留既有规则的阻断结论；不补作物、地块或实物面积，覆盖率不等同质检通过"},
             "protected_sha256": {str(p.relative_to(ROOT)): sha(p) for p in protected}}
    write_json(JOURNAL, state)
    print(json.dumps({k: state[k] for k in ("stage", "backup_sha256", "before_coverage", "rows", "rasterization", "preflight")}, ensure_ascii=False, indent=2))


def do_import() -> None:
    state = read_state()
    assert state["stage"] == "prepared", "Import may be executed only once; inspect journal"
    assert sha(DATABASE) == state["backup_sha256"] == sha(BACKUP)
    assert sha(CSV_PATH) == state["csv_sha256"]
    # Mark BEFORE the non-idempotent API call, so an interruption cannot replay it.
    state["stage"] = "import_started_do_not_replay"
    write_json(JOURNAL, state)
    with db.connect(DATABASE) as conn:
        result = ingest.import_bytes(conn, CSV_PATH.read_bytes(), CSV_PATH.name,
                                     store_root=config.storage_root(), source=SOURCE,
                                     scene_id=SCENE, intended_use=None, source_link=None,
                                     license_status=None, acquired_at=None)
    state["import"] = result
    state["stage"] = "imported"
    write_json(JOURNAL, state)
    print(json.dumps(result, ensure_ascii=False))


def export() -> None:
    state = read_state()
    assert state["stage"] == "imported", "No successful import to export"
    assert sha(BACKUP) == state["backup_sha256"]
    for name, expected in state["protected_sha256"].items():
        assert sha(ROOT / name) == expected, "Protected file changed: " + name
    vid = state["import"]["version_id"]
    with db.connect_readonly(DATABASE) as conn, db.connect_readonly(BACKUP) as previous:
        coverage = scenes.association_coverage(conn)
        members = scenes.scene_members(conn, SCENE)
        assert any(row["version_id"] == vid for row in members["roles"]["attribute"])
        version = dict(conn.execute("SELECT * FROM asset_versions WHERE version_id=?", (vid,)).fetchone())
        assert version["source"] == SOURCE and version["declared_scene_id"] == SCENE
        assert version["object_id"] == sha(CSV_PATH) and version["content_type"] == "attribute"
        assert all(version[k] is None for k in ("source_link", "license_status", "acquired_at", "intended_use"))
        attributes = [dict(r) for r in conn.execute("SELECT * FROM attribute_records WHERE version_id=? ORDER BY row_index", (vid,))]
        assert len(attributes) == 3
        assert all(r["field_id"] is None and r["crop"] is None and r["area_raw"] is None
                   and r["area_unit_raw"] is None and r["area_m2"] is None for r in attributes)
        raw_members = [dict(r) for r in conn.execute("SELECT * FROM scene_members WHERE version_id=?", (vid,))]
        assert len(raw_members) == 1 and raw_members[0]["role"] == "attribute"
        checks = [dict(r) for r in conn.execute("SELECT * FROM quality_checks WHERE version_id=? ORDER BY check_id", (vid,))]
        events = [dict(r) for r in conn.execute("SELECT * FROM import_events WHERE version_id=?", (vid,))]
        dispositions = [dict(r) for r in conn.execute("SELECT disp_id,version_id,action,operator,created_at FROM dispositions ORDER BY disp_id")]
        after_counts = counts(conn)
        # Every original business-table row must remain byte-for-byte equivalent.
        preserved = {}
        tables = [r[0] for r in previous.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]
        for table in tables:
            quoted = '"' + table.replace('"', '""') + '"'
            old = Counter(tuple(r) for r in previous.execute("SELECT * FROM " + quoted))
            current = Counter(tuple(r) for r in conn.execute("SELECT * FROM " + quoted))
            assert not (old - current), "Existing rows changed: " + table
            preserved[table] = sum(old.values())
    evidence = {"description": "正式库只读审计导出；非应用原页面", "scene_id": SCENE,
                "source": SOURCE, "backup_sha256": state["backup_sha256"],
                "annotation_sha256": state["annotation_sha256"], "csv_sha256": sha(CSV_PATH),
                "rows": state["rows"], "rasterization": state["rasterization"],
                "preflight": state["preflight"], "before_coverage": state["before_coverage"],
                "after_coverage": coverage, "before_counts": state["before_counts"],
                "after_counts": after_counts, "members": members, "member_records": raw_members,
                "version": version, "attributes": attributes, "import_events": events,
                "quality_checks": checks, "overall_quality": quality.overall_status(checks),
                "dispositions": dispositions, "disposition_actions": dict(Counter(r["action"] for r in dispositions)),
                "preserved_business_rows": preserved, "protected_file_count": len(state["protected_sha256"]),
                "boundary": "覆盖率只统计影像/属性/文本资料类别；像素统计CSV仍有既有农业字段缺失阻断，未声称质检通过或农业精度提高"}
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    write_json(EVIDENCE / "R2b-正式库只读证据.json", evidence)
    # The submitted evidence also contains a byte-identical copy of the CSV.
    (EVIDENCE / CSV_PATH.name).write_bytes(CSV_PATH.read_bytes())
    disposition_path = EVIDENCE / "R2b-全部处置清单.csv"
    with disposition_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["disp_id", "version_id", "action", "operator", "created_at"])
        writer.writeheader()
        writer.writerows(dispositions)
    rendered = html.escape(json.dumps(evidence, ensure_ascii=False, indent=2))
    (EVIDENCE / "R2b-正式库只读证据.html").write_text(
        '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>R2b正式库只读证据</title>'
        '<style>body{max-width:1100px;margin:32px auto;font:16px sans-serif}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style>'
        '<h1>R2b 正式库只读审计导出</h1><p>非应用原页面。覆盖率与质检通过率是不同指标；完整资料仍可能质检阻断。</p><pre>'
        + rendered + '</pre></html>', encoding="utf-8")
    print(json.dumps({k: evidence[k] for k in ("rows", "after_coverage", "member_records", "overall_quality", "before_counts", "after_counts", "dispositions", "protected_file_count")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "import", "export"))
    args = parser.parse_args()
    {"prepare": prepare, "import": do_import, "export": export}[args.stage]()
