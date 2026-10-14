"""把人工合成样本导入演示数据库（data/p0）。

演示数据来源字段统一标注为人工合成，避免被当作真实农业数据。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from agrolake import config, db, ingest  # noqa: E402

SYNTH_DIR = ROOT / "data" / "p0" / "samples_synth"
SOURCE = "SYNTHETIC-人工合成演示"


def main() -> int:
    conn = db.connect()
    db.init_schema(conn)
    store = config.storage_root()
    specs = [
        ("s1_ok.png", {"scene_id": "S1"}),
        ("s1_blurry.png", {"scene_id": "S1"}),
        ("s1_empty.png", {"scene_id": "S1"}),
        ("s1_broken.png", {"scene_id": "S1"}),
        ("s1_attrs.csv", {"scene_id": "S1"}),
        ("s1_note.json", {"scene_id": "S1"}),
        ("s1_note_mismatch.json", {"scene_id": "S1"}),
    ]
    n = 0
    for name, kw in specs:
        path = SYNTH_DIR / name
        if not path.exists():
            print(f"跳过（不存在）：{name}")
            continue
        res = ingest.import_path(conn, path, store_root=store, source=SOURCE, **kw)
        print(f"{name} -> version {res['version_id']} ({res['event_type']})")
        n += 1
    print(f"已导入 {n} 个合成样本到演示库。{SOURCE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
