"""生成明确标注的人工合成样本，用于规则与流程演示/测试。

重要：这些样本全部为人工合成，不是真实农业样本，也不能作为业务效果证据。
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SYNTHETIC_NOTE = "SYNTHETIC — 人工合成样本，仅用于演示/测试，非真实农业数据"


def make_png(color, size=(96, 64), noise=False) -> bytes:
    from PIL import Image

    img = Image.new("RGB", size, color)
    if noise:
        px = img.load()
        for y in range(size[1]):
            for x in range(size[0]):
                px[x, y] = ((x * 7) % 256, (y * 11) % 256, ((x + y) * 13) % 256)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def main() -> int:
    out = PROJECT_ROOT / "data" / "p0" / "samples_synth"
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"synthetic": True, "note": SYNTHETIC_NOTE, "files": []}

    (out / "s1_ok.png").write_bytes(make_png((30, 120, 90), noise=True))
    (out / "s1_blurry.png").write_bytes(make_png((30, 120, 90)))  # 近似纯色
    (out / "s1_empty.png").write_bytes(b"")
    (out / "s1_broken.png").write_bytes(b"not-a-real-png")
    (out / "s1_attrs.csv").write_text(
        "scene_id,field_id,crop,面积,单位\n"
        "S1,F1,水稻,1,公顷\n"
        "S1,F2,水稻,3,亩\n"
        "S1,F3,水稻,5,acre\n",
        encoding="utf-8",
    )
    (out / "s1_note.json").write_text(
        json.dumps({"scene_id": "S1", "note": "合成复核记录", "synthetic": True},
                   ensure_ascii=False),
        encoding="utf-8",
    )
    (out / "s1_note_mismatch.json").write_text(
        json.dumps({"scene_id": "S9", "note": "合成：场景失配", "synthetic": True},
                   ensure_ascii=False),
        encoding="utf-8",
    )

    for p in sorted(out.glob("*")):
        manifest["files"].append({"name": p.name, "bytes": p.stat().st_size})
    (out / "MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已生成人工合成样本：{out}")
    print(f"共 {len(manifest['files'])} 个文件。{SYNTHETIC_NOTE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
