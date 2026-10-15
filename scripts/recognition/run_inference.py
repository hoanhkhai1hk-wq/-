"""CLI: real inference with the direct SMP adapter (DEV-20261015-007).

Windows CLI with main guard. Args: --input, --checkpoint, --output-dir
plus optional --expected-input-sha256.

DEV-20261015-007 fixes (Codex REV-20261015-001 #3/#4/#5/#6):
- The output directory is created exclusively. An existing directory is never
  written to (no error.json overwrite, no success/failure mixing); the CLI
  prints a structured failure and exits non-zero leaving it byte-for-byte.
- ``sha256_of`` honours its byte limit: the input raster is bounded at 40 MiB
  and the checkpoint at 250 MiB, and the boundary is applied before any decode.
- The input SHA256 is always recorded. A fixed smoke input can be pinned with
  --expected-input-sha256; a plain input is validated by real format/size/band
  checks instead of a hardcoded hash. The trusted model hash is retained.
- A spatial-tag or NoData read-back mismatch is now a hard failure.
"""

import argparse
import csv
import hashlib
import json
import os
import sys
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

EXPECTED_CKPT_SHA256 = "d04671c498c181a2bca4b9c1a82fb8dcc93384eaf55841cb6672692f6fb21d3c"
MAX_INPUT_BYTES = 40 * 1024 * 1024     # 40 MiB input raster
MAX_CKPT_BYTES = 250 * 1024 * 1024     # 250 MiB model checkpoint


def sha256_of(path, max_bytes):
    """SHA256 of a file, refusing anything larger than max_bytes before reading."""
    size = os.path.getsize(path)
    if size > max_bytes:
        raise ValueError(f"file too large: {size} bytes > {max_bytes} byte limit ({path})")
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_preview(path, image, mask):
    """3-panel PNG: input first-3-band RGB, colour mask, overlay."""
    import numpy as np
    from PIL import Image, ImageDraw

    rgb = image[:, :, :3]
    preview_valid = (mask != 255) & np.isfinite(rgb).all(axis=2)
    rgb8 = np.full(rgb.shape, 90, dtype=np.uint8)
    if preview_valid.any():
        # Only valid RGB values contribute to the stretch. Float64 also avoids
        # overflow when a valid float32 source has a very wide value range.
        values = rgb[preview_valid].astype(np.float64)
        lo, hi = np.percentile(values, (2, 98))
        rng = max(float(hi - lo), 1e-6)
        rgb8[preview_valid] = np.clip((values - lo) / rng * 255.0, 0, 255).astype(np.uint8)

    color = np.zeros((*mask.shape, 3), dtype=np.uint8)
    color[mask == 1] = (60, 180, 75)      # field interior - green
    color[mask == 2] = (230, 60, 50)      # boundary - red
    color[mask == 0] = (245, 245, 245)    # background - near white
    color[mask == 255] = (90, 90, 90)     # NoData / unlabelled - grey
    color[~preview_valid] = (90, 90, 90)

    overlay = (rgb8 * 0.55 + color * 0.45).astype(np.uint8)

    bar = 18
    h, w = mask.shape
    panel = Image.new("RGB", (w * 3 + 20, h + bar), (30, 30, 30))
    for i, arr in enumerate((rgb8, color, overlay)):
        panel.paste(Image.fromarray(arr), (i * (w + 10), bar))
    d = ImageDraw.Draw(panel)
    label = "input RGB b1-3" if preview_valid.any() else "input RGB: no valid preview pixels"
    d.text((2, 2), f"{label} | PREDICTION classes | overlay (not ground truth) | NoData grey", fill=(255, 255, 255))
    panel.save(path)


def main():
    parser = argparse.ArgumentParser(description="FTW v1 direct SMP adapter inference")
    parser.add_argument("--input", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--expected-input-sha256", default=None,
        help="optional: require the input to match this SHA256 (fixed smoke input). "
             "When omitted, the input is validated by real format/size/band checks.",
    )
    args = parser.parse_args()

    # Exclusive output directory: never touch an existing directory.
    try:
        os.makedirs(args.output_dir, exist_ok=False)
    except FileExistsError:
        print(json.dumps({
            "status": "failed",
            "stage": "output-dir",
            "error": f"output directory already exists: {os.path.abspath(args.output_dir)}",
            "note": "refusing to write; existing results left byte-for-byte unchanged",
        }, ensure_ascii=False))
        return 1
    started = time.time()

    def fail(stage, exc):
        payload = {
            "status": "failed",
            "stage": stage,
            "error": f"{type(exc).__name__}: {exc}",
            "elapsed_seconds": round(time.time() - started, 3),
        }
        with open(os.path.join(args.output_dir, "error.json"), "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(json.dumps(payload, ensure_ascii=False))
        return 1

    try:
        in_sha = sha256_of(args.input, MAX_INPUT_BYTES)
        ck_sha = sha256_of(args.checkpoint, MAX_CKPT_BYTES)
        if args.expected_input_sha256 is not None and in_sha != args.expected_input_sha256:
            raise ValueError(
                f"input SHA256 mismatch: {in_sha} != expected {args.expected_input_sha256}"
            )
        if ck_sha != EXPECTED_CKPT_SHA256:
            raise ValueError(f"checkpoint SHA256 mismatch: {ck_sha}")

        import torch
        import segmentation_models_pytorch as _smp  # noqa: F401  (version record)

        from recognition import (
            load_model, read_image, write_mask, read_mask_back,
            verify_mask_readback, predict, class_counts, field_table,
        )

        load_t0 = time.time()
        model, hp, load_report = load_model(args.checkpoint)
        load_report["load_seconds"] = round(time.time() - load_t0, 3)

        image, valid, meta = read_image(args.input, max_bytes=MAX_INPUT_BYTES)
        t1 = time.time()
        mask, infer_stats = predict(model, image, valid)
        infer_seconds = round(time.time() - t1, 3)

        mask_path = os.path.join(args.output_dir, "mask.tif")
        write_mask(mask_path, mask, meta["geotiff_tags"])
        back_arr, back_tags = read_mask_back(mask_path)
        mask_back = verify_mask_readback(mask, back_arr, back_tags, meta["geotiff_tags"])

        counts = class_counts(mask)
        fields = field_table(mask)
        with open(os.path.join(args.output_dir, "class_pixels.csv"), "w", newline="", encoding="utf-8") as f:
            wcsv = csv.writer(f)
            wcsv.writerow(["class", "meaning", "pixels"])
            wcsv.writerow([0, "background", counts["background_0"]])
            wcsv.writerow([1, "field_interior", counts["field_interior_1"]])
            wcsv.writerow([2, "boundary", counts["boundary_2"]])
            wcsv.writerow([255, "nodata/unlabelled", counts["nodata_255"]])
        with open(os.path.join(args.output_dir, "field_pixels.csv"), "w", newline="", encoding="utf-8") as f:
            wcsv = csv.writer(f)
            wcsv.writerow(["field_id", "pixels", "min_row", "min_col", "max_row", "max_col"])
            for row in fields:
                wcsv.writerow([row["field_id"], row["pixels"], row["min_row"], row["min_col"], row["max_row"], row["max_col"]])
        write_preview(os.path.join(args.output_dir, "preview.png"), image, mask)

        run = {
            "task": "DEV-20261015-007",
            "status": "success",
            "adapter": "direct SMP 0.5.0 (not ftw-tools 1.4.3 CLI)",
            "versions": {
                "python": sys.version.split()[0],
                "torch": torch.__version__,
                "segmentation_models_pytorch": _smp.__version__,
            },
            "input": {
                "path": os.path.abspath(args.input),
                "sha256": in_sha,
                "expected_input_sha256": args.expected_input_sha256,
                "sha256_pinned": args.expected_input_sha256 is not None,
                "shape_hw": meta["shape_hw"],
                "bands": meta["bands"],
                "dtype": meta["dtype"],
                "band_description": meta["description"],
                "gdal_nodata_source": meta["gdal_nodata_source"],
                "gdal_nodata_value": meta["gdal_nodata_value"],
                "geotiff_tags": meta["geotiff_tags"],
            },
            "checkpoint": {
                "path": os.path.abspath(args.checkpoint),
                "sha256": ck_sha,
                "load_report": load_report,
            },
            "inference": {
                **infer_stats,
                "seconds": infer_seconds,
                "input_sha256": in_sha,
                "model_sha256": ck_sha,
                "temporal_identity": "temporal/window identity of this input has NOT been independently verified",
            },
            "mask_back_read": mask_back,
            "class_counts": counts,
            "field_count": len(fields),
            "notes": [
                "pixel units only; no verified square metres or NDVI",
                "preview first RGB group = bands 1-3; temporal identity unconfirmed",
                "no independent ground truth: no IoU/accuracy reported",
                "field_count = candidate 4-connected regions, NOT verified real field count",
            ],
            "elapsed_seconds": round(time.time() - started, 3),
        }
        with open(os.path.join(args.output_dir, "run.json"), "w", encoding="utf-8") as f:
            json.dump(run, f, ensure_ascii=False, indent=2)
        print(json.dumps({"status": "success", "output_dir": os.path.abspath(args.output_dir)}, ensure_ascii=False))
        return 0
    except Exception as exc:  # noqa: BLE001 - structured failure state required
        return fail("pipeline", exc)


if __name__ == "__main__":
    sys.exit(main())
