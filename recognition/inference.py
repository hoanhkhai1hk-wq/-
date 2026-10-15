"""Tiled inference adapted from the FTW approach, without ftw-tools.

Differences vs ftw-cli 1.4.3 inference (documented in docs/模型适配说明.md):
- tifffile instead of rasterio/GDAL; single process, 2 torch threads.
- reflect edge padding of the whole raster so every valid input pixel falls
  inside a window interior (torchgeo GridGeoSampler leaves edge pixels to the
  zero-initialized background; this adapter does not).
- Same preprocessing (/3000), same bilinear up x2 before the model and
  nearest down after argmax, patch 256 / padding 16 / stride 224.
- No claim of numeric equivalence with the original GridGeoSampler pipeline.

DEV-20261015-007 fixes (Codex REV-20261015-001 #2):
- Invalid / non-finite pixels are neutral-filled with 0 BEFORE padding,
  bilinear interpolation and the model, so NaN/Inf can never reach the
  network and contaminate neighbouring valid predictions.
- The valid mask is checked against the image shape; the tensor entering the
  model and the logits leaving it are asserted finite with the expected class
  count. An input with no valid pixels is rejected instead of writing a fake
  coverage=1 success.
"""

import numpy as np
import torch

PATCH = 256
PADDING = 16
STRIDE = 224
RESIZE_FACTOR = 2
PREPROCESS_DIVISOR = 3000.0
NODATA_CLASS = 255
VALID_CLASSES = (0, 1, 2)


def _window_starts(total, window, stride):
    if total <= window:
        return [0]
    starts = list(range(0, total - window + 1, stride))
    if starts[-1] + window < total:
        starts.append(total - window)
    return starts


def _pad_to_window(arr2d_or_3d, window):
    """Reflect-pad the leading axes up to at least `window`."""
    need = [(window - s) if s < window else 0 for s in arr2d_or_3d.shape[:-1]]
    pad = [(0, 0)] * arr2d_or_3d.ndim
    for i, n in enumerate(need):
        if n > 0:
            pad[i] = (n // 2, n - n // 2)
    if any(p != (0, 0) for p in pad):
        return np.pad(arr2d_or_3d, pad, mode="reflect"), pad
    return arr2d_or_3d, [(0, 0)] * arr2d_or_3d.ndim


def predict(model, image, valid):
    """Run tiled inference.

    image: HxWx8 float32 raw (not normalized), valid: HxW bool.
    Returns (mask uint8 HxW with {0,1,2,255}, stats dict).
    """
    if image.ndim != 3:
        raise ValueError(f"expected HxWxC image, got ndim {image.ndim}")
    if valid.shape != image.shape[:2]:
        raise ValueError(
            f"valid mask shape {valid.shape} does not match image {image.shape[:2]}"
        )
    valid = np.asarray(valid, dtype=bool)
    if not valid.any():
        raise ValueError("input has no valid pixels; refusing to run inference")

    h, w = image.shape[:2]
    pad = PADDING
    n_classes = len(VALID_CLASSES)

    # Neutral-fill invalid pixels BEFORE any interpolation or model call.
    clean = np.where(valid[:, :, None], image, 0.0).astype(np.float32)

    # Reflect-pad so edge pixels land in window interiors (real predictions).
    img = np.pad(clean, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
    img, extra_pad = _pad_to_window(img, PATCH)
    dy, dx = extra_pad[0][0], extra_pad[1][0]
    hp, wp = img.shape[:2]

    torch.set_num_threads(2)
    starts_y = _window_starts(hp, PATCH, STRIDE)
    starts_x = _window_starts(wp, PATCH, STRIDE)

    mask = np.zeros((h, w), dtype=np.uint8)
    covered = np.zeros((h, w), dtype=bool)
    n_windows = 0

    for top in starts_y:
        for left in starts_x:
            patch = img[top:top + PATCH, left:left + PATCH]
            x = torch.from_numpy(patch.transpose(2, 0, 1)).unsqueeze(0).float()
            # Assert finiteness at the model entry (invalid pixels are 0 now).
            if not bool(torch.isfinite(x).all()):
                raise RuntimeError("non-finite values reached the model")
            x = x / PREPROCESS_DIVISOR
            x = torch.nn.functional.interpolate(
                x, size=(PATCH * RESIZE_FACTOR, PATCH * RESIZE_FACTOR), mode="bilinear",
                align_corners=False,
            )
            with torch.inference_mode():
                logits = model(x)
            if logits.ndim != 4 or int(logits.shape[1]) != n_classes:
                raise RuntimeError(
                    f"unexpected model output shape {tuple(logits.shape)}; expected {n_classes} classes"
                )
            if not bool(torch.isfinite(logits).all()):
                raise RuntimeError("model produced non-finite logits")
            pred = logits.argmax(dim=1).unsqueeze(0).float()
            pred = torch.nn.functional.interpolate(
                pred, size=(PATCH, PATCH), mode="nearest"
            )[0, 0].numpy().astype(np.uint8)

            # Window interior in padded coords -> original coords.
            iy0, iy1 = top + pad, top + PATCH - pad
            ix0, ix1 = left + pad, left + PATCH - pad
            oy0, oy1 = iy0 - pad - dy, iy1 - pad - dy
            ox0, ox1 = ix0 - pad - dx, ix1 - pad - dx
            oy0, ox0 = max(oy0, 0), max(ox0, 0)
            oy1, ox1 = min(oy1, h), min(ox1, w)
            if oy0 >= oy1 or ox0 >= ox1:
                continue
            sy0 = oy0 + dy + pad - top
            sx0 = ox0 + dx + pad - left
            mask[oy0:oy1, ox0:ox1] = pred[sy0:sy0 + (oy1 - oy0), sx0:sx0 + (ox1 - ox0)]
            covered[oy0:oy1, ox0:ox1] = True
            n_windows += 1

    # NoData / non-finite pixels are marked, never faked as background.
    mask[~valid] = NODATA_CLASS
    uncovered_valid = int(np.count_nonzero(valid & ~covered))
    if uncovered_valid:
        raise RuntimeError(f"{uncovered_valid} valid pixels received no prediction")
    total_valid = int(np.count_nonzero(valid))
    covered_valid = int(np.count_nonzero(valid & covered))
    coverage = covered_valid / total_valid if total_valid else 0.0

    stats = {
        "patch": PATCH,
        "padding": PADDING,
        "stride": STRIDE,
        "resize_factor": RESIZE_FACTOR,
        "windows": n_windows,
        "valid_pixels": total_valid,
        "covered_valid_pixels": covered_valid,
        "coverage_ratio_valid": coverage,
        "uncovered_valid_pixels": 0,
    }
    return mask, stats
