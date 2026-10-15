"""GeoTIFF reading/writing via tifffile (no rasterio/GDAL install).

Reads one 8-band page, checks layout/dtype/size/finiteness, captures GeoTIFF
spatial tags (33550/33922/34264/34735/34736/34737) and GDAL metadata tags,
then writes the uint8 class mask back with the original spatial tags
preserved verbatim and GDAL NoData updated to 255.

DEV-20261015-007 fixes (Codex REV-20261015-001):
- A numeric GDAL NoData value (tag 42113) is now parsed and excluded from the
  valid mask; a pixel equal to the declared NoData in ANY band is invalid.
  When the source declares no NoData, no value (not even 0) is assumed.
- Hard pre-decode bounds are enforced BEFORE ``page.asarray``: input file size,
  per-side pixels, total pixels and estimated decoded byte size. No large
  array is ever allocated to discover an over-limit input.
- ``verify_mask_readback`` turns a spatial-tag or NoData read-back mismatch
  into a hard failure (previously it was only recorded inside a success).
"""

import os

import numpy as np
import tifffile

GEO_TAG_CODES = (33550, 33922, 34264, 34735, 34736, 34737)
GDAL_METADATA = 42112
GDAL_NODATA = 42113
NODATA_CLASS = 255

# Pre-decode safety bounds (checked before page.asarray).
MAX_SOURCE_BYTES = 40 * 1024 * 1024      # 40 MiB input raster file
MAX_SIDE = 4096                          # max pixels per side
MAX_PIXELS = 1_048_576                   # max total pixels (H*W)
MAX_DECODE_BYTES = 64 * 1024 * 1024      # 64 MiB estimated decoded array

# TIFF field types used by tifffile extratags
_ASCII, _SHORT, _LONG, _DOUBLE = 2, 3, 4, 12

_FLOAT32_ITEMSIZE = np.dtype(np.float32).itemsize


def _parse_nodata(raw):
    """Return the declared NoData value as a float, or None if absent/non-numeric."""
    if raw is None:
        return None
    try:
        return float(str(raw))
    except (TypeError, ValueError):
        return None


def read_image(path, *, max_bytes=MAX_SOURCE_BYTES, max_side=MAX_SIDE,
               max_pixels=MAX_PIXELS, max_decode_bytes=MAX_DECODE_BYTES):
    """Return (image HxWx8 float32, valid bool mask HxW, meta dict).

    Enforces file-size/shape/decode bounds before decoding. The valid mask
    excludes non-finite pixels and any pixel equal to the declared GDAL NoData
    in any band; when no NoData is declared, none is assumed.
    """
    size = os.path.getsize(path)
    if size > max_bytes:
        raise ValueError(f"raster file too large: {size} bytes > {max_bytes} bytes limit")
    with tifffile.TiffFile(str(path)) as tf:
        if len(tf.pages) != 1:
            raise ValueError(f"expected a single-page TIFF, got {len(tf.pages)} pages")
        page = tf.pages[0]
        if page.ndim != 3 or page.shape[2] != 8:
            raise ValueError(
                f"expected an 8-band (H, W, 8) raster, got shape {page.shape}; "
                "RGB cannot be copied to 8 channels"
            )
        if page.dtype not in (np.uint16, np.int16, np.uint8, np.float32, np.float64):
            raise ValueError(f"unexpected raster dtype {page.dtype}")

        h, w = int(page.shape[0]), int(page.shape[1])
        if h > max_side or w > max_side:
            raise ValueError(f"raster side exceeds {max_side} px: {h}x{w}")
        if h * w > max_pixels:
            raise ValueError(f"raster pixel count exceeds {max_pixels}: {h * w}")
        est_decoded = h * w * 8 * _FLOAT32_ITEMSIZE
        if est_decoded > max_decode_bytes:
            raise ValueError(
                f"estimated decoded size exceeds {max_decode_bytes} bytes: {est_decoded}"
            )

        meta = {
            "shape_hw": (h, w),
            "bands": int(page.shape[2]),
            "dtype": str(page.dtype),
            "description": None,
        }
        desc = page.tags.get(270)
        if desc is not None and isinstance(desc.value, str):
            meta["description"] = desc.value
        tags = {}
        for code in GEO_TAG_CODES:
            tg = page.tags.get(code)
            if tg is not None:
                tags[code] = tg.value  # raw value kept for verbatim re-write
        gdal_meta = page.tags.get(GDAL_METADATA)
        if gdal_meta is not None and isinstance(gdal_meta.value, str):
            meta["gdal_metadata"] = gdal_meta.value
        nodata_tag = page.tags.get(GDAL_NODATA)
        nodata_raw = str(nodata_tag.value) if nodata_tag is not None else None
        nodata_value = _parse_nodata(nodata_raw)
        meta["gdal_nodata_source"] = nodata_raw
        meta["gdal_nodata_value"] = nodata_value

        # All bounds are satisfied; only now decode.
        arr = page.asarray().astype(np.float32)

    valid = np.isfinite(arr).all(axis=2)
    if nodata_value is not None:
        valid &= ~(arr == nodata_value).any(axis=2)
    meta["geotiff_tags"] = {code: _to_jsonable(v) for code, v in tags.items()}
    return arr, valid, meta


def _to_jsonable(value):
    """Native-type conversion that preserves numeric values (no str casting)."""
    if isinstance(value, np.ndarray):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, (tuple, list)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def _extratag_for(code, value):
    """Convert a captured tag value into a tifffile extratags tuple."""
    if isinstance(value, str):
        return (code, _ASCII, 0, value, True)
    if isinstance(value, (tuple, list)):
        if all(isinstance(v, str) for v in value):
            return (code, _ASCII, 0, "".join(value), True)
        if all(float(v).is_integer() for v in value) and code in (34735,):
            return (code, _SHORT, len(value), [int(v) for v in value], True)
        return (code, _DOUBLE, len(value), [float(v) for v in value], True)
    raise ValueError(f"unsupported tag value type for code {code}: {type(value)!r}")


def write_mask(path, mask, source_tags, nodata=NODATA_CLASS):
    """Write uint8 class mask, preserving source GeoTIFF tags, NoData=nodata."""
    if mask.dtype != np.uint8:
        raise ValueError("mask must be uint8")
    extratags = []
    for code, value in sorted(source_tags.items()):
        extratags.append(_extratag_for(code, value))
    extratags.append((GDAL_NODATA, _ASCII, 0, str(nodata), True))
    tifffile.imwrite(
        str(path),
        mask,
        photometric="minisblack",
        planarconfig="contig",
        software="recognition-adapter DEV-20261015-007",
        extratags=extratags,
    )


def read_mask_back(path):
    """Re-read a written mask and its tags for independent verification."""
    with tifffile.TiffFile(str(path)) as tf:
        page = tf.pages[0]
        arr = page.asarray()
        out = {"shape": tuple(int(v) for v in arr.shape), "dtype": str(arr.dtype), "tags": {}}
        for code in GEO_TAG_CODES:
            tg = page.tags.get(code)
            out["tags"][code] = _to_jsonable(tg.value) if tg is not None else None
        nd = page.tags.get(GDAL_NODATA)
        out["gdal_nodata"] = str(nd.value) if nd is not None else None
    return arr, out


def verify_mask_readback(mask, back_arr, back_tags, source_tags):
    """Hard-fail verification of a written mask.

    Raises ValueError on shape/dtype/pixel mismatch, on a NoData tag that is
    not 255, or on any GeoKey/GeoTIFF spatial tag that fails to round-trip when
    the source carried a spatial reference. Returns an honest summary dict
    that records 'absent_in_source' rather than claiming preservation when the
    source had no spatial reference to begin with.
    """
    if tuple(back_arr.shape) != tuple(mask.shape) or back_arr.dtype != mask.dtype:
        raise ValueError(
            f"mask re-read shape/dtype mismatch: {back_arr.shape}/{back_arr.dtype} "
            f"vs {mask.shape}/{mask.dtype}"
        )
    if not (back_arr == mask).all():
        raise ValueError("mask re-read pixel mismatch")
    if str(back_tags.get("gdal_nodata")) != str(NODATA_CLASS):
        raise ValueError(
            f"mask NoData read-back {back_tags.get('gdal_nodata')!r} != {NODATA_CLASS}"
        )

    src_has_spatial = any(source_tags.get(c) is not None for c in (33550, 33922, 34264, 34735))
    if src_has_spatial:
        for code in GEO_TAG_CODES:
            got = back_tags["tags"].get(code)
            want = source_tags.get(code)
            if got != want:
                raise ValueError(f"spatial tag {code} not preserved: {got!r} != {want!r}")

    return {
        "shape": back_tags["shape"],
        "dtype": back_tags["dtype"],
        "gdal_nodata": back_tags["gdal_nodata"],
        "spatial_tags_preserved": bool(src_has_spatial),
        "spatial_reference_state": "preserved" if src_has_spatial else "absent_in_source",
    }
