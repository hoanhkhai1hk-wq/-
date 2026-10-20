"""Direct SMP adapter for the FTW v1 CC-BY checkpoint (DEV-20261015-006).

This is NOT ftw-tools 1.4.3 and must not be described as such.
"""

from importlib import import_module

# Keep optional FTW dependencies out of the RGB entry point. In particular,
# RGB inference must configure its CPU runtime before importing torch.
_EXPORTS = {
    "load_model": "checkpoint",
    "read_image": "raster_io",
    "write_mask": "raster_io",
    "read_mask_back": "raster_io",
    "verify_mask_readback": "raster_io",
    "predict": "inference",
    "class_counts": "tables",
    "field_table": "tables",
}


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f".{_EXPORTS[name]}", __name__), name)
    globals()[name] = value
    return value

__all__ = [
    "load_model",
    "read_image",
    "write_mask",
    "read_mask_back",
    "verify_mask_readback",
    "predict",
    "class_counts",
    "field_table",
]
