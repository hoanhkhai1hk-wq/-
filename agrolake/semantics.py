"""语义字典、字段别名与单位口径。

规则与字典带版本；保留原始值，仅对已定义别名/单位标准化。
未定义或不合法的单位不默认为平方米。
"""
from __future__ import annotations

import json
import math
from functools import lru_cache
from typing import Any

from . import config


@lru_cache(maxsize=1)
def load_semantic_rules() -> dict[str, Any]:
    with open(config.SEMANTIC_RULES_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def semantic_version() -> str:
    return load_semantic_rules()["semantic_version"]


def _alias_index(section: str) -> dict[str, str]:
    rules = load_semantic_rules()[section]
    index: dict[str, str] = {}
    if section == "field_aliases":
        for canonical, aliases in rules.items():
            for a in aliases:
                index[a.strip().lower()] = canonical
    else:  # category_aliases
        for canonical, aliases in rules.items():
            for a in aliases:
                index[a.strip().lower()] = canonical
    return index


def resolve_columns(header: list[str]) -> tuple[dict[str, str], list[str]]:
    """把 CSV 表头映射到标准列；返回 (canonical->原列名字符串, 未识别列)。

    映射值为表头中的原始字符串（保留空白），供后续按名取值。
    """
    idx = _alias_index("field_aliases")
    mapping: dict[str, str] = {}
    unknown: list[str] = []
    for col in header:
        raw = str(col)
        canonical = idx.get(raw.strip().lower())
        if canonical and canonical not in mapping:
            mapping[canonical] = raw
        else:
            unknown.append(raw)
    return mapping, unknown


def map_category(raw: str) -> str | None:
    return _alias_index("category_aliases").get(str(raw).strip().lower())


def category_dict_version() -> str:
    """类别/字段字典版本，随语义字典一同发布。"""
    return load_semantic_rules()["semantic_version"]


def normalize_category(raw: Any) -> tuple[str | None, str | None]:
    """把原始地物类别标准化。返回 (标准值 | None, 说明 | None)。

    未知或空值不回退猜测：标准值为 None，并返回可解释的说明，由调用方保留原值并登记。
    """
    if raw is None or str(raw).strip() == "":
        return None, "地物类别值为空，保留为空并记录"
    std = map_category(raw)
    if std is None:
        return None, f"未知地物类别（保留原值，不凭空识别）：{raw!r}"
    return std, None


def asset_identity(filename: str, scene_id: str | None, source: str | None) -> tuple[str, dict]:
    """业务资产身份 = (规范化文件名, 声明场景, 声明来源)。

    同名文件出现在不同场景或来源时视为不同资产，避免被当作同一资产的版本更新。
    返回 (资产键, 身份明细)。
    """
    fn = str(filename).strip().lower()
    scene = (str(scene_id).strip() if scene_id not in (None, "") else "") or "-"
    src = (str(source).strip() if source not in (None, "") else "") or "-"
    key = f"{fn}|scene={scene}|source={src}"
    return key, {"filename": fn, "declared_scene_id": scene, "source": src}


def normalize_unit(unit_raw: str | None) -> str | None:
    """返回单位规则键（m2/ha/mu），未定义返回 None。"""
    if unit_raw is None:
        return None
    units = load_semantic_rules()["units"]
    key = str(unit_raw).strip().lower()
    for name, spec in units.items():
        if key in [a.lower() for a in spec["aliases"]] or key == name.lower():
            return name
    return None


def convert_area(area_raw: Any, unit_raw: str | None) -> tuple[float | None, str, str]:
    """面积换算为平方米。

    返回 (area_m2 | None, status, reason)。status ∈ {"ok","error"}。
    负数、非有限、空值与未定义单位均判为错误，不默认平方米。
    """
    if area_raw is None or str(area_raw).strip() == "":
        return None, "error", "面积值缺失"
    try:
        value = float(str(area_raw).strip())
    except (TypeError, ValueError):
        return None, "error", f"面积值无法解析为数值：{area_raw!r}"
    if not math.isfinite(value):
        return None, "error", f"面积值非有限：{area_raw!r}"
    if value < 0:
        return None, "error", f"面积值为负数：{area_raw!r}"
    unit_key = normalize_unit(unit_raw)
    if unit_key is None:
        return None, "error", f"未定义面积单位：{unit_raw!r}"
    factor = load_semantic_rules()["units"][unit_key]["factor_to_m2"]
    result = value * factor
    if not math.isfinite(result):
        return None, "error", f"换算结果非有限：{area_raw!r}×{unit_key}"
    return result, "ok", f"{unit_key}×{factor}→m2"
