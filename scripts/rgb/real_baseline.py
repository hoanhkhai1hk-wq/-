"""Candidate selection for a conservative RGB internal baseline (TRN-20261017-002).

This module ONLY classifies existing DATA-20261016-005 manifest records into
three dispositions and writes a selection package.  It never trains, never
predicts, never creates train/val/test splits, and never modifies any source
or candidate file.  ``experiment_train`` merely marks records as allowed for a
separate *experimental internal* baseline; ``semantic_reviewed`` stays false for
every record and no generalization metrics are claimed.

CLI::

    python scripts/rgb/real_baseline.py select --input <005 dir> --output <fresh dir under reports/rgb>

``train`` is intentionally not a command.  Root is located from ``__file__``;
relative paths resolve against the project root, never against the caller's
working directory.  Real record counts (734/121/48/855 and so on) are never
hard-coded: everything is derived from the manifest at run time.
"""

import argparse
import csv
import json
import importlib.util
import os
import sys
from datetime import datetime

_HERE = os.path.dirname(os.path.abspath(__file__))
_SPEC = importlib.util.spec_from_file_location("rgb_selection_input", os.path.join(_HERE, "train_smoke.py"))
ts = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ts)

SCHEMA_VERSION = 1
TASK_ID = "TRN-20261017-002"
# original label -> derived class (0 Field, 1 Field ridge/Fieldridge, 2 Road)
_LABEL_CLASS = {"Field": 0, "Field ridge": 1, "Fieldridge": 1, "Road": 2}
_DISPOSITIONS = ("excluded_user_decision", "deferred_alias", "experiment_train")


class BaselineError(Exception):
    """Structured failure of the candidate selection module."""


# --------------------------------------------------------------------------- #
# Output location (fresh directory under reports/rgb, never near the input)
# --------------------------------------------------------------------------- #
def _resolve_output_dir(output_dir, root, input_dir):
    if output_dir is not None and not isinstance(output_dir, str):
        raise BaselineError(f"output_dir 必须为字符串: {output_dir!r}")
    if output_dir == "":
        raise BaselineError("output_dir 不能为空字符串")
    reports_rgb = os.path.join(root, "reports", "rgb")
    if output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        output_dir = os.path.join(reports_rgb, f"{TASK_ID}-select-{stamp}-{os.getpid()}")
    elif not os.path.isabs(output_dir):
        output_dir = os.path.join(root, output_dir)
    output_dir = os.path.abspath(output_dir)
    if not ts._is_within(output_dir, reports_rgb):
        raise BaselineError(f"输出目录必须位于 reports/rgb 内: {output_dir}")
    if ts._norm(output_dir) == ts._norm(reports_rgb):
        raise BaselineError("输出目录不能是 reports/rgb 本身")
    if ts._norm(output_dir) == ts._norm(input_dir):
        raise BaselineError("输出目录不得与输入目录相同")
    if ts._is_within(input_dir, output_dir):
        raise BaselineError("输出目录不得是输入目录的祖先")
    if ts._is_within(output_dir, input_dir):
        raise BaselineError("输出目录不得位于输入目录内")
    if os.path.exists(output_dir):
        raise BaselineError(f"输出目录已存在，拒绝覆盖/重跑: {output_dir}")
    return output_dir


def _load_json(path, name):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError) as exc:
        raise BaselineError(f"{name} 读取/解析失败: {exc}") from exc


# --------------------------------------------------------------------------- #
# Per-record verification (original JSON is the only truth for aliases)
# --------------------------------------------------------------------------- #
def _check_file(path, sha, nbytes, sid, what):
    if not os.path.isfile(path):
        raise BaselineError(f"{sid}: {what} 不存在: {path}")
    if type(nbytes) is not int or nbytes <= 0 or os.path.getsize(path) != nbytes:
        raise BaselineError(f"{sid}: {what} 字节数不符")
    if ts._sha256_file(path) != sha:
        raise BaselineError(f"{sid}: {what} SHA 不符")


def _check_labels(json_abs, rec, sid):
    """Validate shapes/labels/label_mapping/alias consistency from the source JSON."""
    obj = _load_json(json_abs, f"{sid} 源JSON")
    shapes = obj.get("shapes") if isinstance(obj, dict) else None
    if not isinstance(shapes, list) or not shapes:
        raise BaselineError(f"{sid}: shapes 缺失或为空")
    labels = []
    for i, shape in enumerate(shapes):
        if not isinstance(shape, dict) or shape.get("shape_type") != "polygon":
            raise BaselineError(f"{sid}: shape[{i}] 缺失或 shape_type 不是 polygon")
        label = shape.get("label")
        if not isinstance(label, str) or label not in _LABEL_CLASS:
            raise BaselineError(f"{sid}: 未知标签 {label!r}")
        labels.append(label)
    mapping = rec.get("label_mapping")
    if not isinstance(mapping, list) or len(mapping) != len(labels):
        raise BaselineError(f"{sid}: label_mapping 缺失或与 shapes 数量不一致")
    for i, (entry, label) in enumerate(zip(mapping, labels)):
        if (
            not isinstance(entry, dict)
            or type(entry.get("shape_index")) is not int
            or type(entry.get("derived_class")) is not int
            or entry.get("shape_index") != i
            or entry.get("original_label") != label
            or entry.get("derived_class") != _LABEL_CLASS[label]
        ):
            raise BaselineError(f"{sid}: label_mapping[{i}] 与源JSON不一致")
    fridge = {i for i, label in enumerate(labels) if label == "Fieldridge"}
    alias = rec.get("alias_normalized")
    if not isinstance(alias, list):
        raise BaselineError(f"{sid}: alias_normalized 不是列表")
    used = set()
    for entry in alias:
        idx = entry.get("shape_index") if isinstance(entry, dict) else None
        if type(idx) is not int or idx not in fridge or idx in used or entry.get("from") != "Fieldridge" or entry.get("to") != "Field ridge":
            raise BaselineError(f"{sid}: alias_normalized 与原JSON不一致，拒绝伪造: {entry!r}")
        used.add(idx)
    if used != fridge:
        raise BaselineError(f"{sid}: alias_normalized 覆盖与原JSON的 Fieldridge 集合不一致，拒绝伪造")
    return labels


def _verify_source(rec, dataset, sid):
    image_abs = dataset._resolve_source_rel(rec.get("image_path"), sid)
    _check_file(image_abs, rec.get("image_sha256"), rec.get("image_bytes"), sid, "源图")
    json_abs = dataset._resolve_source_rel(rec.get("json_path"), sid)
    _check_file(json_abs, rec.get("json_sha256"), rec.get("json_bytes"), sid, "源JSON")
    return _check_labels(json_abs, rec, sid)


def _verify_candidate_pixels(dataset, index, rec, sid):
    """dataset[index] pixel counts must match the manifest pixel_stats exactly."""
    target = dataset[index]["target"]
    unique, counts_t = __import__("torch").unique(target, return_counts=True)
    hist = {int(k): int(v) for k, v in zip(unique.tolist(), counts_t.tolist())}
    stats = rec.get("pixel_stats")
    if not isinstance(stats, dict):
        raise BaselineError(f"{sid}: 缺少 pixel_stats（真实 schema 必填，不放松）")
    if type(stats.get("denominator")) is not int or stats.get("denominator") != int(target.numel()):
        raise BaselineError(f"{sid}: pixel_stats.denominator 与掩膜总像素不符")
    class_pixels = stats.get("class_pixels")
    if not isinstance(class_pixels, dict):
        raise BaselineError(f"{sid}: pixel_stats.class_pixels 缺失")
    for c in (0, 1, 2):
        if type(class_pixels.get(str(c))) is not int or class_pixels.get(str(c)) != hist.get(c, 0):
            raise BaselineError(f"{sid}: 类 {c} 像素计数与掩膜不符")
    if type(stats.get("ignore_pixels")) is not int or stats.get("ignore_pixels") != hist.get(255, 0):
        raise BaselineError(f"{sid}: 忽略像素计数与掩膜不符")


def _build_record(rec, sid, status, disposition):
    record = {
        "sample_id": sid,
        "dataset": rec.get("dataset"),
        "status": status,
        "disposition": disposition,
        "image_path": rec.get("image_path"),
        "image_sha256": rec.get("image_sha256"),
        "json_path": rec.get("json_path"),
        "json_sha256": rec.get("json_sha256"),
        "width": rec.get("width"),
        "height": rec.get("height"),
        "semantic_reviewed": False,
        "independent_test_available": False,
        "source_relationship": "unknown",
    }
    if disposition == "excluded_user_decision":
        record["review_reasons"] = rec.get("review_reasons")
        record["unsupported_reasons"] = rec.get("unsupported_reasons")
    else:
        record["mask_path"] = rec.get("mask_path")
        record["mask_sha256"] = rec.get("mask_sha256")
        if disposition == "experiment_train":
            record["experimental_training_allowed"] = True
    return record


# --------------------------------------------------------------------------- #
# Selection
# --------------------------------------------------------------------------- #
def select_candidates(input_dir=None, output_dir=None, root=None):
    root = ts._resolve_root(root)
    dataset = ts.RGBInspectionDataset(input_dir=input_dir, root=root)
    input_abs = dataset.input_dir
    out = _resolve_output_dir(output_dir, root, input_abs)

    manifest = _load_json(os.path.join(input_abs, "manifest.json"), "manifest.json")
    records = manifest.get("records")
    if not isinstance(records, list) or not records:
        raise BaselineError("manifest.records 缺失或为空")

    cand_index = {rec["sample_id"]: i for i, rec in enumerate(dataset.records)}
    seen = set()
    out_records = []
    counts = {name: 0 for name in _DISPOSITIONS}
    for rec in records:
        if not isinstance(rec, dict):
            raise BaselineError("manifest 中存在非对象记录")
        sid = ts._validate_sample_id(rec.get("sample_id"))
        key = sid.casefold()
        if key in seen:
            raise BaselineError(f"重复 sample_id: {sid}")
        seen.add(key)
        status = rec.get("status")
        if status == "review":
            # review rows are fully verified too (source/SHA/bytes/JSON labels)
            _verify_source(rec, dataset, sid)
            disposition = "excluded_user_decision"
        elif status == "candidate":
            if sid not in cand_index:
                raise BaselineError(f"{sid}: 候选未被数据集接受")
            labels = _verify_source(rec, dataset, sid)
            # alias decision comes from the ORIGINAL JSON only, never alias_normalized
            disposition = "deferred_alias" if "Fieldridge" in labels else "experiment_train"
            _verify_candidate_pixels(dataset, cand_index[sid], rec, sid)
        else:
            raise BaselineError(f"{sid}: 未知 status: {status!r}")
        counts[disposition] += 1
        out_records.append(_build_record(rec, sid, status, disposition))

    summary = {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "kind": "candidate_selection",
        "input_dir": ts._rel(input_abs, root),
        "manifest_sha256": ts._sha256_file(os.path.join(input_abs, "manifest.json")),
        "policy_sha256": ts._sha256_file(os.path.join(input_abs, "policy.json")),
        "total_records": len(out_records),
        "dispositions": counts,
        "semantic_reviewed": False,
        "not_training_ready": True,
        "split_created": False,
        "trained_on_real_data": False,
        "source_relationship_default": "unknown",
        "independent_test_available_default": False,
        "notes": [
            "review=excluded_user_decision；含Fieldridge(以原JSON为准)=deferred_alias；其余candidate=experiment_train。",
            "experiment_train 仅为未来实验性内部基线许可，不等于训练就绪；无独立验证/测试集，不报泛化精度。",
            "所有条目 semantic_reviewed=false、source_relationship=unknown；选择命令不执行训练，独立训练记录见fit_baseline.py产物。",
        ],
        "records": out_records,
    }

    os.makedirs(out)
    with open(os.path.join(out, "selection.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    csv_path = os.path.join(out, "records.csv")
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "sample_id", "dataset", "status", "disposition", "image_path", "json_path",
            "mask_path", "semantic_reviewed", "independent_test_available",
            "source_relationship", "experimental_training_allowed",
        ])
        for record in out_records:
            writer.writerow([
                record["sample_id"], record.get("dataset") or "", record["status"], record["disposition"],
                record["image_path"], record["json_path"], record.get("mask_path") or "",
                record["semantic_reviewed"], record["independent_test_available"],
                record["source_relationship"], "true" if record.get("experimental_training_allowed") else "",
            ])
    with open(os.path.join(out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in summary.items() if k != "records"}, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out, "README.md"), "w", encoding="utf-8") as f:
        f.write(
            f"# 候选选择结果（{TASK_ID}）\n\n"
            "- 本目录仅是对既有候选清单的**分类选择**，未训练、未预测、未切分、未修改任何原文件。\n"
            "- `selection.json`：全部记录与去向；`records.csv`：UTF-8 BOM 表格；`summary.json`：计数摘要。\n"
            "- 去向：review→excluded_user_decision；原JSON含Fieldridge→deferred_alias；其余candidate→experiment_train。\n"
            "- experiment_train 仅表示允许未来实验性内部基线；semantic_reviewed 仍为 false，来源关系未知。\n"
        )
    return summary


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser():
    parser = argparse.ArgumentParser(
        prog="real_baseline",
        description="RGB 内部基线候选选择（仅选择/分类；无训练、无预测命令）。",
    )
    sub = parser.add_subparsers(dest="command")
    select = sub.add_parser("select", help="对候选清单做三去向选择并输出选择包")
    select.add_argument("--input", default=None, help="005 候选目录（reports/labels 或 reports/rgb 内）")
    select.add_argument("--output", default=None, help="输出目录（reports/rgb 下全新目录）")
    select.add_argument("--root", default=None, help="项目根（默认按脚本位置）")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command != "select":
        parser.print_help()
        return 0 if args.command is None else 2
    try:
        summary = select_candidates(input_dir=args.input, output_dir=args.output, root=args.root)
    except ts.RGBError as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2
    except BaselineError as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001
        print(f"未预期错误：{exc}", file=sys.stderr)
        return 1
    print(json.dumps({
        "kind": summary["kind"],
        "total_records": summary["total_records"],
        "dispositions": summary["dispositions"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
