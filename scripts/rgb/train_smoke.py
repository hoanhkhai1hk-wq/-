"""RGB input adaptation and synthetic training smoke drill (TRN-20261016-001).

This module is a **software flow drill only**.  It never trains on the real
candidate data: real RGB/mask tensors are only *read and inspected*, while the
finite forward/loss/backward/optimizer/checkpoint steps run exclusively on tiny
in-memory synthetic 32x32 samples.  The tiny random network is a smoke-test
stub -- it is not an agricultural model, does not replace or modify the
existing FTW weights, and does not pre-select any final contest model.

Run with the existing CPU interpreter::

    .venv-recognition/Scripts/python.exe -B scripts/rgb/train_smoke.py smoke --output <fresh dir>
    .venv-recognition/Scripts/python.exe -B scripts/rgb/train_smoke.py inspect --input <005 dir>

Root is located from ``__file__`` so the script runs from any working directory.
Standard library + Torch + Numpy + Pillow only; the ``recognition`` package is
never imported.
"""

import argparse
import hashlib
import json
import os
import re
import sys
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

DEFAULT_CANDIDATE_DIR = os.path.join(
    "reports", "labels", "RGB-candidates-DATA-20261016-005"
)
# Leading English class tokens, in order, that identify each class id.  Chinese
# descriptions after the English tokens are allowed.
_EXPECTED_IDENTITY = {
    0: ["Field"],
    1: ["Field", "ridge"],
    2: ["Road"],
    255: ["ignore"],
}
_SAMPLE_ID_RE = re.compile(r"[A-Za-z0-9_-]{1,80}")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_WINDOWS_RESERVED = {
    "AUX", "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8",
    "COM9", "CON", "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7",
    "LPT8", "LPT9", "NUL", "PRN",
}
_MASK_VALUES = {0, 1, 2, 255}


class RGBError(Exception):
    """Structured failure for the RGB drill module."""


# --------------------------------------------------------------------------- #
# Path / hashing helpers
# --------------------------------------------------------------------------- #
def _project_root():
    return os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
    )


def _resolve_root(root):
    return _project_root() if root is None else os.path.abspath(root)


def _norm(path):
    return os.path.normcase(os.path.realpath(os.path.abspath(path)))


def _is_within(child, parent):
    c, p = _norm(child), _norm(parent)
    try:
        return os.path.commonpath([c, p]) == p
    except ValueError:
        return False


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _rel(path, root):
    return os.path.relpath(path, root).replace("\\", "/")


# --------------------------------------------------------------------------- #
# Validation helpers
# --------------------------------------------------------------------------- #
def _validate_sample_id(sid):
    if not isinstance(sid, str) or not _SAMPLE_ID_RE.fullmatch(sid):
        raise RGBError(f"非法 sample_id: {sid!r}")
    if sid.split(".")[0].upper() in _WINDOWS_RESERVED:
        raise RGBError(f"保留名 sample_id: {sid!r}")
    return sid


def _positive_int(value, field, sid):
    """Require a positive, non-bool integer manifest field."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RGBError(f"{sid}: {field} 必须为正整数（非 bool）: {value!r}")
    return value


def _sha256_str(value, field, sid):
    """Require a 64-char lowercase hex SHA-256 string (no re-hashing here)."""
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise RGBError(f"{sid}: {field} 必须为 64 位小写十六进制字符串: {value!r}")
    return value


def _english_tokens(text):
    if not isinstance(text, str):
        raise RGBError(f"类别描述不是字符串: {text!r}")
    tokens = []
    for tok in text.split():
        if tok.isascii() and tok.isalpha():
            tokens.append(tok)
        else:
            break
    return tokens


def _validate_policy(policy):
    if not isinstance(policy, dict):
        raise RGBError("policy.json 根不是对象")
    class_ids = policy.get("class_ids")
    if not isinstance(class_ids, dict):
        raise RGBError("policy.class_ids 缺失或不是对象")
    if set(str(k) for k in class_ids) != {"0", "1", "2", "255"}:
        raise RGBError("policy.class_ids 必须恰含 0/1/2/255")
    for cid, expected in _EXPECTED_IDENTITY.items():
        text = class_ids.get(str(cid))
        if _english_tokens(text) != expected:
            raise RGBError(f"policy 类别 {cid} 身份不符: {text!r}")
    if policy.get("no_background_class") is not True:
        raise RGBError("policy.no_background_class 必须为 true（0 是田块，不是背景）")
    alias = policy.get("alias_assumption")
    if not isinstance(alias, dict):
        raise RGBError("policy.alias_assumption 必须是对象")
    if alias.get("map") != {"Fieldridge": "Field ridge"}:
        raise RGBError("policy.alias_assumption.map 不符")
    if alias.get("status") != "pending_semantic_review":
        raise RGBError("policy.alias_assumption.status 必须为 pending_semantic_review")
    tp = policy.get("training_policy")
    if not isinstance(tp, dict):
        raise RGBError("policy.training_policy 必须是对象")
    if tp.get("semantic_reviewed") is not False:
        raise RGBError("policy.training_policy.semantic_reviewed 必须为 false")
    if tp.get("not_training_ready") is not True:
        raise RGBError("policy.training_policy.not_training_ready 必须为 true")
    if tp.get("no_train_test_split") is not True:
        raise RGBError("policy.training_policy.no_train_test_split 必须为 true")


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class RGBInspectionDataset(Dataset):
    """Read-only view over the DATA-20261016-005 candidate batch.

    ``len()`` counts candidates only; ``review`` records are excluded and their
    number is exposed as ``excluded_review_count``.  Constructor validation is
    strict: unknown or bad fields raise :class:`RGBError` instead of being
    silently skipped.
    """

    def __init__(self, input_dir=None, root=None):
        self.root = _resolve_root(root)
        if input_dir is None:
            input_dir = os.path.join(self.root, DEFAULT_CANDIDATE_DIR)
        elif not os.path.isabs(input_dir):
            # Relative input paths resolve against the project root, not os.getcwd().
            input_dir = os.path.join(self.root, input_dir)
        self.input_dir = os.path.abspath(input_dir)

        allowed_inputs = [
            os.path.join(self.root, "reports", "labels"),
            os.path.join(self.root, "reports", "rgb"),
        ]
        if not any(_is_within(self.input_dir, base) for base in allowed_inputs):
            raise RGBError(f"input_dir 不在允许范围（reports/labels 或 reports/rgb）: {self.input_dir}")

        self._source_roots = [
            os.path.join(self.root, "\u9093"),
            os.path.join(self.root, "hy0713 (1)"),
            os.path.join(self.root, "reports", "rgb"),
            os.path.join(self.root, "reports", "labels"),
        ]

        manifest = self._load_json(os.path.join(self.input_dir, "manifest.json"), "manifest.json")
        if not isinstance(manifest, dict):
            raise RGBError("manifest.json 根不是对象")
        records = manifest.get("records")
        if not isinstance(records, list):
            raise RGBError("manifest.records 不是列表")

        policy = self._load_json(os.path.join(self.input_dir, "policy.json"), "policy.json")
        _validate_policy(policy)
        self.policy = policy

        self.records = []
        self.excluded_review_count = 0
        seen_ids = set()
        seen_images = set()
        seen_jsons = set()
        seen_masks = set()
        for index, rec in enumerate(records):
            if not isinstance(rec, dict):
                raise RGBError(f"records[{index}] 不是对象")
            sid = _validate_sample_id(rec.get("sample_id"))
            key = sid.casefold()
            if key in seen_ids:
                raise RGBError(f"重复 sample_id: {sid}")
            seen_ids.add(key)
            status = rec.get("status")
            if status == "review":
                # Review rows only need counting; their candidate fields are not required.
                self.excluded_review_count += 1
                continue
            if status != "candidate":
                raise RGBError(f"{sid}: 未知 status: {status!r}")
            if rec.get("semantic_reviewed") is not False:
                raise RGBError(f"{sid}: semantic_reviewed 必须为 false")
            if rec.get("not_training_ready") is not True:
                raise RGBError(f"{sid}: not_training_ready 必须为 true")
            _positive_int(rec.get("width"), "width", sid)
            _positive_int(rec.get("height"), "height", sid)
            _positive_int(rec.get("image_bytes"), "image_bytes", sid)
            _positive_int(rec.get("json_bytes"), "json_bytes", sid)
            _sha256_str(rec.get("image_sha256"), "image_sha256", sid)
            _sha256_str(rec.get("json_sha256"), "json_sha256", sid)
            _sha256_str(rec.get("mask_sha256"), "mask_sha256", sid)
            image_abs = self._resolve_source_rel(rec.get("image_path"), sid)
            json_abs = self._resolve_source_rel(rec.get("json_path"), sid)
            mask_abs = self._resolve_derived_rel(rec.get("mask_path"), sid)
            image_key = _norm(image_abs)
            if image_key in seen_images:
                raise RGBError(f"{sid}: 重复源图引用")
            seen_images.add(image_key)
            json_key = _norm(json_abs)
            if json_key in seen_jsons:
                raise RGBError(f"{sid}: 重复源 JSON 引用")
            seen_jsons.add(json_key)
            mask_key = _norm(mask_abs)
            if mask_key in seen_masks:
                raise RGBError(f"{sid}: 重复候选 mask 引用")
            seen_masks.add(mask_key)
            self.records.append(rec)

        if not self.records:
            raise RGBError("没有可用的 candidate 记录")

    def _load_json(self, path, name):
        if not os.path.isfile(path):
            raise RGBError(f"缺少 {name}: {path}")
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, ValueError) as exc:
            raise RGBError(f"{name} 读取/解析失败: {exc}") from exc

    def _resolve_source_rel(self, rel, sid):
        if not isinstance(rel, str) or not rel or os.path.isabs(rel):
            raise RGBError(f"{sid}: 源路径非法: {rel!r}")
        norm = os.path.normpath(rel)
        if norm.startswith(".."):
            raise RGBError(f"{sid}: 源路径越界: {rel!r}")
        abs_path = os.path.join(self.root, norm)
        if not any(_is_within(abs_path, base) for base in self._source_roots):
            raise RGBError(f"{sid}: 源路径不在允许根内: {rel!r}")
        if not os.path.isfile(abs_path):
            raise RGBError(f"{sid}: 源文件不存在: {rel!r}")
        return abs_path

    def _resolve_derived_rel(self, rel, sid):
        if not isinstance(rel, str) or not rel or os.path.isabs(rel):
            raise RGBError(f"{sid}: 派生路径非法: {rel!r}")
        norm = os.path.normpath(rel)
        if norm.startswith(".."):
            raise RGBError(f"{sid}: 派生路径越界: {rel!r}")
        abs_path = os.path.join(self.input_dir, norm)
        if not _is_within(abs_path, self.input_dir):
            raise RGBError(f"{sid}: 派生路径越界: {rel!r}")
        return abs_path

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        rec = self.records[index]
        sid = rec["sample_id"]
        width, height = rec.get("width"), rec.get("height")
        image_path = self._resolve_source_rel(rec["image_path"], sid)
        mask_path = self._resolve_derived_rel(rec.get("mask_path"), sid)
        if not os.path.isfile(mask_path):
            raise RGBError(f"{sid}: mask 不存在: {rec.get('mask_path')!r}")

        if _sha256_file(image_path) != rec.get("image_sha256"):
            raise RGBError(f"{sid}: 源图 SHA 不符")
        if os.path.getsize(image_path) != rec.get("image_bytes"):
            raise RGBError(f"{sid}: 源图字节数不符 manifest")
        try:
            with Image.open(image_path) as im:
                if im.mode != "RGB":
                    raise RGBError(f"{sid}: 源图不是严格 RGB: {im.mode}")
                if im.size != (width, height):
                    raise RGBError(f"{sid}: 源图尺寸与 manifest 不符: {im.size}")
                arr = np.array(im, dtype=np.uint8)  # explicit copy off PIL buffer
        except RGBError:
            raise
        except (OSError, ValueError) as exc:
            raise RGBError(f"{sid}: 源图读取失败: {exc}") from exc
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise RGBError(f"{sid}: 源图维度异常: {arr.shape}")
        image = torch.from_numpy(arr.astype(np.float32) / 255.0).permute(2, 0, 1).contiguous()

        if _sha256_file(mask_path) != rec.get("mask_sha256"):
            raise RGBError(f"{sid}: mask SHA 不符")
        try:
            with Image.open(mask_path) as m:
                if m.format != "PNG":
                    raise RGBError(f"{sid}: mask 不是真正的 PNG: {m.format}")
                if m.mode != "L":
                    raise RGBError(f"{sid}: mask 模式不是 L: {m.mode}")
                if m.size != (width, height):
                    raise RGBError(f"{sid}: mask 尺寸与 manifest 不符: {m.size}")
                mask = np.array(m, dtype=np.uint8)  # explicit copy
        except RGBError:
            raise
        except (OSError, ValueError) as exc:
            raise RGBError(f"{sid}: mask 读取失败: {exc}") from exc

        values = set(int(v) for v in np.unique(mask))
        if not values.issubset(_MASK_VALUES):
            raise RGBError(f"{sid}: mask 值域非法: {sorted(values)}")
        valid_pixels = int(np.count_nonzero((mask == 0) | (mask == 1) | (mask == 2)))
        if valid_pixels <= 0:
            raise RGBError(f"{sid}: mask 无有效像素")
        target = torch.from_numpy(mask.astype(np.int64))

        return {
            "image": image,
            "target": target,
            "sample_id": sid,
            "status": rec["status"],
            "semantic_reviewed": rec.get("semantic_reviewed"),
            "training_ready": not rec.get("not_training_ready"),
        }


# --------------------------------------------------------------------------- #
# Loss and model
# --------------------------------------------------------------------------- #
def segmentation_loss(logits, target):
    """Mean cross-entropy over valid pixels; class 0 is a real class.

    ``ignore_index=255`` excludes ignore pixels from the mean (0 gradient), and
    an all-ignore target raises :class:`RGBError` instead of producing NaN.
    Returns a backward-able scalar tensor.
    """
    if not torch.is_tensor(logits) or not torch.is_tensor(target):
        raise RGBError("segmentation_loss 需要 Tensor 输入")
    if logits.dim() != 4 or target.dim() != 3:
        raise RGBError("形状必须为 logits[B,3,H,W] 与 target[B,H,W]")
    b, c, h, w = logits.shape
    if c != 3:
        raise RGBError(f"logits 第二维必须为 3 类，实际 {c}")
    if not torch.is_floating_point(logits):
        raise RGBError(f"logits 必须为浮点，实际 {logits.dtype}")
    if tuple(target.shape) != (b, h, w):
        raise RGBError(f"target 形状不匹配: {tuple(target.shape)} != {(b, h, w)}")
    if target.dtype != torch.int64:
        raise RGBError(f"target 必须为 int64，实际 {target.dtype}")
    if not bool(torch.isfinite(logits).all()):
        raise RGBError("logits 含非有限值")
    values = set(int(v) for v in torch.unique(target).tolist())
    if not values.issubset(_MASK_VALUES):
        raise RGBError(f"target 值域非法: {sorted(values)}")
    if bool((target == 255).all()):
        raise RGBError("target 全为忽略像素，无有效像素")
    return F.cross_entropy(logits, target, ignore_index=255)


class SmokeNet(nn.Module):
    """Tiny random 3-in/3-out network for the synthetic drill only.

    Preserves spatial size; no pretrained weights, no download, no FTW
    reference.  Not a contest model.
    """

    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 8, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, out_channels, 1),
        )

    def forward(self, x):
        return self.net(x)


class _SyntheticPNGDataset(Dataset):
    """Reads the synthetic PNGs back through explicit RGB/mask conversion."""

    def __init__(self, samples):
        self.samples = list(samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, mask_path = self.samples[index]
        with Image.open(image_path) as im:
            if im.mode != "RGB":
                raise RGBError(f"合成图不是 RGB: {im.mode}")
            arr = np.array(im, dtype=np.uint8)
        with Image.open(mask_path) as m:
            if m.format != "PNG" or m.mode != "L":
                raise RGBError(f"合成 mask 不是 PNG/L: {m.format}/{m.mode}")
            mask = np.array(m, dtype=np.uint8)
        image = torch.from_numpy(arr.astype(np.float32) / 255.0).permute(2, 0, 1).contiguous()
        target = torch.from_numpy(mask.astype(np.int64))
        return {"image": image, "target": target}


# --------------------------------------------------------------------------- #
# Output directory helper
# --------------------------------------------------------------------------- #
def _resolve_output_dir(output_dir, root, prefix):
    reports_rgb = os.path.join(root, "reports", "rgb")
    if output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        output_dir = os.path.join(reports_rgb, f"{prefix}-{stamp}-{os.getpid()}")
    elif not os.path.isabs(output_dir):
        # Relative output paths resolve against the project root, not os.getcwd().
        output_dir = os.path.join(root, output_dir)
    output_dir = os.path.abspath(output_dir)
    if not _is_within(output_dir, reports_rgb):
        raise RGBError(f"输出目录必须位于 reports/rgb 内: {output_dir}")
    if _norm(output_dir) == _norm(reports_rgb):
        raise RGBError("输出目录不能是 reports/rgb 本身")
    if _is_within(output_dir, os.path.join(root, "reports", "labels")):
        raise RGBError("输出目录不得位于 reports/labels 内")
    if os.path.exists(output_dir):
        raise RGBError(f"输出目录已存在，拒绝覆盖/重跑: {output_dir}")
    return output_dir


# --------------------------------------------------------------------------- #
# Synthetic smoke drill
# --------------------------------------------------------------------------- #
def _make_synthetic_sample(seed, size=32):
    """Small deterministic RGB image + 0/1/2/255 mask with an ignore border."""
    rng = np.random.default_rng(seed)
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[..., 0] = 30
    img[..., 1] = 60
    img[..., 2] = 90
    img[:, : size // 2] = (200, 40, 40)
    img[size // 2 :, :] = (40, 200, 40)
    jitter = int(rng.integers(0, 40))
    img = np.clip(img.astype(np.int32) + jitter, 0, 255).astype(np.uint8)

    mask = np.full((size, size), 255, dtype=np.uint8)  # border stays ignore
    mask[2 : size // 2, 2 : size // 2] = 0
    mask[size // 2 : size - 2, 2 : size // 2] = 1
    mask[2 : size - 2, size // 2 + 2 : size - 2] = 2
    return img, mask


def run_smoke(output_dir=None, root=None, steps=12, seed=17):
    """Finite CPU forward/loss/backward/step/checkpoint drill on synthetic data.

    Validates parameters and the output location *before* creating anything.
    Writes only into a brand-new directory under ``reports/rgb``.
    """
    root = _resolve_root(root)
    if isinstance(steps, bool) or not isinstance(steps, int) or not 1 <= steps <= 100:
        raise RGBError("steps 必须为 1..100 的整数（非 bool）")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2 ** 32 - 1:
        raise RGBError("seed 必须为 0..2**32-1 的整数（非 bool）")
    out = _resolve_output_dir(output_dir, root, "TRN-20261016-001-smoke")

    saved_torch_rng = torch.get_rng_state()
    saved_np_state = np.random.get_state()
    saved_threads = torch.get_num_threads()
    try:
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.set_num_threads(1)

        os.makedirs(out)
        images_dir = os.path.join(out, "synthetic", "images")
        masks_dir = os.path.join(out, "synthetic", "masks")
        os.makedirs(images_dir)
        os.makedirs(masks_dir)

        samples = []
        for i in range(4):
            img, mask = _make_synthetic_sample(seed * 100 + i)
            ip = os.path.join(images_dir, f"syn_{i}.png")
            mp = os.path.join(masks_dir, f"syn_{i}.png")
            Image.fromarray(img).save(ip, format="PNG")
            Image.fromarray(mask).save(mp, format="PNG")
            samples.append((ip, mp))

        dataset = _SyntheticPNGDataset(samples)
        loader = DataLoader(dataset, batch_size=2, shuffle=True)

        net = SmokeNet()
        before = {k: v.detach().clone() for k, v in net.state_dict().items()}
        optimizer = torch.optim.SGD(net.parameters(), lr=0.05)

        # Fixed reference batch: an independent, unshuffled DataLoader over the
        # same synthetic directory, taken after the network is initialised so it
        # is deterministic and independently reproducible.  ``initial_loss`` and
        # ``final_loss`` are both measured on this *same* batch (before any update
        # and after all updates), so they share one口径 and ``final_loss`` can be
        # recomputed from the saved checkpoint by an outside verifier.
        reference_loader = DataLoader(dataset, batch_size=2, shuffle=False)
        reference_batch = next(iter(reference_loader))
        reference_batch_indices = [0, 1]

        net.eval()
        with torch.no_grad():
            initial_loss = float(
                segmentation_loss(net(reference_batch["image"]), reference_batch["target"]).item()
            )

        net.train()
        def _cycles():
            while True:
                for batch in loader:
                    yield batch

        stream = _cycles()
        optimizer_steps = 0
        last_step_loss_before_update = initial_loss
        for _ in range(steps):
            batch = next(stream)
            logits = net(batch["image"])
            loss = segmentation_loss(logits, batch["target"])
            if not bool(torch.isfinite(loss)):
                raise RGBError("loss 为 NaN/Inf，演练失败")
            optimizer.zero_grad()
            loss.backward()
            for name, param in net.named_parameters():
                if param.grad is not None and not bool(torch.isfinite(param.grad).all()):
                    raise RGBError(f"梯度非有限（{name}），演练失败")
            optimizer.step()
            for name, param in net.named_parameters():
                if not bool(torch.isfinite(param).all()):
                    raise RGBError(f"参数非有限（{name}），演练失败")
            optimizer_steps += 1
            last_step_loss_before_update = float(loss.item())

        net.eval()
        with torch.no_grad():
            final_loss = float(
                segmentation_loss(net(reference_batch["image"]), reference_batch["target"]).item()
            )
            output_logits = net(reference_batch["image"])

        parameter_changed = any(
            not torch.equal(before[k], v.detach()) for k, v in net.state_dict().items()
        )
        if not parameter_changed:
            raise RGBError("参数未发生变化，演练失败")

        checkpoint_path = os.path.join(out, "synthetic_state.pt")
        torch.save(net.state_dict(), checkpoint_path)
        checkpoint_sha = _sha256_file(checkpoint_path)

        loaded = torch.load(checkpoint_path, weights_only=True, map_location="cpu")
        net_reloaded = SmokeNet()
        net_reloaded.load_state_dict(loaded, strict=True)
        net_reloaded.eval()
        with torch.no_grad():
            roundtrip_equal = bool(
                torch.equal(net(reference_batch["image"]), net_reloaded(reference_batch["image"]))
            )
        if not roundtrip_equal:
            raise RGBError("checkpoint 严格重载后同输入输出不一致，演练失败")

        artifacts = {}
        for name in os.listdir(images_dir):
            p = os.path.join(images_dir, name)
            artifacts[_rel(p, root)] = _sha256_file(p)
        for name in os.listdir(masks_dir):
            p = os.path.join(masks_dir, name)
            artifacts[_rel(p, root)] = _sha256_file(p)
        artifacts[_rel(checkpoint_path, root)] = checkpoint_sha

        summary = {
            "task_id": "TRN-20261016-001",
            "kind": "synthetic_smoke",
            "trained_on_real_data": False,
            "production_model": False,
            "real_dataset_training_ready": False,
            "optimizer_steps": optimizer_steps,
            "parameter_changed": bool(parameter_changed),
            "checkpoint_roundtrip_equal": roundtrip_equal,
            "initial_loss": initial_loss,
            "final_loss": final_loss,
            "reference_batch_indices": reference_batch_indices,
            "loss_reference": "same_fixed_synthetic_batch_before_after",
            "last_step_loss_before_update": last_step_loss_before_update,
            "seed": seed,
            "synthetic_sample_count": len(samples),
            "synthetic_size": 32,
            "input_shape": list(reference_batch["image"].shape),
            "input_dtype": str(reference_batch["image"].dtype),
            "target_shape": list(reference_batch["target"].shape),
            "target_dtype": str(reference_batch["target"].dtype),
            "output_shape": list(output_logits.shape),
            "output_dtype": str(output_logits.dtype),
            "checkpoint": _rel(checkpoint_path, root),
            "checkpoint_sha256": checkpoint_sha,
            "artifact_sha256": artifacts,
            "note": "合成软件演练，非农田模型训练；未使用任何真实候选数据；不预选最终模型。",
        }
        with open(os.path.join(out, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        with open(os.path.join(out, "README.md"), "w", encoding="utf-8") as f:
            f.write(
                "# 合成训练演练（TRN-20261016-001）\n\n"
                "本目录是一次**软件流程演练**的产物：在 4 张 32x32 合成 RGB 图与 0/1/2/255\n"
                "掩膜上执行了有限的 CPU 前向、损失、反向与参数更新，并保存/重载了随机小网络。\n\n"
                "- 训练数据：**仅内部合成样本**；未使用任何真实候选/照片。\n"
                "- 网络：微型随机 3 输入/3 输出网络，**不是参赛模型**，不替代/修改 FTW。\n"
                "- 结果字段见 `summary.json`；`optimizer_steps` 为真实步数，loss 不保证下降。\n"
                "- 全部产物只读参考，不代表任何农业精度或效果。\n"
            )
        return summary
    finally:
        torch.set_rng_state(saved_torch_rng)
        np.random.set_state(saved_np_state)
        torch.set_num_threads(saved_threads)


# --------------------------------------------------------------------------- #
# Real (read-only) inspection
# --------------------------------------------------------------------------- #
def inspect_real(input_dir=None, output_dir=None, root=None):
    """Read and validate every candidate tensor; never touch a model/optimizer.

    The fresh output location is resolved and validated *before* any input is
    traversed, and the directory is created only after *all* records validate
    successfully.  The output must not sit inside the input directory.
    """
    root = _resolve_root(root)
    if input_dir is None:
        resolved_input = os.path.join(root, DEFAULT_CANDIDATE_DIR)
    elif not os.path.isabs(input_dir):
        # Relative input paths resolve against the project root, not os.getcwd().
        resolved_input = os.path.join(root, input_dir)
    else:
        resolved_input = input_dir
    resolved_input = os.path.abspath(resolved_input)

    out = _resolve_output_dir(output_dir, root, "TRN-20261016-001-inspect")
    if _is_within(out, resolved_input):
        raise RGBError("输出目录不得位于输入目录内")

    dataset = RGBInspectionDataset(input_dir=resolved_input, root=root)

    class_totals = {0: 0, 1: 0, 2: 0}
    ignore_total = 0
    records = []
    for index in range(len(dataset)):
        item = dataset[index]
        rec = dataset.records[index]
        target = item["target"]
        unique, counts = torch.unique(target, return_counts=True)
        hist = {int(k): int(v) for k, v in zip(unique.tolist(), counts.tolist())}
        class_pixels = {c: hist.get(c, 0) for c in (0, 1, 2)}
        ignore_pixels = hist.get(255, 0)
        for c in (0, 1, 2):
            class_totals[c] += class_pixels[c]
        ignore_total += ignore_pixels
        image = item["image"]
        records.append({
            "sample_id": item["sample_id"],
            "status": item["status"],
            "semantic_reviewed": item["semantic_reviewed"],
            "training_ready": item["training_ready"],
            # Source paths are relative to the project root; mask_path is relative
            # to this input directory.  Shown so a record traces back to its origin.
            "image_path": rec.get("image_path"),
            "image_sha256": rec.get("image_sha256"),
            "json_path": rec.get("json_path"),
            "json_sha256": rec.get("json_sha256"),
            "mask_path": rec.get("mask_path"),
            "mask_sha256": rec.get("mask_sha256"),
            "image_shape": list(image.shape),
            "image_dtype": str(image.dtype),
            "image_min": float(image.min()),
            "image_max": float(image.max()),
            "target_shape": list(target.shape),
            "target_dtype": str(target.dtype),
            "class_pixels": class_pixels,
            "ignore_pixels": ignore_pixels,
            "total_pixels": int(target.numel()),
        })

    os.makedirs(out)

    summary = {
        "task_id": "TRN-20261016-001",
        "kind": "real_input_inspection",
        "optimizer_steps": 0,
        "trained_on_real_data": False,
        "semantic_reviewed": False,
        "training_ready": False,
        "split_created": False,
        "candidate_count": len(dataset),
        "excluded_review_count": dataset.excluded_review_count,
        "input_dir": _rel(dataset.input_dir, root),
        "manifest_sha256": _sha256_file(os.path.join(dataset.input_dir, "manifest.json")),
        "policy_sha256": _sha256_file(os.path.join(dataset.input_dir, "policy.json")),
        "class_pixel_totals": {str(k): v for k, v in class_totals.items()},
        "ignore_pixel_total": ignore_total,
        "class_ids": {"0": "Field 田块", "1": "Field ridge 田埂", "2": "Road 道路", "255": "ignore 忽略"},
        "note": "只读张量检查；未调用模型/优化器/反向，未训练，未切分。像素统计不等于精度或真值。",
    }
    with open(os.path.join(out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out, "records.json"), "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    return summary


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser():
    parser = argparse.ArgumentParser(
        prog="train_smoke",
        description="RGB 输入适配与合成训练演练（仅软件流程；不训练真实农田模型；无真实训练命令）。",
    )
    sub = parser.add_subparsers(dest="command")
    smoke = sub.add_parser("smoke", help="合成 32x32 小样本上的有限 CPU 演练")
    smoke.add_argument("--output", default=None, help="输出目录（reports/rgb 下全新目录）")
    smoke.add_argument("--steps", type=int, default=12, help="优化步数，1..100")
    smoke.add_argument("--seed", type=int, default=17, help="随机种子")
    smoke.add_argument("--root", default=None, help="项目根（默认按脚本位置）")

    inspect = sub.add_parser("inspect", help="只读检查真实候选张量，不训练")
    inspect.add_argument("--input", default=None, help="005 候选目录（reports/labels 或 reports/rgb 内）")
    inspect.add_argument("--output", default=None, help="输出目录（reports/rgb 下全新目录）")
    inspect.add_argument("--root", default=None, help="项目根（默认按脚本位置）")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 0
    try:
        if args.command == "smoke":
            summary = run_smoke(output_dir=args.output, root=args.root, steps=args.steps, seed=args.seed)
            print(json.dumps({
                "kind": summary["kind"],
                "optimizer_steps": summary["optimizer_steps"],
                "initial_loss": summary["initial_loss"],
                "final_loss": summary["final_loss"],
                "parameter_changed": summary["parameter_changed"],
                "checkpoint_roundtrip_equal": summary["checkpoint_roundtrip_equal"],
            }, ensure_ascii=False))
        else:
            summary = inspect_real(input_dir=args.input, output_dir=args.output, root=args.root)
            print(json.dumps({
                "kind": summary["kind"],
                "candidate_count": summary["candidate_count"],
                "excluded_review_count": summary["excluded_review_count"],
                "optimizer_steps": summary["optimizer_steps"],
            }, ensure_ascii=False))
        return 0
    except RGBError as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001 - surface unexpected failures clearly
        print(f"未预期错误：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
