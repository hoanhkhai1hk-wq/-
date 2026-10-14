"""内容寻址对象存储与路径边界。

- 对象文件名只由内容 SHA256 推导，绝不使用用户提供的文件名拼接目标路径。
- 所有写入路径必须解析在存储根目录之内，否则抛 PathBoundaryError。
- 原始文件不被覆盖：相同内容复用同一对象文件，不同内容必然得到不同对象文件。
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path


class PathBoundaryError(Exception):
    """目标路径逃逸出存储根目录。"""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def safe_name(original_filename: str) -> str:
    """仅取基名用于元数据展示，剥离任何目录分量与上跳。"""
    name = str(original_filename).replace("\\", "/").split("/")[-1]
    name = name.strip()
    return name or "unnamed"


def ensure_within(root: Path, candidate: Path) -> Path:
    """解析并确认 candidate 位于 root 内，返回解析后的绝对路径。"""
    root_r = root.resolve()
    cand_r = candidate.resolve()
    try:
        cand_r.relative_to(root_r)
    except ValueError as exc:
        raise PathBoundaryError(
            f"路径越界：{cand_r} 不在存储根 {root_r} 内"
        ) from exc
    return cand_r


def object_relpath(sha: str) -> str:
    if len(sha) < 3 or any(c not in "0123456789abcdef" for c in sha):
        raise ValueError(f"非法内容哈希：{sha!r}")
    return f"objects/{sha[:2]}/{sha}"


def object_path(root: Path, sha: str) -> Path:
    target = root / object_relpath(sha)
    ensure_within(root, target)
    return target


def store_bytes(root: Path, data: bytes) -> tuple[str, Path, bool]:
    """按内容存储字节，返回 (sha256, 路径, 是否新建)。"""
    sha = sha256_bytes(data)
    target = object_path(root, sha)
    if target.exists():
        return sha, target, False
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".tmp")
    ensure_within(root, tmp)
    with open(tmp, "wb") as fh:
        fh.write(data)
    os.replace(tmp, target)
    return sha, target, True


def store_file(root: Path, src: Path) -> tuple[str, Path, bool]:
    """按内容存储磁盘文件，返回 (sha256, 对象路径, 是否新建)。"""
    src = Path(src)
    if not src.is_file():
        raise FileNotFoundError(f"源文件不存在：{src}")
    sha = sha256_file(src)
    target = object_path(root, sha)
    if target.exists():
        return sha, target, False
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".tmp")
    ensure_within(root, tmp)
    with open(src, "rb") as rf, open(tmp, "wb") as wf:
        for block in iter(lambda: rf.read(1 << 20), b""):
            wf.write(block)
    os.replace(tmp, target)
    return sha, target, True


def relative_to_root(root: Path, path: Path) -> str:
    """返回相对存储根的 POSIX 风格相对路径，供数据库记录。"""
    return ensure_within(root, path).relative_to(root.resolve()).as_posix()
