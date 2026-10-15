"""独立内容寻址存储与路径边界（DEV-20261015-009，并发修复 DEV-20261015-010）。

与 P0 的 agrolake.storage 采用相同的对象布局 ``objects/<sha[:2]>/<sha>``，
但本模块独立实现，不导入或以任何方式修改 P0 包。规则：

- 对象文件名只由内容 SHA256 推导，绝不使用用户提供的文件名拼接路径。
- 所有写入路径必须解析在存储根目录之内，否则抛 PathBoundaryError。
- 相同内容复用同一对象文件；不同内容必然得到不同对象文件，历史不覆盖。
- DEV-010 修复：并发写不再共用同一 ``<sha>.tmp``；每个写入者使用存储目录内
  唯一的暂存文件再用硬链接发布到内容寻址目标，因此并发提交不再互相打断；
  既有目标对象不被覆盖（同摘要即同内容，已存在时直接复用）。
"""
from __future__ import annotations

import hashlib
import os
import tempfile
import threading
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


def _normalize_for_compare(path) -> str:
    """把路径规范为可比较形式：真实路径 + 去扩展前缀 + 统一大小写/分隔符。

    Windows 下 ``Path.resolve()`` 对本机已存在父目录会返回 ``\\\\?\\`` 扩展前缀，
    对不存在父目录则不会，直接字符串比较会误判越界。这里统一处理。
    """
    real = os.path.realpath(os.fspath(path))
    if real.startswith("\\\\?\\"):
        real = real[4:]
    return os.path.normcase(real)


def ensure_within(root: Path, candidate: Path) -> Path:
    """确认 candidate 位于 root 内，返回原 candidate（不做解析以免引入扩展前缀）。"""
    root_n = _normalize_for_compare(root)
    cand_n = _normalize_for_compare(candidate)
    if cand_n != root_n and not cand_n.startswith(root_n + os.sep):
        raise PathBoundaryError(
            f"路径越界：{candidate} 不在存储根 {root} 内"
        )
    return Path(candidate)


def object_relpath(sha: str) -> str:
    if len(sha) < 3 or any(c not in "0123456789abcdef" for c in sha):
        raise ValueError(f"非法内容哈希：{sha!r}")
    return f"objects/{sha[:2]}/{sha}"


def object_path(root: Path, sha: str) -> Path:
    target = Path(root) / object_relpath(sha)
    ensure_within(root, target)
    return target


# 同一进程内按内容摘要串行化写入，避免 12 个并发提交对同一目标互相打断。
_staging_locks = [threading.Lock() for _ in range(64)]


def _staging_lock(sha: str) -> threading.Lock:
    return _staging_locks[int(sha[:8], 16) % len(_staging_locks)]


def store_bytes(root: Path, data: bytes) -> tuple[str, Path, bool]:
    """按内容存储字节，返回 (sha256, 对象绝对路径, 是否本次新建)。

    并发安全：
    - 每个写入者在目标目录内创建唯一暂存文件（``.staging-*.tmp``），不再共用
      固定 ``<sha>.tmp``，因此并发写不会出现 PermissionError / FileNotFound。
    - 用同卷原子硬链接发布到内容寻址目标；目标已存在时核对摘要再复用，
      不覆盖既有对象，也不删除任何文件。
    """
    sha = sha256_bytes(data)
    target = object_path(root, sha)
    if target.is_file():
        if target.stat().st_size != len(data) or sha256_file(target) != sha:
            raise ValueError('已有内容对象损坏，保留原对象并拒绝复用')
        return sha, target, False
    target.parent.mkdir(parents=True, exist_ok=True)

    lock = _staging_lock(sha)
    with lock:
        if target.is_file():
            if target.stat().st_size != len(data) or sha256_file(target) != sha:
                raise ValueError('已有内容对象损坏，保留原对象并拒绝复用')
            return sha, target, False
        fd, tmp_name = tempfile.mkstemp(
            prefix=".staging-", suffix=".tmp", dir=os.fspath(target.parent)
        )
        tmp = Path(tmp_name)
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        try:
            # NTFS硬链接原子发布完整内容；目标存在时失败，跨进程也不覆盖。
            # 成功暂存是同一内容的硬链接，失败暂存亦保留，不执行删除。
            os.link(tmp, target)
            created = True
        except FileExistsError:
            if target.stat().st_size != len(data) or sha256_file(target) != sha:
                raise ValueError('已有内容对象损坏，保留原对象并拒绝复用')
            created = False
    return sha, target, created


def relative_to_root(root: Path, path: Path) -> str:
    """返回相对存储根的 POSIX 风格相对路径，供数据库记录。"""
    ensure_within(root, path)
    rel = os.path.relpath(
        os.path.realpath(os.fspath(path)), os.path.realpath(os.fspath(root))
    )
    return rel.replace(os.sep, "/")
