"""进程级资源所有权。先锁定DB/存储/输出，再恢复旧作业；锁文件保留。"""
from pathlib import Path
import os


class OwnershipError(RuntimeError):
    pass


class ResourceLock:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.file = None

    def acquire(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fh = open(self.path, 'a+b')
        try:
            fh.seek(0)
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            fh.close()
            raise OwnershipError(f'识别服务已占用此资源，请先停止原服务：{self.path}') from exc
        self.file = fh
        return self

    def close(self):
        if self.file is not None:
            # Windows关闭文件句柄、POSIX关闭描述符都会释放内核锁，进程异常退出亦同。
            self.file.close()
            self.file = None
