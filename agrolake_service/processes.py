"""仅管理本服务启动的进程树。Windows Job Object确保venv子孙进程一并退出。"""
import os
import signal
import subprocess


class ProcessTree:
    def __init__(self, command, *, cwd, env, stdout):
        self.job = None
        self.process = None
        if os.name == 'nt':
            import ctypes as c
            from ctypes import wintypes as w
            class Basic(c.Structure):
                _fields_ = [('process_time', c.c_int64), ('job_time', c.c_int64),
                            ('flags', w.DWORD), ('min_ws', c.c_size_t), ('max_ws', c.c_size_t),
                            ('active_limit', w.DWORD), ('affinity', c.c_size_t),
                            ('priority', w.DWORD), ('scheduling', w.DWORD)]
            class IO(c.Structure):
                _fields_ = [(name,c.c_uint64) for name in ('read_ops','write_ops','other_ops',
                                                         'read_bytes','write_bytes','other_bytes')]
            class Extended(c.Structure):
                _fields_ = [('basic',Basic), ('io',IO), ('process_memory',c.c_size_t),
                            ('job_memory',c.c_size_t), ('peak_process',c.c_size_t), ('peak_job',c.c_size_t)]
            kernel = c.WinDLL('kernel32', use_last_error=True)
            kernel.CreateJobObjectW.argtypes = [c.c_void_p,w.LPCWSTR]
            kernel.CreateJobObjectW.restype = w.HANDLE
            kernel.SetInformationJobObject.argtypes = [w.HANDLE,c.c_int,c.c_void_p,w.DWORD]
            kernel.SetInformationJobObject.restype = w.BOOL
            kernel.AssignProcessToJobObject.argtypes = [w.HANDLE,w.HANDLE]
            kernel.AssignProcessToJobObject.restype = w.BOOL
            kernel.TerminateJobObject.argtypes = [w.HANDLE,w.UINT]
            kernel.TerminateJobObject.restype = w.BOOL
            kernel.CloseHandle.argtypes = [w.HANDLE]
            kernel.CloseHandle.restype = w.BOOL
            self.kernel = kernel
            self.job = kernel.CreateJobObjectW(None,None)
            if not self.job:
                raise c.WinError(c.get_last_error())
            try:
                limits = Extended()
                limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
                if not kernel.SetInformationJobObject(self.job,9,c.byref(limits),c.sizeof(limits)):
                    raise c.WinError(c.get_last_error())
                self.process = subprocess.Popen(command,cwd=cwd,env=env,stdout=stdout,
                    stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
                if not kernel.AssignProcessToJobObject(self.job,int(self.process._handle)):
                    raise c.WinError(c.get_last_error())
            except BaseException:
                if self.process is not None and self.process.poll() is None:
                    # 仅在未能加入Job时回收刚启动的自身PID树，不扫描其他Python。
                    subprocess.run([os.path.join(os.environ['SystemRoot'],'System32','taskkill.exe'),
                                    '/PID',str(self.process.pid),'/T','/F'],capture_output=True,timeout=15)
                    self.process.wait(timeout=15)
                self.close()
                raise
        else:
            self.process = subprocess.Popen(command,cwd=cwd,env=env,stdout=stdout,
                                           stderr=subprocess.STDOUT,start_new_session=True)

    def terminate(self):
        if os.name == 'nt' and self.job:
            if not self.kernel.TerminateJobObject(self.job,1):
                import ctypes
                raise ctypes.WinError(ctypes.get_last_error())
        elif self.process is not None:
            try:
                os.killpg(self.process.pid,signal.SIGKILL)
            except ProcessLookupError:
                pass
        if self.process is not None:
            self.process.wait(timeout=15)

    def close(self):
        if self.job:
            self.kernel.CloseHandle(self.job)
            self.job = None
        elif os.name != 'nt' and self.process is not None:
            # 即使父进程已正常结束，也不保留其派生后台子进程。
            try:
                os.killpg(self.process.pid,signal.SIGKILL)
            except ProcessLookupError:
                pass
