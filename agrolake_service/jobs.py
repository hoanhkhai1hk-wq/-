"""单 worker 串行作业队列与识别持久化（DEV-20261015-009）。

- 单 worker 串行消费；状态真实变化 queued -> running -> succeeded/failed。
- 每次数据库操作由调用线程自行新建并关闭连接（连接不跨线程复用）。
- 重启把遗留 queued/running 标为 interrupted，绝不悄悄重放。
- 成功前把候选区域、类计数与 run.json 事实在同一事务内写入。
"""
from __future__ import annotations

import json
import queue
import threading
import uuid
from pathlib import Path

from . import config as cfg
from . import db, store, validation
from .ownership import ResourceLock


class JobManager:
    def __init__(self, config: cfg.ServiceConfig, runner, *, auto_start: bool = False):
        self.config = config
        self.runner = runner
        self.runs_root = Path(config.runs_root)
        self.store_root = Path(config.store_root)
        self.db_path = Path(config.db_path)
        self._queue: queue.Queue = queue.Queue()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._closed = False
        self._lifecycle = threading.RLock()
        self._locks = []

        try:
            resources = {self.db_path.with_name(self.db_path.name + '.recognition.lock'),
                         self.store_root / '.recognition.lock',self.runs_root / '.recognition.lock'}
            for path in sorted(resources,key=str):
                self._locks.append(ResourceLock(path).acquire())
            # 独占权确认后才能恢复旧作业。第二实例不会改写任何SQLite记录。
            conn = db.connect(self.db_path)
            try:
                db.init_schema(conn)
                self._mark_interrupted(conn)
            finally:
                conn.close()
        except BaseException:
            for lock in reversed(self._locks):
                lock.close()
            raise
        if auto_start:
            self.start()

    # ---- worker lifecycle -------------------------------------------------

    def start(self) -> None:
        if self._closed:
            raise RuntimeError('服务已停止，不能再次启动同一manager')
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._worker_loop, name="rec-worker", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 30.0) -> None:
        with self._lifecycle:
            if not self._locks:
                return
            self._closed = True
            self._stop.set()
            cancel = getattr(self.runner, 'cancel', None)
            if callable(cancel):
                cancel()
            if self._thread is not None:
                self._thread.join(timeout=timeout)
                if self._thread.is_alive():
                    raise RuntimeError('识别工作线程尚未退出，保留所有权锁，不能启动第二服务')
            conn = db.connect(self.db_path)
            try:
                self._mark_interrupted(conn)
            finally:
                conn.close()
            for lock in reversed(self._locks):
                lock.close()
            self._locks = []

    def _worker_loop(self) -> None:
        while not self._stop.is_set():
            try:
                job_id = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                self._process(job_id)
            except Exception as exc:  # noqa: BLE001 - worker must never die silently
                self._record_unexpected(job_id, exc)
            finally:
                self._queue.task_done()

    @staticmethod
    def _mark_interrupted(conn) -> None:
        now = db.now()
        conn.execute(
            "UPDATE rec_jobs SET status='interrupted',"
            " error=COALESCE(error, '服务重启：作业未完成，已标记中断，未自动重放'),"
            " finished_at=? WHERE status IN ('queued','running')",
            (now,),
        )
        conn.execute(
            "UPDATE rec_runs SET status='interrupted', finished_at=?"
            " WHERE status='running'",
            (now,),
        )
        conn.commit()

    # ---- submission -------------------------------------------------------

    def submit(self, normalized: dict) -> dict:
        with self._lifecycle:
            if self._closed:
                raise validation.ValidationError('识别服务正在停止，请稍后重启再提交',503)
            return self._submit(normalized)

    def _submit(self, normalized: dict) -> dict:
        """存储内容、登记输入并排队作业；返回 {'job_id','created','input_sha256','sample_kind','request_id'}。

        同 request_id 不同内容 -> 抛 IdempotencyConflict。
        """
        content = normalized["content"]
        sha, obj_path, _created = store.store_bytes(self.store_root, content)
        relpath = store.relative_to_root(self.store_root, obj_path)
        sample_kind = self.config.classify_sample(sha)
        request_id = normalized.get("request_id")
        fp = validation.fingerprint(normalized)

        conn = db.connect(self.db_path)
        try:
            if request_id:
                existing = conn.execute(
                    "SELECT request_id, request_fingerprint, job_id FROM rec_requests"
                    " WHERE request_id = ?",
                    (request_id,),
                ).fetchone()
                if existing:
                    return self._idempotent_result(conn, existing, normalized, fp)

            now = db.now()
            conn.execute(
                "INSERT OR IGNORE INTO rec_inputs(sha256, object_relpath, size_bytes,"
                " original_filename, sample_kind, created_at) VALUES (?,?,?,?,?,?)",
                (sha, relpath, len(content), normalized["filename"], sample_kind, now),
            )
            input_row = conn.execute(
                "SELECT input_id FROM rec_inputs WHERE sha256 = ?", (sha,)
            ).fetchone()
            input_id = input_row["input_id"]

            cur = conn.execute(
                "INSERT INTO rec_jobs(input_id, input_sha256, original_filename, source,"
                " source_link, license_status, scene_id, sample_kind, request_id, status,"
                " model_path, model_sha256, cli_path, worker_python, created_at)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    input_id, sha, normalized["filename"], normalized["source"],
                    normalized["source_link"], normalized["license_status"],
                    normalized["scene_id"], sample_kind, request_id, "queued",
                    str(self.config.model_path), self.config.model_sha256,
                    str(self.config.cli_script), str(self.config.worker_python), now,
                ),
            )
            job_id = cur.lastrowid
            if request_id:
                # submit由生命周期锁串行化；同一资源仅有一个manager，不掩盖存储异常。
                conn.execute(
                    "INSERT INTO rec_requests(request_id, request_fingerprint, job_id,"
                    " created_at) VALUES (?,?,?,?)",(request_id, fp, job_id, now))
            conn.commit()
        finally:
            conn.close()

        self._queue.put(job_id)
        return {
            "job_id": job_id,
            "created": True,
            "input_sha256": sha,
            "sample_kind": sample_kind,
            "request_id": request_id,
        }

    def _idempotent_result(self, conn, existing, normalized, fp) -> dict:
        if existing["request_fingerprint"] != fp:
            raise IdempotencyConflict(
                f"request_id {existing['request_id']!r} 已存在但内容不同，拒绝复用"
            )
        job = conn.execute(
            "SELECT input_sha256, sample_kind FROM rec_jobs WHERE job_id = ?",
            (existing["job_id"],),
        ).fetchone()
        return {
            "job_id": existing["job_id"],
            "created": False,
            "input_sha256": job["input_sha256"] if job else normalized["content_sha256"],
            "sample_kind": job["sample_kind"] if job else None,
            "request_id": existing["request_id"],
        }

    # ---- processing -------------------------------------------------------

    def _process(self, job_id: int) -> None:
        conn = db.connect(self.db_path)
        try:
            job = conn.execute("SELECT * FROM rec_jobs WHERE job_id = ?", (job_id,)).fetchone()
            if job is None or job["status"] != "queued":
                # 非 queued（例如已被标为 interrupted）不重放。
                return

            started = db.now()
            conn.execute(
                "UPDATE rec_jobs SET status='running', started_at=?,"
                " attempt_count = attempt_count + 1 WHERE job_id = ?",
                (started, job_id),
            )
            conn.commit()

            run_dir = self.runs_root / f"job-{job_id:06d}-{uuid.uuid4().hex}"
            run_dir.mkdir(parents=True, exist_ok=False)
            cli_out = run_dir / "cli_output"            # 不提前创建，交给 CLI 独占创建
            log_path = run_dir / "job.log"
            run_dir_rel = run_dir.relative_to(self.runs_root).as_posix()
            cli_rel = (run_dir / "cli_output").relative_to(self.runs_root).as_posix()
            log_rel = log_path.relative_to(self.runs_root).as_posix()

            input_row = conn.execute(
                "SELECT object_relpath FROM rec_inputs WHERE input_id = ?", (job["input_id"],)
            ).fetchone()
            input_path = store.ensure_within(self.store_root,self.store_root / input_row["object_relpath"])

            conn.execute(
                "UPDATE rec_jobs SET run_dir_relpath=?, cli_output_relpath=?, log_relpath=?"
                " WHERE job_id = ?",
                (run_dir_rel, cli_rel, log_rel, job_id),
            )
            conn.commit()

            result = self.runner.run(input_path, job["input_sha256"], cli_out, log_path)
            run_seq = conn.execute(
                "SELECT COUNT(*) AS n FROM rec_runs WHERE job_id = ?", (job_id,)
            ).fetchone()["n"] + 1
            finished = db.now()

            if getattr(result, "success", False):
                self._write_success(conn, job_id, run_seq, result, cli_rel, started, finished)
            else:
                self._write_failure(conn, job_id, run_seq, result, cli_rel, started, finished)
        finally:
            conn.close()

    def _write_success(self, conn, job_id, run_seq, result, cli_rel, started, finished) -> None:
        counts = result.class_counts or {}
        run_json_rel = (Path(cli_rel) / "run.json").as_posix()
        with conn:  # 单一事务：区域 + 类计数 + run.json 事实 + 作业状态
            cur = conn.execute(
                "INSERT INTO rec_runs(job_id, run_seq, status, exit_code, adapter,"
                " versions_json, params_json, input_sha256, model_sha256, elapsed_seconds,"
                " field_count, output_relpath, run_json_relpath, error, started_at, finished_at)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    job_id, run_seq, "succeeded", result.exit_code, result.adapter,
                    json.dumps(result.versions, ensure_ascii=False),
                    json.dumps(result.params, ensure_ascii=False),
                    result.input_sha256, result.model_sha256, result.elapsed_seconds,
                    result.field_count, cli_rel, run_json_rel, None, started, finished,
                ),
            )
            run_id = cur.lastrowid
            for name in cfg.FIXED_ARTIFACTS:
                artifact = store.ensure_within(self.runs_root,self.runs_root/cli_rel/name)
                conn.execute('INSERT INTO rec_artifacts(job_id,run_id,name,sha256,size_bytes) VALUES (?,?,?,?,?)',
                             (job_id,run_id,name,store.sha256_file(artifact),artifact.stat().st_size))
            for region in result.regions or []:
                conn.execute(
                    "INSERT INTO rec_regions(job_id, run_id, field_id, pixels, min_row,"
                    " min_col, max_row, max_col, area_unit) VALUES (?,?,?,?,?,?,?,?,?)",
                    (
                        job_id, run_id, region["field_id"], region["pixels"],
                        region["min_row"], region["min_col"], region["max_row"],
                        region["max_col"], "pixel",
                    ),
                )
            for class_value, key in db.CLASS_PIXEL_KEY.items():
                conn.execute(
                    "INSERT INTO rec_class_counts(job_id, run_id, class_value, meaning, pixels)"
                    " VALUES (?,?,?,?,?)",
                    (job_id, run_id, class_value, db.CLASS_MEANINGS[class_value],
                     int(counts.get(key, 0))),
                )
            conn.execute(
                "UPDATE rec_jobs SET status='succeeded', error=NULL, finished_at=?,"
                " model_sha256=? WHERE job_id = ?",
                (finished, result.model_sha256, job_id),
            )

    def _write_failure(self, conn, job_id, run_seq, result, cli_rel, started, finished) -> None:
        err = getattr(result, "error", None) or "未知失败"
        with conn:
            conn.execute(
                "INSERT INTO rec_runs(job_id, run_seq, status, exit_code, input_sha256,"
                " model_sha256, elapsed_seconds, output_relpath, error, started_at, finished_at)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    job_id, run_seq, "failed", getattr(result, "exit_code", None),
                    getattr(result, "input_sha256", None), getattr(result, "model_sha256", None),
                    getattr(result, "elapsed_seconds", None), cli_rel, err, started, finished,
                ),
            )
            conn.execute(
                "UPDATE rec_jobs SET status='failed', error=?, finished_at=? WHERE job_id = ?",
                (err, finished, job_id),
            )

    def _record_unexpected(self, job_id, exc) -> None:
        conn = db.connect(self.db_path)
        try:
            with conn:
                conn.execute(
                    "UPDATE rec_jobs SET status='failed', error=?, finished_at=?"
                    " WHERE job_id = ? AND status IN ('queued','running')",
                    (f"worker 异常：{type(exc).__name__}: {exc}", db.now(), job_id),
                )
        except Exception:  # noqa: BLE001 - best effort
            pass
        finally:
            conn.close()

    # ---- queries ----------------------------------------------------------

    def _job_row(self, job_id):
        conn = db.connect(self.db_path)
        try:
            return conn.execute("SELECT * FROM rec_jobs WHERE job_id = ?", (job_id,)).fetchone()
        finally:
            conn.close()

    def get_job(self, job_id: int) -> dict | None:
        conn = db.connect(self.db_path)
        try:
            conn.execute('BEGIN')  # 各张结果表与状态取自同一个读快照
            job = conn.execute("SELECT * FROM rec_jobs WHERE job_id = ?", (job_id,)).fetchone()
            if job is None:
                return None
            run = conn.execute(
                "SELECT * FROM rec_runs WHERE job_id = ? ORDER BY run_seq DESC LIMIT 1",
                (job_id,),
            ).fetchone()
            counts = conn.execute(
                "SELECT class_value, meaning, pixels FROM rec_class_counts WHERE job_id = ?"
                " ORDER BY class_value",
                (job_id,),
            ).fetchall()
            region_count = conn.execute(
                "SELECT COUNT(*) AS n FROM rec_regions WHERE job_id = ?", (job_id,)
            ).fetchone()["n"]
            result = self._job_to_dict(job, run, counts, region_count)
            result['artifact_checksums'] = {r['name']:{'sha256':r['sha256'],'size_bytes':r['size_bytes']}
                for r in conn.execute('SELECT name,sha256,size_bytes FROM rec_artifacts WHERE job_id=?',(job_id,))}
            return result
        finally:
            conn.close()

    def list_jobs(self, limit: int = 100) -> list:
        conn = db.connect(self.db_path)
        try:
            rows = conn.execute(
                "SELECT * FROM rec_jobs ORDER BY job_id DESC LIMIT ?", (int(limit),)
            ).fetchall()
            return [self._job_to_dict(r, None, None, None) for r in rows]
        finally:
            conn.close()

    def get_fields(self, job_id: int) -> list | None:
        conn = db.connect(self.db_path)
        try:
            job = conn.execute(
                "SELECT job_id FROM rec_jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
            if job is None:
                return None
            rows = conn.execute(
                "SELECT field_id, pixels, min_row, min_col, max_row, max_col, area_unit"
                " FROM rec_regions WHERE job_id = ? ORDER BY field_id",
                (job_id,),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def artifact_path(self, job_id: int, name: str) -> Path | None:
        """返回可下载产物的绝对路径；仅成功作业、仅固定文件名，且必须落在 runs 根内。"""
        if name not in cfg.FIXED_ARTIFACTS:
            return None
        conn = db.connect(self.db_path)
        try:
            job = conn.execute(
                "SELECT status, cli_output_relpath FROM rec_jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
        finally:
            conn.close()
        if job is None or job["status"] != "succeeded" or not job["cli_output_relpath"]:
            return None
        base = self.runs_root / job["cli_output_relpath"]
        try:
            target = store.ensure_within(self.runs_root, base / name)
        except store.PathBoundaryError:
            return None
        return target if target.is_file() else None

    def read_artifact(self, job_id: int, name: str) -> bytes | None:
        path = self.artifact_path(job_id,name)
        if path is None:
            return None
        conn = db.connect(self.db_path)
        try:
            row = conn.execute('SELECT sha256,size_bytes FROM rec_artifacts WHERE job_id=? AND name=?',
                               (job_id,name)).fetchone()
        finally:
            conn.close()
        data = path.read_bytes()
        if row is None or len(data)!=row['size_bytes'] or store.sha256_bytes(data)!=row['sha256']:
            raise ArtifactIntegrityError('识别产物已变化或缺少校验记录，请保留文件并重新运行新的识别作业')
        return data

    @staticmethod
    def _job_to_dict(job, run, counts, region_count) -> dict:
        out = {
            "job_id": job["job_id"],
            "status": job["status"],
            "input_sha256": job["input_sha256"],
            "original_filename": job["original_filename"],
            "source": job["source"],
            "source_link": job["source_link"],
            "license_status": job["license_status"],
            "scene_id": job["scene_id"],
            "sample_kind": job["sample_kind"],
            "request_id": job["request_id"],
            "error": job["error"],
            "attempt_count": job["attempt_count"],
            "created_at": job["created_at"],
            "started_at": job["started_at"],
            "finished_at": job["finished_at"],
            "model_sha256": job["model_sha256"],
            "artifacts": list(cfg.FIXED_ARTIFACTS) if job['status'] == 'succeeded' else [],
        }
        if run is not None:
            out["run"] = {
                "run_seq": run["run_seq"],
                "status": run["status"],
                "exit_code": run["exit_code"],
                "adapter": run["adapter"],
                "versions": json.loads(run["versions_json"]) if run["versions_json"] else None,
                "params": json.loads(run["params_json"]) if run["params_json"] else None,
                "elapsed_seconds": run["elapsed_seconds"],
                "field_count": run["field_count"],
            }
        if counts is not None:
            out["class_counts"] = {str(c["class_value"]): c["pixels"] for c in counts}
            out["class_meaning"] = {str(c["class_value"]): c["meaning"] for c in counts}
        if region_count is not None:
            out["region_count"] = region_count
        return out


class IdempotencyConflict(Exception):
    """同 request_id 但内容不同。"""


class ArtifactIntegrityError(Exception):
    """下载内容与作业完成时的摘要不一致。"""


class JobNotFound(Exception):
    """未知作业。"""
