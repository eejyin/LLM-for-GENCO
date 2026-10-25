"""Persistent local subprocess jobs. No shell commands are accepted from the LLM."""
import json
import os
import signal
import subprocess
import threading
import time
import uuid
from pathlib import Path

class JobManager:
    def __init__(self, root: Path):
        self.root = root
        self.jobs = {}
        self.processes = {}
        self.lock = threading.RLock()
        for path in root.glob("*/job.json"):
            try:
                job = json.loads(path.read_text())
                if job["status"] in ("queued", "running"):
                    job["status"] = "interrupted"
                    job["error"] = "Server restarted; previous process state is unknown. Check the recorded PID before retrying."
                self.jobs[job["id"]] = job
            except (OSError, ValueError, KeyError):
                continue

    def persist(self, job):
        target = self.root / job["id"] / "job.json"
        temp = target.with_suffix(".tmp")
        temp.write_text(json.dumps(job, ensure_ascii=False, indent=2))
        temp.replace(target)

    def create(self, kind, metadata=None):
        with self.lock:
            job_id = uuid.uuid4().hex[:12]
            folder = self.root / job_id
            folder.mkdir(parents=True)
            job = {"id": job_id, "kind": kind, "status": "queued", "created_at": time.time(), **(metadata or {})}
            self.jobs[job_id] = job
            self.persist(job)
            return job

    def launch(self, job, command, env):
        def work():
            folder = self.root / job["id"]
            with (folder / "run.log").open("w") as log:
                try:
                    with self.lock:
                        if job["status"] == "cancelled":
                            return
                        proc = subprocess.Popen(command, cwd=folder, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                        self.processes[job["id"]] = proc
                        job.update(status="running", pid=proc.pid, command=command)
                        self.persist(job)
                    code = proc.wait()
                    with self.lock:
                        if job["status"] != "cancelled":
                            job["status"] = "completed" if code == 0 else "failed"
                        job["exit_code"] = code
                except Exception as exc:
                    with self.lock:
                        job.update(status="failed", error=str(exc))
                    log.write(str(exc))
                finally:
                    with self.lock:
                        job["finished_at"] = time.time()
                        self.processes.pop(job["id"], None)
                        self.persist(job)
        threading.Thread(target=work, daemon=True).start()
        return dict(job)

    def get(self, job_id):
        with self.lock:
            if job_id not in self.jobs:
                raise KeyError(job_id)
            job = dict(self.jobs[job_id])
        path = self.root / job_id / "run.log"
        if path.exists():
            with path.open("rb") as handle:
                handle.seek(max(0, path.stat().st_size - 12000))
                job["log_tail"] = handle.read().decode(errors="replace")
        else:
            job["log_tail"] = ""
        job["artifacts"] = [str(p.relative_to(self.root / job_id)) for p in (self.root / job_id).rglob("*")
                            if p.is_file() and not p.is_symlink() and p.suffix in (".csv", ".json", ".yaml", ".pt", ".ckpt", ".parquet")][:80]
        return job

    def cancel(self, job_id):
        with self.lock:
            job = self.jobs[job_id]
            if job["status"] not in ("queued", "running"):
                return dict(job)
            job["status"] = "cancelled"
            proc = self.processes.get(job_id)
            if proc and proc.poll() is None:
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                def kill_later():
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        try: os.killpg(proc.pid, signal.SIGKILL)
                        except ProcessLookupError: pass
                threading.Thread(target=kill_later, daemon=True).start()
            self.persist(job)
            return dict(job)
