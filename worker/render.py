"""Render role: runs scripts/run_swap.sh for a whole project (shot_id "all") or one shot (shot_id "<n>")."""
from __future__ import annotations

import os
import queue
import signal
import subprocess
import threading
import time

from contracts.python.messages import ErrorCode, JobRequest, Output, Stage
from utils import swap_project as sp

from .config import REPO_ROOT, Config
from .loop import JobContext, JobFailed

#: How many log lines a failure message quotes.
_TAIL_LINES = 20


def popen_group_kwargs() -> dict:
    """Start the pipeline in its own process group, so a cancel stops it with everything it started."""
    if os.name == "nt":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


def kill_tree(proc: subprocess.Popen, timeout: float = 15) -> None:
    """Stop a process started with popen_group_kwargs and everything it spawned."""
    if proc.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True)
    else:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        if os.name != "nt":
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        proc.kill()
        proc.wait()


def failure_tail(text: str, lines: int = _TAIL_LINES) -> str:
    return "\n".join(sp.clean_log_lines(text, lines))


class RenderHandler:
    def __init__(self, config: Config):
        self.config = config
        self.last_proc: subprocess.Popen | None = None      # for tests and diagnostics

    def handle(self, job: JobRequest, ctx: JobContext) -> Output:
        proj = job.project_id
        d = sp.project_dir(proj)
        if d is None or not (d / "project.json").exists():
            raise JobFailed(ErrorCode.SHOT_NOT_FOUND, f"Project '{proj}' not found.")
        m = sp.meta(proj)
        char = m.get("char") or ""
        if not char or not (sp.CHARS / char / "cutout.png").exists():
            raise JobFailed(ErrorCode.SHOT_NOT_FOUND, f"Character '{char}' not found.")
        if not m.get("details"):
            dt = sp.CHARS / char / "details.txt"
            m["details"] = dt.read_text(encoding="utf-8").strip() if dt.exists() else "Female"

        if job.shot_id == "all":
            only = ""
            ctx.report(Stage.RESOLVING, 1, "Preparing the project", force=True)
            sp.prepare_case(proj)
        else:
            if not job.shot_id.isdigit():
                raise JobFailed(ErrorCode.INVALID_SHOT, f"'{job.shot_id}' is not a shot number.")
            n = int(job.shot_id)
            if not sp.is_swapped(proj, n):
                raise JobFailed(ErrorCode.INVALID_SHOT, f"Shot {n} is kept from the original video (nobody to swap), "
                                                        "so there is nothing to re-render.")
            ov = job.overrides or {}
            sp.apply_shot_edit(proj, n, note=ov.get("note"), identity=ov.get("identity") or "same",
                               change_light=bool(ov.get("change_light")), light=ov.get("relight_strength", 1.0),
                               clothes=ov.get("clothes") or "unchanged", new_take=bool(ov.get("new_take")))
            ctx.report(Stage.RESOLVING, 1, f"Re-rendering shot {n}", force=True)
            if not (d / "case.env").exists():
                sp.prepare_case(proj)
            only = str(n)

        env = sp.pipeline_env(proj, m, only)
        text, rc = self._run(proj, env, only, ctx)
        if rc != 0:
            raise JobFailed(ErrorCode.INTERNAL_ERROR, f"The pipeline failed (exit {rc}). Last lines:\n"
                                                      + failure_tail(text))
        final = d / "output" / "final" / "swapped.mp4"
        if not final.exists():
            raise JobFailed(ErrorCode.INTERNAL_ERROR, "The pipeline finished without a final video. Last lines:\n"
                                                      + failure_tail(text))
        ctx.report(Stage.FINALIZING, 99, "Cutting the shots", force=True)
        man = sp.build_manifest(proj, job.job_id)
        w, h = man.get("width"), man.get("height")
        return Output(filename="swapped.mp4", path=f"{proj}/output/final/swapped.mp4", bytes=final.stat().st_size,
                      duration_seconds=man.get("duration_seconds"), resolution=f"{w}x{h}" if w and h else None,
                      extra={"manifest": f"{proj}/manifest.json", "shots": len(man["shots"])})

    def _run(self, proj: str, env: dict, only: str, ctx: JobContext) -> tuple[str, int]:
        """Run the pipeline, teeing its output to output/run.log and turning it into progress events; stops it when
        the job is cancelled. Returns (this run's log text, exit code)."""
        fh = sp.open_run_log(proj, only)
        parts: list[str] = [f"===== ONLY={only or 'all'}\n"]
        try:
            proc = subprocess.Popen(self.config.run_cmd, cwd=REPO_ROOT, env=env, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                                    bufsize=1, **popen_group_kwargs())
        except OSError as error:
            fh.close()
            raise JobFailed(ErrorCode.INTERNAL_ERROR, f"Cannot start the pipeline: {error}") from error
        self.last_proc = proc
        lines: queue.Queue[str | None] = queue.Queue()

        def pump() -> None:
            for line in proc.stdout:                  # type: ignore[union-attr]
                lines.put(line)
            lines.put(None)

        reader = threading.Thread(target=pump, name="pipeline-output", daemon=True)
        reader.start()
        next_cancel_check = 0.0
        eof = False
        try:
            while not eof:
                try:
                    item = lines.get(timeout=0.5)
                    while True:                       # drain what is there
                        if item is None:
                            eof = True
                            break
                        fh.write(item); parts.append(item)
                        item = lines.get_nowait()
                except queue.Empty:
                    pass
                fh.flush()
                stage, pct, msg = sp.parse_progress("".join(parts), only)
                ctx.report(Stage(stage), pct, msg)
                if time.monotonic() >= next_cancel_check:
                    next_cancel_check = time.monotonic() + self.config.cancel_poll_seconds
                    if not eof and ctx.cancelled():
                        kill_tree(proc)
                        sp.stop_comfy_jobs(proj)
                        reader.join(timeout=5)
                        fh.write(f"[cancelled] {time.strftime('%H:%M:%S')} stopped from the web-app\n")
                        raise JobFailed(ErrorCode.CANCELLED, "Cancelled.")
            rc = proc.wait()
        finally:
            if proc.poll() is None:
                kill_tree(proc)
            fh.close()
        return "".join(parts), rc
