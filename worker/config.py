"""Worker configuration, all from the environment (plus an optional worker/.env).

    REDIS_URL             redis://:<password>@127.0.0.1:6379/0 - must match the web-app's
    STREAM_PREFIX         video                                - must match the web-app's
    SWAP_CASES_DIR        project folders (default <repo>/cases)       - shared with the web-app
    SWAP_CHARS_DIR        character library (default <repo>/characters) - shared with the web-app
    WORKER_NAME           unique per process (default swap-<role>-<pid>)
    WORKER_CLAIM_IDLE_MS  a job read but unacked for longer is taken over (render 4 h, util 2 min)
    SWAP_RUN_CMD          the pipeline command (default "bash scripts/run_swap.sh"; tests use a fake)
    PLANNER_GPU           GPU of the util worker (edit planner, insightface), nvidia-smi numbering (default 0)
"""
from __future__ import annotations

import os
import shlex
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

ROLES = ("render", "util")


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    try:
        return int(raw) if raw else default
    except ValueError:
        return default


def _env_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    return Path(raw).expanduser().resolve() if raw else default


@dataclass(slots=True)
class Config:
    role: str = "render"
    redis_url: str = field(default_factory=lambda: os.environ.get("REDIS_URL") or "redis://127.0.0.1:6379/0")
    stream_prefix: str = field(default_factory=lambda: os.environ.get("STREAM_PREFIX") or "video")
    cases_dir: Path = field(default_factory=lambda: _env_path("SWAP_CASES_DIR", REPO_ROOT / "cases"))
    chars_dir: Path = field(default_factory=lambda: _env_path("SWAP_CHARS_DIR", REPO_ROOT / "characters"))
    worker_name: str = ""
    #: 0 = role default: render jobs run for up to an hour and must not be taken over while still running.
    claim_idle_ms: int = field(default_factory=lambda: _env_int("WORKER_CLAIM_IDLE_MS", 0))
    #: One attempt: a failed swap is reported, never silently re-run for another hour.
    max_attempts: int = 1
    #: How long to block on XREADGROUP before looping (also the shutdown granularity).
    block_ms: int = field(default_factory=lambda: _env_int("WORKER_BLOCK_MS", 5000))
    run_cmd: list[str] = field(
        default_factory=lambda: shlex.split(os.environ.get("SWAP_RUN_CMD") or "bash scripts/run_swap.sh"))
    planner_gpu: str = field(default_factory=lambda: os.environ.get("PLANNER_GPU") or "0")
    #: Seconds between checks of the cancel flag while the pipeline runs.
    cancel_poll_seconds: float = 5.0
    #: Minimum seconds between two progress events of one job.
    progress_interval: float = 1.0
    #: Seconds between capability advertisements (the document's TTL is 120 s).
    advertise_seconds: float = 30.0

    def __post_init__(self) -> None:
        if self.role not in ROLES:
            raise ValueError(f"unknown role {self.role!r}; expected one of: {', '.join(ROLES)}")
        if not self.worker_name:
            self.worker_name = os.environ.get("WORKER_NAME") or f"swap-{self.role}-{os.getpid()}"
        if not self.claim_idle_ms:
            self.claim_idle_ms = 4 * 3600 * 1000 if self.role == "render" else 120_000


def load(role: str) -> Config:
    """Config for one role, reading worker/.env first if present (existing variables win)."""
    _load_dotenv(REPO_ROOT / "worker" / ".env")
    return Config(role=role)


def _load_dotenv(path: Path) -> None:
    """Minimal .env support so the worker needs no extra dependency."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))
