"""Util role: short jobs that must not wait behind a render (they run on the planner GPU, PLANNER_GPU).

  action plan_edit       project demoN, shot <n>, overrides.request: the edit planner (Qwen3-VL, utils/edit_planner)
                         turns a free-form request into the shot-edit settings
  action add_character   project "characters", shot <name>: <CHARS>/<name>/_upload/photo.* (+ face.*) becomes
                         cutout.png + face.png (utils.swap_project.process_character)
"""
from __future__ import annotations

import json
import shutil
from collections.abc import Callable

from contracts.python.messages import ErrorCode, JobRequest, Output, Stage
from utils import swap_project as sp

from .config import Config
from .loop import JobContext, JobFailed

CHARACTERS_PROJECT = "characters"


def _load_planner() -> Callable[[str, list[str], dict], dict]:
    """utils.edit_planner.plan; the module keeps the model loaded, so it loads once per worker process."""
    from utils.edit_planner import plan
    return plan


class UtilHandler:
    def __init__(self, config: Config, planner: Callable[[str, list[str], dict], dict] | None = None,
                 process_character: Callable | None = None):
        self.config = config
        self.planner = planner
        self.process_character = process_character or sp.process_character

    def handle(self, job: JobRequest, ctx: JobContext) -> Output:
        action = (job.overrides or {}).get("action")
        if action == "plan_edit":
            return self._plan_edit(job, ctx)
        if action == "add_character":
            return self._add_character(job, ctx)
        raise JobFailed(ErrorCode.INVALID_SHOT, f"Unknown action {action!r}; expected plan_edit or add_character.")

    def _plan_edit(self, job: JobRequest, ctx: JobContext) -> Output:
        proj = job.project_id
        d = sp.project_dir(proj)
        if d is None or not (d / "project.json").exists():
            raise JobFailed(ErrorCode.SHOT_NOT_FOUND, f"Project '{proj}' not found.")
        if not job.shot_id.isdigit() or not sp.is_swapped(proj, int(job.shot_id)):
            raise JobFailed(ErrorCode.INVALID_SHOT, f"Shot {job.shot_id} is kept from the original video (nobody to "
                                                    "swap), so it cannot be edited.")
        n = int(job.shot_id)
        request = str((job.overrides or {}).get("request") or "").strip()
        if not request:
            raise JobFailed(ErrorCode.INVALID_SHOT, "Describe the change first.")
        ctx.report(Stage.RESOLVING, 10, "Reading the shot", force=True)
        frames, info = sp.plan_inputs(proj, n)
        ctx.report(Stage.RENDERING, 40, "Planning the edit", force=True)
        if self.planner is None:
            self.planner = _load_planner()
        p = self.planner(request, frames, info)
        fields = sp.plan_to_fields(p)
        out = d / "output" / "gen" / f"plan_{n:02d}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"job_id": job.job_id, "request": request, "plan": p, "fields": fields,
                                   "created": sp.utcnow()}, indent=1, ensure_ascii=False), encoding="utf-8")
        return Output(filename=out.name, path=f"{proj}/output/gen/{out.name}",
                      extra={"possible": bool(p.get("possible", True)), "explanation": str(p.get("explanation") or ""),
                             "plan": p, "fields": fields})

    def _add_character(self, job: JobRequest, ctx: JobContext) -> Output:
        if job.project_id != CHARACTERS_PROJECT:
            raise JobFailed(ErrorCode.INVALID_SHOT, f"add_character needs project '{CHARACTERS_PROJECT}'.")
        name = sp.normalize_name(job.shot_id)
        if not name or name != job.shot_id:
            raise JobFailed(ErrorCode.INVALID_SHOT, f"Invalid character name '{job.shot_id}'.")
        d = sp.CHARS / name
        up = d / "_upload"
        photo = next(iter(sorted(up.glob("photo.*"))), None) if up.is_dir() else None
        if photo is None:
            raise JobFailed(ErrorCode.SHOT_NOT_FOUND, f"No uploaded photo for character '{name}'.")
        face = next(iter(sorted(up.glob("face.*"))), None)
        ctx.report(Stage.RENDERING, 30, "Cutting out the character", force=True)
        try:
            self.process_character(name, photo, face, None)
        except sp.NoPersonFound:
            shutil.rmtree(d, ignore_errors=True)      # lets the user try again with the same name
            raise JobFailed(ErrorCode.INVALID_SHOT, "No person found in that photo.") from None
        except ValueError as error:
            shutil.rmtree(d, ignore_errors=True)
            raise JobFailed(ErrorCode.INVALID_SHOT, str(error)) from None
        shutil.rmtree(up, ignore_errors=True)
        return Output(filename="cutout.png", path=f"{name}/cutout.png", extra={"character": name})
