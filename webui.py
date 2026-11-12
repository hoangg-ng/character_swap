#!/usr/bin/env python3
"""Swap character - web UI for the pipeline (Gradio).

Workflow: upload a source video, say who to replace, pick a character -> "Swap character" runs the standard
pipeline (scripts/run_swap.sh) in a new project (Demo 1, Demo 2, ... = cases/demoN). The result is the final video
and the list of its shots; any swapped shot can be re-rendered with an extra instruction ("edit this shot"), after
which the final video is rebuilt.

Characters live in characters/<name>/ (cutout.png, face.png|jpg, details.txt); new ones can be added from photos.
Start: bash scripts/start_ui.sh

The project logic lives in utils/swap_project.py and is shared with the web-app worker (worker/): this Gradio UI keeps
working alongside the web-app, and both see the same projects (SWAP_CASES_DIR) and characters (SWAP_CHARS_DIR).
"""
from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import time
from pathlib import Path

import gradio as gr

from utils import swap_project as sp

ROOT = Path(__file__).resolve().parent
CASES = sp.CASES
CHARS = sp.CHARS
RUNS: dict[str, subprocess.Popen] = {}          # project -> running pipeline
SEEN: dict[str, float] = {}                     # project -> final video mtime shown

CSS = """
:root, .dark { --accent: #3b82f6; --bg: #121212; --panel: #1c1c1e; --panel2: #232326; --line: #2e2e32;
  --text: #f2f2f2; --muted: #9a9aa2; }
body, .gradio-container { background: var(--bg) !important; color: var(--text); }
.gradio-container { max-width: 1440px !important; padding: 0 20px !important; }
#topbar { display: flex; align-items: center; gap: 12px; padding: 14px 4px 10px; border-bottom: 1px solid var(--line);
  margin-bottom: 14px; }
#topbar .logo { width: 30px; height: 30px; border-radius: 8px; background: linear-gradient(135deg, #3b82f6, #a855f7); }
#topbar .name { font-size: 18px; font-weight: 600; letter-spacing: .2px; }
#topbar .sub { color: var(--muted); font-size: 13px; margin-left: 6px; }
.panel { background: var(--panel) !important; border: 1px solid var(--line) !important; border-radius: 12px !important;
  padding: 14px !important; }
.section-title { font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: .08em; color: var(--muted);
  margin: 4px 0 6px; }
#swap-btn { background: var(--accent) !important; color: white !important; font-weight: 600; font-size: 15px;
  border-radius: 10px !important; min-height: 46px; }
#status { background: var(--panel2); border: 1px solid var(--line); border-radius: 10px; padding: 10px 14px; }
#status .stage { font-weight: 600; } #status .meta { color: var(--muted); font-size: 13px; }
.bar { height: 6px; background: #2c2c30; border-radius: 4px; margin-top: 8px; overflow: hidden; }
.bar > div { height: 100%; background: var(--accent); transition: width .5s; }
footer { display: none !important; }
"""
STAGES = sp.STAGES
IDENT, KEEP = sp.IDENT, sp.KEEP


# ---------- projects ----------
_num = sp.num
projects, next_project, meta = sp.projects, sp.next_project, sp.meta


def first_project():
    ps = projects()
    return ps[0][1] if ps else None


# ---------- characters ----------
characters, char_face = sp.characters, sp.char_face


def char_gallery():
    return [(str(char_face(n) or sp.CHARS / n / "cutout.png"), n) for n in characters()]


def add_character(name, photo, face_photo, details):
    try:
        d = sp.process_character(name, photo, face_photo, details or "")
    except ValueError as e:                    # NoPersonFound, or no name / photo
        return str(e), gr.update()
    return f"Added **{d.name}**.", gr.update(value=char_gallery())


# ---------- running ----------
pipeline_env = sp.pipeline_env


def launch(proj: str, env: dict) -> None:
    fh = sp.open_run_log(proj, env.get("ONLY") or "")
    RUNS[proj] = subprocess.Popen(["bash", "scripts/run_swap.sh"], cwd=ROOT, env=env, stdout=fh,
                                  stderr=subprocess.STDOUT, start_new_session=True)


def start_swap(video, who, sel, keep, quality, relight, protect):
    if not video:
        return "Upload a source video.", gr.update()
    if not sel:
        return "Pick a character.", gr.update()
    proj = next_project(); d = sp.CASES / proj
    (d / "input").mkdir(parents=True); (d / "output").mkdir()
    ext = Path(video).suffix or ".mp4"; shutil.copy2(video, d / "input" / f"video{ext}")
    who = (who or "the woman").strip().replace('"', "'")
    m = dict(char=sel, details=(sp.CHARS / sel / "details.txt").read_text().strip(), keep=bool(keep),
             q="720" if quality.startswith("720") else "480", relight=bool(relight), protect=(protect or "").strip(),
             who=who, created=time.strftime("%Y-%m-%d %H:%M"))
    (d / "project.json").write_text(json.dumps(m, indent=1))
    sp.prepare_case(proj)                      # character files -> input/, case.env
    launch(proj, pipeline_env(proj, m))
    return f"Started **Demo {_num(d)}**.", gr.update(choices=projects(), value=proj)


shot_edits = sp.shot_edits


def plan_edit(proj, shot, request):
    """Free-form request -> editable plan (explanation, note, identity, light, clothes, new take)."""
    nothing = ("", "", "same", False, 1.0, "unchanged", True)
    if not (proj and shot):
        return ("Select a shot first.",) + nothing[1:]
    if not (request or "").strip():
        return ("Describe the change first.",) + nothing[1:]
    if not sp.is_swapped(proj, shot):
        return (f"Shot {shot} is kept from the original video (nobody to swap), so it cannot be edited.",) + nothing[1:]
    from utils.edit_planner import plan
    frames, info = sp.plan_inputs(proj, shot)
    p = plan(request.strip(), frames, info)
    if not p.get("possible", True):
        return (f"**Not possible:** {p.get('explanation', '')}",) + nothing[1:]
    f = sp.plan_to_fields(p)
    keep = {"unchanged": KEEP[0], "video": KEEP[1], "own": KEEP[2]}[f["clothes"]]
    return (f"**Plan:** {p.get('explanation', '')}  \nCheck the settings below, adjust if needed, then re-render.",
            f["note"], f["identity"], f["change_light"], f["relight_strength"], keep, f["new_take"])


def rerender_shot(proj, shot, note, identity, change_light, light, keep, new_take):
    if not (proj and shot):
        return "Select a shot first."
    if proj in RUNS and RUNS[proj].poll() is None:
        return "This project is still running."
    if not sp.is_swapped(proj, shot):
        return f"Shot {shot} is kept from the original video (nobody to swap), so there is nothing to re-render."
    e = sp.apply_shot_edit(proj, shot, note, identity, change_light, light, keep, new_take)
    launch(proj, pipeline_env(proj, meta(proj), only=str(shot)))
    return f"Re-rendering shot {shot} with {json.dumps(e, ensure_ascii=False)}. The final video is rebuilt afterwards."


def stop(proj):
    p = RUNS.get(proj)
    if not p or p.poll() is not None:
        return "Nothing is running."
    os.killpg(os.getpgid(p.pid), signal.SIGTERM)
    sp.stop_comfy_jobs(proj)                   # remove this project's jobs from the GPU queues
    return "Stopped."


# ---------- status & results ----------
last_run_log = sp.last_run_log


def status_html(proj: str) -> str:
    if not proj:
        return "<div id='status'><span class='meta'>No project yet. Upload a video and press Swap character.</span></div>"
    txt = last_run_log(proj)
    running = proj in RUNS and RUNS[proj].poll() is None
    label, k = sp.stage_of(txt)
    stage = label or ("Starting" if running else "Ready")
    if stage == "Generating shots":
        done, started = len(re.findall(r"shot \d+: done", txt)), len(re.findall(r"generating on", txt))
        stage += f" ({done}/{max(1, started)})"
    if not running and txt and "[done]" not in txt:
        stage = "Stopped"
    pct = int(100 * k / len(STAGES))
    m = meta(proj)
    info = (f"{m.get('char', '?')} · replaces “{m.get('who', '')}” · {m.get('q', '480')}p"
            + (" · keeps the video's clothes" if m.get("keep") else "") + (" · running" if running else ""))
    return (f"<div id='status'><div class='stage'>{stage}</div><div class='meta'>{info}</div>"
            f"<div class='bar'><div style='width:{pct}%'></div></div></div>")


def shots(proj: str):
    """([(thumbnail, caption)], [shot numbers]) cut from the final video at the pipeline's shot boundaries."""
    items, nums = [], []
    for s in (sp.cut_shots(proj) if proj else []):
        if s["thumb"] is None:
            continue
        items.append((str(s["thumb"]), f"Shot {s['n']} · {s['start']:.1f}–{s['end']:.1f}s · "
                                       f"{'swapped' if s['swapped'] else 'original'}"))
        nums.append(s["n"])
    return items, nums


def refresh(proj):
    final = sp.CASES / proj / "output" / "final" / "swapped.mp4" if proj else None
    if final and final.exists():
        SEEN[proj] = final.stat().st_mtime
    items, nums = shots(proj) if proj else ([], [])
    return status_html(proj), (str(final) if final and final.exists() else None), items, nums


def tick(proj):
    log = "\n".join(sp.clean_log_lines(last_run_log(proj))) if proj else ""
    final = sp.CASES / proj / "output" / "final" / "swapped.mp4" if proj else None
    if final and final.exists() and final.stat().st_mtime > SEEN.get(proj, 0) and not (proj in RUNS and RUNS[proj].poll() is None):
        s, f, items, nums = refresh(proj)                  # a run just finished: show the new final and shots
        return s, log, f, items, nums
    return status_html(proj), log, gr.update(), gr.update(), gr.update()


def pick_shot(evt: gr.SelectData, proj, nums):
    i = nums[evt.index] if nums and evt.index < len(nums) else None
    if not i:
        return None, None, ""
    clip = sp.CASES / proj / "output" / "gen" / "_ui_shots" / f"s{i:02d}.mp4"
    e = shot_edits(proj).get(str(i), {})
    note = e.get("note", "") if isinstance(e, dict) else str(e)
    return (str(clip) if clip.exists() else None), i, note


# ---------- layout ----------
def build() -> gr.Blocks:
    theme = gr.themes.Base(primary_hue="blue", neutral_hue="zinc", font=[gr.themes.GoogleFont("Inter"), "system-ui"])
    with gr.Blocks(title="Swap character", theme=theme, css=CSS,
                   js="() => { document.body.classList.add('dark'); }") as ui:
        gr.HTML("<div id='topbar'><div class='logo'></div><div class='name'>Swap character</div>"
                "<div class='sub'>replace a person in a video with your character</div></div>")
        sel_char, shot_nums, cur_shot = gr.State(None), gr.State([]), gr.State(None)
        with gr.Row(equal_height=False):
            with gr.Column(scale=4, elem_classes="panel"):
                gr.HTML("<div class='section-title'>1 · Source video</div>")
                video = gr.Video(sources=["upload"], show_label=False, height=240)
                gr.HTML("<div class='section-title'>2 · Who to replace</div>")
                who = gr.Textbox(show_label=False, placeholder="e.g. the woman in the red dress", lines=1)
                gr.HTML("<div class='section-title'>3 · Character</div>")
                gal = gr.Gallery(value=char_gallery(), show_label=False, columns=4, height=160, allow_preview=False,
                                 object_fit="cover")
                chosen = gr.Markdown("*Pick a character above.*")
                with gr.Accordion("Add a character", open=False):
                    cn = gr.Textbox(label="Name")
                    with gr.Row():
                        cp = gr.Image(type="filepath", label="Full-body photo", height=160)
                        cf = gr.Image(type="filepath", label="Face photo (optional)", height=160)
                    cd = gr.Textbox(label="Description (head to toe)", lines=2,
                                    placeholder="Female, slim, long black hair, ... wearing ...")
                    cb = gr.Button("Add character"); cm = gr.Markdown()
                with gr.Accordion("Options", open=False):
                    keep = gr.Checkbox(label="Keep the video's clothes", value=False)
                    quality = gr.Radio(["480p · quick test", "720p · final"], value="480p · quick test", label="Quality")
                    relight = gr.Checkbox(label="Re-light to the scene", value=True)
                    protect = gr.Textbox(label="People to keep unchanged (optional)", placeholder="e.g. the man")
                go = gr.Button("Swap character", elem_id="swap-btn")
                msg = gr.Markdown()
            with gr.Column(scale=7, elem_classes="panel"):
                with gr.Row():
                    proj = gr.Dropdown(projects(), value=first_project(), label="Project", scale=4)
                    rb = gr.Button("Refresh", scale=1); sb = gr.Button("Stop", variant="stop", scale=1)
                status = gr.HTML(status_html(first_project()))
                final = gr.Video(label="Final video", height=380, interactive=False)
                gr.HTML("<div class='section-title'>Shots · select one to edit</div>")
                shotgal = gr.Gallery(show_label=False, columns=5, height=230, allow_preview=False, object_fit="cover")
                with gr.Row(equal_height=False):
                    shotvid = gr.Video(label="Selected shot", height=220, interactive=False, scale=5)
                    with gr.Column(scale=5):
                        req = gr.Textbox(label="Edit this shot (any language)", lines=2,
                                         placeholder="e.g. she looks too pale and not like Giang, softer smile")
                        pb = gr.Button("Plan edit")
                        pmsg = gr.Markdown()
                        with gr.Accordion("Edit settings", open=True):
                            note = gr.Textbox(label="Appearance note (English, added to this shot's prompt)", lines=2)
                            ident = gr.Radio(list(IDENT), value="same", label="Identity strength")
                            with gr.Row():
                                chl = gr.Checkbox(label="Change light", value=False)
                                light = gr.Slider(0, 1, value=1.0, step=0.1, label="Re-light strength")
                            keepc = gr.Radio(KEEP, value="unchanged", label="Clothes")
                            take = gr.Checkbox(label="New take (another variation)", value=True)
                        rr = gr.Button("Re-render shot", variant="primary"); rmsg = gr.Markdown()
                with gr.Accordion("Log", open=False):
                    logbox = gr.Textbox(show_label=False, lines=14, autoscroll=True)

        def on_char(evt: gr.SelectData):
            n = characters()[evt.index]
            return n, f"Character: **{n}**"
        gal.select(on_char, None, [sel_char, chosen])
        cb.click(add_character, [cn, cp, cf, cd], [cm, gal])
        go.click(start_swap, [video, who, sel_char, keep, quality, relight, protect], [msg, proj])
        outs = [status, final, shotgal, shot_nums]
        proj.change(refresh, proj, outs); rb.click(refresh, proj, outs); ui.load(refresh, proj, outs)
        shotgal.select(pick_shot, [proj, shot_nums], [shotvid, cur_shot, note])
        pb.click(plan_edit, [proj, cur_shot, req], [pmsg, note, ident, chl, light, keepc, take])
        rr.click(rerender_shot, [proj, cur_shot, note, ident, chl, light, keepc, take], rmsg)
        sb.click(stop, proj, msg)
        gr.Timer(4).tick(tick, proj, [status, logbox, final, shotgal, shot_nums])
    return ui


def main() -> None:
    os.chdir(ROOT)
    pf = ROOT / ".ui_password"
    pw = os.environ.get("UI_PASSWORD") or (pf.read_text().strip() if pf.exists() else "swap")
    build().queue().launch(server_name=os.environ.get("UI_HOST", "127.0.0.1"),
                           server_port=int(os.environ.get("UI_PORT", "7860")),
                           auth=(os.environ.get("UI_USER", "swap"), pw), allowed_paths=[str(CASES), str(CHARS)],
                           share=False)


if __name__ == "__main__":
    main()
