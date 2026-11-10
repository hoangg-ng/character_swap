#!/usr/bin/env python3
"""Shot-aware character swap for multi-shot clips.

The H3 character-swap LoRA expects ONE continuous shot. This wrapper
  1. detects scene cuts (frame accurate),
  2. cuts the clip into shots,
  3. runs character_swap.py on every shot except those in --keep,
  4. re-assembles all shots at 24 fps / the delivery size, over the ORIGINAL audio.

Example
  python swap_shots.py --video hug_scene.mp4 --character phuc.jpg --keep 3 \
      --subject "woman with long black hair in a white jacket" \
      --character-desc "..." --resolution 720

Any argument not listed below is forwarded to character_swap.py unchanged.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import queue
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT_FPS = 24


def sh(cmd: list[str]) -> str:
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"failed: {' '.join(map(str, cmd))}\n{r.stderr[-2000:]}")
    return r.stdout + r.stderr


def probe(path: Path) -> dict:
    out = sh(["ffprobe", "-v", "error", "-select_streams", "v", "-count_frames", "-print_format", "json",
              "-show_entries", "stream=width,height,avg_frame_rate,start_time,nb_read_frames", str(path)])
    s = json.loads(out[out.index("{"):])["streams"][0]
    n, d = s["avg_frame_rate"].split("/")
    return {"w": s["width"], "h": s["height"], "fps": float(n) / float(d),
            "start": float(s.get("start_time", 0) or 0), "frames": int(s["nb_read_frames"])}


def detect_cuts(path: Path, info: dict, threshold: float) -> list[int]:
    err = subprocess.run(["ffmpeg", "-hide_banner", "-i", str(path), "-an", "-vf",
                          f"select='gt(scene,{threshold})',showinfo", "-f", "null", "-"],
                         capture_output=True, text=True).stderr
    times = [float(t) for t in re.findall(r"pts_time:([0-9.]+)", err)]
    return sorted({round((t - info["start"]) * info["fps"]) for t in times})


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", required=True, type=Path)
    p.add_argument("--character", required=True, type=Path)
    p.add_argument("--keep", default="", help="1-based shot numbers to leave untouched, e.g. '3' or '1,3'")
    p.add_argument("--cuts", default="", help="manual cut frame indices (overrides detection), e.g. '33,92'")
    p.add_argument("--cut-threshold", type=float, default=0.2)
    p.add_argument("--min-frames", type=int, default=60,
                   help="shots shorter than this (source frames) are ping-pong extended for generation")
    p.add_argument("--resolution", default="720")
    p.add_argument("--job", default="", help="reuse a job name to resume (finished shots are skipped)")
    p.add_argument("--comfy-urls", default="http://127.0.0.1:8188",
                   help="comma-separated ComfyUI servers, one per GPU; shots are spread across them")
    p.add_argument("--list-shots", action="store_true", help="print detected shots and exit")
    args, passthrough = p.parse_known_args()

    info = probe(args.video)
    cuts = ([int(c) for c in args.cuts.split(",") if c.strip()] if args.cuts
            else detect_cuts(args.video, info, args.cut_threshold))
    bounds = [0] + [c for c in cuts if 0 < c < info["frames"]] + [info["frames"]]
    shots = [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]
    keep = {int(k) for k in args.keep.split(",") if k.strip()}
    for i, (a, b) in enumerate(shots, 1):
        print(f"shot {i}: frames {a:4d}-{b - 1:4d}  ({(b - a) / info['fps']:.2f}s)  "
              f"{'KEEP' if i in keep else 'swap'}")
    if args.list_shots:
        return

    job = args.job or f"{args.video.stem}_{dt.datetime.now():%Y%m%d_%H%M%S}"
    job_dir = ROOT / "outputs" / job
    seg_dir = job_dir / "shots"
    seg_dir.mkdir(parents=True, exist_ok=True)

    # delivery size: short side = --resolution if numeric, else source size
    if args.resolution.isdigit() and args.resolution not in ("480", "768"):
        s = int(args.resolution)
        W, H = ((round(info["w"] * s / info["h"] / 2) * 2, s) if info["w"] >= info["h"]
                else (s, round(info["h"] * s / info["w"] / 2) * 2))
    else:
        W, H = info["w"], info["h"]

    # ---- cut every shot ----
    srcs = {}
    for i, (a, b) in enumerate(shots, 1):
        src = seg_dir / f"shot{i:02d}_src.mp4"
        t0, t1 = a / info["fps"], b / info["fps"]
        sh(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(args.video),
            "-vf", f"setpts=PTS-STARTPTS,trim=start_frame={a}:end_frame={b},setpts=PTS-STARTPTS",
            "-af", f"asetpts=PTS-STARTPTS,atrim=start={t0}:end={t1},asetpts=PTS-STARTPTS",
            "-c:v", "libx264", "-crf", "10", "-pix_fmt", "yuv420p", "-c:a", "aac", str(src)])
        n = b - a
        if n < args.min_frames:
            # ping-pong loop (fwd, rev, fwd, ...) so the model sees enough motion context;
            # only the first n frames of the result are used (normalisation trims to the shot length)
            reps = -(-args.min_frames // n)
            ext = seg_dir / f"shot{i:02d}_src_ext.mp4"
            chain = "".join(f"[0:v]{'reverse,' if k % 2 else ''}setpts=PTS-STARTPTS[v{k}];" for k in range(reps))
            sh(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(src), "-filter_complex",
                chain + "".join(f"[v{k}]" for k in range(reps)) + f"concat=n={reps}:v=1:a=0[out]",
                "-map", "[out]", "-c:v", "libx264", "-crf", "10", "-pix_fmt", "yuv420p", str(ext)])
            print(f"shot {i}: {n} frames < {args.min_frames}, ping-pong extended x{reps} for generation")
            src = ext
        srcs[i] = src

    def done_output(i: int) -> Path | None:
        sub = ROOT / "outputs" / f"{job}/shot{i:02d}"
        cands = sorted(sub.glob("final_*.mp4")) or sorted(sub.glob("save_*.mp4"))
        return cands[0] if cands else None

    # ---- swap shots in parallel: one worker per ComfyUI server (one server per GPU) ----
    urls = [u.strip() for u in args.comfy_urls.split(",") if u.strip()]
    todo = [i for i in range(1, len(shots) + 1) if i not in keep and not done_output(i)]
    # longest shots first so the GPUs finish at about the same time
    todo.sort(key=lambda i: shots[i - 1][0] - shots[i - 1][1])
    free: queue.Queue[str] = queue.Queue()
    for u in urls:
        free.put(u)

    def swap(i: int) -> tuple[int, int]:
        url = free.get()
        try:
            log = seg_dir / f"shot{i:02d}.log"
            cmd = [sys.executable, "-u", str(ROOT / "character_swap.py"), "--video", str(srcs[i]),
                   "--character", str(args.character), "--resolution", args.resolution,
                   "--job", f"{job}/shot{i:02d}", "--comfy-url", url, *passthrough]
            print(f"=== shot {i}: swapping on {url} (log: {log})", flush=True)
            with open(log, "w") as fh:
                rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT).returncode
            print(f"=== shot {i}: {'done' if rc == 0 else f'FAILED rc={rc}'}", flush=True)
            return i, rc
        finally:
            free.put(url)

    skipped = [i for i in range(1, len(shots) + 1) if i not in keep and i not in todo]
    if skipped:
        print(f"already done, skipping: {skipped}")
    with ThreadPoolExecutor(max_workers=len(urls)) as ex:
        failed = [i for i, rc in ex.map(swap, todo) if rc != 0]
    if failed:
        sys.exit(f"shots {failed} failed - see logs in {seg_dir}; rerun the same command to resume")

    # ---- normalise every shot to OUT_FPS, delivery size and the exact source duration ----
    normalized = []
    for i, (a, b) in enumerate(shots, 1):
        result = seg_dir / f"shot{i:02d}_src.mp4" if i in keep else done_output(i)
        dur = (b - a) / info["fps"]
        norm = seg_dir / f"shot{i:02d}_norm.mp4"
        sh(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(result), "-an",
            "-vf", f"fps={OUT_FPS},scale={W}:{H}:flags=lanczos,setsar=1,"
                   f"tpad=stop_mode=clone:stop_duration=1",
            "-t", f"{dur:.6f}", "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p", str(norm)])
        normalized.append(norm)

    lst = seg_dir / "concat.txt"
    lst.write_text("".join(f"file '{n.resolve()}'\n" for n in normalized))
    final = job_dir / f"{args.video.stem}_swapped_{H if W >= H else W}p.mp4"
    sh(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst),
        "-i", str(args.video), "-map", "0:v", "-map", "1:a?", "-c:v", "libx264", "-crf", "16",
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-shortest", str(final)])
    print(f"\nFINAL -> {final}")


if __name__ == "__main__":
    main()
