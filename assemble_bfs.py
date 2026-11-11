#!/usr/bin/env python3
"""Assemble a full clip from per-shot BFS results, keeping everything outside the SAM mask pixel-exact.

For every swapped shot: out = source * (1 - a) + generated * a, where a = the shot's own per-frame SAM mask
(saved by run_bfs.py as mask_save_*.png), feathered. Frames where SAM found nobody have a = 0 -> pure source.
The leading sacrificial frames are dropped, the shot is trimmed to its real length and scaled to the source size.
Shots without a generated result (or listed in --keep) come from the source. Original audio over the whole clip.

Usage:
  python assemble_bfs.py --out outputs/final_bfs_sam --shot 1=outputs/bfs_hide/shot01 --shot 2=outputs/bfs_sam/shot02 ...
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
FPS_SRC = 30000 / 1001


def read(path: Path, start: int = 0, end: int = 10 ** 9) -> list:
    cap, out, i = cv2.VideoCapture(str(path)), [], 0
    while True:
        ok, f = cap.read()
        if not ok or i >= end:
            break
        if i >= start:
            out.append(f)
        i += 1
    return out


def write(frames: list, path: Path) -> None:
    h, w = frames[0].shape[:2]
    p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
                          "-r", "24", "-i", "-", "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p", str(path)],
                         stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(np.ascontiguousarray(f).tobytes())
    p.stdin.close(); p.wait()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--cuts", required=True, help="shot boundaries in source frames (as used by bfs_full.py)")
    ap.add_argument("--size", default="source", help="output WxH, default = source size (even-rounded)")
    ap.add_argument("--shot", action="append", default=[], help="N=<shot result dir> (contains save_*.mp4, "
                    "mask_save_*.png, lead.txt; its parent holds shotNN_in.mp4)")
    ap.add_argument("--feather", type=float, default=3.0)
    ap.add_argument("--raw", action="store_true", help="no pixel composite at all (pure generated frames)")
    ap.add_argument("--color-match", action="store_true",
                    help="match the generated frame's colour to the source (fitted on the background, outside the "
                         "mask) before blending -> no colour zone at the mask edge")
    ap.add_argument("--letterbox", action="store_true",
                    help="re-apply the source's black letterbox bars (detected per shot) with a small margin; BFS "
                         "paints into the bars and leaves a dark line at the bar edge")
    ap.add_argument("--bar-margin", type=int, default=3)
    ap.add_argument("--tone-person", type=float, default=0.0,
                    help="strength (0-1) of matching the generated person's tone to the original person "
                         "(tone_match.py), applied only to shots where the generated person is clearly brighter")
    ap.add_argument("--tone-trigger", type=float, default=1.25,
                    help="apply --tone-person when generated/original person lightness exceeds this ratio")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    global FPS_SRC
    cap = cv2.VideoCapture(str(args.video)); total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    FPS_SRC = cap.get(cv2.CAP_PROP_FPS) or FPS_SRC
    sw, sh = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)); cap.release()
    OW, OH = (sw // 2 * 2, sh // 2 * 2) if args.size == "source" else map(int, args.size.lower().split("x"))
    b = [0] + [int(c) for c in args.cuts.split(",")] + [total]
    shots = {i + 1: (b[i], b[i + 1]) for i in range(len(b) - 1)}
    given = {int(k): Path(v) for k, v in (s.split("=", 1) for s in args.shot)}
    args.out.mkdir(parents=True, exist_ok=True)
    parts, log = [], {}
    for i, (a, e) in shots.items():
        src = read(args.video, a, e)
        n = int(round(len(src) * 24 / FPS_SRC))
        if i not in given:
            frames = [src[min(int(round(k * FPS_SRC / 24)), len(src) - 1)] for k in range(n)]
            frames = [cv2.resize(f, (OW, OH), interpolation=cv2.INTER_LANCZOS4) for f in frames]
            log[i] = "source"
        else:
            d = given[i]
            gen_p = sorted(d.glob("save_*.mp4"), key=lambda p: p.stat().st_mtime)[-1]
            lead = int((d / "lead.txt").read_text()) if (d / "lead.txt").exists() else 1
            inp = d.parent / f"{d.name}_in.mp4"
            gen, srcin = read(gen_p), read(inp)
            masks = [] if args.raw else sorted(d.glob("mask_save_*.png"))
            prot = [] if args.raw else sorted(d.glob("prot_save_*.png"))
            frames = []
            for k in range(lead, lead + n):
                g = gen[min(k, len(gen) - 1)].astype(np.float32)
                s = srcin[min(k, len(srcin) - 1)].astype(np.float32)
                if masks:
                    m = cv2.imread(str(masks[min(k, len(masks) - 1)]), 0)
                    m = cv2.resize(m, (g.shape[1], g.shape[0]), interpolation=cv2.INTER_NEAREST) > 127
                    if args.color_match:
                        bg = cv2.erode((~m).astype(np.uint8), np.ones((31, 31), np.uint8)) > 0
                        if bg.sum() > 1000:
                            for ch in range(3):        # per-channel linear fit gen -> src on background pixels
                                x, y = g[..., ch][bg], s[..., ch][bg]
                                A = np.vstack([x, np.ones_like(x)]).T
                                k, c0 = np.linalg.lstsq(A, y, rcond=None)[0]
                                g[..., ch] = g[..., ch] * k + c0
                    al = cv2.GaussianBlur(m.astype(np.float32), (0, 0), args.feather)[..., None]
                    g = s * (1 - al) + g * al                      # outside her mask: original pixels
                if prot and not masks:
                    # restore only the protected person (the man) from the source; she is untouched
                    pm = cv2.imread(str(prot[min(k, len(prot) - 1)]), 0)
                    pm = cv2.resize(pm, (g.shape[1], g.shape[0]), interpolation=cv2.INTER_NEAREST) > 127
                    pm = cv2.erode(pm.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
                    al = cv2.GaussianBlur(pm.astype(np.float32), (0, 0), args.feather)[..., None]
                    g = g * (1 - al) + s * al
                frames.append(cv2.resize(np.clip(g, 0, 255).astype(np.uint8), (OW, OH),
                                         interpolation=cv2.INTER_LANCZOS4))
            how = "SAM-mask composite" if masks else ("man restored from source" if prot else "raw (no composite)")
            log[i] = f"{gen_p} | lead {lead} | {how}"
            if args.tone_person > 0:
                from utils import tone_match as tm
                srcs = [cv2.resize(srcin[min(k, len(srcin) - 1)], (OW, OH)) for k in range(lead, lead + n)]
                lg, ls = [], []
                for f, sf in list(zip(frames, srcs))[::6]:           # sample: person lightness, generated vs source
                    bg_, bs_ = tm.person_box(f), tm.person_box(sf)
                    if bg_ and bs_:
                        lg.append(cv2.cvtColor(f[bg_[1]:bg_[3], bg_[0]:bg_[2]], cv2.COLOR_BGR2LAB)[..., 0].mean())
                        ls.append(cv2.cvtColor(sf[bs_[1]:bs_[3], bs_[0]:bs_[2]], cv2.COLOR_BGR2LAB)[..., 0].mean())
                ratio = float(np.mean(lg) / max(np.mean(ls), 1.0)) if lg else 1.0
                if ratio > args.tone_trigger:
                    frames = [tm.tone_frame(sf, f, args.tone_person) for f, sf in zip(frames, srcs)]
                    log[i] += f" | person toned (generated {ratio:.2f}x brighter)"
        if args.letterbox and i in given:
            # rows that are black in every source frame of this shot = letterbox bars
            rm = np.stack([cv2.resize(f, (OW, OH)).mean(axis=(1, 2)) for f in src]).max(axis=0)
            content = np.where(rm > 12)[0]
            if len(content) and (content[0] > 0 or content[-1] < OH - 1):
                top, bot = max(content[0] + args.bar_margin, 0), min(content[-1] - args.bar_margin, OH - 1)
                for f in frames:
                    f[:top] = 0
                    f[bot + 1:] = 0
                log[i] += f" | letterbox rows <{top} and >{bot}"
        p = args.out / f"s{i}.mp4"
        write(frames, p)
        parts.append(p)
        print(f"shot {i}: {log[i]}")
    (args.out / "list.txt").write_text("".join(f"file '{p.resolve()}'\n" for p in parts))
    final = args.out / "swapped.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(args.out / "list.txt"),
                    "-i", str(args.video), "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-crf", "16",
                    "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(final)], check=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(args.video), "-i", str(final), "-filter_complex",
                    "[0:v]scale=-2:360[a];[1:v]scale=-2:360[b];[a][b]hstack", "-map", "1:a", "-c:v", "libx264",
                    "-crf", "18", "-pix_fmt", "yuv420p", str(args.out / "side_by_side.mp4")], check=True)
    json.dump({str(k): v for k, v in log.items()}, open(args.out / "picks.json", "w"), indent=1)
    print(f"FINAL -> {final}")


if __name__ == "__main__":
    main()
