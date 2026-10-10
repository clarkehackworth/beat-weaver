"""python -m beat_sim <map folder> [--difficulty Expert] [--json out.json] [--video out.mp4|out.gif --start 10 --end 20]"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from beat_sim import evaluate, to_dict
from beat_sim.music import find_audio
from beat_weaver.parsers.beatmap_parser import parse_map_folder


def summary_line(name: str, ev: dict) -> str:
    r, b = ev["rhythm"], ev["body"]
    music = f" music={r.stats['energy_corr']:.2f}/{r.stats['onset_hit']:.2f}" if "energy_corr" in r.stats else ""
    return (f"{name:12s} composite={ev['composite']:5.1f} rhythm={r.score:5.1f} body={b.score:5.1f}  "
            f"nps={r.stats['nps']:.2f} streams/min={r.stats['streams_per_min']:.1f} dots={r.stats['dots']:.2f} "
            f"doubles={r.stats['doubles']:.2f}{music}  fast={b.left.too_fast + b.right.too_fast} resets={b.left.resets + b.right.resets}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("folder", type=Path)
    ap.add_argument("--difficulty", help="e.g. Expert, ExpertPlus; default: all")
    ap.add_argument("--json", type=Path, help="write per-difficulty reports here")
    ap.add_argument("--video", type=Path, help="stick-figure animation (.mp4 via ffmpeg, or .gif) of the first matching difficulty")
    ap.add_argument("--start", type=float, default=0.0, help="video start, seconds")
    ap.add_argument("--end", type=float, help="video end, seconds (default: whole song)")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--quiet", action="store_true", help="one line per difficulty, no band table")
    args = ap.parse_args()

    out = {}
    for bm in parse_map_folder(args.folder):
        name = bm.difficulty_info.difficulty
        if args.difficulty and name.lower() != args.difficulty.lower():
            continue
        ev = evaluate(bm.notes, bm.metadata.bpm, name, return_trajectories=bool(args.video), audio=find_audio(args.folder))
        out[name] = to_dict(ev)
        print(summary_line(name, ev))
        if not args.quiet:
            print(ev["rhythm"].table())
        if args.video:
            from beat_sim.viz import render

            render(ev["trajectories"], bm.notes, args.video, args.start, args.end, args.fps)
            print(f"wrote {args.video}")
            args.video = None
    if args.json:
        args.json.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
