"""Generate-and-score loop: run a grid of generation settings on the container, score each map here, rank.

    python -m beat_sim.loop --tag r1 --songs lying sawadika \\
        --grid "notes-per-beat=1.0,1.5,2.0" "min-gap=0.1,0.15" --extra="--two-stage --temperature 0.9"

Each grid cell is one `beat-weaver generate` call per song inside the beat-weaver container on $BW_HOST,
written to /data/generate/<tag>/<song>__<cell>/. Maps are copied to ~/Downloads/<tag>/ and scored with
beat_sim.evaluate. Results land in ~/Downloads/<tag>/results.csv, ranked by mean composite over songs.
Re-running with the same tag skips cells whose map folder already exists.
"""

from __future__ import annotations

import argparse
import os
import csv
import itertools
import json
import shlex
import subprocess
import tarfile
import io
from pathlib import Path

from beat_sim import evaluate, to_dict
from beat_sim.music import find_audio

# ssh target of the machine running the beat-weaver container, e.g. "user@gpu-box". Set BW_HOST in the environment.
HOST = os.environ.get("BW_HOST", "")
CONTAINER = "beat-weaver"
REMOTE_ROOT = "/data/generate"
CHECKPOINT = "models/current"  # see /data/models/README.md in the container
ENV = "PYTHONPATH=/tmp/bwft NUMBA_CPU_NAME=generic NUMBA_CACHE_DIR=/tmp/nc_generic"


def _need_host() -> None:
    if not HOST:
        raise SystemExit("BW_HOST is not set: export BW_HOST=user@host (the machine running the beat-weaver container)")


def remote(cmd: str, check: bool = True) -> str:
    _need_host()
    full = f"docker exec {CONTAINER} sh -c {shlex.quote(cmd)}"
    r = subprocess.run(["ssh", HOST, full], capture_output=True, text=True)
    if check and r.returncode:
        raise RuntimeError(f"remote failed: {cmd}\n{r.stderr[-2000:]}")
    return r.stdout


def sync_code(repo: Path) -> None:
    _need_host()
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        for name in ("beat_weaver", "configs", "pyproject.toml"):
            tf.add(repo / name, arcname=name)
    subprocess.run(["ssh", HOST, f"docker exec -i {CONTAINER} sh -c 'mkdir -p /tmp/bwft && tar -xf - -C /tmp/bwft'"],
                   input=buf.getvalue(), check=True)


def parse_grid(specs: list[str]) -> list[dict[str, str]]:
    axes = []
    for spec in specs:
        key, _, vals = spec.partition("=")
        axes.append([(key.strip(), v.strip()) for v in vals.split(",")])
    return [dict(combo) for combo in itertools.product(*axes)] if axes else [{}]


SWITCHES = {"beat-focus", "allow-near-converging", "two-stage"}  # store_true flags: grid value 1 = pass, 0 = omit


def cell_flags(cell: dict[str, str]) -> str:
    out = []
    for k, v in cell.items():
        if k in SWITCHES:
            if v not in ("0", "false", "False", ""):
                out.append(f"--{k}")
        else:
            out.append(f"--{k} {v}")
    return " ".join(out)


def cell_name(cell: dict[str, str]) -> str:
    return "_".join(f"{k.replace('-', '')}{v}" for k, v in cell.items()) or "default"


def fetch(remote_dir: str, local_dir: Path) -> None:
    _need_host()
    local_dir.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(["ssh", HOST, f"docker exec {CONTAINER} tar -cf - -C {shlex.quote(str(Path(remote_dir).parent))} {shlex.quote(Path(remote_dir).name)}"],
                       capture_output=True, check=True)
    with tarfile.open(fileobj=io.BytesIO(r.stdout)) as tf:
        tf.extractall(local_dir.parent, filter="data")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True, help="run name; folder under /data/generate and ~/Downloads")
    ap.add_argument("--songs", nargs="+", default=["lying", "sawadika"], help="names of /data/generate/<song>.mp3")
    ap.add_argument("--grid", nargs="*", default=[], help='axes like "notes-per-beat=1.0,1.5"')
    ap.add_argument("--extra", default="--two-stage", help="flags added to every generate call")
    ap.add_argument("--checkpoint", default=CHECKPOINT)
    ap.add_argument("--difficulty", default="Expert")
    ap.add_argument("--no-sync", action="store_true", help="skip pushing local code to the container")
    ap.add_argument("--score-only", action="store_true", help="don't generate; rescore what is already in ~/Downloads/<tag>")
    ap.add_argument("--seeds", type=int, default=1, help="generate each cell this many times (seeds 0..N-1) and average")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parent.parent
    local_root = Path.home() / "Downloads" / args.tag
    local_root.mkdir(parents=True, exist_ok=True)
    cells = parse_grid(args.grid)
    seeds = list(range(args.seeds))
    print(f"{len(cells)} cells x {len(args.songs)} songs x {len(seeds)} seeds -> {local_root}")

    if not args.score_only:
        if not args.no_sync:
            sync_code(repo)
        for cell in cells:
            for song in args.songs:
                for seed in seeds:
                    name = f"{song}__{cell_name(cell)}" + (f"__s{seed}" if len(seeds) > 1 else "")
                    out_dir = f"{REMOTE_ROOT}/{args.tag}/{name}"
                    if (local_root / name / "Info.dat").exists():
                        print(f"skip {name} (exists)")
                        continue
                    flags = cell_flags(cell) + f" --seed {seed}"
                    cmd = (f"cd /data && {ENV} python -m beat_weaver.cli generate --checkpoint {args.checkpoint} "
                           f"--audio generate/{song}.mp3 --difficulty {args.difficulty} --output {out_dir} {args.extra} {flags}")
                    print(f"generate {name} ...", end="", flush=True)
                    log = remote(cmd)
                    print(" " + (log.strip().splitlines() or ["done"])[-1])
                    fetch(out_dir, local_root / name)

    from beat_weaver.parsers.beatmap_parser import parse_map_folder

    rows = []
    for cell in cells:
        for song in args.songs:
            for seed in seeds:
                name = f"{song}__{cell_name(cell)}" + (f"__s{seed}" if len(seeds) > 1 else "")
                folder = local_root / name
                if not folder.exists():
                    continue
                bms = [b for b in parse_map_folder(folder) if b.difficulty_info.difficulty.lower() == args.difficulty.lower()]
                if not bms:
                    continue
                bm = bms[0]
                ev = evaluate(bm.notes, bm.metadata.bpm, bm.difficulty_info.difficulty, audio=find_audio(folder))
                (folder / "report.json").write_text(json.dumps(to_dict(ev), indent=2))
                r, b = ev["rhythm"], ev["body"]
                rows.append({"cell": cell_name(cell), "song": song, "seed": seed, "composite": round(ev["composite"], 1),
                             "rhythm": round(r.score, 1), "body": round(b.score, 1),
                             **{k: round(v, 3) for k, v in r.stats.items()},
                             "too_fast": b.left.too_fast + b.right.too_fast, "resets": b.left.resets + b.right.resets,
                             "out_of_band": ",".join(k for k, ok in r.in_band.items() if not ok)})
                music = f" music={r.stats['energy_corr']:.2f}/{r.stats['onset_hit']:.2f}" if "energy_corr" in r.stats else ""
                print(f"{name:40s} composite={ev['composite']:5.1f} rhythm={r.score:5.1f} body={b.score:5.1f} "
                      f"nps={r.stats['nps']:.2f} streams={r.stats['streams_per_min']:.1f}/min{music}")

    if not rows:
        return
    with open(local_root / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    import statistics

    by_cell: dict[str, dict[str, list[float]]] = {}
    for r in rows:
        by_cell.setdefault(r["cell"], {}).setdefault(r["song"], []).append(r["composite"])
    print("\nranked by mean composite (per song: mean ± sd over seeds):")
    for cell, songs in sorted(by_cell.items(), key=lambda kv: -statistics.mean(v for s in kv[1].values() for v in s)):
        allv = [v for s in songs.values() for v in s]
        per = "  ".join(f"{s}={statistics.mean(v):.1f}±{statistics.pstdev(v):.1f}" for s, v in songs.items())
        print(f"  {statistics.mean(allv):5.1f}  {cell:40s} {per}")
    print(f"\n{local_root / 'results.csv'}")


if __name__ == "__main__":
    main()
