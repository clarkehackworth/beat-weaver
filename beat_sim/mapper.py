"""One song in, one map folder out, with the best map we can make at each requested difficulty.

    python -m beat_sim.mapper song.mp3 --difficulties Easy Normal Hard Expert ExpertPlus --candidates 4

For each difficulty: generate `candidates` maps on the container (different seeds, and two density
variants around the difficulty's preset), score every one with beat_sim.evaluate against that
difficulty's favourites bands, keep the best. Then merge the winners into a single playable map
folder with one Info.dat listing every difficulty, in ~/Downloads/<song>_map/.

Presets per difficulty come from the favourites bands (notes/second at the band centre, less the
~25% that doubles add) and the playtest-tuned flags from the r1-r8 grids.
"""

from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
from pathlib import Path

from beat_sim import evaluate, to_dict
from beat_sim.loop import CHECKPOINT, CONTAINER, ENV, HOST, REMOTE_ROOT, fetch, remote, sync_code
from beat_sim.music import find_audio
from beat_sim.rhythm import BANDS

DIFFS = ("Easy", "Normal", "Hard", "Expert", "ExpertPlus")
RANK = {"Easy": 1, "Normal": 3, "Hard": 5, "Expert": 7, "ExpertPlus": 9}

# Flags that are not density. doubles_bias ~1.0 lands near the favourites' 0.25 doubles rate; lower
# difficulties get fewer doubles and a wider min gap so the schedule can't form streams at all.
PRESET = {
    "Easy":       dict(min_gap=0.40, doubles_bias=0.3, flow_penalty=4, dynamics=1, vision_penalty=6, stream_bias=1.0),
    "Normal":     dict(min_gap=0.25, doubles_bias=0.5, flow_penalty=4, dynamics=1, vision_penalty=6, stream_bias=1.0),
    "Hard":       dict(min_gap=0.15, doubles_bias=0.8, flow_penalty=4, dynamics=1, vision_penalty=6, stream_bias=1.0),
    "Expert":     dict(min_gap=0.10, doubles_bias=1.0, flow_penalty=4, dynamics=1, vision_penalty=6, stream_bias=1.0),
    "ExpertPlus": dict(min_gap=0.10, doubles_bias=1.0, flow_penalty=4, dynamics=1, vision_penalty=6, stream_bias=0.8),
}
DOUBLES_FRACTION = {"Easy": 0.10, "Normal": 0.15, "Hard": 0.20, "Expert": 0.25, "ExpertPlus": 0.25}
DENSITY_VARIANTS = (0.85, 1.15)  # ponytail: two hedges around the preset; widen if winners sit at an edge


def target_nps(diff: str) -> float:
    lo, hi = BANDS[diff]["nps"]
    return (lo + hi) / 2 / (1 + DOUBLES_FRACTION[diff])


def flags_for(diff: str, nps: float, seed: int) -> str:
    p = PRESET[diff]
    return (f"--two-stage --notes-per-second {nps:.2f} --min-gap {p['min_gap']} --doubles-bias {p['doubles_bias']} "
            f"--flow-penalty {p['flow_penalty']} --dynamics {p['dynamics']} --vision-penalty {p['vision_penalty']} "
            f"--stream-bias {p['stream_bias']} --seed {seed}")


def upload_song(song: Path) -> str:
    """Put the song in /data/generate on the container if it isn't there; return its remote name."""
    name = song.name
    if not remote(f"ls {REMOTE_ROOT}/{shlex.quote(name)} 2>/dev/null", check=False).strip():
        with open(song, "rb") as f:
            subprocess.run(["ssh", HOST, f"docker exec -i {CONTAINER} sh -c 'cat > {shlex.quote(REMOTE_ROOT + '/' + name)}'"],
                           stdin=f, check=True)
    return name


def merge(winners: dict[str, Path], out: Path, song_name: str) -> Path:
    """One folder, one Info.dat, every winning <Diff>.dat, audio copied once."""
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    first = next(iter(winners.values()))
    info = json.loads((first / "Info.dat").read_text())
    audio = find_audio(first)
    shutil.copy(audio, out / audio.name)
    info["_songFilename"] = audio.name
    info["_songName"] = song_name
    beatmaps = []
    for diff in DIFFS:
        if diff not in winners:
            continue
        src = winners[diff]
        entry = json.loads((src / "Info.dat").read_text())["_difficultyBeatmapSets"][0]["_difficultyBeatmaps"][0]
        shutil.copy(src / entry["_beatmapFilename"], out / f"{diff}.dat")
        entry["_difficulty"], entry["_difficultyRank"], entry["_beatmapFilename"] = diff, RANK[diff], f"{diff}.dat"
        beatmaps.append(entry)
    info["_difficultyBeatmapSets"] = [{"_beatmapCharacteristicName": "Standard", "_difficultyBeatmaps": beatmaps}]
    (out / "Info.dat").write_text(json.dumps(info, indent=2))
    return out


LADDER_RATIO = 1.10  # each difficulty must be at least this much denser (notes/s) than the one below


def pick_ladder(cands: dict[str, list[tuple[float, float, Path, dict]]]) -> dict[str, tuple[float, float, Path, dict]]:
    """Best composite per difficulty subject to nps rising by LADDER_RATIO up the ladder (DP over candidates).

    cands[diff] = [(composite, nps, folder, ev), ...]. Without this, each level independently picks its
    best-scoring map and Hard and Expert come out the same density. Falls back to independent picks if
    no chain satisfies the ratio.
    """
    diffs = [d for d in DIFFS if cands.get(d)]
    best: list[list[tuple[float, int]]] = []  # per level, per candidate: (total, index of predecessor)
    for li, d in enumerate(diffs):
        row = []
        for c in cands[d]:
            if li == 0:
                row.append((c[0], -1))
                continue
            prev = [(best[li - 1][pi][0], pi) for pi, pc in enumerate(cands[diffs[li - 1]])
                    if best[li - 1][pi][0] > float("-inf") and c[1] >= LADDER_RATIO * pc[1]]
            row.append((c[0] + max(prev)[0], max(prev)[1]) if prev else (float("-inf"), -1))
        best.append(row)
    if diffs and max(r[0] for r in best[-1]) > float("-inf"):
        ci = max(range(len(best[-1])), key=lambda i: best[-1][i][0])
        chosen = {}
        for li in range(len(diffs) - 1, -1, -1):
            chosen[diffs[li]] = cands[diffs[li]][ci]
            ci = best[li][ci][1]
        return chosen
    print("warning: no candidate set satisfies the difficulty ladder; picking each level independently")
    return {d: max(cands[d], key=lambda c: c[0]) for d in diffs}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("song", type=Path, help="audio file (mp3/ogg/wav)")
    ap.add_argument("--difficulties", nargs="+", default=list(DIFFS), choices=DIFFS)
    ap.add_argument("--candidates", type=int, default=4, help="maps generated per difficulty (seeds x density variants)")
    ap.add_argument("--checkpoint", default=CHECKPOINT)
    ap.add_argument("--out", type=Path, help="default ~/Downloads/<song>_map")
    ap.add_argument("--no-sync", action="store_true")
    ap.add_argument("--score-only", action="store_true", help="rescore candidates already fetched")
    args = ap.parse_args()

    stem = args.song.stem
    work = Path.home() / "Downloads" / f"{stem}_candidates"
    out = args.out or Path.home() / "Downloads" / f"{stem}_map"
    work.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parent.parent

    if not args.score_only:
        if not args.no_sync:
            sync_code(repo)
        remote_song = upload_song(args.song)
        for diff in args.difficulties:
            base = target_nps(diff)
            for i in range(args.candidates):
                seed, variant = divmod(i, len(DENSITY_VARIANTS))
                nps = base * DENSITY_VARIANTS[variant]
                name = f"{diff}__nps{nps:.2f}__s{seed}"
                if (work / name / "Info.dat").exists():
                    continue
                out_dir = f"{REMOTE_ROOT}/{stem}_candidates/{name}"
                cmd = (f"cd /data && {ENV} python -m beat_weaver.cli generate --checkpoint {args.checkpoint} "
                       f"--audio {REMOTE_ROOT}/{shlex.quote(remote_song)} --difficulty {diff} --output {out_dir} "
                       f"{flags_for(diff, nps, seed)}")
                print(f"generate {name} ...", end="", flush=True)
                log = remote(cmd)
                print(" " + (log.strip().splitlines() or ["done"])[-1])
                fetch(out_dir, work / name)

    from beat_weaver.parsers.beatmap_parser import parse_map_folder

    cands: dict[str, list] = {}
    for diff in args.difficulties:
        for folder in sorted(work.glob(f"{diff}__*")):
            bms = [b for b in parse_map_folder(folder) if b.difficulty_info.difficulty == diff]
            if not bms:
                continue
            bm = bms[0]
            ev = evaluate(bm.notes, bm.metadata.bpm, diff, audio=find_audio(folder))
            (folder / "report.json").write_text(json.dumps(to_dict(ev), indent=2))
            r = ev["rhythm"]
            bad = [k for k, ok in r.in_band.items() if not ok]
            print(f"  {folder.name:34s} composite={ev['composite']:5.1f} nps={r.stats['nps']:.2f} "
                  f"doubles={r.stats['doubles']:.2f} out={','.join(bad) or '-'}")
            cands.setdefault(diff, []).append((ev["composite"], r.stats["nps"], folder, ev))
    picked = pick_ladder(cands)
    winners = {d: c[2] for d, c in picked.items()}
    report = {d: {"folder": c[2].name, **to_dict(c[3])} for d, c in picked.items()}

    if not winners:
        print("no candidates found")
        return
    merge(winners, out, stem)
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(f"\nmap folder: {out}")
    for diff in args.difficulties:
        if diff in report:
            print(f"  {diff:10s} {report[diff]['composite']:5.1f}  {report[diff]['folder']}")


if __name__ == "__main__":
    main()
