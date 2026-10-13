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
from beat_sim.meta import extract_cover, song_meta
from beat_sim.loop import CHECKPOINT, CONTAINER, ENV, HOST, REMOTE_ROOT, fetch, remote, sync_code
from beat_sim.music import find_audio
from beat_sim.rhythm import BANDS

DIFFS = ("Easy", "Normal", "Hard", "Expert", "ExpertPlus")
RANK = {"Easy": 1, "Normal": 3, "Hard": 5, "Expert": 7, "ExpertPlus": 9}

# Flags that are not density. doubles_bias ~1.0 lands near the favourites' 0.25 doubles rate; lower
# difficulties get fewer doubles and a wider min gap so the schedule can't form streams at all.
# max_doubles caps the doubles rate, which otherwise swings 0.1-0.8 by song and let Hard out-dense Expert.
PRESET = {
    "Easy":       dict(min_gap=0.40, doubles_bias=0.3, flow_penalty=4, dynamics=1,   vision_penalty=6, stream_bias=1.0, column_max_run=0, reset_gap=0.5, max_doubles=0.15, beat_floor=0.3, max_run=0,  beat_focus=False),
    "Normal":     dict(min_gap=0.25, doubles_bias=0.5, flow_penalty=4, dynamics=1,   vision_penalty=6, stream_bias=1.0, column_max_run=0, reset_gap=0.5, max_doubles=0.2,  beat_floor=0.3, max_run=0,  beat_focus=False),
    "Hard":       dict(min_gap=0.15, doubles_bias=0.8, flow_penalty=4, dynamics=1,   vision_penalty=6, stream_bias=1.0, column_max_run=0, reset_gap=0.5, max_doubles=0.25, beat_floor=0.3, max_run=12, beat_focus=False),
    "Expert":     dict(min_gap=0.10, doubles_bias=1.0, flow_penalty=4, dynamics=1,   vision_penalty=6, stream_bias=0.6, column_max_run=4, reset_gap=0.5, max_doubles=0.4,  beat_floor=0.3, max_run=20, beat_focus=False),
    "ExpertPlus": dict(min_gap=0.10, doubles_bias=1.0, flow_penalty=4, dynamics=1.5, vision_penalty=6, stream_bias=0.6, column_max_run=4, reset_gap=0.5, max_doubles=0.45, beat_floor=0.3, max_run=32, beat_focus=False),
}
# beat_floor / max_run / beat_focus / stream_bias 0.6 / dynamics 1.5: added after Remedy's Expert+ came out as one
# 210-note eighth-note line (93% half-beat gaps, 17 rests/min). Quiet beats now rest, runs are capped and
# end in a whole-beat rest, extras bunch on the strongest beats, chains take hot sixteenths, loud bars fill more.
DOUBLES_FRACTION = {"Easy": 0.10, "Normal": 0.15, "Hard": 0.20, "Expert": 0.25, "ExpertPlus": 0.25}
DENSITY_SCALE = {"Expert": 1.1, "ExpertPlus": 1.1}  # ponytail: generated nps lands ~18% over target and the scorer picked the low candidate; tune per play test
DENSITY_BONUS = {"Expert": 10.0, "ExpertPlus": 10.0}  # max points added to the ranking score (not the report)
DENSITY_VARIANTS = (0.85, 1.0, 1.15)
# bsmg-guideline shaping applied to every candidate (see beat_weaver/model/shaping.py and docs/MAPPING_GUIDELINES.md)
SHAPING = {"drift_bias": 1.5, "momentum_bias": 0.5, "flow_window": 1.5, "repeats": 0.7, "emphasis": 0.08}
DOWNMAP_FROM = {"Easy": "Hard", "Normal": "Hard", "Hard": "Hard"}  # ponytail: Hard is generated; Easy/Normal are thinned from it  # ponytail: two hedges around the preset; widen if winners sit at an edge


def density_bonus(diff: str, nps: float) -> float:
    """Ranking bonus for density: 0 at the bottom of the difficulty's nps band, full at its midpoint, flat above.

    The composite only counts metrics in band, so it always preferred the lightest in-band candidate. Capped at the
    midpoint so a denser chart that leaves the band (and loses points there) is never rewarded for it.
    """
    lo, hi = BANDS[diff]["nps"]
    return DENSITY_BONUS.get(diff, 0.0) * min(max((nps - lo) / (hi - lo), 0.0), 0.5) / 0.5


def target_nps(diff: str) -> float:
    lo, hi = BANDS[diff]["nps"]
    return (lo + hi) / 2 / (1 + DOUBLES_FRACTION[diff]) * DENSITY_SCALE.get(diff, 1.0)


def flags_for(diff: str, nps: float, seed: int, downmap: tuple[str, ...] = ()) -> str:
    p = PRESET[diff]
    sh = " ".join(f"--{k.replace('_', '-')} {v}" for k, v in SHAPING.items())
    dm = (" --downmap " + " ".join(downmap)) if downmap else ""
    return (("--beat-focus " if p["beat_focus"] else "") + f"--beat-floor {p['beat_floor']} --max-run {p['max_run']} {sh}{dm} "
            f"--two-stage --notes-per-second {nps:.2f} --min-gap {p['min_gap']} --doubles-bias {p['doubles_bias']} "
            f"--flow-penalty {p['flow_penalty']} --dynamics {p['dynamics']} --vision-penalty {p['vision_penalty']} "
            f"--stream-bias {p['stream_bias']} --column-max-run {p['column_max_run']} "
            f"--reset-gap {p['reset_gap']} --max-doubles {p['max_doubles']} --seed {seed}")


def upload_song(song: Path, stem: str | None = None) -> str:
    """Put the song in /data/generate on the container if it isn't there; return its remote name."""
    name = f"{stem}{song.suffix}" if stem else song.name
    if not remote(f"ls {REMOTE_ROOT}/{shlex.quote(name)} 2>/dev/null", check=False).strip():
        with open(song, "rb") as f:
            subprocess.run(["ssh", HOST, f"docker exec -i {CONTAINER} sh -c 'cat > {shlex.quote(REMOTE_ROOT + '/' + name)}'"],
                           stdin=f, check=True)
    return name


def merge(winners: dict[str, Path], out: Path, meta: dict, cover: Path | None = None) -> Path:
    """One folder, one Info.dat, every winning <Diff>.dat, audio copied once.

    `meta` overrides Info.dat fields (song name, artist, mapper, preview). `cover` is an image file to copy in.
    """
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    first = next(iter(winners.values()))
    info = json.loads((first / "Info.dat").read_text())
    audio = find_audio(first)
    shutil.copy(audio, out / audio.name)
    info["_songFilename"] = audio.name
    info.update(meta)
    if cover:
        shutil.copy(cover, out / cover.name)
    info["_coverImageFilename"] = cover.name if cover else ""
    beatmaps = []
    for diff in DIFFS:
        if diff not in winners:
            continue
        src = winners[diff]
        entries = json.loads((src / "Info.dat").read_text())["_difficultyBeatmapSets"][0]["_difficultyBeatmaps"]
        entry = next((e for e in entries if e["_difficulty"] == diff), entries[0])  # candidate folders may hold downmaps too
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


def score_candidate(arg: tuple[str, str, bool]):
    """Score one candidate folder in a worker process. Returns (composite, nps, report dict, out-of-band names) or None.

    With reuse=True a report.json already in the folder is used as is instead of rescoring (minutes per map).
    """
    from beat_weaver.parsers.beatmap_parser import parse_map_folder

    folder, diff, reuse = Path(arg[0]), arg[1], arg[2]
    cached = folder / "report.json"
    if reuse and cached.exists():
        d = json.loads(cached.read_text())
        return d["composite"], d["rhythm"]["stats"]["nps"], d, [k for k, ok in d["rhythm"]["in_band"].items() if not ok]
    bms = [b for b in parse_map_folder(folder) if b.difficulty_info.difficulty == diff]
    if not bms:
        return None
    bm = bms[0]
    ev = evaluate(bm.notes, bm.metadata.bpm, diff, audio=find_audio(folder))
    d = to_dict(ev)
    (folder / "report.json").write_text(json.dumps(d, indent=2))
    return d["composite"], d["rhythm"]["stats"]["nps"], d, [k for k, ok in d["rhythm"]["in_band"].items() if not ok]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("song", type=Path, help="audio file (mp3/ogg/wav)")
    ap.add_argument("--name", help="short name for output folders and the remote copy (default: the file name)")
    ap.add_argument("--difficulties", nargs="+", default=list(DIFFS), choices=DIFFS)
    ap.add_argument("--candidates", type=int, default=4, help="maps generated per difficulty (seeds x density variants)")
    ap.add_argument("--checkpoint", default=CHECKPOINT)
    ap.add_argument("--out", type=Path, help="default ~/Downloads/<song>_map")
    ap.add_argument("--no-sync", action="store_true")
    ap.add_argument("--workers", type=int, default=6, help="parallel scoring processes")
    ap.add_argument("--reuse-reports", action="store_true", help="use each candidate's saved report.json instead of rescoring")
    ap.add_argument("--title", help="song title for Info.dat (default: audio tags, else parsed from the file name)")
    ap.add_argument("--artist", help="song artist for Info.dat (same fallbacks)")
    ap.add_argument("--mapper", help="level author name (default: $BW_MAPPER, then the user config file, then BeatWeaver AI)")
    ap.add_argument("--no-web-cover", action="store_true", help="don't look the cover up on YouTube / iTunes")
    ap.add_argument("--cover", type=Path, help="cover image to put in the map (default: art embedded in the audio, else looked up on the web)")
    ap.add_argument("--score-only", action="store_true", help="rescore candidates already fetched")
    args = ap.parse_args()

    stem = args.name or args.song.stem
    work = Path.home() / "Downloads" / f"{stem}_candidates"
    out = args.out or Path.home() / "Downloads" / f"{stem}_map"
    work.mkdir(parents=True, exist_ok=True)
    meta = song_meta(args.song, args.title, args.artist, args.mapper)
    print(f"metadata: title={meta['_songName']!r} artist={meta['_songAuthorName']!r} mapper={meta['_levelAuthorName']!r} "
          f"preview={meta['_previewStartTime']}s")
    repo = Path(__file__).resolve().parent.parent

    if not args.score_only:
        if not args.no_sync:
            sync_code(repo)
        remote_song = upload_song(args.song, args.name)
        gen_diffs = {DOWNMAP_FROM.get(d, d) for d in args.difficulties}
        for diff in [d for d in DIFFS if d in gen_diffs]:
            downmap = tuple(d for d in args.difficulties if DOWNMAP_FROM.get(d) == diff and d != diff)
            base = target_nps(diff)
            for i in range(args.candidates):
                seed, variant = divmod(i, len(DENSITY_VARIANTS))
                nps = base * DENSITY_VARIANTS[variant]
                name = f"{diff}__nps{nps:.2f}__s{seed}"
                if (work / name / "Info.dat").exists():
                    continue
                out_dir = shlex.quote(f"{REMOTE_ROOT}/{stem}_candidates/{name}")
                cmd = (f"cd /data && {ENV} python -m beat_weaver.cli generate --checkpoint {args.checkpoint} "
                       f"--audio {REMOTE_ROOT}/{shlex.quote(remote_song)} --difficulty {diff} --output {out_dir} "
                       f"{flags_for(diff, nps, seed, downmap)}")
                print(f"generate {name} ...", end="", flush=True)
                log = remote(cmd)
                print(" " + (log.strip().splitlines() or ["done"])[-1])
                fetch(f"{REMOTE_ROOT}/{stem}_candidates/{name}", work / name)

    from beat_weaver.parsers.beatmap_parser import parse_map_folder

    from concurrent.futures import ProcessPoolExecutor

    folders = [(diff, f) for diff in args.difficulties for f in sorted(work.glob(f"{DOWNMAP_FROM.get(diff, diff)}__*"))
               if (f / f"{diff}.dat").exists()]
    cands: dict[str, list] = {}
    with ProcessPoolExecutor(max_workers=min(args.workers, max(len(folders), 1))) as ex:
        for (diff, folder), res in zip(folders, ex.map(score_candidate, [(str(f), d, args.reuse_reports) for d, f in folders])):
            if res is None:
                continue
            composite, nps, ev, bad = res
            print(f"  {folder.name:34s} composite={composite:5.1f} nps={nps:.2f} "
                  f"doubles={ev['rhythm']['stats']['doubles']:.2f} out={','.join(bad) or '-'}")
            cands.setdefault(diff, []).append((composite + density_bonus(diff, nps), nps, folder, ev))
    picked = pick_ladder(cands)
    winners = {d: c[2] for d, c in picked.items()}
    report = {d: {"folder": c[2].name, **c[3]} for d, c in picked.items()}

    if not winners:
        print("no candidates found")
        return
    cover_dir = work / "_cover"
    cover_dir.mkdir(exist_ok=True)
    cover_name, cover_src = extract_cover(args.song, cover_dir, args.cover, meta["_songName"], meta["_songAuthorName"],
                                          web=not args.no_web_cover)
    print(f"cover: {cover_src or 'none found'}")
    merge(winners, out, meta, cover_dir / cover_name if cover_name else None)
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(f"\nmap folder: {out}")
    for diff in args.difficulties:
        if diff in report:
            print(f"  {diff:10s} {report[diff]['composite']:5.1f}  {report[diff]['folder']}")


if __name__ == "__main__":
    main()
