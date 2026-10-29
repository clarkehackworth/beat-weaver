#!/usr/bin/env python3
"""Batch-generate playable Beat Saber maps from a set of audio files, on a GPU host.

Feed it a folder (or files) of songs and it produces one playable map folder per song,
each with every requested difficulty, correct song/artist metadata, a preview, and a cover.
Generation runs as `beat-weaver generate --two-stage` inside a `beat-weaver` Docker container on
a GPU machine over ssh (the same host the training runs on); scoring and metadata happen here.

    # every song in a folder, all five difficulties, one playable folder per song
    python scripts/batch_generate.py /path/to/songs \
        --bw-host user@gpu-host --author "Your Name" \
        --out ~/Downloads/maps

    # only the hardest two difficulties, 8 in parallel, no web cover lookup
    python scripts/batch_generate.py /path/to/songs \
        --difficulties Expert ExpertPlus --concurrency 8 --no-web-cover

    # dry run: list the songs and the per-difficulty presets it would use (no container touched)
    python scripts/batch_generate.py /path/to/songs --dry

How it works:
  1. Syncs this repository's `beat_weaver/` into the container (so the CLI there has the
     current flags), unless the container already matches.
  2. Uploads all audio to the container's /data in one stream.
  3. Generates each (song, difficulty) pair concurrently in waves sized by --concurrency,
     using that difficulty's preset from `generation.json` beside the model (see --checkpoint).
  4. Fetches the results and merges each song's difficulties into one folder with a single
     Info.dat (all levels, author, title/artist, preview, cover).

Output layout: <out>/<song name>/  containing <difficulty>.dat per level, one Info.dat, the
song audio as song.egg, and cover.jpg. Copy any folder straight into Beat Saber_Data/CustomLevels/.
"""
from __future__ import annotations

import argparse
import io
import json
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

CONTAINER = "beat-weaver"
REMOTE_DATA = "/data"
REMOTE_INBOX = "/data/bw_batch_inbox"
REMOTE_OUT = "/data/bw_batch_out"
ENV = ("export PYTHONPATH=/tmp/bwft NUMBA_CPU_NAME=generic NUMBA_CACHE_DIR=/tmp/nc_generic "
       "OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1")
ALL_DIFFS = ["Easy", "Normal", "Hard", "Expert", "ExpertPlus"]
RANK = {"Easy": 1, "Normal": 3, "Hard": 5, "Expert": 7, "ExpertPlus": 9}
AUDIO_EXTS = {".mp3", ".ogg", ".wav", ".flac", ".m4a"}


# Fallback presets when generation.json is not available. These match the production
# model's generation.json (see beat_sim/mapper.py for the scoring-tuned values).
DEFAULT_PRESET = {
    "Easy":       dict(nps=1.33, min_gap=0.4, doubles_bias=0.3, flow_penalty=4, dynamics=1, vision_penalty=6, stream_bias=1.0, column_max_run=0, reset_gap=0.5, max_doubles=0.15),
    "Normal":     dict(nps=1.94, min_gap=0.25, doubles_bias=0.5, flow_penalty=4, dynamics=1, vision_penalty=6, stream_bias=1.0, column_max_run=0, reset_gap=0.5, max_doubles=0.2),
    "Hard":       dict(nps=2.8, min_gap=0.15, doubles_bias=0.8, flow_penalty=4, dynamics=1, vision_penalty=6, stream_bias=1.0, column_max_run=0, reset_gap=0.5, max_doubles=0.25),
    "Expert":     dict(nps=3.6, min_gap=0.1, doubles_bias=1.0, flow_penalty=4, dynamics=1, vision_penalty=6, stream_bias=1.0, column_max_run=4, reset_gap=0.5, max_doubles=0.4),
    "ExpertPlus": dict(nps=4.6, min_gap=0.1, doubles_bias=1.0, flow_penalty=4, dynamics=1, vision_penalty=6, stream_bias=0.8, column_max_run=4, reset_gap=0.5, max_doubles=0.45),
}


def ssh(host: str, cmd: str, check: bool = True, binary: bool = False) -> subprocess.CompletedProcess:
    r = subprocess.run(["ssh", host, cmd],
                       capture_output=True, text=not binary)
    if check and r.returncode:
        raise RuntimeError(f"ssh failed: {cmd}\n{r.stderr[-2000:] if not binary else ''}")
    return r


def dexec(host: str, container: str, cmd: str, check: bool = True) -> str:
    return ssh(host, f"docker exec {container} sh -c {shlex.quote(cmd)}", check=check).stdout


def sync_code(host: str, container: str) -> None:
    """Push this repo's beat_weaver/ + configs into the container so its CLI is current."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        for name in ("beat_weaver", "configs", "pyproject.toml"):
            p = REPO / name
            if p.exists():
                tf.add(p, arcname=name)
    buf.seek(0)
    r = subprocess.run(["ssh", host, f"docker exec -i {container} sh -c 'mkdir -p /tmp/bwft && tar -xf - -C /tmp/bwft'"],
                       input=buf.getvalue(), capture_output=True)
    if r.returncode:
        raise RuntimeError(f"code sync failed: {r.stderr[-300:]}")


def upload_audio(host: str, container: str, files: list[Path]) -> None:
    dexec(host, container, f"rm -rf {REMOTE_INBOX} && mkdir -p {REMOTE_INBOX} {REMOTE_OUT}")
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        for f in files:  # songs can come from several folders; flatten by name (names are de-duped in main)
            tf.add(f, arcname=f.name)
    r = subprocess.run(["ssh", host, f"docker exec -i {container} sh -c 'tar -xf - -C {REMOTE_INBOX}'"],
                       input=buf.getvalue(), capture_output=True)
    if r.returncode:
        raise RuntimeError(f"remote untar failed: {r.stderr[-300:]}")


def load_presets(host: str, container: str, checkpoint: str, remote: bool) -> dict:
    """Read generation.json beside the checkpoint; fall back to the built-in presets."""
    if remote:
        txt = dexec(host, container, f"cat {shlex.quote(REMOTE_DATA + '/' + checkpoint + '/generation.json')} 2>/dev/null", check=False).strip()
    else:
        p = REPO / checkpoint / "generation.json"
        txt = p.read_text().strip() if p.exists() else ""
    if not txt:
        return dict(DEFAULT_PRESET)
    data = json.loads(txt)
    presets = {}
    for d in ALL_DIFFS:
        v = data.get(d, {})
        presets[d] = dict(nps=v.get("notes_per_second", DEFAULT_PRESET[d]["nps"]),
                          min_gap=v.get("min_gap", DEFAULT_PRESET[d]["min_gap"]),
                          doubles_bias=v.get("doubles_bias", DEFAULT_PRESET[d]["doubles_bias"]),
                          flow_penalty=v.get("flow_penalty", DEFAULT_PRESET[d]["flow_penalty"]),
                          dynamics=v.get("dynamics", DEFAULT_PRESET[d]["dynamics"]),
                          vision_penalty=v.get("vision_penalty", DEFAULT_PRESET[d]["vision_penalty"]),
                          stream_bias=v.get("stream_bias", DEFAULT_PRESET[d]["stream_bias"]),
                          column_max_run=v.get("column_max_run", DEFAULT_PRESET[d]["column_max_run"]),
                          reset_gap=v.get("reset_gap", DEFAULT_PRESET[d]["reset_gap"]),
                          max_doubles=v.get("max_doubles", DEFAULT_PRESET[d]["max_doubles"]))
    return presets


def flags_for(p: dict, seed: int, diff: str = "", downmap: tuple[str, ...] = ()) -> str:
    from beat_sim.mapper import HARD_RECIPE, RECIPE, RECIPE_DIFFS
    if diff == "Hard":  # one Hard generation also writes the requested Easy/Normal, thinned from it
        dm = (" --downmap " + " ".join(downmap)) if downmap else ""
        return f"{HARD_RECIPE} --notes-per-second {p['nps']} --seed {seed}{dm}"
    if diff in RECIPE_DIFFS:  # play-tested recipe (docs/MAPPING_GUIDELINES.md); only the density comes from the preset
        return f"{RECIPE} --notes-per-second {p['nps']} --seed {seed}"
    return (f"--two-stage --notes-per-second {p['nps']} --min-gap {p['min_gap']} "
            f"--doubles-bias {p['doubles_bias']} --flow-penalty {p['flow_penalty']} "
            f"--dynamics {p['dynamics']} --vision-penalty {p['vision_penalty']} "
            f"--stream-bias {p['stream_bias']} --column-max-run {p['column_max_run']} "
            f"--reset-gap {p['reset_gap']} --max-doubles {p['max_doubles']} --seed {seed}")


def build_wave(n: int, tasks: list[tuple], presets: dict, checkpoint: str, seed: int, downmap: tuple[str, ...] = ()) -> Path:
    lines = ["#!/bin/bash", "set -u", "cd /data", ENV]
    lines.append("run(){ python -m beat_weaver.cli generate --checkpoint " + checkpoint
                 + " --audio \"$1\" --difficulty \"$2\" --output \"$3\" $4 > \"$3.log\" 2>&1; }")
    lines.append(f"mkdir -p {REMOTE_OUT}")
    for song_name, diff, *nps in tasks:  # an optional third item overrides the preset density (ladder retry)
        outdir = f"{REMOTE_OUT}/{Path(song_name).stem}__{diff}"
        p = {**presets[diff], "nps": round(nps[0], 2)} if nps else presets[diff]
        lines.append(f"run {shlex.quote(REMOTE_INBOX + '/' + song_name)} {diff} {shlex.quote(outdir)} {shlex.quote(flags_for(p, seed, diff, downmap))} &")
    lines.append("wait")
    lines.append(f"echo DONE_WAVE_{n}")
    script = Path(tempfile.gettempdir()) / f"bw_batch_wave{n}.sh"
    script.write_text("\n".join(lines) + "\n")
    return script


def deploy_and_run_wave(host: str, container: str, n: int, script: Path) -> None:
    r = subprocess.run(["scp", "-q", str(script), f"{host}:/tmp/bw_batch_wave{n}.sh"], capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f"scp wave failed: {r.stderr[-300:]}")
    ssh(host, f"docker cp /tmp/bw_batch_wave{n}.sh {container}:/data/bw_batch_wave{n}.sh")
    dexec(host, container, f"bash /data/bw_batch_wave{n}.sh")


def merge_song(orig: Path, outdir: Path, dats: dict[str, Path], author: str, web_cover: bool) -> str:
    """`dats` maps difficulty -> its generated map folder (Info.dat, <diff>.dat, song.egg)."""
    from beat_sim.meta import song_meta, extract_cover
    from beat_weaver.model.exporter import jump_settings
    # The .dat beats are in the generator's detected BPM, so Info.dat must carry that BPM, and the
    # generator already transcoded the audio to ogg (Beat Saber can't play an mp3 renamed .egg).
    infos = {d: json.loads((p / "Info.dat").read_text()) for d, p in dats.items()}
    bpms = {d: i["_beatsPerMinute"] for d, i in infos.items()}
    if len(set(bpms.values())) > 1:
        raise ValueError(f"difficulties detected different BPMs {bpms}; regenerate the odd one out")
    first = next(iter(dats))
    base = infos[first]
    outdir.mkdir(parents=True, exist_ok=True)
    shutil.copy(dats[first] / base["_songFilename"], outdir / "song.egg")
    for diff, p in dats.items():
        shutil.copy(p / f"{diff}.dat", outdir / f"{diff}.dat")
    base.update(song_meta(orig, mapper=author, web=web_cover))
    base["_songFilename"] = "song.egg"
    cover_name, cover_src = extract_cover(orig, outdir, None, base["_songName"], base["_songAuthorName"], web=web_cover)
    base["_coverImageFilename"] = cover_name
    # NJS and offset from the exporter: offset 0 left slow and mid-tempo Easy/Normal at 1.5-1.75 s reaction time
    beatmaps = [{"_difficulty": d, "_difficultyRank": RANK[d], "_beatmapFilename": f"{d}.dat",
                 "_noteJumpMovementSpeed": (js := jump_settings(d, base["_beatsPerMinute"]))[0],
                 "_noteJumpStartBeatOffset": js[1]}
                for d in ALL_DIFFS if d in dats]
    base["_difficultyBeatmapSets"] = [{"_beatmapCharacteristicName": "Standard", "_difficultyBeatmaps": beatmaps}]
    (outdir / "Info.dat").write_text(json.dumps(base, indent=2))
    order = [d for d in ALL_DIFFS if d in dats]
    nps = {d: _nps(outdir / f"{d}.dat", base["_beatsPerMinute"]) for d in order}
    for lo, hi in zip(order, order[1:]):
        if nps[hi] < LADDER_RATIO * nps[lo]:
            print(f"  [ladder] {outdir.name}: {hi} {nps[hi]:.2f} nps is not {LADDER_RATIO}x {lo} {nps[lo]:.2f}")
    return cover_src or "no cover"


LADDER_RATIO = 1.10  # same as beat_sim.mapper: each level at least this much denser than the one below


def _nps(dat: Path, bpm: float) -> float:
    beats = [n["_time"] for n in json.loads(dat.read_text())["_notes"]]
    span = (max(beats) - min(beats)) * 60 / bpm if beats else 0
    return len(beats) / span if span else 0.0


# Remote: notes per second of each song's Expert, Expert+ and Hard (same span measure as _nps), as JSON
# {stem: [e, e+, hard or null]}.
_NPS_PY = """
import json, pathlib, sys
def nps(folder, diff):
    bpm = json.loads((folder / "Info.dat").read_text())["_beatsPerMinute"]
    t = [n["_time"] for n in json.loads((folder / (diff + ".dat")).read_text())["_notes"] if n["_type"] in (0, 1)]
    return len(t) / ((max(t) - min(t)) * 60 / bpm)
out = {}
for d in pathlib.Path(sys.argv[1]).glob("*__Expert"):
    stem = d.name[:-len("__Expert")]
    try:
        r = [nps(d, "Expert"), nps(d.with_name(stem + "__ExpertPlus"), "ExpertPlus")]
    except (OSError, ValueError, KeyError):
        continue
    try:
        r.append(nps(d.with_name(stem + "__Hard"), "Hard"))
    except (OSError, ValueError, KeyError):
        r.append(None)
    out[stem] = r
print(json.dumps(out))
"""


def ladder_retry(args, files: list[Path], presets: dict, downmap: tuple[str, ...], rounds: int = 4) -> None:
    """Regenerate Expert wherever it isn't LADDER_RATIO apart from Hard below and Expert+ above.

    On slow songs the onset head finds few candidate hits, so Expert overshoots its target and Expert+ can't
    go higher (4.6, 5.4 and 6.2 nps targets gave the same 1041 notes). Moving Expert is the only lever.
    It aims at the geometric middle of Hard and Expert+ (or 5% under Expert+'s limit when Hard isn't there),
    scaling the old target by how far Expert has to move. Density jumps unevenly with the target (one song went
    3.00 -> 4.22 nps on a small change), so once a target has been too sparse and another too dense it bisects.
    """
    target = {f.stem: presets["Expert"]["nps"] for f in files}
    lo: dict[str, float] = {}  # highest target that came out too sparse (Expert under LADDER_RATIO x Hard)
    hi: dict[str, float] = {}  # lowest target that came out too dense (Expert+ under LADDER_RATIO x Expert)
    for rnd in range(1, rounds + 1):
        nps = json.loads(dexec(args.bw_host, args.container,
                               f"python -c {shlex.quote(_NPS_PY)} {shlex.quote(REMOTE_OUT)}", check=False) or "{}")
        by_stem = {f.stem: f.name for f in files}
        redo = []
        for s, (e, eplus, hard) in nps.items():
            if s not in by_stem:
                continue
            if eplus < LADDER_RATIO * e:
                hi[s] = min(hi.get(s, target[s]), target[s])
            elif hard and e < LADDER_RATIO * hard:
                lo[s] = max(lo.get(s, target[s]), target[s])
            else:
                continue
            if s in lo and s in hi:
                t = (lo[s] + hi[s]) / 2
            else:
                goal = (hard * eplus) ** 0.5 if hard and eplus >= LADDER_RATIO ** 2 * hard else eplus / (LADDER_RATIO * 1.05)
                t = target[s] * goal / e
            redo.append((by_stem[s], "Expert", t))
        if not redo:
            return
        print(f"[ladder] round {rnd}: regenerating Expert for {len(redo)} song(s)", flush=True)
        for i in range(0, len(redo), args.concurrency):
            wave = redo[i:i + args.concurrency]
            for name, _, t in wave:
                target[Path(name).stem] = t
            script = build_wave(900 + rnd * 10 + i // args.concurrency, wave, presets, args.checkpoint, args.seed, downmap)
            deploy_and_run_wave(args.bw_host, args.container, 900 + rnd * 10 + i // args.concurrency, script)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("songs", nargs="+", help="audio file(s) or folder(s) of audio files")
    ap.add_argument("--bw-host", default="", help="ssh target of the GPU host running the container (user@host)")
    ap.add_argument("--checkpoint", default="models/current", help="checkpoint folder in the container (default models/current)")
    ap.add_argument("--container", default=CONTAINER, help="container name (default beat-weaver)")
    ap.add_argument("--difficulties", nargs="+", choices=ALL_DIFFS, default=list(ALL_DIFFS))
    ap.add_argument("--concurrency", type=int, default=16, help="songs generated in parallel per wave (default 16)")
    ap.add_argument("--out", type=Path, default=Path.home() / "Downloads" / "maps", help="output dir, one folder per song")
    ap.add_argument("--author", default=None, help="level author for Info.dat (default from beat-weaver config, else 'BeatWeaver AI')")
    ap.add_argument("--seed", type=int, default=1, help="generation seed (default 1)")
    ap.add_argument("--no-web-cover", action="store_true", help="don't look covers up on YouTube/iTunes")
    ap.add_argument("--no-sync", action="store_true", help="skip pushing repo code to the container")
    ap.add_argument("--keep-remote", action="store_true", help="leave generated files in the container")
    ap.add_argument("--no-validate", action="store_true", help="skip the post-merge structure/playability gate")
    ap.add_argument("--dry", action="store_true", help="list songs and presets, touch nothing remote")
    args = ap.parse_args()

    if not args.dry and not args.bw_host:
        raise SystemExit("Set --bw-host (the ssh target running the beat-weaver container) or use --dry.")

    files: list[Path] = []
    for s in args.songs:
        p = Path(s).expanduser()
        if p.is_dir():
            files += [f for f in sorted(p.iterdir()) if f.is_file() and f.suffix.lower() in AUDIO_EXTS]
        elif p.is_file():
            files.append(p)
        else:
            print(f"warning: not found: {p}", file=sys.stderr)
    # de-dup by name, keep first; warn on name collisions (they'd collide in the container inbox)
    seen: dict[str, Path] = {}
    for f in files:
        if f.name in seen:
            print(f"warning: duplicate file name {f.name!r}; keeping {seen[f.name]} and ignoring {f}", file=sys.stderr)
        else:
            seen[f.name] = f
    files = list(seen.values())
    files.sort(key=lambda p: p.name.lower())
    if not files:
        raise SystemExit("no audio files found")

    presets = load_presets(args.bw_host, args.container, args.checkpoint, remote=not args.dry and bool(args.bw_host))
    preset_desc = ", ".join(f"{d} {presets[d]['nps']} nps" for d in args.difficulties)
    print(f"{len(files)} songs x {len(args.difficulties)} difficulties ({preset_desc}) -> {args.out}")
    if args.dry:
        for f in files:
            print(f"  {f.name}")
        return

    if not args.no_sync:
        print("syncing repo code to container ...")
        sync_code(args.bw_host, args.container)
    print(f"uploading {len(files)} songs ...")
    upload_audio(args.bw_host, args.container, files)

    # With Hard requested, Easy and Normal come from the Hard job (--downmap) instead of being generated alone.
    downmap = tuple(d for d in ("Easy", "Normal") if d in args.difficulties) if "Hard" in args.difficulties else ()
    tasks: list[tuple[str, str]] = []
    have = set(dexec(args.bw_host, args.container, f"cd {REMOTE_OUT} && ls -d */*.dat 2>/dev/null", check=False).split("\n"))
    for f in files:
        for d in (d for d in args.difficulties if d not in downmap):
            if f"{f.stem}__{d}/{d}.dat" in have:
                print(f"  [have] {f.stem}__{d}")
            else:
                tasks.append((f.name, d))
    waves = [tasks[i:i + args.concurrency] for i in range(0, len(tasks), args.concurrency)] or [[]]
    print(f"{len(tasks)} generations in {len(waves)} wave(s) of up to {args.concurrency}")
    for i, wave in enumerate(waves, 1):
        if not wave:
            continue
        t0 = time.time()
        print(f"wave {i}/{len(waves)} ({len(wave)} jobs) ...", flush=True)
        script = build_wave(i, wave, presets, args.checkpoint, args.seed, downmap)
        deploy_and_run_wave(args.bw_host, args.container, i, script)
        print(f"  wave {i} done in {time.time()-t0:.0f}s", flush=True)

    if "Expert" in args.difficulties and "ExpertPlus" in args.difficulties:
        ladder_retry(args, files, presets, downmap)

    # fetch results into a temp dir, then merge per song
    fetch_tmp = Path(tempfile.mkdtemp(prefix="bw_batch_"))
    r = subprocess.run(["ssh", args.bw_host, f"docker exec {args.container} tar -cf - -C {REMOTE_DATA} bw_batch_out"],
                       capture_output=True)
    if r.returncode:
        raise RuntimeError(f"fetch failed: {r.stderr[-300:]!r}")
    with tarfile.open(fileobj=io.BytesIO(r.stdout)) as tf:
        tf.extractall(fetch_tmp, filter="data")
    fetch_root = fetch_tmp / "bw_batch_out"
    print(f"fetched {sum(1 for _ in fetch_root.rglob('*.dat')) if fetch_root.exists() else 0} .dat files")

    # merge per song
    ok = miss = 0
    for f in files:
        dats = {}
        for d in args.difficulties:
            p = fetch_root / f"{f.stem}__{'Hard' if d in downmap else d}"
            if (p / f"{d}.dat").exists():
                dats[d] = p
            elif (log := fetch_root / f"{p.name}.log").exists():
                print(f"  [fail] {p.name}: " + (log.read_text().strip().splitlines() or ["(empty log)"])[-1])
        if not dats:
            miss += 1
            print(f"[skip] {f.name}: no difficulties generated")
            continue
        outdir = args.out / f.stem
        try:
            cover_src = merge_song(f, outdir, dats, args.author, web_cover=not args.no_web_cover)
        except Exception as e:  # keep going; report the failure
            print(f"[warn] {f.name}: merge failed: {type(e).__name__} {e}")
            continue
        if len(dats) < len(args.difficulties):
            print(f"  [note] {f.name}: {len(dats)}/{len(args.difficulties)} difficulties (missing: "
                  f"{[d for d in args.difficulties if d not in dats]})")
        ok += 1
        print(f"[ok] {f.stem}  ({len(dats)} diffs, cover: {cover_src})")

    shutil.rmtree(fetch_tmp, ignore_errors=True)
    if not args.keep_remote:
        dexec(args.bw_host, args.container, f"rm -rf {REMOTE_INBOX} {REMOTE_OUT}")

    # post-merge structure + playability gate (no torch; runs here, not in the container)
    if not args.no_validate:
        written = [args.out / f.stem for f in files if (args.out / f.stem / "Info.dat").exists()]
        if written:
            from beat_weaver.model.map_check import check_paths
            reports = check_paths(written)
            n_struct = sum(len(r.structure_issues) for r in reports)
            n_play = sum(len(r.playability_issues) for r in reports)
            print(f"\n[validate] {len(reports)} folder(s): {n_struct} structural issue(s), {n_play} playability warning(s)")
            for r in reports:
                if r.structure_issues:
                    for i in r.structure_issues:
                        print(f"  [struct] {Path(r.folder).name}/{i.difficulty}: {i.message} ({i.count})")
            if n_struct:
                print("[validate] structural issues found — fix before shipping to CustomLevels")

    print(f"\n{ok} map(s) written to {args.out}, {miss} failed. Copy any folder to Beat Saber_Data/CustomLevels/.")


if __name__ == "__main__":
    main()
