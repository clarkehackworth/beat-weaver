"""Command-line interface for Beat Weaver data pipeline."""

import argparse
import collections
import os
import sys
import logging
from pathlib import Path


def cmd_download(args: argparse.Namespace) -> None:
    from beat_weaver.sources.beatsaver import BeatSaverClient

    client = BeatSaverClient()

    # Count already-downloaded maps for the summary
    dest = Path(args.output)
    existing = sum(1 for d in dest.iterdir() if d.is_dir()) if dest.exists() else 0

    downloaded = client.download_maps(
        dest_dir=dest,
        min_score=args.min_score,
        min_upvotes=args.min_upvotes,
        max_maps=args.max_maps,
        workers=args.workers,
    )
    newly = len(downloaded) - existing if len(downloaded) > existing else len(downloaded)
    limit = f" (limit: {args.max_maps})" if args.max_maps > 0 else " (no limit)"
    print(f"Done: {len(downloaded)} maps in {args.output} ({newly} new){limit}")


def cmd_extract_official(args: argparse.Namespace) -> None:
    from beat_weaver.sources.unity_extractor import extract_official_maps

    bundles_dir = (
        Path(args.beat_saber)
        / "Beat Saber_Data"
        / "StreamingAssets"
        / "aa"
        / "StandaloneWindows64"
    )
    extracted = extract_official_maps(bundles_dir, Path(args.output), beat_saber_path=Path(args.beat_saber))
    print(f"Extracted {len(extracted)} map folders to {args.output}")


def cmd_build_manifest(args: argparse.Namespace) -> None:
    from beat_weaver.model.audio import build_audio_manifest, save_manifest

    raw_dirs = [Path(d) for d in args.input]
    manifest = build_audio_manifest(raw_dirs)
    save_manifest(manifest, Path(args.output))
    print(f"Built audio manifest: {len(manifest)} entries -> {args.output}")


def _detect_source(map_folder: Path, input_root: Path) -> str:
    """Detect map source from its path relative to the input root."""
    try:
        rel = map_folder.relative_to(input_root)
        parts = [p.lower() for p in rel.parts]
    except ValueError:
        parts = []
    if "official" in parts:
        return "official"
    if "beatsaver" in parts:
        return "beatsaver"
    return "local_custom"


def _process_single_folder(
    map_folder: Path, source: str,
) -> list:
    """Process a single map folder (top-level for pickling by ProcessPoolExecutor)."""
    from beat_weaver.pipeline.processor import process_map_folder
    from beat_weaver.sources.beatsaver import load_beatsaver_meta

    beatmaps = process_map_folder(map_folder, source=source, source_id=map_folder.name)

    # Inject BeatSaver score/votes into metadata if available
    if source == "beatsaver":
        meta = load_beatsaver_meta(map_folder)
        if meta:
            stats = meta.get("stats", {})
            score = stats.get("score")
            upvotes = stats.get("upvotes")
            downvotes = stats.get("downvotes")
            for bm in beatmaps:
                if score is not None:
                    bm.metadata.score = score
                if upvotes is not None:
                    bm.metadata.upvotes = upvotes
                if downvotes is not None:
                    bm.metadata.downvotes = downvotes

    return beatmaps


def _find_map_folders(root: Path) -> list[Path]:
    """Every folder under ``root`` that holds a map's info file, sorted.

    Map makers ship the file as ``Info.dat`` or ``info.dat`` (the parser and the
    audio manifest already accept both). Matching only ``Info.dat`` silently
    dropped ~23% of downloaded community maps and 42% of a hand-picked favourites
    set. Like ``Path.rglob`` this does not follow symlinked directories.
    """
    found = []
    for dirpath, _dirs, files in os.walk(root):
        if any(name.lower() == "info.dat" for name in files):
            found.append(Path(dirpath))
    return sorted(found)


def cmd_process(args: argparse.Namespace) -> None:
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from beat_weaver.storage.writer import write_parquet

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    # Collect all map folders up front
    folders = []
    for map_folder in _find_map_folders(input_dir):
        source = _detect_source(map_folder, input_dir)
        folders.append((map_folder, source))

    all_beatmaps = []
    with ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(_process_single_folder, folder, source): folder
            for folder, source in folders
        }
        for future in as_completed(futures):
            try:
                all_beatmaps.extend(future.result())
            except Exception:
                logging.getLogger(__name__).warning(
                    "Failed to process %s", futures[future], exc_info=True,
                )

    write_parquet(all_beatmaps, output_dir)
    print(f"Processed {len(all_beatmaps)} beatmaps to {output_dir}")


def cmd_train(args: argparse.Namespace) -> None:
    from beat_weaver.model.audio import build_audio_manifest, save_manifest, load_manifest
    from beat_weaver.model.config import ModelConfig
    from beat_weaver.model.dataset import BeatSaberDataset, warm_mel_cache
    from beat_weaver.model.training import train

    config = ModelConfig()
    if args.config:
        config = ModelConfig.load(Path(args.config))
    if args.epochs:
        config.max_epochs = args.epochs
    if args.batch_size:
        config.batch_size = args.batch_size

    manifest_path = Path(args.audio_manifest)
    if not manifest_path.exists():
        print(f"Audio manifest not found: {manifest_path}")
        return

    data_dir = Path(args.data)

    # Pre-compute mel spectrograms in parallel before training
    warm_mel_cache(data_dir, manifest_path, config)

    train_ds = BeatSaberDataset(data_dir, manifest_path, config, split="train")
    val_ds = BeatSaberDataset(data_dir, manifest_path, config, split="val")

    print(f"Training: {len(train_ds)} samples, Validation: {len(val_ds)} samples")

    resume = Path(args.resume) if args.resume else None
    init = Path(args.init_from) if args.init_from else None
    probe_audio = Path(args.probe_audio) if args.probe_audio else None
    best_ckpt = train(config, train_ds, val_ds, Path(args.output), resume_from=resume, init_from=init,
                      probe_audio=probe_audio)
    print(f"Training complete. Best checkpoint: {best_ckpt}")


def cmd_generate(args: argparse.Namespace) -> None:
    import torch
    from beat_weaver.model.audio import (
        beat_align_spectrogram, compute_mel_spectrogram, compute_mel_with_onset,
        detect_bpm, load_audio,
    )
    from beat_weaver.model.config import ModelConfig
    from beat_weaver.model.exporter import export_notes
    from beat_weaver.model.inference import generate_full_song
    from beat_weaver.model.transformer import BeatWeaverModel

    ckpt_dir = Path(args.checkpoint)
    config = ModelConfig.load(ckpt_dir / "config.json")

    model = BeatWeaverModel(config)
    model.load_state_dict(
        torch.load(ckpt_dir / "model.pt", map_location="cpu", weights_only=True),
    )
    model.eval()
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    audio, sr = load_audio(Path(args.audio), sr=config.sample_rate)

    bpm = args.bpm
    if bpm is None:
        bpm = detect_bpm(audio, sr=sr)
        print(f"Auto-detected BPM: {bpm:.1f}")

    if config.use_onset_features:
        mel = compute_mel_with_onset(audio, sr=sr, n_mels=config.n_mels,
                                     n_fft=config.n_fft, hop_length=config.hop_length)
    else:
        mel = compute_mel_spectrogram(audio, sr=sr, n_mels=config.n_mels,
                                      n_fft=config.n_fft, hop_length=config.hop_length)

    # Beat-align to match training data preprocessing
    mel = beat_align_spectrogram(mel, sr=sr, hop_length=config.hop_length, bpm=bpm)
    mel_tensor = torch.from_numpy(mel)

    from beat_weaver.model.inference import last_music_second

    notes = generate_full_song(
        model, mel_tensor, args.difficulty, config, bpm,
        temperature=args.temperature,
        seed=args.seed,
        onset_guidance=args.onset_guidance,
        two_stage=args.two_stage, notes_per_beat=args.notes_per_beat, min_gap_seconds=args.min_gap,
        dynamics=args.dynamics, dynamics_range=tuple(args.dynamics_range), lead_in_seconds=args.lead_in, flow_penalty=args.flow_penalty,
        end_seconds=last_music_second(audio, sr),
        notes_per_second=args.notes_per_second, stream_bias=args.stream_bias, doubles_bias=args.doubles_bias, parallel_bias=args.parallel_bias, parallel_streak=args.parallel_streak, stack_bias=args.stack_bias, stack_target=args.stack_rate / 100, bomb_bias=args.bomb_bias, bomb_gap_bias=args.bomb_gap_bias, bomb_row_bias=args.bomb_row_bias, diagonal_bias=args.diagonal_bias, hand_balance=args.hand_balance, crossover_penalty=args.crossover_penalty, vision_penalty=args.vision_penalty, vision_max_run=args.vision_max_run, column_max_run=args.column_max_run, raised_max_run=args.raised_max_run, reversal_penalty=args.reversal_penalty, beat_focus=args.beat_focus, block_near_converging=not args.allow_near_converging, streams=args.streams, stacks=args.stacks,
        reset_seconds=args.reset_gap, max_doubles=args.max_doubles, max_ebpm=args.max_ebpm, beat_floor=args.beat_floor, max_run=args.max_run,
        drift_bias=args.drift_bias, momentum_bias=args.momentum_bias, flow_window=args.flow_window,
        hand_bias=args.hand_bias, precision_prior=args.precision_prior,
        extras_out=(model_extras := []),
    )
    n_windows = max(1, (mel.shape[1] - 1) // (config.max_audio_len - min(config.max_audio_len // 4, 1024)) + 1) if mel.shape[1] > config.max_audio_len else 1

    # Rule-based shaping after decoding (bsmg guidelines): repeated sections, emphasis, then downmapping.
    from beat_weaver.model import shaping
    strength, hop_s = shaping.onset_strength(audio, sr)
    if args.repeats > 0:
        pairs = shaping.find_repeats(mel.T, 60.0 / bpm / 16, bpm, bars=args.repeat_bars, min_sim=args.repeats)
        notes = shaping.repeat_sections(notes, bpm, pairs, bars=args.repeat_bars)
        print(f"Repeated sections: {len(pairs)} block(s) of {args.repeat_bars} bars copied")
    if args.vocal_snap > 0:
        lead, peaks, _ = shaping.vocal_signal(audio, sr)
        before = [n.time_seconds for n in notes]
        notes = shaping.vocal_snap(notes, bpm, lead, peaks, hop_s, min_lead=args.vocal_snap)
        moved = sum(a != b.time_seconds for a, b in zip(before, notes))
        print(f"Vocal snap: {moved} hit(s) moved onto harmonic onsets ({(lead >= args.vocal_snap).mean():.0%} of song vocal-led)")
    if args.emphasis > 0 and args.difficulty not in ("Easy", "Normal"):
        before = len(notes)
        notes = shaping.emphasize(notes, bpm, strength, hop_s, top_frac=args.emphasis, sustain_frac=args.sliders,
                                  difficulty=args.difficulty)
        print(f"Emphasis: {len(notes) - before} note(s) added as stacks/sliders")
    by_diff = {args.difficulty: notes}
    if args.downmap:
        by_diff = shaping.downmap_ladder(notes, bpm, args.difficulty, [*args.downmap, args.difficulty],
                                         strength=strength, hop_s=hop_s)
    by_diff = {d: shaping.gate_patterns(ns, bpm, d) for d, ns in by_diff.items()}
    if args.reset_gap > 0 or args.max_ebpm > 0:
        from beat_weaver.model.inference import drop_fast_same_way
        swing = 60.0 / args.max_ebpm if args.max_ebpm > 0 else 0.0
        by_diff = {d: drop_fast_same_way(ns, args.reset_gap, swing) for d, ns in by_diff.items()}
    if args.downmap:
        # Downmapped levels inherit the source's swing speed, so cap each at its own limit (10% under the validator's)
        from beat_weaver.model.inference import drop_fast_same_way
        from beat_weaver.model.map_check import MAX_EBPM
        by_diff = {d: ns if d == args.difficulty else drop_fast_same_way(ns, args.reset_gap, 60.0 / (MAX_EBPM[d] * 0.9))
                   for d, ns in by_diff.items()}

    song_name = Path(args.audio).stem
    output = Path(args.output) if args.output else Path(f"output/{song_name}")
    events = None
    if not args.no_lights:
        from beat_weaver.model.lighting import light_events
        events = {d: light_events(ns, bpm, strength, hop_s) for d, ns in by_diff.items()}
    extras = None
    if args.bombs or args.walls or model_extras:
        # Model-generated bombs/walls (vocab 495 checkpoints) go on the generated difficulty; rule extras on all
        from beat_weaver.model.exporter import extras_to_v2
        extras = {d: shaping.bombs_and_walls(ns, bpm, bombs=args.bombs, walls=args.walls) for d, ns in by_diff.items()}
        mb, mw = extras_to_v2(model_extras)
        n_mb, n_mw = len(mb), len(mw)
        if model_extras:
            ln = [w["_duration"] for w in mw]
            at = collections.Counter(b["_time"] for b in mb)
            note_beats = {n.beat for n in by_diff[args.difficulty]}
            print(f"Model bombs: {len(mb)}, {sum(at[b['_time']] > 1 for b in mb)} in groups, "
                  f"{sum(b['_time'] not in note_beats for b in mb)} on beats with no note")
            print(f"Model walls: {sum(d >= 1 for d in ln)}/{len(ln)} at least 1 beat, "
                  f"{sum(d >= 2 for d in ln)} at least 2 beats, {sum(w['_width'] >= 3 for w in mw)} 3+ wide")
        mb, mw = shaping.limit_extras(mb, mw, by_diff[args.difficulty], bpm, walls_per_bar=args.max_walls_per_bar)
        if model_extras:
            print(f"Model extras kept: {len(mb)}/{n_mb} bomb(s), {len(mw)}/{n_mw} wall(s)")
        extras[args.difficulty] = (extras[args.difficulty][0] + mb, extras[args.difficulty][1] + mw)
    if extras:
        extras = {d: (b, shaping.drop_overlapping_walls(w)) for d, (b, w) in extras.items()}
        print("Extras: " + ", ".join(f"{d} {len(b)} bomb(s), {len(w)} wall(s)" for d, (b, w) in extras.items()) + f" ({len(model_extras)} from the model)")
    export_notes(by_diff, bpm, song_name, Path(args.audio), output, args.difficulty, events=events, extras=extras)
    counts = ", ".join(f"{d} {len(n)}" for d, n in by_diff.items())
    print(f"Generated map: {output} ({n_windows} window(s), notes: {counts})")


def cmd_validate(args: argparse.Namespace) -> int:
    """Validate Beat Saber map folders (structure + playability, no torch).

    Exits non-zero if any folder has structural issues or playability
    warnings exceed --max-issues (default: structural only fail the build).
    """
    from beat_weaver.model.map_check import check_paths
    reports = check_paths(args.maps)
    for r in reports:
        print(r.summary())
    n_struct = sum(len(r.structure_issues) for r in reports)
    n_play = sum(len(r.playability_issues) for r in reports)
    print(f"\n{len(reports)} folders: {n_struct} structural issues, {n_play} playability warnings")
    if n_struct or n_play > args.max_issues:
        print("VALIDATE FAILED")
        return 1
    print("VALIDATE PASSED")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> None:
    import json as _json
    import torch
    from beat_weaver.model.audio import (
        compute_mel_spectrogram, load_audio, load_manifest, beat_align_spectrogram,
    )
    from beat_weaver.model.config import ModelConfig
    from beat_weaver.model.dataset import BeatSaberDataset
    from beat_weaver.model.evaluate import evaluate_map
    from beat_weaver.model.inference import generate
    from beat_weaver.model.tokenizer import decode_tokens
    from beat_weaver.model.transformer import BeatWeaverModel

    ckpt_dir = Path(args.checkpoint)
    config = ModelConfig.load(ckpt_dir / "config.json")

    model = BeatWeaverModel(config)
    model.load_state_dict(
        torch.load(ckpt_dir / "model.pt", map_location="cpu", weights_only=True),
    )
    model.eval()

    test_ds = BeatSaberDataset(Path(args.data), Path(args.audio_manifest), config, split="test")
    results = []

    for i in range(len(test_ds)):
        mel, tokens, mask = test_ds[i]
        sample = test_ds.samples[i]

        gen_tokens = generate(model, mel, sample["difficulty"], config, temperature=0)
        gen_notes = decode_tokens(gen_tokens, sample["bpm"])

        from beat_weaver.schemas.normalized import Note
        ref_notes = [
            Note(beat=n["beat"], time_seconds=n["time_seconds"],
                 x=n["x"], y=n["y"], color=n["color"], cut_direction=n["cut_direction"])
            for n in sample["notes"]
        ]

        metrics = evaluate_map(gen_notes, ref_notes, sample["bpm"])
        metrics["song_hash"] = sample["song_hash"]
        metrics["difficulty"] = sample["difficulty"]
        results.append(metrics)

    output_path = Path(args.output) if args.output else Path("evaluation_results.json")
    output_path.write_text(_json.dumps(results, indent=2), encoding="utf-8")
    print(f"Evaluated {len(results)} maps. Results: {output_path}")


def cmd_run(args: argparse.Namespace) -> None:
    from beat_weaver.pipeline.batch import PipelineConfig, run_pipeline

    config = PipelineConfig(
        beat_saber_path=Path(args.beat_saber),
        raw_dir=Path(args.raw_dir),
        output_dir=Path(args.output),
        cache_dir=Path(args.cache_dir),
        include_local=not args.no_local,
        include_beatsaver=not args.no_beatsaver,
        include_official=not args.no_official,
        min_score=args.min_score,
        max_beatsaver_maps=args.max_maps,
    )
    result = run_pipeline(config)
    print(f"Done: {result.total_songs} songs, {result.total_beatmaps} beatmaps, {result.total_notes} notes")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="beat-weaver",
        description="Beat Weaver training data pipeline",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable debug logging"
    )
    sub = parser.add_subparsers(dest="command")

    # download
    dl = sub.add_parser("download", help="Download custom maps from BeatSaver")
    dl.add_argument("--min-score", type=float, default=0.75,
                     help="Minimum rating score 0.0-1.0 (default: 0.75)")
    dl.add_argument("--min-upvotes", type=int, default=5,
                     help="Minimum upvotes (default: 5)")
    dl.add_argument("--max-maps", type=int, default=0,
                     help="Max maps to download (default: 0 = unlimited)")
    dl.add_argument("--workers", type=int, default=8,
                     help="Parallel download threads (default: 8)")
    dl.add_argument("--output", default="data/raw/beatsaver")

    # extract-official
    ext = sub.add_parser("extract-official", help="Extract maps from Unity bundles")
    ext.add_argument(
        "--beat-saber",
        default=r"C:\Program Files (x86)\Steam\steamapps\common\Beat Saber",
    )
    ext.add_argument("--output", default="data/raw/official")

    # build-manifest
    bm = sub.add_parser("build-manifest", help="Build audio manifest from raw map folders")
    bm.add_argument("--input", nargs="+", default=["data/raw"],
                     help="Raw map directories to scan (default: data/raw)")
    bm.add_argument("--output", default="data/audio_manifest.json",
                     help="Output manifest JSON path")

    # process
    proc = sub.add_parser("process", help="Normalize raw maps into Parquet")
    proc.add_argument("--input", default="data/raw")
    proc.add_argument("--output", default="data/processed")

    # run (full pipeline)
    run = sub.add_parser("run", help="Run full pipeline")
    run.add_argument(
        "--beat-saber",
        default=r"C:\Program Files (x86)\Steam\steamapps\common\Beat Saber",
    )
    run.add_argument("--raw-dir", default="data/raw")
    run.add_argument("--output", default="data/processed")
    run.add_argument("--cache-dir", default="data/cache")
    run.add_argument("--min-score", type=float, default=0.7)
    run.add_argument("--max-maps", type=int, default=100)
    run.add_argument("--no-local", action="store_true")
    run.add_argument("--no-beatsaver", action="store_true")
    run.add_argument("--no-official", action="store_true")

    # train
    tr = sub.add_parser("train", help="Train the ML model")
    tr.add_argument("--data", default="data/processed", help="Processed data directory")
    tr.add_argument("--audio-manifest", required=True, help="Audio manifest JSON path")
    tr.add_argument("--output", default="output/training", help="Output directory for checkpoints")
    tr.add_argument("--config", default=None, help="Optional JSON config override")
    tr.add_argument("--epochs", type=int, default=None, help="Max epochs")
    tr.add_argument("--batch-size", type=int, default=None, help="Batch size")
    tr.add_argument("--resume", default=None, help="Resume from checkpoint directory")
    tr.add_argument("--init-from", default=None,
                    help="Load only model weights from this checkpoint (fresh optimizer/schedule)")
    tr.add_argument("--probe-audio", default=None,
                    help="Song to generate after every epoch (fixed flags) and log doubles/reversal/left/nps shares: decoder drift that val loss misses")

    # generate
    gen = sub.add_parser("generate", help="Generate a Beat Saber map from audio")
    gen.add_argument("--checkpoint", required=True, help="Model checkpoint directory")
    gen.add_argument("--audio", required=True, help="Input audio file")
    gen.add_argument("--difficulty", default="Expert",
                     choices=["Easy", "Normal", "Hard", "Expert", "ExpertPlus"])
    gen.add_argument("--output", default=None, help="Output map folder")
    gen.add_argument("--bpm", type=float, default=None,
                     help="Song BPM (auto-detected from audio if not provided)")
    gen.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    gen.add_argument("--seed", type=int, default=None, help="Random seed")
    gen.add_argument("--onset-guidance", type=float, default=0.0,
                     help="Weight on the encoder's per-frame onset logit added to note-position "
                          "logits, tying note timing to the audio (0 = off)")
    gen.add_argument("--two-stage", action="store_true",
                     help="Note timing from the encoder's onset head, placement from the decoder")
    gen.add_argument("--notes-per-beat", type=float, default=None,
                     help="Two-stage density override (default: onset head's own expectation)")
    gen.add_argument("--min-gap", type=float, default=0.2,
                     help="Two-stage: minimum seconds between consecutive notes")
    gen.add_argument("--notes-per-second", type=float, default=None,
                     help="Two-stage density target in notes/second (overrides --notes-per-beat; favourites Expert ~3.8, Expert+ ~4.8)")
    gen.add_argument("--stream-bias", type=float, default=1.0,
                     help="Two-stage: keep a run going while the next frame's onset prob >= this x the bar peak (1 = off, 0.6 = long streams)")
    gen.add_argument("--stack-rate", type=float, default=1.3,
                     help="With --stack-bias: stacked notes per 100 to aim for (real Expert median 1.3, p75 3.3)")
    gen.add_argument("--stack-bias", type=float, default=0.0,
                     help="With --stacks: logit bonus for continuing a hand's note as a stack, until --stack-rate")
    gen.add_argument("--doubles-bias", type=float, default=0.0,
                     help="Logit bonus for a right-hand note right after a left-hand note (0 = off; ~1.5 reaches favourites' doubles rate)")
    gen.add_argument("--parallel-bias", type=float, default=0.0,
                     help="Logit bonus for a right-hand note beside the left one (same row, 1-2 columns right) with the same cut (0 = off)")
    gen.add_argument("--parallel-streak", type=float, default=0.0,
                     help="Extra parallel bonus per parallel instant already in a row (up to 3), for longer in-sync stretches (0 = off)")
    gen.add_argument("--vision-penalty", type=float, default=3.0,
                     help="Logit penalty on notes in the middle-row centre cells (vision blocks); favourites keep ~2%% there")
    gen.add_argument("--vision-max-run", type=int, default=2,
                     help="Max consecutive hits containing a centre-cell note before those cells are masked (0 = no cap)")
    gen.add_argument("--reversal-penalty", type=float, default=0.0,
                     help="Logit penalty on a 2+ cell cut reversed within half a beat of the same hand's last note, while over 43%% of such fast swings are big reversals (real Expert median; Expert+ 53%%). Try 3-6")
    gen.add_argument("--column-max-run", type=int, default=0,
                     help="Max consecutive notes one hand puts in the same column before it must move (0 = no cap; 4 tames long up/down straights)")
    gen.add_argument("--raised-max-run", type=int, default=0,
                     help="Max consecutive notes one hand puts in the upper two rows before it drops to the bottom row (0 = no cap; real Expert median longest run is 5)")
    gen.add_argument("--reset-gap", type=float, default=0.5,
                     help="Seconds: a hand may not cut the same way (within 45 degrees, or both up / both down) again this soon (0 = off)")
    gen.add_argument("--max-doubles", type=float, default=1.0,
                     help="Cap on the fraction of hits that use both hands (1.0 = no cap)")
    gen.add_argument("--max-ebpm", type=float, default=0.0,
                     help="Per-hand effective BPM cap: 60 / seconds between one hand's swings (0 = off; favourites p90: Easy 140, Normal 240, Hard 280, Expert 310, Expert+ 385)")
    gen.add_argument("--beat-floor", type=float, default=0.0,
                     help="Two-stage: a beat whose onset peak is under this fraction of its bar's peak gets no note (0 = every beat gets one)")
    gen.add_argument("--stacks", action="store_true",
                     help="Let a hand put several same-cut notes at one instant (stacks, towers, windows). Only with a "
                          "checkpoint fine-tuned on stack tokens (run 5+); older ones put random mass there")
    gen.add_argument("--streams", type=float, default=0.0, metavar="FRAC",
                     help="Two-stage: play the schedule's longest dense chains (8+ hits within half a beat) as streams, one "
                          "alternating note per hit, until FRAC of notes are in one (real Expert median 0.10; 0 = off)")
    gen.add_argument("--max-run", type=int, default=0,
                     help="Two-stage: longest run of notes within half a beat of each other; the run then rests a whole beat (0 = no cap)")
    gen.add_argument("--precision-prior", type=float, default=0.0,
                     help="Two-stage: weight on the rhythm precision prior; off-beat 1/8 and 1/16 frames need a stronger "
                          "onset than 1/4 positions to be picked (0 = off, 1 = full table)")
    gen.add_argument("--beat-focus", action="store_true",
                     help="Two-stage: give a bar's extra notes to its strongest beats first, leaving rests between streams (Expert+)")
    gen.add_argument("--allow-near-converging", action="store_true",
                     help="Allow same-time notes on adjacent cells whose cuts are 135 deg apart with one aimed at the other (blocked by default)")
    gen.add_argument("--flow-penalty", type=float, default=2.0,
                     help="Logit penalty for a cut that breaks the saber's arc from the hand's last note (0 = off)")
    gen.add_argument("--hand-bias", type=float, default=0.0,
                     help="Logit weight on forehand/backhand placement comfort: forehand (down) cuts in the inner lanes, "
                          "backhand (up) in the outer lanes, no down-cuts on the top row (0 = off)")
    gen.add_argument("--drift-bias", type=float, default=0.0,
                     help="Logit bonus for a hand moving one lane over in the direction it last moved, reversing at the "
                          "edges (piano streams / jumps). Favourites move ~0.75 lanes per swing; 0 = off")
    gen.add_argument("--momentum-bias", type=float, default=0.0,
                     help="Logit bonus for cuts that continue the hand's rotation sense, flipping after 3 same-sense turns (0 = off)")
    gen.add_argument("--flow-window", type=float, default=1.5,
                     help="Beats after which a hand counts as reset; --flow-penalty fades to 0 over the last beat (0 = never fades)")
    gen.add_argument("--downmap", nargs="*", default=[], choices=["Easy", "Normal", "Hard", "Expert"], metavar="DIFF",
                     help="Also write these lower difficulties, derived from the generated one by thinning ~20%% per "
                          "step on the weakest beats (bsmg downmapping); Easy/Normal also lose crossovers and the top row")
    gen.add_argument("--repeats", type=float, default=0.0, metavar="SIM",
                     help="Copy the first occurrence's notes onto repeated song sections whose beat-synchronous mel "
                          "similarity is >= SIM (0.7 is a good start; odd repeats are mirrored; 0 = off)")
    gen.add_argument("--repeat-bars", type=int, default=4, help="Section block length in bars for --repeats")
    gen.add_argument("--emphasis", type=float, default=0.0, metavar="FRAC",
                     help="Turn the loudest FRAC of isolated hits into stacks (0.08 is a good start; skipped on Easy/Normal; 0 = off)")
    gen.add_argument("--sliders", type=float, default=0.0, metavar="FRAC",
                     help="With --emphasis: FRAC of isolated hits on sustained sounds become 2-dot sliders (0 = off)")
    gen.add_argument("--no-lights", action="store_true", help="Skip the rules-based lightshow (lighting.light_events)")
    gen.add_argument("--vocal-snap", type=float, default=0.0, metavar="LEAD",
                     help="Snap hits in vocal-led stretches (harmonic/percussive onset ratio >= LEAD, try 1.5) onto harmonic onsets (0 = off)")
    gen.add_argument("--bombs", action="store_true", help="Add bomb resets at 2/3 of same-cut rests (shaping.bombs_and_walls)")
    gen.add_argument("--walls", action="store_true", help="Add a full-width crouch wall in rests of 4+ beats")
    gen.add_argument("--crossover-penalty", type=float, default=0.0,
                     help="Logit penalty on a hand's notes in the other half while the last 16 notes exceed the training crossover rate (21%%; try 2-4)")
    gen.add_argument("--hand-balance", type=float, default=0.0,
                     help="Steer single notes to the hand that is behind (logit push scaled by the L/R imbalance; try 5-15)")
    gen.add_argument("--diagonal-bias", type=float, default=0.0,
                     help="Logit boost on diagonal cuts (the decoder over-uses plain up/down; try 0.5-1.5)")
    gen.add_argument("--bomb-gap-bias", type=float, default=0.0,
                     help="Logit boost on opening a bomb-only position between scheduled notes (model puts ~1%% there; try 2-4; needs an extras checkpoint and --two-stage)")
    gen.add_argument("--bomb-row-bias", type=float, default=0.0,
                     help="Logit boost on adding more bombs to a bomb-only position (rows/frames; the model gives ~8%% where real maps have ~38%%; try 1-3)")
    gen.add_argument("--bomb-bias", type=float, default=0.0,
                     help="Logit boost on BOMB tokens after each right-hand note (model under-emits bombs ~15x; try 2-4; needs an extras checkpoint)")
    gen.add_argument("--max-walls-per-bar", type=float, default=1.0, metavar="N",
                     help="Cap on model-made walls per bar after dropping short, 3-wide and note-covering ones (0 = none)")
    gen.add_argument("--dynamics", type=float, default=1.0,
                     help="Two-stage: how much each bar's note budget follows the music (0 = flat/staccato, 1 = full)")
    gen.add_argument("--dynamics-range", type=float, nargs=2, default=[0.0, 2.0], metavar=("LO", "HI"),
                     help="Two-stage: clamp each bar's budget to LO..HI x the mean density (0.7 1.3 evens out busy and sparse sections)")
    gen.add_argument("--lead-in", type=float, default=2.0,
                     help="Seconds at the start of the song with no notes")

    # validate (no torch — structure + playability gate for exported map folders)
    val = sub.add_parser("validate", help="Validate map folders (structure + playability, no torch)")
    val.add_argument("maps", nargs="+", help="Map folder(s) or parent dir(s) to validate")
    val.add_argument("--max-issues", type=int, default=100000,
                     help="Fail if total playability warnings exceed this (structural issues always fail)")

    # evaluate
    ev = sub.add_parser("evaluate", help="Evaluate model on test data")
    ev.add_argument("--checkpoint", required=True, help="Model checkpoint directory")
    ev.add_argument("--data", default="data/processed", help="Test data directory")
    ev.add_argument("--audio-manifest", required=True, help="Audio manifest JSON path")
    ev.add_argument("--output", default=None, help="Output JSON path for results")

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    commands = {
        "download": cmd_download,
        "extract-official": cmd_extract_official,
        "build-manifest": cmd_build_manifest,
        "process": cmd_process,
        "run": cmd_run,
        "train": cmd_train,
        "generate": cmd_generate,
        "validate": cmd_validate,
        "evaluate": cmd_evaluate,
    }

    handler = commands.get(args.command)
    if handler:
        sys.exit(handler(args) or 0)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
