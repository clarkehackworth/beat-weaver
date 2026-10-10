"""Command-line interface for Beat Weaver data pipeline."""

import argparse
import os
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
    best_ckpt = train(config, train_ds, val_ds, Path(args.output), resume_from=resume, init_from=init)
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
        dynamics=args.dynamics, lead_in_seconds=args.lead_in, flow_penalty=args.flow_penalty,
        end_seconds=last_music_second(audio, sr),
        notes_per_second=args.notes_per_second, stream_bias=args.stream_bias, doubles_bias=args.doubles_bias, vision_penalty=args.vision_penalty, vision_max_run=args.vision_max_run, beat_focus=args.beat_focus, block_near_converging=not args.allow_near_converging,
    )
    n_windows = max(1, (mel.shape[1] - 1) // (config.max_audio_len - min(config.max_audio_len // 4, 1024)) + 1) if mel.shape[1] > config.max_audio_len else 1

    song_name = Path(args.audio).stem
    output = Path(args.output) if args.output else Path(f"output/{song_name}")
    export_notes(notes, bpm, song_name, Path(args.audio), output, args.difficulty)
    print(f"Generated map: {output} ({n_windows} window(s), {len(notes)} notes)")


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
    gen.add_argument("--doubles-bias", type=float, default=0.0,
                     help="Logit bonus for a right-hand note right after a left-hand note (0 = off; ~1.5 reaches favourites' doubles rate)")
    gen.add_argument("--vision-penalty", type=float, default=3.0,
                     help="Logit penalty on notes in the middle-row centre cells (vision blocks); favourites keep ~2%% there")
    gen.add_argument("--vision-max-run", type=int, default=2,
                     help="Max consecutive hits containing a centre-cell note before those cells are masked (0 = no cap)")
    gen.add_argument("--beat-focus", action="store_true",
                     help="Two-stage: give a bar's extra notes to its strongest beats first, leaving rests between streams (Expert+)")
    gen.add_argument("--allow-near-converging", action="store_true",
                     help="Allow same-time notes on adjacent cells whose cuts are 135 deg apart with one aimed at the other (blocked by default)")
    gen.add_argument("--flow-penalty", type=float, default=2.0,
                     help="Logit penalty for a cut that breaks the saber's arc from the hand's last note (0 = off)")
    gen.add_argument("--dynamics", type=float, default=1.0,
                     help="Two-stage: how much each bar's note budget follows the music (0 = flat/staccato, 1 = full)")
    gen.add_argument("--lead-in", type=float, default=2.0,
                     help="Seconds at the start of the song with no notes")

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
        "evaluate": cmd_evaluate,
    }

    handler = commands.get(args.command)
    if handler:
        handler(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
