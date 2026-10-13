[![CI](https://github.com/asfilion/beat-weaver/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/asfilion/beat-weaver/actions/workflows/ci.yml)

# beat-weaver

AI-powered Beat Saber track generator — feed in a song, get a playable custom map.

## What is this?

Beat Weaver uses machine learning to automatically generate [Beat Saber](https://beatsaber.com/) note maps from audio files. Instead of manually placing blocks, you provide a song and the model outputs block positions, orientations, and timing for both sabers.

## Features

- **Audio-to-map generation** — provide an audio file, get a playable v2 Beat Saber map (BPM auto-detected or manual)
- **Difficulty selection** — generate for Easy, Normal, Hard, Expert, or ExpertPlus
- **Seeded generation** — use a fixed seed for repeatable tracks, or randomize for variety
- **Grammar-constrained decoding** — generated maps always follow valid Beat Saber structure
- **Quality metrics** — onset F1, parity violations, NPS accuracy, beat alignment, pattern diversity
- **Song in, best map out** — `python -m beat_sim.mapper song.mp3` generates candidates for every difficulty, scores them against reference maps and a kinematic body simulation, and keeps the best set

See **[ARCHITECTURE.md](ARCHITECTURE.md)** for how it works and how this fork differs from upstream.

## Requirements

- Python 3.11+
- For training: NVIDIA GPU with CUDA support
  - Medium conformer (9.4M params): 8GB+ VRAM
  - Large conformer (62M params): 24GB+ VRAM
- Beat Saber installation (Steam) — only needed for extracting official maps

## Installation

```bash
git clone https://github.com/asfilion/beat-weaver.git
cd beat-weaver

# Core (data pipeline only)
pip install -e .

# With ML model dependencies (required for training and generation)
pip install -e ".[ml]"

# Development (adds pytest)
pip install -e ".[ml,dev]"
```

## Quick Start: Training a Model

This is the end-to-end workflow from a fresh clone to a trained model. The production recipe is in [TRAINING.md](TRAINING.md).

### Step 1: Download training data

Download community maps from [BeatSaver](https://beatsaver.com/). This downloads maps with a rating score >= 0.75 and >= 5 upvotes. The download is resumable — you can stop and restart without losing progress.

```bash
# Download ~55K community maps (this takes several hours)
beat-weaver download --min-score 0.75 --output data/raw/beatsaver
```

### Step 2: Extract official maps (optional)

If you have Beat Saber installed via Steam, you can also extract the 214 official/DLC maps. These are higher quality and are sampled at a fixed share of each training batch (20% in the default configs, 10% in the production config).

```bash
# Windows (default Steam path)
beat-weaver extract-official --output data/raw/official

# Custom install path
beat-weaver extract-official --beat-saber "/path/to/Beat Saber" --output data/raw/official
```

### Step 3: Process raw maps into Parquet

Parse all downloaded/extracted maps into a normalized Parquet format for training.

```bash
beat-weaver process --input data/raw --output data/processed
```

### Step 4: Build the audio manifest

Create a JSON mapping from song hash to audio file path. This tells the training pipeline where to find each song's audio.

```bash
beat-weaver build-manifest --input data/raw --output data/audio_manifest.json
```

### Step 5: Train the model

This trains a single-stage model, the original upstream recipe. To reproduce the model that the two-stage
generator uses (main run, then a placement fine-tune on a favourites-weighted mix), follow
**[TRAINING.md](TRAINING.md)** instead. Choose a config based on your hardware:

| Config | Params | VRAM | File |
|--------|--------|------|------|
| Small | 1M | 4GB | `configs/small.json` |
| Medium | 6.5M | 6GB | `configs/medium.json` |
| Medium Conformer | 9.4M | 8GB | `configs/medium_conformer.json` |
| **Large Conformer** | **62M** | **24GB+** | **`configs/large_conformer.json`** |

```bash
# Train with the large conformer config (recommended if you have 24GB+ VRAM)
beat-weaver train \
  --config configs/large_conformer.json \
  --audio-manifest data/audio_manifest.json \
  --data data/processed \
  --output output/training

# Or with the medium conformer for 8GB GPUs
beat-weaver train \
  --config configs/medium_conformer.json \
  --audio-manifest data/audio_manifest.json \
  --data data/processed \
  --output output/training
```

On the first run, mel spectrograms are pre-computed and cached to `data/processed/mel_cache/` (~30GB for 23K songs, takes ~25 minutes). Subsequent runs reuse the cache.

Training logs to TensorBoard:

```bash
tensorboard --logdir output/training/tensorboard
```

### Step 6: Resume training (if interrupted)

Always resume from the `best/` checkpoint (never from numbered epoch checkpoints, which may be overwritten during training).

```bash
beat-weaver train \
  --config configs/large_conformer.json \
  --audio-manifest data/audio_manifest.json \
  --data data/processed \
  --output output/training \
  --resume output/training/checkpoints/best
```

### Step 7: Generate a map

Generation needs a trained model. **The weights are not in this repository.** Either train one
(see **[TRAINING.md](TRAINING.md)** for the full recipe) or copy an existing model folder. The commands
below use `models/current`, a folder holding `model.pt` and `config.json`, or a link to one. See
[ARCHITECTURE.md](ARCHITECTURE.md#where-the-model-lives) for how the production model is laid out.
If you just trained with Step 5, the single-stage checkpoint is `output/training/checkpoints/best`,
and you can generate with it by dropping `--two-stage`. For the best maps, use the two-stage model
from TRAINING.md and always pass `--two-stage`.

```bash
# Best results: candidates for every difficulty, scored, best kept, merged into one map folder
python -m beat_sim.mapper song.mp3
python -m beat_sim.mapper song.mp3 --difficulties Expert ExpertPlus --candidates 6

# One map, one difficulty (BPM auto-detected)
beat-weaver generate \
  --checkpoint models/current \
  --audio song.ogg \
  --difficulty ExpertPlus \
  --two-stage \
  --output my_map/

# With explicit BPM and seed for reproducibility
beat-weaver generate \
  --checkpoint models/current \
  --audio song.ogg \
  --difficulty ExpertPlus \
  --two-stage \
  --bpm 128 \
  --seed 42 \
  --output my_map/
```

`--downmap Easy Normal` writes those difficulties into the same folder, thinned from the generated one; `--repeats 0.7` copies repeated sections; `--emphasis 0.08` adds stacks on the loudest hits; `--drift-bias 1.5` makes hands sweep sideways (piano streams). The mapper passes all of these. Each flag's guideline is in [docs/MAPPING_GUIDELINES.md](docs/MAPPING_GUIDELINES.md).

Recommended flags per difficulty are in `generation.json` beside the production model's weights (if you have that folder), and are also the presets in `beat_sim/mapper.py`. To use a checkpoint you just trained instead, pass `--checkpoint output/training/checkpoints/best`.

The output folder can be copied directly to `Beat Saber_Data/CustomLevels/` to play in-game.

### Step 7b: Batch-generate a folder of songs

`scripts/batch_generate.py` turns a folder (or list) of audio files into one playable map folder per song, each with every requested difficulty, correct song/artist metadata, a preview start, and cover art. It runs generation as `beat-weaver generate --two-stage` inside the `beat-weaver` container on your GPU host over ssh — the same host you train on — so a laptop can drive the whole batch. It picks each difficulty's flags from `generation.json` beside the model (falling back to the same presets as `beat_sim/mapper.py`).

```bash
# All songs in a folder, all five difficulties, one playable folder per song, author set
python scripts/batch_generate.py /path/to/songs \
  --bw-host user@gpu-host \
  --author "Your Name" \
  --out ~/Downloads/maps

# A subset of difficulties, 8 in parallel, skip the web cover lookup
python scripts/batch_generate.py /path/to/songs \
  --bw-host user@gpu-host \
  --difficulties Expert ExpertPlus --concurrency 8 --no-web-cover

# Preview: list the songs and the per-difficulty presets it would use (touches nothing remote)
python scripts/batch_generate.py /path/to/songs --dry
```

It syncs this repo's `beat_weaver/` into the container so the CLI there has the current flags, uploads all audio in one stream, generates each (song, difficulty) pair in waves of `--concurrency`, then merges each song's difficulties into a single folder: `<out>/<song>/` with one `Info.dat`, a `<difficulty>.dat` per level, the audio as `song.egg`, and `cover.jpg` (embedded art first, then a YouTube/iTunes lookup unless `--no-web-cover`). With `--keep-remote`, generated (song, difficulty) pairs stay in the container and are skipped on the next run, so you can re-run it to fill in failures; each job's log is kept beside its output and the last line is printed for any that failed. Copy any folder to `Beat Saber_Data/CustomLevels/`.

### Step 8: Evaluate (optional)

```bash
beat-weaver evaluate \
  --checkpoint models/current \
  --audio-manifest data/audio_manifest.json \
  --data data/processed
```

## All-in-One Pipeline

If you want to download, extract, process, and build the manifest in a single command:

```bash
beat-weaver run --beat-saber "/path/to/Beat Saber" --output data/processed
```

Note: this runs with conservative defaults (`--max-maps 100`). For full training data, use the individual steps above.

## Architecture

Generation runs in two stages. A Conformer encoder reads the song; its onset head and a scheduler decide **when** notes happen, and the token decoder decides **what** each note is (hand, cell, cut direction) under playability rules. Several candidates per difficulty are then scored against reference maps and a kinematic body simulation, and the best are kept.

```
Audio -> [Conformer Encoder] -> onset head -> scheduler (when) ─┐
                             └-> [Token Decoder] (what) ─────────┴-> notes -> beat_sim scoring -> best map per difficulty
```

Full details, including what changed from the upstream single-stage design, are in **[ARCHITECTURE.md](ARCHITECTURE.md)**. See [RESEARCH.md](RESEARCH.md) for research details and [plans/](plans/) for implementation plans.

## Project Status

- **Data pipeline** — complete (parsers for v2/v3/v4 maps, BeatSaver downloader, Unity extractor, Parquet storage)
- **ML model** — complete (tokenizer, audio preprocessing, Conformer/Transformer encoder, training loop, inference, exporter, evaluation)
- **Baseline training** — complete (small model: 16 epochs, 23K songs, 60.6% token accuracy, generates playable maps)
- **Model improvements** — complete (dataset filtering, SpecAugment, onset features, RoPE, color balance loss, Conformer encoder)
- **Conformer training** — complete (9.4M params, best val_loss=2.23, 59.4% accuracy at epoch 26, Expert+ only)
- **Two-stage generation** — complete (onset-head timing, placement-only fine-tune, playability masks; production model `beat-weaver-placement-v1`)
- **Map evaluation and selection** — complete (`beat_sim`: reference-band scoring, music fit, kinematic body simulation, multi-difficulty mapper)

## Tests

```bash
# Run all tests (ML tests auto-skip without ML deps)
python -m pytest tests/ -v
```

## License

[MIT](LICENSE)
