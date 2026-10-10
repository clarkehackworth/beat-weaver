# Training the model

How to train the model that generation uses, in the order that produced the production model
(`beat-weaver-placement-v1`). For how the model is used at generation time, see
[ARCHITECTURE.md](ARCHITECTURE.md). The README's quick start is a shorter, single-run version of
steps 1 to 5 here.

The recipe has three training phases:

| Phase | What | Config | Took (RTX 5060 Ti, 16 GB) |
|---|---|---|---|
| A. Main run | Learn the whole token stream from audio | `configs/medium_conformer_favorites.json` | about 31 hours for 24 epochs |
| B. Placement fine-tune | Retrain only the left/right placement tokens | `configs/medium_conformer_placement.json` | about 7.7 hours for 6 epochs |
| C. Pick the checkpoint | Compare epochs on the placement loss | `scripts/score_placement.py` | minutes |

You can skip A if you already have a main-run checkpoint, and skip B and C if you only want a
single-stage model. Generation with `--two-stage` is meant for the placement model.

## 0. Setup

```bash
git clone https://github.com/clarkehackworth/beat-weaver.git
cd beat-weaver
pip install -e ".[ml]"        # torch, librosa, soundfile, pandas, tensorboard
pip install -e ".[sim]"       # only for beat_sim scoring, not for training
```

You need:

- **A CUDA GPU.** The medium Conformer (about 9.4M parameters) trains in 8 GB with the settings below.
  Production ran on a 16 GB card at about 13 samples per second.
- **ffmpeg** on the path, for decoding audio.
- **About 30 GB of disk** for the cached spectrograms, plus the raw maps.
- **Numba on CPU-only containers:** if audio processing fails with a CPU-architecture error,
  set `NUMBA_CPU_NAME=generic` and `NUMBA_CACHE_DIR=/tmp/nc_generic`.

Tests that need torch or librosa only run with the `ml` extra installed.

## 1. Get the maps

Maps are grouped under `data/raw/` by where they came from. The training code reads the **folder
name** to decide a map's source, so the layout matters:

```
data/raw/
├── beatsaver/     community maps, downloaded           -> source "beatsaver"
├── official/      the game's own maps, extracted       -> source "official"
└── favorites/     any other folder of map folders      -> source "local_custom"
```

Any folder under `data/raw/` that is not named `official` or `beatsaver` is read as `local_custom`.
That is where hand-picked maps go. Each map is a folder holding `Info.dat`, its difficulty `.dat`
files and the audio.

```bash
# Community maps rated >= 0.75 with >= 5 upvotes. Resumable. Several hours.
beat-weaver download --min-score 0.75 --output data/raw/beatsaver

# Official maps, from a Beat Saber install (add --beat-saber "/path/to/Beat Saber" if not the default)
beat-weaver extract-official --output data/raw/official

# Your own favourites: copy the map folders into data/raw/favorites/<one folder per map>
```

**Why the favourites folder exists.** It lets you steer the model toward maps you like. In the
production run, hand-picked favourites made up 15% of every training batch and official maps 10%,
though they are a small share of the data. The remaining 75% is community maps, weighted by their
BeatSaver score.

## 2. Process and index

```bash
# Parse every map into Parquet. Finds Info.dat in any letter case.
beat-weaver process --input data/raw --output data/processed

# Map each song to its audio file
beat-weaver build-manifest --input data/raw --output data/audio_manifest.json
```

If a map appears in more than one source, one copy is kept, chosen in the order official, then
local custom, then everything else. Songs whose audio cannot be decoded are left out. The first
training run also builds a spectrogram cache in `data/processed/mel_cache/` (about 30 GB, about
25 minutes on many cores). Later runs reuse it.

## 3. Phase A: the main run

```bash
beat-weaver train \
  --config configs/medium_conformer_favorites.json \
  --audio-manifest data/audio_manifest.json \
  --data data/processed \
  --output output/training
```

This trains on Expert and Expert+ maps only (`min_difficulty: Expert`, Standard characteristic,
50 to 300 BPM). Settings that matter, all in the config:

| Setting | Value | Why |
|---|---|---|
| `learning_rate`, `warmup_steps` | 3e-5, 1000 | Larger values made this model diverge. Medium models need a low rate and a long warmup. |
| `batch_size`, `gradient_accumulation_steps` | 8, 4 | An effective batch of 32 inside 8 GB. |
| `source_ratios` | official 0.10, local_custom 0.15 | Share of each batch from those sources. The rest goes to community maps. |
| `token_dropout` | 0.3 | Replaces 30% of decoder input tokens with padding, so the decoder cannot predict the map from its own past and must look at the audio. |
| `onset_loss_weight` | 0.5 | Trains a head on the encoder output to predict where notes are. This is the head that generation reads for timing. |
| `early_stopping_patience` | 10 | Stops after ten epochs without a better validation loss. |

Production stopped itself at epoch 24 of 50 after 31 hours. The best checkpoint is
`output/training/checkpoints/best`, and every epoch is also saved as `epoch_NNN`.

Watch training with `tensorboard --logdir output/training/tensorboard`.

**If it is interrupted,** resume from `best`, never from a numbered epoch:

```bash
beat-weaver train --config configs/medium_conformer_favorites.json \
  --audio-manifest data/audio_manifest.json --data data/processed \
  --output output/training --resume output/training/checkpoints/best
```

Resuming from `epoch_013` retrains epoch 12 and saves it back to `epoch_013`, which overwrote a good
checkpoint once. Resume restores the model, optimizer, scheduler and gradient scaler together.
Missing the gradient scaler caused NaN losses on resume before it was saved.

**A lower loss is not proof the model uses the audio.** The first models reached good validation
loss while ignoring the song. The token dropout and onset loss above are what fixed that. If you
change the training, check that two different songs give different maps from the same seed.

## 4. Phase B: placement fine-tune

Two-stage generation takes timing from the onset head, so the decoder only ever has to choose what
goes at each position. This phase trains it on exactly that: the loss covers the left and right note
tokens and nothing else.

Start from the main run's weights, but with a fresh optimizer and schedule (`--init-from`, not
`--resume`). Production started from `epoch_014` of the main run, the epoch with the lowest
validation loss (2.5188, which is also what `best` holds). Placement loss was not measured for the
other main-run epochs, so a different starting epoch was never compared.

```bash
beat-weaver train \
  --config configs/medium_conformer_placement.json \
  --audio-manifest data/audio_manifest.json --data data/processed \
  --output output/placement \
  --init-from output/training/checkpoints/epoch_014
```

Changes from the main run: `placement_only: true`, learning rate 1e-5, warmup 200, at most 8
epochs, early stopping after 3. Production stopped at epoch 6 after about 7.7 hours.

`configs/medium_conformer_placement_hard.json` adds Hard maps to the same recipe. It scored worse
on Expert placement and is kept only as a record of that experiment.

## 5. Phase C: pick the checkpoint

The best epoch by validation loss is not necessarily the one to ship, and loss from the main run
and the placement run measure different tokens, so they cannot be compared directly. Score each
candidate the same way:

```bash
python scripts/score_placement.py \
  output/training/checkpoints/epoch_014 \
  output/placement/checkpoints/epoch_001 output/placement/checkpoints/epoch_003 output/placement/checkpoints/best
```

Production results (lower is better):

| Checkpoint | Placement validation loss |
|---|---|
| main run, epoch 14 | 2.6719 |
| placement run, epoch 1 | 2.5419 |
| placement run, epoch 3 | **2.5374** (shipped) |
| placement run, epoch 6 | 2.5680 |

Epoch 3 is the lowest, and later epochs got worse, so more training would not have helped. When
the numbers are close, generate a few maps from each candidate and play them, because small loss
differences do not always show up in play.

## 6. Install it as the production model

Generation reads the model from `models/current` in the data volume. The weights are not in git.

```bash
mkdir -p models/my-model-v1
cp output/placement/checkpoints/epoch_003/{model.pt,config.json,training_state.json} models/my-model-v1/
ln -sfn my-model-v1 models/current
```

Add a `MODEL.md` beside the weights saying what it is, its validation loss and how it was trained,
so whoever picks it up next can tell. To replace it later, add a new versioned folder and repoint
`current`; leave older folders where they are.

## 7. Check that it generates

```bash
beat-weaver generate --checkpoint models/current --audio song.mp3 \
  --difficulty ExpertPlus --two-stage --output test_map/
python -m beat_sim test_map/ --difficulty ExpertPlus        # score it against reference maps
```

For several candidates per difficulty with the best kept, use `python -m beat_sim.mapper song.mp3`.
The scoring bands were measured from favourite and official maps with
`python -m beat_sim.calibrate_bands`. If your favourites folder differs a lot from mine, rerun that.

## Recreating this model from nothing, in short

1. Put official, community and favourite maps under `data/raw/{official,beatsaver,favorites}`.
2. `beat-weaver process`, then `beat-weaver build-manifest`.
3. Train with `medium_conformer_favorites.json` until it stops itself.
4. Fine-tune with `medium_conformer_placement.json --init-from` an epoch of step 3.
5. Score the placement epochs with `scripts/score_placement.py` and take the lowest.
6. Copy that checkpoint to `models/<name>` and repoint `models/current`.

## Reference: where each number came from

- Main run: 60,678 training and 7,603 validation samples, 24 epochs, average 4,621 s per epoch.
- Placement run: 6 epochs, average 4,610 s per epoch. Both ran at about 13 samples per second.
- The shipped checkpoint is `output/placement/checkpoints/epoch_003`; its `training_state.json`
  records `epoch: 2` because the counter starts at zero.
