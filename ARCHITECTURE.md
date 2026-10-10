# Architecture

How beat-weaver turns a song into a Beat Saber map, and how this fork differs from
[upstream](https://github.com/asfilion/beat-weaver) (forked at `9a045e4`, 2026-02-17).

## The short version

A song goes through two stages and a selection loop:

```
song.mp3
  │
  ├─ BPM detection + refinement ────────────────┐
  ├─ log-mel spectrogram + onset channel         │
  │                                              ▼
  │                                   ┌────────────────────┐
  └──────────────────────────────────►│ Conformer encoder  │
                                      └─────────┬──────────┘
                              audio memory      │
                 ┌──────────────────────────────┴───────────────┐
                 ▼                                              ▼
       Stage 1: WHEN                                 Stage 2: WHAT
       onset head → per-bar budget →                 token decoder picks hand, cell,
       scheduler picks note frames                   cut direction for each frame,
                 │                                   under masks and penalties
                 └──────────────────►  merged note list  ◄──────┘
                                              │
                                   lead-in / tail trimmed
                                              │
                                   ┌──────────▼───────────┐
                                   │ beat_sim evaluator    │  rhythm bands, music fit,
                                   │ (several candidates)  │  kinematic body sim
                                   └──────────┬───────────┘
                                              ▼
                                 best map per difficulty, merged
                                 into one playable map folder
```

The model in production is `models/current` in the data volume. See
[models/README.md](#where-the-model-lives) below.

## Stage 0: audio front end

`beat_weaver/model/audio.py`

- **Mel spectrogram:** 80 log-mel bins at 22.05 kHz, hop 512, plus one onset-strength channel.
- **Beat alignment:** frames are resampled to a 1/16-beat grid, so one bar is 64 frames. Every
  downstream step works in that grid.
- **BPM:** `detect_bpm` uses librosa's beat tracker, then `refine_bpm` searches ±1.5 BPM for the
  tempo whose beat phase locks tightest to the audio's onset energy. Trackers are often off by a
  fraction of a BPM. A 0.7 BPM error walks the grid a full beat every 85 seconds, which made
  verses start empty and end crowded. The tempo only moves when the best lock is at least 5%
  sharper than the detector's own value. On a song with a steady pulse the gain is large (lying:
  80.7 corrected to 80.0, a 12.7% gain). On songs without one it is under 0.5% and the score is
  noise, so the detector's tempo is kept.

## The model

`beat_weaver/model/transformer.py`, `config.py`

An encoder-decoder transformer. Production uses the medium Conformer (about 9.4M parameters).

- **Encoder:** linear projection, then Conformer blocks (half feed-forward, self-attention with
  RoPE, depthwise convolution, half feed-forward). Its output is the audio memory.
- **Memory positions (fork):** a fixed sinusoidal position is added to the memory. RoPE only makes
  positions relative inside the encoder's self-attention, so without this the decoder could match
  audio by content but never by place in the bar.
- **Onset head (fork):** a linear layer on the memory predicting, per frame, whether a note starts
  there. It is trained with its own loss and is what Stage 1 reads.
- **Decoder:** token embedding, RoPE self-attention, cross-attention to the memory.
- **Vocabulary:** 355 tokens. Special tokens, difficulty, 64 indexed bar tokens, 64 position-in-bar
  tokens, and compound note tokens: 108 per hand, one for each cell (4 × 3) and direction (9),
  plus an empty token per hand.

## Stage 1: when notes happen

`_onset_schedule` in `beat_weaver/model/inference.py`

The decoder never learned to read timing from the audio (see the history below), so timing comes
from the onset head instead.

1. **Budget per bar.** The target density is `--notes-per-second`, converted to notes per beat for
   the song's tempo and capped at 2.5 per beat. `--dynamics` scales each bar's share by how loud
   the onset head thinks that bar is, so quiet sections thin out and choruses fill up. The song
   average is unchanged.
2. **Frame picking.** Each beat has a queue of frames ranked by onset probability. Beats take turns,
   strongest beat first, and every beat gets at least one note before any beat gets a second.
   `--min-gap` keeps notes apart.
3. **Streams.** `--stream-bias` lets a pick chain into the next frames while they stay hot, so a
   run of sixteenths stays together. `--beat-focus` gives a bar's extra notes to its strongest
   beats, which leaves rests between streams. Expert+ needs this, or high density spreads into one
   unbroken stream.

## Stage 2: what each note is

`generate` in `beat_weaver/model/inference.py`

The decoder writes the token sequence autoregressively. Stage 1's frames are forced in as position
tokens, so the decoder only chooses left note, right note or empty at each one. Every logit passes
through these rules first:

| Rule | Kind | Why |
|---|---|---|
| Grammar mask | hard | Valid token order, bars in sequence, no END before the audio ends. |
| One cell per note | hard | Two notes can't share a cell. |
| Outward crossover | hard | A hand that crossed to the far side can't cut away from the body. |
| Converging sabers | hard | A right note can't cut into the left note at the same time on the same or an adjacent cell. |
| Near-converging | hard, flag | Cuts 135° apart with one aimed at the other. Blocked unless `--allow-near-converging`. |
| Vision cap | hard | After `--vision-max-run` consecutive hits in the middle-row centre cells, those cells close. |
| Vision penalty | soft | `--vision-penalty` keeps most notes out of the player's line of sight. |
| Saber arc | soft | `--flow-penalty`: the cheap next cut continues the hand's follow-through from its last note. |
| Doubles | soft | `--doubles-bias` raises both-hands-at-once notes toward the reference rate. |

## Stage 3: trimming

`generate_full_song` in `beat_weaver/model/inference.py`

Long songs are generated in overlapping windows and stitched at the overlap midpoint. Then:

- **Lead-in:** no notes in the first `--lead-in` seconds (default 2).
- **Tail:** no notes after the audio goes quiet for good, found from the waveform's loudness.

## Choosing the best map: beat_sim

`beat_sim/`, documented in [beat_sim/README.md](beat_sim/README.md)

Generation is stochastic, so the system makes several candidates and scores them.

- **Rhythm and layout (`rhythm.py`):** 17 statistics per map, including density, stream runs,
  doubles, dots, row and column use, vision blocks and collisions. Each is checked against the
  10th–90th percentile band of reference maps at the same difficulty. The bands come from 535
  hand-picked favourites plus 303 official maps, about 2,500 difficulties.
- **Music fit (`music.py`):** whether note density per bar follows the song's loudness, and the
  share of notes on an audio onset. Needs ffmpeg and the song file.
- **Body (`body.py`, `ik.py`, `planner.py`, `metrics.py`):** a two-arm kinematic skeleton with
  joint limits. It solves inverse kinematics per note, interpolates smooth swings, and flags
  transitions faster than anything in the reference maps. It only subtracts points; it never adds
  them.
- **Composite:** the rhythm score is the grade, minus body penalties.

Three entry points:

| Command | Use |
|---|---|
| `python -m beat_sim <map folder>` | Score one map. `--video` renders a stick-figure animation. |
| `python -m beat_sim.loop` | Grid search over generation flags, averaged over seeds. For tuning. |
| `python -m beat_sim.mapper song.mp3` | Song in, finished map out. Generates candidates per difficulty, picks the best set so that each level is at least 10% denser than the one below, and merges them into one folder. |

## Where the model lives

In the data volume (`/data` inside the beat-weaver container):

```
models/
├── README.md
├── current -> beat-weaver-placement-v1
└── beat-weaver-placement-v1/
    ├── model.pt, config.json, training_state.json
    ├── MODEL.md          what it is and how to run it
    └── generation.json   recommended flags per difficulty
```

`current` is the stable name. To promote a new model, add a versioned folder and repoint the link.
Training checkpoints with optimizer state stay under `output/`.

## How this fork differs from upstream

Upstream is a complete, working single-stage system: the decoder generates timing and placement
together from the audio, and maps are produced with one `generate` call. This fork keeps its data
pipeline, tokenizer layout, encoder and decoder, and changes how they are trained and used.

### Why it changed

Measured on the upstream-style model, the decoder ignored the audio. Two different songs gave the
same first window token for token, and the loss on a human map was the same with its own audio as
with another song's. Every song got the same map, which played as repetitive patterns over the
wrong sounds.

### What changed

| Area | Upstream | This fork |
|---|---|---|
| Who decides timing | Decoder, from its own token history | Encoder onset head plus a scheduler (two-stage) |
| Decoder's job | Timing and placement | Placement only, at given frames |
| Memory positions | None after the encoder | Sinusoidal positions added, so the decoder can address audio by place |
| Training | Plain teacher forcing | 30% token dropout on decoder inputs, onset loss on a new onset head, placement-only fine-tune |
| Bar tokens | One `BAR` token, counted | 64 indexed `BAR_0..BAR_63` tokens (vocab 291 → 355) |
| BPM | Tracker estimate | Tracker estimate refined to the tightest beat-phase lock |
| Density | Whatever the decoder produced | Target notes per second, shaped by song dynamics |
| Playability rules | Grammar only | Collision, near-collision, crossover, vision-block masks; arc, doubles and vision penalties |
| Start and end | Notes from the first frame and past the music | Lead-in silence and tail trim |
| Output selection | One map per call | Several candidates scored against reference maps, best kept, all difficulties merged |
| Evaluation | Onset F1, NPS, parity, diversity against held-out data | Plus beat_sim: reference bands, music fit, kinematic body simulation |
| Data | BeatSaver + official, about 23% of community maps silently dropped | Case-insensitive Info.dat lookup, hand-picked favourites as a weighted source, undecodable audio excluded |
| Audio export | Copied as-is, so an mp3 input made an unplayable map | Transcoded to Ogg Vorbis, the only format Beat Saber plays |

The upstream generation path still exists. Without `--two-stage`, `generate` behaves as before,
with the fork's grammar fixes and masks applied.

### Lessons recorded along the way

- A lower validation loss did not mean better maps. The decoder could lower its loss from token
  history alone, so audio grounding had to be measured directly.
- Single-seed scores swing by up to 20 points. Compare settings with `--seeds 3` or more.
- Reference bands beat hand-set thresholds. Physical limits from biomechanics flagged half of all
  transitions in maps people love; percentiles of those maps did not.

## Where to read more

- [beat_sim/README.md](beat_sim/README.md): evaluator, loop and mapper details, calibration.
- [CODEBASE_REFERENCE.md](CODEBASE_REFERENCE.md): module-by-module reference (upstream).
- [RESEARCH.md](RESEARCH.md): map format and model research (upstream, extended).
