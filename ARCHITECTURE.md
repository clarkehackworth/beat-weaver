# Architecture

How beat-weaver turns a song into a Beat Saber map, and how this fork differs from
[upstream](https://github.com/asfilion/beat-weaver) (forked at `9a045e4`, 2026-02-17).

## The short version

Two kinds of machinery work together. A small neural network, trained on human-made maps, supplies
the musical judgement: where notes are likely, and which hand, cell and cut direction fit. Ordinary
code does everything else: audio processing, density, rests, playability rules, picking the best
map, and writing the files.

Every part below is labelled with what drives it:

| Label | Meaning |
|---|---|
| **AI** | Learned. The behaviour comes from weights trained on map data. The weights are fixed when generating. |
| **Rules** | Hand-written logic or standard signal processing. Deterministic, nothing learned. |
| **AI + rules** | A learned model proposes, and rules constrain or reshape what it proposes. |

Some rules use numbers measured from reference maps, such as the scoring bands. Those are
calibrated from data but fixed: nothing updates them by training.

```
song.mp3
  │
  ▼
[RULES]  Audio front end: mel spectrogram, tempo detection and refinement
  │
  ▼
[AI]     Conformer encoder: audio -> learned features ──────────────────┐
  │                                                                     │
  ▼                                                                     │
[AI]     Onset head: odds of a note at each 1/16 beat                   │
  │                                                                     │
  ▼                                                                     ▼
[RULES]  Scheduler: note budget per  ───frames───►  [AI + RULES]  Token decoder: hand, cell and
         bar, which frames get one                                 cut direction at each frame,
                                                                   with bad placements blocked
                                                                   or discouraged by rules
                                                                              │
  ┌───────────────────────────────────────────────────────────────────────────┘
  ▼
[RULES]  Trim: lead-in silence, nothing after the music ends
  │
  ▼
[RULES]  beat_sim: score several candidates, keep the best per difficulty
  │
  ▼
[RULES]  Export: title, artist, mapper, preview, cover, Ogg audio -> one map folder
```

The model in production is `models/current` in the data volume. See
[models/README.md](#where-the-model-lives) below.

### Who decides what

| Step | Driven by | What decides |
|---|---|---|
| Mel spectrogram, 1/16-beat grid | Rules | Fixed signal processing |
| Tempo | Rules | librosa's beat tracker (classical dynamic programming), then a fixed search for the tightest beat lock |
| Audio understanding | AI | Conformer encoder |
| Odds of a note at each frame | AI | Onset head |
| How many notes, and which frames | Rules, fed by AI | Scheduler. The onset head's odds are its only learned input |
| Hand, cell and cut direction | AI + rules | Decoder scores placements, rules remove or reweight some, then one is sampled |
| Collisions, crossovers, vision blocks | Rules | Hard masks, enforced whatever the model predicts |
| Flow, doubles, centre avoidance | Rules acting on AI scores | Soft penalties and biases that a strong model preference can outweigh |
| Lead-in and tail trim | Rules | Loudness threshold on the waveform |
| Scoring candidates | Rules | Statistics against reference bands, audio analysis, a kinematic simulation |
| Best map, difficulty ladder | Rules | Search over the scored candidates |
| Title, artist, mapper, preview, cover, audio export | Rules | File tags, file-name parsing, ffmpeg |
| Training (offline) | AI | Gradient descent on human-made maps |
| Data pipeline | Rules | Map parsers and the Parquet writer |

### What the AI decides, and what it doesn't

The AI decides:

- **Where notes are likely.** The onset head's per-frame odds. Measured when it was added, its
  top-k precision against held-out human maps was 0.56, against 0.08 for chance.
- **Placement style.** For each chosen frame, each hand's cell and cut direction, or nothing,
  shaped by the notes before it.
- **Variety between candidates.** Sampling from the decoder's distribution, set by `--seed` and
  temperature.

Rules decide:

- **The tempo and beat grid, the overall density, and the rests and streams.**
- **Which placements are possible at all.** Same-time collisions, crossovers and vision blocks are
  hard rules, enforced whatever the model predicts.
- **How the song starts and ends,** and **which candidate wins** at each difficulty.
- **All metadata.**

What follows from that:

- Flags such as `--notes-per-second` or `--beat-focus` change the map at once, because they
  control rules. Changing the timing signal or the placement style needs a different model.
- The score that picks the best map is not a model. It does not improve by training, only by
  recalibrating its bands against reference maps.
- Not measured: how much the production decoder's placement depends on the audio itself rather
  than on the notes before it. Timing is the part known to come from the audio, through the
  onset head.

## Stage 0: audio front end (rules)

`beat_weaver/model/audio.py`

**Driven by: rules.** Standard signal processing; no learned model.

- **Mel spectrogram:** 80 log-mel bins at 22.05 kHz, hop 512, plus one onset-strength channel.
- **Beat alignment:** frames are resampled to a 1/16-beat grid, so one bar is 64 frames. Every
  downstream step works in that grid.
- **BPM:** `detect_bpm` uses librosa's beat tracker (classical dynamic programming, not a neural
  network), then `refine_bpm` searches ±1.5 BPM for the
  tempo whose beat phase locks tightest to the audio's onset energy. Trackers are often off by a
  fraction of a BPM. A 0.7 BPM error walks the grid a full beat every 85 seconds, which made
  verses start empty and end crowded. The tempo only moves when the best lock is at least 5%
  sharper than the detector's own value. On a song with a steady pulse the gain is large (one test song:
  80.7 corrected to 80.0, a 12.7% gain). On songs without one it is under 0.5% and the score is
  noise, so the detector's tempo is kept.

## The model (AI)

`beat_weaver/model/transformer.py`, `config.py`

**Driven by: AI.** Everything in this section is learned from map data in an offline training run.
At generation time the weights are fixed; the only randomness is sampling.

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

## Stage 1: when notes happen (AI + rules)

`_onset_schedule` in `beat_weaver/model/inference.py`

**Driven by: AI + rules.** The onset head (AI) supplies the odds that a note starts at each frame.
Everything after that is a deterministic algorithm: how many notes, which frames, where streams and
rests fall.

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

## Stage 2: what each note is (AI + rules)

`generate` in `beat_weaver/model/inference.py`

**Driven by: AI + rules.** The decoder (AI) scores every possible placement. Rules then remove some
placements outright or nudge the scores, and the final choice is sampled from what is left.

The decoder writes the token sequence autoregressively. Stage 1's frames are forced in as position
tokens (rules), so at each frame the decoder chooses, for each hand, a note (cell and cut direction)
or nothing. Before sampling, every score passes through the rules below. All of them are
hand-written. Hard rules hold whatever the model predicts; soft rules can be outweighed by a strong
model preference:

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

## Stage 2b: shaping (rules)

`beat_weaver/model/shaping.py`, after the decoder, before export. Four passes from the bsmg mapping
guidelines, all plain numpy: repeated song sections get the first occurrence's notes (mirrored on odd
repeats), the loudest isolated hits become stacks, lower difficulties are thinned from the generated
one (to each difficulty's target density, weakest beats first, hits removed whole so parity holds),
and a pattern ladder (`TIERS`) reshapes every exported difficulty to what it may contain: Easy is
up/down singles on its own side of the grid a beat apart, Normal adds horizontals, Hard adds
diagonals, crossovers, the top row and stacks, Expert adds sliders, Expert+ adds towers and windows; Easy
and Normal are also held to 3 and 5 distinct hand/cell placements per 2 bars. Flags `--repeats`, `--emphasis`,
`--sliders`, `--downmap`. `beat_weaver/model/lighting.py` then writes a rules-based lightshow into each
difficulty's `_events` from the notes and the loudness envelope. See
[docs/MAPPING_GUIDELINES.md](docs/MAPPING_GUIDELINES.md).

## Stage 3: trimming (rules)

`generate_full_song` in `beat_weaver/model/inference.py`

**Driven by: rules.** Windowing, stitching and trimming are plain code around the model's output.

Long songs are generated in overlapping windows and stitched at the overlap midpoint. Then:

- **Lead-in:** no notes in the first `--lead-in` seconds (default 2).
- **Tail:** no notes after the audio goes quiet for good, found from the waveform's loudness.

## Choosing the best map: beat_sim (rules)

`beat_sim/`, documented in [beat_sim/README.md](beat_sim/README.md)

**Driven by: rules.** There is no neural network in the evaluator or the mapper. Some thresholds were
measured from reference maps (the bands, the speed caps), so they are calibrated from data but
fixed. Nothing is trained.

Generation is stochastic, so the system makes several candidates and scores them.

- **Rhythm and layout (`rhythm.py`):** 20 statistics per map, including density, stream runs,
  doubles, dots, row and column use, lateral hand motion (piano streams vs inline), vision blocks and collisions. Each is checked against the
  10th–90th percentile band of reference maps at the same difficulty. The bands come from 535
  hand-picked favourites plus 303 official maps, about 2,500 difficulties.
- **Music fit (`music.py`):** whether note density per bar follows the song's loudness, and the
  share of notes on an audio onset. Needs ffmpeg and the song file.
- **Body (`body.py`, `ik.py`, `planner.py`, `metrics.py`):** a two-arm kinematic skeleton with
  joint limits. It solves inverse kinematics per note (a numerical optimization, not a learned
  policy), interpolates smooth swings, and flags
  transitions faster than anything in the reference maps. It only subtracts points; it never adds
  them.
- **Composite:** the rhythm score is the grade, minus body penalties.
- **Map metadata (`meta.py`):** title and artist from the audio file's tags or its file name, the
  mapper name from your own config file, a preview from the loudest ten seconds, a cover if the file
  has one. File-name parsing uses regular expressions, not a model.

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
Training checkpoints with optimizer state stay under `output/`. The weights are not in git: see
[TRAINING.md](TRAINING.md) to produce them.

## How this fork differs from upstream

Upstream is a complete, working single-stage system: the decoder generates timing and placement
together from the audio, and maps are produced with one `generate` call. This fork keeps its data
pipeline, tokenizer layout, encoder and decoder, and changes how they are trained and used. The
effect is that more of the decisions are made by hand-written code: density, rests and streams,
playability, trimming, selection and metadata all moved out of the model.

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
| Vocabulary | Color notes only | Stacks (several notes per hand at one position), 12 BOMB and 128 WALL tokens (vocab 495) |
| Generation flags | Sampling settings | Play-tested recipes for Hard, Expert and Expert+ (`RECIPE`, `HARD_RECIPE`); Easy and Normal downmapped from Hard |
| Validation | None | `validate` command: structure, reaction time, swing speed, parity, wide doubles, difficulty ladder |

The upstream generation path still exists. Without `--two-stage`, `generate` behaves as before,
with the fork's grammar fixes and masks applied.

### Lessons recorded along the way

- A lower validation loss did not mean better maps. The decoder could lower its loss from token
  history alone, so audio grounding had to be measured directly.
- Single-seed scores swing by up to 20 points. Compare settings with `--seeds 3` or more.
- Reference bands beat hand-set thresholds. Physical limits from biomechanics flagged half of all
  transitions in maps people love; percentiles of those maps did not.

## Where to read more

- [docs/MAPPING_GUIDELINES.md](docs/MAPPING_GUIDELINES.md): bsmg.wiki mapping guidelines, which flag or metric enforces each, and the backlog.
- [TRAINING.md](TRAINING.md): how to train the model, step by step, in the order that produced the production model.
- [beat_sim/README.md](beat_sim/README.md): evaluator, loop and mapper details, calibration.
- [CODEBASE_REFERENCE.md](CODEBASE_REFERENCE.md): module-by-module reference (upstream).
- [RESEARCH.md](RESEARCH.md): map format and model research (upstream, extended).
