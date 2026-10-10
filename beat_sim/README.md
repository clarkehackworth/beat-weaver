# beat_sim — map evaluator and generate-and-score loop

Rates a Beat Saber note list two ways and combines them:

1. **Rhythm** (`rhythm.py`): density, streams, doubles, dots, layout, each checked
   against the p10–p90 band of the user's favourite maps at that difficulty.
   The rhythm score is the weighted fraction of metrics in band. This is the grade.
2. **Body** (`body.py` … `metrics.py`): a two-arm kinematic skeleton with joint
   limits, IK per note, min-jerk interpolation, caps set at the p99 of what
   favourite maps produce. This only takes points off for the inhuman.

3. **Music** (`music.py`): does note density per bar track the song's loudness
   (`energy_corr`), and do notes sit on audio onsets (`onset_hit`)? Plus the
   lead-in before the first note. Only scored when the song file is beside the map.

`evaluate(notes, bpm, difficulty, audio=...)` returns all of it plus a `composite` 0..100 to rank by.

```
pip install -e ".[sim]"
python -m beat_sim path/to/map_folder [--difficulty Expert] [--json out.json] [--quiet]
python -m beat_sim path/to/map_folder --difficulty Expert --video song.mp4        # whole song, needs ffmpeg
python -m beat_sim path/to/map_folder --difficulty Expert --video clip.gif --start 30 --end 40
```

The video is a two-panel stick figure (front view with the note grid, top view
with notes approaching along z). Red is the left arm, blue the right. Frames
stream to disk one at a time, so a full song is fine; use .mp4 for anything
longer than a short clip.

```python
from beat_sim import evaluate
ev = evaluate(beatmap.notes, bpm, "Expert")   # list[Note] from beat_weaver.schemas.normalized
ev["composite"]                               # 0..100, rank candidate maps by this
print(ev["rhythm"].table())                   # every metric vs its favourites band, < or > marks out-of-band
ev["body"].right.too_fast                     # kinematic detail
```

## One song in, one map out

```
python -m beat_sim.mapper song.mp3                       # Easy..ExpertPlus, 4 candidates each
python -m beat_sim.mapper song.mp3 --difficulties Expert ExpertPlus --candidates 6
```

Per difficulty it generates candidates on the container (seeds × two density
hedges around a preset derived from that difficulty's favourites bands), scores
each with `evaluate`, keeps the best, and merges the winners into
`~/Downloads/<song>_map/` with one Info.dat. Candidates stay in
`~/Downloads/<song>_candidates/` with a `report.json` each; `--score-only`
re-picks without regenerating. Presets live in `mapper.PRESET`.

Bands for all five difficulties come from 535 favourite + 303 official maps
(`python -m beat_sim.calibrate_bands <folders...>` regenerates them).

## The loop

```
python -m beat_sim.loop --tag r1 --songs lying sawadika \
    --grid "notes-per-beat=1.0,1.5,2.0" "min-gap=0.1,0.15"   # --two-stage is the default; add flags with --extra="--temperature 0.9"
```

Runs every grid cell per song as `beat-weaver generate` inside the beat-weaver
container on docker.lan (code synced first), copies maps to `~/Downloads/<tag>/`,
scores each, writes `results.csv`, prints cells ranked by mean composite.
`--seeds N` generates each cell N times and reports mean ± sd per song, which
you want once score gaps between cells are under ~5 points (one seed is noisy). Re-running
the same tag skips cells already generated; `--score-only` rescores without
generating (after changing bands or weights). Playtest the top one or two.

## Generation flags these metrics respond to

| Flag | Metric | Note |
|---|---|---|
| `--notes-per-beat` | nps, streams | 1.3 won grids r1 and r2 at Expert |
| `--notes-per-second` | nps | overrides notes-per-beat; favourites hold NPS steady across tempos (Expert ~3.8, Expert+ ~4.8) |
| `--stream-bias` | longest_run, hand_gap_p10 | <1 chains hot frames into runs; 1 = off |
| `--dynamics` | energy_corr | 0 = flat per-bar budget (staccato), 1 = budget follows the onset head |
| `--flow-penalty` | resets, feel | arc penalty: the cheap next cut continues the hand's follow-through |
| `--lead-in` | lead_in | seconds of silence before the first note, default 2 |
| `--doubles-bias` | doubles | logit bonus for a right note after a left note; 1.0 ≈ favourites median 0.25 |
| `--vision-penalty`, `--vision-max-run` | vision, vision_run3 | keep the middle-row centre cells clear; hard cap on consecutive hits there |
| (always on) | converging | a right note may not cut into the left note placed at the same time |
| (automatic) | — | notes after the song goes quiet for good are dropped |

## Recalibrating

- Rhythm bands: `BANDS` in `rhythm.py`, from the favourites folder stats (see git history for the script).
- Body caps: `python -m beat_sim.calibrate <folder of map folders> [n_maps]` prints p50/p90/p99; paste p99s into `metrics.py`.
- Composite weights: `WEIGHTS` in `rhythm.py` and the penalties in `composite()` in `__init__.py`.

## Pipeline

| Module | Does |
|---|---|
| `body.py` | 6-DoF arm (shoulder 3, elbow 1, wrist 2), forward kinematics, joint limits and speed caps, grip tilt |
| `swing.py` | note → swing: contact point, cut direction, blade axis and hand position at approach / contact / follow-through |
| `ik.py` | `scipy.optimize.least_squares` with bounds; residual = blade through point + hand position + blade axis + stay-near-previous-pose |
| `planner.py` | IK per swing seeded from the previous pose, min-jerk joint interpolation at 100 Hz, tip speed and acceleration |
| `music.py` | ffmpeg-decoded RMS envelope; energy correlation and onset-hit metrics |
| `calibrate_music.py` | bands for the music metrics over a folder of maps with audio |
| `rhythm.py` | density / stream / layout stats and favourites bands, `score(notes, bpm, difficulty)` |
| `loop.py` | generate-and-score grid runner against the container |
| `calibrate.py` | body-cap percentiles over a folder of maps |
| `viz.py` | stick-figure .mp4 (ffmpeg) or .gif (Pillow), streamed frame by frame, `render(trajs, notes, path, start, end)` |
| `metrics.py` | hits, too-fast (joint speed / tip speed / tip acceleration caps), parity resets, crossovers, joint-limit stress, extension, composite score |

Parity emerges from the geometry: two consecutive same-direction cuts force a
reset swing, which costs time and shows up as `resets`.

## Body model knobs

Everything physical is a module-level constant with a comment stating what it
was set against. Swing amplitude scales with the gap to the nearest same-hand
note (`swing.amplitude`), so streams are flicks and slow notes are full swings.
Crossover means the far column only; favourites use the near-centre column 18%
of the time.

Known ceiling: favourite Expert maps contain quarter-second transitions that
the model rates at 40–55 m/s tip speed, because it has no torso turn and
always completes the follow-through. The p99 caps absorb that. A torso yaw
joint is the upgrade path if body flags ever need to be sharper than "beyond
anything in the favourites".

## Not modelled (on purpose)

Dynamics (torque, inertia), torso/head/legs, saber-body collision, bombs,
walls, arcs/chains, Beat Saber score accuracy, fatigue over a song. Add the
first of these only when the GIF shows the model cheating in a way the
metrics miss.
