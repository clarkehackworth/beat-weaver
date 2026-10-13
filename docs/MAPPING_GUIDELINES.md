# Mapping guidelines

What the Beat Saber mapping community (bsmg.wiki) says a good map does, and how beat-weaver applies
each point. Sources: [Basic](https://bsmg.wiki/mapping/basic-mapping.html),
[Intermediate](https://bsmg.wiki/mapping/intermediate-mapping.html),
[Downmapping](https://bsmg.wiki/mapping/downmapping.html),
[Extended](https://bsmg.wiki/mapping/extended-mapping.html) and the
[Glossary](https://bsmg.wiki/mapping/glossary.html). Numbers marked *favourites* were measured over
the 535 hand-picked reference maps used for the evaluator bands.

Three columns matter for each guideline: what the wiki says, where beat-weaver enforces it
(generation flag, evaluator metric, validator check), and what is still open. The open items are
collected in the [backlog](#backlog) at the end.

## 1. Representation: the map follows the music

| Guideline | Wiki | beat-weaver |
|---|---|---|
| Map real sounds. Most songs need only 1/1, 1/2, 1/4 precision; 1/8 and finer is usually overmapping. | Basic: Timing | Onset head on a 1/16 grid, `--min-gap`. No precision prior yet (backlog 8). |
| Undermapping is fine, overmapping is not. | Basic | Scheduler budget `--notes-per-second`, rests on quiet beats (`--beat-floor`). |
| Density follows the song's energy. | Intermediate: Rhythm choice | `--dynamics`; evaluator `energy_corr`, `onset_hit`. |
| Emphasis ladder: single, 2-slider, stack, 3-slider, window, tower. Big sound, big swing. If everything is emphasized nothing is. | Basic: Emphasis | `--emphasis FRAC` stacks the loudest isolated hits; `--sliders FRAC` 2-dot sliders on held sounds; at Expert+ the loudest quarter of the emphasised hits become towers or windows (`shaping.emphasize`). Doubles via `--doubles-bias`/`--max-doubles`. |
| Consistency: the same sound mapped the same way; similar sections mapped similarly, mirrored or varied within the motif. | Intermediate: Consistency, Structure | `--repeats SIM` finds repeated 4-bar blocks by beat-synchronous mel similarity and copies the first occurrence onto them, mirroring odd repeats (`shaping.find_repeats`, `repeat_sections`). |
| Lead-in and tail: no notes in silence. | Basic | `--lead-in`, music-end trim; evaluator `lead_in`. |

## 2. Flow and parity

| Guideline | Wiki | beat-weaver |
|---|---|---|
| Parity: forehand/backhand alternate per hand. A "double directional" (same swing twice in a row) is a reset. | Basic: Parity | `--reset-gap` masks same-way cuts within the gap; `--flow-penalty` prefers the cut that continues the saber's arc; body sim counts `resets`; validator flags double directionals within 1.5 beats above the favourites' p90 rate per difficulty. |
| Time scaling: resets are fine when the hand has had time (about 1.5 beats) to come back. | Intermediate: Spacing | `--flow-window` fades the flow penalty to 0 over the last beat of the window (default 1.5). |
| Rotation amounts: 180° and 135° are safe at any speed; 90° maybe; 45° is a reset. Larger changes need more time. | Basic: Parity | Arc penalty is cosine-shaped so 180° is cheapest and 45° worst. |
| Momentum: a wrist turning one way wants to keep turning, up to a point; then it must unwind. | Intermediate: Momentum and Tension | `--momentum-bias` rewards cuts that keep the rotation sense, flipping to a penalty after 3 same-sense turns. |
| Swing curvature: positive (natural arc) fine to 135°; negative beyond 45° is harsh. | Intermediate: Rotations | Partly covered by the arc penalty. Explicit curvature term: backlog 6b. |
| Avoid awkward same-colour curves, triangles (three cuts rotating one way), Shrado angles, staircase hitbox abuse, Loloppe notes. | Basic, Glossary | Momentum tension cap covers triangles. The rest: backlog 10. |

## 3. Spatial patterns

| Guideline | Wiki | beat-weaver |
|---|---|---|
| Hands move. Streams sweep sideways (piano stream: alternating colour and direction across the grid), jumps change lane or row. A hand parked in one cell is "inline". | Glossary: Piano Stream, Jump, Inline | `--drift-bias` rewards landing one lane over in the hand's current drift direction, reversing at the edges. Evaluator `lateral` (mean lanes moved per same-hand swing, favourites Expert 0.55–1.20), `inline` (share that stays put, p90 0.35), `zigzag3` (monotonic 3-lane runs per 100 pairs, Expert 0.7–9.4). |
| Vision blocks: nothing on the two centre middle cells that hides the next note. | Basic: Vision | `--vision-penalty`, `--vision-max-run`; evaluator `vision`, `vision_run3`. |
| Crossovers: outward-facing cuts after crossing are near unhittable; inward is fine. Easy and Normal: none. | Basic, Downmapping | Outward crossover tokens masked; `far_cross` metric; downmapper drops crossovers for Easy/Normal. |
| Handclaps, arm tangles, controller clash: hands converging on the same cell or crossing with opposite vertical cuts. | Basic | Shared cell masked; converging pairs masked; validator "crossed hands". Arm-tangle check: backlog 10. |
| Top row sparingly (shoulder strain), especially at Easy/Normal. Down-cuts on the top row and inward doubles on the edges are hard for short players. Forehand comfortable in the inner lanes, backhand in the outer. | Basic: Difficulty, Intermediate: Handedness | `top_row` metric; the ladder keeps Easy/Normal off the top row. `--hand-bias` adds a zero-mean comfort table to each hand's logits (`inference._hand_comfort`): forehand inner, backhand outer, no down-cut on the top row or up-cut on the bottom, no outward horizontal on the edge. |
| Hitbox: no two notes of one colour that one swipe cannot cover. | Basic | Validator "wide doubles"; evaluator `converging`. |

## 4. Difficulty spread (downmapping)

| Guideline | Wiki | beat-weaver |
|---|---|---|
| Lower difficulties are the top difficulty distilled: keep the key moments, drop about 20% of the notes per step. | Downmapping | `--downmap Easy Normal` derives lower difficulties from the generated one (`shaping.downmap`): hits removed whole (parity untouched), weakest beats first (off-grid, quiet, non-downbeat), never two neighbours in a row inside a beat so streams become half-time streams. The mapper generates Hard and downmaps Easy and Normal from it; Expert and Expert+ are generated at their own density. |
| Easy and Normal: no crossovers, few top-row notes, dots acceptable, 2–3 (Easy) or 4–5 (Normal) unique patterns per half-jump interval. | Downmapping | Pattern ladder below (`shaping.TIERS`, `gate_patterns`), applied to every exported difficulty. `max_cells` caps distinct (hand, cell) placements per 2 bars (Easy 3, Normal 5); extras move to the hand's commonest cell. |
| Each difficulty is denser than the one below. | Basic: Difficulty | Mapper `LADDER_RATIO` 1.10; validator ladder check. |
| Reference densities (notes/s): Easy ~0.8, Normal ~1.1, Hard ~1.9, Expert ~2.4 (official). | Basic | Favourites bands per difficulty in `beat_sim/rhythm.py` (`nps`). |

### Pattern ladder

Each difficulty may use the patterns of its row and every row above it. `shaping.gate_patterns` runs on every
exported difficulty (generated or downmapped) and converts anything above the tier into the nearest allowed
shape rather than dropping it: diagonals lose their sideways part, horizontals become alternating up/down
cuts, a crossed-over hand moves back to its own side, top-row notes drop a row, stacks and sliders collapse
to one note, doubles thin to the cap and lane jumps are shortened. Hits closer than the minimum gap are
dropped. Parity is repaired afterwards.

| Difficulty | Unlocks | Cuts | Hand side | Rows | Doubles | Lane jump | Min gap |
|---|---|---|---|---|---|---|---|
| Easy | singles, up/down, dots | up, down, dot | own half only | bottom two | ≤ 18% | 1 | 1 beat |
| Normal | horizontals, 2-lane moves | + left, right | own half only | bottom two | ≤ 20% | 2 | 1/2 beat |
| Hard | diagonals, inward crossovers, top row, stacks | all | free | all | free | 3 | none |
| Expert | sliders, piano streams (drift bias) | all | free | all | free | 3 | none |
| Expert+ | towers (3 in an outer column) and windows (one-cell gap) on the loudest quarter of emphasised hits | all | free | all | free | 3 | none |

Outward crossovers are masked at generation for every difficulty. Downmapping aims at the absolute density in
`shaping.TARGET_NPS`, not just a ratio of the source. Easy's target (1.2/s) sits near the bottom of the favourites
band on purpose: the favourites are an Expert+ player's picks and their Easy runs well above official Easy
(~0.8/s). The evaluator's Easy `updown` band is open at the top because the tier makes every cut vertical.

## 5. Jump settings

| Guideline | Wiki | beat-weaver |
|---|---|---|
| Set reaction time, not NJS. Expert+ 500–600 ms, Hard 600–750 ms, Easy/Normal 750–1200 ms. NJS 16 is the Expert+ baseline; above 25 never. Keep NJS low and use the offset to fix reaction time. | Intermediate: Jump Settings | `exporter.jump_settings` computes `_noteJumpStartBeatOffset` from BPM and NJS to pin reaction time per difficulty (Easy 1200, Normal 1050, Hard 900, Expert 750, Expert+ 600 ms, matching favourites' medians). NJS: Easy 10, Normal 11, Hard 12, Expert 15, Expert+ 18. Validator flags reaction times outside 400–1500 ms. |

## 6. Playability and comfort

| Guideline | Wiki | beat-weaver |
|---|---|---|
| Effective BPM: 100–150 natural, 150–200 tiring, 200+ speed maps. Same-hand sequences feel much faster than alternating ones. | Intermediate: Swing Speed | Evaluator `ebpm` (p90 of 60 / per-hand swing gap, favourites p90: Easy 142, Normal 239, Hard 280, Expert 308, Expert+ 384); `--max-ebpm` masks a hand that swung too recently so the other hand takes the slot (mapper presets use the favourites p90); validator flags eBPM above the cap with 10% headroom. |
| Gallops (two singles then a double) are hard and add unintended emphasis. | Glossary | A gallop is a same-hand gap under the eBPM cap, so `--max-ebpm` stops most of them. Explicit check: backlog 10. |
| Leaning, swaying, crouching: regular, predictable, started on forehand swings. | Intermediate: Full Body Motion | Out of scope until walls exist: backlog 12. |
| Bombs: represent a sound; keep 1/4 beat clear before and after notes; resets at 2/3 of the gap. | Intermediate: Bombs | No bombs generated: backlog 12. |
| Arcs and chains: sparingly, for select swings. Walls: standard full-height and crouch only, never 3-wide, never the whole grid. | Extended, Basic | Not generated: backlog 12. |
| Playtest with fresh players; mapper blindness is real. | Intermediate: Readability | `beat_sim` body simulation stands in. Player feedback system: CLAUDE.md open question. |
| Lighting: lights follow the notes and the song's energy; dark in rests. | Basic Lighting | Rules-based lightshow on every export (`lighting.light_events`, `--no-lights` to skip): back lasers and the hand's rotating laser flash in the note colour, ring lights fade per bar, centre lights on the loudest third of bars, ring spin every 8 bars, zoom on loudness jumps, laser speed from density, lights out in rests over 2 beats and at the end. |

## Where each rule lives

- Generation flags: `beat-weaver generate --help`; presets in `beat_sim/mapper.py` (`PRESET`, `SHAPING`).
- Post-decode shaping: `beat_weaver/model/shaping.py` (downmap, repeats, emphasis).
- Evaluator metrics and bands: `beat_sim/rhythm.py`; recalibrate with `python -m beat_sim.calibrate_bands <favourites folder>`.
- Jump settings: `beat_weaver/model/exporter.py`.
- Validator: `beat-weaver validate <map folder>` (`beat_weaver/model/map_check.py`).
- Lighting: `beat_weaver/model/lighting.py`.

## Backlog

Numbering continues from the review that produced this document. Items 1–7, 9, 11, 14, 15 and 16 shipped on 2026-10-12.

7. ~~Forehand/backhand placement biases~~ shipped: `--hand-bias` (`inference._hand_comfort`), mapper `SHAPING["hand_bias"]` 0.5.
8. **Rhythm precision prior.** Require a stronger onset score for 1/8 and 1/16 placements than for 1/2, 1/3 and 1/4; snap vocal-led sections to the intended rhythm. (Basic: Timing)
9. ~~Effective BPM cap~~ shipped: evaluator `ebpm`, `--max-ebpm` in generation (mapper presets per difficulty), validator check.
10. **Named bad patterns as explicit checks.** Arm tangles, handclaps, gallops, staircase hitbox, inline spam, Shrado angle, Loloppe notes, triangles beyond the momentum cap. Evaluator counts plus masks where cheap. (Basic, Glossary)
11. ~~Validator additions~~ shipped: reaction-time range, max eBPM, double directionals within 1.5 beats. The density ladder already covered all five difficulties. Inline rate stays an evaluator metric only.
12. **Bombs, walls, arcs, chains.** Bomb resets at 2/3 of a gap for long rests; cosmetic bombs clear of saber paths; standard dodge and crouch walls with time to return; arcs on select sustained notes. Each needs tokenizer or exporter support. (Intermediate: Bombs, Arcs; Extended)
13. **Mapping Extensions, 360/90 modes.** Multi-lane and rotation mapping. Separate characteristic; low priority. (Extended)
14. ~~Lighting~~ shipped: `lighting.light_events`, on by default, `--no-lights` to skip.
15. ~~Windows and towers~~ shipped: `shaping.emphasize` with `difficulty="ExpertPlus"`, gated by `TIERS[..]["towers"]`.
16. ~~Unique-pattern limit for Easy/Normal~~ shipped: `TIERS[..]["max_cells"]` (Easy 3, Normal 5 distinct hand/cell placements per 2 bars).
17. **Swing curvature term.** Penalise negative-curvature rotations above 45° in the arc penalty. (Intermediate: Rotations)
