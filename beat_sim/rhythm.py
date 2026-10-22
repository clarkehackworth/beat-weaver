"""Rhythm and layout statistics of a note list, scored against bands from the user's favourite maps.

A metric is "in band" when it falls between the p10 and p90 of favourite maps at the same
difficulty. The rhythm score is the weighted fraction of metrics in band, 0..100.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from beat_weaver.schemas.normalized import Note

# (p10, p90) per metric over 535 favourite + 303 official map folders, Standard characteristic (python -m beat_sim.calibrate_bands).
# Music metrics and lead_in: 431 Expert / 390 Expert+ favourite + official difficulties with audio (calibrate_music),
# measured with the per-note nearest-onset metric; Easy-Hard reuse the Expert values. from "Beat Saber Favorites-2026.10.05".
# Regenerate with `python -m beat_sim.calibrate <favourites folder>`.
BANDS = {
    'Easy': {  # 367 maps
        'converging': (0.000, 0.000),
        'near_converging': (0.000, 0.000),
        'vision': (0.000, 0.078),
        'vision_run3': (0.000, 0.000),
        'nps': (1.087, 1.829),
        'gap_le_half': (0.000, 0.069),
        'gap_1': (0.031, 0.537),
        'gap_ge2': (0.000, 0.712),
        'streams_per_min': (0.000, 0.000),
        'longest_run': (1.000, 3.000),
        'hand_gap_p10': (0.462, 0.978),
        'doubles': (0.062, 0.365),
        'dots': (0.000, 0.065),
        'top_row': (0.000, 0.058),
        'outer_cols': (0.287, 0.465),
        'far_cross': (0.000, 0.010),
        'updown': (0.590, 1.000),  # the Easy tier allows only up/down/dot cuts (shaping.TIERS), so all-vertical is by design
        'lateral': (0.250, 1.166),
        'inline': (0.000, 0.618),
        'zigzag3': (0.000, 4.919),
        'ebpm': (0.0, 141.7),
        'tangle': (0.000, 0.000), 'gallop': (0.000, 0.958), 'inline4': (0.000, 2.366), 'staircase': (0.000, 0.000), 'loloppe': (0.000, 0.000), 'shrado': (0.000, 0.000),
        'fine': (0.000, 0.016),
        'energy_corr': (0.289, 1.0),
        'onset_hit': (0.624, 1.0),
        'lead_in': (1.6, 7.4),
    },
    'Normal': {  # 425 maps
        'converging': (0.000, 0.000),
        'near_converging': (0.000, 0.000),
        'vision': (0.000, 0.069),
        'vision_run3': (0.000, 0.000),
        'nps': (1.661, 2.792),
        'gap_le_half': (0.000, 0.370),
        'gap_1': (0.241, 0.733),
        'gap_ge2': (0.000, 0.296),
        'streams_per_min': (0.000, 3.909),
        'longest_run': (1.000, 8.000),
        'hand_gap_p10': (0.300, 0.631),
        'doubles': (0.062, 0.343),
        'dots': (0.000, 0.066),
        'top_row': (0.000, 0.166),
        'outer_cols': (0.308, 0.497),
        'far_cross': (0.000, 0.031),
        'updown': (0.565, 0.833),
        'lateral': (0.221, 1.114),
        'inline': (0.000, 0.553),
        'zigzag3': (0.000, 6.192),
        'ebpm': (0.0, 239.0),
        'tangle': (0.000, 0.000), 'gallop': (0.000, 3.246), 'inline4': (0.000, 3.345), 'staircase': (0.000, 0.000), 'loloppe': (0.000, 0.000), 'shrado': (0.000, 0.000),
        'fine': (0.000, 0.200),
        'energy_corr': (0.289, 1.0),
        'onset_hit': (0.624, 1.0),
        'lead_in': (1.6, 7.4),
    },
    'Hard': {  # 508 maps
        'converging': (0.000, 0.003),
        'near_converging': (0.000, 0.000),
        'vision': (0.000, 0.061),
        'vision_run3': (0.000, 0.000),
        'nps': (2.400, 4.090),
        'gap_le_half': (0.132, 0.761),
        'gap_1': (0.102, 0.592),
        'gap_ge2': (0.000, 0.122),
        'streams_per_min': (0.303, 17.067),
        'longest_run': (4.000, 49.300),
        'hand_gap_p10': (0.211, 0.438),
        'doubles': (0.072, 0.369),
        'dots': (0.000, 0.075),
        'top_row': (0.000, 0.215),
        'outer_cols': (0.336, 0.530),
        'far_cross': (0.000, 0.036),
        'updown': (0.520, 0.843),
        'lateral': (0.469, 1.117),
        'inline': (0.000, 0.388),
        'zigzag3': (0.000, 8.071),
        'ebpm': (0.0, 280.0),
        'tangle': (0.000, 0.000), 'gallop': (0.000, 6.274), 'inline4': (0.000, 3.782), 'staircase': (0.000, 0.000), 'loloppe': (0.000, 0.000), 'shrado': (0.000, 0.000),
        'fine': (0.000, 0.286),
        'energy_corr': (0.289, 1.0),
        'onset_hit': (0.624, 1.0),
        'lead_in': (1.6, 7.4),
    },
    'Expert': {  # 644 maps
        'converging': (0.000, 0.005),
        'near_converging': (0.000, 0.000),
        'vision': (0.000, 0.072),
        'vision_run3': (0.000, 0.000),
        'nps': (3.092, 5.221),
        'gap_le_half': (0.399, 0.896),
        'gap_1': (0.041, 0.450),
        'gap_ge2': (0.000, 0.061),
        'streams_per_min': (4.189, 22.038),
        'longest_run': (10.000, 128.000),
        'hand_gap_p10': (0.171, 0.341),
        'doubles': (0.081, 0.385),
        'dots': (0.000, 0.077),
        'top_row': (0.018, 0.262),
        'outer_cols': (0.365, 0.538),
        'far_cross': (0.000, 0.050),
        'updown': (0.447, 0.797),
        'lateral': (0.547, 1.201),
        'inline': (0.000, 0.353),
        'zigzag3': (0.665, 9.364),
        'ebpm': (0.0, 308.0),
        'tangle': (0.000, 0.001), 'gallop': (0.000, 7.785), 'inline4': (0.000, 3.368), 'staircase': (0.000, 0.000), 'loloppe': (0.000, 0.000), 'shrado': (0.000, 0.000),
        'fine': (0.000, 0.276),
        'energy_corr': (0.289, 1.0),
        'onset_hit': (0.624, 1.0),
        'lead_in': (1.6, 7.4),
    },
    'ExpertPlus': {  # 577 maps
        'converging': (0.000, 0.004),
        'near_converging': (0.000, 0.000),
        'vision': (0.000, 0.072),
        'vision_run3': (0.000, 0.000),
        'nps': (3.913, 6.524),
        'gap_le_half': (0.524, 0.945),
        'gap_1': (0.021, 0.336),
        'gap_ge2': (0.000, 0.041),
        'streams_per_min': (6.544, 23.138),
        'longest_run': (15.000, 205.400),
        'hand_gap_p10': (0.137, 0.287),
        'doubles': (0.091, 0.368),
        'dots': (0.000, 0.095),
        'top_row': (0.100, 0.307),
        'outer_cols': (0.399, 0.547),
        'far_cross': (0.000, 0.088),
        'updown': (0.359, 0.752),
        'lateral': (0.427, 1.275),
        'inline': (0.000, 0.372),
        'zigzag3': (0.790, 10.860),
        'ebpm': (0.0, 384.0),
        'tangle': (0.000, 0.006), 'gallop': (0.000, 9.355), 'inline4': (0.000, 3.457), 'staircase': (0.000, 0.000), 'loloppe': (0.000, 0.000), 'shrado': (0.000, 0.000),
        'fine': (0.000, 0.511),
        'energy_corr': (0.286, 1.0),
        'onset_hit': (0.621, 1.0),
        'lead_in': (1.6, 6.5),
    },
}

WEIGHTS = {  # what matters most, per the alignment discussion
    "nps": 3, "gap_le_half": 2, "streams_per_min": 2, "longest_run": 1, "gap_1": 1, "gap_ge2": 1, "hand_gap_p10": 1,
    "doubles": 1, "dots": 2, "far_cross": 2, "top_row": 1, "outer_cols": 1, "updown": 1, "ebpm": 2,
    # only scored when audio is available (see score(extra=...)); dropped from the denominator otherwise
    "energy_corr": 3, "onset_hit": 2, "lead_in": 1,
    "vision": 2, "vision_run3": 2,  # notes in the middle-row centre cells (hide what follows), and runs of 3+ hits there
    "converging": 2,  # same-time pairs whose cuts drive the sabers into each other (favourites median 0)
    "near_converging": 1,  # 45 deg off head-on with one cut aimed at the other note; legal but very hard
    # sideways motion of each hand between its own swings (bsmg "piano stream" / "jump" shapes vs "inline")
    "lateral": 2, "inline": 1, "zigzag3": 1,
    # bsmg named bad patterns (backlog 10), bands from the 535 favourites (tangle/staircase: p90 is ~0, so any is out of band)
    "tangle": 1, "gallop": 1, "inline4": 1, "staircase": 1,
    "loloppe": 1, "shrado": 1,  # ponytail: bands (0, 0) by definition, not yet measured on the favourites; recalibrate when the folder is at hand
    "fine": 1,  # share of notes at 1/8 beat or finer (offset-corrected, triplets on-grid)
}
# Graded instead of pass/fail: full credit in the middle half of the band, fading to 0 at either edge.
# A 210-note stream passed Expert+'s (15, 205) band as "in band" and scored the same as a 60-note one.
GRADED = ("longest_run", "gap_le_half", "lateral")


def _credit(k: str, v: float, lo: float, hi: float) -> float:
    if k not in GRADED:
        return float(lo <= v <= hi)
    half = (hi - lo) / 2 or 1.0
    return max(0.0, min(1.0, 2.0 * (1.0 - abs(v - (lo + hi) / 2) / half)))


_CUT = {0: (0, 1), 1: (0, -1), 2: (-1, 0), 3: (1, 0), 4: (-1, 1), 5: (1, 1), 6: (-1, -1), 7: (1, -1)}


def _near_converging(l: Note, r: Note) -> bool:
    """Adjacent same-time pair, cuts >= 135 deg apart, at least one aimed at the other (not already head-on)."""
    if l.cut_direction not in _CUT or r.cut_direction not in _CUT or _converging(l, r):
        return False
    dx, dy = r.x - l.x, r.y - l.y
    if max(abs(dx), abs(dy)) != 1:
        return False
    lv, rv = _CUT[l.cut_direction], _CUT[r.cut_direction]
    cos = (lv[0] * rv[0] + lv[1] * rv[1]) / ((lv[0] ** 2 + lv[1] ** 2) * (rv[0] ** 2 + rv[1] ** 2)) ** 0.5
    return cos <= -0.7 and (lv[0] * dx + lv[1] * dy > 0 or rv[0] * -dx + rv[1] * -dy > 0)


def _converging(l: Note, r: Note) -> bool:
    if _cuts_through(l, r) or _cuts_through(r, l):
        return True
    if l.cut_direction not in _CUT or r.cut_direction not in _CUT:
        return False
    dx, dy = r.x - l.x, r.y - l.y
    if max(abs(dx), abs(dy)) > 1:
        return False
    if dx == 0 and dy == 0:
        return True
    lv, rv = _CUT[l.cut_direction], _CUT[r.cut_direction]
    return lv[0] * dx + lv[1] * dy > 0 and rv[0] * -dx + rv[1] * -dy > 0


def _tangle(l: Note, r: Note) -> bool:
    if l.x <= r.x or l.cut_direction not in _CUT or r.cut_direction not in _CUT:
        return False
    return _CUT[l.cut_direction][1] * _CUT[r.cut_direction][1] < 0


def _staircase(a: Note, b: Note) -> bool:
    dx, dy = b.x - a.x, b.y - a.y
    if abs(dx) != 1 or abs(dy) != 1 or a.cut_direction not in _CUT:
        return False
    v = _CUT[a.cut_direction]
    return (v[0], v[1]) not in ((dx, dy), (-dx, -dy))  # a cut along the diagonal is one clean swipe


def _loloppe(a: Note, b: Note) -> bool:
    """bsmg glossary: two same-direction blocks side by side that one swing can only hit by abusing the hitbox.

    Side by side = orthogonally adjacent with the shared cut running across the line that joins them (along
    it is a stack, which is one clean swing)."""
    dx, dy = b.x - a.x, b.y - a.y
    if abs(dx) + abs(dy) != 1 or a.cut_direction != b.cut_direction or a.cut_direction not in _CUT:
        return False
    v = _CUT[a.cut_direction]
    return v[0] * dx + v[1] * dy == 0


def _shrado(a: Note, b: Note) -> bool:
    """bsmg glossary: an outward-facing diagonal-down block in a far lane, then an up block of the same colour
    in a lane 2+ closer (3 lanes spanned); players cut the up block diagonally. ``a``, ``b`` consecutive same-hand swings."""
    out = (a.x == 0 and a.cut_direction == 6) or (a.x == 3 and a.cut_direction == 7)
    return out and b.cut_direction == 0 and abs(b.x - a.x) >= 2


def _cuts_through(a: Note, b: Note) -> bool:
    """a's blade path (1-2 cells along its cut) runs over b's cell: the saber hits the wrong colour first."""
    if a.cut_direction not in range(8):
        return False
    v = _CUT[a.cut_direction]
    return any((a.x + k * v[0], a.y + k * v[1]) == (b.x, b.y) for k in (1, 2))
AUDIO_METRICS = ("energy_corr", "onset_hit")


@dataclass
class RhythmReport:
    difficulty: str
    stats: dict[str, float]
    in_band: dict[str, bool]
    score: float  # 0..100

    def to_dict(self) -> dict:
        return {"difficulty": self.difficulty, "score": self.score, "stats": self.stats, "in_band": self.in_band}

    def table(self) -> str:
        bands = BANDS[self.difficulty]
        lines = [f"{'metric':16s} {'value':>8s}  {'band':>14s}"]
        for k, v in self.stats.items():
            lo, hi = bands[k]
            mark = " " if self.in_band[k] else ("<" if v < lo else ">")
            lines.append(f"{k:16s} {v:8.2f} {mark} {lo:6.2f}..{hi:<6.2f}")
        return "\n".join(lines)


def stats(notes: list[Note], bpm: float) -> dict[str, float]:
    n = [x for x in notes if 0 <= x.x <= 3 and 0 <= x.y <= 2]
    if len(n) < 2:
        return {k: 0.0 for k in WEIGHTS}
    t = np.array([x.time_seconds for x in n])
    dur = max(t[-1] - t[0], 1e-6)
    beat = 60.0 / bpm
    ts = Counter(np.round(t, 3))
    ut = np.array(sorted(ts))
    g = np.diff(ut) / beat if len(ut) > 1 else np.array([np.inf])
    runs, c = [], 1
    for x in g:
        if x <= 0.55:
            c += 1
        else:
            runs.append(c)
            c = 1
    runs.append(c)
    r = np.array(runs)
    hg = []
    for hand in (0, 1):
        ht = np.array(sorted({round(x.time_seconds, 3) for x in n if x.color == hand}))
        if len(ht) > 1:
            hg.append(np.diff(ht))
    hg = np.concatenate(hg) if hg else np.array([np.inf])
    vision = {(1, 1), (2, 1)}
    hit_vis = [any((x.x, x.y) in vision for x in n if round(x.time_seconds, 3) == tt) for tt in ut]
    run3 = 0
    run = 0
    for i, hv in enumerate(hit_vis):
        run = run + 1 if hv and (i == 0 or g[i - 1] <= 2.0) else (1 if hv else 0)
        run3 += run == 3
    by_t: dict[float, list[Note]] = {}
    for x in n:
        by_t.setdefault(round(x.time_seconds, 3), []).append(x)
    # Lateral motion per hand between its own consecutive swings within a beat: how far it moves
    # sideways, how often it stays put (inline), and monotonic 3+ lane runs (the piano-stream shape).
    dxs, inline, mono3, inline4, shrado = [], 0, 0, 0, 0
    for hand in (0, 1):
        h = sorted((x for x in n if x.color == hand), key=lambda x: x.time_seconds)
        run, lastd, parked = 1, 0, 1
        for a, b in zip(h, h[1:]):
            if (b.time_seconds - a.time_seconds) / beat > 1.01:
                run, lastd, parked = 1, 0, 1
                continue
            shrado += _shrado(a, b)
            d = b.x - a.x
            dxs.append(abs(d))
            inline += d == 0 and b.y == a.y
            parked = parked + 1 if d == 0 and b.y == a.y else 1  # inline spam: 4+ swings parked on one cell
            inline4 += parked == 4
            run = run + 1 if d and (lastd == 0 or (d > 0) == (lastd > 0)) else (2 if d else 1)
            lastd = d or lastd
            mono3 += run == 3
    pairs = max(len(dxs), 1)
    # Effective BPM (bsmg: the BPM at which the hand's swings would be 1/2 notes = one per beat): 60 / the
    # per-hand gap in seconds, p90 over swing gaps inside 2 s. Same-hand notes under 0.3 beat apart
    # (stacks, sliders, towers) are one swing.
    hgs = []
    for hand in (0, 1):
        swings = []
        for x in sorted((x for x in n if x.color == hand), key=lambda x: x.time_seconds):
            if not swings or x.time_seconds - swings[-1] >= 0.3 * beat:
                swings.append(x.time_seconds)
        hgs += [b - a for a, b in zip(swings, swings[1:]) if b - a <= 2.0]
    ebpm = float(np.percentile(60.0 / np.array(hgs), 90)) if hgs else 0.0
    # gallop (bsmg glossary): two singles then a double, all within half a beat of each other
    sizes = [ts[tt] for tt in ut]
    gallop = sum(sizes[i] == 1 and sizes[i + 1] == 1 and sizes[i + 2] > 1 and g[i] <= 0.55 and g[i + 1] <= 0.55
                 for i in range(len(sizes) - 2))
    # timing precision (bsmg Basic: Timing): most songs need only 1/1, 1/2, 1/4 (and triplets); 1/8 and finer is overmapping
    # Scored at the beat offset that fits best: many community maps carry a constant offset that is not precision.
    off = (np.array([x.beat for x in n])[None, :] - np.arange(0, 1, 0.01)[:, None]) % 1.0  # (offsets, notes); not `g`, the note gaps
    on_grid = (np.abs(off * 4 - np.round(off * 4)) < 0.05) | (np.abs(off * 3 - np.round(off * 3)) < 0.05)
    fine = float(1.0 - on_grid.mean(axis=1).max())
    # arm tangle: red right of blue at the same time with opposite vertical cuts, so the arms cross and pull apart
    tangle = sum(_tangle(l, r) for ns in by_t.values() for l in ns if l.color == 0 for r in ns if r.color == 1)
    # staircase hitbox abuse: two same-colour notes at one time on a diagonal, cut across the diagonal
    staircase = sum(_staircase(a, b) for ns in by_t.values() for a in ns for b in ns if a is not b and a.color == b.color) // 2
    loloppe = sum(_loloppe(a, b) for ns in by_t.values() for a in ns for b in ns if a is not b and a.color == b.color) // 2
    conv = sum(_converging(l, r) for ns in by_t.values() for l in ns if l.color == 0 for r in ns if r.color == 1)
    near = sum(_near_converging(l, r) for ns in by_t.values() for l in ns if l.color == 0 for r in ns if r.color == 1)
    return {
        "near_converging": near / max(len(ut), 1),
        "converging": conv / max(len(ut), 1),
        "vision": sum((x.x, x.y) in vision for x in n) / len(n),
        "vision_run3": run3 / max(len(ut), 1),
        "nps": len(n) / dur,
        "gap_le_half": float(np.mean(g <= 0.55)),
        "gap_1": float(np.mean(np.isclose(g, 1, atol=0.13))),
        "gap_ge2": float(np.mean(g >= 2)),
        "streams_per_min": float(np.sum(r >= 4) / (dur / 60)),
        "longest_run": float(r.max()),
        "hand_gap_p10": float(np.percentile(hg, 10)),
        "doubles": sum(1 for v in ts.values() if v > 1) / len(ts),
        "dots": sum(x.cut_direction == 8 for x in n) / len(n),
        "top_row": sum(x.y == 2 for x in n) / len(n),
        "outer_cols": sum(x.x in (0, 3) for x in n) / len(n),
        "far_cross": sum((x.color == 0 and x.x == 3) or (x.color == 1 and x.x == 0) for x in n) / len(n),
        "updown": sum(x.cut_direction in (0, 1) for x in n) / len(n),
        "lateral": float(np.mean(dxs)) if dxs else 0.0,
        "inline": inline / pairs,
        "zigzag3": 100.0 * mono3 / pairs,
        "ebpm": ebpm,
        "tangle": tangle / max(len(ut), 1),
        "gallop": 100.0 * gallop / max(len(ut), 1),
        "inline4": 100.0 * inline4 / pairs,
        "staircase": staircase / max(len(ut), 1),
        "loloppe": loloppe / max(len(ut), 1),
        "shrado": 100.0 * shrado / pairs,
        "fine": fine,
    }


def score(notes: list[Note], bpm: float, difficulty: str = "Expert", extra: dict[str, float] | None = None) -> RhythmReport:
    """`extra` carries metrics computed elsewhere (beat_sim.music stats); missing ones don't count."""
    difficulty = difficulty if difficulty in BANDS else "Expert"
    st = stats(notes, bpm)
    if notes:
        st["lead_in"] = float(min(n.time_seconds for n in notes))
    st.update(extra or {})
    bands = BANDS[difficulty]
    inb = {k: bool(bands[k][0] <= v <= bands[k][1]) for k, v in st.items()}
    total = sum(WEIGHTS[k] for k in st)
    return RhythmReport(difficulty, st, inb, 100.0 * sum(WEIGHTS[k] * _credit(k, v, *bands[k]) for k, v in st.items()) / total)


if __name__ == "__main__":
    # 12-note half-beat streams separated by a 1-beat rest: dense, with runs, no doubles, no top row
    notes = []
    for bar in range(40):
        for i in range(12):
            t = bar * 3.5 + i * 0.25
            notes.append(Note(beat=t * 2, time_seconds=t, x=1 + i % 2, y=1, color=i % 2, cut_direction=(1, 0)[(i // 2) % 2]))
    r = score(notes, 120.0)
    assert r.in_band["nps"] and r.in_band["streams_per_min"] and r.in_band["longest_run"], r.table()
    assert not r.in_band["doubles"] and not r.in_band["top_row"], "a pure stream has no doubles and no top row"
    assert _credit("longest_run", 110, 15, 205) == 1.0 and _credit("longest_run", 210, 15, 205) == 0.0 and 0 < _credit("longest_run", 180, 15, 205) < 1
    sparse = score(notes[::4], 120.0)
    assert sparse.score < r.score and not sparse.in_band["nps"]
    assert "energy_corr" not in r.stats and r.in_band["lead_in"] is False, "no audio: no music metrics; lead-in 0 s is out of band"
    withaudio = score(notes, 120.0, extra={"energy_corr": 0.5, "onset_hit": 0.6})
    assert withaudio.in_band["energy_corr"] and withaudio.score > r.score
    print("rhythm ok")
    print(r.table())
