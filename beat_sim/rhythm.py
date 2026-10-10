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
# Music metrics and lead_in come from the audio subset of the favourites (calibrate_music). from "Beat Saber Favorites-2026.10.05".
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
        'updown': (0.590, 0.819),
        'energy_corr': (0.32, 1.0),
        'onset_hit': (0.42, 1.0),
        'lead_in': (1.7, 12.0),
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
        'energy_corr': (0.32, 1.0),
        'onset_hit': (0.42, 1.0),
        'lead_in': (1.7, 12.0),
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
        'energy_corr': (0.32, 1.0),
        'onset_hit': (0.42, 1.0),
        'lead_in': (1.7, 12.0),
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
        'energy_corr': (0.32, 1.0),
        'onset_hit': (0.42, 1.0),
        'lead_in': (1.7, 12.0),
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
        'energy_corr': (0.28, 1.0),
        'onset_hit': (0.4, 1.0),
        'lead_in': (1.7, 11.0),
    },
}

WEIGHTS = {  # what matters most, per the alignment discussion
    "nps": 3, "gap_le_half": 2, "streams_per_min": 2, "longest_run": 1, "gap_1": 1, "gap_ge2": 1, "hand_gap_p10": 1,
    "doubles": 1, "dots": 2, "far_cross": 2, "top_row": 1, "outer_cols": 1, "updown": 1,
    # only scored when audio is available (see score(extra=...)); dropped from the denominator otherwise
    "energy_corr": 3, "onset_hit": 2, "lead_in": 1,
    "vision": 2, "vision_run3": 2,  # notes in the middle-row centre cells (hide what follows), and runs of 3+ hits there
    "converging": 2,  # same-time pairs whose cuts drive the sabers into each other (favourites median 0)
    "near_converging": 1,  # 45 deg off head-on with one cut aimed at the other note; legal but very hard
}
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
    if l.cut_direction not in _CUT or r.cut_direction not in _CUT:
        return False
    dx, dy = r.x - l.x, r.y - l.y
    if max(abs(dx), abs(dy)) > 1:
        return False
    if dx == 0 and dy == 0:
        return True
    lv, rv = _CUT[l.cut_direction], _CUT[r.cut_direction]
    return lv[0] * dx + lv[1] * dy > 0 and rv[0] * -dx + rv[1] * -dy > 0
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
    return RhythmReport(difficulty, st, inb, 100.0 * sum(WEIGHTS[k] for k, ok in inb.items() if ok) / total)


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
    sparse = score(notes[::4], 120.0)
    assert sparse.score < r.score and not sparse.in_band["nps"]
    assert "energy_corr" not in r.stats and r.in_band["lead_in"] is False, "no audio: no music metrics; lead-in 0 s is out of band"
    withaudio = score(notes, 120.0, extra={"energy_corr": 0.5, "onset_hit": 0.6})
    assert withaudio.in_band["energy_corr"] and withaudio.score > r.score
    print("rhythm ok")
    print(r.table())
