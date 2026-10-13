"""Rules-based lightshow for v2 maps (bsmg.wiki basic-lighting): lights follow the notes and the song's energy.

Event types: 0 back lasers, 1 ring lights, 2/3 left/right rotating lasers, 4 centre lights, 8 ring spin,
9 ring zoom, 12/13 laser speeds. Values: 0 off, 1/5 blue/red on, 2/6 flash, 3/7 fade.
"""

from __future__ import annotations

import numpy as np

from beat_weaver.schemas.normalized import Note

ON, FLASH, FADE = 1, 2, 3  # add RED to colour the event red
RED = 4
BACK, RING, LASER_L, LASER_R, CENTRE, SPIN, ZOOM, SPEED_L, SPEED_R = 0, 1, 2, 3, 4, 8, 9, 12, 13


def _ev(beat: float, typ: int, val: int) -> dict:
    return {"_time": round(beat, 3), "_type": typ, "_value": val}


def light_events(notes: list[Note], bpm: float, strength: np.ndarray, hop_s: float) -> list[dict]:
    """Lightshow from the notes and a loudness envelope.

    Per hit: back lasers flash in the hit's colour; left/right rotating lasers flash with their hand.
    Per bar: ring lights fade in the bar's dominant colour, brighter bars (top third) also light the
    centre. Every 8 bars the rings spin; a loudness jump zooms them; laser speed tracks density.
    Lights go out in rests longer than 2 beats.
    """
    if not notes:
        return []
    beat_s = 60.0 / bpm
    ev: list[dict] = []
    hits: dict[float, list[Note]] = {}
    for n in notes:
        hits.setdefault(round(n.beat, 3), []).append(n)
    times = sorted(hits)
    # bar loudness from the envelope
    n_bars = int(times[-1] / 4) + 2
    bar_loud = np.zeros(n_bars)
    for b in range(n_bars):
        lo, hi = int(b * 4 * beat_s / hop_s), int((b + 1) * 4 * beat_s / hop_s)
        seg = strength[lo:hi]
        bar_loud[b] = float(seg.mean()) if len(seg) else 0.0
    loud_cut = np.quantile(bar_loud[bar_loud > 0], 0.67) if (bar_loud > 0).any() else 1e9
    prev_t = None
    for t in times:
        h = hits[t]
        red = sum(n.color == 0 for n in h) >= sum(n.color == 1 for n in h)
        ev.append(_ev(t, BACK, FLASH + (RED if red else 0)))
        for n in h:
            ev.append(_ev(t, LASER_L if n.color == 0 else LASER_R, FLASH + (RED if n.color == 0 else 0)))
        if prev_t is not None and t - prev_t > 2.0:  # rest: lights out after the last hit
            ev.append(_ev(prev_t + 1.0, BACK, 0))
            ev.append(_ev(prev_t + 1.0, CENTRE, 0))
        prev_t = t
    # bars
    first_bar, last_bar = int(times[0] // 4), int(times[-1] // 4)
    for b in range(first_bar, last_bar + 1):
        bar_hits = [n for t in times if b * 4 <= t < (b + 1) * 4 for n in hits[t]]
        if not bar_hits:
            ev.append(_ev(b * 4, RING, 0))
            continue
        red = sum(n.color == 0 for n in bar_hits) >= len(bar_hits) / 2
        ev.append(_ev(b * 4, RING, FADE + (RED if red else 0)))
        if bar_loud[b] >= loud_cut:
            ev.append(_ev(b * 4, CENTRE, ON + (RED if not red else 0)))  # opposite colour to the rings
        elif b > 0 and bar_loud[b - 1] >= loud_cut:
            ev.append(_ev(b * 4, CENTRE, 0))
        if (b - first_bar) % 8 == 0:
            ev.append(_ev(b * 4, SPIN, 0))
            speed = int(np.clip(len(bar_hits) / 2, 1, 8))
            ev.append(_ev(b * 4, SPEED_L, speed))
            ev.append(_ev(b * 4, SPEED_R, speed))
        if b > 0 and bar_loud[b - 1] > 0 and bar_loud[b] > 1.6 * bar_loud[b - 1]:  # drop
            ev.append(_ev(b * 4, ZOOM, 0))
    end = times[-1] + 2.0
    ev += [_ev(end, t, 0) for t in (BACK, RING, CENTRE, LASER_L, LASER_R)]
    return sorted(ev, key=lambda e: (e["_time"], e["_type"]))
