"""Post-decode shaping of a note list, following bsmg.wiki mapping guidelines.

Three rule-based passes that run on decoded notes (no torch):

* ``downmap``        — derive a lower difficulty from a higher one by thinning (bsmg "Downmapping").
* ``repeat_sections`` — detect repeated song sections from the audio and copy the first occurrence's
  notes onto the repeats (bsmg "Consistency": similar sections, similar mapping).
* ``emphasize``      — turn singles on the loudest onsets into stacks, and notes on sustained sounds
  into short sliders (bsmg "Emphasis": baseline single, bigger sound bigger swing).

All passes return a new sorted list and never move a note in time.
"""
from __future__ import annotations

import collections
import math
from dataclasses import replace

import numpy as np

from beat_weaver.schemas.normalized import Note

CUT_VEC = {0: (0, 1), 1: (0, -1), 2: (-1, 0), 3: (1, 0), 4: (-1, 1), 5: (1, 1), 6: (-1, -1), 7: (1, -1), 8: (0, 0)}
VISION_CELLS = {(1, 1), (2, 1)}
DIFFS = ("Easy", "Normal", "Hard", "Expert", "ExpertPlus")
# Favourites' notes-per-second band midpoints (beat_sim/rhythm.py BANDS). bsmg says ~20% fewer notes per
# step, but the reference maps step harder at the bottom (Hard -> Normal -> Easy is x0.69, x0.65), so a
# downmap keeps the ratio of these targets rather than a flat 0.8.
TARGET_NPS = {"Easy": 1.50, "Normal": 2.23, "Hard": 3.25, "Expert": 4.16, "ExpertPlus": 5.22}
# Easy sits near the bottom of the favourites band (1.09-1.83) rather than its midpoint: the favourites are an
# Expert+ player's picks and their Easy runs well above official Easy (~0.8/s); play test said it felt hard.
_REVERSE = {0: 1, 1: 0, 2: 3, 3: 2, 4: 7, 7: 4, 5: 6, 6: 5, 8: 8}



def _sorted(notes: list[Note]) -> list[Note]:
    return sorted(notes, key=lambda n: (n.time_seconds, n.color, n.x, n.y))


def _hits(notes: list[Note]) -> list[list[Note]]:
    """Notes grouped by time (one swing moment), in order."""
    by_t: dict[float, list[Note]] = {}
    for n in notes:
        by_t.setdefault(round(n.time_seconds, 3), []).append(n)
    return [by_t[t] for t in sorted(by_t)]


# --- 1. downmapping ----------------------------------------------------------------------------------------

def downmap(notes: list[Note], bpm: float, src: str, dst: str, strength: np.ndarray | None = None,
            hop_s: float = 0.0) -> list[Note]:
    """Thin ``notes`` (difficulty ``src``) to difficulty ``dst``, keeping the strongest moments.

    ``strength`` is an optional per-frame onset/loudness array with ``hop_s`` seconds per frame; a hit
    scores by it so loud beats survive. Without it, on-beat hits survive first, then half beats.
    Hits are removed whole (both hands) so parity is untouched, and never two in a row inside a beat so
    streams become half-time streams rather than ragged ones. ``gate_patterns`` then reshapes anything above
    ``dst``'s tier.
    """
    if DIFFS.index(src) <= DIFFS.index(dst):
        return _sorted(notes)
    beat = 60.0 / bpm
    hits = _hits(notes)
    # aim at dst's absolute density; if src is lighter than its own band, fall back to the band ratio
    span = max(hits[-1][0].time_seconds - hits[0][0].time_seconds, 1.0) if hits else 1.0
    ratio = min(TARGET_NPS[dst] * span / max(len(notes), 1), TARGET_NPS[dst] / TARGET_NPS[src])
    target = int(round(len(hits) * ratio))

    def score(h: list[Note]) -> float:
        t = h[0].time_seconds
        pos = (t / beat) % 1.0
        bar_pos = int(t / beat) % 4
        # metric hierarchy: downbeat > beat 3 > beats 2/4 > half beats > the rest
        grid = (1.0 if bar_pos == 0 else 0.75 if bar_pos == 2 else 0.5) if min(pos, 1 - pos) < 0.08 \
            else 0.25 if abs(pos - 0.5) < 0.08 else 0.0
        s = grid + 0.25 * (len(h) > 1)
        if strength is not None and hop_s > 0:
            i = min(int(t / hop_s), len(strength) - 1)
            s += float(strength[i]) / (float(strength.max()) or 1.0)
        return s

    keep = [True] * len(hits)
    # ties thin evenly across the song (golden-ratio spread) instead of front to back
    order = sorted(range(len(hits)), key=lambda i: (score(hits[i]), (i * 0.6180339887) % 1.0))
    order = [i for i in order if i != 0][: len(hits) - target]  # removal candidates; the first hit and the key moments stay
    removed = 0
    # two passes: first never remove neighbours (keeps streams even), then whatever is left
    for strict in (True, False):
        for i in order:
            if removed >= len(hits) - target:
                break
            if not keep[i]:
                continue
            if strict:
                near = [j for j in (i - 1, i + 1) if 0 <= j < len(hits)
                        and abs(hits[j][0].time_seconds - hits[i][0].time_seconds) <= beat * 0.55]
                if any(not keep[j] for j in near):
                    continue
            keep[i] = False
            removed += 1
    out = [n for i, h in enumerate(hits) if keep[i] for n in h]

    return gate_patterns(out, bpm, dst)


def _converging(l: Note, r: Note) -> bool:
    """Same-time pair whose cuts drive the sabers into each other (mirrors beat_sim.rhythm._converging),
    or where one saber must arrive through the other note's cell (``_approaches_through``)."""
    if _cuts_through(l, r) or _cuts_through(r, l) or _approaches_through(l, r) or _approaches_through(r, l):
        return True
    if l.cut_direction == 8 or r.cut_direction == 8:
        return False
    dx, dy = r.x - l.x, r.y - l.y
    if max(abs(dx), abs(dy)) > 1:
        return False
    if dx == 0 and dy == 0:
        return True
    lv, rv = CUT_VEC[l.cut_direction], CUT_VEC[r.cut_direction]
    return lv[0] * dx + lv[1] * dy > 0 and rv[0] * -dx + rv[1] * -dy > 0


def _cuts_through(a: Note, b: Note) -> bool:
    """a's blade path (1-2 cells along its cut) runs over b's cell: the saber hits the wrong colour first."""
    if a.cut_direction not in range(8):
        return False
    v = CUT_VEC[a.cut_direction]
    return any((a.x + k * v[0], a.y + k * v[1]) == (b.x, b.y) for k in (1, 2))


def _approaches_through(a: Note, b: Note) -> bool:
    """a's saber arrives from the cell behind it (one step against its cut) and b sits there: a red up-cut
    over a blue down-cut has each saber come through the other note (seen in play testing). Real Expert
    maps: median 0, p99 0.5% of their doubles."""
    if a.cut_direction not in range(8):
        return False
    v = CUT_VEC[a.cut_direction]
    return (a.x - v[0], a.y - v[1]) == (b.x, b.y)


def repair_parity(notes: list[Note], bpm: float, window_beats: float = 2.0) -> list[Note]:
    """Reverse any cut that repeats the same hand's previous cut (within 45 deg) inside ``window_beats``.

    Thinning removes the swing between two same-direction swings and leaves a double directional;
    the wiki calls these resets and tolerates them only after ~1.5 beats. Dots are left alone.
    """
    beat = 60.0 / bpm
    out: list[Note] = []
    last: dict[int, Note] = {}
    for h in _hits(_sorted(notes)):
        for color in (0, 1):
            grp = [n for n in h if n.color == color]  # a stack is one swing: reverse all of it or none
            if not grp:
                continue
            p = last.get(color)
            lead = grp[0]
            if p and lead.cut_direction != 8 and p.cut_direction != 8 and lead.time_seconds - p.time_seconds < window_beats * beat:
                a, b = CUT_VEC[p.cut_direction], CUT_VEC[lead.cut_direction]
                if a[0] * b[0] + a[1] * b[1] > 0.5:
                    grp = [replace(n, cut_direction=_REVERSE[n.cut_direction]) for n in grp]
            out.extend(grp)
            last[color] = grp[0]
    return _sorted(out)


def downmap_ladder(top_notes: list[Note], bpm: float, top: str, wanted: list[str], **kw) -> dict[str, list[Note]]:
    """{difficulty: notes} for every difficulty in ``wanted`` at or below ``top``, derived from ``top_notes``."""
    out = {top: _sorted(top_notes)}
    prev = top
    for d in reversed(DIFFS[: DIFFS.index(top)]):
        if d in wanted:
            out[d] = downmap(out[prev], bpm, prev, d, **kw)
            prev = d
    return {d: out[d] for d in wanted if d in out}


# --- 1b. pattern ladder --------------------------------------------------------------------------------------
# What each difficulty may contain, cumulative up the ladder (bsmg basic-mapping "Difficulty", downmapping).
#   cuts        allowed cut directions (0 up, 1 down, 2 left, 3 right, 4-7 diagonals, 8 dot)
#   own_half    each hand stays on its side of the grid (no crossovers, inward or outward)
#   top_row     may use y=2
#   stacks      same-colour note groups (emphasis)
#   sliders     same-colour dot groups
#   max_doubles share of swings that are doubles (None = no cap)
#   max_jump    lanes a hand may move between consecutive swings
#   min_gap     beats between consecutive swings (either hand); closer hits are dropped
#   towers      3-note same-colour stacks and 2-note windows (one-cell gap) on the biggest moments
#   max_cells   distinct (hand, cell) placements per 2 bars; extras move to the hand's commonest cell
#               (bsmg downmapping: 2-3 unique patterns per half-jump at Easy, 4-5 at Normal)
TIERS = {
    "Easy": dict(max_cells=3, cuts={0, 1, 8}, own_half=True, top_row=False, stacks=False, sliders=False, max_doubles=0.18, max_jump=1, min_gap=1.0, towers=False),
    "Normal": dict(max_cells=5, cuts={0, 1, 2, 3, 8}, own_half=True, top_row=False, stacks=False, sliders=False, max_doubles=0.20, max_jump=2, min_gap=0.5, towers=False),
    "Hard": dict(max_cells=None, cuts=set(range(9)), own_half=False, top_row=True, stacks=True, sliders=False, max_doubles=None, max_jump=3, min_gap=0.0, towers=False),
    "Expert": dict(max_cells=None, cuts=set(range(9)), own_half=False, top_row=True, stacks=True, sliders=True, max_doubles=None, max_jump=3, min_gap=0.0, towers=False),
    "ExpertPlus": dict(max_cells=None, cuts=set(range(9)), own_half=False, top_row=True, stacks=True, sliders=True, max_doubles=None, max_jump=3, min_gap=0.0, towers=True),
}


def _simplify_cut(cut: int, allowed: set[int], prev: int | None) -> int:
    """Nearest allowed cut: diagonals lose their sideways part, horizontals alternate up/down with the hand's
    previous cut (a dot if nothing vertical is allowed)."""
    if cut in allowed:
        return cut
    if cut >= 4:
        v = 0 if CUT_VEC[cut][1] > 0 else 1
        h = 3 if CUT_VEC[cut][0] > 0 else 2
        # A downward diagonal after a vertical cut keeps its sideways part where the tier allows it (Normal);
        # otherwise Normal came out 98% up/down against 57-83% in the favourites.
        if v == 1 and prev in (0, 1) and h in allowed:
            return h
        if v in allowed:
            return v
    if cut in (2, 3) and {0, 1} <= allowed:
        return _REVERSE[prev] if prev in (0, 1) else 1
    return 8


def _limit_cells(notes: list[Note], bpm: float, max_cells: int, window_beats: float = 8.0) -> list[Note]:
    """Per window, keep at most ``max_cells`` distinct (hand, cell) placements: the rarest move to the hand's
    commonest cell in that window, unless that cell is taken by the other hand at the same time."""
    beat = 60.0 / bpm
    out = list(notes)
    by_window: dict[int, list[int]] = {}
    for i, n in enumerate(out):
        by_window.setdefault(int(n.time_seconds / beat // window_beats), []).append(i)
    for idx in by_window.values():
        counts: dict[tuple[int, int, int], int] = {}
        for i in idx:
            n = out[i]
            counts[(n.color, n.x, n.y)] = counts.get((n.color, n.x, n.y), 0) + 1
        keep = set(sorted(counts, key=lambda k: -counts[k])[:max_cells])
        for i in idx:
            n = out[i]
            if (n.color, n.x, n.y) in keep:
                continue
            home = max((k for k in keep if k[0] == n.color), key=lambda k: counts[k], default=None)
            if home is None:
                continue
            same_t = {(m.x, m.y) for m in out if abs(m.time_seconds - n.time_seconds) < 1e-3 and m is not n}
            if (home[1], home[2]) not in same_t:
                out[i] = replace(n, x=home[1], y=home[2])
    return out


def gate_patterns(notes: list[Note], bpm: float, difficulty: str) -> list[Note]:
    """Convert anything above ``difficulty``'s tier in ``TIERS`` into the nearest allowed pattern.

    Notes are kept (the rhythm is the map) and reshaped: cuts simplified, crossovers moved back to the
    hand's side, top row pulled down, stacks and sliders collapsed, doubles thinned, big lane jumps
    shortened. Parity is repaired afterwards.
    """
    t = TIERS[difficulty]
    beat = 60.0 / bpm
    hits = _hits(_sorted(notes))
    if t["min_gap"] > 0:  # hits too close to the previous kept one go
        kept, last_t = [], -1e9
        for h in hits:
            if h[0].time_seconds - last_t >= t["min_gap"] * beat - 1e-3:
                kept.append(h)
                last_t = h[0].time_seconds
        hits = kept
    # stacks / sliders: keep the first note of a same-colour group
    for h in hits:
        for color in (0, 1):
            grp = [n for n in h if n.color == color]
            if len(grp) > 1 and (not t["stacks"] or (not t["sliders"] and all(n.cut_direction == 8 for n in grp))):
                for n in grp[1:]:
                    h.remove(n)
            elif len(grp) > 1 and not t["towers"]:
                wide = max(max(abs(a.x - b.x), abs(a.y - b.y)) for a in grp for b in grp) > 1  # diagonal stacks are adjacent
                for n in grp[2 if not wide else 1:]:  # towers to stacks, windows to singles
                    h.remove(n)
    # doubles: thin to the cap, alternating which hand keeps the note
    if t["max_doubles"] is not None:
        doubles = [i for i, h in enumerate(hits) if len({n.color for n in h}) > 1]
        extra = len(doubles) - int(round(t["max_doubles"] * len(hits)))
        if extra > 0:
            for i in doubles[:: max(1, len(doubles) // extra)][:extra]:
                hits[i] = [n for n in hits[i] if n.color == i % 2] or hits[i][:1]
    out: list[Note] = []
    last: dict[int, Note] = {}
    for h in hits:
        taken: set[tuple[int, int]] = set()
        placed = []
        for n in sorted(h, key=lambda n: n.color):
            p = last.get(n.color)
            cut = _simplify_cut(n.cut_direction, t["cuts"], p.cut_direction if p else None)
            x, y = n.x, n.y
            if t["own_half"]:
                x = min(x, 1) if n.color == 0 else max(x, 2)
            if p and abs(x - p.x) > t["max_jump"]:
                x = p.x + (t["max_jump"] if x > p.x else -t["max_jump"])
            if not t["top_row"] and y == 2:
                y = 1 if x in (0, 3) else 0
                cut = cut if cut in (0, 1, 8) else 8
            if (x, y) in VISION_CELLS and (x, y) != (n.x, n.y):
                y = 0  # a moved note must not become a vision block
            if (x, y) in taken:
                continue  # landed on the other hand's note; this one goes
            taken.add((x, y))
            placed.append(replace(n, x=x, y=y, cut_direction=cut))
        for n in placed:
            last[n.color] = n
        out.extend(placed)
    if t["max_cells"]:
        out = _limit_cells(out, bpm, t["max_cells"])
    out = repair_parity(_sorted(out), bpm)
    return _sorted(n for h in _resolve_collisions(_hits(out)) for n in h)


def _resolve_collisions(hits: list[list[Note]]) -> list[list[Note]]:
    """No same-time pair may converge or cut through the other colour (emphasis, repeats and parity repair
    can all create one). Try, in order: straighten diagonals, reverse one offender's cut, drop the extra
    notes of a same-colour group, drop the offender."""
    def bad(h):
        return [(l, r) for l in h if l.color == 0 for r in h if r.color == 1 if _converging(l, r)]
    for h in hits:
        if not bad(h):
            continue
        for k, n in enumerate(h):  # 1. diagonals to straight
            if 4 <= n.cut_direction <= 7:
                h[k] = replace(n, cut_direction=0 if CUT_VEC[n.cut_direction][1] > 0 else 1)
        for l, r in bad(h):  # 2. reverse the cut that runs through the other note
            for n in (l, r):
                other = r if n is l else l
                if (_cuts_through(n, other) or _approaches_through(n, other)) and n in h:
                    cand = replace(n, cut_direction=_REVERSE[n.cut_direction])
                    if not _converging(cand, other) and not _converging(other, cand):
                        h[h.index(n)] = cand
        if bad(h):  # 3. a same-colour group (stack/tower) keeps only its first note
            for color in (0, 1):
                grp = [n for n in h if n.color == color]
                for n in grp[1:]:
                    h.remove(n)
        while bad(h):  # 4. drop the offender (the note whose blade runs through the other)
            l, r = bad(h)[0]
            h.remove(l if _cuts_through(l, r) or _approaches_through(l, r) else r)
    return hits


# --- 2. section repeats -------------------------------------------------------------------------------------

def find_repeats(feat: np.ndarray, hop_s: float, bpm: float, bars: int = 4, min_sim: float = 0.7) -> list[tuple[float, float]]:
    """Repeated sections in a (frames, dims) feature matrix, as (source_start_s, repeat_start_s) pairs.

    The song is cut into ``bars``-bar blocks on the beat grid; a block whose beat-synchronous feature
    sequence has cosine similarity >= ``min_sim`` to an earlier block is a repeat of it. Earliest match
    wins. Blocks never overlap, so the output is safe to apply left to right.

    Measured over 45 songs: unrelated 4-bar blocks sit at ~0 (p90 0.25, p99 under 0.6), repeats at
    0.7-0.95; 8-bar blocks blur repeats to 0.5-0.6, so 4 bars and 0.7 are the defaults.
    """
    beat = 60.0 / bpm
    block_s = bars * 4 * beat
    n_blocks = int(len(feat) * hop_s // block_s)
    if n_blocks < 2:
        return []
    per_beat = max(1, int(round(beat / hop_s)))
    vecs = []
    for b in range(n_blocks):
        f0 = int(round(b * block_s / hop_s))
        seg = feat[f0: f0 + int(round(block_s / hop_s))]
        nb = len(seg) // per_beat
        v = seg[: nb * per_beat].reshape(nb, per_beat, -1).mean(axis=1)
        v = v - v.mean(axis=0, keepdims=True)  # ignore absolute loudness; compare shape
        v = v.ravel()
        vecs.append(v / (np.linalg.norm(v) or 1.0))
    L = min(len(v) for v in vecs)
    pairs = []
    for b in range(1, n_blocks):
        for a in range(b):
            if float(vecs[a][:L] @ vecs[b][:L]) >= min_sim:
                pairs.append((a * block_s, b * block_s))
                break
    return pairs


def repeat_sections(notes: list[Note], bpm: float, pairs: list[tuple[float, float]], bars: int = 4,
                    mirror_odd: bool = True) -> list[Note]:
    """Replace the notes inside each repeat block with the source block's notes, shifted in time.

    ``pairs`` come from :func:`find_repeats`. Odd-numbered repeats are mirrored left/right (bsmg: copy or
    mirror, keep the motif) so the song does not play literally the same twice. Note count is kept
    close to the original block's so density still follows the song: if the repeat block had fewer
    notes than the source, the source is thinned to match.
    """
    if not pairs:
        return _sorted(notes)
    beat = 60.0 / bpm
    block_s = bars * 4 * beat
    out = list(notes)
    for k, (src_t, dst_t) in enumerate(pairs):
        src = [n for n in notes if src_t <= n.time_seconds < src_t + block_s]
        dst = [n for n in out if dst_t <= n.time_seconds < dst_t + block_s]
        if not src:
            continue
        out = [n for n in out if not (dst_t <= n.time_seconds < dst_t + block_s)]
        shift = dst_t - src_t
        moved = [replace(n, beat=n.beat + shift / beat, time_seconds=n.time_seconds + shift) for n in src]
        if mirror_odd and k % 2 == 1:
            moved = [mirror(n) for n in moved]
        if 0 < len(dst) < len(moved):
            hits = _hits(moved)
            keep = int(round(len(hits) * len(dst) / len(moved)))
            idx = set(np.linspace(0, len(hits) - 1, keep).round().astype(int).tolist())
            moved = [n for i, h in enumerate(hits) if i in idx for n in h]
        out.extend(moved)
    return _sorted(out)


_MIRROR_DIR = {0: 0, 1: 1, 2: 3, 3: 2, 4: 5, 5: 4, 6: 7, 7: 6, 8: 8}


def mirror(n: Note) -> Note:
    return replace(n, x=3 - n.x, color=1 - n.color, cut_direction=_MIRROR_DIR[n.cut_direction])


# --- 3. emphasis ----------------------------------------------------------------------------------------------

def emphasize(notes: list[Note], bpm: float, strength: np.ndarray, hop_s: float, top_frac: float = 0.08,
              sustain_frac: float = 0.0, min_gap_beats: float = 0.9, difficulty: str = "Expert") -> list[Note]:
    """Stacks on the loudest ``top_frac`` of hits; short dot sliders on sustained sounds; on tiers with
    ``towers`` (Expert+) the loudest quarter of those become towers (3 in a column, vertical cuts, outer
    columns so no vision cell) or windows (a one-cell gap along a horizontal cut).

    A stack is a second same-colour, same-direction note adjacent to the first along the cut axis
    (bsmg "Stack": a bigger, faster swing). Only hits with at least ``min_gap_beats`` of room before
    and after get one, so stacks mark peaks rather than clutter streams; vision cells and the other
    hand's cell are never used. ``sustain_frac`` of hits whose loudness stays high for the next half
    beat (a held note) get a 2-dot slider continuing along the cut instead; 0 disables sliders.
    """
    hits = _hits(notes)
    if len(hits) < 4 or not len(strength):
        return _sorted(notes)
    beat = 60.0 / bpm
    smax = float(strength.max()) or 1.0

    def loud(t: float, span: float = 0.0) -> float:
        i0 = min(int(t / hop_s), len(strength) - 1)
        i1 = min(int((t + span) / hop_s) + 1, len(strength))
        return float(strength[i0: max(i1, i0 + 1)].mean()) / smax

    times = [h[0].time_seconds for h in hits]
    room = [
        (i == 0 or times[i] - times[i - 1] >= min_gap_beats * beat)
        and (i == len(hits) - 1 or times[i + 1] - times[i] >= min_gap_beats * beat)
        for i in range(len(hits))
    ]
    ranked = sorted((i for i in range(len(hits)) if room[i]), key=lambda i: -loud(times[i]))
    n_stack = int(round(top_frac * len(hits)))
    stack_idx = set(ranked[:n_stack])
    big_idx = set(ranked[: n_stack // 4]) if TIERS[difficulty]["towers"] else set()
    slide_idx: set[int] = set()
    if sustain_frac > 0:
        held = sorted((i for i in range(len(hits)) if room[i] and i not in stack_idx),
                      key=lambda i: -loud(times[i], 0.5 * beat))
        slide_idx = {i for i in held[: int(round(sustain_frac * len(hits)))] if loud(times[i], 0.5 * beat) > 0.6 * loud(times[i])}

    out = list(notes)
    for i, h in enumerate(hits):
        occupied = {(n.x, n.y) for n in h}
        for n in h:
            if n.cut_direction == 8:
                continue
            dx, dy = CUT_VEC[n.cut_direction]
            if i in big_idx and n.cut_direction <= 1 and n.x in (0, 3):  # tower: the whole outer column
                for yy in (0, 1, 2):
                    if (n.x, yy) not in occupied:
                        out.append(replace(n, y=yy))
                        occupied.add((n.x, yy))
                continue
            if i in big_idx and n.cut_direction in (2, 3):  # window: same row, one cell gap, along the cut
                for cell in ((n.x + 2 * dx, n.y), (n.x - 2 * dx, n.y)):
                    if 0 <= cell[0] <= 3 and cell not in occupied and cell not in VISION_CELLS:
                        out.append(replace(n, x=cell[0]))
                        occupied.add(cell)
                        break
                continue
            if i in stack_idx and n.cut_direction <= 3:  # straight cuts only: a diagonal stack reads as a wide double
                # second note sits in line with the first along the cut so one swipe covers both
                for cell in ((n.x - dx, n.y - dy), (n.x + dx, n.y + dy)):
                    if 0 <= cell[0] <= 3 and 0 <= cell[1] <= 2 and cell not in occupied and cell not in VISION_CELLS:
                        out.append(replace(n, x=cell[0], y=cell[1]))
                        occupied.add(cell)
                        break
            elif i in slide_idx:
                step = beat / 8
                cell = (n.x + dx, n.y + dy)
                if 0 <= cell[0] <= 3 and 0 <= cell[1] <= 2 and cell not in occupied and cell not in VISION_CELLS:
                    out.append(replace(n, x=cell[0], y=cell[1], cut_direction=8,
                                       beat=n.beat + 0.125, time_seconds=n.time_seconds + step))
    return _sorted(out)


def onset_strength(audio: np.ndarray, sr: int, hop: int = 512) -> tuple[np.ndarray, float]:
    """(per-frame loudness rise, seconds per frame) from mono PCM; no librosa needed."""
    n = len(audio) // hop
    if n == 0:
        return np.zeros(1), hop / sr
    rms = np.sqrt((np.asarray(audio[: n * hop], dtype=np.float32).reshape(n, hop) ** 2).mean(axis=1))
    rise = np.diff(np.log1p(rms * 100), prepend=0.0).clip(min=0)
    return rise + 0.25 * rms / (rms.max() or 1.0), hop / sr


# --- 4. vocal snapping (backlog 8) ------------------------------------------------------------------------------
def vocal_signal(audio: np.ndarray, sr: int, hop: int = 512) -> tuple[np.ndarray, np.ndarray, float]:
    """(vocal-lead per frame, harmonic onset peak times in seconds, seconds per frame).

    HPSS splits the song into harmonic (vocals, sustained instruments) and percussive parts; lead is the
    one-bar-smoothed ratio of harmonic to percussive onset strength, so >1 means the melody is driving.
    Needs librosa (ML extra)."""
    import librosa  # ponytail: only vocal_snap needs it; keep the core import-free

    harm, perc = librosa.effects.hpss(audio)
    oh = librosa.onset.onset_strength(y=harm, sr=sr, hop_length=hop)
    op = librosa.onset.onset_strength(y=perc, sr=sr, hop_length=hop)
    hop_s = hop / sr
    win = max(int(round(2.0 / hop_s)), 1)  # ~2 s, about a bar at 120 BPM
    k = np.ones(win) / win
    lead = np.convolve(oh, k, "same") / (np.convolve(op, k, "same") + 1e-6)
    peaks = librosa.util.peak_pick(oh, pre_max=3, post_max=3, pre_avg=10, post_avg=10, delta=0.3 * (oh.std() or 1.0), wait=4)
    return lead, np.asarray(peaks, dtype=np.float64) * hop_s, hop_s


def vocal_snap(notes: list[Note], bpm: float, lead: np.ndarray, peak_times: np.ndarray, hop_s: float,
               min_lead: float = 1.5, tol_beats: float = 0.25) -> list[Note]:
    """Move hits in vocal-led stretches (``lead`` >= ``min_lead``) onto the nearest harmonic onset.

    bsmg: vocal rhythms rarely sit on the drum grid; map what the voice does. The onset time is re-quantised
    to whichever of the 1/8 or 1/12 grids is nearer so triplet phrasing survives. A hit moves at most
    ``tol_beats`` and never onto a time another hit already holds.
    """
    if len(peak_times) == 0:
        return _sorted(notes)
    beat = 60.0 / bpm
    hits = _hits(notes)
    taken = {round(h[0].time_seconds, 3) for h in hits}
    out: list[Note] = []
    for h in hits:
        t = h[0].time_seconds
        f = min(int(t / hop_s), len(lead) - 1)
        pt = float(peak_times[np.argmin(np.abs(peak_times - t))])
        if lead[f] < min_lead or abs(pt - t) > tol_beats * beat:
            out.extend(h)
            continue
        b = pt / beat
        nb = min((round(b * 8) / 8, round(b * 12) / 12), key=lambda q: abs(q - b))
        nt = nb * beat
        if round(nt, 3) in taken and round(nt, 3) != round(t, 3):
            out.extend(h)
            continue
        taken.discard(round(t, 3))
        taken.add(round(nt, 3))
        out.extend(replace(n, beat=nb, time_seconds=nt) for n in h)
    return _sorted(out)


# --- 5. bombs and walls (backlog 12) ------------------------------------------------------------------------
# ponytail: rules at export, as v2 dicts. The model still emits colour notes only; arcs and chains are v3 and the
# exporter writes v2, so they stay out until the tokenizer learns them.
def bombs_and_walls(notes: list[Note], bpm: float, bombs: bool = True, walls: bool = True,
                    rest_beats: float = 2.0, wall_rest_beats: float = 4.0) -> tuple[list[dict], list[dict]]:
    """(v2 bomb notes, v2 obstacles) for ``notes``.

    Bomb resets (bsmg Intermediate: Bombs): when a hand rests >= ``rest_beats`` and comes back with the same
    cut it left on, a bomb at 2/3 of the gap sits where the saber must not be (low for a coming down-cut,
    high for an up-cut), so the player resets instead of double-directional-ing. Kept >= 1/4 beat from any note.
    Crouch walls (bsmg Basic: Walls): one full-width crouch wall in every rest of >= ``wall_rest_beats``.
    """
    beat = 60.0 / bpm
    ns = _sorted(notes)
    times = np.array([n.time_seconds for n in ns]) if ns else np.zeros(0)
    out_b: list[dict] = []
    if bombs:
        for color in (0, 1):
            h = [n for n in ns if n.color == color]
            for a, b in zip(h, h[1:]):
                gap = b.time_seconds - a.time_seconds
                if gap < rest_beats * beat or a.cut_direction != b.cut_direction or b.cut_direction not in (0, 1):
                    continue
                t = a.time_seconds + gap * 2 / 3
                if np.any(np.abs(times - t) < 0.25 * beat):
                    continue
                out_b.append({"_time": round(t / beat, 4), "_lineIndex": b.x, "_lineLayer": 0 if b.cut_direction == 1 else 2,
                              "_type": 3, "_cutDirection": 8})
    out_w: list[dict] = []
    if walls:
        hits = _hits(ns)
        for p, q in zip(hits, hits[1:]):
            gap = q[0].time_seconds - p[0].time_seconds
            if gap >= wall_rest_beats * beat:
                start = p[0].time_seconds / beat + 1.0
                out_w.append({"_time": round(start, 4), "_lineIndex": 0, "_type": 1, "_width": 4,
                              "_duration": round(gap / beat - 2.0, 4)})
        out_w += _lane_walls(ns, out_w)
    return out_b, out_w


def _lane_walls(ns: list[Note], taken: list[dict], every: float = 4.0, dur: float = 2.0) -> list[dict]:
    """Dodge walls in whichever half of the grid (lanes 0-1 or 2-3) has no note for the wall's span plus a half beat
    either side, else 1-wide in an empty outer lane; at most one per ``every`` beats, none overlapping ``taken``."""
    out: list[dict] = []
    t = ns[0].beat if ns else 0.0
    end = ns[-1].beat if ns else 0.0
    while t + dur <= end:
        if not any(w["_time"] < t + dur + 0.5 and t < w["_time"] + w["_duration"] + 0.5 for w in taken):
            span = [n for n in ns if t - 0.5 <= n.beat < t + dur + 0.5]
            lanes = {n.x for n in span}
            busy = {x // 2 for x in lanes}
            free = [h for h in (0, 1) if h not in busy]
            outer = [x for x in (0, 3) if x not in lanes]
            # a note beside the candidate wall cutting toward it: that side is out too
            if any(n.x == 2 and n.cut_direction in (2, 4, 6) for n in span): free = [h for h in free if h != 0]; outer = [x for x in outer if x != 0] if 1 in lanes else outer
            if any(n.x == 1 and n.cut_direction in (3, 5, 7) for n in span): free = [h for h in free if h != 1]; outer = [x for x in outer if x != 3] if 2 in lanes else outer
            if any(n.x == 1 and n.cut_direction in (2, 4, 6) for n in span): outer = [x for x in outer if x != 0]
            if any(n.x == 2 and n.cut_direction in (3, 5, 7) for n in span): outer = [x for x in outer if x != 3]
            if free and len(busy) == 1:  # a half is free and notes still need the other, so the wall forces a lean
                out.append({"_time": round(t, 4), "_lineIndex": 2 * free[0], "_type": 0, "_width": 2, "_duration": dur})
            elif lanes and outer:  # else a 1-wide wall in an empty outer lane: a lighter lean
                out.append({"_time": round(t, 4), "_lineIndex": outer[0], "_type": 0, "_width": 1, "_duration": dur})
            if free and len(busy) == 1 or lanes and outer:
                t += every
                continue
        t += 0.5
    return out


def limit_extras(bombs: list[dict], walls: list[dict], notes: list[Note], bpm: float,
                 walls_per_bar: float = 0.5, min_wall_beats: float = 1.0) -> tuple[list[dict], list[dict]]:
    """Filter model-made v2 bombs and walls down to what the bsmg guidelines allow.

    Walls: drop any shorter than ``min_wall_beats`` (the model's half-beat side flickers), any full-height wall
    3+ lanes wide (bsmg: never 3-wide, never the whole grid), and any with a note in its lanes while active
    or in the half beat after (any row: a note under a crouch wall is hidden while ducking). The rest are capped at ``walls_per_bar``
    per 4 beats, longest first, by keeping wall starts at least ``4 / walls_per_bar`` beats apart. Bombs: one in a note's saber path (``_in_saber_path``) or on a hand's route between two notes
    (``_on_swing_route``) moves to the nearest clear cell at the same beat, or is dropped.
    """
    ns = _sorted(notes)

    def covers(w: dict, n: Note) -> bool:
        # any row: a note under a crouch wall's lane is hidden while ducking (played: "notes under a wall, hard to hit");
        # a half beat either side is reserved for moving the saber out / standing back up. A note in the lane next
        # to an active wall whose cut swings toward the wall (played: "blocks on the other side of walls") also goes.
        lo, hi = w["_lineIndex"], w["_lineIndex"] + w["_width"]
        active = w["_time"] - 0.5 <= n.beat < w["_time"] + w["_duration"] + 0.5
        into = (n.x == lo - 1 and n.cut_direction in (3, 5, 7)) or (n.x == hi and n.cut_direction in (2, 4, 6))
        return active and (lo <= n.x < hi or (into and w["_time"] <= n.beat < w["_time"] + w["_duration"]))

    keep = [w for w in walls if w["_duration"] >= min_wall_beats and not (w["_type"] == 0 and w["_width"] >= 3)
            and not any(covers(w, n) for n in ns)]
    # longest first, each kept wall starts at least 4 / walls_per_bar beats from every other, so no section bunches
    gap = 4 / walls_per_bar if walls_per_bar > 0 else float("inf")
    spaced: list[dict] = []
    for w in sorted(keep, key=lambda w: -w["_duration"]) if walls_per_bar > 0 else []:
        if all(abs(w["_time"] - k["_time"]) >= gap for k in spaced):
            spaced.append(w)
    keep = sorted(spaced, key=lambda w: w["_time"])
    out_b: list[dict] = []
    at = collections.Counter(b["_time"] for b in bombs)
    for b in bombs:  # keep the model's timing; move a lone bomb in a saber path to the nearest clear cell
        near = [n for n in ns if abs(n.beat - b["_time"]) <= 0.5]
        if at[b["_time"]] > 1:  # part of a row/frame: moving one cell breaks the shape, so drop just that cell.
            # A quarter-beat window: a 3-4 cell row almost always touches some note within half a beat, and only
            # 4% of real Expert bombs are that close (16% at half a beat)
            if not any(_in_saber_path(b, n, 0.25) for n in near) and not _on_swing_route(b, ns):
                out_b.append(b)
            continue
        cells = sorted(((x, y) for x in range(4) for y in range(3)),
                       key=lambda c: abs(c[0] - b["_lineIndex"]) + abs(c[1] - b["_lineLayer"]))
        for x, y in cells:
            m = {**b, "_lineIndex": x, "_lineLayer": y}
            if not any(_in_saber_path(m, n) for n in near) and not _on_swing_route(m, ns) and not any(
                    o["_lineIndex"] == x and o["_lineLayer"] == y and abs(o["_time"] - b["_time"]) < 0.5 for o in out_b):
                out_b.append(m)
                break
    return out_b, keep


def _in_saber_path(b: dict, n: Note, window: float = 0.5) -> bool:
    """A bomb blocks a note's swing if it is in the note's column within 1/4 beat, or in a neighbouring cell
    (including diagonals) within half a beat: the saber passes through or next to it on the way in or out
    (played: a bomb beside a note, then the two swapped one hit later). 1 beat flagged 35% of real Expert bombs."""
    dt = abs(n.beat - b["_time"])
    near = abs(n.x - b["_lineIndex"]) <= 1 and abs(n.y - b["_lineLayer"]) <= 1
    return (n.x == b["_lineIndex"] and dt < 0.25) or (near and dt <= window)


_CUT_STEP = {0: (0, 1), 1: (0, -1), 2: (-1, 0), 3: (1, 0), 4: (-1, 1), 5: (1, 1), 6: (-1, -1), 7: (1, -1), 8: (0, 0)}


def _on_swing_route(b: dict, ns: list[Note], max_gap: float = 2.0, reach: float = 0.75) -> bool:
    """A bomb sits on a hand's route between the note before it and the note after it (both within
    ``max_gap`` beats of the bomb): the follow-through cell one step past the first cut, the wind-up cell one
    step before the second, and the straight lines joining them. Played: bombs in the saber's path between
    notes, which the half-beat ``_in_saber_path`` window missed (3-7 per map, gaps up to 2 beats)."""
    c = (b["_lineIndex"], b["_lineLayer"])
    for hand in (0, 1):
        before = [n for n in ns if n.color == hand and b["_time"] - max_gap <= n.beat <= b["_time"]]
        after = [n for n in ns if n.color == hand and b["_time"] < n.beat <= b["_time"] + max_gap]
        if not before or not after:
            continue
        p, q = before[-1], after[0]
        (dpx, dpy), (dqx, dqy) = _CUT_STEP.get(p.cut_direction, (0, 0)), _CUT_STEP.get(q.cut_direction, (0, 0))
        route = [(p.x, p.y), (p.x + dpx, p.y + dpy), (q.x - dqx, q.y - dqy), (q.x, q.y)]
        if min(_seg_dist(c, a, z) for a, z in zip(route, route[1:])) < reach:
            return True
    return False


def _seg_dist(c: tuple, a: tuple, z: tuple) -> float:
    vx, vy = z[0] - a[0], z[1] - a[1]
    t = 0.0 if vx == vy == 0 else max(0.0, min(1.0, ((c[0] - a[0]) * vx + (c[1] - a[1]) * vy) / (vx * vx + vy * vy)))
    return math.hypot(c[0] - a[0] - t * vx, c[1] - a[1] - t * vy)


def drop_overlapping_walls(walls: list[dict]) -> list[dict]:
    """Drop walls that share a lane with a longer wall while active, or start within a half beat of its end
    (played: walls inside other walls). Rule and model walls are merged before this, so it catches both."""
    kept: list[dict] = []
    for w in sorted(walls, key=lambda w: -w["_duration"]):
        if not any(w["_lineIndex"] < k["_lineIndex"] + k["_width"] and k["_lineIndex"] < w["_lineIndex"] + w["_width"]
                   and w["_time"] < k["_time"] + k["_duration"] + 0.5 and k["_time"] < w["_time"] + w["_duration"] + 0.5
                   for k in kept):
            kept.append(w)
    return sorted(kept, key=lambda w: w["_time"])


if __name__ == "__main__":  # self-check
    def N(t, x, y, c, d, bpm=120.0):
        return Note(beat=t * bpm / 60, time_seconds=t, x=x, y=y, color=c, cut_direction=d)

    bpm = 120.0
    # 32 bars of quarter-note alternating hands, left in col 1 and right in col 2
    base = [N(i * 0.5, 1 + i % 2, 1 if i % 4 else 2, i % 2, (1, 0)[(i // 2) % 2]) for i in range(256)]
    hard = downmap(base, bpm, "ExpertPlus", "Hard")
    assert 0.55 * len(base) < len(hard) < 0.7 * len(base), len(hard)
    fixed = repair_parity([N(0, 0, 0, 0, 1), N(0.5, 0, 0, 0, 1), N(1.0, 0, 0, 0, 8), N(4.0, 0, 0, 0, 1)], bpm)
    assert [n.cut_direction for n in fixed] == [1, 0, 8, 1], "repeat reversed, dot kept, far-apart repeat allowed"
    assert {n.time_seconds for n in hard} <= {n.time_seconds for n in base}
    easy = downmap(base, bpm, "ExpertPlus", "Easy")
    assert len(easy) < len(hard) and all(n.y < 2 for n in easy)
    ladder = downmap_ladder(base, bpm, "ExpertPlus", ["Easy", "Hard", "ExpertPlus"])
    assert len(ladder["Easy"]) < len(ladder["Hard"]) < len(ladder["ExpertPlus"])

    hop_s = 0.01
    rng = np.random.default_rng(0)
    block = rng.random((int(16 / hop_s), 4))  # 8 bars at 120 bpm = 16 s
    feat = np.concatenate([block, rng.random(block.shape), block, block])
    pairs = find_repeats(feat, hop_s, bpm, bars=8)
    assert pairs == [(0.0, 32.0), (0.0, 48.0)], pairs
    rep = repeat_sections(base, bpm, pairs, bars=8)
    a = [(n.x, n.color) for n in rep if n.time_seconds < 16]
    b = [(n.x, n.color) for n in rep if 32 <= n.time_seconds < 48]
    c = [(n.x, n.color) for n in rep if 48 <= n.time_seconds < 64]
    assert a == b and c == [(3 - x, 1 - col) for x, col in a], "block 3 copies, block 4 mirrors"

    strength = np.zeros(int(130 / hop_s))
    strength[int(8.0 / hop_s)] = 1.0  # one loud moment at 8 s
    sparse = [N(t, 0, 1, 0, 1) for t in range(0, 128, 2)]  # every 2 s in column 0: room on both sides
    emp = emphasize(sparse, bpm, strength, hop_s, top_frac=1 / len(_hits(sparse)) + 1e-9)
    extra = [n for n in emp if n not in sparse]
    assert len(extra) == 1 and extra[0].time_seconds == 8.0 and extra[0].cut_direction != 8, extra
    print("shaping ok")
