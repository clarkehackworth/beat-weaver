"""Autoregressive generation with grammar-constrained decoding."""

from __future__ import annotations

import collections

import torch
import torch.nn.functional as F

from beat_weaver.model.config import ModelConfig
from beat_weaver.model.tokenizer import (
    BAR_BASE,
    BAR_COUNT,
    COLS,
    DIRS,
    ROWS,
    SUBDIVISIONS_PER_BAR,
    _decode_note_token,
    bar_token,
    is_bar_token,
    DIFF_EASY,
    DIFF_EXPERT_PLUS,
    END,
    LEFT_BASE,
    LEFT_COUNT,
    LEFT_EMPTY,
    POS_BASE,
    POS_COUNT,
    RIGHT_BASE,
    RIGHT_COUNT,
    RIGHT_EMPTY,
    START,
    VOCAB_SIZE,
    BOMB_BASE,
    BOMB_COUNT,
    WALL_BASE,
    WALL_COUNT,
    decode_extras,
    decode_tokens,
    difficulty_to_token,
    is_bomb_token,
    is_wall_token,
)
from beat_weaver.model.transformer import BeatWeaverModel
from beat_weaver.schemas.normalized import Note


def _audio_bars(real_frames: int) -> int:
    """Number of bars (64 frames each) needed to cover ``real_frames`` of audio."""
    return max(1, min(BAR_COUNT, -(-real_frames // SUBDIVISIONS_PER_BAR)))


# Saber travel direction for each cut direction (0=up 1=down 2=left 3=right
# 4=up-left 5=up-right 6=down-left 7=down-right 8=any). Two consecutive cuts on
# one hand "flow" when the second roughly reverses the first (a down-swing is
# followed by an up-swing). The exception is 8 (any direction).
_CUT_VEC = torch.tensor(
    [[0, 1], [0, -1], [-1, 0], [1, 0], [-1, 1], [1, 1], [-1, -1], [1, -1], [0, 0]],
    dtype=torch.float32,
)
_CUT_UNIT = torch.nn.functional.normalize(_CUT_VEC, dim=1)


def _same_way(a: int, b: int) -> bool:
    """Two cuts need a reset between them: within 45 degrees, or both swinging up (or both down). Up-left then
    up-right is 90 degrees apart but still two upswings (seen in play testing: 0.09 s apart)."""
    va, vb = _CUT_VEC[a], _CUT_VEC[b]
    return float(va @ vb) > 0 or float(va[1] * vb[1]) > 0


def _note_tokens_on_cell(base: int, x: int, y: int) -> slice:
    """The 9 compound tokens (all cut directions) for one grid cell of one hand."""
    start = base + x * (ROWS * DIRS) + y * DIRS  # same layout as _encode_note_token
    return slice(start, start + DIRS)


FOLLOW_THROUGH = 0.75  # grid cells the hand continues past a note along its cut
CURVATURE = 0.0  # arc penalty added for a negative-curvature rotation above 45 degrees; 0.3 cost ~1.5 points and blew up `updown` (see MAPPING_GUIDELINES 17)
VISION_CELLS = ((1, 1), (2, 1))  # middle row, centre columns: a note here hides the ones behind it


def _converging_right_tokens(left_cell: tuple[int, int], left_dir: int, block_near: bool = True) -> torch.Tensor:
    """RIGHT note tokens whose cut would drive into the LEFT note just placed at the same time.

    Two sabers converge when each cut points at the other note and the notes are on
    the same or adjacent cells: the blades meet in the middle. Also any directional
    RIGHT note on the LEFT note's own cell. Favourite Expert+ maps: median 0 per map.

    ``block_near`` also blocks the near miss: cuts 135 degrees apart (45 off head-on) with
    at least one aimed at the other note. Legal, but the blades pass within inches;
    reference maps use it in ~4 of 1000 doubles.

    Arm tangle (bsmg Basic): a LEFT note in the right half (crossed) with a vertical
    cut blocks RIGHT notes in the left half cutting the other way vertically, so
    the crossed arms never pull apart through each other.
    """
    m = torch.zeros(VOCAB_SIZE, dtype=torch.bool)
    if left_cell[0] >= 2 and left_dir < 8 and _CUT_VEC[left_dir][1] != 0:
        for x in (0, 1):
            for y in range(ROWS):
                sl = _note_tokens_on_cell(RIGHT_BASE, x, y)
                for d in range(8):
                    if _CUT_VEC[d][1] * _CUT_VEC[left_dir][1] < 0:
                        m[sl.start + d] = True
    # cut-through: a RIGHT cut whose blade path (1-2 cells along the cut) runs over the LEFT note's cell
    for x in range(COLS):
        for y in range(ROWS):
            sl = _note_tokens_on_cell(RIGHT_BASE, x, y)
            for d in range(8):
                rv = _CUT_VEC[d]
                if any((x + k * int(rv[0]), y + k * int(rv[1])) == left_cell for k in (1, 2)):
                    m[sl.start + d] = True
    # approach: a RIGHT cut whose saber arrives through the LEFT note's cell (one step against its cut)
    for x in range(COLS):
        for y in range(ROWS):
            sl = _note_tokens_on_cell(RIGHT_BASE, x, y)
            for d in range(8):
                rv = _CUT_VEC[d]
                if (x - int(rv[0]), y - int(rv[1])) == left_cell:
                    m[sl.start + d] = True
    if left_dir < 0 or left_dir == 8:
        return m
    lv = _CUT_VEC[left_dir]
    cx, cy = left_cell[0] - int(lv[0]), left_cell[1] - int(lv[1])  # ... and the LEFT saber arrives through the RIGHT note's cell
    if 0 <= cx < COLS and 0 <= cy < ROWS:
        m[_note_tokens_on_cell(RIGHT_BASE, cx, cy)] = True
    for k in (1, 2):  # ... and the LEFT blade's path runs over the RIGHT note's cell, whatever its cut
        cx, cy = left_cell[0] + k * int(lv[0]), left_cell[1] + k * int(lv[1])
        if 0 <= cx < COLS and 0 <= cy < ROWS:
            m[_note_tokens_on_cell(RIGHT_BASE, cx, cy)] = True
    for x in range(COLS):
        for y in range(ROWS):
            dx, dy = x - left_cell[0], y - left_cell[1]
            if max(abs(dx), abs(dy)) > 1:
                continue
            sl = _note_tokens_on_cell(RIGHT_BASE, x, y)
            if dx == 0 and dy == 0:
                m[sl.start: sl.start + 8] = True  # all 8 directional cuts; a dot is left alone
                continue
            if lv[0] * dx + lv[1] * dy <= 0:
                continue  # left cut doesn't point at this cell
            for d in range(8):
                rv = _CUT_VEC[d]
                if rv[0] * -dx + rv[1] * -dy > 0:
                    m[sl.start + d] = True
    if block_near:  # near miss: one cut aimed at the other note, directions >= 135 degrees apart
        lu = torch.nn.functional.normalize(lv.float(), dim=0)
        for x in range(COLS):
            for y in range(ROWS):
                dx, dy = x - left_cell[0], y - left_cell[1]
                if max(abs(dx), abs(dy)) != 1:
                    continue
                sl = _note_tokens_on_cell(RIGHT_BASE, x, y)
                l_at = lv[0] * dx + lv[1] * dy > 0
                for d in range(8):
                    rv = _CUT_VEC[d]
                    r_at = rv[0] * -dx + rv[1] * -dy > 0
                    if (l_at or r_at) and float(lu @ torch.nn.functional.normalize(rv.float(), dim=0)) <= -0.7:
                        m[sl.start + d] = True
    return m


def _reset_tokens(base: int, last_dir: int) -> torch.Tensor:
    """Boolean (VOCAB_SIZE,) mask of this hand's tokens that cut the same way as its last cut (``_same_way``).

    Cutting the same way twice in a row means the saber has to come back first (a reset).
    Dots are left alone. ponytail: no slider exception; add one if fast same-way sliders are wanted.
    """
    m = torch.zeros(VOCAB_SIZE, dtype=torch.bool)
    if last_dir < 0 or last_dir == 8:
        return m
    same = [d for d in range(8) if _same_way(d, last_dir)]
    for x in range(COLS):
        for y in range(ROWS):
            sl = _note_tokens_on_cell(base, x, y)
            for d in same:
                m[sl.start + d] = True
    return m


CROSSOVER_TARGET = 0.21  # share of notes in the other hand's half, Expert training data
STACK_TARGET = 0.013  # default stacked notes per note: real Expert median 1.3 per 100 (p75 3.3)
DIAGONAL_TARGET = 0.30  # share of cuts 4-7 in the Expert training data
BOMB_TARGET = 0.05  # bombs per note, Expert training data


def _bomb_push(bias: float, n_bombs: int, n_notes: int) -> float:
    """Logit push on BOMB tokens: ``bias`` while under the training rate, none up to twice it, -inf above."""
    rate = n_bombs / max(n_notes, 1)
    return bias if rate < BOMB_TARGET else 0.0 if rate < 2 * BOMB_TARGET else float("-inf")


def _vision_tokens(base: int) -> torch.Tensor:
    """Boolean (VOCAB_SIZE,) mask of this hand's note tokens on the vision-block cells."""
    m = torch.zeros(VOCAB_SIZE, dtype=torch.bool)
    for x, y in VISION_CELLS:
        m[_note_tokens_on_cell(base, x, y)] = True
    return m


_ANGLE = {0: 90, 1: 270, 2: 180, 3: 0, 4: 135, 5: 45, 6: 225, 7: 315}


def _swings_back_through(px: int, py: int, pd: int, qx: int, qy: int, qd: int) -> bool:
    """A hand that cut (px, py) along pd ends past that cell; to cut (qx, qy) along qd it must first wind up
    on the cell's far side. True when that wind-up path runs back through the note itself: the hand travels
    toward an edge and the next cut asks it to come from beyond the edge (played: a red note on the right
    wanting an inward cut while the saber was already heading out). Real Expert maps: median 0, p90 2.2% of
    a hand's swings within a beat; generated maps 3-9%."""
    if not (0 <= pd < 8 and 0 <= qd < 8):
        return False
    a, b = _CUT_UNIT[pd].tolist(), _CUT_UNIT[qd].tolist()
    hx, hy = px + a[0], py + a[1]  # where the saber is after following through
    wx, wy = qx - b[0], qy - b[1]  # where it has to be to start the next cut
    if (qx - hx) ** 2 + (qy - hy) ** 2 < 0.75 ** 2:
        return False  # the saber is already at the note: no travel to speak of
    vx, vy = wx - hx, wy - hy
    t = max(0.0, min(1.0, ((qx - hx) * vx + (qy - hy) * vy) / (vx * vx + vy * vy or 1.0)))
    return (qx - hx - t * vx) ** 2 + (qy - hy - t * vy) ** 2 < 0.5 ** 2


def _swing_back_tokens(base: int, prev_cell: tuple[int, int], prev_direction: int) -> torch.Tensor:
    """Bool (VOCAB_SIZE,) mask of this hand's notes that ``_swings_back_through`` from its last cut."""
    m = torch.zeros(VOCAB_SIZE, dtype=torch.bool)
    for x in range(COLS):
        for y in range(ROWS):
            for d in range(8):
                if _swings_back_through(*prev_cell, prev_direction, x, y, d):
                    m[_note_tokens_on_cell(base, x, y).start + d] = True
    return m


def _big_reversal_tokens(base: int, prev_cell: tuple[int, int], prev_direction: int) -> torch.Tensor:
    """Bool mask over the vocab: this hand's notes 2+ cells from its last cell whose cut turns 135+ degrees
    from its last cut (a long slash reversed). Dots and same-way cuts are never flagged."""
    mask = torch.zeros(VOCAB_SIZE, dtype=torch.bool)
    if prev_direction not in _ANGLE:
        return mask
    px, py = prev_cell
    for x in range(4):
        for y in range(3):
            if (x - px) ** 2 + (y - py) ** 2 < 4:
                continue
            for d in range(8):
                turn = abs((_ANGLE[d] - _ANGLE[prev_direction] + 180) % 360 - 180)
                if turn >= 135:
                    mask[base + x * 27 + y * 9 + d] = True
    return mask


REVERSAL_TARGET = 0.43  # of a hand's swings within half a beat of its last, the share that are big reversals: real Expert per-map median (p10 0.09, p90 0.74; Expert+ median 0.53)


def _arc_penalty(base: int, prev_cell: tuple[int, int] | None, prev_direction: int) -> torch.Tensor:
    """Per-token penalty (0..1) for how badly a note breaks the saber's arc from the hand's last cut.

    After cutting cell A along d_A the hand sits at A + FOLLOW_THROUGH * d_A. The
    natural next cut through cell B is the direction from that point to B, so the
    penalty is (1 - cos) / 2 between each token's cut and that direction. On the
    same cell this is exactly "reverse the last cut"; on a cell off to one side it
    favours a cut that sweeps through B along the way, which is the arc a mapper
    draws. Dots cost 0.5 (neutral). Returns a (VOCAB_SIZE,) tensor, non-zero only
    over this hand's 108 note tokens.

    Swing curvature (bsmg Intermediate: Rotations): the wrist turns one way from
    the last cut into the travel to B and should keep turning that way into the
    new cut. When the new cut turns back the other way (negative curvature) and
    the two cuts are more than 45 degrees apart, CURVATURE is added. A straight
    reversal (180 degrees) has no sense change and costs nothing extra.
    """
    pen = torch.zeros(VOCAB_SIZE)
    if prev_direction < 0 or prev_direction == 8 or prev_cell is None:
        return pen
    prev = _CUT_UNIT[prev_direction]
    hand = torch.tensor(prev_cell, dtype=torch.float32) + FOLLOW_THROUGH * prev
    cells = torch.tensor([[x, y] for x in range(COLS) for y in range(ROWS)], dtype=torch.float32)  # token order
    u = torch.nn.functional.normalize(cells - hand, dim=1)  # (12, 2) ideal cut direction per cell
    cos = u @ _CUT_UNIT.T  # (12, 9)
    cos[:, 8] = 0.0
    turn_in = prev[0] * u[:, 1] - prev[1] * u[:, 0]  # (12,) sense of last cut -> travel
    turn_out = u[:, 0:1] * _CUT_UNIT[:, 1] - u[:, 1:2] * _CUT_UNIT[:, 0]  # (12, 9) sense of travel -> new cut
    negative = (turn_in.unsqueeze(1) * turn_out < -1e-6) & ((_CUT_UNIT @ prev) < 0.75).unsqueeze(0)  # > 45 deg
    pen[base: base + COLS * ROWS * DIRS] = ((1.0 - cos) / 2 + CURVATURE * negative).clamp(max=1.0).reshape(-1)
    return pen


def _column_tokens(base: int, x: int) -> slice:
    return slice(base + x * ROWS * DIRS, base + (x + 1) * ROWS * DIRS)


def _drift_bonus(base: int, prev_cell: tuple[int, int] | None, drift: int) -> torch.Tensor:
    """(VOCAB_SIZE,) 0/1 mask of this hand's tokens one lane over in its drift direction.

    drift is the sign of the hand's last lateral move (0 = none: both neighbours). At the edge the
    drift reverses, so a hand sweeping right across the grid comes back left: the piano-stream shape.
    """
    m = torch.zeros(VOCAB_SIZE)
    if prev_cell is None:
        return m
    x = prev_cell[0]
    for nx in ((x + 1, x - 1) if drift == 0 else (x + drift,) if 0 <= x + drift < COLS else (x - drift,)):
        if 0 <= nx < COLS:
            m[_column_tokens(base, nx)] = 1.0
    # zero-mean over this hand's note tokens: shapes *where* a note goes without lifting notes over
    # the hand's EMPTY token (which would raise the doubles rate)
    notes = slice(base, base + COLS * ROWS * DIRS)
    m[notes] -= m[notes].mean()
    return m


def _hand_comfort(base: int) -> torch.Tensor:
    """(VOCAB_SIZE,) placement comfort for one hand (bsmg basic-mapping "Difficulty", intermediate "Handedness").

    Forehand swings (down, down-diagonals) sit naturally in the inner lanes, backhand swings (up,
    up-diagonals) in the outer lanes; a down-cut on the top row and an up-cut on the bottom row are
    strain. Horizontal cuts facing away from the body's centre on the outer lane are a reach. +1
    comfortable, -1 strained, zero-mean over the hand's note tokens so it never adds notes.
    """
    m = torch.zeros(VOCAB_SIZE)
    left = base == LEFT_BASE
    inner, outer = (1, 0) if left else (2, 3)
    outward = 2 if left else 3  # the horizontal cut pointing away from the centre
    for x in range(COLS):
        for y in range(ROWS):
            for d in range(DIRS):
                v = 0.0
                fore, back = d in (1, 6, 7), d in (0, 4, 5)
                if fore and x == inner or back and x == outer:
                    v += 1.0
                if fore and x == outer or back and x == inner:
                    v -= 0.5
                if fore and y == 2 or back and y == 0:
                    v -= 1.0
                if d == outward and x == outer:
                    v -= 1.0
                m[base + (x * ROWS + y) * DIRS + d] = v
    notes = slice(base, base + COLS * ROWS * DIRS)
    m[notes] -= m[notes].mean()
    return m


_COMFORT = {LEFT_BASE: _hand_comfort(LEFT_BASE), RIGHT_BASE: _hand_comfort(RIGHT_BASE)}


# Rotation sense between two cuts: sign of the 2-D cross product (counter-clockwise > 0). Dots are 0.
_ROT = torch.sign(_CUT_UNIT[:, 0:1] * _CUT_UNIT[:, 1].unsqueeze(0) - _CUT_UNIT[:, 1:2] * _CUT_UNIT[:, 0].unsqueeze(0))
MOMENTUM_TENSION = 3  # same-sense rotations in a row before the wrist is wound up and should unwind


def _momentum(base: int, prev_direction: int, rot: int) -> torch.Tensor:
    """(VOCAB_SIZE,) in [-1, 1]: +1 for cuts that keep this hand's rotation going, -1 for the reverse.

    bsmg "Momentum and Tension": a wrist turning one way wants to keep turning, up to a point. Past
    MOMENTUM_TENSION consecutive same-sense turns the sign flips so the hand unwinds. rot is the
    signed count of consecutive same-sense turns so far.
    """
    m = torch.zeros(VOCAB_SIZE)
    if prev_direction < 0 or prev_direction == 8 or rot == 0:
        return m
    sense = _ROT[prev_direction]  # (9,) rotation sense of each next cut relative to the last
    want = 1.0 if abs(rot) < MOMENTUM_TENSION else -1.0
    v = (sense * (1 if rot > 0 else -1) * want).repeat(COLS * ROWS)
    m[base: base + COLS * ROWS * DIRS] = v - v.mean()  # zero-mean: see _drift_bonus
    return m


# Cut directions pointing back toward the hand's own side for a hand that has crossed over:
# left hand in columns 2-3 may not cut Left/UpLeft/DownLeft, right hand in columns 0-1 may
# not cut Right/UpRight/DownRight. 300 real Expert maps: a right note in column 0 cuts
# L/UL/DL 80% of the time and R/UR/DR 1% (column 1: 0-8%); the saber reaching across keeps
# travelling the way it went. (This table was reversed until 2026-10-20; play testing found it.)
_OUTWARD = {LEFT_BASE: ((2, 3), (2, 4, 6)), RIGHT_BASE: ((0, 1), (3, 5, 7))}
# And in the hand's own outer column (left x=0, right x=3), no inward cut: real Expert 1.3% (top), 3.8% (middle),
# 1.6% (bottom) cut inward there, fix9 2-12%. Played: a right note at the top-right corner cutting
# up-left after a down-right, and another cutting down-left after a down.
_INWARD_OUTER = {LEFT_BASE: ((0,), (3, 5, 7)), RIGHT_BASE: ((3,), (2, 4, 6))}


def _cuts_back(base: int, x: int, d: int) -> bool:
    """A note this hand should not get: crossed over and cutting back home, or on its outer column cutting inward."""
    return any(x in cols and d in dirs for cols, dirs in (_OUTWARD[base], _INWARD_OUTER[base]))


def _crossover_outward_tokens(base: int) -> list[int]:
    return [base + x * 27 + y * 9 + d for x in range(4) for y in range(3) for d in range(9) if _cuts_back(base, x, d)]


def _stack_tokens(last_token: int, base: int) -> torch.Tensor:
    """This hand's note tokens that may follow ``last_token`` at the same instant: the next cell along the cut line
    with the same cut (bsmg stack, tower; any adjacent cell for a dot). Ascending cell order, like bomb clusters.
    Real Expert same-hand pairs: 2.56 per 100 notes adjacent along the cut, 0.48 across it, 1.41 farther apart
    (windows); any higher cell made mostly far pairs, which gate_patterns then collapsed to singles."""
    cell, d = divmod(last_token - base, DIRS)
    x, y = divmod(cell, ROWS)
    if d == 8:
        steps = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if (dx, dy) != (0, 0)]
    else:
        vx, vy = (int(v) for v in _CUT_VEC[d])
        steps = [(vx, vy), (-vx, -vy)]
    cells = {(x + dx) * ROWS + y + dy for dx, dy in steps if 0 <= x + dx < COLS and 0 <= y + dy < ROWS}
    return torch.tensor([base + c * DIRS + d for c in sorted(cells) if c > cell], dtype=torch.long)


def _position_closed(token: int) -> bool:
    """True once a position's RIGHT slot (and any BOMB/WALL extras) has been written."""
    return token == RIGHT_EMPTY or RIGHT_BASE <= token < RIGHT_BASE + RIGHT_COUNT or is_bomb_token(token) or is_wall_token(token)


def _build_grammar_mask(
    last_token: int, last_pos_in_bar: int = -1, current_bar: int = -1,
    audio_bars: int | None = None, left_cell: tuple[int, int] | None = None, stacks: bool = False,
) -> torch.Tensor:
    """Build a boolean mask over the vocabulary for valid next tokens.

    Returns a tensor of shape (VOCAB_SIZE,) where True = allowed.

    Args:
        last_token: The most recently generated token.
        last_pos_in_bar: The last POS offset used in the current bar (-1 if none).
            Used to enforce strictly increasing positions within a bar,
            preventing multiple notes at the same beat.
        current_bar: Index of the bar currently open (-1 before the first bar).
            The only bar token allowed next is current_bar + 1, so bars can
            neither repeat nor skip and the sequence cannot outrun the audio.
        audio_bars: Bars of real audio in the window. When given, END is not
            allowed until the last of those bars is open, and no bar may follow
            it. Without this the model stops early: END is legal after every
            bar, so a small chance per bar compounds over 64 bars (measured on
            a trained model: windows ending after 14 and 17 of 64 bars).
        left_cell: (x, y) of the LEFT note just placed at this position, if any.
            The RIGHT note may not occupy the same cell: two blocks cannot share
            a grid square at one instant. Human maps never do this; the model
            did it 1.5-2.5% of the time. Nor a column left of it: red right of
            blue at one instant crosses the arms (45-map batch: 295 such doubles).

    Grammar rules:
        START      → DIFF_*
        DIFF_*     → BAR_0
        BAR_k      → POS_* | BAR_k+1 | END
        POS_*      → LEFT_* | LEFT_EMPTY
        LEFT_*     → LEFT_* (higher cell, same cut: a stack) | RIGHT_* | RIGHT_EMPTY
        RIGHT_*    → RIGHT_* (higher cell, same cut) | BOMB_* | WALL_* | POS_* (strictly >) | BAR_k+1 | END
        BOMB_c     → BOMB_c' (c' > c) | WALL_* | POS_* (strictly >) | BAR_k+1 | END
        WALL_*     → POS_* (strictly >) | BAR_k+1 | END
    A notes-only model (355 logits) never sees the BOMB/WALL rows: generate() pads its logits with -inf.
    The stack continuations need ``stacks=True``: a checkpoint trained one-note-per-hand puts random mass there.
    """
    mask = torch.zeros(VOCAB_SIZE, dtype=torch.bool)
    next_bar = current_bar + 1
    next_bar_ok = next_bar < BAR_COUNT

    if last_token == START:
        # After START → only difficulty tokens
        mask[DIFF_EASY: DIFF_EXPERT_PLUS + 1] = True

    elif DIFF_EASY <= last_token <= DIFF_EXPERT_PLUS:
        # After DIFF → the first bar
        mask[bar_token(0)] = True

    elif is_bar_token(last_token):
        # After BAR_k → POS, BAR_k+1 (if any audio left), or END
        mask[POS_BASE: POS_BASE + POS_COUNT] = True
        if next_bar_ok:
            mask[bar_token(next_bar)] = True
        mask[END] = True

    elif POS_BASE <= last_token < POS_BASE + POS_COUNT:
        # After POS → LEFT note or LEFT_EMPTY
        mask[LEFT_EMPTY] = True
        mask[LEFT_BASE: LEFT_BASE + LEFT_COUNT] = True
        mask[_crossover_outward_tokens(LEFT_BASE)] = False

    elif last_token == LEFT_EMPTY or (LEFT_BASE <= last_token < LEFT_BASE + LEFT_COUNT):
        # After LEFT → RIGHT note or RIGHT_EMPTY (never on the LEFT note's cell)
        mask[RIGHT_EMPTY] = True
        mask[RIGHT_BASE: RIGHT_BASE + RIGHT_COUNT] = True
        mask[_crossover_outward_tokens(RIGHT_BASE)] = False
        if left_cell is not None:
            mask[_note_tokens_on_cell(RIGHT_BASE, *left_cell)] = False
            mask[RIGHT_BASE: RIGHT_BASE + left_cell[0] * ROWS * DIRS] = False  # no RIGHT note left of the LEFT one: arms would cross
        if stacks and LEFT_BASE <= last_token < LEFT_BASE + LEFT_COUNT:  # or the LEFT stack goes on
            mask[_stack_tokens(last_token, LEFT_BASE)] = True
            mask[_crossover_outward_tokens(LEFT_BASE)] = False

    elif _position_closed(last_token):
        # After RIGHT (or its BOMB/WALL extras) → extras not yet used, POS (strictly increasing), BAR, or END
        if last_token == RIGHT_EMPTY or (RIGHT_BASE <= last_token < RIGHT_BASE + RIGHT_COUNT):
            mask[BOMB_BASE: BOMB_BASE + BOMB_COUNT] = True
        if stacks and RIGHT_BASE <= last_token < RIGHT_BASE + RIGHT_COUNT:  # or the RIGHT stack goes on
            mask[_stack_tokens(last_token, RIGHT_BASE)] = True
            mask[_crossover_outward_tokens(RIGHT_BASE)] = False
            if left_cell is not None:
                mask[_note_tokens_on_cell(RIGHT_BASE, *left_cell)] = False
                mask[RIGHT_BASE: RIGHT_BASE + left_cell[0] * ROWS * DIRS] = False
        elif is_bomb_token(last_token):  # more bombs at this instant, in ascending cell order
            mask[last_token + 1: BOMB_BASE + BOMB_COUNT] = True
        if not is_wall_token(last_token):
            mask[WALL_BASE: WALL_BASE + WALL_COUNT] = True
        min_next = last_pos_in_bar + 1
        if min_next < POS_COUNT:
            mask[POS_BASE + min_next: POS_BASE + POS_COUNT] = True
        if next_bar_ok:
            mask[bar_token(next_bar)] = True
        mask[END] = True

    else:
        # Unknown state — allow everything except PAD/START
        mask[2:] = True

    if audio_bars is not None:
        audio_bars = min(audio_bars, BAR_COUNT)
        if current_bar < audio_bars - 1:
            mask[END] = False  # audio remains: the sequence must reach the last bar
        else:
            mask[BAR_BASE: BAR_BASE + BAR_COUNT] = False  # last bar is open: only notes or END

    return mask


# Rhythm precision prior over the 16 frames of a beat: 1/1, 1/2 and 1/4 (quarter-beat) positions full weight,
# 1/8 positions 0.6, 1/16 positions 0.4. Index = frame offset within the beat.
PRECISION_WEIGHT = torch.tensor([1.0 if i % 4 == 0 else 0.6 if i % 2 == 0 else 0.4 for i in range(SUBDIVISIONS_PER_BAR // 4)])


def _onset_schedule(
    model: BeatWeaverModel, memory: torch.Tensor, n_frames: int,
    notes_per_beat: float | None = None, min_gap: int = 4, dynamics: float = 1.0,
    stream_bias: float = 1.0,
    dynamics_range: tuple[float, float] = (0.0, 2.0),
    beat_focus: bool = False,
    beat_floor: float = 0.0,
    precision_prior: float = 0.0,
) -> list[int]:
    """Stage 1 of two-stage decoding: the frames that get a note.

    Read straight off the encoder's onset head (trained per frame on where the
    map's notes fall, top-k precision 0.56 vs 0.08 chance), with no decoder
    involved. Each bar gets the head's own expected count there (sum of its
    probabilities), or ``notes_per_beat * 4`` when overridden, spread across
    the bar's beats with ``min_gap`` frames between notes.

    ``dynamics`` shapes the overridden budget to the music: each bar's share is
    scaled by (its expected count / the window mean) ** dynamics, clamped to
    ``dynamics_range`` (default [0, 2]), so the song-wide density still averages ``notes_per_beat`` but quiet
    bars thin out and loud bars fill up. 0 is a flat budget (every bar the same,
    which plays as staccato), 1 follows the onset head fully. A narrower range
    (0.7, 1.3) evens it out: played, one song at the full range had a few 7 notes/s
    sections between long 2 notes/s ones.

    ``stream_bias`` lets a run keep going: after a frame is picked, the frame
    ``min_gap`` later is taken too while its probability is at least
    ``stream_bias`` x the bar's peak (and budget remains). 1.0 is off; the
    round-robin alone caps runs at about one note per beat, which plays as
    Expert. Favourite Expert+ maps have runs of 13-150 notes.

    ``beat_focus``: once every beat has its one note, give the rest of the bar's
    budget to the strongest beat until it is full, then the next. Weak beats keep a
    single note, which leaves a rest of about a beat between streams. Without it the
    extras are spread evenly, so at Expert+ density no gap is ever longer than half
    a beat and the whole song plays as one unbroken stream.

    ``beat_floor``: a beat whose peak probability is under ``beat_floor`` x the
    bar's peak gets no note at all. 0 (off) guarantees every beat one note, so
    rests longer than a beat only appear when a whole bar's budget rounds to 0 and
    Expert+ plays as one 200-note line. 0.5 lets quiet beats rest.

    ``precision_prior`` (bsmg Basic: Timing): most songs need only 1/1, 1/2 and
    1/4 precision; 1/8 and finer is usually overmapping. Each frame's probability
    is scaled by PRECISION_WEIGHT[offset within the beat] ** precision_prior
    before ranking, so an off-beat 1/8 or 1/16 needs a stronger onset than a
    quarter to be picked. 0 is off; 1 applies the table as is. The per-bar
    budget is taken from the unscaled probabilities, so density is unchanged.
    """
    p = torch.sigmoid(model.onset_head(memory)[0, :n_frames, 0].float())
    beat = SUBDIVISIONS_PER_BAR // 4
    sums = torch.tensor([p[b0: b0 + SUBDIVISIONS_PER_BAR].sum().item() for b0 in range(0, n_frames, SUBDIVISIONS_PER_BAR)])
    if precision_prior > 0:
        p = p * (PRECISION_WEIGHT.to(p.device) ** precision_prior).repeat(n_frames // beat + 1)[:n_frames]
    bars = [p[b0: b0 + SUBDIVISIONS_PER_BAR] for b0 in range(0, n_frames, SUBDIVISIONS_PER_BAR)]
    shape = (sums / sums.mean().clamp(min=1e-6)).clamp(min=0.0) ** dynamics if notes_per_beat else torch.ones_like(sums)
    shape = shape.clamp(*dynamics_range)
    chosen: list[int] = []
    for b0, pb, sh in zip(range(0, n_frames, SUBDIVISIONS_PER_BAR), bars, shape.tolist()):
        k = round(pb.sum().item()) if notes_per_beat is None else round(notes_per_beat * len(pb) / 16 * sh)
        # Round-robin over the bar's beats, strongest frame first within each, so
        # the budget is spread through the bar instead of bunching on its loudest
        # moment (which played as a burst then a wait).
        queues = [
            [b0 + q0 + i for i in torch.argsort(pb[q0: q0 + beat], descending=True).tolist()]
            for q0 in range(0, len(pb), beat)
        ]
        peak = pb.max().item() * stream_bias
        per_beat = [0] * len(queues)
        picked = 0
        # quiet beats (under beat_floor x the bar's peak) are skipped and never reserved for
        live = [pb[j * beat: (j + 1) * beat].max().item() >= pb.max().item() * beat_floor for j in range(len(queues))]
        # Beats take turns strongest-first, so a bar's remainder (6 notes over 4 beats = 2,2,1,1) goes to the
        # beats the onset head likes best. Always starting at beat 1 gave beats 1-2 twice the notes of 3-4.
        order = sorted((j for j in range(len(queues)) if live[j]), key=lambda j: -pb[j * beat: (j + 1) * beat].max().item())
        while picked < k and any(queues[j] for j in order):
            before = picked
            focus = beat_focus and all(per_beat[j] > 0 for j in order)
            for qi in order:
                if focus and picked > before:
                    break  # one pick per pass, always from the strongest beat that still has room
                q = queues[qi]
                # a beat that already has a note yields to beats that have none
                reserve = sum(1 for j in order if j != qi and per_beat[j] == 0)
                if per_beat[qi] > 0 and picked >= k - reserve:
                    continue
                while q:
                    f = q.pop(0)
                    if all(abs(f - c) >= min_gap for c in chosen):
                        chosen.append(f)
                        picked += 1
                        per_beat[qi] += 1
                        # stream: chain forward through hot frames, across beats if they stay hot, but
                        # always leave one note of budget for every later beat that has none yet. Without
                        # that reserve a chain ate the whole bar and every bar played front-loaded.
                        while stream_bias < 1.0:
                            g = f + min_gap
                            if g >= b0 + len(pb) or p[g].item() < peak or any(abs(g - c) < min_gap for c in chosen):
                                break
                            gi = (g - b0) // beat
                            reserve = sum(1 for j in order if j > gi and per_beat[j] == 0)
                            if picked >= k - reserve:
                                break
                            chosen.append(g)
                            picked += 1
                            per_beat[gi] += 1
                            for qq in queues:
                                if g in qq:
                                    qq.remove(g)
                            f = g
                        break
                if picked >= k:
                    break
            if picked == before:  # every remaining queue was skipped or exhausted
                break
    return sorted(chosen)



def _stream_frames(schedule: list[int], share: float, beat: int = SUBDIVISIONS_PER_BAR // 4, min_len: int = 8) -> set[int]:
    """Frames of the schedule's dense chains (hits no more than half a beat apart, ``min_len`` or longer), taken
    longest first until ``share`` of the scheduled notes sit in one. Inside those the decoder plays a stream: one
    note per hit, hands alternating. Real Expert maps put a median 10% of their notes in streams (p75 18%, 83%
    of maps have one); the flat doubles lift and same-hand pairs broke every chain here within a few notes."""
    if share <= 0 or not schedule:
        return set()
    chains, run = [], [schedule[0]]
    for a, b in zip(schedule, schedule[1:]):
        if b - a <= beat // 2:
            run.append(b)
        else:
            chains.append(run)
            run = [b]
    chains.append(run)
    out: set[int] = set()
    for c in sorted((c for c in chains if len(c) >= min_len), key=len, reverse=True):
        if len(out) >= share * len(schedule):
            break
        out.update(c)
    return out


def _cap_runs(schedule: list[int], max_run: int, beat: int = SUBDIVISIONS_PER_BAR // 4) -> list[int]:
    """Drop frames so no run (notes within half a beat of each other) exceeds ``max_run``.

    Once a run is full, every frame is dropped until a gap of a whole beat opens, so
    the stream ends in a real rest. ponytail: the dropped budget is not re-spent, the
    density variants in beat_sim.mapper hedge that; re-spend elsewhere if winners sit low.
    """
    if max_run <= 0:
        return schedule
    kept: list[int] = []
    run = 0
    for f in sorted(schedule):
        gap = f - kept[-1] if kept else beat
        if run >= max_run and gap < beat:
            continue
        run = run + 1 if gap <= beat // 2 else 1
        kept.append(f)
    return kept


def _sample_with_filter(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
) -> int:
    """Sample a token from logits with temperature, top-k, and top-p filtering."""
    if temperature <= 0:
        return logits.argmax().item()

    logits = logits / temperature

    # Top-k filtering
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        values, _ = torch.topk(logits, top_k)
        min_val = values[-1]
        logits = torch.where(logits < min_val, torch.full_like(logits, float("-inf")), logits)

    # Top-p (nucleus) filtering
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        # Remove tokens with cumulative prob above threshold
        sorted_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
        sorted_logits[sorted_mask] = float("-inf")
        # Scatter back
        logits = torch.zeros_like(logits).scatter(0, sorted_indices, sorted_logits)

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1).item()


@torch.no_grad()
def generate(
    model: BeatWeaverModel,
    mel_spectrogram: torch.Tensor,
    difficulty: str,
    config: ModelConfig,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    seed: int | None = None,
    mel_mask: torch.Tensor | None = None,
    audio_bars: int | None = None,
    flow_penalty: float = 2.0,
    onset_guidance: float = 0.0,
    two_stage: bool = False,
    notes_per_beat: float | None = None,
    min_gap: int = 4,
    dynamics: float = 1.0,
    dynamics_range: tuple[float, float] = (0.0, 2.0),
    stream_bias: float = 1.0,
    doubles_bias: float = 0.0,
    stack_bias: float = 0.0,
    stack_target: float = STACK_TARGET,
    bomb_bias: float = 0.0,
    bomb_gap_bias: float = 0.0,
    bomb_row_bias: float = 0.0,
    diagonal_bias: float = 0.0,
    hand_balance: float = 0.0,
    crossover_penalty: float = 0.0,
    vision_penalty: float = 3.0,
    vision_max_run: int = 2,
    column_max_run: int = 0,
    raised_max_run: int = 0,
    reversal_penalty: float = 0.0,
    beat_focus: bool = False,
    block_near_converging: bool = True,
    streams: float = 0.0,
    stacks: bool = False,
    reset_gap: int = 0,
    max_doubles: float = 1.0,
    ebpm_gap: int = 0,
    beat_floor: float = 0.0,
    max_run: int = 0,
    drift_bias: float = 0.0,
    momentum_bias: float = 0.0,
    flow_window: float = 1.5,
    hand_bias: float = 0.0,
    precision_prior: float = 0.0,
) -> list[int]:
    """Generate a token sequence autoregressively.

    Args:
        model: Trained BeatWeaverModel.
        mel_spectrogram: (n_mels, T_audio) — single spectrogram (no batch dim).
        difficulty: Difficulty name (e.g., "Expert").
        config: Model configuration.
        temperature: Sampling temperature (0 = greedy).
        top_k: Top-k filtering (0 = disabled).
        top_p: Top-p / nucleus filtering (1.0 = disabled).
        seed: Random seed for reproducibility.
        mel_mask: (T_audio,) — True for valid positions.
        vision_penalty: Logit penalty on notes in the two middle-row centre cells,
            which sit in the player's line of sight and hide what follows. The
            decoder puts 25-30% of notes there; favourite Expert+ maps 2% (p90 8%).
        vision_max_run: hard cap on consecutive hits that include such a note.
            After that many, centre cells are masked until a hit lands elsewhere.
        raised_max_run: hard cap on one hand's consecutive notes in rows 1-2. After that many, the hand's
            next note goes to the bottom row (unless that leaves it nothing). Played, fix7, two songs:
            long upper-row chains tire the arms even with every whole-song stat in the real Expert range.
        column_max_run: hard cap on one hand's consecutive notes in the same
            column. The arc penalty makes "reverse the last cut" cheapest on the
            same cell, so without this a hand chains straight up/down/up/down in
            place for 10-20+ notes. After that many, the column is masked for
            that hand and the arc penalty picks a sweep into a neighbour. 0 = off.
        reset_gap: frames (1/16 beat). A hand's note this soon after its last
            one may not cut within 45 degrees of that cut (forced reset). The
            arc penalty alone left ~4 resets per map. 0 = off.
        ebpm_gap: frames (1/16 beat) a hand must wait between swings (bsmg swing speed: 60 / max eBPM
            seconds). A hand that swung more recently sits this slot out unless the other hand also did
            (0 = off).
        max_doubles: cap on the fraction of hits that use both hands. The
            doubles rate otherwise swings 0.1-0.8 by song, and a Hard with 0.8
            doubles came out denser than the Expert. 1.0 = off.
        flow_penalty: Logit penalty, times _arc_penalty, applied to each note
            token for how badly it breaks the saber's arc from that hand's last
            cut: after cutting a cell the hand continues along the cut, and the
            cheap next cut sweeps from there through the next cell. On the same
            cell that means reversing; off to one side it means a diagonal or
            sideways cut that connects the two. 0 disables it. Human Expert maps
            repeat a direction ~4-5% of the time, the model ~7-12%, so this is a
            nudge the model can still override, not a ban.
        onset_guidance: Weight on the encoder's per-frame onset logit, added to
            the logit of each POS token (POS p in bar b is audio frame
            64*b + p). The decoder alone does not reliably use the audio (on
            held-out charts its loss was the same with the right song's audio
            and a wrong one's), but the encoder's onset head does locate where
            notes fall (top-k precision 0.56 vs 0.08 chance). This ties note
            timing to the song directly. The logits are centred on the window
            mean so the overall density is unchanged: it moves notes to the
            onsets, it does not add or remove them. 0 disables it.
        two_stage: Rhythm from the audio, placement from the decoder. The
            decoder's cross-attention never attended to the audio at any point
            in training (entropy at the uniform ceiling; forcing it onto the
            right frame changed the loss by nothing), so every POS/BAR/END
            token is taken from the onset head (see ``_onset_schedule``) and
            the decoder only chooses what goes at each given position. A
            scheduled position must hold a note, so LEFT_EMPTY + RIGHT_EMPTY
            is forbidden there.
        notes_per_beat: Two-stage note density override (default: the onset
            head's own expected count).

    Returns:
        List of token IDs including START and END.
    """
    if seed is not None:
        torch.manual_seed(seed)

    model.eval()
    device = next(model.parameters()).device

    # Prepare mel: add batch dimension
    mel = mel_spectrogram.unsqueeze(0).to(device)  # (1, n_mels, T_audio)
    if mel_mask is not None:
        mel_mask = mel_mask.unsqueeze(0).to(device)  # (1, T_audio)

    # Encode audio once
    memory = model.encode_audio(mel, mel_mask)

    # Per-frame onset guidance, centred on the window's real audio and padded to
    # the full BAR_COUNT * 64 frames so every bar can slice 64 entries.
    guidance = None
    if onset_guidance > 0:
        onset = model.onset_head(memory)[0, :, 0].float()
        n_real = min(onset.numel(), (audio_bars or BAR_COUNT) * SUBDIVISIONS_PER_BAR)
        guidance = torch.zeros(BAR_COUNT * SUBDIVISIONS_PER_BAR, device=device)
        m = min(onset.numel(), guidance.numel())
        guidance[:m] = onset[:m] - onset[:n_real].mean()

    n_bars = min(audio_bars or BAR_COUNT, BAR_COUNT)
    schedule = None
    if two_stage:
        schedule = _cap_runs(_onset_schedule(model, memory, n_bars * SUBDIVISIONS_PER_BAR, notes_per_beat, min_gap, dynamics, stream_bias, dynamics_range, beat_focus, beat_floor, precision_prior), max_run)
    stream_frames = _stream_frames(schedule, streams) if schedule else set()
    sched_i = 0

    # Start with [START, DIFF_x]
    diff_token = difficulty_to_token(difficulty)
    tokens = [START, diff_token]
    last_pos_in_bar = -1  # Track last POS offset in current bar
    current_bar = -1  # Index of the open bar; grammar only permits current_bar + 1 next
    left_cell: tuple[int, int] | None = None  # cell of the LEFT note at the open position
    last_dir = {LEFT_BASE: -1, RIGHT_BASE: -1}  # previous cut direction per hand
    last_cell: dict[int, tuple[int, int] | None] = {LEFT_BASE: None, RIGHT_BASE: None}
    left_dir_open = -1  # cut direction of the LEFT note at the open position
    open_left: list[tuple[int, int]] = []  # every LEFT cell at the open position (a stack has several)
    vision_run = 0  # consecutive hits (POS tokens) that included a vision-cell note
    hit_has_vision = False  # whether the open position has one so far
    n_stack = 0  # notes that continue a same-hand stack
    col_run = {LEFT_BASE: 0, RIGHT_BASE: 0}  # consecutive notes per hand in its last column
    raised_run = {LEFT_BASE: 0, RIGHT_BASE: 0}  # consecutive notes per hand in rows 1-2
    last_frame = {LEFT_BASE: -10**9, RIGHT_BASE: -10**9}  # frame of each hand's last note
    drift = {LEFT_BASE: 0, RIGHT_BASE: 0}  # sign of the hand's last lateral move
    rot = {LEFT_BASE: 0, RIGHT_BASE: 0}  # signed count of consecutive same-sense rotations
    beat_frames = SUBDIVISIONS_PER_BAR // 4
    hits = doubles = 0
    n_hand = {LEFT_BASE: 0, RIGHT_BASE: 0}
    recent_cross: collections.deque = collections.deque(maxlen=16)
    cross_tokens = torch.tensor([LEFT_BASE + i for i in range(108) if i // 27 >= 2]
                                + [RIGHT_BASE + i for i in range(108) if i // 27 <= 1], dtype=torch.long, device=device)
    n_diag = n_bombs = 0
    n_pairs = n_rev = 0  # same-hand pairs within half a beat, and how many were big reversals
    hold_extras, inserted_from, in_gap_pos = False, -1, False
    diag_tokens = torch.tensor([b + i for b in (LEFT_BASE, RIGHT_BASE) for i in range(108) if i % 9 in (4, 5, 6, 7)],
                               dtype=torch.long, device=device)
    resets = {(b, d): _reset_tokens(b, d).to(device) for b in (LEFT_BASE, RIGHT_BASE) for d in range(8)}
    swing_back = {(b, c, d): _swing_back_tokens(b, c, d).to(device) for b in (LEFT_BASE, RIGHT_BASE)
                  for c in ((x, y) for x in range(COLS) for y in range(ROWS)) for d in range(8)}
    vis = {LEFT_BASE: _vision_tokens(LEFT_BASE).to(device), RIGHT_BASE: _vision_tokens(RIGHT_BASE).to(device)}

    while len(tokens) < config.max_seq_len - 1:  # extras-only positions add 3+ tokens a step: count tokens, not steps
        # Two-stage: timing tokens come from the schedule, not the decoder
        if schedule is not None and (is_bar_token(tokens[-1]) or _position_closed(tokens[-1])):
            # Timing is scheduled, but the decoder may still add bombs or a wall to the position, or open a
            # bomb-only position between scheduled ones (92% of Expert bombs sit on a beat with no note).
            in_bar = sched_i < len(schedule) and schedule[sched_i] // SUBDIVISIONS_PER_BAR == current_bar
            # Token budget: every scheduled note still to come needs POS, LEFT, RIGHT, plus a BAR per bar and END.
            # Extras (bombs, walls, bomb-only positions) only get what is left, so the schedule always fits: without
            # one song's last window hit the 2048 cap at bar 55 of 64 (661 half-beat wall tokens) and the outro
            # went silent.
            reserve = 3 * (len(schedule) - sched_i) + (n_bars - current_bar) + 1
            if config.vocab_size > BOMB_BASE and not hold_extras and len(tokens) + reserve < config.max_seq_len - 8:
                token_tensor = torch.tensor([tokens], dtype=torch.long, device=device)
                token_mask = torch.ones(1, len(tokens), dtype=torch.bool, device=device)
                extra_logits = model.decoder(token_tensor, memory, token_mask, mel_mask)[0, -1]
                gm = _build_grammar_mask(tokens[-1], last_pos_in_bar, current_bar, audio_bars).to(device)
                extra_logits[~gm] = float("-inf")
                n_notes = n_hand[LEFT_BASE] + n_hand[RIGHT_BASE]
                push = _bomb_push(bomb_bias, n_bombs, n_notes) if bomb_bias > 0 else 0.0
                if push == float("-inf") or (n_bombs + 1) / max(n_notes, 20) > 2 * BOMB_TARGET:  # 20: a couple early on
                    extra_logits[BOMB_BASE: BOMB_BASE + BOMB_COUNT] = float("-inf")  # over twice the training rate
                elif RIGHT_BASE <= tokens[-1] < RIGHT_BASE + RIGHT_COUNT or tokens[-1] == RIGHT_EMPTY:
                    extra_logits[BOMB_BASE: BOMB_BASE + BOMB_COUNT] += push
                insert_hi = schedule[sched_i] % SUBDIVISIONS_PER_BAR if in_bar else POS_COUNT
                insert = slice(POS_BASE + last_pos_in_bar + 1, POS_BASE + insert_hi)
                if bomb_row_bias > 0 and in_gap_pos and is_bomb_token(tokens[-1]) and push != float("-inf"):
                    # 38% of real Expert bomb instants have 2+ bombs (rows, frames); after a gap bomb the model gives
                    # ~8%, having written few bombs earlier in the song. Lift more bombs (ascending cells only).
                    extra_logits[BOMB_BASE: BOMB_BASE + BOMB_COUNT] += bomb_row_bias
                if bomb_gap_bias > 0 and push != float("-inf") and not in_gap_pos:  # not mid-cluster
                    # The model puts ~1% on opening a bomb-only position (it cannot tell from the audio which songs
                    # are bomb-heavy: 72% of Expert bombs sit in maps with 100+); lift those POS tokens.
                    extra_logits[insert] += bomb_gap_bias
                if not torch.isfinite(extra_logits[BOMB_BASE: BOMB_BASE + BOMB_COUNT]).any():
                    extra_logits[insert] = float("-inf")  # a bomb-only position with no bomb allowed is pointless
                # Sample "a bomb/wall here", "a bomb-only position at an earlier POS" or "move on" (any other
                # timing token); the schedule still owns which note position comes next.
                next_token = _sample_with_filter(extra_logits, temperature, top_k, top_p)
                if next_token >= BOMB_BASE:
                    n_bombs += BOMB_BASE <= next_token < BOMB_BASE + BOMB_COUNT
                    tokens.append(next_token)
                    continue
                if insert.start <= next_token < insert.stop:
                    # Open a bomb-only position only if the decoder agrees the position is empty: it must pick
                    # LEFT_EMPTY after the POS (a note there would be off the schedule).
                    pos_logits = model.decoder(torch.tensor([tokens + [next_token]], dtype=torch.long, device=device),
                                               memory, torch.ones(1, len(tokens) + 1, dtype=torch.bool, device=device),
                                               mel_mask)[0, -1]
                    if pos_logits[LEFT_EMPTY] >= pos_logits[LEFT_BASE: LEFT_BASE + LEFT_COUNT].max():
                        tokens += [next_token, LEFT_EMPTY, RIGHT_EMPTY]
                        inserted_from, last_pos_in_bar, left_cell = last_pos_in_bar, next_token - POS_BASE, None
                        open_left = []
                        in_gap_pos = True
                    else:
                        hold_extras = True  # the decoder wanted a note there: leave it to the schedule
                    continue
            if tokens[-1] == RIGHT_EMPTY and tokens[-2] == LEFT_EMPTY:
                # The decoder opened a bomb-only position and then put nothing in it: take it back, and stop it
                # reopening one before the next scheduled position
                del tokens[-3:]
                last_pos_in_bar, hold_extras, in_gap_pos = inserted_from, True, False
                continue
            hold_extras = False
            if sched_i < len(schedule) and schedule[sched_i] // SUBDIVISIONS_PER_BAR == current_bar:
                next_token = POS_BASE + schedule[sched_i] % SUBDIVISIONS_PER_BAR
                sched_i += 1
            elif current_bar + 1 < n_bars:
                next_token = bar_token(current_bar + 1)
            else:
                next_token = END
            tokens.append(next_token)
            if is_bar_token(next_token):
                last_pos_in_bar, current_bar = -1, next_token - BAR_BASE
            elif next_token != END:
                last_pos_in_bar, left_cell = next_token - POS_BASE, None
                open_left = []
                hits += 1
            if next_token == END:
                break
            continue

        # Prepare decoder input
        token_tensor = torch.tensor([tokens], dtype=torch.long, device=device)
        token_mask = torch.ones(1, len(tokens), dtype=torch.bool, device=device)

        logits = model.decoder(token_tensor, memory, token_mask, mel_mask)
        # logits: (1, seq_len, vocab_size) — take last position
        next_logits = logits[0, -1]  # (vocab_size,)
        if next_logits.numel() < VOCAB_SIZE:  # notes-only checkpoint: it can never emit bombs or walls
            next_logits = torch.cat([next_logits, torch.full((VOCAB_SIZE - next_logits.numel(),), float("-inf"), device=device)])

        # Apply grammar mask (with position tracking for one-note-per-color-per-beat)
        grammar_mask = _build_grammar_mask(
            tokens[-1], last_pos_in_bar, current_bar, audio_bars, left_cell, stacks=stacks,
        ).to(device)
        if schedule is not None and tokens[-1] == LEFT_EMPTY:
            grammar_mask[RIGHT_EMPTY] = False  # a scheduled position holds a note
        for c in open_left:  # no RIGHT note on any cell of the open LEFT stack
            grammar_mask[_note_tokens_on_cell(RIGHT_BASE, *c)] = False
        next_logits[~grammar_mask] = float("-inf")
        grammar_only = next_logits.clone()  # fallback if the style masks below leave nothing

        # Onset guidance: POS p of the open bar is frame 64*bar + p
        if guidance is not None and current_bar >= 0:
            f0 = current_bar * SUBDIVISIONS_PER_BAR
            next_logits[POS_BASE: POS_BASE + POS_COUNT] += (
                onset_guidance * guidance[f0: f0 + POS_COUNT]
            )

        # Flow: favour the cut that continues the saber's arc from the hand's last note, fading out as
        # the hand has had time to reset; then the lateral drift and rotation momentum shaping.
        if POS_BASE <= tokens[-1] < POS_BASE + POS_COUNT:
            hand_base = LEFT_BASE
        elif tokens[-1] == LEFT_EMPTY or LEFT_BASE <= tokens[-1] < LEFT_BASE + LEFT_COUNT:
            hand_base = RIGHT_BASE
        else:
            hand_base = None
        if hand_base is not None and hand_bias > 0:
            next_logits += hand_bias * _COMFORT[hand_base].to(device)
        if hand_base is not None and last_cell[hand_base] is not None:
            frame = current_bar * SUBDIVISIONS_PER_BAR + last_pos_in_bar
            gap_beats = (frame - last_frame[hand_base]) / beat_frames
            fresh = min(1.0, max(0.0, flow_window - gap_beats)) if flow_window > 0 else 1.0
            if flow_penalty > 0 and fresh > 0:
                next_logits -= flow_penalty * fresh * _arc_penalty(hand_base, last_cell[hand_base], last_dir[hand_base]).to(device)
            if drift_bias > 0 and gap_beats <= 1.0:
                next_logits += drift_bias * _drift_bonus(hand_base, last_cell[hand_base], drift[hand_base]).to(device)
            if momentum_bias > 0 and fresh > 0:
                next_logits += momentum_bias * fresh * _momentum(hand_base, last_dir[hand_base], rot[hand_base]).to(device)

        # Collision: a RIGHT note may not cut into the LEFT note placed at this same time
        if left_cell is not None and (tokens[-1] == LEFT_EMPTY or LEFT_BASE <= tokens[-1] < LEFT_BASE + LEFT_COUNT):
            next_logits[_converging_right_tokens(left_cell, left_dir_open, block_near_converging).to(device)] = float("-inf")

        # Vision: keep the centre of the middle row mostly clear, and never stack it
        if vision_penalty > 0 or vision_max_run > 0:
            for base in (LEFT_BASE, RIGHT_BASE):
                next_logits[vis[base]] -= vision_penalty
                if vision_max_run > 0 and vision_run >= vision_max_run:
                    next_logits[vis[base]] = float("-inf")

        # Big fast reversals: a 2+ cell slash reversed within half a beat. Of such fast same-hand swings, real Expert
        # maps make 43% big reversals (median), Expert+ 53%; extras8 made 55% and played as Expert+. Penalise
        # them while over the Expert median.
        if reversal_penalty > 0 and hand_base is not None and last_cell[hand_base] is not None \
                and n_rev > REVERSAL_TARGET * max(n_pairs, 20):
            frame = current_bar * SUBDIVISIONS_PER_BAR + last_pos_in_bar
            if (frame - last_frame[hand_base]) / beat_frames <= 0.5:
                next_logits[_big_reversal_tokens(hand_base, last_cell[hand_base], last_dir[hand_base]).to(device)] -= reversal_penalty

        # Straights: stop a hand chaining back-and-forths in one column
        if column_max_run > 0:
            for base in (LEFT_BASE, RIGHT_BASE):
                if col_run[base] >= column_max_run:
                    x = last_cell[base][0]
                    next_logits[base + x * 27: base + (x + 1) * 27] = float("-inf")

        # Arm fatigue: after raised_max_run notes in rows 1-2, a hand drops to the bottom row
        if raised_max_run > 0:
            for base in (LEFT_BASE, RIGHT_BASE):
                if raised_run[base] >= raised_max_run:
                    raised = torch.tensor([base + x * 27 + y * 9 + d for x in range(4) for y in (1, 2) for d in range(9)],
                                          device=device)
                    kept = next_logits[raised].clone()
                    next_logits[raised] = float("-inf")
                    if not torch.isfinite(next_logits[base: base + 108]).any():
                        next_logits[raised] = kept  # a forced note with only raised cells left: keep it
        # Swing speed: a hand that swung within 60/max_ebpm s sits this slot out (the other hand takes it)
        if ebpm_gap > 0:
            frame = current_bar * SUBDIVISIONS_PER_BAR + last_pos_in_bar
            if POS_BASE <= tokens[-1] < POS_BASE + POS_COUNT:
                if last_cell[LEFT_BASE] is not None and frame - last_frame[LEFT_BASE] < ebpm_gap \
                        and not (last_cell[RIGHT_BASE] is not None and frame - last_frame[RIGHT_BASE] < ebpm_gap):
                    next_logits[LEFT_BASE: LEFT_BASE + LEFT_COUNT] = float("-inf")
            elif LEFT_BASE <= tokens[-1] < LEFT_BASE + LEFT_COUNT:  # the slot already holds a note
                if last_cell[RIGHT_BASE] is not None and frame - last_frame[RIGHT_BASE] < ebpm_gap:
                    next_logits[RIGHT_BASE: RIGHT_BASE + RIGHT_COUNT] = float("-inf")

        # Parity: no same-way cut by a hand that just cut, unless it had time to reset
        if reset_gap > 0:
            frame = current_bar * SUBDIVISIONS_PER_BAR + last_pos_in_bar
            base = LEFT_BASE if POS_BASE <= tokens[-1] < POS_BASE + POS_COUNT else RIGHT_BASE
            if 0 <= last_dir[base] < 8 and frame - last_frame[base] < reset_gap:
                next_logits[resets[(base, last_dir[base])]] = float("-inf")
            # and no cut the saber would have to swing back through the note to reach, within a beat
            if 0 <= last_dir[base] < 8 and last_cell[base] is not None and frame - last_frame[base] <= beat_frames:
                next_logits[swing_back[(base, last_cell[base], last_dir[base])]] = float("-inf")

        # Streams: inside a chosen dense chain of the schedule (``_stream_frames``) each hit is one note and the
        # hands alternate, so the chain plays as a stream instead of doubles and same-hand pairs breaking it.
        if stream_frames and current_bar * SUBDIVISIONS_PER_BAR + last_pos_in_bar in stream_frames:
            if POS_BASE <= tokens[-1] < POS_BASE + POS_COUNT:
                if last_frame[LEFT_BASE] > last_frame[RIGHT_BASE]:
                    next_logits[LEFT_BASE: LEFT_BASE + LEFT_COUNT] = float("-inf")  # right's turn
                elif last_frame[RIGHT_BASE] > last_frame[LEFT_BASE]:
                    next_logits[LEFT_EMPTY] = float("-inf")  # left's turn
            elif LEFT_BASE <= tokens[-1] < LEFT_BASE + LEFT_COUNT:
                next_logits[RIGHT_BASE: RIGHT_BASE + RIGHT_COUNT] = float("-inf")  # no double
            elif tokens[-1] == LEFT_EMPTY:
                next_logits[RIGHT_EMPTY] = float("-inf")  # the hit needs a note

        # Stacks: run 5 learned them but emits ~0.2 per 100 notes (real Expert 1.3); lift the stack continuation
        # after a note until stack_target is reached. Needs stacks=True, else the grammar masks them.
        if stack_bias > 0 and n_stack < stack_target * max(n_hand[LEFT_BASE] + n_hand[RIGHT_BASE], 50):
            for base, size in ((LEFT_BASE, LEFT_COUNT), (RIGHT_BASE, RIGHT_COUNT)):
                if base <= tokens[-1] < base + size:
                    next_logits[_stack_tokens(tokens[-1], base).to(device)] += stack_bias

        # Doubles: after a LEFT note, lift every RIGHT note over RIGHT_EMPTY so both hands hit together more often.
        # Favourite Expert+ maps double 9-45% of hits (median 25%); the decoder alone lands near the floor.
        if LEFT_BASE <= tokens[-1] < LEFT_BASE + LEFT_COUNT:
            if hits and doubles / hits >= max_doubles:
                next_logits[RIGHT_BASE: RIGHT_BASE + RIGHT_COUNT] = float("-inf")
            elif doubles_bias > 0:
                next_logits[RIGHT_BASE: RIGHT_BASE + RIGHT_COUNT] += doubles_bias

        # Hand balance: the decoder leans right (one song, extras4: 666 R vs 370 L, singles 449 vs 153). Push the empty
        # slot of the hand that is ahead so the other hand takes the single: at a POS, LEFT_EMPTY gets less likely
        # when right leads; after a LEFT note, RIGHT_EMPTY gets more likely. imb in (-1, 1), > 0 = right-heavy.
        if hand_balance > 0:
            imb = (n_hand[RIGHT_BASE] - n_hand[LEFT_BASE]) / (n_hand[RIGHT_BASE] + n_hand[LEFT_BASE] + 8)
            if POS_BASE <= tokens[-1] < POS_BASE + POS_COUNT:
                next_logits[LEFT_EMPTY] -= hand_balance * imb
            elif LEFT_BASE <= tokens[-1] < LEFT_BASE + LEFT_COUNT:
                next_logits[RIGHT_EMPTY] += hand_balance * imb

        # Crossovers: the decoder clusters hands-on-the-wrong-side notes (one song's outro: 34-40% vs 21% in the training
        # data), where the saber has to come around the other arm. Penalise crossed cells while the last 16 notes
        # are over the training rate.
        if crossover_penalty > 0 and len(recent_cross) == 16 and sum(recent_cross) > CROSSOVER_TARGET * 16 and (
                POS_BASE <= tokens[-1] < POS_BASE + POS_COUNT or LEFT_BASE <= tokens[-1] < LEFT_BASE + LEFT_COUNT):
            next_logits[cross_tokens] -= crossover_penalty

        # Diagonals: the decoder collapses onto plain up/down cuts (89-98% on one song vs 57% in the training data);
        # lift the four diagonal cuts of both hands on every note slot.
        # Only while the running diagonal share is under the training data's 30%: a fixed boost feeds back into a
        # 90%-diagonal regime (one song: 0.6 -> 11%, 0.8 -> 90%).
        n_notes = n_hand[LEFT_BASE] + n_hand[RIGHT_BASE]
        if diagonal_bias > 0 and n_diag < DIAGONAL_TARGET * max(n_notes, 1) and (
                POS_BASE <= tokens[-1] < POS_BASE + POS_COUNT or LEFT_BASE <= tokens[-1] < LEFT_BASE + LEFT_COUNT):
            next_logits[diag_tokens] += diagonal_bias

        # Bombs: the model emits ~15x fewer than the training data (0.3 vs 5 per 100 notes on Expert); lift every BOMB
        # token after a RIGHT note. Notes-only checkpoints have -inf there, so this is a no-op for them.
        if bomb_bias > 0 and RIGHT_BASE <= tokens[-1] < RIGHT_BASE + RIGHT_COUNT:
            next_logits[BOMB_BASE: BOMB_BASE + BOMB_COUNT] += _bomb_push(bomb_bias, n_bombs, n_notes)

        if not torch.isfinite(next_logits).any():
            next_logits = grammar_only

        # Sample
        next_token = _sample_with_filter(next_logits, temperature, top_k, top_p)
        n_stack += any(b <= tokens[-1] < b + c and b <= next_token < b + c for b, c in ((LEFT_BASE, LEFT_COUNT), (RIGHT_BASE, RIGHT_COUNT)))
        tokens.append(next_token)
        in_gap_pos = False  # back on the schedule
        n_bombs += BOMB_BASE <= next_token < BOMB_BASE + BOMB_COUNT

        # Update position tracking
        if is_bar_token(next_token):
            last_pos_in_bar = -1  # Reset on new bar
            current_bar = next_token - BAR_BASE
        elif POS_BASE <= next_token < POS_BASE + POS_COUNT:
            last_pos_in_bar = next_token - POS_BASE
            left_cell, left_dir_open = None, -1
            open_left = []
            vision_run = vision_run + 1 if hit_has_vision else 0  # close the previous hit
            hits += 1
            hit_has_vision = False
        elif LEFT_BASE <= next_token < LEFT_BASE + LEFT_COUNT:
            x, y, d = _decode_note_token(next_token, LEFT_BASE)
            n_hand[LEFT_BASE] += 1
            if last_cell[LEFT_BASE] is not None and 0 <= last_dir[LEFT_BASE] < 8 and d < 8 \
                    and (current_bar * SUBDIVISIONS_PER_BAR + last_pos_in_bar - last_frame[LEFT_BASE]) / beat_frames <= 0.5:
                n_pairs += 1
                n_rev += (x - last_cell[LEFT_BASE][0]) ** 2 + (y - last_cell[LEFT_BASE][1]) ** 2 >= 4 \
                    and abs((_ANGLE[d] - _ANGLE[last_dir[LEFT_BASE]] + 180) % 360 - 180) >= 135
            recent_cross.append(x >= 2)
            n_diag += 4 <= d <= 7
            col_run[LEFT_BASE] = col_run[LEFT_BASE] + 1 if last_cell[LEFT_BASE] and last_cell[LEFT_BASE][0] == x else 1
            raised_run[LEFT_BASE] = raised_run[LEFT_BASE] + 1 if y >= 1 else 0
            if last_cell[LEFT_BASE] is not None and current_bar * SUBDIVISIONS_PER_BAR + last_pos_in_bar - last_frame[LEFT_BASE] <= beat_frames:
                drift[LEFT_BASE] = (x > last_cell[LEFT_BASE][0]) - (x < last_cell[LEFT_BASE][0]) or drift[LEFT_BASE]
                sense = int(_ROT[last_dir[LEFT_BASE], d].item()) if 0 <= last_dir[LEFT_BASE] < 8 and d < 8 else 0
                rot[LEFT_BASE] = rot[LEFT_BASE] + sense if sense and (rot[LEFT_BASE] == 0 or (sense > 0) == (rot[LEFT_BASE] > 0)) else sense
            else:
                drift[LEFT_BASE], rot[LEFT_BASE] = 0, 0
            last_frame[LEFT_BASE] = current_bar * SUBDIVISIONS_PER_BAR + last_pos_in_bar
            left_cell, left_dir_open = (x, y), d
            open_left.append((x, y))
            last_dir[LEFT_BASE], last_cell[LEFT_BASE] = d, (x, y)
            hit_has_vision |= (x, y) in VISION_CELLS
        elif RIGHT_BASE <= next_token < RIGHT_BASE + RIGHT_COUNT:
            x, y, d = _decode_note_token(next_token, RIGHT_BASE)
            n_hand[RIGHT_BASE] += 1
            if last_cell[RIGHT_BASE] is not None and 0 <= last_dir[RIGHT_BASE] < 8 and d < 8 \
                    and (current_bar * SUBDIVISIONS_PER_BAR + last_pos_in_bar - last_frame[RIGHT_BASE]) / beat_frames <= 0.5:
                n_pairs += 1
                n_rev += (x - last_cell[RIGHT_BASE][0]) ** 2 + (y - last_cell[RIGHT_BASE][1]) ** 2 >= 4 \
                    and abs((_ANGLE[d] - _ANGLE[last_dir[RIGHT_BASE]] + 180) % 360 - 180) >= 135
            recent_cross.append(x <= 1)
            n_diag += 4 <= d <= 7
            col_run[RIGHT_BASE] = col_run[RIGHT_BASE] + 1 if last_cell[RIGHT_BASE] and last_cell[RIGHT_BASE][0] == x else 1
            raised_run[RIGHT_BASE] = raised_run[RIGHT_BASE] + 1 if y >= 1 else 0
            if last_cell[RIGHT_BASE] is not None and current_bar * SUBDIVISIONS_PER_BAR + last_pos_in_bar - last_frame[RIGHT_BASE] <= beat_frames:
                drift[RIGHT_BASE] = (x > last_cell[RIGHT_BASE][0]) - (x < last_cell[RIGHT_BASE][0]) or drift[RIGHT_BASE]
                sense = int(_ROT[last_dir[RIGHT_BASE], d].item()) if 0 <= last_dir[RIGHT_BASE] < 8 and d < 8 else 0
                rot[RIGHT_BASE] = rot[RIGHT_BASE] + sense if sense and (rot[RIGHT_BASE] == 0 or (sense > 0) == (rot[RIGHT_BASE] > 0)) else sense
            else:
                drift[RIGHT_BASE], rot[RIGHT_BASE] = 0, 0
            last_frame[RIGHT_BASE] = current_bar * SUBDIVISIONS_PER_BAR + last_pos_in_bar
            doubles += LEFT_BASE <= tokens[-2] < LEFT_BASE + LEFT_COUNT
            last_dir[RIGHT_BASE], last_cell[RIGHT_BASE] = d, (x, y)
            hit_has_vision |= (x, y) in VISION_CELLS

        if next_token == END:
            break

    return tokens


def last_music_second(audio, sr: int, floor: float = 0.1, hop: int = 2205) -> float:
    """Time at which the song goes quiet for good: last 100 ms frame whose RMS is above ``floor`` x the peak.
    0.02 (-34 dB) counted the ring-out of the final hit as music: played, one song had notes 0.3 and 0.9 s after its
    last hit, at 5% and 2% of peak. 0.1 (-20 dB) ends it at the hit and keeps a quiet outro (~-13 dB)."""
    import numpy as np

    n = len(audio) // hop
    if n == 0:
        return 0.0
    rms = np.sqrt((np.asarray(audio[: n * hop], dtype=np.float32).reshape(n, hop) ** 2).mean(axis=1))
    loud = np.flatnonzero(rms > floor * rms.max())
    return float((loud[-1] + 1) * hop / sr) if len(loud) else 0.0


def generate_full_song(
    model: BeatWeaverModel,
    mel_spectrogram: torch.Tensor,
    difficulty: str,
    config: ModelConfig,
    bpm: float,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    seed: int | None = None,
    onset_guidance: float = 0.0,
    two_stage: bool = False,
    notes_per_beat: float | None = None,
    min_gap_seconds: float = 0.2,
    dynamics: float = 1.0,
    dynamics_range: tuple[float, float] = (0.0, 2.0),
    lead_in_seconds: float = 2.0,
    flow_penalty: float = 2.0,
    end_seconds: float | None = None,
    notes_per_second: float | None = None,
    stream_bias: float = 1.0,
    doubles_bias: float = 0.0,
    stack_bias: float = 0.0,
    stack_target: float = STACK_TARGET,
    bomb_bias: float = 0.0,
    bomb_gap_bias: float = 0.0,
    bomb_row_bias: float = 0.0,
    diagonal_bias: float = 0.0,
    hand_balance: float = 0.0,
    crossover_penalty: float = 0.0,
    vision_penalty: float = 3.0,
    vision_max_run: int = 2,
    column_max_run: int = 0,
    raised_max_run: int = 0,
    reversal_penalty: float = 0.0,
    beat_focus: bool = False,
    block_near_converging: bool = True,
    streams: float = 0.0,
    stacks: bool = False,
    reset_seconds: float = 0.0,
    max_doubles: float = 1.0,
    max_ebpm: float = 0.0,
    beat_floor: float = 0.0,
    max_run: int = 0,
    drift_bias: float = 0.0,
    momentum_bias: float = 0.0,
    flow_window: float = 1.5,
    hand_bias: float = 0.0,
    precision_prior: float = 0.0,
    extras_out: list | None = None,
) -> list[Note]:
    """Generate a complete Beat Saber map by processing audio in overlapping windows.

    For short audio that fits in a single window, this is equivalent to calling
    generate() + decode_tokens(). For longer audio, the mel is split into
    overlapping windows, each generating a token sequence that is decoded to
    notes and merged using midpoint ownership in the overlap zones.

    Args:
        model: Trained BeatWeaverModel.
        mel_spectrogram: (n_mels, T_audio) — full beat-aligned spectrogram.
        difficulty: Difficulty name (e.g., "Expert").
        config: Model configuration.
        bpm: Song BPM (needed for beat offset calculation and token decoding).
        temperature: Sampling temperature.
        top_k: Top-k filtering (0 = disabled).
        top_p: Top-p / nucleus filtering (1.0 = disabled).
        seed: Random seed for reproducibility.

    Returns:
        List of Note objects spanning the full song, sorted by beat.
    """
    # ponytail: min spacing between notes in frames (1/16 beats); ~an 8th at 140 BPM, a 1/4 at 80
    min_gap = max(2, round(min_gap_seconds * bpm / 60 * SUBDIVISIONS_PER_BAR / 4))
    reset_gap = round(reset_seconds * bpm / 60 * SUBDIVISIONS_PER_BAR / 4)
    ebpm_gap = round(60.0 / max_ebpm * bpm / 60 * SUBDIVISIONS_PER_BAR / 4) if max_ebpm > 0 else 0
    if notes_per_second is not None:
        # Favourites hold notes/second roughly constant across tempos, so a slow song gets more notes per beat.
        notes_per_beat = min(notes_per_second * 60.0 / bpm, 2.5)  # above ~2.5/beat the 16th grid saturates into one run
    lead_in_beat = lead_in_seconds * bpm / 60.0  # no notes before this: the player needs to see the song start
    end_beat = end_seconds * bpm / 60.0 if end_seconds is not None else float("inf")  # and none after the music stops

    total_frames = mel_spectrogram.shape[1]
    max_len = config.max_audio_len
    # The model sometimes keeps writing notes into the zero padding after the
    # audio ends (the last window is real audio + zeros). Nothing past the real
    # audio can be a valid note, so drop it. Frames are 1/16 beat.
    audio_end_beat = total_frames / 16.0

    # Single window — generate directly
    if total_frames <= max_len:
        tokens = generate(
            model, mel_spectrogram, difficulty, config,
            temperature=temperature, top_k=top_k, top_p=top_p, seed=seed,
            audio_bars=_audio_bars(total_frames), onset_guidance=onset_guidance,
            two_stage=two_stage, notes_per_beat=notes_per_beat, min_gap=min_gap, dynamics=dynamics, dynamics_range=dynamics_range,
            flow_penalty=flow_penalty, stream_bias=stream_bias, doubles_bias=doubles_bias, stack_bias=stack_bias, stack_target=stack_target, bomb_bias=bomb_bias, bomb_gap_bias=bomb_gap_bias, bomb_row_bias=bomb_row_bias, diagonal_bias=diagonal_bias, hand_balance=hand_balance, crossover_penalty=crossover_penalty, vision_penalty=vision_penalty, vision_max_run=vision_max_run, column_max_run=column_max_run, raised_max_run=raised_max_run, reversal_penalty=reversal_penalty, beat_focus=beat_focus, block_near_converging=block_near_converging, streams=streams, stacks=stacks,
            reset_gap=reset_gap, max_doubles=max_doubles, ebpm_gap=ebpm_gap, beat_floor=beat_floor, max_run=max_run,
            drift_bias=drift_bias, momentum_bias=momentum_bias, flow_window=flow_window, hand_bias=hand_bias, precision_prior=precision_prior,
        )
        if extras_out is not None:
            extras_out.extend(e for es in decode_extras(tokens, bpm) for e in es if lead_in_beat <= e.beat < min(audio_end_beat, end_beat))
        return [n for n in decode_tokens(tokens, bpm) if lead_in_beat <= n.beat < min(audio_end_beat, end_beat)]

    # Multi-window generation
    overlap = min(max_len // 4, 1024)
    stride = max_len - overlap

    # Compute window start positions
    starts: list[int] = []
    pos = 0
    while pos < total_frames:
        starts.append(pos)
        if pos + max_len >= total_frames:
            break
        pos += stride

    all_window_notes: list[tuple[int, list[Note]]] = []
    all_window_extras: list[list] = []

    for i, start in enumerate(starts):
        end = start + max_len
        window_mel = mel_spectrogram[:, start:end]

        # Zero-pad last window if needed
        if window_mel.shape[1] < max_len:
            pad_size = max_len - window_mel.shape[1]
            window_mel = torch.nn.functional.pad(window_mel, (0, pad_size))

        # Use different seed per window for variety (if seed provided)
        window_seed = seed + i if seed is not None else None

        tokens = generate(
            model, window_mel, difficulty, config,
            temperature=temperature, top_k=top_k, top_p=top_p, seed=window_seed,
            audio_bars=_audio_bars(min(max_len, total_frames - start)),
            onset_guidance=onset_guidance, two_stage=two_stage, notes_per_beat=notes_per_beat, min_gap=min_gap,
            dynamics=dynamics, dynamics_range=dynamics_range, flow_penalty=flow_penalty, stream_bias=stream_bias, doubles_bias=doubles_bias, stack_bias=stack_bias, stack_target=stack_target, bomb_bias=bomb_bias, bomb_gap_bias=bomb_gap_bias, bomb_row_bias=bomb_row_bias, diagonal_bias=diagonal_bias, hand_balance=hand_balance, crossover_penalty=crossover_penalty, vision_penalty=vision_penalty, vision_max_run=vision_max_run, column_max_run=column_max_run, raised_max_run=raised_max_run, reversal_penalty=reversal_penalty, beat_focus=beat_focus, block_near_converging=block_near_converging, streams=streams, stacks=stacks,
            reset_gap=reset_gap, max_doubles=max_doubles, ebpm_gap=ebpm_gap, beat_floor=beat_floor, max_run=max_run,
            drift_bias=drift_bias, momentum_bias=momentum_bias, flow_window=flow_window, hand_bias=hand_bias, precision_prior=precision_prior,
        )

        # Decode tokens — notes have beats relative to window start (bar 0)
        window_notes = decode_tokens(tokens, bpm)
        window_extras = [e for es in decode_extras(tokens, bpm) for e in es] if extras_out is not None else []

        # Offset all beats by the window's start position in frames
        # Each frame = 1/16th note subdivision, beat = frame / 16
        beat_offset = start / 16.0
        for note in (*window_notes, *window_extras):
            note.beat += beat_offset
            note.time_seconds = note.beat * 60.0 / bpm

        all_window_notes.append((start, window_notes))
        all_window_extras.append(window_extras)

    # Merge with midpoint ownership — each window owns notes up to the
    # midpoint of its overlap with the next window, and from the midpoint
    # of its overlap with the previous window.
    result: list[Note] = []
    for i, (start, notes) in enumerate(all_window_notes):
        min_beat = 0.0
        max_beat = audio_end_beat

        if i > 0:
            prev_start = all_window_notes[i - 1][0]
            # Overlap zone between prev and current: [start, prev_start + max_len)
            overlap_mid_frame = (start + prev_start + max_len) / 2.0
            min_beat = overlap_mid_frame / 16.0

        if i < len(all_window_notes) - 1:
            next_start = all_window_notes[i + 1][0]
            # Overlap zone between current and next: [next_start, start + max_len)
            overlap_mid_frame = (next_start + start + max_len) / 2.0
            max_beat = overlap_mid_frame / 16.0

        result.extend(n for n in notes if min_beat <= n.beat < max_beat)
        if extras_out is not None:
            extras_out.extend(e for e in all_window_extras[i] if min_beat <= e.beat < max_beat and lead_in_beat <= e.beat < end_beat)

    result = [n for n in result if lead_in_beat <= n.beat < end_beat]
    result.sort(key=lambda n: (n.beat, n.color))
    if reset_seconds > 0:
        # Windows are decoded independently, so a seam can join two same-way cuts
        result = drop_fast_same_way(result, reset_seconds)
    return result


def drop_fast_same_way(notes: list, reset_seconds: float, min_swing_seconds: float = 0.0) -> list:
    """Drop a hand's note that cuts the same way (``_same_way``) as its previous note under ``reset_seconds``
    later, or that the saber would swing back through to reach within a beat (``_swings_back_through``), or that
    comes under ``min_swing_seconds`` after the hand's last kept swing (the ``max_ebpm`` cap: the decoder lets
    both hands through when both just swung, so a fast stream still held 12% of one song's gaps under it).
    Also drops a crossed-over note cutting back toward its own side, or an outer-column note cutting inward (``_cuts_back``), which parity repair, collision
    reversal and emphasis can create after decoding. Also run after the CLI's shaping passes, whose copied sections
    and added notes can join two."""
    kept, last = [], {}
    for n in sorted(notes, key=lambda n: (n.beat, n.color)):
        if _cuts_back(RIGHT_BASE if n.color else LEFT_BASE, n.x, n.cut_direction):
            continue
        p = last.get(n.color)
        if p and p.beat == n.beat:  # a same-beat group (stack, tower, window, slider) is one swing: keep it whole
            kept.append(n)
            continue
        if p and n.time_seconds - p.time_seconds < reset_seconds and 8 not in (p.cut_direction, n.cut_direction) \
                and _same_way(p.cut_direction, n.cut_direction):
            continue
        if p and n.beat - p.beat <= 1.0 and _swings_back_through(p.x, p.y, p.cut_direction, n.x, n.y, n.cut_direction):
            continue
        if p and n.time_seconds - p.time_seconds < min_swing_seconds - 1e-6:
            continue
        kept.append(n)
        last[n.color] = n
    return kept
