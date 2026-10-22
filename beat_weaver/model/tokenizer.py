"""Token vocabulary and encode/decode for Beat Saber maps.

Converts between NormalizedBeatmap and integer token sequences.
No PyTorch dependency — pure Python + existing dataclasses.

Token vocabulary (355 tokens):
    0       PAD
    1       START
    2       END
    3-7     DIFF_Easy .. DIFF_ExpertPlus
    8       (unused; was the single BAR token)
    9-72    POS_0 .. POS_63  (1/16th note positions in a 4-beat bar)
    73      LEFT_EMPTY
    74-181  LEFT_x_y_d  (4 cols × 3 rows × 9 dirs)
    182     RIGHT_EMPTY
    183-290 RIGHT_x_y_d (4 cols × 3 rows × 9 dirs)
    291-354 BAR_0 .. BAR_63  (absolute bar index within the audio window)

Bars are indexed, not counted. With a single BAR token the decoder had to
count bars itself across 1000+ tokens and drifted: measured on a trained
model, a 64-bar window produced 127 BAR tokens, spreading the right number
of notes over twice the time (half density) and pushing the tail past the
audio. BAR_k names the bar outright, so the grammar can force BAR_k → BAR_k+1
and the audio position maps to the token directly.

Compound note encoding: base + x * 27 + y * 9 + direction

A hand may emit several note tokens at one POS, in ascending token order (cell, then cut), for a stack,
tower, window or slider; the grammar only allows a higher cell with the same cut after the first.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from beat_weaver.schemas.normalized import Bomb, Note, NormalizedBeatmap, Obstacle

logger = logging.getLogger(__name__)

# ── Special tokens ──────────────────────────────────────────────────────────

PAD = 0
START = 1
END = 2

# ── Difficulty tokens ───────────────────────────────────────────────────────

DIFF_EASY = 3
DIFF_NORMAL = 4
DIFF_HARD = 5
DIFF_EXPERT = 6
DIFF_EXPERT_PLUS = 7

_DIFF_NAMES = ["Easy", "Normal", "Hard", "Expert", "ExpertPlus"]
_DIFF_TO_TOKEN = {name: DIFF_EASY + i for i, name in enumerate(_DIFF_NAMES)}
_TOKEN_TO_DIFF = {v: k for k, v in _DIFF_TO_TOKEN.items()}

# Case-insensitive lookup + aliases (BeatSaver maps use inconsistent casing)
_DIFF_LOOKUP = {name.lower(): token for name, token in _DIFF_TO_TOKEN.items()}
_DIFF_LOOKUP["expert+"] = DIFF_EXPERT_PLUS

# ── Bar / position tokens ──────────────────────────────────────────────────

POS_BASE = 9
POS_COUNT = 64  # 4 beats × 16 subdivisions

# ── Note tokens ─────────────────────────────────────────────────────────────

LEFT_EMPTY = 73
LEFT_BASE = 74
LEFT_COUNT = 108  # 4 × 3 × 9

RIGHT_EMPTY = 182
RIGHT_BASE = 183
RIGHT_COUNT = 108

# Indexed bar tokens live after the note tokens so every pre-existing ID is unchanged.
BAR_BASE = 291
BAR_COUNT = 64  # max_audio_len (4096 frames) / 64 subdivisions per bar
VOCAB_SIZE_NOTES_ONLY = BAR_BASE + BAR_COUNT  # 355: checkpoints before bombs/walls

# Optional per-position extras after RIGHT: BOMB tokens in ascending cell order (a cluster at one instant: rows,
# frames; 66% of Expert bombs share their instant), then at most one WALL token.
BOMB_BASE = VOCAB_SIZE_NOTES_ONLY  # 355
BOMB_COUNT = COLS_ROWS = 12  # x * 3 + y
WALL_BASE = BOMB_BASE + BOMB_COUNT  # 367
WALL_DURATIONS = (0.5, 1.0, 2.0, 4.0)  # beats, bucketed
WALL_COUNT = 4 * 4 * 2 * len(WALL_DURATIONS)  # x, width, type (0 full, 1 crouch), duration -> 128
VOCAB_SIZE = WALL_BASE + WALL_COUNT  # 495


def is_bomb_token(token_id: int) -> bool:
    return BOMB_BASE <= token_id < BOMB_BASE + BOMB_COUNT


def is_wall_token(token_id: int) -> bool:
    return WALL_BASE <= token_id < WALL_BASE + WALL_COUNT


def encode_wall(x: int, width: int, crouch: bool, duration_beats: float) -> int:
    width = max(1, min(width, 4 - x))
    d = min(range(len(WALL_DURATIONS)), key=lambda i: abs(WALL_DURATIONS[i] - duration_beats))
    return WALL_BASE + ((x * 4 + (width - 1)) * 2 + int(crouch)) * len(WALL_DURATIONS) + d


def decode_wall(token: int) -> tuple[int, int, bool, float]:
    """-> (x, width, crouch, duration_beats)"""
    v = token - WALL_BASE
    d, v = v % len(WALL_DURATIONS), v // len(WALL_DURATIONS)
    crouch, v = bool(v % 2), v // 2
    return v // 4, v % 4 + 1, crouch, WALL_DURATIONS[d]


def bar_token(bar_index: int) -> int:
    """Token ID for bar ``bar_index`` (0-based within the audio window)."""
    if not 0 <= bar_index < BAR_COUNT:
        raise ValueError(f"bar index {bar_index} outside 0..{BAR_COUNT - 1}")
    return BAR_BASE + bar_index


def is_bar_token(token_id: int) -> bool:
    return BAR_BASE <= token_id < BAR_BASE + BAR_COUNT

# Grid constants
COLS = 4
ROWS = 3
DIRS = 9
SUBDIVISIONS_PER_BEAT = 16
BEATS_PER_BAR = 4
SUBDIVISIONS_PER_BAR = SUBDIVISIONS_PER_BEAT * BEATS_PER_BAR  # 64


def _encode_note_token(base: int, x: int, y: int, direction: int) -> int:
    """Encode a note placement into a compound token ID."""
    assert 0 <= x < COLS
    assert 0 <= y < ROWS
    assert 0 <= direction < DIRS
    return base + x * 27 + y * 9 + direction


def _decode_note_token(token: int, base: int) -> tuple[int, int, int]:
    """Decode a compound note token into (x, y, direction)."""
    offset = token - base
    x = offset // 27
    y = (offset % 27) // 9
    direction = offset % 9
    return x, y, direction


def difficulty_to_token(difficulty: str) -> int:
    """Map difficulty name to token ID. Raises ValueError for unknown."""
    token = _DIFF_LOOKUP.get(difficulty.lower())
    if token is None:
        raise ValueError(
            f"Unknown difficulty {difficulty!r}. "
            f"Expected one of: {_DIFF_NAMES}"
        )
    return token


def token_to_difficulty(token: int) -> str:
    """Map token ID to difficulty name. Raises ValueError for non-difficulty token."""
    name = _TOKEN_TO_DIFF.get(token)
    if name is None:
        raise ValueError(f"Token {token} is not a difficulty token (expected {DIFF_EASY}-{DIFF_EXPERT_PLUS})")
    return name


def _quantize_beat(beat: float) -> tuple[int, int]:
    """Quantize a beat value to (bar_index, subdivision_within_bar).

    Returns 0-based bar index and 0-63 subdivision position.
    """
    total_subdivisions = round(beat * SUBDIVISIONS_PER_BEAT)
    bar_index = total_subdivisions // SUBDIVISIONS_PER_BAR
    sub_in_bar = total_subdivisions % SUBDIVISIONS_PER_BAR
    return bar_index, sub_in_bar


def encode_beatmap(beatmap: NormalizedBeatmap) -> list[int]:
    """Encode a NormalizedBeatmap into a token ID sequence.

    Steps:
        1. Sort notes by beat
        2. Quantize beats to 1/16th note grid
        3. Group notes by quantized position
        4. Emit: START DIFF_x BAR_0 POS_p LEFT_tok... RIGHT_tok... ... BAR_1 ... END
        5. A hand's notes at one position all go in, ascending token order (stacks)
    """
    difficulty = beatmap.difficulty_info.difficulty
    diff_token = difficulty_to_token(difficulty)

    # Sort notes by beat
    notes = sorted(beatmap.notes, key=lambda n: n.beat)

    if not notes:
        return [START, diff_token, END]

    # Group notes by (bar_index, subdivision); bombs and walls (on-grid only) keyed the same way
    groups: dict[tuple[int, int], list[Note]] = {}
    for note in notes:
        key = _quantize_beat(note.beat)
        groups.setdefault(key, []).append(note)
    bombs: dict[tuple[int, int], set[int]] = {}
    for b in beatmap.bombs:
        if 0 <= b.x < COLS and 0 <= b.y < ROWS:
            bombs.setdefault(_quantize_beat(b.beat), set()).add(BOMB_BASE + b.x * ROWS + b.y)
    walls: dict[tuple[int, int], int] = {}
    for w in sorted(beatmap.obstacles, key=lambda w: w.beat):
        if 0 <= w.x < COLS and w.width >= 1 and w.duration_beats > 0:
            walls.setdefault(_quantize_beat(w.beat), encode_wall(w.x, w.width, w.height <= 3, w.duration_beats))
    for key in (*bombs, *walls):
        groups.setdefault(key, [])

    # Find the range of bars we need. Bars past the audio window cannot be
    # named (and are never in the labels: dataset clips notes to the window).
    max_bar = min(max(bar for bar, _ in groups), BAR_COUNT - 1)

    tokens: list[int] = [START, diff_token]

    for bar_idx in range(max_bar + 1):
        tokens.append(bar_token(bar_idx))

        # Collect all subdivisions in this bar that have notes
        bar_subs = sorted(
            sub for (b, sub) in groups if b == bar_idx
        )

        for sub in bar_subs:
            tokens.append(POS_BASE + sub)

            group_notes = groups[(bar_idx, sub)]
            # Every note of a hand at this instant, in ascending token order (cell, then cut): a stack, tower,
            # window or slider is one swing and reads as one group, like a bomb cluster. 73% of real Expert maps
            # have stacks (median 1.3 per 100 notes); keeping one note per hand hid every one from the model.
            for base, color, empty in ((LEFT_BASE, 0, LEFT_EMPTY), (RIGHT_BASE, 1, RIGHT_EMPTY)):
                hand = sorted({_encode_note_token(base, n.x, n.y, n.cut_direction) for n in group_notes if n.color == color})
                tokens.extend(hand or [empty])

            tokens.extend(sorted(bombs.get((bar_idx, sub), ())))
            if (bar_idx, sub) in walls:
                tokens.append(walls[(bar_idx, sub)])

    tokens.append(END)
    return tokens


def decode_extras(token_ids: list[int], bpm: float) -> tuple[list[Bomb], list[Obstacle]]:
    """Bombs and walls from a token sequence (decode_tokens() returns the colour notes)."""
    bombs: list[Bomb] = []
    walls: list[Obstacle] = []
    current_bar, beat = -1, 0.0
    for tok in token_ids:
        if is_bar_token(tok):
            current_bar = tok - BAR_BASE
        elif POS_BASE <= tok < POS_BASE + POS_COUNT:
            beat = (current_bar * SUBDIVISIONS_PER_BAR + tok - POS_BASE) / SUBDIVISIONS_PER_BEAT
        elif is_bomb_token(tok):
            v = tok - BOMB_BASE
            bombs.append(Bomb(beat=beat, time_seconds=beat * 60.0 / bpm, x=v // ROWS, y=v % ROWS))
        elif is_wall_token(tok):
            x, width, crouch, dur = decode_wall(tok)
            walls.append(Obstacle(beat=beat, time_seconds=beat * 60.0 / bpm, duration_beats=dur, x=x,
                                  y=2 if crouch else 0, width=width, height=3 if crouch else 5))
    return bombs, walls


def decode_tokens(token_ids: list[int], bpm: float) -> list[Note]:
    """Decode a token ID sequence back into a list of Note objects.

    Args:
        token_ids: Token sequence (should start with START and end with END).
        bpm: Beats per minute for computing time_seconds.

    Returns:
        List of Note objects sorted by beat.
    """
    notes: list[Note] = []
    current_bar = -1
    current_sub = 0

    i = 0
    while i < len(token_ids):
        tok = token_ids[i]

        if tok in (PAD, START, END) or DIFF_EASY <= tok <= DIFF_EXPERT_PLUS:
            i += 1
            continue

        if is_bar_token(tok):
            current_bar = tok - BAR_BASE
            i += 1
            continue

        if POS_BASE <= tok < POS_BASE + POS_COUNT:
            current_sub = tok - POS_BASE
            beat = (current_bar * SUBDIVISIONS_PER_BAR + current_sub) / SUBDIVISIONS_PER_BEAT
            time_seconds = beat * 60.0 / bpm

            # LEFT notes (or LEFT_EMPTY), then RIGHT notes (or RIGHT_EMPTY); a hand may have several (a stack)
            i += 1
            for base, color, empty in ((LEFT_BASE, 0, LEFT_EMPTY), (RIGHT_BASE, 1, RIGHT_EMPTY)):
                if i < len(token_ids) and token_ids[i] == empty:
                    i += 1
                    continue
                while i < len(token_ids) and base <= token_ids[i] < base + LEFT_COUNT:
                    x, y, d = _decode_note_token(token_ids[i], base)
                    notes.append(Note(beat=beat, time_seconds=time_seconds, x=x, y=y, color=color, cut_direction=d))
                    i += 1
            continue

        # Unknown token — skip
        i += 1

    return sorted(notes, key=lambda n: n.beat)


@dataclass
class TokenInfo:
    """Human-readable description of a token for debugging."""

    id: int
    name: str
    category: str


def describe_token(token_id: int) -> TokenInfo:
    """Return human-readable info about a token ID."""
    if token_id == PAD:
        return TokenInfo(token_id, "PAD", "special")
    if token_id == START:
        return TokenInfo(token_id, "START", "special")
    if token_id == END:
        return TokenInfo(token_id, "END", "special")
    if DIFF_EASY <= token_id <= DIFF_EXPERT_PLUS:
        name = _TOKEN_TO_DIFF[token_id]
        return TokenInfo(token_id, f"DIFF_{name}", "difficulty")
    if is_bar_token(token_id):
        return TokenInfo(token_id, f"BAR_{token_id - BAR_BASE}", "structure")
    if POS_BASE <= token_id < POS_BASE + POS_COUNT:
        pos = token_id - POS_BASE
        beat_in_bar = pos / SUBDIVISIONS_PER_BEAT
        return TokenInfo(token_id, f"POS_{pos} (beat {beat_in_bar:.4g})", "position")
    if token_id == LEFT_EMPTY:
        return TokenInfo(token_id, "LEFT_EMPTY", "note")
    if LEFT_BASE <= token_id < LEFT_BASE + LEFT_COUNT:
        x, y, d = _decode_note_token(token_id, LEFT_BASE)
        return TokenInfo(token_id, f"LEFT({x},{y},d={d})", "note")
    if token_id == RIGHT_EMPTY:
        return TokenInfo(token_id, "RIGHT_EMPTY", "note")
    if RIGHT_BASE <= token_id < RIGHT_BASE + RIGHT_COUNT:
        x, y, d = _decode_note_token(token_id, RIGHT_BASE)
        return TokenInfo(token_id, f"RIGHT({x},{y},d={d})", "note")
    return TokenInfo(token_id, f"UNKNOWN_{token_id}", "unknown")
