"""Tests for the Beat Saber map tokenizer."""

import logging

import pytest

from beat_weaver.model.tokenizer import (
    BAR_BASE,
    BAR_COUNT,
    bar_token,
    is_bar_token,
    DIFF_EASY,
    DIFF_EXPERT,
    DIFF_EXPERT_PLUS,
    DIFF_HARD,
    DIFF_NORMAL,
    END,
    LEFT_BASE,
    LEFT_EMPTY,
    POS_BASE,
    RIGHT_BASE,
    RIGHT_EMPTY,
    START,
    VOCAB_SIZE,
    _decode_note_token,
    _encode_note_token,
    _quantize_beat,
    decode_tokens,
    describe_token,
    difficulty_to_token,
    encode_beatmap,
    token_to_difficulty,
)
from beat_weaver.schemas.normalized import (
    DifficultyInfo,
    Note,
    NormalizedBeatmap,
    SongMetadata,
)


def _make_beatmap(difficulty: str, notes: list[Note]) -> NormalizedBeatmap:
    return NormalizedBeatmap(
        metadata=SongMetadata(source="test", source_id="test", bpm=120.0),
        difficulty_info=DifficultyInfo(
            characteristic="Standard",
            difficulty=difficulty,
            difficulty_rank=7,
            note_jump_speed=16.0,
            note_jump_offset=0.0,
        ),
        notes=notes,
    )


class TestVocabulary:
    def test_vocab_size(self):
        assert VOCAB_SIZE == 495

    def test_token_ranges_no_overlap(self):
        """All token ranges should be contiguous and non-overlapping."""
        # PAD=0, START=1, END=2, DIFF=3-7, BAR=8, POS=9-72,
        # LEFT_EMPTY=73, LEFT=74-181, RIGHT_EMPTY=182, RIGHT=183-290
        assert LEFT_BASE + 108 == RIGHT_EMPTY  # 74 + 108 = 182
        assert RIGHT_BASE + 108 == BAR_BASE  # 183 + 108 = 291
        assert BAR_BASE + BAR_COUNT == 355  # notes-only vocab; BOMB 355-366, WALL 367-494


class TestCompoundTokenEncoding:
    def test_encode_origin_up(self):
        """Note at (0,0) with direction 0 (up) → base offset."""
        tok = _encode_note_token(LEFT_BASE, 0, 0, 0)
        assert tok == LEFT_BASE

    def test_encode_max_position(self):
        """Note at (3,2) with direction 8 → max offset."""
        tok = _encode_note_token(LEFT_BASE, 3, 2, 8)
        expected = LEFT_BASE + 3 * 27 + 2 * 9 + 8
        assert tok == expected
        assert tok < RIGHT_EMPTY  # within LEFT range

    def test_round_trip_all_positions(self):
        """Encode then decode every valid (x, y, d) combination."""
        for base in (LEFT_BASE, RIGHT_BASE):
            for x in range(4):
                for y in range(3):
                    for d in range(9):
                        tok = _encode_note_token(base, x, y, d)
                        rx, ry, rd = _decode_note_token(tok, base)
                        assert (rx, ry, rd) == (x, y, d)


class TestQuantization:
    def test_beat_zero(self):
        bar, sub = _quantize_beat(0.0)
        assert bar == 0
        assert sub == 0

    def test_beat_one(self):
        """Beat 1.0 = 16 subdivisions into bar 0."""
        bar, sub = _quantize_beat(1.0)
        assert bar == 0
        assert sub == 16

    def test_bar_boundary(self):
        """Beat 4.0 = start of bar 1."""
        bar, sub = _quantize_beat(4.0)
        assert bar == 1
        assert sub == 0

    def test_sixteenth_note(self):
        """Beat 0.25 = 4 subdivisions."""
        bar, sub = _quantize_beat(0.25)
        assert bar == 0
        assert sub == 4

    def test_fractional_quantizes(self):
        """Beat 0.13 rounds to nearest 1/16th (0.125 = sub 2)."""
        bar, sub = _quantize_beat(0.13)
        assert bar == 0
        assert sub == 2  # 0.125 * 16 = 2


class TestDifficultyTokens:
    def test_all_difficulties(self):
        assert difficulty_to_token("Easy") == DIFF_EASY
        assert difficulty_to_token("Normal") == DIFF_NORMAL
        assert difficulty_to_token("Hard") == DIFF_HARD
        assert difficulty_to_token("Expert") == DIFF_EXPERT
        assert difficulty_to_token("ExpertPlus") == DIFF_EXPERT_PLUS

    def test_round_trip(self):
        for name in ["Easy", "Normal", "Hard", "Expert", "ExpertPlus"]:
            assert token_to_difficulty(difficulty_to_token(name)) == name

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            difficulty_to_token("SuperEasy")


class TestEncodeBeatmap:
    def test_empty_map(self):
        bm = _make_beatmap("Expert", [])
        tokens = encode_beatmap(bm)
        assert tokens == [START, DIFF_EXPERT, END]

    def test_single_note(self):
        """A single left note at beat 0."""
        note = Note(beat=0.0, time_seconds=0.0, x=1, y=0, color=0, cut_direction=1)
        bm = _make_beatmap("Hard", [note])
        tokens = encode_beatmap(bm)

        assert tokens[0] == START
        assert tokens[1] == DIFF_HARD
        assert tokens[2] == bar_token(0)
        assert tokens[3] == POS_BASE + 0  # subdivision 0
        # Left note token
        expected_left = _encode_note_token(LEFT_BASE, 1, 0, 1)
        assert tokens[4] == expected_left
        # Right empty
        assert tokens[5] == RIGHT_EMPTY
        assert tokens[6] == END

    def test_both_hands(self):
        """Two notes at same beat, one per hand."""
        notes = [
            Note(beat=0.0, time_seconds=0.0, x=1, y=0, color=0, cut_direction=1),
            Note(beat=0.0, time_seconds=0.0, x=2, y=0, color=1, cut_direction=1),
        ]
        bm = _make_beatmap("Expert", notes)
        tokens = encode_beatmap(bm)

        # Should have LEFT and RIGHT tokens (not EMPTY)
        left_tok = tokens[4]
        right_tok = tokens[5]
        assert LEFT_BASE <= left_tok < LEFT_BASE + 108
        assert RIGHT_BASE <= right_tok < RIGHT_BASE + 108

    def test_multi_bar(self):
        """Notes in bar 0 and bar 1."""
        notes = [
            Note(beat=0.0, time_seconds=0.0, x=0, y=0, color=0, cut_direction=0),
            Note(beat=4.0, time_seconds=2.0, x=0, y=0, color=0, cut_direction=0),
        ]
        bm = _make_beatmap("Expert", notes)
        tokens = encode_beatmap(bm)

        # Count BAR tokens
        bar_count = sum(1 for t in tokens if is_bar_token(t))
        assert bar_count == 2  # bar 0 and bar 1

    def test_same_hand_stack_round_trips(self):
        """A hand's notes at one beat (stack, tower) are all encoded, ascending cell order, and all decode."""
        notes = [
            Note(beat=0.0, time_seconds=0.0, x=1, y=1, color=0, cut_direction=1),
            Note(beat=0.0, time_seconds=0.0, x=1, y=0, color=0, cut_direction=1),
            Note(beat=0.0, time_seconds=0.0, x=3, y=2, color=1, cut_direction=1),
            Note(beat=0.0, time_seconds=0.0, x=3, y=0, color=1, cut_direction=1),
            Note(beat=0.0, time_seconds=0.0, x=3, y=1, color=1, cut_direction=1),
        ]
        tokens = encode_beatmap(_make_beatmap("Expert", notes))
        assert tokens[3:9] == [POS_BASE, _encode_note_token(LEFT_BASE, 1, 0, 1), _encode_note_token(LEFT_BASE, 1, 1, 1),
                               _encode_note_token(RIGHT_BASE, 3, 0, 1), _encode_note_token(RIGHT_BASE, 3, 1, 1),
                               _encode_note_token(RIGHT_BASE, 3, 2, 1)]
        back = decode_tokens(tokens, 120.0)
        assert sorted((n.x, n.y, n.color) for n in back) == sorted((n.x, n.y, n.color) for n in notes)


class TestDecodeTokens:
    def test_empty_sequence(self):
        notes = decode_tokens([START, DIFF_EXPERT, END], bpm=120.0)
        assert notes == []

    def test_round_trip_single_note(self):
        """Encode then decode a single note."""
        original = Note(beat=0.0, time_seconds=0.0, x=2, y=1, color=0, cut_direction=3)
        bm = _make_beatmap("Expert", [original])
        tokens = encode_beatmap(bm)
        decoded = decode_tokens(tokens, bpm=120.0)

        assert len(decoded) == 1
        n = decoded[0]
        assert n.x == original.x
        assert n.y == original.y
        assert n.color == original.color
        assert n.cut_direction == original.cut_direction
        assert n.beat == pytest.approx(0.0)

    def test_round_trip_multiple_notes(self):
        """Encode then decode several notes across bars."""
        originals = [
            Note(beat=0.0, time_seconds=0.0, x=1, y=0, color=0, cut_direction=1),
            Note(beat=0.0, time_seconds=0.0, x=2, y=0, color=1, cut_direction=1),
            Note(beat=1.0, time_seconds=0.5, x=0, y=2, color=0, cut_direction=0),
            Note(beat=4.5, time_seconds=2.25, x=3, y=1, color=1, cut_direction=8),
        ]
        bm = _make_beatmap("Expert", originals)
        tokens = encode_beatmap(bm)
        decoded = decode_tokens(tokens, bpm=120.0)

        assert len(decoded) == len(originals)
        for orig, dec in zip(
            sorted(originals, key=lambda n: (n.beat, n.color)),
            sorted(decoded, key=lambda n: (n.beat, n.color)),
        ):
            assert dec.x == orig.x
            assert dec.y == orig.y
            assert dec.color == orig.color
            assert dec.cut_direction == orig.cut_direction

    def test_time_seconds_computed(self):
        """Verify time_seconds = beat * 60 / bpm."""
        note = Note(beat=2.0, time_seconds=1.0, x=0, y=0, color=0, cut_direction=0)
        bm = _make_beatmap("Expert", [note])
        tokens = encode_beatmap(bm)
        decoded = decode_tokens(tokens, bpm=120.0)

        assert len(decoded) == 1
        assert decoded[0].time_seconds == pytest.approx(1.0)


class TestDescribeToken:
    def test_special_tokens(self):
        assert describe_token(0).name == "PAD"
        assert describe_token(1).name == "START"
        assert describe_token(2).name == "END"

    def test_difficulty_token(self):
        info = describe_token(DIFF_EXPERT)
        assert "Expert" in info.name
        assert info.category == "difficulty"

    def test_position_token(self):
        info = describe_token(POS_BASE + 16)
        assert "POS_16" in info.name
        assert info.category == "position"

    def test_note_token(self):
        tok = _encode_note_token(LEFT_BASE, 2, 1, 3)
        info = describe_token(tok)
        assert "LEFT(2,1,d=3)" == info.name


class TestIndexedBars:
    def test_roundtrip_preserves_absolute_bar(self):
        """A note in bar 40 decodes back to bar 40 even when bars 1-39 are empty."""
        from beat_weaver.schemas.normalized import DifficultyInfo, NormalizedBeatmap, SongMetadata
        notes = [Note(beat=0.0, time_seconds=0, x=0, y=0, color=0, cut_direction=1),
                 Note(beat=161.25, time_seconds=0, x=3, y=2, color=1, cut_direction=0)]
        bm = NormalizedBeatmap(
            metadata=SongMetadata(source="t", source_id="t", hash="t", bpm=120.0),
            difficulty_info=DifficultyInfo(characteristic="Standard", difficulty="Expert",
                                           difficulty_rank=0, note_jump_speed=0, note_jump_offset=0),
            notes=notes)
        tokens = encode_beatmap(bm)
        assert tokens[2] == bar_token(0) and bar_token(40) in tokens
        out = decode_tokens(tokens, 120.0)
        assert sorted(n.beat for n in out) == [0.0, 161.25]

    def test_bars_past_window_are_dropped_not_crashed(self):
        from beat_weaver.schemas.normalized import DifficultyInfo, NormalizedBeatmap, SongMetadata
        notes = [Note(beat=10.0, time_seconds=0, x=0, y=0, color=0, cut_direction=1),
                 Note(beat=900.0, time_seconds=0, x=0, y=0, color=1, cut_direction=1)]
        bm = NormalizedBeatmap(
            metadata=SongMetadata(source="t", source_id="t", hash="t", bpm=120.0),
            difficulty_info=DifficultyInfo(characteristic="Standard", difficulty="Expert",
                                           difficulty_rank=0, note_jump_speed=0, note_jump_offset=0),
            notes=notes)
        tokens = encode_beatmap(bm)
        assert max(t for t in tokens if is_bar_token(t)) == bar_token(BAR_COUNT - 1)
        assert all(n.beat < 256 for n in decode_tokens(tokens, 120.0))


def test_bombs_and_walls_round_trip():
    from beat_weaver.model.tokenizer import decode_extras, encode_wall, decode_wall, is_bomb_token, is_wall_token
    from beat_weaver.schemas.normalized import Bomb, Obstacle

    for x in range(4):
        for w in range(1, 5 - x):
            for crouch in (False, True):
                for d in (0.5, 1.0, 2.0, 4.0):
                    assert decode_wall(encode_wall(x, w, crouch, d)) == (x, w, crouch, d)
    assert decode_wall(encode_wall(3, 4, False, 3.0))[1] == 1  # width clipped to the grid, 3 beats -> nearest bucket
    bm = _make_beatmap("Expert", [Note(beat=1.0, time_seconds=0.5, x=0, y=0, color=0, cut_direction=1)])
    bm.bombs = [Bomb(beat=1.0, time_seconds=0.5, x=3, y=1), Bomb(beat=1.0, time_seconds=0.5, x=2, y=1),  # a cluster
               Bomb(beat=1.0, time_seconds=0.5, x=2, y=1),  # duplicate cell: kept once
               Bomb(beat=2.5, time_seconds=1.25, x=1, y=2)]  # on an empty position
    bm.obstacles = [Obstacle(beat=4.0, time_seconds=2.0, duration_beats=2.0, x=0, y=2, width=4, height=3)]
    toks = encode_beatmap(bm)
    assert sum(is_bomb_token(t) for t in toks) == 3 and sum(is_wall_token(t) for t in toks) == 1
    bombs, walls = decode_extras(toks, 120.0)
    assert [(b.beat, b.x, b.y) for b in bombs] == [(1.0, 2, 1), (1.0, 3, 1), (2.5, 1, 2)]  # ascending cell order
    assert [(w.beat, w.x, w.width, w.height, w.duration_beats) for w in walls] == [(4.0, 0, 4, 3, 2.0)]
    assert [(n.beat, n.x) for n in decode_tokens(toks, 120.0)] == [(1.0, 0)]
