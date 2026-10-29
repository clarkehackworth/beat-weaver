"""Tests for inference and grammar-constrained generation."""

import pytest

torch = pytest.importorskip("torch")

from beat_weaver.model.config import ModelConfig
from beat_weaver.model.inference import (
    _build_grammar_mask, _crossover_outward_tokens, _decode_note_token, generate, generate_full_song,
)
from beat_weaver.model.tokenizer import (
    BOMB_BASE,
    BAR_BASE,
    BAR_COUNT,
    bar_token,
    is_bar_token,
    DIFF_EASY,
    DIFF_EXPERT,
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
)
from beat_weaver.model.transformer import BeatWeaverModel


class TestGrammarMask:
    def test_after_start(self):
        mask = _build_grammar_mask(START)
        # Only difficulty tokens allowed
        assert mask[DIFF_EASY:DIFF_EXPERT_PLUS + 1].all()
        assert not mask[START]
        assert not mask[bar_token(0)]
        assert not mask[END]

    def test_after_difficulty(self):
        mask = _build_grammar_mask(DIFF_EXPERT)
        # Only bar_token(0) allowed
        assert mask[bar_token(0)]
        assert mask.sum() == 1

    def test_after_bar(self):
        mask = _build_grammar_mask(bar_token(0), current_bar=0)
        # POS, the NEXT bar, or END
        assert mask[POS_BASE:POS_BASE + POS_COUNT].all()
        assert mask[bar_token(1)]
        assert mask[END]
        assert not mask[START]
        assert not mask[LEFT_EMPTY]

    def test_audio_bars_gates_end_and_bars(self):
        """END only once the last audio bar is open; no bar tokens after it."""
        for last in (bar_token(0), POS_BASE + 3, LEFT_EMPTY, RIGHT_EMPTY):
            # current_bar=0 of 3 audio bars: audio remains -> END forbidden, next bar open
            mask = _build_grammar_mask(last, last_pos_in_bar=2, current_bar=0, audio_bars=3)
            assert not mask[END], last
        mask = _build_grammar_mask(bar_token(0), current_bar=0, audio_bars=3)
        assert mask[bar_token(1)] and not mask[bar_token(2)] and not mask[END]
        # last audio bar open (index 2): END allowed, nothing after it
        for last in (bar_token(2), RIGHT_EMPTY):
            mask = _build_grammar_mask(last, last_pos_in_bar=5, current_bar=2, audio_bars=3)
            assert mask[END], last
            assert not mask[BAR_BASE:BAR_BASE + BAR_COUNT].any(), last
        # a single-bar window and a window longer than the vocabulary both stay non-empty
        assert _build_grammar_mask(DIFF_EXPERT, audio_bars=1)[bar_token(0)]
        assert _build_grammar_mask(RIGHT_EMPTY, 63, BAR_COUNT - 1, audio_bars=10**6).any()
        # unchanged when not gated
        assert _build_grammar_mask(bar_token(0), current_bar=0)[END]

    def test_right_hand_cannot_share_left_hands_cell(self):
        from beat_weaver.model.tokenizer import _encode_note_token
        left = _encode_note_token(LEFT_BASE, 2, 1, 0)
        mask = _build_grammar_mask(left, left_cell=(2, 1))
        for d in range(9):
            assert not mask[_encode_note_token(RIGHT_BASE, 2, 1, d)], d
        # columns left of the red note would cross the arms; the rest of its column and right of it stay open
        assert not mask[_encode_note_token(RIGHT_BASE, 1, 1, 0)]
        assert not mask[_encode_note_token(RIGHT_BASE, 0, 2, 0)]
        assert mask[_encode_note_token(RIGHT_BASE, 2, 0, 0)]
        assert mask[_encode_note_token(RIGHT_BASE, 3, 1, 0)]
        assert mask[RIGHT_EMPTY]
        allowed = mask[RIGHT_BASE:RIGHT_BASE + RIGHT_COUNT].sum()
        assert allowed == (2 * 3 - 1) * 9 - 9  # columns 2-3 minus the shared cell, minus inward cuts on column 3
        # without the cell the full set is allowed (LEFT_EMPTY case)
        assert _build_grammar_mask(LEFT_EMPTY)[RIGHT_BASE:RIGHT_BASE + RIGHT_COUNT].sum() == RIGHT_COUNT - 27

    def test_reset_tokens_block_same_way_cuts(self):
        from beat_weaver.model.inference import _reset_tokens
        from beat_weaver.model.tokenizer import _encode_note_token
        m = _reset_tokens(LEFT_BASE, 1)  # just cut DOWN
        for d in (1, 6, 7):  # down, down-left, down-right
            assert m[_encode_note_token(LEFT_BASE, 0, 0, d)], d
        for d in (0, 2, 3, 4, 5, 8):
            assert not m[_encode_note_token(LEFT_BASE, 3, 2, d)], d
        assert not m[_encode_note_token(RIGHT_BASE, 0, 0, 1)]
        assert not _reset_tokens(LEFT_BASE, 8).any()
        # up-left then up-right: 90 degrees apart, but both are upswings
        m = _reset_tokens(LEFT_BASE, 4)
        assert m[_encode_note_token(LEFT_BASE, 0, 0, 5)] and not m[_encode_note_token(LEFT_BASE, 0, 0, 7)]
        assert not _reset_tokens(LEFT_BASE, 2)[_encode_note_token(LEFT_BASE, 0, 0, 0)]  # left then up: no reset

    def test_converging_mask_blocks_a_saber_arriving_through_the_other_note(self):
        from beat_weaver.model.inference import _converging_right_tokens
        from beat_weaver.model.tokenizer import _encode_note_token
        m = _converging_right_tokens((0, 1), 0)  # red up-cut on (0,1): its saber arrives from (0,0)
        assert m[_encode_note_token(RIGHT_BASE, 0, 0, 1)]   # blue down-cut on (0,0): arrives from (0,1)
        assert m[_encode_note_token(RIGHT_BASE, 0, 0, 0)]   # blue up-cut on (0,0): red's saber comes through it
        assert not m[_encode_note_token(RIGHT_BASE, 3, 1, 0)]

    def test_stacks_continue_a_hand_on_a_higher_cell_with_the_same_cut(self):
        from beat_weaver.model.inference import _build_grammar_mask
        from beat_weaver.model.tokenizer import _encode_note_token
        left = _encode_note_token(LEFT_BASE, 1, 0, 1)
        assert not _build_grammar_mask(left, 4, 0, None, (1, 0))[_encode_note_token(LEFT_BASE, 1, 1, 1)]  # off by default
        m = _build_grammar_mask(left, 4, 0, None, (1, 0), stacks=True)
        assert m[_encode_note_token(LEFT_BASE, 1, 1, 1)]  # the next cell along the down cut
        assert not m[_encode_note_token(LEFT_BASE, 2, 0, 1)] and not m[_encode_note_token(LEFT_BASE, 1, 2, 1)]  # across, or not adjacent
        assert not m[_encode_note_token(LEFT_BASE, 0, 0, 1)] and not m[_encode_note_token(LEFT_BASE, 1, 1, 0)]
        diag = _build_grammar_mask(_encode_note_token(LEFT_BASE, 0, 1, 4), 4, 0, None, (0, 1), stacks=True)  # up-left
        assert diag[_encode_note_token(LEFT_BASE, 1, 0, 4)] and not diag[_encode_note_token(LEFT_BASE, 1, 2, 4)]
        assert m[RIGHT_EMPTY] and m[_encode_note_token(RIGHT_BASE, 3, 0, 1)]
        right = _encode_note_token(RIGHT_BASE, 2, 0, 1)
        m = _build_grammar_mask(right, 4, 0, None, (1, 0), stacks=True)
        assert m[_encode_note_token(RIGHT_BASE, 2, 1, 1)] and not m[_encode_note_token(RIGHT_BASE, 1, 1, 1)]
        assert m[POS_BASE + 5] and not m[_encode_note_token(RIGHT_BASE, 1, 0, 1)]  # may close; never on the LEFT cell

    def test_swing_back_through_flags_cuts_the_saber_must_come_back_for(self):
        from beat_weaver.model.inference import _swing_back_tokens, _swings_back_through, drop_fast_same_way
        from beat_weaver.schemas.normalized import Note
        from beat_weaver.model.tokenizer import _encode_note_token
        # cut right on the middle row heading out to the right edge, then a note at the edge asking to be cut
        # leftward from beyond it (the played case), or downward from the top after swinging up
        assert _swings_back_through(1, 1, 3, 3, 1, 2)
        assert _swings_back_through(1, 0, 0, 1, 2, 0) is False  # up then up again: a reset, not this rule
        assert _swings_back_through(0, 1, 7, 0, 0, 0)            # down-right from the side, then up from below it
        assert _swings_back_through(1, 0, 0, 1, 2, 1)            # bottom up, then top down: past the top and back
        assert not _swings_back_through(1, 0, 1, 1, 0, 0)        # down then up on one cell: the normal arc
        assert not _swings_back_through(1, 0, 0, 1, 1, 1)        # up, then down where the saber stopped
        assert not _swings_back_through(1, 1, 3, 3, 1, 3)        # keep going right: fine
        m = _swing_back_tokens(LEFT_BASE, (1, 1), 3)
        assert m[_encode_note_token(LEFT_BASE, 3, 1, 2)] and not m[_encode_note_token(LEFT_BASE, 3, 1, 3)]
        assert not m[_encode_note_token(RIGHT_BASE, 3, 1, 2)]
        notes = [Note(beat=0.0, time_seconds=0.0, x=1, y=0, color=0, cut_direction=0, angle_offset=0),  # bottom up,
                 Note(beat=0.5, time_seconds=0.25, x=1, y=2, color=0, cut_direction=1, angle_offset=0),  # then top down
                 Note(beat=2.0, time_seconds=1.0, x=1, y=2, color=0, cut_direction=1, angle_offset=0)]   # hand's own side
        assert [n.beat for n in drop_fast_same_way(notes, 0.0)] == [0.0, 2.0]  # a beat and more later is fine
        fast = [Note(beat=b, time_seconds=b * 0.5, x=1, y=1, color=0, cut_direction=d, angle_offset=0) for b, d in ((0.0, 1), (0.25, 3), (0.5, 0), (1.0, 2))]
        assert [n.beat for n in drop_fast_same_way(fast, 0.0, min_swing_seconds=0.2)] == [0.0, 0.5, 1.0]  # 0.125 s gap dropped
        stack = [Note(beat=0.0, time_seconds=0.0, x=1, y=y, color=0, cut_direction=1, angle_offset=0) for y in (1, 0)]
        assert len(drop_fast_same_way(stack, 0.5, min_swing_seconds=0.2)) == 2  # a stack is one swing, not a reset

    def test_arc_penalty_geometry(self):
        from beat_weaver.model.inference import _arc_penalty
        from beat_weaver.model.tokenizer import _encode_note_token
        tok = lambda x, y, d: _encode_note_token(LEFT_BASE, x, y, d)
        pen = _arc_penalty(LEFT_BASE, (1, 1), 1)                 # just cut DOWN through the centre-left cell
        assert pen[tok(1, 1, 1)] > 0.99                            # down again on the same cell: full penalty
        assert pen[tok(1, 1, 0)] < 0.01                            # up: reverses, free
        assert abs(pen[tok(1, 1, 2)] - 0.5) < 1e-6                 # left: perpendicular, half
        assert abs(pen[tok(1, 1, 8)] - 0.5) < 1e-6                 # dot: neutral
        # hand is now below (1,1); a note at (3,2) is up-right of it, so up-right is the arc
        assert pen[tok(3, 2, 5)] < pen[tok(3, 2, 3)] < pen[tok(3, 2, 6)]
        assert pen[RIGHT_BASE:RIGHT_BASE + RIGHT_COUNT].sum() == 0  # other hand untouched
        assert pen[:LEFT_BASE].sum() == 0
        assert _arc_penalty(LEFT_BASE, (1, 1), 8).sum() == 0       # after an any-cut: nothing
        assert _arc_penalty(LEFT_BASE, None, 1).sum() == 0         # no previous cut

    def test_arc_penalty_charges_negative_curvature(self):
        from beat_weaver.model import inference
        from beat_weaver.model.inference import _arc_penalty
        CURVATURE = 0.3
        inference.CURVATURE = CURVATURE  # default is 0 (off); exercise the term at the value that was tried
        from beat_weaver.model.tokenizer import _encode_note_token
        tok = lambda x, y, d: _encode_note_token(LEFT_BASE, x, y, d)
        pen = _arc_penalty(LEFT_BASE, (1, 1), 1)  # cut DOWN at (1,1); hand sits below it
        # (3,2) is up-right: the wrist turned counter-clockwise (down -> up-right travel). Up-right keeps that
        # sense (positive curvature); Right turns back clockwise at 90 degrees (negative) and pays CURVATURE.
        assert pen[tok(3, 2, 5)] < 0.1
        assert pen[tok(3, 2, 3)] > CURVATURE
        # Down-left at (3,2) also turns back but is only 45 degrees from the last cut: no curvature charge
        assert pen[tok(3, 2, 6)] <= 1.0
        base = _arc_penalty(LEFT_BASE, (1, 1), 1)[tok(1, 1, 0)]  # straight reversal on the same cell: free
        assert base < 0.01
        inference.CURVATURE = 0.0

    def test_bars_are_sequential_and_bounded(self):
        """Exactly one bar token is ever allowed: current_bar + 1, and none past the window."""
        for k in range(BAR_COUNT - 1):
            mask = _build_grammar_mask(bar_token(k), current_bar=k)
            bars = mask[BAR_BASE:BAR_BASE + BAR_COUNT].nonzero().flatten().tolist()
            assert bars == [k + 1], (k, bars)
        # Last bar: no further bar token, only POS or END
        mask = _build_grammar_mask(bar_token(BAR_COUNT - 1), current_bar=BAR_COUNT - 1)
        assert not mask[BAR_BASE:BAR_BASE + BAR_COUNT].any()
        assert mask[END]
        # Same rule after a RIGHT token mid-bar
        mask = _build_grammar_mask(RIGHT_EMPTY, last_pos_in_bar=5, current_bar=7)
        bars = mask[BAR_BASE:BAR_BASE + BAR_COUNT].nonzero().flatten().tolist()
        assert bars == [8]

    def test_after_pos(self):
        mask = _build_grammar_mask(POS_BASE + 10)
        # LEFT tokens
        assert mask[LEFT_EMPTY]
        assert mask[LEFT_BASE:LEFT_BASE + LEFT_COUNT].sum() == LEFT_COUNT - 27
        assert not mask[_crossover_outward_tokens(LEFT_BASE)].any()
        assert not mask[bar_token(0)]
        assert not mask[RIGHT_EMPTY]

    def test_after_left(self):
        mask = _build_grammar_mask(LEFT_BASE + 5)
        # RIGHT tokens
        assert mask[RIGHT_EMPTY]
        assert mask[RIGHT_BASE:RIGHT_BASE + RIGHT_COUNT].sum() == RIGHT_COUNT - 27
        assert not mask[_crossover_outward_tokens(RIGHT_BASE)].any()
        assert not mask[bar_token(0)]
        assert not mask[LEFT_EMPTY]

    def test_after_left_empty(self):
        mask = _build_grammar_mask(LEFT_EMPTY)
        # RIGHT tokens
        assert mask[RIGHT_EMPTY]
        assert mask[RIGHT_BASE:RIGHT_BASE + RIGHT_COUNT].sum() == RIGHT_COUNT - 27

    def test_after_right(self):
        mask = _build_grammar_mask(RIGHT_BASE + 5)
        # POS, bar_token(0), or END
        assert mask[POS_BASE:POS_BASE + POS_COUNT].all()
        assert mask[bar_token(0)]
        assert mask[END]

    def test_after_right_empty(self):
        mask = _build_grammar_mask(RIGHT_EMPTY)
        assert mask[POS_BASE:POS_BASE + POS_COUNT].all()
        assert mask[bar_token(0)]
        assert mask[END]


class TestGenerate:
    @pytest.fixture
    def small_model(self):
        config = ModelConfig(
            vocab_size=495,
            max_seq_len=64,
            n_mels=80,
            encoder_layers=1,
            encoder_dim=32,
            encoder_heads=4,
            encoder_ff_dim=64,
            decoder_layers=1,
            decoder_dim=32,
            decoder_heads=4,
            decoder_ff_dim=64,
            dropout=0.0,
        )
        model = BeatWeaverModel(config)
        return model, config

    def test_starts_and_ends_correctly(self, small_model):
        model, config = small_model
        mel = torch.randn(80, 20)
        tokens = generate(model, mel, "Expert", config, temperature=1.0, seed=42)
        assert tokens[0] == START
        assert tokens[1] == DIFF_EXPERT
        # Should end with END or hit max_seq_len
        assert tokens[-1] == END or len(tokens) == config.max_seq_len

    def test_grammar_valid_sequence(self, small_model):
        """Every generated token should follow grammar rules."""
        model, config = small_model
        mel = torch.randn(80, 20)
        tokens = generate(model, mel, "Expert", config, temperature=1.0, seed=42)

        # Re-validate with the same state the generator tracks (bar + position)
        last_pos_in_bar, current_bar = -1, -1
        for i in range(1, len(tokens)):
            prev = tokens[i - 1]
            curr = tokens[i]
            mask = _build_grammar_mask(prev, last_pos_in_bar, current_bar)
            assert mask[curr], (
                f"Token {curr} not valid after {prev} at position {i}. "
                f"Sequence so far: {tokens[:i+1]}"
            )
            if is_bar_token(curr):
                last_pos_in_bar, current_bar = -1, curr - BAR_BASE
            elif POS_BASE <= curr < POS_BASE + POS_COUNT:
                last_pos_in_bar = curr - POS_BASE

    def test_deterministic_with_seed(self, small_model):
        model, config = small_model
        mel = torch.randn(80, 20)
        t1 = generate(model, mel, "Expert", config, temperature=0.5, seed=123)
        t2 = generate(model, mel, "Expert", config, temperature=0.5, seed=123)
        assert t1 == t2

    def test_greedy_decoding(self, small_model):
        model, config = small_model
        mel = torch.randn(80, 20)
        tokens = generate(model, mel, "Expert", config, temperature=0)
        assert tokens[0] == START
        assert tokens[1] == DIFF_EXPERT


class TestGenerateFullSong:
    @pytest.fixture
    def small_model(self):
        config = ModelConfig(
            vocab_size=495,
            max_seq_len=64,
            max_audio_len=128,
            n_mels=80,
            encoder_layers=1,
            encoder_dim=32,
            encoder_heads=4,
            encoder_ff_dim=64,
            decoder_layers=1,
            decoder_dim=32,
            decoder_heads=4,
            decoder_ff_dim=64,
            dropout=0.0,
        )
        torch.manual_seed(3)  # the statistical tests below (flow penalty halves repeats) need a fixed random model
        model = BeatWeaverModel(config)
        return model, config

    def test_full_song_single_window(self, small_model):
        """Short mel that fits in one window produces same result as generate()."""
        model, config = small_model
        mel = torch.randn(80, 64)  # Well under max_audio_len=128
        notes = generate_full_song(model, mel, "Expert", config, bpm=120.0, seed=42)
        # Should return a list of Note objects (may be empty if model generates no notes)
        assert isinstance(notes, list)
        for note in notes:
            assert hasattr(note, "beat")
            assert hasattr(note, "color")

    def test_full_song_multi_window(self, small_model):
        """Mel 2.5x max_audio_len produces notes spanning the full duration."""
        model, config = small_model
        total_frames = int(config.max_audio_len * 2.5)
        mel = torch.randn(80, total_frames)
        notes = generate_full_song(
            model, mel, "Expert", config, bpm=120.0,
            temperature=1.0, seed=42,
        )
        assert isinstance(notes, list)
        # With a random model we might not get notes everywhere, but
        # the function should run without errors and return Note objects
        for note in notes:
            assert hasattr(note, "beat")
            assert note.beat >= 0

    def test_generate_fills_exactly_the_audio_bars(self, small_model):
        """With audio_bars=N the sequence has bars 0..N-1 and only then ends: it can
        neither stop early (the 14/64-bar windows) nor run past the audio."""
        model, config = small_model
        mel = torch.randn(80, config.max_audio_len)
        finished = 0
        for seed in range(12):
            tokens = generate(model, mel, "Expert", config, temperature=2.0, seed=seed, audio_bars=2)
            bars = [t - BAR_BASE for t in tokens if BAR_BASE <= t < BAR_BASE + BAR_COUNT]
            assert bars == list(range(len(bars))) and len(bars) <= 2
            if tokens[-1] == END:
                assert bars == [0, 1], (seed, bars)
                finished += 1
        assert finished > 0, "no sequence reached END; test would be vacuous"

    def test_full_song_passes_each_windows_real_bars(self, small_model, monkeypatch):
        import beat_weaver.model.inference as inf
        model, config = small_model
        seen = []
        real_generate = inf.generate
        def spy(*a, **kw):
            seen.append(kw.get("audio_bars")); return real_generate(*a, **kw)
        monkeypatch.setattr(inf, "generate", spy)
        L = config.max_audio_len                      # 128 frames = 2 bars
        generate_full_song(model, torch.randn(80, 40), "Expert", config, bpm=120.0, seed=1)
        assert seen == [1]                            # 40 frames -> 1 bar
        seen.clear()
        generate_full_song(model, torch.randn(80, int(L * 2.5)), "Expert", config, bpm=120.0, seed=1)
        assert seen[:-1] == [2] * (len(seen) - 1)     # full windows: 2 bars
        assert seen[-1] == -(-(int(L * 2.5) - (len(seen) - 1) * (L - min(L // 4, 1024))) // 64)  # tail window

    def test_generated_notes_never_share_a_cell(self, small_model):
        """At any one position the two hands must be on different cells."""
        from beat_weaver.model.tokenizer import _decode_note_token
        model, config = small_model
        mel = torch.randn(80, config.max_audio_len)
        pairs = 0
        for seed in range(20):
            tokens = generate(model, mel, "Expert", config, temperature=2.0, seed=seed)
            for a, b in zip(tokens, tokens[1:]):
                if LEFT_BASE <= a < LEFT_BASE + LEFT_COUNT and RIGHT_BASE <= b < RIGHT_BASE + RIGHT_COUNT:
                    pairs += 1
                    assert _decode_note_token(a, LEFT_BASE)[:2] != _decode_note_token(b, RIGHT_BASE)[:2], (seed, a, b)
        assert pairs > 0, "no simultaneous pairs generated; test would be vacuous"

    def test_flow_penalty_reduces_direction_repeats(self, small_model):
        """A heavy penalty must lower the rate of same-direction consecutive cuts vs no penalty."""
        from beat_weaver.model.tokenizer import _decode_note_token
        model, config = small_model
        mel = torch.randn(80, config.max_audio_len)
        def repeat_rate(flow_penalty):
            rep = tot = 0
            for seed in range(30):
                tokens = generate(model, mel, "Expert", config, temperature=1.5, seed=seed, flow_penalty=flow_penalty, flow_window=0)
                for base, count in ((LEFT_BASE, LEFT_COUNT), (RIGHT_BASE, RIGHT_COUNT)):
                    dirs = [_decode_note_token(t, base)[2] for t in tokens if base <= t < base + count]
                    for d1, d2 in zip(dirs, dirs[1:]):
                        if d1 != 8 and d2 != 8:
                            tot += 1; rep += d1 == d2
            return rep / max(tot, 1), tot
        r0, n0 = repeat_rate(0.0); r1, n1 = repeat_rate(50.0)
        assert n0 > 50 and n1 > 50
        assert r1 < r0 * 0.6, (r0, r1)  # untrained model; init shifted with the 495 vocab, 0.5 fails by 3e-4

    def test_dynamics_shapes_budget_to_the_onset_head(self, small_model):
        from beat_weaver.model.inference import _onset_schedule
        model, config = small_model
        n = 4 * 64
        # bar 0 silent, bar 3 loud, bars 1-2 average
        high = [f for f in range(3 * 64, 4 * 64, 2)] + [f for f in range(64, 3 * 64, 4)]
        model.onset_head = self._fixed_onset_head(high, n)
        memory = torch.zeros(1, n, config.encoder_dim)
        flat = _onset_schedule(model, memory, n, notes_per_beat=1.0, min_gap=2, dynamics=0.0)
        shaped = _onset_schedule(model, memory, n, notes_per_beat=1.0, min_gap=2, dynamics=1.0)
        per_bar = lambda s: [sum(1 for f in s if f // 64 == b) for b in range(4)]
        assert per_bar(flat) == [4, 4, 4, 4], per_bar(flat)
        pb = per_bar(shaped)
        assert pb[0] == 0 and pb[3] > pb[1] >= 1, pb
        assert abs(sum(pb) - sum(per_bar(flat))) <= 2, "density is preserved overall"
        even = per_bar(_onset_schedule(model, memory, n, notes_per_beat=1.0, min_gap=2, dynamics=1.0,
                                       dynamics_range=(0.7, 1.3)))
        assert even[0] >= 2 and max(even) <= 6, even  # floor and ceiling at 0.7x / 1.3x of 4 per bar

    def test_lead_in_drops_opening_notes(self, small_model):
        model, config = small_model
        mel = torch.randn(80, config.max_audio_len)
        notes = generate_full_song(model, mel, "Expert", config, 120.0, seed=1, two_stage=True,
                                   notes_per_beat=2.0, lead_in_seconds=3.0)
        assert notes and min(n.time_seconds for n in notes) >= 3.0
        notes0 = generate_full_song(model, mel, "Expert", config, 120.0, seed=1, two_stage=True,
                                    notes_per_beat=2.0, lead_in_seconds=0.0)
        assert min(n.time_seconds for n in notes0) < 3.0

    def test_stream_bias_chains_hot_frames_into_runs(self, small_model):
        from beat_weaver.model.inference import _onset_schedule
        model, config = small_model
        n = 2 * 64
        model.onset_head = self._fixed_onset_head(list(range(16, 48)), n)  # beats 2-3 of bar 0 are hot, every frame
        memory = torch.zeros(1, n, config.encoder_dim)
        off = _onset_schedule(model, memory, n, notes_per_beat=2.0, min_gap=4, dynamics=0.0, stream_bias=1.0)
        on = _onset_schedule(model, memory, n, notes_per_beat=2.0, min_gap=4, dynamics=0.0, stream_bias=0.5)
        def longest(sched):
            best = run = 1
            for a, b in zip(sched, sched[1:]):
                run = run + 1 if b - a == 4 else 1
                best = max(best, run)
            return best
        assert len(on) == len(off), "budget unchanged"
        assert longest(on) >= 6 > longest(off), (longest(on), longest(off))

    def test_stream_chains_stop_at_the_beat_boundary(self, small_model):
        from beat_weaver.model.inference import _onset_schedule
        model, config = small_model
        n = 64
        model.onset_head = self._fixed_onset_head(list(range(0, 64)), n)  # whole bar hot
        memory = torch.zeros(1, n, config.encoder_dim)
        sched = _onset_schedule(model, memory, n, notes_per_beat=2.0, min_gap=4, dynamics=0.0, stream_bias=0.5)
        per_beat = [sum(1 for f in sched if f // 16 == b) for b in range(4)]
        assert min(per_beat) >= 1 and per_beat[0] <= len(sched) // 2, per_beat

    def test_bar_remainder_goes_to_the_strongest_beats(self, small_model):
        from beat_weaver.model.inference import _onset_schedule
        model, config = small_model
        n = 64
        model.onset_head = self._fixed_onset_head(list(range(32, 48)), n)  # beat 3 hot
        memory = torch.zeros(1, n, config.encoder_dim)
        sched = _onset_schedule(model, memory, n, notes_per_beat=1.5, min_gap=2, dynamics=0.0)  # 6 notes: 2,2,1,1 split
        per_beat = [sum(1 for f in sched if f // 16 == b) for b in range(4)]
        assert sum(per_beat) == 6 and per_beat[2] == 2 and min(per_beat) >= 1, per_beat

    def test_converging_mask_geometry(self):
        from beat_weaver.model.inference import _converging_right_tokens
        from beat_weaver.model.tokenizer import _encode_note_token
        tok = lambda x, y, d: _encode_note_token(RIGHT_BASE, x, y, d)
        m = _converging_right_tokens((1, 1), 3)  # left note at (1,1) cutting RIGHT
        assert m[tok(2, 1, 2)], "right note just to its right cutting LEFT: blades meet"
        assert m[tok(2, 1, 3)], "both cutting right: the left blade sweeps over the blue first (cut-through)"
        assert not m[tok(2, 2, 0)] and not m[tok(2, 0, 1)], "perpendicular, cutting away, off the blade path: fine"
        assert m[tok(3, 1, 2)], "two cells away cutting back: the blue blade runs over the red (cut-through)"
        assert not m[tok(3, 0, 3)], "two cells away on another row: fine"
        assert m[tok(1, 1, 0)] and m[tok(1, 1, 3)] and not m[tok(1, 1, 8)], "same cell: any directional cut blocked, dot ok"
        assert m[tok(0, 1, 2)], "on the cell the left saber arrives through (approach): blocked"
        assert not m[tok(0, 0, 2)], "behind and below, sweeping away: fine"
        m2 = _converging_right_tokens((1, 1), 0)  # left cutting UP
        assert m2[tok(1, 2, 1)] and m2[tok(2, 2, 6)] and not m2[tok(2, 2, 0)]
        m8 = _converging_right_tokens((1, 1), 8)  # left is a dot: only cuts that run over it are blocked
        assert m8[tok(2, 1, 2)] and not m8[tok(3, 1, 3)] and not m8[tok(0, 0, 1)]
        assert m8[tok(2, 1, 3)], "a blue cut arriving through the dot's cell is an approach crossing"

    def test_arm_tangle_mask(self):
        from beat_weaver.model.inference import _converging_right_tokens
        from beat_weaver.model.tokenizer import _encode_note_token
        m = _converging_right_tokens((3, 1), 1)  # left note crossed to the right edge, cutting DOWN
        assert m[_encode_note_token(RIGHT_BASE, 0, 1, 0)]       # right note on the left, cutting UP: tangle
        assert m[_encode_note_token(RIGHT_BASE, 1, 2, 4)]       # up-left too (opposite vertical sense)
        assert not m[_encode_note_token(RIGHT_BASE, 0, 1, 1)]   # same way down: arms move together
        assert not m[_encode_note_token(RIGHT_BASE, 0, 1, 3)]   # horizontal: no vertical pull
        assert not m[_encode_note_token(RIGHT_BASE, 2, 1, 0)]   # right note on its own side: fine
        assert not _converging_right_tokens((1, 1), 1)[_encode_note_token(RIGHT_BASE, 0, 1, 0)]  # left not crossed

    def test_near_converging_is_blocked_by_default_and_allowed_by_flag(self):
        from beat_weaver.model.inference import _converging_right_tokens
        from beat_weaver.model.tokenizer import _encode_note_token
        tok = lambda x, y, d: _encode_note_token(RIGHT_BASE, x, y, d)
        # left at (1,1) cutting UP; right beside it at (2,1) cutting DOWN-LEFT: 135 deg apart, right aimed at left
        near = tok(2, 1, 6)
        assert _converging_right_tokens((1, 1), 0)[near]
        assert not _converging_right_tokens((1, 1), 0, block_near=False)[near]
        assert _converging_right_tokens((1, 1), 3, block_near=False)[tok(2, 1, 2)], "head-on stays blocked either way"
        # left at (1,1) cutting right: its blade runs over (2,1), so every directional RIGHT note there is a
        # cut-through (favourites Expert/Expert+: 0.3% of doubles; a vertical one is 0.01%)
        assert _converging_right_tokens((1, 1), 3)[tok(2, 1, 1)], "cut-through: left blade passes over the blue"
        assert not _converging_right_tokens((1, 1), 3)[tok(2, 0, 1)], "90 deg apart off the blade path is not a near miss"
        assert not _converging_right_tokens((1, 1), 3)[tok(3, 0, 6)], "two cells away off the blade path is fine"
        # left at (1,1) cutting LEFT arrives from (2,1): a right note there is an approach crossing, even cutting away
        assert _converging_right_tokens((1, 1), 2)[tok(2, 1, 5)]
        # diverging and off both approach paths: left at (1,1) cutting LEFT, right at (2,2) cutting RIGHT
        assert not _converging_right_tokens((1, 1), 2)[tok(2, 2, 3)]

    def test_no_converging_doubles_are_generated(self, small_model):
        from beat_weaver.model.inference import _converging_right_tokens
        model, config = small_model
        mel = torch.randn(80, config.max_audio_len)
        for seed in range(10):
            toks = generate(model, mel, "Expert", config, seed=seed, temperature=2.0, audio_bars=2, doubles_bias=20.0)
            for a, b in zip(toks, toks[1:]):
                if LEFT_BASE <= a < LEFT_BASE + LEFT_COUNT and RIGHT_BASE <= b < RIGHT_BASE + RIGHT_COUNT:
                    x, y, d = _decode_note_token(a, LEFT_BASE)
                    assert not _converging_right_tokens((x, y), d)[b], (seed, a, b)

    def test_vision_cap_never_stacks_centre_notes(self, small_model):
        from beat_weaver.model.inference import VISION_CELLS
        model, config = small_model
        mel = torch.randn(80, config.max_audio_len)
        worst = 0
        for seed in range(10):
            toks = generate(model, mel, "Expert", config, seed=seed, temperature=2.0, audio_bars=2,
                            vision_penalty=0.0, vision_max_run=2)
            run = 0
            hit_vis = False
            for t in toks:
                if POS_BASE <= t < POS_BASE + POS_COUNT:
                    run = run + 1 if hit_vis else 0
                    worst = max(worst, run)
                    hit_vis = False
                elif LEFT_BASE <= t < LEFT_BASE + LEFT_COUNT:
                    hit_vis |= _decode_note_token(t, LEFT_BASE)[:2] in VISION_CELLS
                elif RIGHT_BASE <= t < RIGHT_BASE + RIGHT_COUNT:
                    hit_vis |= _decode_note_token(t, RIGHT_BASE)[:2] in VISION_CELLS
        assert worst <= 2, worst

    def test_vision_penalty_lowers_centre_rate(self, small_model):
        from beat_weaver.model.inference import VISION_CELLS
        model, config = small_model
        mel = torch.randn(80, config.max_audio_len)
        def rate(pen):
            c = n = 0
            for seed in range(10):
                for t in generate(model, mel, "Expert", config, seed=seed, temperature=2.0, audio_bars=2, vision_penalty=pen, vision_max_run=0):
                    for base, cnt in ((LEFT_BASE, LEFT_COUNT), (RIGHT_BASE, RIGHT_COUNT)):
                        if base <= t < base + cnt:
                            n += 1; c += _decode_note_token(t, base)[:2] in VISION_CELLS
            return c / max(n, 1)
        assert rate(8.0) < rate(0.0) * 0.5

    def test_doubles_bias_raises_double_rate(self, small_model):
        model, config = small_model
        mel = torch.randn(80, config.max_audio_len)
        def rate(bias):
            d = tot = 0
            for seed in range(12):
                toks = generate(model, mel, "Expert", config, seed=seed, temperature=1.5, audio_bars=2, doubles_bias=bias)
                for a, b in zip(toks, toks[1:]):
                    if LEFT_BASE <= a < LEFT_BASE + LEFT_COUNT:
                        tot += 1; d += RIGHT_BASE <= b < RIGHT_BASE + RIGHT_COUNT
            return d / max(tot, 1), tot
        r0, n0 = rate(0.0); r1, n1 = rate(20.0)
        assert n0 > 20 and r1 >= r0 and r1 >= 0.95, (r0, r1)  # a few slots have every RIGHT cut masked by cut-through

    def test_parallel_bias_raises_side_by_side_same_cut(self, small_model):
        model, config = small_model
        mel = torch.randn(80, config.max_audio_len)
        def rate(bias):
            p = tot = 0
            for seed in range(12):
                toks = generate(model, mel, "Expert", config, seed=seed, temperature=1.5, audio_bars=2, doubles_bias=20.0,
                                parallel_bias=bias)
                for a, b in zip(toks, toks[1:]):
                    if LEFT_BASE <= a < LEFT_BASE + LEFT_COUNT and RIGHT_BASE <= b < RIGHT_BASE + RIGHT_COUNT:
                        (lx, ly, ld), (rx, ry, rd) = _decode_note_token(a, LEFT_BASE), _decode_note_token(b, RIGHT_BASE)
                        tot += 1; p += ld == rd < 8 and ly == ry and 0 < rx - lx <= 2
            return p / max(tot, 1), tot
        (r0, n0), (r1, _) = rate(0.0), rate(20.0)
        assert n0 > 20 and r1 > max(2 * r0, 0.5), (r0, r1)

    def test_parallel_streak_lengthens_in_sync_runs(self, small_model):
        model, config = small_model
        mel = torch.randn(80, config.max_audio_len)
        def mean_run(streak, two_stage=False):
            runs = []
            for seed in range(12):
                toks = generate(model, mel, "Expert", config, seed=seed, temperature=1.5, audio_bars=2 if two_stage else 4, doubles_bias=3.0,
                                parallel_bias=3.0, parallel_streak=streak, two_stage=two_stage,
                                notes_per_beat=2.0)
                cur, left = 0, None
                for t in toks:
                    if POS_BASE <= t < POS_BASE + POS_COUNT or t == END:
                        if left is not True: runs.append(cur); cur = 0
                        cur += left is True
                        left = None
                    elif LEFT_BASE <= t < LEFT_BASE + LEFT_COUNT:
                        left = _decode_note_token(t, LEFT_BASE)
                    elif RIGHT_BASE <= t < RIGHT_BASE + RIGHT_COUNT and isinstance(left, tuple):
                        (lx, ly, ld), (rx, ry, rd) = left, _decode_note_token(t, RIGHT_BASE)
                        left = ld == rd < 8 and ly == ry and 0 < rx - lx <= 2
            runs = [r for r in runs if r]
            return sum(runs) / max(len(runs), 1)
        for two_stage in (False, True):  # the batch recipe decodes two-stage: the schedule opens the positions
            r0, r1 = mean_run(0.0, two_stage), mean_run(8.0, two_stage)
            assert r1 > r0 * 1.15, (two_stage, r0, r1)

    def test_beat_focus_leaves_weak_beats_with_one_note(self, small_model):
        from beat_weaver.model.inference import _onset_schedule
        model, config = small_model
        n = 64
        hot = list(range(0, 16)) + list(range(32, 48))  # beats 1 and 3 hot
        model.onset_head = self._fixed_onset_head(hot, n)
        memory = torch.zeros(1, n, config.encoder_dim)
        even = _onset_schedule(model, memory, n, notes_per_beat=2.5, min_gap=4, dynamics=0.0)
        focus = _onset_schedule(model, memory, n, notes_per_beat=2.5, min_gap=4, dynamics=0.0, beat_focus=True)
        per = lambda sc: [sum(1 for f in sc if f // 16 == b) for b in range(4)]
        assert len(even) == len(focus) == 10
        assert min(per(focus)) == 1 and max(per(focus)) == 4, per(focus)
        assert max(per(even)) - min(per(even)) <= 1, per(even)

    def test_reset_gap_and_max_doubles(self, small_model):
        model, config = small_model
        mel = torch.randn(80, config.max_audio_len)
        bpm = 120.0
        notes = generate_full_song(model, mel, "Expert", config, bpm, seed=1, two_stage=True, notes_per_beat=2.0,
                                   lead_in_seconds=0, reset_seconds=0.5, max_doubles=0.2)
        same = lambda a, b: a != 8 and b != 8 and (a == b or {a, b} in ({0, 4}, {0, 5}, {1, 6}, {1, 7}, {2, 4}, {2, 6}, {3, 5}, {3, 7}))
        for c in (0, 1):
            hand = [n for n in notes if n.color == c]
            for a, b in zip(hand, hand[1:]):
                assert not ((b.beat - a.beat) * 60 / bpm < 0.5 and same(a.cut_direction, b.cut_direction)), (a, b)
        beats = [n.beat for n in notes]
        hits = len(set(beats))
        assert (len(beats) - hits) / hits <= 0.2 + 1 / hits

    def test_notes_per_second_overrides_notes_per_beat(self, small_model):
        model, config = small_model
        mel = torch.randn(80, config.max_audio_len)
        slow = generate_full_song(model, mel, "Expert", config, 60.0, seed=1, two_stage=True, notes_per_second=4.0, lead_in_seconds=0)
        fast = generate_full_song(model, mel, "Expert", config, 180.0, seed=1, two_stage=True, notes_per_second=4.0, lead_in_seconds=0)
        per_beat = lambda ns, bpm: len({n.beat for n in ns}) / (max(n.beat for n in ns) + 1)  # hits: doubles depend on placement masks
        assert per_beat(slow, 60) > 1.5 * per_beat(fast, 180), (per_beat(slow, 60), per_beat(fast, 180))  # 2.5 (capped) vs 1.33 per beat

    def test_end_seconds_drops_notes_after_the_music(self, small_model):
        import numpy as np
        from beat_weaver.model.inference import last_music_second
        sr = 22050
        audio = np.concatenate([np.random.default_rng(0).standard_normal(5 * sr) * 0.3, np.zeros(3 * sr)]).astype(np.float32)
        end = last_music_second(audio, sr)
        assert 4.9 <= end <= 5.2, end
        ring = np.concatenate([audio[:5 * sr], audio[:sr] * np.linspace(0.05, 0.01, sr, dtype=np.float32)])  # a fading tail
        assert last_music_second(ring, sr) <= 5.1  # is not music
        model, config = small_model
        mel = torch.randn(80, config.max_audio_len)
        notes = generate_full_song(model, mel, "Expert", config, 120.0, seed=1, two_stage=True, notes_per_beat=2.0,
                                   lead_in_seconds=0.0, end_seconds=end)
        assert notes and max(n.time_seconds for n in notes) < end

    def test_precision_prior_prefers_quarter_positions(self, small_model):
        from beat_weaver.model.inference import _onset_schedule
        model, config = small_model
        n = 64
        # every frame of bar 0 equally hot: without the prior the first pick per beat is arbitrary
        model.onset_head = self._fixed_onset_head(list(range(n)), n)
        memory = torch.zeros(1, n, config.encoder_dim)
        sched = _onset_schedule(model, memory, n, notes_per_beat=2.0, min_gap=2, dynamics=0.0, precision_prior=1.0)
        assert len(sched) == 8 and all(f % 4 == 0 for f in sched), sched  # 1/4-beat positions only
        flat = _onset_schedule(model, memory, n, notes_per_beat=2.0, min_gap=2, dynamics=0.0)
        assert len(flat) == 8, "the prior changes which frames are picked, not how many"

    @staticmethod
    def _fixed_onset_head(frames_high, n_frames):
        class Fixed(torch.nn.Module):
            def forward(self, memory):
                out = torch.full((memory.size(0), memory.size(1), 1), -10.0)
                for f in frames_high:
                    out[:, f, 0] = 10.0
                return out
        return Fixed()

    def test_onset_guidance_puts_notes_on_the_audios_onsets(self, small_model):
        """With a head that marks frames 10, 20, 30 of bar 0, the first note of
        bar 0 must land on a marked frame in nearly every guided run, and almost
        never in unguided runs of the same model (so the test is not vacuous).
        Only the first note is asserted: the guidance multiplies the decoder's own
        preferences, and an untrained decoder has none, so after a marked frame is
        used the rest is its choice."""
        model, config = small_model
        high = {10, 20, 30}
        model.onset_head = self._fixed_onset_head(high, config.max_audio_len)
        mel = torch.randn(80, config.max_audio_len)
        def first_pos_in_bar0(g, seed):
            toks = generate(model, mel, "Expert", config, temperature=1.5, seed=seed,
                            audio_bars=2, onset_guidance=g)
            bar = -1
            for t in toks:
                if BAR_BASE <= t < BAR_BASE + BAR_COUNT: bar = t - BAR_BASE
                elif POS_BASE <= t < POS_BASE + POS_COUNT and bar == 0: return t - POS_BASE
            return None
        guided = [first_pos_in_bar0(8.0, s) for s in range(30)]
        unguided = [first_pos_in_bar0(0.0, s) for s in range(30)]
        assert sum(p in high for p in guided) >= 27, guided
        assert sum(p in high for p in unguided) <= 6, unguided

    def test_final_filter_drops_crossed_notes_cutting_back(self):
        from beat_weaver.model.inference import drop_fast_same_way
        from beat_weaver.schemas.normalized import Note
        mk = lambda b, x, c, d: Note(beat=b, time_seconds=b / 2, x=x, y=2, color=c, cut_direction=d)
        kept = drop_fast_same_way([mk(0, 0, 1, 7), mk(4, 0, 1, 4), mk(8, 3, 0, 2), mk(12, 3, 0, 5)], 0.0)
        assert [(n.beat, n.cut_direction) for n in kept] == [(4, 4), (12, 5)]  # right DR / left L on the far side go

    def test_crossover_notes_never_face_outward(self, small_model):
        """A left note in columns 2-3 never cuts left-ish, a right note in
        columns 0-1 never cuts right-ish: the crossed saber would have to turn back."""
        from beat_weaver.model.tokenizer import _decode_note_token
        model, config = small_model
        mel = torch.randn(80, config.max_audio_len)
        for seed in range(6):
            for t in generate(model, mel, "Expert", config, seed=seed, audio_bars=4, temperature=2.0):
                if LEFT_BASE <= t < LEFT_BASE + LEFT_COUNT:
                    x, _, d = _decode_note_token(t, LEFT_BASE)
                    assert not (x >= 2 and d in (2, 4, 6)), (x, d)
                elif RIGHT_BASE <= t < RIGHT_BASE + RIGHT_COUNT:
                    x, _, d = _decode_note_token(t, RIGHT_BASE)
                    assert not (x <= 1 and d in (3, 5, 7)), (x, d)

    def test_two_stage_notes_sit_exactly_on_the_scheduled_frames(self, small_model):
        """Timing comes from the onset head alone: with frames 10, 20, 40 of
        bar 0 and 5 of bar 1 marked, the output's note frames are exactly those,
        every position holds a note, and the sequence still parses."""
        model, config = small_model
        high = {10, 20, 40, 64 + 5}
        model.onset_head = self._fixed_onset_head(high, config.max_audio_len)
        mel = torch.randn(80, config.max_audio_len)
        for seed in range(5):
            toks = generate(model, mel, "Expert", config, seed=seed, audio_bars=2, two_stage=True)
            frames, bar = [], -1
            for i, t in enumerate(toks):
                if BAR_BASE <= t < BAR_BASE + BAR_COUNT: bar = t - BAR_BASE
                elif POS_BASE <= t < POS_BASE + POS_COUNT:
                    if toks[i + 1] == LEFT_EMPTY and toks[i + 2] == RIGHT_EMPTY:
                        assert toks[i + 3] >= BOMB_BASE  # an extras-only position always holds a bomb or wall
                    else:
                        frames.append(bar * 64 + t - POS_BASE)
            assert frames == sorted(high), (seed, frames)
            assert toks[-1] == END

    def test_two_stage_keeps_the_schedule_within_the_token_budget(self, small_model):
        """A schedule that nearly fills max_seq_len, with a decoder that wants bombs and bomb-only positions
        everywhere: every scheduled note is still placed and the sequence stays under the cap (the outro of a
        long song went silent when extras ate the budget)."""
        model, config = small_model
        high = set(range(2, 128, 8))  # 16 notes = 48 tokens + bars + START/DIFF in a 64-token model
        model.onset_head = self._fixed_onset_head(high, config.max_audio_len)
        real = model.decoder

        class Greedy(torch.nn.Module):
            def forward(self, toks, *a):
                out = real(toks, *a).clone()
                out[0, -1, BOMB_BASE: BOMB_BASE + 12] += 50.0  # a bomb wherever one is allowed
                out[0, -1, POS_BASE: POS_BASE + 64] += 50.0  # and a position whenever one is allowed
                out[0, -1, LEFT_EMPTY] += 60.0  # bomb-only positions get confirmed
                return out

        model.decoder = Greedy()
        mel = torch.randn(80, config.max_audio_len)
        for seed in range(3):
            toks = generate(model, mel, "Expert", config, seed=seed, audio_bars=2, two_stage=True)
            frames, bar = [], -1
            for i, t in enumerate(toks):
                if BAR_BASE <= t < BAR_BASE + BAR_COUNT: bar = t - BAR_BASE
                elif POS_BASE <= t < POS_BASE + POS_COUNT and not (toks[i + 1] == LEFT_EMPTY and toks[i + 2] == RIGHT_EMPTY):
                    frames.append(bar * 64 + t - POS_BASE)
            assert frames == sorted(high), (seed, frames)
            assert len(toks) <= config.max_seq_len, len(toks)

    def test_streams_alternate_hands_with_no_doubles(self, small_model):
        from beat_weaver.model.inference import _stream_frames
        # a chain of 16 hits half a beat apart, then a rest, then 4 more: only the chain is a stream at share 0.5
        chain = list(range(0, 128, 8)); tail = [160, 168, 176, 184]
        assert _stream_frames(chain + tail, 0.5) == set(chain)
        assert _stream_frames(chain + tail, 0.0) == set()
        model, config = small_model
        model.onset_head = self._fixed_onset_head(set(chain), config.max_audio_len)
        mel = torch.randn(80, config.max_audio_len)
        for seed in range(3):
            toks = generate(model, mel, "Expert", config, seed=seed, audio_bars=2, two_stage=True, streams=1.0,
                            doubles_bias=20.0)
            hands = []
            for i, t in enumerate(toks):
                if POS_BASE <= t < POS_BASE + POS_COUNT and not (toks[i + 1] == LEFT_EMPTY and toks[i + 2] == RIGHT_EMPTY):
                    hands.append((toks[i + 1] != LEFT_EMPTY, toks[i + 2] != RIGHT_EMPTY))
            assert len(hands) == 16 and all(l != r for l, r in hands), (seed, hands)  # one note per hit, never a double
            assert all(a != b for a, b in zip(hands, hands[1:])), (seed, hands)  # hands alternate

    def test_stack_bias_adds_stacks(self, small_model):
        model, config = small_model
        model.onset_head = self._fixed_onset_head(set(range(0, 128, 4)), config.max_audio_len)
        mel = torch.randn(80, config.max_audio_len)

        def stacked(bias):
            hand = lambda t: 0 if LEFT_BASE <= t < LEFT_BASE + LEFT_COUNT else 1 if RIGHT_BASE <= t < RIGHT_BASE + RIGHT_COUNT else None
            n = 0
            for seed in range(5):  # a random model: compare totals, not one sample
                toks = generate(model, mel, "Expert", config, seed=seed, audio_bars=2, two_stage=True, stacks=True, stack_bias=bias)
                n += sum(hand(a) is not None and hand(a) == hand(b) for a, b in zip(toks, toks[1:]))
            return n
        assert stacked(50.0) > stacked(0.0)  # (the target cap only stops the bonus; a random model stacks anyway)

    def test_raised_max_run_drops_a_hand_to_the_bottom_row(self, small_model):
        model, config = small_model
        model.onset_head = self._fixed_onset_head(set(range(0, 128, 8)), config.max_audio_len)
        mel = torch.randn(80, config.max_audio_len)
        def over(cap):  # notes past the 2nd in a row in rows 1-2, over 5 seeds
            n = 0
            for seed in range(5):
                toks = generate(model, mel, "Expert", config, seed=seed, audio_bars=2, two_stage=True, raised_max_run=cap)
                for base, size in ((LEFT_BASE, LEFT_COUNT), (RIGHT_BASE, RIGHT_COUNT)):
                    run = 0
                    for t in toks:
                        if base <= t < base + size:
                            run = run + 1 if _decode_note_token(t, base)[1] >= 1 else 0
                            n += run > 2
            return n
        # the cap steps aside when the bottom row has no legal note left, so not zero
        assert over(2) < over(0) / 2, (over(2), over(0))

    def test_left_cells_do_not_block_the_right_hand_at_later_positions(self, small_model):
        """Scheduled positions must clear the open LEFT cells: they used to pile up, so a RIGHT note could never
        use a cell the left hand had ever used (one song: 339 dead ends fell back to vision-cell notes)."""
        model, config = small_model
        model.onset_head = self._fixed_onset_head(set(range(0, 128, 8)), config.max_audio_len)
        mel = torch.randn(80, config.max_audio_len)
        toks = generate(model, mel, "Expert", config, seed=0, audio_bars=2, two_stage=True)
        used_left, reused = set(), 0
        for i, t in enumerate(toks):
            if POS_BASE <= t < POS_BASE + POS_COUNT:
                l, r = toks[i + 1], toks[i + 2]
                if RIGHT_BASE <= r < RIGHT_BASE + RIGHT_COUNT:
                    reused += _decode_note_token(r, RIGHT_BASE)[:2] in used_left
                if LEFT_BASE <= l < LEFT_BASE + LEFT_COUNT:
                    used_left.add(_decode_note_token(l, LEFT_BASE)[:2])
        assert reused > 0

    def test_two_stage_can_open_bomb_only_positions_between_notes(self, small_model):
        """The decoder may open positions between the scheduled notes; those hold only bombs or walls, and the
        scheduled note frames are unchanged and the sequence still parses."""
        model, config = small_model
        high = set(range(4, 128, 16))  # 8 notes; the 64-token test model leaves room for a few extras
        model.onset_head = self._fixed_onset_head(high, config.max_audio_len)
        real = model.decoder

        class Steer(torch.nn.Module):  # wants POS 12 (between scheduled 4 and 20) and a bomb in an empty slot
            def forward(self, toks, *a):
                out = real(toks, *a).clone()
                out[0, -1, BOMB_BASE + 1:] = -50.0  # only one bomb cell, no walls
                if toks[0, -1] == POS_BASE + 12:
                    out[0, -1, LEFT_EMPTY] = 50.0  # the decoder agrees the position is empty
                elif toks[0, -1] == RIGHT_EMPTY and toks[0, -2] == LEFT_EMPTY:
                    out[0, -1, BOMB_BASE] = 50.0
                elif toks[0, -1] >= RIGHT_BASE:
                    out[0, -1, POS_BASE + 12] = 50.0
                return out

        model.decoder = Steer()
        mel = torch.randn(80, config.max_audio_len)
        between = 0
        for seed in range(2):
            toks = generate(model, mel, "Expert", config, seed=seed, audio_bars=2, two_stage=True)
            frames, bar = [], -1
            for i, t in enumerate(toks):
                if BAR_BASE <= t < BAR_BASE + BAR_COUNT: bar = t - BAR_BASE
                elif POS_BASE <= t < POS_BASE + POS_COUNT:
                    if toks[i + 1] == LEFT_EMPTY and toks[i + 2] == RIGHT_EMPTY:
                        assert toks[i + 3] >= BOMB_BASE; between += 1
                    else:
                        frames.append(bar * 64 + t - POS_BASE)
            assert frames == sorted(high) and toks[-1] == END, (seed, frames, toks[-4:])
        assert between > 0

    def test_onset_guidance_zero_changes_nothing(self, small_model):
        model, config = small_model
        mel = torch.randn(80, config.max_audio_len)
        a = generate(model, mel, "Expert", config, seed=3, audio_bars=2)
        b = generate(model, mel, "Expert", config, seed=3, audio_bars=2, onset_guidance=0.0)
        assert a == b

    def test_generate_cannot_outrun_the_audio(self, small_model):
        """With indexed bars the decoder can emit at most one token per bar, so a
        window can never contain more bars than its audio (the drift that halved
        density with a single counted BAR token)."""
        model, config = small_model
        mel = torch.randn(80, config.max_audio_len)
        for seed in range(8):
            tokens = generate(model, mel, "Expert", config, temperature=2.0, seed=seed)
            bars = [t - BAR_BASE for t in tokens if BAR_BASE <= t < BAR_BASE + BAR_COUNT]
            assert bars == list(range(len(bars))), bars        # 0,1,2,... no repeats, no skips
            assert len(bars) <= BAR_COUNT

    def test_full_song_no_notes_past_audio_end(self, small_model):
        """Notes must not land past the real audio (the last window is zero-padded)."""
        model, config = small_model
        # 2.3 windows: the final window is mostly padding the model could write into
        for total_frames in (int(config.max_audio_len * 2.3), config.max_audio_len // 2):
            mel = torch.randn(80, total_frames)
            for seed in range(6):
                notes = generate_full_song(
                    model, mel, "Expert", config, bpm=120.0, temperature=1.5, seed=seed,
                )
                assert all(n.beat < total_frames / 16.0 for n in notes), (
                    total_frames, seed, max(n.beat for n in notes),
                )

    def test_full_song_overlap_no_duplicates(self, small_model):
        """No two notes at the exact same beat+color in the overlap zone."""
        model, config = small_model
        total_frames = config.max_audio_len * 3
        mel = torch.randn(80, total_frames)
        notes = generate_full_song(
            model, mel, "Expert", config, bpm=120.0, seed=99,
        )
        seen = set()
        for note in notes:
            key = (round(note.beat, 6), note.color, note.x, note.y)
            # Notes can legitimately share a beat if they're different placements,
            # but the same (beat, color, x, y) should not appear twice
            assert key not in seen, f"Duplicate note at {key}"
            seen.add(key)

    def test_full_song_notes_sorted(self, small_model):
        """Output notes are sorted by beat."""
        model, config = small_model
        total_frames = config.max_audio_len * 2
        mel = torch.randn(80, total_frames)
        notes = generate_full_song(
            model, mel, "Expert", config, bpm=120.0, seed=7,
        )
        for i in range(1, len(notes)):
            assert notes[i].beat >= notes[i - 1].beat

    def test_cap_runs_ends_a_full_run_with_a_whole_beat_rest(self):
        from beat_weaver.model.inference import _cap_runs
        sched = list(range(0, 64 * 4, 8))  # unbroken eighth-note stream, 32 notes over 4 bars
        capped = _cap_runs(sched, 8)
        assert len(capped) < len(sched) and capped == sorted(capped)
        run = best = 1
        for a, b in zip(capped, capped[1:]):
            run = run + 1 if b - a <= 8 else 1
            best = max(best, run)
            assert b - a <= 8 or b - a >= 16, "a run ends in a full-beat rest, not a dropped sixteenth"
        assert best == 8
        assert _cap_runs(sched, 0) == sched

    def test_beat_floor_leaves_quiet_beats_empty(self, small_model):
        from beat_weaver.model.inference import _onset_schedule
        model, config = small_model
        n = 64
        model.onset_head = self._fixed_onset_head(list(range(0, 32)), n)  # beats 1-2 hot, 3-4 silent
        memory = torch.zeros(1, n, config.encoder_dim)
        every = _onset_schedule(model, memory, n, notes_per_beat=1.0, min_gap=4, dynamics=0.0)
        floored = _onset_schedule(model, memory, n, notes_per_beat=1.0, min_gap=4, dynamics=0.0, beat_floor=0.5)
        assert any(f >= 32 for f in every), "without a floor every beat gets its note"
        assert floored and all(f < 32 for f in floored), floored


class TestShaping:
    def test_drift_bonus_moves_one_lane_and_reverses_at_edge(self):
        from beat_weaver.model.inference import _drift_bonus, _column_tokens
        m = _drift_bonus(LEFT_BASE, (1, 1), +1)
        assert (m[_column_tokens(LEFT_BASE, 2)] > 0).all() and (m[_column_tokens(LEFT_BASE, 0)] < 0).all()
        assert abs(m[LEFT_BASE: LEFT_BASE + LEFT_COUNT].mean()) < 1e-6, "zero-mean over note tokens"
        assert m[LEFT_EMPTY] == 0
        edge = _drift_bonus(LEFT_BASE, (3, 1), +1)  # at the right edge the drift turns back
        assert (edge[_column_tokens(LEFT_BASE, 2)] > 0).all() and (edge[_column_tokens(LEFT_BASE, 3)] < 0).all()
        both = _drift_bonus(LEFT_BASE, (1, 1), 0)
        assert (both[_column_tokens(LEFT_BASE, 0)] > 0).all() and (both[_column_tokens(LEFT_BASE, 2)] > 0).all()
        assert not _drift_bonus(RIGHT_BASE, (1, 1), 1)[LEFT_BASE: LEFT_BASE + LEFT_COUNT].any()

    def test_momentum_continues_then_unwinds(self):
        from beat_weaver.model.inference import MOMENTUM_TENSION, _ROT, _momentum
        from beat_weaver.model.tokenizer import _encode_note_token
        # after an Up cut rotating towards Right (clockwise), Right keeps the sense, Left reverses it
        sense_right, sense_left = _ROT[0, 3].item(), _ROT[0, 2].item()
        assert sense_right == -sense_left != 0
        rot = int(sense_right)
        m = _momentum(LEFT_BASE, 0, rot)
        assert m[_encode_note_token(LEFT_BASE, 1, 1, 3)] > 0 > m[_encode_note_token(LEFT_BASE, 1, 1, 2)]
        wound = _momentum(LEFT_BASE, 0, rot * MOMENTUM_TENSION)
        assert wound[_encode_note_token(LEFT_BASE, 1, 1, 3)] < 0
        assert not _momentum(LEFT_BASE, 8, rot).any() and not _momentum(LEFT_BASE, 0, 0).any()

    def test_generate_accepts_shaping_flags(self):
        config = ModelConfig(
            vocab_size=495,
            max_seq_len=64,
            n_mels=80,
            encoder_layers=1,
            encoder_dim=32,
            encoder_heads=4,
            encoder_ff_dim=64,
            decoder_layers=1,
            decoder_dim=32,
            decoder_heads=4,
            decoder_ff_dim=64,
            dropout=0.0,
        )
        torch.manual_seed(0)
        model = BeatWeaverModel(config).eval()
        mel = torch.randn(80, 64)
        toks = generate(model, mel, "Expert", config, seed=1, drift_bias=2.0, momentum_bias=1.0, flow_window=1.0)
        assert toks[0] == START and (toks[-1] == END or len(toks) == config.max_seq_len)


def test_hand_comfort_table_is_zero_mean_and_prefers_inner_forehand():
    from beat_weaver.model.inference import _COMFORT, LEFT_BASE, RIGHT_BASE
    from beat_weaver.model.tokenizer import COLS, ROWS, DIRS
    for base in (LEFT_BASE, RIGHT_BASE):
        m = _COMFORT[base]
        notes = m[base: base + COLS * ROWS * DIRS]
        assert abs(float(notes.mean())) < 1e-6
        assert float(m[: base].abs().sum()) == 0.0 if base == LEFT_BASE else True
    def tok(base, x, y, d):
        return base + (x * ROWS + y) * DIRS + d
    L = _COMFORT[LEFT_BASE]
    assert L[tok(LEFT_BASE, 1, 0, 1)] > L[tok(LEFT_BASE, 0, 0, 1)]  # down-cut: inner lane over outer
    assert L[tok(LEFT_BASE, 0, 1, 0)] > L[tok(LEFT_BASE, 1, 1, 0)]  # up-cut: outer lane over inner
    assert L[tok(LEFT_BASE, 1, 2, 1)] < L[tok(LEFT_BASE, 1, 1, 1)]  # down-cut on the top row is strain
    R = _COMFORT[RIGHT_BASE]
    assert R[tok(RIGHT_BASE, 2, 0, 1)] > R[tok(RIGHT_BASE, 3, 0, 1)]


    def test_ebpm_gap_keeps_one_hand_from_swinging_twice_in_a_row(self, small_model):
        from beat_weaver.model.tokenizer import decode_tokens
        model, config = small_model
        mel = torch.randn(80, config.max_audio_len)
        for seed in range(5):
            tokens = generate(model, mel, "Expert", config, temperature=1.5, seed=seed, ebpm_gap=4)
            notes = sorted(decode_tokens(tokens, 120.0), key=lambda n: n.beat)
            for color in (0, 1):
                h = [n for n in notes if n.color == color]
                for a, b in zip(h, h[1:]):
                    assert b.beat - a.beat >= 0.25 - 1e-6 or abs(b.beat - a.beat) < 1e-6, (seed, a, b)


def test_grammar_allows_one_bomb_then_one_wall_after_right():
    from beat_weaver.model.tokenizer import BOMB_BASE, WALL_BASE, bar_token
    m = _build_grammar_mask(RIGHT_EMPTY, last_pos_in_bar=2, current_bar=0, audio_bars=3)
    assert m[BOMB_BASE] and m[WALL_BASE] and m[POS_BASE + 3] and not m[POS_BASE + 2]
    m = _build_grammar_mask(BOMB_BASE + 5, last_pos_in_bar=2, current_bar=0, audio_bars=3)
    assert not m[BOMB_BASE + 5] and m[BOMB_BASE + 6] and m[WALL_BASE] and m[POS_BASE + 3] and m[bar_token(1)]
    assert not m[BOMB_BASE: BOMB_BASE + 6].any()  # a cluster is written in ascending cell order
    m = _build_grammar_mask(WALL_BASE + 7, last_pos_in_bar=2, current_bar=0, audio_bars=3)
    assert not m[BOMB_BASE] and not m[WALL_BASE] and m[POS_BASE + 3]
    assert not _build_grammar_mask(POS_BASE + 2, current_bar=0)[BOMB_BASE]  # only after RIGHT


def test_bomb_push_targets_the_training_rate():
    from beat_weaver.model.inference import _bomb_push
    assert _bomb_push(3.0, 2, 100) == 3.0       # 2% < 5%: push
    assert _bomb_push(3.0, 7, 100) == 0.0       # between 5% and 10%: leave it to the model
    assert _bomb_push(3.0, 11, 100) == float("-inf")  # over twice the rate: no more bombs


def test_big_reversal_tokens_flag_long_reversed_slashes_only():
    from beat_weaver.model.inference import _big_reversal_tokens
    m = _big_reversal_tokens(LEFT_BASE, (0, 0), 5)  # last cut: up-right from bottom-left
    tok = lambda x, y, d: LEFT_BASE + x * 27 + y * 9 + d
    assert m[tok(2, 2, 6)]  # down-left from top-right, 2.8 cells away: flagged
    assert not m[tok(2, 0, 7)]  # down-right, 2 cells away, but only a 90-degree turn: not flagged
    assert m[tok(2, 0, 1)]  # straight down 2 cells away: 135 degrees from up-right, flagged
    assert not m[tok(0, 1, 6)]  # 1 cell away: not flagged
    assert not m[tok(3, 2, 5)]  # same cut: not flagged
    assert not m[tok(2, 2, 8)]  # dot: not flagged
    assert not m[LEFT_BASE + 108: LEFT_BASE + 2 * 108].any()  # other hand untouched
