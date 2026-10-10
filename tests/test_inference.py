"""Tests for inference and grammar-constrained generation."""

import pytest

torch = pytest.importorskip("torch")

from beat_weaver.model.config import ModelConfig
from beat_weaver.model.inference import (
    _build_grammar_mask, _crossover_outward_tokens, _decode_note_token, generate, generate_full_song,
)
from beat_weaver.model.tokenizer import (
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
        # every other cell still allowed, and RIGHT_EMPTY too
        assert mask[_encode_note_token(RIGHT_BASE, 1, 1, 0)]
        assert mask[_encode_note_token(RIGHT_BASE, 2, 0, 0)]
        assert mask[RIGHT_EMPTY]
        assert mask[RIGHT_BASE:RIGHT_BASE + RIGHT_COUNT].sum() == RIGHT_COUNT - 9 - 18  # minus crossover-outward tokens
        # without the cell the full set is allowed (LEFT_EMPTY case)
        assert _build_grammar_mask(LEFT_EMPTY)[RIGHT_BASE:RIGHT_BASE + RIGHT_COUNT].sum() == RIGHT_COUNT - 18

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
        assert mask[LEFT_BASE:LEFT_BASE + LEFT_COUNT].sum() == LEFT_COUNT - 18
        assert not mask[_crossover_outward_tokens(LEFT_BASE)].any()
        assert not mask[bar_token(0)]
        assert not mask[RIGHT_EMPTY]

    def test_after_left(self):
        mask = _build_grammar_mask(LEFT_BASE + 5)
        # RIGHT tokens
        assert mask[RIGHT_EMPTY]
        assert mask[RIGHT_BASE:RIGHT_BASE + RIGHT_COUNT].sum() == RIGHT_COUNT - 18
        assert not mask[_crossover_outward_tokens(RIGHT_BASE)].any()
        assert not mask[bar_token(0)]
        assert not mask[LEFT_EMPTY]

    def test_after_left_empty(self):
        mask = _build_grammar_mask(LEFT_EMPTY)
        # RIGHT tokens
        assert mask[RIGHT_EMPTY]
        assert mask[RIGHT_BASE:RIGHT_BASE + RIGHT_COUNT].sum() == RIGHT_COUNT - 18

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
            vocab_size=355,
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
            vocab_size=355,
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
                tokens = generate(model, mel, "Expert", config, temperature=1.5, seed=seed, flow_penalty=flow_penalty)
                for base, count in ((LEFT_BASE, LEFT_COUNT), (RIGHT_BASE, RIGHT_COUNT)):
                    dirs = [_decode_note_token(t, base)[2] for t in tokens if base <= t < base + count]
                    for d1, d2 in zip(dirs, dirs[1:]):
                        if d1 != 8 and d2 != 8:
                            tot += 1; rep += d1 == d2
            return rep / max(tot, 1), tot
        r0, n0 = repeat_rate(0.0); r1, n1 = repeat_rate(50.0)
        assert n0 > 50 and n1 > 50
        assert r1 < r0 * 0.5, (r0, r1)

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
        assert not m[tok(2, 1, 3)], "both cutting right: fine"
        assert not m[tok(2, 1, 0)] and not m[tok(2, 1, 1)], "perpendicular: fine"
        assert not m[tok(3, 1, 2)], "two cells away: not adjacent, allowed"
        assert m[tok(1, 1, 0)] and m[tok(1, 1, 3)] and not m[tok(1, 1, 8)], "same cell: any directional cut blocked, dot ok"
        assert not m[tok(0, 1, 3)], "behind the left cut: fine"
        m2 = _converging_right_tokens((1, 1), 0)  # left cutting UP
        assert m2[tok(1, 2, 1)] and m2[tok(2, 2, 6)] and not m2[tok(2, 2, 0)]
        assert _converging_right_tokens((1, 1), 8).sum() == 0

    def test_near_converging_is_blocked_by_default_and_allowed_by_flag(self):
        from beat_weaver.model.inference import _converging_right_tokens
        from beat_weaver.model.tokenizer import _encode_note_token
        tok = lambda x, y, d: _encode_note_token(RIGHT_BASE, x, y, d)
        # left at (1,1) cutting UP; right beside it at (2,1) cutting DOWN-LEFT: 135 deg apart, right aimed at left
        near = tok(2, 1, 6)
        assert _converging_right_tokens((1, 1), 0)[near]
        assert not _converging_right_tokens((1, 1), 0, block_near=False)[near]
        assert _converging_right_tokens((1, 1), 3, block_near=False)[tok(2, 1, 2)], "head-on stays blocked either way"
        assert not _converging_right_tokens((1, 1), 3)[tok(2, 1, 1)], "90 deg apart is not a near miss"
        assert not _converging_right_tokens((1, 1), 3)[tok(3, 1, 6)], "two cells away is fine"
        # neither aimed at the other: left at (1,1) cutting LEFT, right at (2,1) cutting UP-RIGHT, 135 apart but diverging
        assert not _converging_right_tokens((1, 1), 2)[tok(2, 1, 5)]

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
        assert n0 > 20 and r1 >= r0 and r1 == 1.0, (r0, r1)

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

    def test_notes_per_second_overrides_notes_per_beat(self, small_model):
        model, config = small_model
        mel = torch.randn(80, config.max_audio_len)
        slow = generate_full_song(model, mel, "Expert", config, 60.0, seed=1, two_stage=True, notes_per_second=4.0, lead_in_seconds=0)
        fast = generate_full_song(model, mel, "Expert", config, 180.0, seed=1, two_stage=True, notes_per_second=4.0, lead_in_seconds=0)
        per_beat = lambda ns, bpm: len(ns) / (max(n.beat for n in ns) + 1)
        assert per_beat(slow, 60) > 2 * per_beat(fast, 180), (per_beat(slow, 60), per_beat(fast, 180))

    def test_end_seconds_drops_notes_after_the_music(self, small_model):
        import numpy as np
        from beat_weaver.model.inference import last_music_second
        sr = 22050
        audio = np.concatenate([np.random.default_rng(0).standard_normal(5 * sr) * 0.3, np.zeros(3 * sr)]).astype(np.float32)
        end = last_music_second(audio, sr)
        assert 4.9 <= end <= 5.2, end
        model, config = small_model
        mel = torch.randn(80, config.max_audio_len)
        notes = generate_full_song(model, mel, "Expert", config, 120.0, seed=1, two_stage=True, notes_per_beat=2.0,
                                   lead_in_seconds=0.0, end_seconds=end)
        assert notes and max(n.time_seconds for n in notes) < end

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

    def test_crossover_notes_never_face_outward(self, small_model):
        """A left note in columns 2-3 never cuts right-ish, a right note in
        columns 0-1 never cuts left-ish: those are the near-unhittable blocks."""
        from beat_weaver.model.tokenizer import _decode_note_token
        model, config = small_model
        mel = torch.randn(80, config.max_audio_len)
        for seed in range(6):
            for t in generate(model, mel, "Expert", config, seed=seed, audio_bars=4, temperature=2.0):
                if LEFT_BASE <= t < LEFT_BASE + LEFT_COUNT:
                    x, _, d = _decode_note_token(t, LEFT_BASE)
                    assert not (x >= 2 and d in (3, 5, 7)), (x, d)
                elif RIGHT_BASE <= t < RIGHT_BASE + RIGHT_COUNT:
                    x, _, d = _decode_note_token(t, RIGHT_BASE)
                    assert not (x <= 1 and d in (2, 4, 6)), (x, d)

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
                    frames.append(bar * 64 + t - POS_BASE)
                    assert not (toks[i + 1] == LEFT_EMPTY and toks[i + 2] == RIGHT_EMPTY)
            assert frames == sorted(high), (seed, frames)
            assert toks[-1] == END

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
