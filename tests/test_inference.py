"""Tests for inference and grammar-constrained generation."""

import pytest

torch = pytest.importorskip("torch")

from beat_weaver.model.config import ModelConfig
from beat_weaver.model.inference import _build_grammar_mask, generate, generate_full_song
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
        assert mask[LEFT_BASE:LEFT_BASE + LEFT_COUNT].all()
        assert not mask[bar_token(0)]
        assert not mask[RIGHT_EMPTY]

    def test_after_left(self):
        mask = _build_grammar_mask(LEFT_BASE + 5)
        # RIGHT tokens
        assert mask[RIGHT_EMPTY]
        assert mask[RIGHT_BASE:RIGHT_BASE + RIGHT_COUNT].all()
        assert not mask[bar_token(0)]
        assert not mask[LEFT_EMPTY]

    def test_after_left_empty(self):
        mask = _build_grammar_mask(LEFT_EMPTY)
        # RIGHT tokens
        assert mask[RIGHT_EMPTY]
        assert mask[RIGHT_BASE:RIGHT_BASE + RIGHT_COUNT].all()

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
