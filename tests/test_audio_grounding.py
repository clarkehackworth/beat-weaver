"""Token dropout and the onset-alignment loss: the two measures that make the
decoder use the audio instead of predicting the map from its own past."""
import pytest

torch = pytest.importorskip("torch")

from beat_weaver.model.config import ModelConfig
from beat_weaver.model.tokenizer import (
    BAR_BASE, DIFF_EXPERT, END, LEFT_EMPTY, PAD, POS_BASE, POS_COUNT, RIGHT_EMPTY, START,
    SUBDIVISIONS_PER_BAR, bar_token,
)
from beat_weaver.model.training import apply_token_dropout, note_frame_targets
from beat_weaver.model.transformer import BeatWeaverModel


def _small_config(**kw):
    base = dict(
        max_seq_len=64, max_audio_len=256, n_mels=8, encoder_layers=1, encoder_dim=32,
        encoder_heads=4, encoder_ff_dim=64, decoder_layers=1, decoder_dim=32,
        decoder_heads=4, decoder_ff_dim=64, dropout=0.0, use_onset_features=False,
    )
    base.update(kw)
    return ModelConfig(**base)


class TestNoteFrameTargets:
    def test_frames_come_from_bar_and_pos_tokens(self):
        # bar 0 pos 5, bar 0 pos 20, bar 2 pos 3  -> frames 5, 20, 2*64+3
        seq = [START, DIFF_EXPERT, bar_token(0), POS_BASE + 5, LEFT_EMPTY, RIGHT_EMPTY,
               POS_BASE + 20, LEFT_EMPTY, RIGHT_EMPTY, bar_token(1), bar_token(2),
               POS_BASE + 3, LEFT_EMPTY, RIGHT_EMPTY, END]
        tok = torch.tensor([seq])
        t = note_frame_targets(tok, 256)
        assert t.shape == (1, 256)
        assert t[0].nonzero().flatten().tolist() == [5, 20, 2 * SUBDIVISIONS_PER_BAR + 3]

    def test_frames_past_audio_are_dropped_and_padding_ignored(self):
        seq = [START, DIFF_EXPERT, bar_token(3), POS_BASE + 0, LEFT_EMPTY, RIGHT_EMPTY, END, PAD, PAD]
        t = note_frame_targets(torch.tensor([seq]), 128)          # frame 192 >= 128
        assert t.sum() == 0
        t = note_frame_targets(torch.tensor([seq]), 256)
        assert t[0].nonzero().flatten().tolist() == [3 * SUBDIVISIONS_PER_BAR]

    def test_batch_rows_are_independent(self):
        a = [START, DIFF_EXPERT, bar_token(0), POS_BASE + 1, LEFT_EMPTY, RIGHT_EMPTY, END, PAD]
        b = [START, DIFF_EXPERT, bar_token(1), POS_BASE + 7, LEFT_EMPTY, RIGHT_EMPTY, END, PAD]
        t = note_frame_targets(torch.tensor([a, b]), 256)
        assert t[0].nonzero().flatten().tolist() == [1]
        assert t[1].nonzero().flatten().tolist() == [64 + 7]


class TestTokenDropout:
    def test_rate_and_protection(self):
        torch.manual_seed(0)
        tok = torch.full((64, 200), 50)
        out = apply_token_dropout(tok, 0.3)
        assert (out[:, :2] == 50).all()                            # START/DIFF never dropped
        rate = (out[:, 2:] == PAD).float().mean().item()
        assert 0.25 < rate < 0.35
        assert ((out == PAD) | (out == 50)).all()                   # only PAD is ever written

    def test_zero_is_identity(self):
        tok = torch.randint(3, 300, (2, 30))
        assert torch.equal(apply_token_dropout(tok, 0.0), tok)


class TestOnsetHead:
    def test_forward_with_onset_shapes_and_forward_unchanged(self):
        cfg = _small_config()
        m = BeatWeaverModel(cfg).eval()
        mel = torch.randn(2, cfg.n_mels, 256); tok = torch.randint(3, 300, (2, 20))
        logits, onset = m.forward_with_onset(mel, tok)
        assert logits.shape == (2, 20, cfg.vocab_size)
        assert onset.shape == (2, 256)
        assert torch.allclose(logits, m(mel, tok))                 # plain forward is unaffected

    def test_onset_head_learns_note_positions_from_audio(self):
        """A few steps of the aux loss alone must make the encoder output predict
        where notes are from the audio: the whole point of the head."""
        torch.manual_seed(0)
        cfg = _small_config()
        m = BeatWeaverModel(cfg)
        # audio where a loud frame marks each note, and tokens placing notes there
        frames = [5, 70, 130, 200]
        mel = torch.zeros(1, cfg.n_mels, 256); mel[0, :, frames] = 5.0
        seq = [START, DIFF_EXPERT]
        for f in frames:
            seq += [bar_token(f // 64)] if f // 64 > (len([x for x in seq if BAR_BASE <= x < BAR_BASE + 64]) - 1) else []
            seq += [POS_BASE + f % 64, LEFT_EMPTY, RIGHT_EMPTY]
        seq.append(END)
        tok = torch.tensor([seq]); target = note_frame_targets(tok, 256)
        assert target.sum() == len(frames)
        opt = torch.optim.Adam(m.parameters(), lr=1e-2)
        for _ in range(60):
            _, onset = m.forward_with_onset(mel, tok)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(onset, target)
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            _, onset = m.forward_with_onset(mel, tok)
        top = onset[0].topk(len(frames)).indices.sort().values.tolist()
        assert top == frames, top


class TestAudioIsUsed:
    """End to end through Trainer.train_epoch, on the simplest task whose answer
    is only in the audio: one bar, three loud frames, a note at each. After
    training, the model must predict a held-out map far better with its own
    audio than with another song's. This is the measurement that exposed the
    bug on the real model (identical loss with the right and the wrong song)."""

    @staticmethod
    def _pairs(n, cfg, seed, k=3):
        g = torch.Generator().manual_seed(seed)
        mels, toks = [], []
        for _ in range(n):
            frames = sorted(torch.randperm(64, generator=g)[:k].tolist())
            mel = torch.zeros(cfg.n_mels, 64); mel[:, frames] = 5.0
            seq = [START, DIFF_EXPERT, bar_token(0)]
            for f in frames:
                seq += [POS_BASE + f, LEFT_EMPTY, RIGHT_EMPTY]
            seq.append(END); seq += [PAD] * (cfg.max_seq_len - len(seq))
            mels.append(mel); toks.append(torch.tensor(seq))
        return torch.stack(mels), torch.stack(toks)

    @staticmethod
    def _pos_accuracy(m, mel, tok):
        with torch.no_grad():
            pred = m(mel, tok[:, :-1]).argmax(-1); t = tok[:, 1:]
            pos = (t >= POS_BASE) & (t < POS_BASE + POS_COUNT)
            return (pred == t)[pos].float().mean().item()

    @pytest.fixture(scope="class")
    def trained(self, tmp_path_factory):
        from beat_weaver.model.training import Trainer
        torch.manual_seed(0)
        cfg = _small_config(max_audio_len=64, max_seq_len=16, token_dropout=0.3, onset_loss_weight=0.5,
                            color_balance_weight=0.0, learning_rate=3e-3, warmup_steps=1,
                            gradient_accumulation_steps=1, batch_size=32)
        mels, toks = self._pairs(512, cfg, seed=1); masks = torch.ones_like(toks, dtype=torch.bool)
        batches = [(mels[i:i+32], torch.ones(32, 64, dtype=torch.bool), toks[i:i+32], masks[i:i+32])
                   for i in range(0, 512, 32)]
        tr = Trainer(BeatWeaverModel(cfg), cfg, tmp_path_factory.mktemp("run"), device=torch.device("cpu"))
        for _ in range(20):
            tr.train_epoch(batches)
        tr.model.eval()
        return tr.model, cfg

    def test_grounded_model_reads_the_audio(self, trained):
        m, cfg = trained
        tm, tt = self._pairs(64, cfg, seed=2)                       # unseen songs
        own = self._pos_accuracy(m, tm, tt)
        wrong = self._pos_accuracy(m, tm.roll(1, dims=0), tt)       # each map under another song's audio
        assert own > 0.5, own                                       # chance is 1/64
        assert wrong < 0.1, wrong

    def test_inference_uses_the_same_memory_as_training(self, trained):
        """generate() must go through encode_audio: calling the bare encoder
        drops the absolute positions the model was trained with."""
        from beat_weaver.model.inference import generate
        m, cfg = trained
        tm, tt = self._pairs(4, cfg, seed=3)
        hits = total = 0
        for i in range(4):
            tokens = generate(m, tm[i], "Expert", cfg, temperature=0.0, audio_bars=1)
            got = {t - POS_BASE for t in tokens if POS_BASE <= t < POS_BASE + POS_COUNT}
            want = set(tm[i][0].nonzero().flatten().tolist())
            hits += len(got & want); total += len(want)
        assert hits / total > 0.5, (hits, total)
