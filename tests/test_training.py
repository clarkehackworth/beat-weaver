"""Tests for training utilities — color balance loss."""

import pytest

torch = pytest.importorskip("torch")

from beat_weaver.model.training import _color_balance_loss
from beat_weaver.model.tokenizer import LEFT_BASE, LEFT_COUNT, RIGHT_BASE, RIGHT_COUNT


class TestColorBalanceLoss:
    def test_balanced_near_zero(self):
        """Loss near zero for 50/50 left/right predictions."""
        # Create logits that predict equal LEFT and RIGHT probability
        batch, seq, vocab = 2, 10, 355
        logits = torch.zeros(batch, seq, vocab)
        # Set equal logits for all LEFT and RIGHT tokens
        logits[:, :, LEFT_BASE:LEFT_BASE + LEFT_COUNT] = 1.0
        logits[:, :, RIGHT_BASE:RIGHT_BASE + RIGHT_COUNT] = 1.0
        loss = _color_balance_loss(logits)
        assert loss.item() < 0.01

    def test_imbalanced_positive(self):
        """Loss > 0 for 100% left predictions."""
        batch, seq, vocab = 2, 10, 355
        logits = torch.full((batch, seq, vocab), -10.0)
        # All probability on LEFT tokens
        logits[:, :, LEFT_BASE:LEFT_BASE + LEFT_COUNT] = 5.0
        loss = _color_balance_loss(logits)
        assert loss.item() > 0.1

    def test_gradient_flows(self):
        """Gradients flow through auxiliary loss."""
        batch, seq, vocab = 2, 10, 355
        logits = torch.randn(batch, seq, vocab, requires_grad=True)
        loss = _color_balance_loss(logits)
        loss.backward()
        assert logits.grad is not None
        assert logits.grad.abs().sum() > 0

    def test_no_note_tokens(self):
        """Returns 0 when no positions have significant note probability."""
        batch, seq, vocab = 2, 10, 355
        logits = torch.zeros(batch, seq, vocab)
        # All probability on PAD/START/END tokens, none on notes
        logits[:, :, 0:8] = 10.0
        loss = _color_balance_loss(logits)
        assert loss.item() == 0.0


def test_placement_targets_keep_only_note_tokens():
    import torch
    from beat_weaver.model.tokenizer import (
        START, END, PAD, LEFT_EMPTY, RIGHT_EMPTY, LEFT_BASE, RIGHT_BASE, POS_BASE, bar_token,
    )
    from beat_weaver.model.training import placement_targets
    from beat_weaver.model.tokenizer import BOMB_BASE, WALL_BASE
    t = torch.tensor([[START, 5, bar_token(0), POS_BASE + 3, LEFT_EMPTY, RIGHT_BASE + 7, BOMB_BASE + 4, WALL_BASE + 9, POS_BASE + 9, LEFT_BASE + 1, RIGHT_EMPTY, END, PAD]])
    out = placement_targets(t).tolist()[0]
    assert out == [PAD, PAD, PAD, PAD, LEFT_EMPTY, RIGHT_BASE + 7, BOMB_BASE + 4, WALL_BASE + 9, PAD, LEFT_BASE + 1, RIGHT_EMPTY, PAD, PAD]
    # a POS that opens an extras-only position (LEFT_EMPTY, RIGHT_EMPTY next) is kept: nothing else schedules it
    t = torch.tensor([[bar_token(0), POS_BASE + 3, LEFT_EMPTY, RIGHT_EMPTY, BOMB_BASE + 1, POS_BASE + 5, LEFT_BASE + 2, RIGHT_EMPTY, END]])
    out = placement_targets(t).tolist()[0]
    assert out == [PAD, POS_BASE + 3, LEFT_EMPTY, RIGHT_EMPTY, BOMB_BASE + 1, PAD, LEFT_BASE + 2, RIGHT_EMPTY, PAD]


def test_token_loss_matches_cross_entropy_and_weights_extras():
    import torch
    from torch import nn
    from beat_weaver.model.tokenizer import PAD, LEFT_BASE, RIGHT_EMPTY, BOMB_BASE, WALL_BASE, POS_BASE, VOCAB_SIZE
    from beat_weaver.model.training import token_loss
    torch.manual_seed(0)
    logits = torch.randn(1, 6, VOCAB_SIZE)
    targets = torch.tensor([[LEFT_BASE + 3, RIGHT_EMPTY, BOMB_BASE + 2, WALL_BASE + 5, POS_BASE + 7, PAD]])
    ref = nn.CrossEntropyLoss(ignore_index=PAD, label_smoothing=0.1)(logits.reshape(-1, VOCAB_SIZE), targets.reshape(-1))
    assert torch.isclose(token_loss(logits, targets, 0.1), ref)
    # weight 0 on extras in placement mode: only the two note targets count
    notes_only = nn.CrossEntropyLoss(ignore_index=PAD, label_smoothing=0.1)(logits[0, :2], targets[0, :2])
    assert torch.isclose(token_loss(logits, targets, 0.1, extras_weight=0.0, placement_only=True), notes_only)
    # outside placement mode the POS is an ordinary target, not an extra
    w0 = token_loss(logits, targets, 0.1, extras_weight=0.0, placement_only=False)
    assert not torch.isclose(w0, notes_only)


def test_drift_stats_measure_doubles_and_big_reversals():
    from beat_weaver.schemas.normalized import Note
    from beat_weaver.model.probe import drift_stats
    N = lambda t, x, y, c, d: Note(beat=t, time_seconds=t / 2, x=x, y=y, color=c, cut_direction=d)
    notes = [
        N(0.0, 1, 0, 0, 1), N(0.0, 2, 0, 1, 1),  # a double
        N(0.5, 1, 2, 0, 0),                      # left: 2 cells up, cut reversed (down -> up): big reversal
        N(0.25, 2, 1, 1, 0),                     # right: 1 cell, reversed, within half a beat: fast but not big
        N(1.25, 2, 1, 1, 8),                     # dot: skipped
        N(3.0, 0, 0, 0, 1),                      # left, 2.5 beats later: not a fast swing
    ]
    s = drift_stats(notes)
    assert s["doubles"] == 1 / 5  # 5 instants, one with both hands
    assert s["reversal"] == 0.5  # fast swings: left 0->0.5 (big), right 0->0.25 (not); the dot and the slow one are skipped
    assert s["left"] == 0.5
    assert abs(s["nps"] - 6 / 1.5) < 1e-9
    assert drift_stats([]) == {"doubles": 0.0, "reversal": 0.0, "left": 0.0, "nps": 0.0}
