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
    t = torch.tensor([[START, 5, bar_token(0), POS_BASE + 3, LEFT_EMPTY, RIGHT_BASE + 7, POS_BASE + 9, LEFT_BASE + 1, RIGHT_EMPTY, END, PAD]])
    out = placement_targets(t).tolist()[0]
    assert out == [PAD, PAD, PAD, PAD, LEFT_EMPTY, RIGHT_BASE + 7, PAD, LEFT_BASE + 1, RIGHT_EMPTY, PAD, PAD]
