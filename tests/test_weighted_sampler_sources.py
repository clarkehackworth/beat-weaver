"""Per-source batch shares in build_weighted_sampler."""
import pytest
from beat_weaver.model.dataset import build_weighted_sampler


class _DS:
    def __init__(self, samples): self.samples = samples
    def __len__(self): return len(self.samples)


def _share(sampler, samples, source):
    w = list(sampler.weights)
    return sum(w[i] for i, s in enumerate(samples) if s["source"] == source) / sum(w)


def test_named_sources_get_exact_shares():
    samples = ([{"source": "official"}] * 10 + [{"source": "local_custom"}] * 40
               + [{"source": "beatsaver", "score": sc} for sc in (0.5, 1.0, 1.5, 1.0) * 50])
    s = build_weighted_sampler(_DS(samples), source_ratios={"official": 0.2, "local_custom": 0.25})
    assert _share(s, samples, "official") == pytest.approx(0.20)
    assert _share(s, samples, "local_custom") == pytest.approx(0.25)
    assert _share(s, samples, "beatsaver") == pytest.approx(0.55)
    # within the remainder, weight follows score
    w = list(s.weights)
    i15 = next(i for i, x in enumerate(samples) if x.get("score") == 1.5)
    i05 = next(i for i, x in enumerate(samples) if x.get("score") == 0.5)
    assert w[i15] == pytest.approx(3 * w[i05])


def test_old_two_way_call_unchanged():
    samples = [{"source": "official"}] * 5 + [{"source": "beatsaver", "score": 1.0}] * 95
    s = build_weighted_sampler(_DS(samples), official_ratio=0.2)
    assert _share(s, samples, "official") == pytest.approx(0.20)


def test_missing_named_source_is_ignored():
    samples = [{"source": "official"}] * 5 + [{"source": "beatsaver", "score": 1.0}] * 95
    s = build_weighted_sampler(_DS(samples), source_ratios={"official": 0.2, "local_custom": 0.25})
    assert _share(s, samples, "official") == pytest.approx(0.20)
    assert _share(s, samples, "beatsaver") == pytest.approx(0.80)


def test_single_source_returns_none():
    assert build_weighted_sampler(_DS([{"source": "beatsaver", "score": 1.0}] * 10)) is None


def test_ratios_leaving_no_remainder_raise():
    samples = [{"source": "official"}] * 5 + [{"source": "beatsaver"}] * 5
    with pytest.raises(ValueError):
        build_weighted_sampler(_DS(samples), source_ratios={"official": 1.0})
