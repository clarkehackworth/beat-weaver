"""Per-epoch drift probe: generate one fixed song with fixed flags and measure what the decoder samples.

Mean cross-entropy did not see the extras4 drift (doubles 26-52% -> 5-13%, big fast reversals 0.11 -> 0.19 at
equal flags; note CE 2.2038 -> 2.1977), so training logs these shares from a real generation instead.
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from pathlib import Path

import torch

from beat_weaver.schemas.normalized import Note

logger = logging.getLogger(__name__)

_ANGLE = {0: 90, 1: 270, 2: 180, 3: 0, 4: 135, 5: 45, 6: 225, 7: 315}

# The extras8 Expert recipe without the reversal/doubles clamps, so drift shows rather than being masked
PROBE_FLAGS = dict(
    two_stage=True, notes_per_second=3.0, min_gap_seconds=0.1, doubles_bias=1.3, flow_penalty=4.0, dynamics=1.0,
    vision_penalty=6.0, stream_bias=1.0, column_max_run=4, diagonal_bias=1.5, hand_balance=10.0,
    crossover_penalty=2.0, seed=1,
)


def drift_stats(notes: list[Note]) -> dict[str, float]:
    """doubles: share of hit instants with both hands. reversal: of a hand's swings within half a beat of its
    last, the share travelling 2+ cells and turning 135+ degrees (real Expert median 0.43, Expert+ 0.53).
    left: share of notes on the left hand. nps: notes per second over the mapped span."""
    if not notes:
        return {"doubles": 0.0, "reversal": 0.0, "left": 0.0, "nps": 0.0}
    per = Counter(n.beat for n in notes)
    fast = big = 0
    for hand in (0, 1):
        s = sorted((n for n in notes if n.color == hand), key=lambda n: n.beat)
        for p, q in zip(s, s[1:]):
            if p.cut_direction > 7 or q.cut_direction > 7 or not 0 < q.beat - p.beat <= 0.5:
                continue
            fast += 1
            turn = abs((_ANGLE[q.cut_direction] - _ANGLE[p.cut_direction] + 180) % 360 - 180)
            big += math.hypot(q.x - p.x, q.y - p.y) >= 2 and turn >= 135
    span = max(n.time_seconds for n in notes) - min(n.time_seconds for n in notes)
    return {
        "doubles": sum(v > 1 for v in per.values()) / len(per),
        "reversal": big / max(fast, 1),
        "left": sum(n.color == 0 for n in notes) / len(notes),
        "nps": len(notes) / max(span, 1e-6),
    }


def load_probe_audio(audio_path: Path, config) -> tuple[torch.Tensor, float]:
    """The CLI's audio pipeline, run once so each epoch's probe only pays for generation."""
    from beat_weaver.model.audio import beat_align_spectrogram, compute_mel_spectrogram, compute_mel_with_onset, detect_bpm, load_audio

    audio, sr = load_audio(audio_path, sr=config.sample_rate)
    bpm = detect_bpm(audio, sr=sr)
    if config.use_onset_features:
        mel = compute_mel_with_onset(audio, sr=sr, n_mels=config.n_mels, n_fft=config.n_fft, hop_length=config.hop_length)
    else:
        mel = compute_mel_spectrogram(audio, sr=sr, n_mels=config.n_mels, n_fft=config.n_fft, hop_length=config.hop_length)
    mel = beat_align_spectrogram(mel, sr=sr, hop_length=config.hop_length, bpm=bpm)
    return torch.from_numpy(mel), bpm


def probe(model, config, mel: torch.Tensor, bpm: float, difficulty: str = "Expert") -> dict[str, float]:
    """Generate the probe song with ``PROBE_FLAGS`` and return ``drift_stats``. Never raises: a failed probe logs
    and returns an empty dict so it cannot take the training run down."""
    from beat_weaver.model.inference import generate_full_song

    was_training = model.training
    try:
        model.eval()
        with torch.no_grad():
            notes = generate_full_song(model, mel, difficulty, config, bpm, **PROBE_FLAGS)
        return drift_stats(notes)
    except Exception:  # noqa: BLE001 - the probe is diagnostic only
        logger.exception("Drift probe failed")
        return {}
    finally:
        if was_training:
            model.train()
