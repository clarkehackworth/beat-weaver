"""Does the map follow the song? Two numbers from the audio alone, no model.

- energy_corr: Pearson correlation between per-bar note count and per-bar audio loudness.
  A flat budget gives ~0 (staccato, notes regardless of the music); favourites are well above.
- onset_hit: fraction of notes within 60 ms of an audio onset (an energy rise). Timing, not density.

Audio is decoded with ffmpeg to mono 22.05 kHz PCM; the envelope is RMS at 10 ms hops. No librosa.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

from beat_weaver.schemas.normalized import Note

SR = 22050
HOP = 220  # ~10 ms


def envelope(audio_path: Path) -> np.ndarray:
    """RMS loudness at 10 ms hops, via ffmpeg. Returns a 1-D float array."""
    cmd = ["ffmpeg", "-v", "error", "-i", str(audio_path), "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"]
    pcm = np.frombuffer(subprocess.run(cmd, capture_output=True, check=True).stdout, dtype=np.float32)
    n = len(pcm) // HOP
    return np.sqrt((pcm[: n * HOP].reshape(n, HOP) ** 2).mean(axis=1))


def onsets(env: np.ndarray, min_sep_s: float = 0.08) -> np.ndarray:
    """Times (s) where loudness rises sharply: peaks of the positive first difference above its mean + 1 sd."""
    d = np.diff(np.log1p(env * 100), prepend=0.0).clip(min=0)
    thr = d.mean() + d.std()
    idx = np.flatnonzero((d > thr) & (d >= np.roll(d, 1)) & (d >= np.roll(d, -1)))
    keep, last = [], -np.inf
    for i in idx:
        if i - last >= min_sep_s * SR / HOP:
            keep.append(i)
            last = i
    return np.array(keep) * HOP / SR


def stats(notes: list[Note], env: np.ndarray, bpm: float) -> dict[str, float]:
    if len(notes) < 2:
        return {"energy_corr": 0.0, "onset_hit": 0.0}
    t = np.array([n.time_seconds for n in notes])
    bar_s = 4 * 60.0 / bpm
    n_bars = int(np.ceil(max(t[-1], len(env) * HOP / SR) / bar_s)) + 1
    counts = np.bincount((t / bar_s).astype(int), minlength=n_bars).astype(float)
    bar_of_hop = (np.arange(len(env)) * HOP / SR / bar_s).astype(int)
    loud = np.bincount(bar_of_hop, weights=env, minlength=n_bars) / np.maximum(np.bincount(bar_of_hop, minlength=n_bars), 1)
    hi = int(max(t[-1], len(env) * HOP / SR) / bar_s) + 1  # song start to the later of last note / audio end
    c, l = counts[:hi], loud[:hi]
    corr = float(np.corrcoef(c, l)[0, 1]) if c.std() > 0 and l.std() > 0 else 0.0
    on = onsets(env)
    if len(on):
        i = np.searchsorted(on, t)
        nearest = np.minimum(np.abs(on[i.clip(0, len(on) - 1)] - t), np.abs(on[(i - 1).clip(0, len(on) - 1)] - t))
        hit = float(np.mean(nearest <= 0.06))  # per note, onset on either side
    else:
        hit = 0.0
    return {"energy_corr": corr, "onset_hit": hit}


def find_audio(folder: Path) -> Path | None:
    for pat in ("*.egg", "*.ogg", "*.mp3", "*.wav"):
        hits = sorted(folder.glob(pat))
        if hits:
            return hits[0]
    return None


if __name__ == "__main__":
    # synthetic: 4 s of silence, then 4 s of 4 Hz clicks, at 120 bpm (bar = 2 s)
    pcm = np.zeros(8 * SR, dtype=np.float32)
    for k in range(16):
        s = 4 * SR + int(k * 0.25 * SR)
        pcm[s: s + 400] = 0.8
    env = np.sqrt((pcm[: len(pcm) // HOP * HOP].reshape(-1, HOP) ** 2).mean(axis=1))
    on = onsets(env)
    assert 14 <= len(on) <= 17 and on.min() >= 3.9, (len(on), on[:3])
    dense_late = [Note(beat=0, time_seconds=4 + k * 0.25, x=1, y=1, color=k % 2, cut_direction=1) for k in range(16)]
    flat = [Note(beat=0, time_seconds=k * 0.5, x=1, y=1, color=k % 2, cut_direction=1) for k in range(16)]
    a, b = stats(dense_late, env, 120.0), stats(flat, env, 120.0)
    assert a["energy_corr"] > 0.8 and a["onset_hit"] > 0.9, a
    assert b["onset_hit"] < 0.6, b
    straddle = [Note(beat=0, time_seconds=4 + k * 0.25 + (0.02 if k % 2 else -0.02), x=1, y=1, color=0, cut_direction=1) for k in range(16)]
    assert stats(straddle, env, 120.0)["onset_hit"] > 0.9, "notes just before and just after onsets both count"
    print("music ok", a, b)
