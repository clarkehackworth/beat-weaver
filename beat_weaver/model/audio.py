"""Audio preprocessing — mel spectrograms and beat-aligned framing.

Depends on librosa and soundfile (optional ML dependencies).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from scipy.interpolate import interp1d

logger = logging.getLogger(__name__)


def load_audio(path: Path, sr: int = 22050) -> tuple[np.ndarray, int]:
    """Load an audio file and resample to target sample rate.

    Returns (audio_mono, sample_rate) where audio_mono is float32 1-D.
    """
    path = Path(path)
    audio, orig_sr = sf.read(str(path), dtype="float32", always_2d=True)
    # Mix to mono
    audio = audio.mean(axis=1)
    # Resample if needed
    if orig_sr != sr:
        audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=sr)
    return audio, sr


def compute_mel_spectrogram(
    audio: np.ndarray,
    sr: int = 22050,
    n_mels: int = 80,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> np.ndarray:
    """Compute log-mel spectrogram.

    Returns float32 array of shape (n_mels, T).
    """
    mel = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length,
        n_mels=n_mels, window="hann",
    )
    # Convert to dB scale (log-magnitude), ref=max
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db.astype(np.float32)


def beat_align_spectrogram(
    mel: np.ndarray,
    sr: int,
    hop_length: int,
    bpm: float,
    subdivisions_per_beat: int = 16,
) -> np.ndarray:
    """Resample spectrogram frames to align with beat subdivisions.

    Each output frame corresponds to one 1/16th note position.

    Args:
        mel: Log-mel spectrogram of shape (n_mels, T_frames).
        sr: Audio sample rate.
        hop_length: STFT hop length used for mel.
        bpm: Beats per minute.
        subdivisions_per_beat: Number of subdivisions per beat (default 16).

    Returns:
        Float32 array of shape (n_mels, T_beats) where T_beats is the total
        number of beat subdivisions covered by the audio.
    """
    n_mels, n_frames = mel.shape

    # Time of each spectrogram frame
    frame_times = librosa.frames_to_time(
        np.arange(n_frames), sr=sr, hop_length=hop_length,
    )

    # Duration of the audio
    duration = frame_times[-1] if n_frames > 0 else 0.0

    # Total beat subdivisions
    beats_per_second = bpm / 60.0
    subs_per_second = beats_per_second * subdivisions_per_beat
    total_subs = int(np.ceil(duration * subs_per_second))

    if total_subs == 0:
        return np.zeros((n_mels, 0), dtype=np.float32)

    # Time of each subdivision
    sub_times = np.arange(total_subs) / subs_per_second

    # Interpolate: for each sub_time, find the nearest frame
    # Use linear interpolation across the time axis
    frame_indices = np.interp(sub_times, frame_times, np.arange(n_frames))

    # Vectorized interpolation across all mel bins at once
    x_coords = np.arange(n_frames, dtype=np.float64)
    f = interp1d(x_coords, mel, axis=1, kind="linear",
                 fill_value="extrapolate", assume_sorted=True)
    aligned = f(frame_indices).astype(np.float32)

    return aligned


def detect_bpm(
    audio: np.ndarray, sr: int = 22050, default: float = 120.0,
) -> float:
    """Estimate the BPM of an audio signal using librosa beat tracking.

    Returns the estimated tempo as a float.  Falls back to *default*
    (120 BPM) when beat tracking cannot determine a tempo (e.g. for very
    short or non-rhythmic audio).
    """
    tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
    # librosa may return an array; extract scalar
    if hasattr(tempo, "__len__"):
        tempo = float(tempo[0]) if len(tempo) > 0 else default
    tempo = float(tempo)
    if tempo <= 0:
        return default
    return refine_bpm(audio, sr, tempo)


def refine_bpm(audio: np.ndarray, sr: int, tempo: float, frac: float = 0.06, step: float = 0.01, hop: int = 220,
               min_prominence: float = 3.0) -> float:
    """Move a rough tempo to the candidate within +-``frac`` of it whose beat grid locks tightest to the audio.

    Beat trackers quantise tempo to their frame rate (129.2 for a 128 or 132 song, 117.45 for 115 or 120),
    and a 1 BPM error walks the beat grid through a full beat every minute, so by mid-song every bar is
    off the music (seen in play testing). Score = phase coherence of the onset envelope at
    the beat, half-beat and quarter-beat rates summed: one vector per frame at its phase in the beat,
    weighted by onset strength; a grid that matches the song keeps them pointing one way for the whole
    song. Measured on 41 official songs against their real BPM (octave errors allowed): tracker 2/41
    within 0.15 BPM, the old +-1.5 BPM binned search 18/41, this 32/41.
    """
    n = len(audio) // hop
    if n < 10:
        return tempo
    rms = np.sqrt((audio[: n * hop].astype(np.float32).reshape(n, hop) ** 2).mean(axis=1))
    d = np.clip(np.diff(np.log1p(rms * 100), prepend=0.0), 0, None)
    w = d / max(d.sum(), 1e-9)
    t = np.arange(n) * hop / sr

    def lock(c: np.ndarray) -> np.ndarray:
        return sum(np.abs(np.exp(2j * np.pi * h * np.outer(c, t) / 60) @ w) for h in (1, 2, 4))

    cands = np.arange(tempo * (1 - frac), tempo * (1 + frac), step)
    scores = np.concatenate([lock(c) for c in np.array_split(cands, max(1, len(cands) // 50))])
    best = float(cands[scores.argmax()])
    # Songs without a steady pulse give a flat, noisy score where "best" is arbitrary, so the detector's
    # tempo is kept unless the peak stands out. Measured peak / median: noise 1.6-1.9, songs 4-16.
    if scores.max() < min_prominence * np.median(scores):
        return tempo
    return float(round(best, 2))


def compute_onset_envelope(
    audio: np.ndarray,
    sr: int = 22050,
    hop_length: int = 512,
) -> np.ndarray:
    """Compute onset strength envelope.

    Returns float32 array of shape (1, T).
    """
    onset = librosa.onset.onset_strength(
        y=audio, sr=sr, hop_length=hop_length,
    )
    return onset.astype(np.float32).reshape(1, -1)


def compute_mel_with_onset(
    audio: np.ndarray,
    sr: int = 22050,
    n_mels: int = 80,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> np.ndarray:
    """Compute log-mel spectrogram with onset strength as extra channel.

    Returns float32 array of shape (n_mels + 1, T).
    """
    mel = compute_mel_spectrogram(audio, sr=sr, n_mels=n_mels, n_fft=n_fft,
                                  hop_length=hop_length)
    onset = compute_onset_envelope(audio, sr=sr, hop_length=hop_length)
    # Align lengths (onset may differ by 1 frame from mel)
    min_len = min(mel.shape[1], onset.shape[1])
    return np.vstack([mel[:, :min_len], onset[:, :min_len]])


# ── Audio manifest ──────────────────────────────────────────────────────────

_AUDIO_EXTENSIONS = {".ogg", ".egg", ".wav", ".mp3", ".flac"}


def _hash_folder(folder: Path) -> str:
    """Compute a content hash for a map folder.

    Uses the same algorithm as ``beat_weaver.pipeline.processor.compute_map_hash``
    so the audio manifest keys match the song hashes in the Parquet data.
    """
    from beat_weaver.pipeline.processor import compute_map_hash

    return compute_map_hash(folder)


def build_audio_manifest(raw_dirs: list[Path]) -> dict[str, str]:
    """Scan raw map folders and build a hash → audio file path mapping.

    Looks for Info.dat files and their referenced audio filenames.
    Falls back to scanning for common audio extensions.
    """
    manifest: dict[str, str] = {}

    for raw_dir in raw_dirs:
        raw_dir = Path(raw_dir)
        if not raw_dir.exists():
            logger.warning("Raw directory not found: %s", raw_dir)
            continue

        for info_file in raw_dir.rglob("Info.dat"):
            folder = info_file.parent
            song_hash = _hash_folder(folder)

            # Try to find audio file referenced in Info.dat
            audio_path = _find_audio_in_folder(folder, info_file)
            if audio_path:
                manifest[song_hash] = str(audio_path)

        # Also check for case-insensitive info.dat
        for info_file in raw_dir.rglob("info.dat"):
            if info_file.name == "Info.dat":
                continue  # already handled above
            folder = info_file.parent
            song_hash = _hash_folder(folder)
            if song_hash not in manifest:
                audio_path = _find_audio_in_folder(folder, info_file)
                if audio_path:
                    manifest[song_hash] = str(audio_path)

    logger.info("Built audio manifest: %d entries", len(manifest))
    return manifest


def _find_audio_in_folder(folder: Path, info_file: Path) -> Path | None:
    """Find the audio file in a map folder."""
    # Try parsing Info.dat for the audio filename
    try:
        import json as _json
        info = _json.loads(info_file.read_text(encoding="utf-8-sig"))
        # v2/v3: _songFilename, v4: audio.songFilename or song.songFilename
        audio_name = (
            info.get("_songFilename")
            or info.get("audio", {}).get("songFilename")
            or info.get("song", {}).get("songFilename")
        )
        if audio_name:
            audio_path = folder / audio_name
            if audio_path.exists():
                return audio_path
    except Exception:
        pass

    # Fallback: scan for common audio files
    for ext in _AUDIO_EXTENSIONS:
        for f in folder.glob(f"*{ext}"):
            return f
    return None


def save_manifest(manifest: dict[str, str], path: Path) -> None:
    """Save audio manifest to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def load_manifest(path: Path) -> dict[str, str]:
    """Load audio manifest from JSON."""
    return json.loads(Path(path).read_text(encoding="utf-8"))
