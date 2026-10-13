"""Export token sequences to playable v2 Beat Saber map folders."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from beat_weaver.model.tokenizer import decode_tokens
from beat_weaver.schemas.normalized import Note

# Jump settings per difficulty: NJS and the reaction time the player gets, from the favourites' medians
# (bsmg.wiki intermediate-mapping "Jump Settings": Expert+ 500-600 ms, Easy/Normal 750-1200 ms, NJS 16 baseline).
_NJS_TABLE = {"Easy": 10, "Normal": 11, "Hard": 12, "Expert": 15, "ExpertPlus": 18}
_REACTION_MS = {"Easy": 1200, "Normal": 1050, "Hard": 900, "Expert": 750, "ExpertPlus": 600}


def jump_settings(difficulty: str, bpm: float) -> tuple[float, float]:
    """(NJS, start beat offset) giving this difficulty its target reaction time at this BPM.

    The game's half jump duration starts at 4 beats and halves while the jump distance exceeds 18 m,
    then adds the offset (floor 0.25 beats). Reaction time = HJD in seconds, so without an offset it
    swings with BPM; the offset pins it.
    """
    njs = _NJS_TABLE.get(difficulty, 16)
    beat = 60.0 / bpm
    hjd = 4.0
    while njs * beat * hjd > 17.999:
        hjd /= 2
    target = _REACTION_MS.get(difficulty, 700) / 1000.0 / beat
    offset = round(max(target, 0.25) - hjd, 2)
    return njs, offset


_DIFFICULTY_RANK = {
    "Easy": 1,
    "Normal": 3,
    "Hard": 5,
    "Expert": 7,
    "ExpertPlus": 9,
}


def _export_audio(audio_path: Path, output_dir: Path) -> str:
    """Place the song in the map folder as Ogg Vorbis (the only format Beat Saber plays)."""
    import subprocess
    if audio_path.suffix.lower() in (".ogg", ".egg"):
        dest = output_dir / f"song{audio_path.suffix}"
        shutil.copy2(audio_path, dest)
        return dest.name
    dest = output_dir / "song.egg"
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-y", "-i", str(audio_path),
         "-vn", "-c:a", "libvorbis", "-q:a", "6", "-f", "ogg", str(dest)],
        check=True,
    )
    return dest.name


def _build_info_dat(
    song_name: str,
    bpm: float,
    difficulty: str | list[str],
    audio_filename: str = "song.ogg",
) -> dict:
    """Build a v2 Info.dat structure listing one difficulty, or several (ordered by rank)."""
    diffs = [difficulty] if isinstance(difficulty, str) else sorted(difficulty, key=lambda d: _DIFFICULTY_RANK.get(d, 7))
    beatmaps = []
    for d in diffs:
        njs, offset = jump_settings(d, bpm)
        beatmaps.append({
            "_difficulty": d,
            "_difficultyRank": _DIFFICULTY_RANK.get(d, 7),
            "_beatmapFilename": f"{d}.dat",
            "_noteJumpMovementSpeed": njs,
            "_noteJumpStartBeatOffset": offset,
        })

    return {
        "_version": "2.0.0",
        "_songName": song_name,
        "_songSubName": "",
        "_songAuthorName": "",
        "_levelAuthorName": "BeatWeaver AI",
        "_beatsPerMinute": bpm,
        "_songTimeOffset": 0,
        "_shuffle": 0,
        "_shufflePeriod": 0.5,
        "_previewStartTime": 12,
        "_previewDuration": 10,
        "_songFilename": audio_filename,
        "_coverImageFilename": "",
        "_environmentName": "DefaultEnvironment",
        "_difficultyBeatmapSets": [
            {
                "_beatmapCharacteristicName": "Standard",
                "_difficultyBeatmaps": beatmaps,
            }
        ],
    }


def _build_difficulty_dat(notes: list, version: str = "2.0.0", events: list[dict] | None = None) -> dict:
    """Build a v2 difficulty .dat structure from decoded notes (and optional lighting events)."""
    v2_notes = []
    for note in notes:
        v2_notes.append({
            "_time": note.beat,
            "_lineIndex": note.x,
            "_lineLayer": note.y,
            "_type": note.color,
            "_cutDirection": note.cut_direction,
        })

    return {
        "_version": version,
        "_notes": sorted(v2_notes, key=lambda n: n["_time"]),
        "_obstacles": [],
        "_events": events or [],
    }


def export_map(
    token_ids: list[int],
    bpm: float,
    song_name: str,
    audio_path: Path,
    output_dir: Path,
    difficulty: str = "Expert",
) -> Path:
    """Export a token sequence to a playable v2 Beat Saber map folder.

    Args:
        token_ids: Generated token sequence.
        bpm: Song BPM.
        song_name: Display name for the song.
        audio_path: Path to the audio file.
        output_dir: Where to create the map folder.
        difficulty: Difficulty name.

    Returns:
        Path to the created map folder.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Decode tokens to notes
    notes = decode_tokens(token_ids, bpm)

    # Determine audio filename — copy to folder as song.ogg/.egg
    audio_filename = _export_audio(Path(audio_path), output_dir)

    # Write Info.dat
    info = _build_info_dat(song_name, bpm, difficulty, audio_filename)
    (output_dir / "Info.dat").write_text(json.dumps(info, indent=2), encoding="utf-8")

    # Write difficulty file
    diff_dat = _build_difficulty_dat(notes)
    (output_dir / f"{difficulty}.dat").write_text(json.dumps(diff_dat, indent=2), encoding="utf-8")

    return output_dir


def export_notes(
    notes: list[Note] | dict[str, list[Note]],
    bpm: float,
    song_name: str,
    audio_path: Path,
    output_dir: Path,
    difficulty: str = "Expert",
    events: dict[str, list[dict]] | None = None,
) -> Path:
    """Export a note list to a playable v2 Beat Saber map folder.

    Unlike export_map() which takes token IDs, this takes pre-decoded notes
    (e.g. from windowed generation where notes have already been merged).

    Args:
        notes: List of Note objects, or {difficulty: notes} for a folder with several difficulties.
        bpm: Song BPM.
        song_name: Display name for the song.
        audio_path: Path to the audio file.
        output_dir: Where to create the map folder.
        difficulty: Difficulty name.
        events: Optional {difficulty: v2 lighting events} (see ``lighting.light_events``).

    Returns:
        Path to the created map folder.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    by_diff = notes if isinstance(notes, dict) else {difficulty: notes}
    audio_filename = _export_audio(Path(audio_path), output_dir)
    info = _build_info_dat(song_name, bpm, list(by_diff), audio_filename)
    (output_dir / "Info.dat").write_text(json.dumps(info, indent=2), encoding="utf-8")
    for d, ns in by_diff.items():
        dat = _build_difficulty_dat(ns, events=(events or {}).get(d))
        (output_dir / f"{d}.dat").write_text(json.dumps(dat, indent=2), encoding="utf-8")
    return output_dir
