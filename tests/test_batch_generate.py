import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


def test_merge_song_writes_each_difficultys_jump_settings(tmp_path, monkeypatch):
    # merge_song once wrote offset 0 for every level, leaving 140-BPM Easy at 1.7 s reaction time
    import batch_generate as b
    from beat_weaver.model.exporter import jump_settings
    from beat_weaver.model.map_check import REACTION_MS, check_paths

    import beat_sim.meta as meta
    monkeypatch.setattr(meta, "song_meta", lambda *a, **k: {"_songName": "Sample Song", "_songAuthorName": "Example Band"})
    monkeypatch.setattr(meta, "extract_cover", lambda *a, **k: ("", None))
    dats = {}
    for d in ("Easy", "Normal", "Hard"):
        src = tmp_path / f"src__{d}"
        src.mkdir()
        (src / "Info.dat").write_text(json.dumps({"_beatsPerMinute": 140.0, "_songFilename": "song.egg"}))
        (src / "song.egg").write_bytes(b"OggS")
        notes = [{"_time": float(t), "_lineIndex": 1 + t % 2, "_lineLayer": 0, "_type": t % 2, "_cutDirection": 1} for t in range(8, 200, 2)]
        (src / f"{d}.dat").write_text(json.dumps({"_version": "2.0.0", "_notes": notes, "_obstacles": [], "_events": []}))
        dats[d] = src
    orig = tmp_path / "Example Band - Sample Song.mp3"
    orig.write_bytes(b"")
    out = tmp_path / "out"
    b.merge_song(orig, out, dats, "Test Mapper", web_cover=False)
    info = json.loads((out / "Info.dat").read_text())
    got = {m["_difficulty"]: (m["_noteJumpMovementSpeed"], m["_noteJumpStartBeatOffset"]) for m in info["_difficultyBeatmapSets"][0]["_difficultyBeatmaps"]}
    assert got == {d: jump_settings(d, 140.0) for d in dats}
    assert not any("reaction time" in i.message for i in check_paths([out])[0].playability_issues), REACTION_MS
