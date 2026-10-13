import numpy as np
import pytest

pytest.importorskip("scipy")

from beat_sim import simulate  # noqa: E402
from beat_weaver.schemas.normalized import Note


def stream(bpm: float, n: int, dirs: list[int], x_left=1, x_right=2) -> list[Note]:
    """Alternating left/right notes, one per beat, cut directions cycling through `dirs`."""
    notes = []
    for i in range(n):
        hand = i % 2
        notes.append(Note(beat=i, time_seconds=i * 60.0 / bpm, x=x_left if hand == 0 else x_right, y=1,
                          color=hand, cut_direction=dirs[(i // 2) % len(dirs)]))
    return notes


def test_clean_stream_scores_high():
    r = simulate(stream(140, 40, [1, 0]))  # down, up, down, up per hand
    assert r.left.hit_rate == 1.0 and r.right.hit_rate == 1.0
    assert r.left.resets == 0 and r.right.resets == 0
    assert r.left.too_fast == 0 and r.right.too_fast == 0
    assert r.score > 85, r.to_dict()


def test_same_direction_stream_is_reset_heavy():
    clean = simulate(stream(140, 40, [1, 0]))
    broken = simulate(stream(240, 40, [1]))  # every cut points down, half a second apart: inside the reset window
    assert broken.left.resets == broken.left.notes - 1
    slow = simulate(stream(60, 40, [1]))  # two seconds apart: the hand has time to come back, legal resets
    assert slow.left.resets == 0
    assert broken.score < clean.score


def flail(gap: float, n: int = 40) -> list[Note]:
    """Both hands corner-to-corner with opposing diagonal cuts every `gap` seconds. Nobody plays this."""
    out = []
    for i in range(n):
        k = i % 2
        out.append(Note(beat=i, time_seconds=i * gap, x=0 if k == 0 else 3, y=0 if k == 0 else 2, color=1, cut_direction=6 if k == 0 else 5))
        out.append(Note(beat=i, time_seconds=i * gap, x=3 if k == 0 else 0, y=0 if k == 0 else 2, color=0, cut_direction=7 if k == 0 else 4))
    return out


def test_flailing_trips_speed_cap():
    r = simulate(flail(0.2))
    assert r.left.too_fast + r.right.too_fast > 40, r.to_dict()
    assert r.score < simulate(stream(140, 40, [1, 0])).score


def test_rhythm_bands_and_composite():
    from beat_sim import evaluate

    dense = evaluate(stream(140, 400, [1, 0]), 140.0)  # 2.3 nps, no streams: below Expert band
    assert not dense["rhythm"].in_band["nps"] and not dense["rhythm"].in_band["streams_per_min"]
    assert 0 <= dense["composite"] <= dense["rhythm"].score


def test_out_of_grid_notes_are_ignored_and_empty_map_is_fine():
    assert simulate([]).score == 100.0
    bad = [Note(beat=0, time_seconds=0, x=7, y=1, color=1, cut_direction=1)]
    assert simulate(bad).right.notes == 0


def test_trajectory_is_continuous():
    _, trajs = simulate(stream(140, 10, [1, 0]), return_trajectories=True)
    for tr in trajs:
        assert np.all(np.isfinite(tr.q))
        assert np.abs(np.diff(tr.q, axis=0)).max() < 0.5, "no teleporting joints between 10 ms samples"


def test_ladder_forces_rising_density_and_prefers_total_score():
    from pathlib import Path

    from beat_sim.mapper import pick_ladder

    c = lambda comp, nps: (comp, nps, Path("x"), {})
    cands = {
        "Hard": [c(89, 3.2), c(80, 2.5)],
        "Expert": [c(93, 3.3), c(79, 4.2)],  # best Expert is too close to best Hard
    }
    got = pick_ladder(cands)
    assert got["Expert"][1] >= 1.1 * got["Hard"][1], got
    assert (got["Hard"][0], got["Expert"][0]) in {(80, 93), (89, 79)}  # either feasible chain; the DP takes the larger total
    assert got["Hard"][0] + got["Expert"][0] == 173
    assert pick_ladder({"Easy": [c(70, 1.0)]})["Easy"][0] == 70


def test_mapper_expertplus_candidates_can_use_beat_focus():
    from beat_sim.mapper import flags_for

    from beat_sim.mapper import PRESET
    assert ("--beat-focus" in flags_for("ExpertPlus", 4.2, 0)) == PRESET["ExpertPlus"]["beat_focus"]
    assert "--seed 1" in flags_for("Expert", 3.3, 1)
    assert "--downmap Easy Normal" in flags_for("Hard", 2.0, 0, ("Easy", "Normal"))
    assert "--downmap" not in flags_for("Hard", 2.0, 0)
    assert "--drift-bias" in flags_for("Expert", 3.3, 1)


def test_merge_writes_song_metadata_and_cover(tmp_path):
    import json

    from beat_sim.mapper import merge

    def fake(diff):
        d = tmp_path / f"{diff}__x"
        d.mkdir(exist_ok=True)
        (d / f"{diff}.dat").write_text("{}")
        (d / "song.egg").write_bytes(b"x")
        entry = {"_difficulty": diff, "_difficultyRank": 1, "_beatmapFilename": f"{diff}.dat",
                 "_noteJumpMovementSpeed": 10, "_noteJumpStartBeatOffset": 0}
        (d / "Info.dat").write_text(json.dumps({
            "_songName": "lying", "_songAuthorName": "", "_levelAuthorName": "BeatWeaver AI", "_beatsPerMinute": 80.0,
            "_previewStartTime": 12, "_coverImageFilename": "",
            "_difficultyBeatmapSets": [{"_beatmapCharacteristicName": "Standard", "_difficultyBeatmaps": [entry]}]}))
        return d

    cover = tmp_path / "cover.png"
    cover.write_bytes(b"png")
    meta = {"_songName": "Lying From You", "_songAuthorName": "Linkin Park & Eminem", "_levelAuthorName": "Test Mapper",
            "_previewStartTime": 117.1}
    out = merge({"Expert": fake("Expert"), "Easy": fake("Easy")}, tmp_path / "out", meta, cover)
    info = json.loads((out / "Info.dat").read_text())
    assert (info["_songName"], info["_songAuthorName"], info["_levelAuthorName"]) == ("Lying From You", "Linkin Park & Eminem", "Test Mapper")
    assert info["_previewStartTime"] == 117.1 and info["_beatsPerMinute"] == 80.0
    assert info["_coverImageFilename"] == "cover.png" and (out / "cover.png").exists()
    assert [b["_difficulty"] for b in info["_difficultyBeatmapSets"][0]["_difficultyBeatmaps"]] == ["Easy", "Expert"]
    assert json.loads((merge({"Easy": fake("Easy")}, tmp_path / "o2", meta, None) / "Info.dat").read_text())["_coverImageFilename"] == ""


def test_mapper_name_resolution_order(tmp_path, monkeypatch):
    import json

    from beat_sim.meta import DEFAULT_MAPPER, mapper_name

    cfg = tmp_path / "config.json"
    monkeypatch.setenv("BW_CONFIG", str(cfg))
    monkeypatch.delenv("BW_MAPPER", raising=False)
    assert mapper_name() == DEFAULT_MAPPER  # no file
    cfg.write_text(json.dumps({"mapper_name": "From Config"}))
    assert mapper_name() == "From Config"
    monkeypatch.setenv("BW_MAPPER", "From Env")
    assert mapper_name() == "From Env"
    assert mapper_name("From Flag") == "From Flag"
    cfg.write_text("not json")
    monkeypatch.delenv("BW_MAPPER")
    assert mapper_name() == DEFAULT_MAPPER  # broken config falls back instead of crashing


def _jpeg(w=160, h=90):
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (w, h), (200, 30, 30)).save(buf, "JPEG")
    return buf.getvalue()


def test_web_cover_prefers_the_result_closest_to_the_file_name(monkeypatch):
    import beat_sim.meta as meta

    def page(*items):
        return "".join(f'"videoRenderer":{{"videoId":"{vid}","title":{{"runs":[{{"text":"{title}"}}]}},"ownerText":{{"runs":[{{"text":"chan"}}]}}' for vid, title in items).encode()

    html = page(("LIVEVIDEO11", "LISA Performs SaWaDiKa | 2026 VMAs"),
                ("COVERVIDEO1", "LISA - SaWaDiKa (Cover by Someone)"),
                ("OFFICIAL111", "LISA - SaWaDiKa (Official Music Video)"))
    asked = []

    def fake_get(url, timeout=10.0):
        asked.append(url)
        return html if "youtube.com/results" in url else _jpeg()

    monkeypatch.setattr(meta, "_get", fake_get)
    data, src = meta.youtube_cover("SaWaDiKa", "LISA", hint="LISA - SaWaDiKa (Official Music Video)")
    assert "OFFICIAL111" in src and "OFFICIAL111" in asked[-1], (src, asked)
    assert all("COVERVIDEO1" not in a for a in asked), "cover versions are never chosen"
    assert "LISA+-+SaWaDiKa" in asked[0] or "LISA%20-%20SaWaDiKa" in asked[0], "searches the file name, not just title and artist"


def test_web_cover_needs_a_match_and_falls_back_to_itunes_then_to_nothing(monkeypatch, tmp_path):
    import json

    import beat_sim.meta as meta

    unrelated = '"videoRenderer":{"videoId":"ZZZZZZZZZZZ","title":{"runs":[{"text":"Cooking pasta"}]}'.encode()
    itunes = json.dumps({"results": [{"trackName": "Gasoline", "artistName": "Halsey", "collectionName": "Badlands",
                                      "artworkUrl100": "https://x/100x100bb.jpg"}]}).encode()

    def fake_get(url, timeout=10.0):
        if "youtube.com/results" in url:
            return unrelated
        if "itunes.apple.com" in url:
            return itunes
        return _jpeg(300, 300)

    monkeypatch.setattr(meta, "_get", fake_get)
    name, src = meta.extract_cover(tmp_path / "x.mp3", tmp_path, None, "Gasoline", "Halsey")
    assert name == "cover.jpg" and "iTunes" in src
    from PIL import Image

    assert Image.open(tmp_path / "cover.jpg").size == (512, 512), "always square, 512 px"

    def offline(url, timeout=10.0):
        raise OSError("no network")

    monkeypatch.setattr(meta, "_get", offline)
    assert meta.extract_cover(tmp_path / "x.mp3", tmp_path / "none", None, "Gasoline", "Halsey")[0] == ""
    assert meta.extract_cover(tmp_path / "x.mp3", tmp_path, None, "Gasoline", "Halsey", web=False)[0] == ""


def test_youtube_thumbnail_is_centre_cropped_to_a_square():
    import io

    from PIL import Image

    from beat_sim.meta import _square_jpeg

    out = Image.open(io.BytesIO(_square_jpeg(_jpeg(1280, 720))))
    assert out.size == (512, 512)
