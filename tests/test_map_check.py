"""Tests for the map-folder validity + playability gate (beat_weaver.model.map_check)."""

import json
from pathlib import Path

import pytest

from beat_weaver.model.map_check import (
    check_difficulty_notes,
    check_map_folder,
)


def _note(t: float, x: int = 0, y: int = 1, color: int = 0, cut: int = 1) -> dict:
    return {"_time": t, "_lineIndex": x, "_lineLayer": y, "_type": color, "_cutDirection": cut}


def _collect(notes: list[dict]) -> list:
    """Run check_difficulty_notes and return the issues it added."""
    issues = []
    add = lambda diff, kind, msg, count=1: issues.append((diff, kind, msg, count))
    check_difficulty_notes(notes, difficulty="T", add=add)
    return issues


def _kinds(issues: list) -> set:
    return {(k, m) for _, k, m, _ in issues}


class TestStructuralChecks:
    def test_clean_notes_pass(self):
        notes = [_note(4.0), _note(4.0625, x=3, color=1), _note(8.0, x=1, y=2)]
        issues = _collect(notes)
        assert issues == []

    def test_out_of_grid_flagged(self):
        issues = _collect([_note(4.0, x=5), _note(4.0, x=-1), _note(4.0, y=3)])
        assert any("outside grid" in m for k, m in _kinds(issues))

    def test_invalid_type_flagged(self):
        issues = _collect([_note(4.0, color=2)])
        assert any("invalid _type" in m for k, m in _kinds(issues))

    def test_bomb_type_is_valid(self):
        # _type 3 is a bomb — legal in v2, not an error.
        issues = _collect([_note(4.0, color=3)])
        assert not any("invalid _type" in m for k, m in _kinds(issues))

    def test_invalid_cut_direction_flagged(self):
        issues = _collect([_note(4.0, cut=9)])
        assert any("invalid _cutDirection" in m for k, m in _kinds(issues))

    def test_negative_time_flagged(self):
        issues = _collect([_note(-1.0)])
        assert any("_time" in m for k, m in _kinds(issues))

    def test_missing_time_flagged(self):
        issues = _collect([{"_lineIndex": 0, "_lineLayer": 0, "_type": 0, "_cutDirection": 1}])
        assert any("_time" in m for k, m in _kinds(issues))

    def test_duplicate_notes_flagged(self):
        notes = [_note(4.0, x=1, y=1), _note(4.0, x=1, y=1)]  # identical
        issues = _collect(notes)
        assert any("duplicate" in m for k, m in _kinds(issues))

    def test_empty_notes_flagged(self):
        issues = _collect([])
        assert any("no notes" in m for k, m in _kinds(issues))


def test_malformed_cell_with_duplicate_does_not_crash():
    """Non-numeric _lineLayer used to empty the row set and crash max()."""
    bad = {**_note(4.0), "_lineLayer": "x"}
    issues = _collect([bad, dict(bad)])
    assert any("outside grid" in m for _, _, m, _ in issues)


class TestBeatGrid:
    """The unit-mismatch guard: _time must be on the 0.0625-beat (1/16 note) grid."""

    def test_on_grid_passes(self):
        # 0.0625, 0.125, 4.0, 12.75 are all clean 1/16-note multiples.
        issues = _collect([_note(0.0625), _note(0.125), _note(4.0), _note(12.75)])
        assert not any("grid" in m for k, m in _kinds(issues))

    def test_off_grid_flagged(self):
        # 0.07 and 4.1 are NOT 1/16-note multiples — a seconds-in-beats unit bug.
        issues = _collect([_note(0.07), _note(4.1)])
        assert any("grid" in m and "_time unit" in m for k, m in _kinds(issues))

    def test_seconds_written_as_beats_flagged(self):
        # A realistic song time of ~4.13s written into _time as if it were beats.
        issues = _collect([_note(4.13), _note(8.27), _note(12.4)])
        assert any("_time unit" in m for k, m in _kinds(issues))


class TestPlayability:
    def test_crossed_hands_flagged(self):
        # Red (left) on the right + blue (right) on the left at the same beat.
        notes = [_note(4.0, x=3, color=0), _note(4.0, x=0, color=1)]
        issues = _collect(notes)
        assert any("crossed hands" in m for k, m in _kinds(issues))

    def test_same_side_no_cross(self):
        # Both on the left half — not crossed.
        notes = [_note(4.0, x=0, color=0), _note(4.0, x=1, color=1)]
        issues = _collect(notes)
        assert not any("crossed hands" in m for k, m in _kinds(issues))

    def test_wide_double_flagged(self):
        # Two red notes far apart on the same row — one swipe can't cover both.
        notes = [_note(4.0, x=0, color=0), _note(4.0, x=3, color=0)]
        issues = _collect(notes)
        assert any("wide doubles" in m for k, m in _kinds(issues))

    def test_swipable_double_not_flagged(self):
        # Adjacent columns, same row — a clean swipe.
        notes = [_note(4.0, x=0, y=1, color=0), _note(4.0, x=1, y=1, color=0)]
        issues = _collect(notes)
        assert not any("wide doubles" in m for k, m in _kinds(issues))

    def test_stacked_double_not_flagged(self):
        # Same column, adjacent rows — a vertical swipe.
        notes = [_note(4.0, x=1, y=0, color=0), _note(4.0, x=1, y=1, color=0)]
        issues = _collect(notes)
        assert not any("wide doubles" in m for k, m in _kinds(issues))


class TestFolderChecks:
    @pytest.fixture
    def good_folder(self, tmp_path: Path) -> Path:
        folder = tmp_path / "Song"
        folder.mkdir()
        (folder / "song.ogg").write_bytes(b"fake-ogg")
        info = {
            "_version": "2.0.0",
            "_songName": "Song",
            "_beatsPerMinute": 120.0,
            "_songFilename": "song.ogg",
            "_difficultyBeatmapSets": [{
                "_beatmapCharacteristicName": "Standard",
                "_difficultyBeatmaps": [
                    {"_difficulty": "Easy", "_difficultyRank": 1, "_beatmapFilename": "Easy.dat",
                     "_noteJumpMovementSpeed": 10, "_noteJumpStartBeatOffset": 0},
                    {"_difficulty": "Expert", "_difficultyRank": 7, "_beatmapFilename": "Expert.dat",
                     "_noteJumpMovementSpeed": 16, "_noteJumpStartBeatOffset": 0},
                ],
            }],
        }
        (folder / "Info.dat").write_text(json.dumps(info))
        # Easy: sparse (1.0 nps) so Expert's 4.0 nps clears the ladder ratio.
        (folder / "Easy.dat").write_text(json.dumps({
            "_version": "2.0.0", "_notes": [_note(i * 4.0) for i in range(4)]}))
        (folder / "Expert.dat").write_text(json.dumps({
            "_version": "2.0.0", "_notes": [_note(i * 1.0) for i in range(4)]}))
        return folder

    def test_clean_folder_ok(self, good_folder: Path):
        report = check_map_folder(good_folder)
        assert report.ok
        assert report.structure_issues == []
        # 4 beats at 1.0-beat spacing over 3 beats = 4 notes / 1.5s = 2.67 nps (Expert)
        # vs Easy 4 notes over 12 beats = 2.0s = 2.0 nps -> ratio 1.33 > 1.10, no ladder warn.
        assert not any("ladder" in i.message for i in report.playability_issues)

    def test_missing_info_dat(self, tmp_path: Path):
        empty = tmp_path / "NoInfo"
        empty.mkdir()
        report = check_map_folder(empty)
        assert not report.ok
        assert any("Info.dat missing" in i.message for i in report.structure_issues)

    def test_missing_difficulty_file(self, good_folder: Path):
        (good_folder / "Expert.dat").unlink()
        report = check_map_folder(good_folder)
        assert not report.ok
        assert any("missing" in i.message for i in report.structure_issues)

    def test_wrong_rank_flagged(self, good_folder: Path):
        info = json.loads((good_folder / "Info.dat").read_text())
        info["_difficultyBeatmapSets"][0]["_difficultyBeatmaps"][0]["_difficultyRank"] = 9
        (good_folder / "Info.dat").write_text(json.dumps(info))
        report = check_map_folder(good_folder)
        assert any("difficultyRank" in i.message for i in report.structure_issues)

    def test_ladder_violation_flagged(self, good_folder: Path):
        # Make Expert less dense than Easy -> ladder warning.
        (good_folder / "Expert.dat").write_text(json.dumps({
            "_version": "2.0.0", "_notes": [_note(i * 4.0) for i in range(4)]}))
        report = check_map_folder(good_folder)
        assert any("ladder" in i.message for i in report.playability_issues)

    def test_off_grid_folder_fails_structure(self, good_folder: Path):
        # Simulate the unit bug: seconds written into _time.
        (good_folder / "Expert.dat").write_text(json.dumps({
            "_version": "2.0.0", "_notes": [_note(4.13), _note(8.27)]}))
        report = check_map_folder(good_folder)
        assert not report.ok
        assert any("_time unit" in i.message for i in report.structure_issues)

    def test_bomb_in_folder_is_not_a_failure(self, good_folder: Path):
        notes = [_note(4.0), _note(4.0625, x=1, color=0), _note(8.0, color=3)]
        (good_folder / "Expert.dat").write_text(json.dumps({"_version": "2.0.0", "_notes": notes}))
        report = check_map_folder(good_folder)
        assert not any("invalid _type" in i.message for i in report.structure_issues)


class TestFixtureIntegration:
    def test_project_fixture_folder(self):
        fixture = Path(__file__).parent / "fixtures" / "v2_map"
        report = check_map_folder(fixture)
        # The fixture is a valid v2 map; it must not be flagged structurally corrupt.
        # (It contains a bomb _type=3 note and a wall — both legal.)
        assert not any("invalid _type" in i.message for i in report.structure_issues)


def test_swing_checks_flag_speed_reaction_and_double_directionals(tmp_path):
    import json
    from beat_weaver.model.map_check import check_map_folder
    bpm = 120.0
    # Easy map: one hand hitting every 1/4 beat (eBPM 480) and always cutting down
    notes = [{"_time": i * 0.25, "_lineIndex": 1, "_lineLayer": 0, "_type": 0, "_cutDirection": 1} for i in range(80)]
    (tmp_path / "Easy.dat").write_text(json.dumps({"_version": "2.0.0", "_notes": notes, "_obstacles": [], "_events": []}))
    (tmp_path / "song.ogg").write_bytes(b"")
    info = {"_version": "2.0.0", "_beatsPerMinute": bpm, "_songFilename": "song.ogg", "_difficultyBeatmapSets": [{
        "_beatmapCharacteristicName": "Standard", "_difficultyBeatmaps": [{
            "_difficulty": "Easy", "_difficultyRank": 1, "_beatmapFilename": "Easy.dat",
            "_noteJumpMovementSpeed": 10, "_noteJumpStartBeatOffset": -3.6}]}]}
    (tmp_path / "Info.dat").write_text(json.dumps(info))
    msgs = [i.message for i in check_map_folder(tmp_path).issues]
    assert any("effective BPM" in m for m in msgs)
    assert any("double directionals" in m for m in msgs)
    assert any("reaction time" in m for m in msgs)


def test_cut_through_flagged():
    from beat_weaver.model.map_check import check_difficulty_notes, MapReport
    rep = MapReport(folder="x")
    notes = [{"_time": 1.0, "_lineIndex": 3, "_lineLayer": 1, "_type": 1, "_cutDirection": 0},
             {"_time": 1.0, "_lineIndex": 3, "_lineLayer": 2, "_type": 0, "_cutDirection": 0}]
    check_difficulty_notes(notes, difficulty="Expert", add=rep.add, bpm=120.0)
    assert any("cut-through" in i.message for i in rep.issues)
