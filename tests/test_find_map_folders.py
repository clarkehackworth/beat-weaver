"""process must discover maps whatever the case of their info file."""
import os

from beat_weaver.cli import _find_map_folders


def _map(root, name, info):
    d = root / name
    d.mkdir(parents=True)
    (d / info).write_text("{}")
    return d


def test_finds_both_cases_and_nested_folders(tmp_path):
    a = _map(tmp_path, "upper", "Info.dat")
    b = _map(tmp_path, "lower", "info.dat")
    c = _map(tmp_path / "nested" / "deeper", "m", "info.dat")
    _map(tmp_path, "no_info", "Expert.dat")           # a folder without an info file is not a map
    assert _find_map_folders(tmp_path) == sorted([a, b, c])


def test_folder_with_both_spellings_counted_once(tmp_path):
    d = _map(tmp_path, "both", "Info.dat")
    (d / "info.dat").write_text("{}")                  # case-sensitive filesystem: two files
    assert _find_map_folders(tmp_path) == [d]


def test_symlinked_directories_are_not_followed(tmp_path):
    real = tmp_path / "real"
    _map(real, "m", "Info.dat")
    raw = tmp_path / "raw"
    raw.mkdir()
    os.symlink(real, raw / "linked")
    assert _find_map_folders(raw) == []                # same as rglob; copy/hardlink instead


def test_old_exact_match_would_miss_lowercase(tmp_path):
    _map(tmp_path, "lower", "info.dat")
    assert list(tmp_path.rglob("Info.dat")) == []      # the original behaviour this replaces
    assert len(_find_map_folders(tmp_path)) == 1
