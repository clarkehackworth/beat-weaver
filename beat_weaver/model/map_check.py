"""Standalone validity and playability checks for Beat Saber map folders.

This is a dependency-light gate (no torch) over the JSON that ``exporter`` writes,
so it can run on a CPU box right after generation. It is used two ways:

  * ``beat-weaver validate <folder-or-dir>`` — human/CI check of finished map folders
  * ``batch_generate.merge_song`` — gate on each freshly merged song

Checks split into two severities:

  * ``structure``   — hard failures: the map would not load or is corrupt
                      (out-of-grid cells, bad enums, negative beats, duplicate notes,
                      missing/invalid difficulty files).
  * ``playability`` — soft warnings: things that load fine but feel wrong
                      (crossed hands, un-swipeable doubles, a difficulty ladder where a
                      higher level is not denser than the one below).

``_time`` in a v2 difficulty file is in **beats** (see ``schemas/v2.py``); all
beat-grid math here is on that unit.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeGuard

GRID_COLUMNS = 4   # lineIndex 0-3
GRID_ROWS = 3      # lineLayer 0-2
NOTE_COLORS = (0, 1)        # _type: 0=red (left), 1=blue (right)
BOMB_TYPE = 3               # _type 3 = bomb (valid; the generator never emits one)
VALID_TYPES = NOTE_COLORS + (BOMB_TYPE,)
VALID_CUTS = set(range(9))         # 0-8

# Display order + the canonical Info.dat rank for each difficulty name.
DIFF_ORDER = {"Easy": 1, "Normal": 2, "Hard": 3, "Expert": 4, "ExpertPlus": 5}
DIFF_RANK = {"Easy": 1, "Normal": 3, "Hard": 5, "Expert": 7, "ExpertPlus": 9}

# Each difficulty should be at least this much denser (NPS) than the one below it.
LADDER_RATIO = 1.10


def _is_num(v: object) -> TypeGuard[int | float]:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


@dataclass
class MapIssue:
    difficulty: str
    kind: str      # "structure" | "playability"
    message: str
    count: int = 1


@dataclass
class MapReport:
    folder: str | Path
    issues: list[MapIssue] = field(default_factory=list)

    def add(self, difficulty: str, kind: str, message: str, count: int = 1) -> None:
        self.issues.append(MapIssue(difficulty, kind, message, count))

    @property
    def structure_issues(self) -> list[MapIssue]:
        return [i for i in self.issues if i.kind == "structure"]

    @property
    def playability_issues(self) -> list[MapIssue]:
        return [i for i in self.issues if i.kind == "playability"]

    @property
    def ok(self) -> bool:
        """True when there are no hard structural failures (playability may warn)."""
        return not self.structure_issues

    def summary(self) -> str:
        n_struct = len(self.structure_issues)
        n_play = len(self.playability_issues)
        status = "OK" if self.ok else "FAIL"
        lines = [f"{self.folder}: {status} ({n_struct} structural, {n_play} playability)"]
        for i in self.issues:
            lines.append(f"  [{i.kind[:4]}] {i.difficulty}: {i.message}"
                         + (f" (x{i.count})" if i.count > 1 else ""))
        return "\n".join(lines)


# Per-hand effective BPM ceilings (favourites p90, see beat_sim/rhythm.py BANDS 'ebpm') with 10% headroom.
MAX_EBPM = {"Easy": 155, "Normal": 265, "Hard": 310, "Expert": 340, "ExpertPlus": 425}
REACTION_MS = (400, 1500)  # bsmg jump settings: Expert+ 500-600 ms up to Easy 1200 ms
# share of a hand's swings that repeat the previous cut inside 1.5 beats (favourites p90 per difficulty)
DOUBLE_DIRECTIONAL_RATE = {"Easy": 0.03, "Normal": 0.03, "Hard": 0.07, "Expert": 0.06, "ExpertPlus": 0.11}


_CUT_XY = {0: (0, 1), 1: (0, -1), 2: (-1, 0), 3: (1, 0), 4: (-1, 1), 5: (1, 1), 6: (-1, -1), 7: (1, -1)}


def _swing_checks(notes: list[dict], difficulty: str, bpm: float, add) -> None:
    """Per-hand speed and parity: effective BPM over 2 s windows and double directionals within 1.5 beats."""
    if bpm <= 0:
        return
    beat = 60.0 / bpm
    dd = swings_total = 0
    gaps: list[float] = []
    for color in (0, 1):
        hand = sorted((n for n in notes if n.get("_type") == color and _is_num(n.get("_time"))), key=lambda n: n["_time"])
        swings: list[dict] = []
        for n in hand:
            if not swings or (n["_time"] - swings[-1]["_time"]) >= 0.3:
                swings.append(n)
        for a, b in zip(swings, swings[1:]):
            g = (b["_time"] - a["_time"]) * beat
            if g <= 2.0:
                gaps.append(g)
            swings_total += 1
            ca, cb = a.get("_cutDirection"), b.get("_cutDirection")
            if b["_time"] - a["_time"] <= 1.5 and ca == cb and ca in range(8):
                dd += 1
    if gaps:
        gaps.sort()
        ebpm = 60.0 / gaps[max(0, int(len(gaps) * 0.1) - 1)]  # p90 of 60/gap
        cap = MAX_EBPM.get(difficulty)
        if cap and ebpm > cap:
            add(difficulty, "playability", f"effective BPM {ebpm:.0f} above {cap} for {difficulty}", 1)
    if swings_total and dd / swings_total > DOUBLE_DIRECTIONAL_RATE.get(difficulty, 0.06):
        add(difficulty, "playability", "double directionals (same cut twice by one hand within 1.5 beats)", dd)


def check_difficulty_notes(
    notes: list[dict],
    *,
    difficulty: str,
    add,
    bpm: float = 0.0,
) -> list[float]:
    """Run structural + playability checks over one difficulty's ``_notes``.

    ``add`` is ``MapReport.add``. Returns the list of note beat times (for NPS).
    """
    beats: list[float] = []
    n = len(notes)
    if n == 0:
        add(difficulty, "structure", "no notes")
        return beats

    bad_grid = bad_type = bad_cut = bad_time = bad_dup = off_grid = 0
    seen: set[tuple] = set()
    by_beat: dict[float, list[dict]] = {}

    for note in notes:
        t = note.get("_time")
        li = note.get("_lineIndex")
        ll = note.get("_lineLayer")
        ty = note.get("_type")
        cd = note.get("_cutDirection")

        if not _is_num(t) or t < 0:
            bad_time += 1
            continue
        grid_ok = _is_num(li) and 0 <= li < GRID_COLUMNS and _is_num(ll) and 0 <= ll < GRID_ROWS
        if not grid_ok:
            bad_grid += 1
        if ty not in VALID_TYPES:
            bad_type += 1
        if cd not in VALID_CUTS:
            bad_cut += 1
        # _time is in beats; every note must sit on the 1/16-note (0.0625 beat) grid.
        # This catches the unit bug where beats are written where seconds (or vice
        # versa) are expected: a seconds value is almost never a clean 0.0625 multiple.
        if abs(float(t) * 16.0 - round(float(t) * 16.0)) > 1e-3:
            off_grid += 1

        t_key = round(float(t), 6)
        key = (t_key, li, ll, ty)
        if key in seen:
            bad_dup += 1
        seen.add(key)

        beats.append(float(t))
        if grid_ok:  # malformed cells are already reported; keep them out of hand-pair math
            by_beat.setdefault(t_key, []).append(note)

    if bad_grid:  add(difficulty, "structure", "notes outside grid (lineIndex 0-3 / lineLayer 0-2)", bad_grid)
    if off_grid:  add(difficulty, "structure", "notes off the 1/16-note (0.0625-beat) grid — check _time unit", off_grid)
    if bad_type:  add(difficulty, "structure", "notes with invalid _type (not 0/1/3)", bad_type)
    if bad_cut:   add(difficulty, "structure", "notes with invalid _cutDirection (not 0-8)", bad_cut)
    if bad_time:  add(difficulty, "structure", "notes with missing/negative _time", bad_time)
    if bad_dup:   add(difficulty, "structure", "exact-duplicate notes (same beat/col/row/type)", bad_dup)

    # --- playability: same-beat hand interactions --------------------------
    # ``_type`` is the color: 0=red (left hand), 1=blue (right hand).
    # ``_lineIndex`` is the column (0-3), ``_lineLayer`` the row (0-2).
    crossed = 0
    wide = through = 0
    for beat_notes in by_beat.values():
        red_cols: list[int] = []
        blue_cols: list[int] = []
        for n in beat_notes:
            col = n.get("_lineIndex")
            if not _is_num(col):
                continue
            if n.get("_type") == 0:
                red_cols.append(int(col))
            elif n.get("_type") == 1:
                blue_cols.append(int(col))
        # crossed hands: a red (left) note on the right half + a blue (right) note
        # on the left half at the same beat forces the sabers to pass each other.
        if any(c >= 2 for c in red_cols) and any(c <= 1 for c in blue_cols):
            crossed += 1
        # cut-through: a note's blade path (1-2 cells along its cut) runs over the other colour's note
        for a in beat_notes:
            d = a.get("_cutDirection")
            if d not in _CUT_XY or not _is_num(a.get("_lineIndex")) or not _is_num(a.get("_lineLayer")):
                continue
            dx, dy = _CUT_XY[d]
            cells = {(a["_lineIndex"] + k * dx, a["_lineLayer"] + k * dy) for k in (1, 2)}
            if any(b.get("_type") not in (a.get("_type"), None) and (b.get("_lineIndex"), b.get("_lineLayer")) in cells
                   for b in beat_notes if b.get("_type") in (0, 1)):
                through += 1
        # wide doubles: >=2 notes of one color at this beat that one straight swipe
        # cannot cleanly cover (not same-row/adjacent and not same-col/adjacent).
        for color in (0, 1):
            side_notes = [n for n in beat_notes if n.get("_type") == color]
            if len(side_notes) < 2:
                continue
            cols = {n.get("_lineIndex") for n in side_notes if _is_num(n.get("_lineIndex"))}
            rows = {n.get("_lineLayer") for n in side_notes if _is_num(n.get("_lineLayer"))}
            cuts = {n.get("_cutDirection") for n in side_notes}
            # a tower (one column, vertical cuts) or window (one row, horizontal cuts) is one swipe along the line
            tower = len(cols) == 1 and cuts <= {0, 1, 8}
            window = len(rows) == 1 and cuts <= {2, 3, 8}
            # a diagonal stack: neighbouring cells along their shared diagonal cut (one swipe; the generator makes these)
            diag = len(side_notes) == 2 and len(cuts) == 1 and (c := next(iter(cuts))) in _CUT_XY and c >= 4 and \
                (side_notes[1].get("_lineIndex", 0) - side_notes[0].get("_lineIndex", 0),
                 side_notes[1].get("_lineLayer", 0) - side_notes[0].get("_lineLayer", 0)) in (_CUT_XY[c], tuple(-v for v in _CUT_XY[c]))
            if not tower and not window and not diag and \
               not (len(cols) == 1 and max(rows) - min(rows) <= 1) and \
               not (len(rows) == 1 and max(cols) - min(cols) <= 1):
                wide += 1

    if crossed:
        add(difficulty, "playability", "crossed hands (red on right + blue on left at same beat)", crossed)
    if wide:
        add(difficulty, "playability", "wide doubles (same-color notes one swipe cannot cover)", wide)
    if through:
        add(difficulty, "playability", "cut-through (a cut's path runs over the other colour's note)", through)
    _swing_checks(notes, difficulty, bpm, add)

    return beats


def _nps_from_beats(beats: list[float], bpm: float) -> float:
    if len(beats) < 2 or bpm <= 0:
        return 0.0
    span_beats = max(beats) - min(beats)
    span_sec = span_beats * 60.0 / bpm
    return len(beats) / span_sec if span_sec > 0 else 0.0


def check_difficulty_file(path: Path, *, difficulty: str, bpm: float, add) -> list[float]:
    """Load one difficulty .dat (v2) and check it. Returns note beats."""
    try:
        data = json.loads(path.read_text())
    except Exception as e:
        add(difficulty, "structure", f"unreadable/invalid JSON: {e}")
        return []
    if str(data.get("_version", "")).split(".")[0] not in ("2", "3", "4"):
        add(difficulty, "structure", f"unexpected _version {data.get('_version')!r}")
    notes = data.get("_notes", [])
    if not isinstance(notes, list):
        add(difficulty, "structure", "_notes is not a list")
        return []
    return check_difficulty_notes(notes, difficulty=difficulty, add=add, bpm=bpm)


def check_map_folder(folder: Path, *, check_ladder: bool = True) -> MapReport:
    """Validate one map folder (Info.dat + its difficulty files).

    Tolerant of a missing Info.dat (reports it) so a half-finished folder still
    yields a readable report instead of raising.
    """
    report = MapReport(folder=str(folder))
    add = report.add

    info_path = folder / "Info.dat"
    if not info_path.exists():
        add("Info", "structure", "Info.dat missing")
        return report

    try:
        info = json.loads(info_path.read_text())
    except Exception as e:
        add("Info", "structure", f"Info.dat invalid JSON: {e}")
        return report

    bpm = info.get("_beatsPerMinute")
    if not _is_num(bpm) or not (40 <= bpm <= 250):
        add("Info", "structure", f"suspicious _beatsPerMinute {bpm!r}")
        bpm = float(bpm) if _is_num(bpm) else 0.0

    song = info.get("_songFilename")
    if not song:
        add("Info", "structure", "no _songFilename")
    elif not (folder / song).exists():
        add("Info", "structure", f"song file {song!r} missing from folder")

    # Collect difficulties from the v2 _difficultyBeatmapSets layout.
    diffs: dict[str, str] = {}
    bmap_settings: dict[str, tuple[float, float]] = {}
    for bset in info.get("_difficultyBeatmapSets", []):
        for bmap in bset.get("_difficultyBeatmaps", []):
            d = bmap.get("_difficulty", "")
            if d:
                diffs[d] = bmap.get("_beatmapFilename", "")
                njs, off = bmap.get("_noteJumpMovementSpeed"), bmap.get("_noteJumpStartBeatOffset", 0)
                if _is_num(njs) and njs > 0 and _is_num(off):
                    bmap_settings[d] = (float(njs), float(off))
    if not diffs:
        add("Info", "structure", "no difficultyBeatmapSets/_difficultyBeatmaps entries")
        return report

    nps_by_diff: dict[str, float] = {}
    for d in sorted(diffs, key=lambda x: DIFF_ORDER.get(x, 0)):
        expected_rank = DIFF_RANK.get(d)
        # re-read rank from info for the rank check
        rank = None
        for bset in info.get("_difficultyBeatmapSets", []):
            for bmap in bset.get("_difficultyBeatmaps", []):
                if bmap.get("_difficulty") == d:
                    rank = bmap.get("_difficultyRank")
        if expected_rank is not None and rank is not None and rank != expected_rank:
            add(d, "structure", f"difficultyRank {rank} != expected {expected_rank} for {d}")
        if bpm and bmap_settings.get(d):
            njs, offset = bmap_settings[d]
            hjd = 4.0
            while njs * (60.0 / bpm) * hjd > 17.999:
                hjd /= 2
            reaction = max(hjd + offset, 0.25) * 60.0 / bpm * 1000
            if not REACTION_MS[0] <= reaction <= REACTION_MS[1]:
                add(d, "playability", f"reaction time {reaction:.0f} ms outside {REACTION_MS[0]}-{REACTION_MS[1]} ms (NJS {njs}, offset {offset})")
        fname = diffs[d]
        path = folder / fname
        if not path.exists():
            add(d, "structure", f"referenced file {fname!r} missing")
            continue
        beats = check_difficulty_file(path, difficulty=d, bpm=bpm, add=add)
        if beats:
            nps_by_diff[d] = _nps_from_beats(beats, bpm)

    if check_ladder:
        order = [d for d in ("Easy", "Normal", "Hard", "Expert", "ExpertPlus") if d in nps_by_diff]
        for lo, hi in zip(order, order[1:]):
            if nps_by_diff[hi] < LADDER_RATIO * nps_by_diff[lo]:
                add(hi, "playability",
                    f"ladder: {hi} {nps_by_diff[hi]:.2f} nps < {LADDER_RATIO}x {lo} {nps_by_diff[lo]:.2f} nps")
    return report


def check_paths(paths: list[Path | str]) -> list[MapReport]:
    """Validate one or more map folders (a folder, or a dir containing folders)."""
    folders: list[Path] = []
    for p in paths:
        p = Path(p).resolve()
        if (p / "Info.dat").exists():
            folders.append(p)
        elif p.is_dir():
            for sub in sorted(p.iterdir()):
                if sub.is_dir() and (sub / "Info.dat").exists():
                    folders.append(sub)
        else:
            folders.append(p)  # missing path / bare file: check_map_folder reports "Info.dat missing"
    return [check_map_folder(f) for f in folders]
