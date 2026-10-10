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
    broken = simulate(stream(140, 40, [1]))  # every cut points down
    assert broken.left.resets == broken.left.notes - 1
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
