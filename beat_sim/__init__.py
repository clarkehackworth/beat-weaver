"""Rates a note list by whether a human body can play it and whether it looks like maps the user likes.

    from beat_sim import evaluate
    ev = evaluate(beatmap.notes, bpm, "Expert")
    ev["composite"]            # 0..100, the number to rank candidate maps by
    ev["rhythm"].table()       # density / streams / layout vs favourite-map bands
    ev["body"].left.too_fast   # kinematic simulator detail
"""

from __future__ import annotations

from pathlib import Path

from beat_sim.metrics import Report, combine, score_hand
from beat_sim.planner import Trajectory, plan
from beat_sim.rhythm import RhythmReport
from beat_sim.rhythm import score as rhythm_score
from beat_sim.swing import swings_for_hand
from beat_weaver.schemas.normalized import Note


def simulate(notes: list[Note], return_trajectories: bool = False) -> Report | tuple[Report, list[Trajectory]]:
    """Kinematic body simulation only."""
    trajs = [plan(swings_for_hand(notes, hand), hand) for hand in (0, 1)]
    report = combine(score_hand(trajs[0]), score_hand(trajs[1]))
    return (report, trajs) if return_trajectories else report


def composite(rhythm: RhythmReport, body: Report) -> float:
    """Rhythm score is the grade; the body sim only takes points off for the inhuman."""
    n = max(body.left.notes + body.right.notes, 1)
    miss = (n - body.left.hits - body.right.hits) / n
    fast = (body.left.too_fast + body.right.too_fast) / n
    resets = (body.left.resets + body.right.resets) / n
    # ponytail: linear penalties, tuned so the current maps lose ~5 points. Refit when playtests disagree.
    return float(max(0.0, rhythm.score - 100 * miss - 50 * fast - 30 * resets))


def evaluate(notes: list[Note], bpm: float, difficulty: str = "Expert", return_trajectories: bool = False,
             audio: "Path | None" = None) -> dict:
    """`audio` (song file) enables the music-following metrics; without it they are left out of the score."""
    body, trajs = simulate(notes, return_trajectories=True)
    extra = None
    if audio is not None:
        from beat_sim.music import envelope, stats as music_stats

        extra = music_stats(notes, envelope(audio), bpm)
    rhythm = rhythm_score(notes, bpm, difficulty, extra)
    out = {"composite": composite(rhythm, body), "rhythm": rhythm, "body": body}
    if return_trajectories:
        out["trajectories"] = trajs
    return out


def to_dict(ev: dict) -> dict:
    return {"composite": ev["composite"], "rhythm": ev["rhythm"].to_dict(), "body": ev["body"].to_dict()}


__all__ = ["simulate", "evaluate", "composite", "to_dict", "Report", "RhythmReport", "Trajectory"]
