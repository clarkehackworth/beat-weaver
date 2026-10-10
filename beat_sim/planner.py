"""Turn one hand's swings into a joint-angle trajectory and per-swing feasibility."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from beat_sim.body import Q_REST, fk
from beat_sim.ik import REACH_TOL, solve
from beat_sim.swing import Swing, amplitude

SWING_HALF = 0.12  # seconds from approach pose to contact (and contact to follow-through)
HZ = 100.0


@dataclass
class SwingResult:
    swing: Swing
    q_approach: np.ndarray
    q_contact: np.ndarray
    q_follow: np.ndarray
    err: float  # blade-to-note distance at contact, metres
    swing_err: float  # worst hand-position error of approach/follow poses (clipped swing)

    @property
    def hit(self) -> bool:
        return self.err <= REACH_TOL


@dataclass
class Trajectory:
    hand: int
    results: list[SwingResult]
    t: np.ndarray = field(default_factory=lambda: np.zeros(0))
    q: np.ndarray = field(default_factory=lambda: np.zeros((0, 6)))  # (n, 6)
    tip: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))  # (n, 3)

    @property
    def qd(self) -> np.ndarray:
        return np.gradient(self.q, 1.0 / HZ, axis=0) if len(self.t) > 1 else np.zeros_like(self.q)

    @property
    def tip_vel(self) -> np.ndarray:
        return np.gradient(self.tip, 1.0 / HZ, axis=0) if len(self.t) > 1 else np.zeros((0, 3))

    @property
    def tip_speed(self) -> np.ndarray:
        return np.linalg.norm(self.tip_vel, axis=1)

    @property
    def tip_accel(self) -> np.ndarray:
        return np.linalg.norm(np.gradient(self.tip_vel, 1.0 / HZ, axis=0), axis=1) if len(self.t) > 1 else np.zeros(0)


def _min_jerk(tau: np.ndarray) -> np.ndarray:
    return 10 * tau**3 - 15 * tau**4 + 6 * tau**5


def plan(swings: list[Swing], hand: int) -> Trajectory:
    mirror = hand == 0
    q = Q_REST.copy()
    keys: list[tuple[float, np.ndarray]] = []
    results: list[SwingResult] = []
    for i, s in enumerate(swings):
        # Shrink the swing when notes are packed tighter than two full swings allow.
        gap_prev = s.t - swings[i - 1].t if i else np.inf
        gap_next = swings[i + 1].t - s.t if i + 1 < len(swings) else np.inf
        half = min(SWING_HALF, 0.4 * gap_prev, 0.4 * gap_next)
        amp = amplitude(min(gap_prev, gap_next))  # ponytail: amplitude and duration both shrink with the gap
        qc, ec = solve(q, mirror, point=s.point, axis=s.axis(0))
        if s.direction is None:
            qa, qf, es = qc, qc, 0.0
        else:
            h = fk(qc, mirror)["hilt"]
            qa, ea = solve(qc, mirror, hilt=s.hilt(-1, h, amp), axis=s.axis(-1, amp))
            qf, ef = solve(qc, mirror, hilt=s.hilt(1, h, amp), axis=s.axis(1, amp))
            es = max(ea, ef)
        q = qf
        results.append(SwingResult(s, qa, qc, qf, ec, es))
        # ponytail: contact pose is scored but not a keyframe, so the blade keeps moving through the note
        # instead of stopping on it (min-jerk has zero velocity at every keyframe).
        keys.append((s.t - half, qa))
        if s.direction is not None:
            keys.append((s.t + half, qf))

    traj = Trajectory(hand, results)
    if not keys:
        return traj
    keys.insert(0, (keys[0][0] - 1.0, Q_REST.copy()))
    t = np.arange(keys[0][0], keys[-1][0] + 1.0 / HZ, 1.0 / HZ)
    kt = np.array([k[0] for k in keys])
    kq = np.array([k[1] for k in keys])
    idx = np.clip(np.searchsorted(kt, t, side="right") - 1, 0, len(kt) - 2)
    tau = np.clip((t - kt[idx]) / np.maximum(kt[idx + 1] - kt[idx], 1e-9), 0, 1)
    qs = kq[idx] + _min_jerk(tau)[:, None] * (kq[idx + 1] - kq[idx])
    traj.t, traj.q = t, qs
    # ponytail: one fk per sample, ~100 µs each. Vectorise Rotation if a full song takes too long.
    traj.tip = np.array([fk(row, mirror)["tip"] for row in qs])
    return traj
