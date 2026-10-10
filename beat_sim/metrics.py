"""Score a planned trajectory: hits, speed, body stress, flow."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from beat_sim.body import QD_MAX, extension, fk, limit_stress
from beat_sim.planner import Trajectory

# ponytail: caps are the p99 of 24k transitions in the user's favourite maps (beat_sim.calibrate), not physiology.
# The planner overshoots on real maps (IK picks far-apart joint solutions for nearby hand positions), so absolute
# human limits flag half of all good transitions. A stronger previous-pose prior in ik.W_SMOOTH is the upgrade path.
TIP_SPEED_MAX = 57.0  # m/s, p99 of favourites (p50 16, p90 32)
TIP_ACCEL_MAX = 3340.0  # m/s^2, p99 of favourites (p50 345, p90 2600)
QD_RATIO_MAX = 4.9  # joint speed / body.QD_MAX, p99 of favourites
STRESS_HIGH = 1.0  # limit_stress above this = a joint pinned at its range
CROSS_X = 0.6  # metres past the midline that counts as a crossover: the far column only (favourites do near-column 18% of the time)


@dataclass
class HandReport:
    hand: int
    notes: int
    hits: int
    too_fast: int  # notes whose preceding travel exceeded a joint speed, tip speed, or tip acceleration cap
    resets: int  # consecutive swings in the same direction (parity break)
    crossovers: int  # notes in the far column for this hand
    stress_mean: float
    stress_peak: float
    extension_mean: float
    tip_speed_p95: float
    tip_speed_max: float

    @property
    def hit_rate(self) -> float:
        return self.hits / self.notes if self.notes else 1.0


@dataclass
class Report:
    left: HandReport
    right: HandReport
    score: float  # 0..100, higher is more playable

    def to_dict(self) -> dict:
        return {"score": self.score, "left": asdict(self.left), "right": asdict(self.right)}


def score_hand(traj: Trajectory) -> HandReport:
    rs = traj.results
    n = len(rs)
    if n == 0:
        return HandReport(traj.hand, 0, 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0)
    mirror = traj.hand == 0
    qd_ratio = np.abs(traj.qd) / QD_MAX if n else np.zeros((0, 6))
    tip_speed, tip_accel = traj.tip_speed, traj.tip_accel

    too_fast = resets = crossovers = 0
    stresses, exts = [], []
    for i, r in enumerate(rs):
        # travel window: from previous follow-through to this approach
        t0 = rs[i - 1].swing.t if i else traj.t[0]
        win = (traj.t >= t0) & (traj.t <= r.swing.t)
        if win.any() and (qd_ratio[win].max() > QD_RATIO_MAX or tip_speed[win].max() > TIP_SPEED_MAX or tip_accel[win].max() > TIP_ACCEL_MAX):
            too_fast += 1
        if i and r.swing.direction is not None and rs[i - 1].swing.direction is not None:
            if float(r.swing.direction @ rs[i - 1].swing.direction) > 0.5:
                resets += 1
        for q in (r.q_approach, r.q_contact, r.q_follow):
            stresses.append(limit_stress(q))
            pose = fk(q, mirror)
            exts.append(extension(pose))
        x = r.swing.point[0]
        if (mirror and x > CROSS_X) or (not mirror and x < -CROSS_X):
            crossovers += 1

    return HandReport(
        hand=traj.hand,
        notes=n,
        hits=sum(r.hit for r in rs),
        too_fast=too_fast,
        resets=resets,
        crossovers=crossovers,
        stress_mean=float(np.mean(stresses)),
        stress_peak=float(np.max(stresses)),
        extension_mean=float(np.mean(exts)),
        tip_speed_p95=float(np.percentile(tip_speed, 95)) if len(tip_speed) else 0.0,
        tip_speed_max=float(tip_speed.max()) if len(tip_speed) else 0.0,
    )


def combine(left: HandReport, right: HandReport) -> Report:
    n = left.notes + right.notes
    if n == 0:
        return Report(left, right, 100.0)
    miss = (n - left.hits - right.hits) / n
    fast = (left.too_fast + right.too_fast) / n
    reset = (left.resets + right.resets) / n
    cross = (left.crossovers + right.crossovers) / n
    stress = (left.stress_mean * left.notes + right.stress_mean * right.notes) / n
    # ponytail: hand-tuned linear penalty. Fit the weights against ranked community maps when you have them.
    penalty = 60 * miss + 25 * fast + 20 * reset + 10 * cross + 15 * min(stress / STRESS_HIGH, 1.0)
    return Report(left, right, float(np.clip(100.0 - penalty, 0.0, 100.0)))
