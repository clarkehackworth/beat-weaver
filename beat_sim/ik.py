"""Inverse kinematics: find joint angles that put the blade through a point, staying close to the previous pose."""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from beat_sim.body import Q_MAX, Q_MIN, blade_point_residual, fk

W_SMOOTH = 0.03  # rad of joint travel costs as much as this many metres of miss
W_AXIS = 0.3  # unit of blade-axis error costs as much as this many metres of miss
REACH_TOL = 0.12  # metres; a note cube is ~0.5 m wide, so this is still a clean centre cut


def solve(
    q_prev: np.ndarray,
    mirror: bool,
    point: np.ndarray | None = None,
    hilt: np.ndarray | None = None,
    axis: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """Find q near q_prev with: blade through `point`, hand at `hilt`, blade along `axis` (each optional).

    Returns (q, position error in metres of whichever of point/hilt was requested).
    """

    def resid(q):
        pose = fk(q, mirror)
        parts = [W_SMOOTH * (q - q_prev)]
        if point is not None:
            parts.append(blade_point_residual(pose, point))
        if hilt is not None:
            parts.append(pose["hilt"] - hilt)
        if axis is not None:
            parts.append(W_AXIS * (pose["axis"] - axis))
        return np.concatenate(parts)

    # ponytail: local solver seeded from the previous pose. Good enough because a human
    # does the same thing; add a second seed from Q_REST if misses look like local minima.
    r = least_squares(resid, np.clip(q_prev, Q_MIN, Q_MAX), bounds=(Q_MIN, Q_MAX), xtol=1e-4, ftol=1e-6, max_nfev=200)
    pose = fk(r.x, mirror)
    err = 0.0
    if point is not None:
        err = max(err, float(np.linalg.norm(blade_point_residual(pose, point))))
    if hilt is not None:
        err = max(err, float(np.linalg.norm(pose["hilt"] - hilt)))
    return r.x, err


if __name__ == "__main__":
    from beat_sim.body import Q_REST

    note = np.array([0.3, 1.15, 0.65])
    q, err = solve(Q_REST, False, point=note, axis=np.array([0.0, 0.0, 1.0]))
    assert err < REACH_TOL, err
    assert np.all(q >= Q_MIN - 1e-9) and np.all(q <= Q_MAX + 1e-9)
    h = fk(q)["hilt"]
    q2, err2 = solve(q, False, hilt=h - [0, 0.2, 0], axis=np.array([0.0, -0.71, 0.71]))
    assert err2 < REACH_TOL and fk(q2)["axis"][1] < -0.5, "down-cut follow-through: hand lower, blade pointing down"
    _, far = solve(Q_REST, False, point=np.array([3.0, 1.0, 0.65]))
    assert far > 0.5, "3 m to the right is out of reach"
    print("ik ok")
