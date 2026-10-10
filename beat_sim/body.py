"""Kinematic arm model: 6 joint angles per arm -> saber blade in world space.

World frame: x right, y up, z forward (toward incoming notes). Right arm is
modelled; the left arm is the right arm mirrored in x. No dynamics.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R

# ponytail: average adult, single body size. Make these a dataclass if you ever
# need per-player calibration.
SHOULDER = np.array([0.20, 1.40, 0.0])  # right shoulder, metres
UPPER_ARM = 0.30
FOREARM = 0.27
GRIP = 0.10  # wrist -> saber hilt
BLADE = 1.00  # hilt -> tip
BLADE_START = 0.15  # hits closer to the hilt than this don't count
GRIP_TILT = np.deg2rad(50.0)  # blade angle below "perpendicular to forearm": 0 = hammer grip, 90 = blade continues the forearm

# q = [sh_flex, sh_abduct, sh_rotate, elbow_flex, wrist_flex, wrist_dev], radians.
# Order matters: these are the calibration knobs for "sane body limits".
Q_MIN = np.deg2rad([-40.0, -30.0, -80.0, 0.0, -70.0, -30.0])
Q_MAX = np.deg2rad([180.0, 160.0, 80.0, 150.0, 70.0, 30.0])
Q_REST = np.deg2rad([30.0, 10.0, 0.0, 60.0, 0.0, 0.0])  # relaxed guard pose

# Angular speed caps, rad/s. ponytail: one cap per joint, no torque model.
QD_MAX = np.deg2rad([900.0, 900.0, 1100.0, 1100.0, 1500.0, 1500.0])


def fk(q: np.ndarray, mirror: bool = False) -> dict[str, np.ndarray]:
    """Forward kinematics. Returns shoulder/elbow/wrist/hilt/tip positions and blade axis."""
    r_sh = R.from_euler("xzy", [-q[0], q[1], q[2]])  # flex (positive = forward), abduct, rotate
    elbow = SHOULDER + r_sh.apply([0.0, -UPPER_ARM, 0.0])
    r_el = r_sh * R.from_euler("x", -q[3])  # positive = forearm swings forward
    wrist = elbow + r_el.apply([0.0, -FOREARM, 0.0])
    r_wr = r_el * R.from_euler("xz", q[4:6])
    hilt = wrist + r_wr.apply([0.0, -GRIP, 0.0])
    axis = r_wr.apply([0.0, -np.sin(GRIP_TILT), np.cos(GRIP_TILT)])  # forward of the fist, tilted toward the forearm line
    tip = hilt + BLADE * axis
    out = {"shoulder": SHOULDER.copy(), "elbow": elbow, "wrist": wrist, "hilt": hilt, "tip": tip, "axis": axis}
    if mirror:
        for v in out.values():
            v[0] = -v[0]
    return out


def blade_point_residual(pose: dict[str, np.ndarray], target: np.ndarray) -> np.ndarray:
    """Vector from the nearest usable point on the blade to the target (zero when on the blade)."""
    start = pose["hilt"] + BLADE_START * pose["axis"]
    seg = pose["tip"] - start
    t = np.clip(np.dot(target - start, seg) / np.dot(seg, seg), 0.0, 1.0)
    return target - (start + t * seg)


def limit_stress(q: np.ndarray) -> float:
    """0 at mid-range, 1 at a joint limit, >1 past it. Sum over joints, quartic so only the edges hurt."""
    mid = (Q_MIN + Q_MAX) / 2
    half = (Q_MAX - Q_MIN) / 2
    return float(np.sum(((q - mid) / half) ** 4))


def extension(pose: dict[str, np.ndarray]) -> float:
    """Shoulder-to-wrist distance as a fraction of full arm length."""
    return float(np.linalg.norm(pose["wrist"] - pose["shoulder"]) / (UPPER_ARM + FOREARM))


if __name__ == "__main__":
    p = fk(Q_REST)
    assert p["tip"][2] > p["hilt"][2] > 0, "rest pose saber should point forward"
    assert abs(np.linalg.norm(p["elbow"] - SHOULDER) - UPPER_ARM) < 1e-9
    assert abs(np.linalg.norm(p["wrist"] - p["elbow"]) - FOREARM) < 1e-9
    hang = np.zeros(6)
    bent = hang.copy()
    bent[3] = np.deg2rad(90)
    assert fk(bent)["wrist"][2] > fk(hang)["wrist"][2], "elbow flexion from hanging should bring the hand forward"
    assert fk(bent)["axis"][1] > 0.5 and fk(bent)["axis"][2] > 0, "forearm horizontal: blade points up-forward"
    raised = Q_REST.copy()
    raised[0] = np.deg2rad(90)
    assert fk(raised)["elbow"][2] > p["elbow"][2], "shoulder flexion should bring the elbow forward"
    assert fk(Q_REST, mirror=True)["shoulder"][0] < 0
    print("body ok")
