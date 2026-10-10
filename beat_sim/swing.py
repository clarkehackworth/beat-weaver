"""Notes -> per-hand swing targets (time, point in space, required blade motion)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from beat_weaver.schemas.normalized import Note

# Grid geometry, metres. Beat Saber: 4 columns 0.6 m apart, 3 rows at ~0.6/1.1/1.6 m... close enough.
COL_X = np.array([-0.9, -0.3, 0.3, 0.9])
ROW_Y = np.array([0.7, 1.15, 1.6])
PLANE_Z = 0.65  # where the blade meets the note, in front of the player

# Cut direction -> unit vector the blade tip travels in the x/y plane. 8 = any.
_DIR = {
    0: (0, 1), 1: (0, -1), 2: (-1, 0), 3: (1, 0),
    4: (-1, 1), 5: (1, 1), 6: (-1, -1), 7: (1, -1),
}
SWING_ANGLE = np.deg2rad(45.0)  # blade tilts this far back before contact and this far forward after
HAND_TRAVEL = 0.20  # metres the hand moves along the cut between approach and contact (and contact and follow)
FULL_SWING_GAP = 0.5  # seconds between notes at which a swing is full size; tighter gaps shrink it
MIN_AMPLITUDE = 0.35  # a stream still needs some wrist flick


def amplitude(gap: float) -> float:
    """Swing size (0..1) a player would use given `gap` seconds to the nearest same-hand note."""
    return float(np.clip(gap / FULL_SWING_GAP, MIN_AMPLITUDE, 1.0))


@dataclass
class Swing:
    t: float
    hand: int  # 0 left, 1 right
    point: np.ndarray  # note centre in world space
    direction: np.ndarray | None  # unit motion vector of the tip, None for dot notes
    note: Note

    def axis(self, phase: float, amp: float = 1.0) -> np.ndarray:
        """Blade axis at phase -1 (approach), 0 (contact), +1 (follow-through), scaled by swing amplitude."""
        if self.direction is None:
            return np.array([0.0, 0.0, 1.0])
        a = phase * amp * SWING_ANGLE
        return np.array([0.0, 0.0, np.cos(a)]) + np.sin(a) * self.direction

    def hilt(self, phase: float, contact_hilt: np.ndarray, amp: float = 1.0) -> np.ndarray:
        """Where the hand is at approach/follow-through, given where it was at contact."""
        if self.direction is None:
            return contact_hilt
        return contact_hilt + phase * amp * HAND_TRAVEL * self.direction


def note_point(note: Note) -> np.ndarray:
    return np.array([COL_X[note.x], ROW_Y[note.y], PLANE_Z])


def note_direction(note: Note) -> np.ndarray | None:
    if note.cut_direction not in _DIR:
        return None
    d = np.array(_DIR[note.cut_direction], dtype=float)
    if note.angle_offset:
        a = np.deg2rad(note.angle_offset)
        d = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]) @ d
    d /= np.linalg.norm(d)
    return np.array([d[0], d[1], 0.0])


def swings_for_hand(notes: list[Note], hand: int) -> list[Swing]:
    out = [
        Swing(n.time_seconds, hand, note_point(n), note_direction(n), n)
        for n in notes
        if n.color == hand and 0 <= n.x <= 3 and 0 <= n.y <= 2
    ]
    out.sort(key=lambda s: s.t)
    # ponytail: stacked same-hand notes at one time collapse to their mean; windows/towers are a v2 problem
    merged: list[Swing] = []
    for s in out:
        if merged and abs(merged[-1].t - s.t) < 1e-3:
            m = merged[-1]
            m.point = (m.point + s.point) / 2
            if m.direction is None:
                m.direction = s.direction
        else:
            merged.append(s)
    return merged


if __name__ == "__main__":
    n = Note(beat=1, time_seconds=0.5, x=3, y=0, color=1, cut_direction=1)
    s = swings_for_hand([n], 1)[0]
    assert s.axis(-1)[1] > 0 > s.axis(1)[1], "down cut: blade tilts up before, down after"
    assert abs(np.linalg.norm(s.axis(-1)) - 1) < 1e-9
    assert s.hilt(-1, np.zeros(3))[1] > 0 > s.hilt(1, np.zeros(3))[1], "hand starts above, ends below"
    assert amplitude(0.1) < amplitude(0.3) < amplitude(1.0) == 1.0
    assert abs(s.hilt(1, np.zeros(3), amplitude(0.1))[1]) < abs(s.hilt(1, np.zeros(3))[1]), "tight gap = smaller swing"
    assert swings_for_hand([n], 0) == []
    dot = Note(beat=1, time_seconds=0.5, x=0, y=2, color=0, cut_direction=8)
    assert swings_for_hand([dot], 0)[0].direction is None
    print("swing ok")
