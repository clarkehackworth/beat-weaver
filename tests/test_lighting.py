"""Rules-based lightshow."""
import numpy as np

from beat_weaver.model.lighting import light_events, BACK, RING, SPIN, LASER_L, LASER_R
from beat_weaver.schemas.normalized import Note

BPM = 120.0


def N(beat, color):
    return Note(beat=beat, time_seconds=beat / 2, x=1 + color, y=0, color=color, cut_direction=1)


def test_light_events_follow_notes_and_bars():
    notes = [N(i, i % 2) for i in range(64)]  # 16 bars of quarter notes
    strength = np.ones(4000)
    strength[2000:] = 3.0  # second half louder
    ev = light_events(notes, BPM, strength, 0.01)
    assert all(set(e) == {"_time", "_type", "_value"} for e in ev)
    assert ev == sorted(ev, key=lambda e: (e["_time"], e["_type"]))
    assert sum(e["_type"] == BACK and e["_value"] in (2, 6) for e in ev) == 64  # one flash per hit
    assert sum(e["_type"] == LASER_L and e["_value"] > 0 for e in ev) == 32
    assert sum(e["_type"] == LASER_R and e["_value"] > 0 for e in ev) == 32
    assert sum(e["_type"] == RING for e in ev) >= 16  # one per bar
    assert sum(e["_type"] == SPIN for e in ev) == 2  # every 8 bars
    assert ev[-1]["_value"] == 0  # lights out at the end
    assert light_events([], BPM, strength, 0.01) == []
