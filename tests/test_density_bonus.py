from beat_sim.mapper import density_bonus
from beat_sim.rhythm import BANDS


def test_density_bonus_rises_to_band_midpoint_and_stops():
    lo, hi = BANDS["Expert"]["nps"]
    assert density_bonus("Expert", lo - 1) == 0
    assert 0 < density_bonus("Expert", lo + 0.2 * (hi - lo)) < density_bonus("Expert", (lo + hi) / 2) == 10
    assert density_bonus("Expert", hi) == 10
    assert density_bonus("Normal", 2.0) == 0
