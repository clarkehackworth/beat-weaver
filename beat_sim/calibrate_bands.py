"""Recompute rhythm BANDS (p10..p90 per metric per difficulty) from folders of reference maps.

    python -m beat_sim.calibrate_bands <folder> [<folder> ...]

Prints a Python dict literal to paste over BANDS in rhythm.py. Music metrics (energy_corr,
onset_hit, lead_in) are kept from the existing table since they need audio; see calibrate_music.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from beat_sim.rhythm import AUDIO_METRICS, BANDS, stats
from beat_weaver.parsers.beatmap_parser import parse_map_folder

DIFFS = ("Easy", "Normal", "Hard", "Expert", "ExpertPlus")


def main() -> None:
    rows: dict[str, list[dict]] = {d: [] for d in DIFFS}
    for root in sys.argv[1:]:
        for f in sorted(Path(root).iterdir()):
            try:
                bms = parse_map_folder(f)
            except Exception:
                continue
            for bm in bms:
                if bm.difficulty_info.characteristic != "Standard" or bm.difficulty_info.difficulty not in rows:
                    continue
                n = [x for x in bm.notes if 0 <= x.x <= 3 and 0 <= x.y <= 2]
                if len(n) >= 60:
                    rows[bm.difficulty_info.difficulty].append(stats(n, bm.metadata.bpm))
    print("BANDS = {")
    for d in DIFFS:
        rs = rows[d]
        keep = {k: v for k, v in BANDS.get(d, BANDS["Expert"]).items() if k in AUDIO_METRICS or k == "lead_in"}
        print(f"    {d!r}: {{  # {len(rs)} maps")
        for k in rs[0]:
            lo, hi = np.percentile([r[k] for r in rs], [10, 90])
            if k in ("vision", "vision_run3", "converging", "near_converging", "far_cross", "gap_ge2", "dots"):  # "none" is never out of band
                lo = 0.0
            print(f"        {k!r}: ({lo:.3f}, {hi:.3f}),")
        for k, v in keep.items():
            print(f"        {k!r}: {v},")
        print("    },")
    print("}")


if __name__ == "__main__":
    main()
