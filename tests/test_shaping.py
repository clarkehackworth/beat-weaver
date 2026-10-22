"""bsmg-guideline shaping passes: downmapping, section repeats, emphasis."""
import numpy as np

from beat_weaver.model import shaping
from beat_weaver.schemas.normalized import Note

BPM = 120.0


def N(t, x, y, c, d):
    return Note(beat=t * BPM / 60, time_seconds=t, x=x, y=y, color=c, cut_direction=d)


def stream(n=256, step=0.5):
    return [N(i * step, 1 + i % 2, 1 if i % 4 else 2, i % 2, (1, 0)[(i // 2) % 2]) for i in range(n)]


def test_downmap_thins_to_band_ratio_and_keeps_times():
    base = stream()
    hard = shaping.downmap(base, BPM, "ExpertPlus", "Hard")
    ratio = shaping.TARGET_NPS["Hard"] / shaping.TARGET_NPS["ExpertPlus"]
    assert abs(len(hard) / len(base) - ratio) < 0.03
    assert {n.time_seconds for n in hard} <= {n.time_seconds for n in base}
    assert shaping.downmap(base, BPM, "Hard", "ExpertPlus") == sorted(base, key=lambda n: (n.time_seconds, n.color, n.x, n.y))


def test_downmap_easy_normal_drop_crossovers_and_top_row():
    base = stream() + [N(1000 + i, 3, 2, 0, 0) for i in range(10)]  # red crossovers on the top row
    easy = shaping.downmap(base, BPM, "Hard", "Easy")
    assert all(n.y < 2 for n in easy)
    assert not any(n.color == 0 and n.x >= 2 for n in easy)
    assert not any((n.x, n.y) in shaping.VISION_CELLS for n in easy if n.time_seconds >= 1000), "pulled-down notes avoid vision cells"


def test_repair_parity_reverses_double_directionals():
    notes = [N(0, 0, 0, 0, 1), N(0.5, 0, 0, 0, 1), N(1.0, 0, 0, 0, 8), N(1.5, 0, 0, 0, 6), N(4.0, 0, 0, 0, 6), N(0.5, 3, 0, 1, 0)]
    fixed = shaping.repair_parity(notes, BPM)
    by_t = {(n.time_seconds, n.color): n.cut_direction for n in fixed}
    assert by_t[(0.5, 0)] == 0, "same cut again within the window is reversed"
    assert by_t[(1.0, 0)] == 8, "dots untouched"
    assert by_t[(1.5, 0)] == 6 and by_t[(4.0, 0)] == 6, "two beats on, a repeat is a legal reset"
    assert by_t[(0.5, 1)] == 0, "other hand independent"
    stack = [N(0, 0, 0, 0, 0), N(0.5, 0, 0, 0, 0), N(0.5, 0, 1, 0, 0)]
    fixed = shaping.repair_parity(stack, BPM)
    assert [n.cut_direction for n in fixed if n.time_seconds == 0.5] == [1, 1], "a stack is reversed as one swing"
    # thinning a stream leaves no double directionals within a beat
    hard = shaping.downmap(stream(step=0.25), BPM, "ExpertPlus", "Easy")
    for hand in (0, 1):
        h = [n for n in hard if n.color == hand]
        for a, b in zip(h, h[1:]):
            if b.time_seconds - a.time_seconds < 1.0:
                assert a.cut_direction != b.cut_direction, (a, b)


def test_downmap_prefers_on_beat_and_loud_hits():
    base = stream(step=0.25)  # half-beat stream at 120 bpm
    strength = np.zeros(int(130 / 0.01))
    strength[int(7.75 / 0.01)] = 1.0  # a loud off-beat hit
    hard = shaping.downmap(base, BPM, "ExpertPlus", "Hard", strength=strength, hop_s=0.01)  # Easy would also drop crossovers
    times = sorted({n.time_seconds for n in hard})
    assert 7.75 in times
    assert all(t in times for t in np.arange(0, 64, 0.5)), "every on-beat hit survives; quarter beats go first"
    assert sum(t % 0.5 == 0.25 for t in times) == len(times) - 128, "only the quarter beats were thinned"
    assert times[0] < 5 and times[-1] > 55, "thinning spreads over the song"


def test_downmap_easy_normal_lose_stacks_and_cap_doubles():
    base = stream()  # every hit is a single
    stacked = base + [Note(beat=n.beat, time_seconds=n.time_seconds, x=n.x, y=0, color=n.color, cut_direction=n.cut_direction) for n in base[::4]]
    doubles = [Note(beat=n.beat, time_seconds=n.time_seconds, x=3 - n.x, y=n.y, color=1 - n.color, cut_direction=n.cut_direction) for n in base]
    normal = shaping.downmap(stacked + doubles, BPM, "Hard", "Normal")
    hits = shaping._hits(normal)
    assert all(len([n for n in h if n.color == c]) <= 1 for h in hits for c in (0, 1)), "no stacks"
    assert sum(len(h) > 1 for h in hits) / len(hits) <= shaping.TIERS["Normal"]["max_doubles"] + 0.02


def test_downmap_easy_normal_have_no_converging_pairs():
    # a red down-right next to a blue down-left on the bottom row would converge once pulled down
    pair = [N(t, 1, 2, 0, 7) for t in range(0, 60, 2)] + [N(t, 2, 2, 1, 6) for t in range(0, 60, 2)]
    easy = shaping.downmap(pair + stream(), BPM, "Hard", "Easy")
    for h in shaping._hits(easy):
        for l in (n for n in h if n.color == 0):
            for r in (n for n in h if n.color == 1):
                assert not shaping._converging(l, r), (l, r)


def test_downmap_ladder_is_monotonic():
    ladder = shaping.downmap_ladder(stream(), BPM, "ExpertPlus", ["Easy", "Normal", "Hard", "Expert", "ExpertPlus"])
    counts = [len(ladder[d]) for d in ("Easy", "Normal", "Hard", "Expert", "ExpertPlus")]
    assert counts == sorted(counts) and counts[0] < counts[-1]


def test_find_repeats_and_copy_with_mirroring():
    hop_s = 0.01
    rng = np.random.default_rng(0)
    block = rng.random((int(16 / hop_s), 4))
    feat = np.concatenate([block, rng.random(block.shape), block, block])
    pairs = shaping.find_repeats(feat, hop_s, BPM, bars=8)
    assert pairs == [(0.0, 32.0), (0.0, 48.0)]
    rep = shaping.repeat_sections(stream(), BPM, pairs, bars=8)
    a = [(n.x, n.color, n.cut_direction) for n in rep if n.time_seconds < 16]
    b = [(n.x, n.color, n.cut_direction) for n in rep if 32 <= n.time_seconds < 48]
    c = [(n.x, n.color, n.cut_direction) for n in rep if 48 <= n.time_seconds < 64]
    assert a == b
    assert c == [(3 - x, 1 - col, shaping._MIRROR_DIR[d]) for x, col, d in a]
    assert shaping.repeat_sections(stream(), BPM, []) == sorted(stream(), key=lambda n: (n.time_seconds, n.color, n.x, n.y))


def test_repeat_keeps_the_quieter_blocks_density():
    hop_s = 0.01
    pairs = [(0.0, 32.0)]
    notes = stream(64) + [n for n in stream() if 32 <= n.time_seconds < 48][::2]  # second block half as dense
    rep = shaping.repeat_sections(notes, BPM, pairs, bars=8)
    dst = [n for n in rep if 32 <= n.time_seconds < 48]
    assert abs(len(dst) - 16) <= 2


def test_emphasize_stacks_only_loud_isolated_hits_off_vision_cells():
    hop_s = 0.01
    sparse = [N(t, 0, 1, 0, 1) for t in range(0, 128, 2)]
    strength = np.zeros(int(130 / hop_s))
    strength[int(8.0 / hop_s)] = 1.0
    emp = shaping.emphasize(sparse, BPM, strength, hop_s, top_frac=1 / len(sparse) + 1e-9)
    extra = [n for n in emp if n not in sparse]
    assert len(extra) == 1 and extra[0].time_seconds == 8.0 and extra[0].color == 0 and extra[0].cut_direction == 1
    assert (extra[0].x, extra[0].y) not in shaping.VISION_CELLS
    # a half-beat stream has no room: nothing added
    dense = stream(step=0.25)
    assert shaping.emphasize(dense, BPM, strength, hop_s, top_frac=0.5) == sorted(dense, key=lambda n: (n.time_seconds, n.color, n.x, n.y))


def test_emphasize_sliders_on_sustained_sound():
    hop_s = 0.01
    sparse = [N(t, 0, 0, 0, 3) for t in range(0, 128, 2)]  # right cuts from the bottom-left cell
    strength = np.full(int(130 / hop_s), 0.1)
    strength[int(8.0 / hop_s): int(8.3 / hop_s)] = 1.0  # held sound at 8 s
    emp = shaping.emphasize(sparse, BPM, strength, hop_s, top_frac=0.0, sustain_frac=1 / len(sparse) + 1e-9)
    extra = [n for n in emp if n not in sparse]
    assert len(extra) == 1 and extra[0].cut_direction == 8 and (extra[0].x, extra[0].y) == (1, 0) and 8.0 < extra[0].time_seconds < 8.1


def test_onset_strength_shape():
    s, hop_s = shaping.onset_strength(np.random.default_rng(0).standard_normal(22050 * 3).astype(np.float32), 22050)
    assert len(s) == 22050 * 3 // 512 and abs(hop_s - 512 / 22050) < 1e-9


def test_gate_patterns_easy_simplifies_everything():
    notes = [
        N(0.0, 2, 2, 0, 4),   # left hand crossed over, top row, diagonal
        N(0.5, 0, 1, 0, 2),   # horizontal -> up/down
        N(1.0, 1, 0, 1, 7),   # right hand crossed over, diagonal
        N(1.5, 3, 0, 1, 1), N(1.5, 3, 1, 1, 1),  # stack -> single
    ]
    out = shaping.gate_patterns(notes, BPM, "Easy")
    assert all(n.cut_direction in shaping.TIERS["Easy"]["cuts"] for n in out)
    assert all(n.y < 2 for n in out)
    assert all((n.x <= 1) if n.color == 0 else (n.x >= 2) for n in out)
    assert len([n for n in out if n.time_seconds == 1.5]) == 1
    assert len(out) == 4


def test_gate_patterns_expert_keeps_everything():
    notes = [N(0.0, 2, 2, 0, 4), N(0.5, 0, 1, 0, 3), N(1.0, 3, 0, 1, 1), N(1.0, 3, 1, 1, 1)]
    out = shaping.gate_patterns(notes, BPM, "ExpertPlus")
    assert [(n.x, n.y, n.cut_direction) for n in out] == [(2, 2, 4), (0, 1, 3), (3, 0, 1), (3, 1, 1)]


def test_gate_patterns_clamps_lane_jumps_and_drops_sliders_on_hard():
    out = shaping.gate_patterns([N(0.0, 2, 0, 1, 1), N(0.5, 3, 0, 1, 0), N(1.0, 2, 0, 1, 1)], BPM, "Easy")
    assert [n.x for n in out] == [2, 3, 2]
    out = shaping.gate_patterns([N(0.0, 0, 0, 1, 1), N(0.5, 3, 0, 1, 0)], BPM, "Normal")
    assert [n.x for n in out] == [2, 3]  # own half first, then at most 2 lanes
    hard = shaping.gate_patterns([N(0.0, 0, 0, 0, 8), N(0.0, 0, 1, 0, 8), N(0.0, 3, 0, 1, 1), N(0.0, 3, 1, 1, 1)], BPM, "Hard")
    assert len(hard) == 3  # slider collapsed, stack kept


def test_emphasize_towers_and_windows_only_on_expert_plus():
    notes = [N(i * 2.0, 0 if i % 2 else 3, 0, i % 2, 1 if i % 2 else 3) for i in range(20)]  # loud, roomy singles
    strength = np.ones(2000); strength[::200] = 5.0  # hits 0, 10 ... loudest (hop 0.1 s -> t=0, 20)
    plus = shaping.emphasize(notes, BPM, strength, 0.1, top_frac=0.4, difficulty="ExpertPlus")
    exp = shaping.emphasize(notes, BPM, strength, 0.1, top_frac=0.4, difficulty="Expert")
    sizes_plus = sorted(len(h) for h in shaping._hits(plus))
    assert sizes_plus[-1] == 3 or any(abs(a.x - b.x) == 2 for h in shaping._hits(plus) for a in h for b in h)
    assert max(len(h) for h in shaping._hits(exp)) <= 2
    # the Expert gate trims a tower back to a stack and a window back to a single
    gated = shaping.gate_patterns(plus, BPM, "Expert")
    assert max(len(h) for h in shaping._hits(gated)) <= 2
    assert not any(abs(a.x - b.x) == 2 for h in shaping._hits(gated) for a in h for b in h)


def test_gate_patterns_limits_distinct_cells_on_easy():
    # left hand wanders over 4 cells within 2 bars; Easy allows 3 (hand, cell) placements in total
    notes = [N(i * 1.0, i % 2, i % 2, 0, (1, 0)[i % 2]) for i in range(8)] + [N(i * 1.0 + 0.5, 2 + i % 2, 0, 1, 1) for i in range(8)]
    out = shaping.gate_patterns(notes, BPM, "Easy")
    assert len({(n.color, n.x, n.y) for n in out}) <= 3
    assert len(out) == 16
    hard = shaping.gate_patterns(notes, BPM, "Hard")
    assert len({(n.color, n.x, n.y) for n in hard}) == 4


def test_gate_patterns_removes_cut_through_pairs():
    # blue stack at the bottom of column 3 cutting up into a red on the top row (seen in a generated Expert map)
    notes = [N(0.0, 3, 0, 1, 0), N(0.0, 3, 1, 1, 0), N(0.0, 3, 2, 0, 0),
             N(2.0, 0, 2, 1, 1), N(2.0, 0, 0, 0, 1)]  # blue top of column 0 cutting down into red two below
    out = shaping.gate_patterns(notes, BPM, "ExpertPlus")
    for h in shaping._hits(out):
        for l in h:
            for r in h:
                if l.color == 0 and r.color == 1:
                    assert not shaping._converging(l, r), (l, r)
    assert len(out) >= 4  # resolved by reversing / trimming the stack, not by deleting everything


def test_vocal_snap_moves_only_led_hits_within_tolerance():
    hop_s = 0.01
    lead = np.zeros(1000)
    lead[400:] = 3.0  # vocal-led from 4 s on
    # 1.06 s: not vocal-led, stays. 4.02 s: beat 8.04 -> nearest grid 8.0, stays. 6.30 s: 0.6 beats off, too far.
    peaks = np.array([1.06, 4.02, 6.30])
    notes = [N(1.0, 0, 0, 0, 1), N(4.0, 1, 0, 1, 1), N(6.0, 2, 0, 0, 1)]
    out = shaping.vocal_snap(notes, BPM, lead, peaks, hop_s)
    assert [n.time_seconds for n in out] == [1.0, 4.0, 6.0]
    # 4.085 s = beat 8.17: nearer the 1/12 grid (8.1667) than 1/8 (8.125), so a triplet placement
    out = shaping.vocal_snap(notes, BPM, lead, np.array([4.085]), hop_s)
    assert abs(out[1].beat - 8 - 1 / 6) < 1e-9 and out[1].time_seconds == out[1].beat * 60 / BPM
    # the target time already held by another hit: stay put
    held = notes + [N(4.0 + 60 / BPM / 6, 2, 0, 0, 1)]
    out = shaping.vocal_snap(held, BPM, lead, np.array([4.085]), hop_s)
    assert sorted(n.time_seconds for n in out) == sorted(n.time_seconds for n in held)


def test_bombs_and_walls():
    # left hand: down-cut, 3-beat rest, down-cut again -> bomb low at 2/3 of the gap
    notes = [N(0.0, 0, 1, 0, 1), N(1.5, 0, 1, 0, 1), N(1.5, 3, 1, 1, 0), N(6.0, 3, 1, 1, 0)]
    bombs, walls = shaping.bombs_and_walls(notes, BPM)
    assert [(b["_time"], b["_lineLayer"], b["_type"]) for b in bombs] == [(2.0, 0, 3), (9.0, 2, 3)]
    assert [w for w in walls if w["_type"] == 1] == [{"_time": 4.0, "_lineIndex": 0, "_type": 1, "_width": 4, "_duration": 7.0}]
    # alternating cuts need no reset; short rests get no wall
    short = shaping.bombs_and_walls([N(0.0, 0, 1, 0, 1), N(1.5, 0, 1, 0, 0)], BPM)
    assert short[0] == [] and not [w for w in short[1] if w["_type"] == 1]


def test_limit_extras_drops_bad_walls_and_caps_the_rest():
    notes = [N(t, 1, 0, 0, 1) for t in np.arange(0, 32, 0.5)]  # beats 0-63, lane 1, bottom row
    w = lambda t, x, ty, wd, d: {"_time": t, "_lineIndex": x, "_type": ty, "_width": wd, "_duration": d}
    walls = [w(4, 0, 0, 1, 0.5),   # too short
             w(8, 0, 0, 3, 2.0),   # full-height 3-wide
             w(12, 1, 0, 1, 2.0),  # covers the lane-1 notes
             w(16, 1, 1, 2, 2.0),  # crouch over notes in its lanes: dropped (hidden while ducking)
             w(20, 3, 0, 1, 4.0), w(28, 3, 0, 1, 1.0), w(36, 0, 0, 1, 2.0),
             w(40, 0, 0, 1, 2.0)]  # lane 0 is clear, but a note sits at 42, inside the half-beat tail: dropped
    bombs = [{"_time": 2.0, "_lineIndex": 1, "_lineLayer": 2, "_type": 3, "_cutDirection": 8},   # on a note's column
             {"_time": 2.0, "_lineIndex": 3, "_lineLayer": 2, "_type": 3, "_cutDirection": 8}]
    b, kept = shaping.limit_extras(bombs, walls, notes, BPM, walls_per_bar=0.25)  # starts >= 16 beats apart
    assert [x["_time"] for x in kept] == [20, 36]  # 28 sits too close to the longer wall at 20
    assert [x["_lineIndex"] for x in b] == [3]  # same instant = a cluster: the one on the notes' column is dropped
    assert shaping.limit_extras(bombs, walls, notes, BPM, walls_per_bar=0)[1] == []


def test_lane_walls_take_the_free_half_and_leave_the_notes_alone():
    # notes only in lanes 2-3 for beats 0-20: the left half (lanes 0-1) is free for walls, one per 4 beats
    notes = [N(t * 60 / BPM, 2 + int(t) % 2, 0, 1, 1) for t in np.arange(0, 20, 0.5)]
    _, walls = shaping.bombs_and_walls(notes, BPM, bombs=False, walls=True)
    lane = [w for w in walls if w["_type"] == 0]
    assert lane and all(w["_lineIndex"] == 0 and w["_width"] == 2 for w in lane)
    assert all(b["_time"] - a["_time"] >= 4 for a, b in zip(lane, lane[1:]))
    # notes in every lane: no free half or outer lane, no wall
    both = [N(t * 60 / BPM, int(2 * t) % 4, 0, 1, 1) for t in np.arange(0, 20, 0.5)]  # every lane, every 2 beats
    assert not [w for w in shaping.bombs_and_walls(both, BPM, bombs=False, walls=True)[1] if w["_type"] == 0]
    # lanes 1-2 busy: no free half, so 1-wide walls in the empty outer lane 0
    mid = [N(t * 60 / BPM, 1 + int(t) % 2, 0, 1, 1) for t in np.arange(0, 20, 0.5)]
    lane = [w for w in shaping.bombs_and_walls(mid, BPM, bombs=False, walls=True)[1] if w["_type"] == 0]
    assert lane and all(w["_lineIndex"] == 0 and w["_width"] == 1 for w in lane)


def test_bombs_beside_a_note_within_a_beat_are_moved_clear():
    notes = [N(0.0, 1, 0, 0, 1)]  # beat 0, lane 1 bottom
    b = lambda t, x, y: {"_time": t, "_lineIndex": x, "_lineLayer": y, "_type": 3, "_cutDirection": 8}
    kept, _ = shaping.limit_extras([b(0.5, 2, 1), b(1.5, 2, 0), b(0.25, 3, 2)], [], notes, BPM)
    # (2,1) is diagonal-adjacent to the note: moved up to (2,2), clear of it; the others were clear already
    assert [(x["_time"], x["_lineIndex"], x["_lineLayer"]) for x in kept] == [(0.5, 2, 2), (1.5, 2, 0), (0.25, 3, 2)]
    # a bomb row at one instant: the cell beside the note is dropped, the rest of the row stays where it is
    row, _ = shaping.limit_extras([b(0.25, x, 1) for x in range(4)], [], notes, BPM)
    assert [x["_lineIndex"] for x in row] == [3]
    # a row half a beat from the note keeps all its cells: row cells only count within a quarter beat
    far, _ = shaping.limit_extras([b(0.4, x, 1) for x in range(4)], [], notes, BPM)
    assert [x["_lineIndex"] for x in far] == [0, 1, 2, 3]


def test_bombs_on_a_hands_route_between_notes_are_moved_off_it():
    # left hand cuts down at (0,0), then up at (0,1) a beat and a half later: the saber comes back up through (0,0)
    notes = [N(0.0, 0, 0, 0, 1), N(1.5 * 60 / BPM, 0, 1, 0, 0)]
    b = {"_time": 1.0, "_lineIndex": 0, "_lineLayer": 0, "_type": 3, "_cutDirection": 8}
    assert shaping._on_swing_route(b, notes)
    kept, _ = shaping.limit_extras([b], [], notes, BPM)
    assert kept and not shaping._on_swing_route(kept[0], notes) and kept[0]["_lineIndex"] >= 2
    # the other side of the grid is clear of that route
    assert not shaping._on_swing_route({**b, "_lineIndex": 3, "_lineLayer": 2}, notes)


def test_overlapping_walls_keep_the_longer():
    w = lambda t, x, wd, d: {"_time": t, "_lineIndex": x, "_type": 0, "_width": wd, "_duration": d}
    kept = shaping.drop_overlapping_walls([w(0, 0, 2, 2.0), w(0.5, 1, 1, 1.0), w(2.25, 0, 1, 1.0), w(0, 3, 1, 1.0), w(3, 0, 1, 1.0)])
    assert [(k["_time"], k["_lineIndex"]) for k in kept] == [(0, 0), (0, 3), (3, 0)]


def test_stacked_notes_whose_swings_cross_are_resolved():
    # red up-cut above a blue down-cut: each saber has to come through the other note's cell
    up, down = N(0.0, 0, 1, 0, 0), N(0.0, 0, 0, 1, 1)
    assert shaping._approaches_through(up, down) and shaping._approaches_through(down, up)
    assert shaping._converging(up, down)
    kept = shaping._resolve_collisions([[up, down]])[0]
    assert len(kept) == 1 or not shaping._converging(*sorted(kept, key=lambda n: n.color))
    # same stack cutting the same way is fine to approach (blue down under red down is cuts-through, handled elsewhere);
    # a pair a cell apart on different rows with the sabers coming from clear cells stays
    assert not shaping._approaches_through(N(0.0, 0, 1, 0, 0), N(0.0, 3, 1, 1, 0))
