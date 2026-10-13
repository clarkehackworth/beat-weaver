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
