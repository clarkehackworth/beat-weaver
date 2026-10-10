"""Stick-figure GIF of the planned trajectories. Front view (x/y) plus top view (x/z). Pillow only."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from beat_sim.body import Q_REST, fk
from beat_sim.planner import Trajectory
from beat_sim.swing import COL_X, PLANE_Z, ROW_Y, note_direction, note_point
from beat_weaver.schemas.normalized import Note

W, H = 420, 480  # one panel
SCALE = 160.0  # px per metre
COLORS = {0: (230, 60, 60), 1: (60, 120, 230)}
NOTE_HALF = 0.22  # metres, drawn note half-size
NOTE_WINDOW = 0.4  # seconds a note stays visible around its hit time


def _front(p):  # x right, y up
    return (W / 2 + p[0] * SCALE, H - 40 - p[1] * SCALE)


def _top(p):  # x right, z up (toward the notes)
    return (W / 2 + p[0] * SCALE, H - 40 - p[2] * SCALE)


def _pose_at(traj: Trajectory, t: float) -> np.ndarray:
    if len(traj.t) == 0 or t < traj.t[0]:
        return Q_REST
    i = min(int(np.searchsorted(traj.t, t)), len(traj.t) - 1)
    return traj.q[i]


def _draw_arm(d: ImageDraw.ImageDraw, pose: dict, proj, color, mirror: bool):
    chain = [pose["shoulder"], pose["elbow"], pose["wrist"], pose["hilt"]]
    d.line([proj(p) for p in chain], fill=(40, 40, 40), width=5)
    d.line([proj(pose["hilt"]), proj(pose["tip"])], fill=color, width=4)
    for p in chain[1:3]:
        x, y = proj(p)
        d.ellipse([x - 4, y - 4, x + 4, y + 4], fill=(40, 40, 40))


def _draw_body(d, proj):
    sh = fk(Q_REST)["shoulder"]
    l, r = proj([-sh[0], sh[1], sh[2]]), proj(sh)
    d.line([l, r], fill=(40, 40, 40), width=5)
    hx, hy = proj([0.0, sh[1] + 0.22, sh[2]])
    d.ellipse([hx - 14, hy - 14, hx + 14, hy + 14], outline=(40, 40, 40), width=4)
    hip = proj([0.0, sh[1] - 0.55, sh[2]])
    d.line([proj([0.0, sh[1], sh[2]]), hip], fill=(40, 40, 40), width=5)


def _draw_notes(d, notes: list[Note], t: float, proj, front: bool):
    for n in notes:
        dt = n.time_seconds - t
        if abs(dt) > NOTE_WINDOW or not (0 <= n.x <= 3 and 0 <= n.y <= 2):
            continue
        p = note_point(n)
        if not front:
            p = p + [0, 0, 4.0 * max(dt, 0.0)]  # notes approach along z until hit time
        cx, cy = proj(p)
        h = NOTE_HALF * SCALE * (0.4 if dt < 0 else 1.0)
        col = COLORS[n.color] if dt >= 0 else tuple(int(c * 0.5 + 120) for c in COLORS[n.color])
        d.rectangle([cx - h, cy - h, cx + h, cy + h], outline=col, width=3)
        dv = note_direction(n)
        if dv is not None and front and dt >= 0:
            d.line([(cx, cy), (cx + dv[0] * h * 0.8, cy - dv[1] * h * 0.8)], fill=col, width=4)


def _frame(trajs: list[Trajectory], notes: list[Note], t: float) -> Image.Image:
    img = Image.new("RGB", (2 * W, H), (245, 245, 245))
    d = ImageDraw.Draw(img)
    # front panel: grid outline
    for x in COL_X:
        for y in ROW_Y:
            cx, cy = _front([x, y, PLANE_Z])
            d.rectangle([cx - 4, cy - 4, cx + 4, cy + 4], outline=(200, 200, 200))
    d.line([(W, 0), (W, H)], fill=(180, 180, 180))
    d.text((8, 8), f"t = {t:6.2f}s   front", fill=(60, 60, 60))
    d.text((W + 8, 8), "top (notes come from above)", fill=(60, 60, 60))

    for off, proj, front in ((0, _front, True), (W, _top, False)):
        pr = lambda p, o=off, f=proj: (f(p)[0] + o, f(p)[1])
        if front:
            _draw_body(d, pr)
        else:
            y0 = pr([0.0, 0.0, PLANE_Z])[1]
            d.line([(off + 20, y0), (off + W - 20, y0)], fill=(200, 200, 200))
        _draw_notes(d, notes, t, pr, front)
        for tr in trajs:
            mirror = tr.hand == 0
            _draw_arm(d, fk(_pose_at(tr, t), mirror), pr, COLORS[tr.hand], mirror)
    return img


def render(trajs: list[Trajectory], notes: list[Note], out: Path, start: float = 0.0, end: float | None = None, fps: int = 30) -> Path:
    """Write a .gif (Pillow, streamed) or .mp4 (piped to ffmpeg). Frames are generated one at a time."""
    if end is None:
        end = max([tr.t[-1] for tr in trajs if len(tr.t)] + [start + 1.0])
    frames = (_frame(trajs, notes, t) for t in np.arange(start, end, 1.0 / fps))
    if out.suffix.lower() == ".mp4":
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{2 * W}x{H}",
               "-r", str(fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", str(out)]
        with subprocess.Popen(cmd, stdin=subprocess.PIPE) as proc:
            for f in frames:
                proc.stdin.write(f.tobytes())
            proc.stdin.close()
        if proc.returncode:
            raise RuntimeError(f"ffmpeg failed with code {proc.returncode}")
        return out
    first = next(frames)
    # Pillow consumes append_images lazily, so a generator keeps only one frame in memory.
    first.save(out, save_all=True, append_images=frames, duration=int(1000 / fps), loop=0, optimize=True)
    return out


if __name__ == "__main__":
    import tempfile

    from beat_sim import simulate

    notes = [Note(beat=i, time_seconds=0.5 + i * 0.4, x=1 + i % 2, y=1, color=i % 2, cut_direction=(1, 0)[(i // 2) % 2]) for i in range(6)]
    _, trajs = simulate(notes, return_trajectories=True)
    d = Path(tempfile.mkdtemp())
    im = Image.open(render(trajs, notes, d / "demo.gif", fps=10))
    assert im.n_frames > 20 and im.size == (2 * W, H), (im.n_frames, im.size)
    mp4 = render(trajs, notes, d / "demo.mp4", fps=10)
    assert mp4.stat().st_size > 1000
    print("viz ok", d)
