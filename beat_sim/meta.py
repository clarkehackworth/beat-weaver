"""Map metadata: what the song is called, who made the map, which part previews, the cover.

Nothing person-specific lives in the repo. The mapper's name comes from, in order:
the --mapper flag, the BW_MAPPER environment variable, then the user config file
(~/.config/beat-weaver/config.json, key "mapper_name"; override the path with BW_CONFIG).

The cover comes from --cover, then art embedded in the audio file, then a web lookup: the thumbnail of the
song's YouTube video (the mp3s usually came from there), else its album art from the iTunes search API.
Web results are accepted only when the video title or album name matches the song; otherwise there is no cover.

Song title and artist come from the audio file's tags when it has them. Downloaded audio
often has none, so the file name is parsed ("Artist - Title (Official Video)"). The result
is printed before generation starts; --title and --artist override it when the guess is wrong.
"""

from __future__ import annotations

import io
import json
import os
import re
import shutil
import subprocess
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

DEFAULT_MAPPER = "BeatWeaver AI"
PREVIEW_SECONDS = 10.0


def config_path() -> Path:
    return Path(os.environ.get("BW_CONFIG", Path.home() / ".config" / "beat-weaver" / "config.json"))


def mapper_name(cli: str | None = None) -> str:
    if cli:
        return cli
    if os.environ.get("BW_MAPPER"):
        return os.environ["BW_MAPPER"]
    try:
        name = json.loads(config_path().read_text()).get("mapper_name")
        if name:
            return str(name)
    except (OSError, ValueError, AttributeError):
        pass
    return DEFAULT_MAPPER


# Bracketed or trailing words that describe the upload, not the song.
_NOISE_WORDS = r"official|video|audio|lyrics?|visuali[sz]er|hd|hq|4k|remaster(?:ed)?|music video|mv|explicit|\d{4}"
_BRACKET = re.compile(r"\s*[\(\[][^\)\]]*\b(?:" + _NOISE_WORDS + r")\b[^\)\]]*[\)\]]", re.I)
_TRAILING = re.compile(r"\s+(?:lyrics?|official (?:video|audio)|hd|hq)(?=\s*(?:[\(\[]|$))", re.I)


_ONLY_NOISE = re.compile(r"^(?:official (?:video|audio|visuali[sz]er|music video)|lyrics?|hd|hq)$", re.I)


def parse_filename(stem: str) -> tuple[str, str]:
    """(title, artist) guessed from a file name. Artist may be empty.

    Splits on a spaced dash. "Artist - Title (Official Video)" and "Title [Official Video] - Artist"
    are both common; the part that carries the upload noise is taken as the title. If neither or both
    do, the first part is the artist. Cannot be right for every name, hence the printout and overrides.
    """
    s = stem.replace(" _26 ", " & ")  # '&' mangled by the downloader
    parts = [p.strip() for p in re.split(r"\s+-\s+", s) if p.strip()]
    parts = [p for p in parts if not _ONLY_NOISE.match(p)] or parts  # e.g. a trailing "Official Visualizer"
    cleaned, noisy = [], []
    for p in parts:
        c = _TRAILING.sub("", _BRACKET.sub("", p)).strip()
        cleaned.append(c or p)
        noisy.append(c != p)
    if len(parts) <= 1:
        return (cleaned[0] if cleaned else stem), ""
    if len(parts) == 2 and noisy[0] and not noisy[1]:
        return cleaned[0], cleaned[1]
    return " - ".join(cleaned[1:]), cleaned[0]


def read_tags(audio: Path) -> dict[str, str]:
    """title / artist from the file's own tags; empty when it has none."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format_tags=title,artist", "-of", "json", str(audio)],
            capture_output=True, text=True, check=True).stdout
        tags = json.loads(out).get("format", {}).get("tags", {})
    except (subprocess.CalledProcessError, ValueError, FileNotFoundError):
        return {}
    return {k.lower(): v.strip() for k, v in tags.items() if k.lower() in ("title", "artist") and v.strip()}


def preview_start(audio: Path, length: float = PREVIEW_SECONDS) -> float:
    """Start (seconds) of the loudest `length`-second window, kept out of the first 5% and last 15% of the song."""
    from beat_sim.music import HOP, SR, envelope

    env = envelope(audio)
    per_s = SR / HOP
    win = int(length * per_s)
    if len(env) <= win * 2:
        return 0.0
    c = np.concatenate([[0.0], np.cumsum(env)])
    sums = c[win:] - c[:-win]  # sums[i] = loudness of env[i : i + win]
    lo, hi = int(0.05 * len(env)), int(0.85 * len(env)) - win
    if hi <= lo:
        return 0.0
    return round(float((lo + int(np.argmax(sums[lo:hi]))) / per_s), 1)


COVER_SIZE = 512  # Beat Saber covers are square; 512 px is the usual size
_UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) beat-weaver cover lookup"}


def _get(url: str, timeout: float = 10.0) -> bytes:
    with urllib.request.urlopen(urllib.request.Request(url, headers=_UA), timeout=timeout) as r:
        return r.read()


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _square_jpeg(data: bytes, size: int = COVER_SIZE) -> bytes:
    """Centre-crop to a square and resize. YouTube thumbnails are 16:9, so the sides are cut."""
    from PIL import Image

    im = Image.open(io.BytesIO(data)).convert("RGB")
    side = min(im.size)
    left, top = (im.width - side) // 2, (im.height - side) // 2
    out = io.BytesIO()
    im.crop((left, top, left + side, top + side)).resize((size, size), Image.LANCZOS).save(out, "JPEG", quality=90)
    return out.getvalue()


def _title_matches(found: str, title: str, artist: str) -> bool:
    """A search result is the right one when it names the song and the artist, and is not a cover version
    (unless the search was for one)."""
    f, t, a = _tokens(found), _tokens(title), _tokens(artist)
    if not t <= f:
        return False
    if a and not (a & f):
        return False
    return not ("cover" in f and "cover" not in t | a)


def youtube_cover(title: str, artist: str, hint: str = "") -> tuple[bytes, str] | None:
    """Thumbnail of the best-matching video in YouTube's search results (no API key, page scraping).

    `hint` is the audio file's own name, which for a YouTube download is the video's title. It is what
    gets searched, and among the results that name the song and artist, the one whose words overlap it
    most wins. Without it "Artist - Title" matched a live performance before the official video.

    ponytail: this parses the results page, which Google can change without notice. If it stops
    matching, the iTunes lookup below still works, and --cover always overrides.
    """
    query = hint or f"{artist} {title}".strip()
    try:
        html = _get("https://www.youtube.com/results?search_query=" + urllib.parse.quote(query)).decode("utf-8", "ignore")
    except OSError:
        return None
    want = _tokens(hint)
    ranked: list[tuple[int, int, str, str]] = []
    seen: set[str] = set()
    for chunk in html.split('"videoRenderer":{')[1:]:
        m = re.match(r'"videoId":"([\w-]{11})"', chunk)
        t = re.search(r'"title":\{"runs":\[\{"text":"((?:[^"\\]|\\.)*)"', chunk)
        if not m or not t or m.group(1) in seen:
            continue
        seen.add(m.group(1))
        if len(seen) > 10:
            break
        vid, vtitle = m.group(1), json.loads(f'"{t.group(1)}"')
        # the uploader's channel name often carries the artist ("Example Band" posting "Sample Song")
        owner = re.search(r'"ownerText":\{"runs":\[\{"text":"((?:[^"\\]|\\.)*)"', chunk)
        if _title_matches(vtitle + " " + (json.loads(f'"{owner.group(1)}"') if owner else ""), title, artist):
            ranked.append((-len(want & _tokens(vtitle)), len(ranked), vid, vtitle))
    for _, _, vid, vtitle in sorted(ranked):
        for name in ("maxresdefault", "sddefault", "hqdefault"):  # best first; not every video has the large ones
            try:
                return _get(f"https://i.ytimg.com/vi/{vid}/{name}.jpg"), f"YouTube video {vid} ({vtitle})"
            except OSError:
                continue
    return None


def itunes_cover(title: str, artist: str) -> tuple[bytes, str] | None:
    """Album art from the iTunes search API (public, no key). Square already."""
    q = urllib.parse.quote(f"{artist} {title}".strip())
    try:
        results = json.loads(_get(f"https://itunes.apple.com/search?term={q}&media=music&entity=song&limit=8")).get("results", [])
    except (OSError, ValueError):
        return None
    for r in results:
        if _title_matches(f"{r.get('trackName', '')} {r.get('artistName', '')}", title, artist) and r.get("artworkUrl100"):
            try:
                return _get(r["artworkUrl100"].replace("100x100bb", "1000x1000bb")), f"iTunes album art ({r['trackName']} / {r['collectionName']})"
            except OSError:
                continue
    return None


def _squash(text: str) -> str:
    """Letters and digits only, so "Samplesong" matches "Sample Song", and "Sample's Song" matches "Samples Song"."""
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _words(text: str) -> set[str]:
    return _tokens(text.replace("'", "").replace("\u2019", ""))


_EXTRA = re.compile(r"\s*[\(\[]([^\)\]]*\b(?:cover|feat\.?|ft\.?|remix|live|reimagined|acoustic)\b[^\)\]]*)[\)\]]", re.I)


def itunes_identify(stem: str, title: str, artist: str) -> tuple[str, str] | None:
    """(title, artist) for a file name, checked against the iTunes catalogue; None when nothing matches.

    A file name can't say which side of the dash is the artist ("Sample Song - Example Band") and often has
    no artist at all ("Sample Song"). A result is taken when its track and artist names both
    appear in the file name; failing that, when the guessed artist is empty, a result whose track
    name appears in it (that is how "Sample Song (feat. Guest Singer)" finds Example Band).
    """
    raw = f"{artist} {title}".strip()
    bare = re.sub(r"[\(\[][^\)\]]*[\)\]]", "", raw).strip()  # "(Cover by ...)" often sinks the search
    results = []
    for q in dict.fromkeys((bare, re.sub(r"\bcover\b", "", bare, flags=re.I).strip(), raw)):  # ponytail: 3 tries, then give up
        try:
            results += json.loads(_get(f"https://itunes.apple.com/search?term={urllib.parse.quote(q)}&media=music&entity=song&limit=15")).get("results", [])
        except (OSError, ValueError):
            continue
    have = _squash(stem.replace(" _26 ", " & "))
    named = [(_EXTRA.sub("", r.get("trackName", "")).strip(), r.get("artistName", "")) for r in results]
    named = [(t, a) for t, a in named if _squash(t) and _squash(a) and _squash(t) in have]
    for t, a in named:
        if _squash(a) in have:
            return t, a
    if not artist and named:
        return named[0]
    return None


def web_cover(title: str, artist: str, hint: str = "") -> tuple[bytes, str] | None:
    """YouTube thumbnail first, then iTunes album art. None when neither matches the song."""
    return youtube_cover(title, artist, hint) or itunes_cover(title, artist)


def extract_cover(audio: Path, out_dir: Path, override: Path | None = None, title: str = "", artist: str = "",
                  web: bool = True) -> tuple[str, str]:
    """Put a cover image in out_dir. Returns (file name, where it came from); ('', '') when there is none.

    Order: --cover, art embedded in the audio file, then the web lookup (needs title; skipped when web=False).
    """
    if override:
        dest = out_dir / f"cover{override.suffix.lower()}"
        shutil.copy(override, dest)
        return dest.name, f"--cover {override}"
    got = _embedded_cover(audio, out_dir)
    if got:
        return got, "art embedded in the audio file"
    if web and title:
        hit = web_cover(title, artist, audio.stem)
        if hit:
            (out_dir / "cover.jpg").write_bytes(_square_jpeg(hit[0]))
            return "cover.jpg", hit[1]
    return "", ""


def _embedded_cover(audio: Path, out_dir: Path) -> str:
    """Art stored inside the audio file, written to out_dir/cover.jpg; '' when there is none."""
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v", "-show_entries", "stream=codec_name",
                            "-of", "csv=p=0", str(audio)], capture_output=True, text=True)
    if probe.returncode or not probe.stdout.strip():
        return ""
    dest = out_dir / "cover.jpg"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(audio), "-an", "-frames:v", "1", str(dest)])
    # ffmpeg can exit non-zero yet write a good image; judge the file, not the exit code
    try:
        from PIL import Image

        with Image.open(dest) as im:
            im.verify()
        return dest.name
    except Exception:
        dest.unlink(missing_ok=True)
        return ""


def song_meta(audio: Path, title: str | None = None, artist: str | None = None, mapper: str | None = None,
              with_preview: bool = True, web: bool = True) -> dict:
    """Info.dat fields for this song. Explicit arguments beat file tags, which beat the file name.

    A file-name guess is checked against iTunes (web=True), which fixes swapped title/artist and fills a
    missing artist. A "(Cover by ...)" / "(feat. ...)" from the file name goes to the sub-name.
    """
    tags = read_tags(audio)
    f_title, f_artist = parse_filename(audio.stem)
    sub = ""
    if web and not (title and artist) and not (tags.get("title") and tags.get("artist")):
        hit = itunes_identify(audio.stem, f_title, f_artist)
        if hit:
            extras = [m.group(1).strip() for m in _EXTRA.finditer(f"{f_title} {f_artist}")]
            extras += [w for w in ("cover",) if w in _words(audio.stem) and not extras and w not in _words(hit[0] + hit[1])]
            extras = [e for e in extras if not _squash(re.sub(r"^(?:feat\.?|ft\.?)\s*", "", e, flags=re.I)) in _squash(hit[1])]
            f_title, f_artist, sub = hit[0], hit[1], ", ".join(e for e in extras if e)
            sub = sub.capitalize() if sub == "cover" else sub
    meta = {
        "_songName": title or tags.get("title") or f_title,
        "_songSubName": sub,
        "_songAuthorName": artist or tags.get("artist") or f_artist,
        "_levelAuthorName": mapper_name(mapper),
        "_previewDuration": int(PREVIEW_SECONDS),
    }
    if with_preview:
        meta["_previewStartTime"] = preview_start(audio)
    return meta


if __name__ == "__main__":
    cases = {
        "Example Band - Sample Song (Official Music Video)": ("Sample Song", "Example Band"),
        "Sample Song [Official HD Music Video] - Example Band": ("Sample Song", "Example Band"),
        "Example Band _26 Guest Act - Other Song (2020)": ("Other Song", "Example Band & Guest Act"),
        "Example Band - Third Song": ("Third Song", "Example Band"),
        "Example Band - Fourth Song Lyrics (Origin)": ("Fourth Song (Origin)", "Example Band"),
        "Example Band - Fifth Song lyrics": ("Fifth Song", "Example Band"),
        "Just A Title": ("Just A Title", ""),
        "Other Act - 5 out of 6 (Official Audio)": ("5 out of 6", "Other Act"),
        "Example Band - Sixth Song (Remastered 2023) - Official Visualizer": ("Sixth Song", "Example Band"),
    }
    for stem, want in cases.items():
        got = parse_filename(stem)
        assert got == want, (stem, got, want)
    old = os.environ.pop("BW_MAPPER", None)
    assert mapper_name("Explicit") == "Explicit"
    os.environ["BW_CONFIG"] = "/nonexistent/config.json"
    assert mapper_name() == DEFAULT_MAPPER
    os.environ["BW_MAPPER"] = "FromEnv"
    assert mapper_name() == "FromEnv"
    if old is not None:
        os.environ["BW_MAPPER"] = old
    else:
        del os.environ["BW_MAPPER"]
    print("meta ok")
