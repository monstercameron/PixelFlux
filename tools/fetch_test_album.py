#!/usr/bin/env python3
"""Build PixelFlux's 50-image test album from real photographs on Wikimedia Commons.

WHY REAL PHOTOS
---------------
The first version of this corpus was procedurally drawn — flat colour bands standing in for
beaches and dogs. It was fine for plumbing (hashing, EXIF, dedup, layout) and worthless for
anything that has to look at a picture. Synthetic scenes have almost no high-frequency detail,
which makes perceptual hashing behave unrealistically, and they cannot exercise a tagger at all:
a model asked to describe two rectangles has nothing to be right or wrong about.

So the corpus is now real photographs, pulled from Wikimedia Commons because it needs no API
key, the licences permit this use, and — the part that matters most — the files carry genuine
camera EXIF. Real bodies, real lenses, real capture dates, real GPS fixes.

WHAT IT COVERS
--------------
Subjects are chosen for the things a photo manager has to tell apart: faces, cars, doors, shoes,
food, animals, landmarks, interiors, street scenes, landscapes. Photos that carry a GPS fix and
a camera model are preferred, so place and camera search have real data underneath them.

The adversarial cases from the synthetic corpus are kept, because they are still the only way to
test dedup and error handling — they are now derived from real photographs rather than drawn:
two byte-identical duplicates, a three-frame near-duplicate burst, several files with EXIF
stripped by re-encoding, and one deliberately truncated file.

LICENSING
---------
Everything downloaded is CC or public domain. ATTRIBUTION.tsv records the author, licence, and
source URL for every file. This corpus is local test data and is not redistributed, but the
attribution is recorded anyway because it costs nothing and the licences ask for it.

USAGE
-----
    python tools/fetch_test_album.py --out testdata/album --clean
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import shutil
import sys
import time
import urllib.parse
import urllib.request

from PIL import Image

API = "https://commons.wikimedia.org/w/api.php"


def say(message: str) -> None:
    """Print without dying on a filename the console codepage cannot represent.

    Commons titles carry accents, umlauts, and CJK freely. On a Windows console defaulting to
    cp1252 an ordinary print() of one of those raises UnicodeEncodeError mid-download and takes
    the whole run down — which is exactly what happened the first time this script ran.
    """
    sys.stdout.write(message.encode("utf-8", "replace").decode("utf-8", "replace") + "\n")
    sys.stdout.flush()

# Wikimedia requires a descriptive User-Agent and will refuse generic ones.
UA = "PixelFlux-testdata/0.1 (local photo-manager test corpus; https://github.com/local/pixelflux)"

# Long edge for stored files. Commons originals run to 40 megapixels and 20 MB; a test corpus
# that takes a minute to ingest stops being run. 2400 keeps real detail — enough that a tagger
# and a perceptual hash both behave as they would on a real library — at about 1 MB per file.
MAX_EDGE = 2400

# (search terms, how many to keep, a short subject label for the manifest)
#
# Terms are phrased to bias towards photographs rather than the diagrams, maps, logos, and
# scanned documents that dominate a naive Commons search.
SUBJECTS = [
    ("portrait photograph woman face",        2, "face"),
    ("portrait photograph man face outdoor",  2, "face"),
    ("sneakers shoes photograph",             2, "shoes"),
    ("classic car photograph street",         2, "car"),
    ("sports car red photograph",             2, "car"),
    ("wooden door old building",              2, "door"),
    ("colourful door house entrance",         2, "door"),
    ("cathedral exterior photograph",         2, "landmark"),
    ("bridge river city photograph",          2, "landmark"),
    ("dog photograph outdoor",                2, "animal"),
    ("cat photograph indoor",                 2, "animal"),
    ("food plate restaurant photograph",      3, "food"),
    ("street market people photograph",       3, "street"),
    ("living room interior photograph",       2, "interior"),
    ("mountain landscape photograph",         2, "landscape"),
    ("beach sea photograph coast",            2, "landscape"),
    ("bicycle photograph street",             2, "bicycle"),
    ("flowers garden photograph macro",       2, "flowers"),
    ("bird wildlife photograph",              2, "wildlife"),
]


def api(params: dict) -> dict:
    """Call the Commons API and return the parsed response."""
    params = {**params, "format": "json", "formatversion": "2"}
    url = f"{API}?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def score(info: dict) -> int:
    """Rank a candidate by how much real metadata it carries.

    A photo manager's test corpus lives or dies on metadata variety, so files with a GPS fix and
    a named camera body are worth far more here than a prettier picture with an empty EXIF block.
    """
    meta = {m["name"]: m["value"] for m in (info.get("metadata") or [])}
    points = 0
    if meta.get("GPSLatitude"):
        points += 4
    if meta.get("Model"):
        points += 3
    if meta.get("DateTimeOriginal") or meta.get("DateTime"):
        points += 2
    if meta.get("FNumber") or meta.get("ISOSpeedRatings"):
        points += 1
    return points


def usable(info: dict) -> bool:
    """Filter out anything that is not a reasonably sized photograph."""
    if info.get("mime") not in ("image/jpeg", "image/png"):
        return False
    if info.get("width", 0) < 1200 or info.get("height", 0) < 900:
        return False
    # Above ~25 MB is usually a panorama stitch or a museum scan; skip rather than spend the
    # bandwidth to downscale it away.
    return 120_000 <= info.get("size", 0) <= 25_000_000


def search(term: str, want: int, seen: set[str]) -> list[dict]:
    """Find the best-metadata photographs matching a search term."""
    try:
        data = api({
            "action": "query",
            "generator": "search",
            "gsrsearch": f"filetype:bitmap {term}",
            "gsrnamespace": 6,
            "gsrlimit": 40,
            "prop": "imageinfo",
            "iiprop": "url|size|mime|extmetadata|metadata",
            "iimetadataversion": "latest",
        })
    except Exception as exc:                                       # noqa: BLE001 - network tool
        say(f"  ! search failed for {term!r}: {exc}")
        return []

    candidates = []
    for page in data.get("query", {}).get("pages", []):
        info = (page.get("imageinfo") or [None])[0]
        if not info or page["title"] in seen or not usable(info):
            continue
        candidates.append((score(info), page["title"], info))

    # Highest metadata score first; title as a tiebreak so the selection is deterministic.
    candidates.sort(key=lambda c: (-c[0], c[1]))

    picked = []
    for points, title, info in candidates:
        if len(picked) >= want:
            break
        seen.add(title)
        picked.append({"title": title, "info": info, "score": points})

    return picked


def download(url: str, attempts: int = 5) -> bytes:
    """Fetch a file's bytes, backing off when Wikimedia rate-limits us.

    Commons returns 429 readily for sequential bulk downloads, and it means it — hammering
    through the failures loses roughly a third of the corpus. Each retry waits longer, and
    there is a fixed pause between files in the caller, which together keep the whole run
    inside what the service is willing to serve.
    """
    delay = 4
    last: Exception | None = None

    for attempt in range(attempts):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(request, timeout=180) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            last = exc
            if exc.code not in (429, 503):
                raise
            say(f"    rate-limited, waiting {delay}s (attempt {attempt + 1}/{attempts})")
            time.sleep(delay)
            delay *= 2
        except Exception as exc:                                   # noqa: BLE001 - network tool
            last = exc
            time.sleep(delay)
            delay *= 2

    raise RuntimeError(f"gave up after {attempts} attempts: {last}")


def slug(title: str) -> str:
    """Turn a Commons file title into a short, safe filename stem."""
    stem = title.removeprefix("File:").rsplit(".", 1)[0]
    stem = re.sub(r"[^A-Za-z0-9]+", "-", stem).strip("-").lower()
    return (stem[:48] or "photo").strip("-")


def store(raw: bytes, path: str, rating: int = 0, keywords: list[str] | None = None) -> tuple[int, int]:
    """Write a downloaded photo, downscaling if needed but keeping its EXIF intact.

    Preserving the original EXIF block through the resize is the entire point of this step —
    a corpus of real photographs whose camera metadata got stripped in processing would be no
    better than the synthetic one it replaces.
    """
    with Image.open(io.BytesIO(raw)) as img:
        img = img.convert("RGB")

        # Reopen the original EXIF block so the curated tags can be merged into it rather than
        # replacing it. Losing the camera metadata in order to add a star rating would be a
        # spectacularly bad trade.
        exif_obj = img.getexif()
        if rating:
            exif_obj[0x4746] = rating              # Rating, 0-5 — what Explorer shows
            exif_obj[0x4749] = rating * 20         # RatingPercent, kept consistent
        if keywords:
            # XPKeywords is UTF-16LE with a null terminator, which is what Windows writes and
            # what MetadataExtractor expects to decode.
            exif_obj[0x9C9E] = ";".join(keywords).encode("utf-16-le") + b"\x00\x00"

        exif = exif_obj.tobytes()

        if max(img.size) > MAX_EDGE:
            ratio = MAX_EDGE / max(img.size)
            img = img.resize((round(img.width * ratio), round(img.height * ratio)), Image.LANCZOS)

        # exif= carries the original block verbatim. Note that its PixelXDimension tag now
        # disagrees with the actual pixel size; that is deliberately left alone, because stale
        # dimension tags are extremely common in the wild and the ingestion path must trust the
        # decoder over EXIF anyway.
        img.save(path, "JPEG", quality=88, exif=exif, optimize=True)
        return img.size


def curate(index: int, subject: str, title: str, place_hint: str, camera: str) -> tuple[int, list[str]]:
    """Invent the metadata a *used* library would already carry: a star rating and keywords.

    The corpus previously had none of this, and it made three of the application's sort and
    filter dimensions decorative. Every rating was 0, so "highest rated" was just capture order;
    every favourite was false, so that filter returned an empty screen; and only six tags existed
    in the whole library, all incidental IPTC left behind by Commons uploaders.

    These are written into the files as real EXIF — Rating (0x4746) and XPKeywords (0x9C9E), the
    tags Windows Explorer itself reads and writes — rather than injected into database rows
    afterwards. That keeps the ingestion path honest: it is reading metadata out of files exactly
    as it would from a library someone has actually curated, and re-importing reproduces it.

    The rating distribution is deliberately lopsided. Real libraries are mostly unrated with a
    thin tail of favourites; a uniform spread would make "4 stars and up" return half the
    library and tell you nothing.
    """
    # index % 11 gives a fixed, reproducible spread: ~45% unrated, ~27% 3-star, and 9% each at
    # 1, 2, 4 and 5 stars.
    rating = [0, 0, 3, 0, 5, 3, 0, 4, 3, 1, 2][index % 11]

    words = {subject}

    # Words lifted from the Commons title, which is a human description of the photograph and
    # the best free source of real tags before any model has run.
    # Five characters, not four. Splitting a title into single words inevitably produces
    # fragments of phrases — "Great George Street" yields "great", "george", "street" — and a
    # tag list full of those is noise that makes the facet useless. A longer floor plus the
    # stopword list below removes most of it. Some still gets through, which is honest: keyword
    # metadata in real libraries is messy, and the good tags arrive when the model runs.
    for word in title.removeprefix("File:").rsplit(".", 1)[0].replace("-", " ").split():
        cleaned = "".join(c for c in word if c.isalpha()).lower()
        if len(cleaned) >= 5 and cleaned not in STOPWORDS:
            words.add(cleaned)

    if place_hint:
        words.add(place_hint)
    if camera:
        words.add(camera.split()[0].lower())

    return rating, sorted(words)[:12]


# Title words that carry no search value. Kept short on purpose: over-filtering here would
# strip the real subject words this corpus depends on.
STOPWORDS = {
    # filler and file-noise
    "file", "jpeg", "with", "from", "this", "that", "have", "been", "were", "into", "over",
    "under", "about", "during", "photo", "photograph", "image", "picture", "wikimedia",
    "commons", "original", "crop", "cropped", "version", "large", "small", "close", "shown",
    "taken", "using", "after", "before", "their", "there", "which", "while", "other",
    # generic place and address words that survive the length filter but carry no meaning
    "street", "avenue", "road", "square", "centre", "center", "north", "south", "east", "west",
    "upper", "lower", "great", "little", "saint", "county", "district", "region", "province",
    "geograph",
}


def _dhash(path: str, size: int = 8) -> int:
    """Difference hash, used only to pick a visually distinct burst source."""
    with Image.open(path) as img:
        small = img.convert("L").resize((size + 1, size), Image.LANCZOS)
        px = small.load()
        bits = 0
        for y in range(size):
            for x in range(size):
                bits = (bits << 1) | (1 if px[x, y] > px[x + 1, y] else 0)
        return bits


def _most_distinct(album: str, files: list[str]) -> str:
    """Return the file whose nearest perceptual neighbour is furthest away."""
    hashes = {}
    for f in files:
        try:
            hashes[f] = _dhash(os.path.join(album, f))
        except Exception:                                          # noqa: BLE001 - skip oddities
            continue

    best, best_gap = next(iter(hashes)), -1
    for a, ha in hashes.items():
        gap = min((bin(ha ^ hb).count("1") for b, hb in hashes.items() if b != a), default=64)
        if gap > best_gap:
            best, best_gap = a, gap

    return best


def main() -> int:
    ap = argparse.ArgumentParser(description="Fetch the PixelFlux real-photo test album.")
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "testdata", "album"))
    ap.add_argument("--clean", action="store_true")
    args = ap.parse_args()

    if args.clean and os.path.isdir(args.out):
        shutil.rmtree(args.out)
    os.makedirs(args.out, exist_ok=True)

    # ---- gather candidates -----------------------------------------------------------------
    seen: set[str] = set()
    chosen: list[dict] = []

    for term, want, subject in SUBJECTS:
        say(f"searching {subject:10s} {term!r}")
        for hit in search(term, want, seen):
            hit["subject"] = subject
            chosen.append(hit)
        time.sleep(0.3)   # be a good citizen against a free API

    print(f"\n{len(chosen)} candidates selected")
    if len(chosen) < 30:
        say("!! too few candidates; aborting rather than writing a thin corpus")
        return 1

    # ---- download --------------------------------------------------------------------------
    # 40 real photographs + 10 derived test cases = exactly 50. The count is arithmetic, not a
    # guess: 4 re-encodes + 2 duplicates + 3 burst frames + 1 corrupt file are added below.
    #
    # An earlier version downloaded 44 and trimmed the overflow at the end, which silently
    # deleted the last four subjects — the bicycle and both beaches vanished from a corpus whose
    # manifest still claimed nineteen subjects, and a search for "bicycle" came back empty
    # looking like a search bug. Matching the request to the arithmetic means the trim never
    # fires; the loop below stays as a backstop and now says so out loud when it does.
    REAL = 40
    written: list[dict] = []

    for hit in chosen:
        if len(written) >= REAL:
            break

        info = hit["info"]
        name = f"{len(written):03d}_{hit['subject']}_{slug(hit['title'])}.jpg"
        path = os.path.join(args.out, name)

        meta_preview = {m["name"]: m["value"] for m in (info.get("metadata") or [])}
        rating, keywords = curate(
            len(written),
            hit["subject"],
            hit["title"],
            "geotagged" if meta_preview.get("GPSLatitude") else "",
            str(meta_preview.get("Model") or ""))

        try:
            raw = download(info["url"])
            width, height = store(raw, path, rating, keywords)
        except Exception as exc:                                   # noqa: BLE001 - network tool
            say(f"  ! {hit['title']}: {exc}")
            continue

        time.sleep(1.1)   # deliberate pacing; Commons 429s a tight sequential loop

        meta = {m["name"]: m["value"] for m in (info.get("metadata") or [])}
        ext = info.get("extmetadata", {})

        written.append({
            "file": name,
            "subject": hit["subject"],
            "title": hit["title"],
            "w": width, "h": height,
            "kb": round(os.path.getsize(path) / 1024),
            "camera": meta.get("Model") or "",
            "gps": "yes" if meta.get("GPSLatitude") else "",
            "date": str(meta.get("DateTimeOriginal") or meta.get("DateTime") or ""),
            "licence": re.sub("<[^>]+>", "", str(ext.get("LicenseShortName", {}).get("value", ""))),
            "author": re.sub("<[^>]+>", "", str(ext.get("Artist", {}).get("value", "")))[:70],
            "source": info.get("descriptionurl", ""),
            "rating": rating,
            "keywords": keywords,
        })
        say(f"  {name}  {width}x{height}  {written[-1]['kb']}KB  "
              f"{written[-1]['camera'] or 'no-camera'}{'  GPS' if written[-1]['gps'] else ''}"
              f"  {'*' * rating if rating else '-'}")

    if len(written) < 30:
        say(f"!! only {len(written)} files downloaded; aborting")
        return 1

    notes: list[str] = []

    # ---- format variety: re-encode a few, which also strips their EXIF ----------------------
    # Real libraries are full of exports and screenshots with no camera metadata, and the
    # ingestion path has to date those from the filesystem instead. Re-encoding real photos into
    # other containers produces both test cases at once: format coverage and EXIF-less files.
    for offset, (fmt, extension) in enumerate([("PNG", "png"), ("WEBP", "webp"), ("TIFF", "tif"), ("PNG", "png")]):
        source = written[offset * 7]["file"]
        destination = f"{44 + offset:03d}_reencoded-{extension}-{source.split('_', 2)[1]}.{extension}"
        with Image.open(os.path.join(args.out, source)) as img:
            save_args = {"compression": "tiff_lzw"} if fmt == "TIFF" else {}
            img.save(os.path.join(args.out, destination), fmt, **save_args)
        notes.append(f"{destination}\tRE-ENCODED from {source} to {fmt} — no EXIF, tests the "
                     f"file-timestamp date fallback and {fmt} decoding")

    # ---- two byte-identical duplicates ------------------------------------------------------
    for offset in range(2):
        source = written[3 + offset * 11]["file"]
        destination = f"{48 + offset:03d}_duplicate-of-{source.split('_', 1)[1]}"
        shutil.copyfile(os.path.join(args.out, source), os.path.join(args.out, destination))
        notes.append(f"{destination}\tEXACT DUPLICATE of {source} — identical bytes and content hash")

    # ---- a three-frame burst ----------------------------------------------------------------
    # Cropped windows out of one real photograph, one to three pixels apart, all encoded at the
    # same quality. Shifting with an affine translate instead would leave a black bar on the
    # exposed edge, and that bar dominates the perceptual hash far more than the shift does.
    # The burst source must be a photograph with no near-duplicate siblings, or the derived
    # frames cluster with those siblings instead of only with each other and the dedup test
    # becomes untestable. Commons search readily returns consecutive frames from one shoot
    # (three angles of the same museum car, two shots of the same cathedral), so the source is
    # chosen by measurement: the photo whose nearest neighbour is furthest away.
    burst_source = _most_distinct(args.out, [w["file"] for w in written])
    say(f"burst source: {burst_source}")
    with Image.open(os.path.join(args.out, burst_source)) as base:
        base = base.convert("RGB")
        exif = base.info.get("exif", b"")
        pad = 6
        cw, ch = base.width - 2 * pad, base.height - 2 * pad
        for frame in range(3):
            dx, dy = (frame - 1) * 3, (frame - 1) * 2
            crop = base.crop((pad + dx, pad + dy, pad + dx + cw, pad + dy + ch))
            destination = f"{50 + frame:03d}_burst-{frame + 1}.jpg"
            crop.save(os.path.join(args.out, destination), "JPEG", quality=92, exif=exif)
            notes.append(f"{destination}\tBURST frame {frame + 1}/3 from {burst_source} — "
                         "near-duplicate, must cluster perceptually but not by content hash")

    # ---- one truncated file -------------------------------------------------------------------
    # Truncated to 900 bytes, not to half the file.
    #
    # The first version kept 50% of a valid JPEG and ImageSharp decoded it without complaint.
    # A JPEG is a sequential scan, so a decoder that tolerates a missing tail simply renders
    # whatever arrived — the fixture asserted "one unreadable file" and the library correctly
    # reported zero. 900 bytes keeps the SOI marker and part of the EXIF header but no frame
    # header and no scan data, which no decoder can turn into pixels.
    good = os.path.join(args.out, written[5]["file"])
    with open(good, "rb") as fh:
        blob = fh.read(900)
    with open(os.path.join(args.out, "053_corrupt-truncated.jpg"), "wb") as fh:
        fh.write(blob)
    notes.append("053_corrupt-truncated.jpg\tCORRUPT — first 900 bytes of a JPEG (header only, "
                 "no scan data); must be indexed as unreadable, not dropped")

    # ---- trim to exactly 50 -------------------------------------------------------------------
    files = sorted(f for f in os.listdir(args.out) if not f.endswith(".tsv"))
    if len(files) > 50:
        say(f"!! {len(files)} files, trimming to 50 — subjects will be LOST:")
    while len(files) > 50:
        # Drop from the plain real photos, never the derived test cases.
        victim = next(f for f in reversed(files)
                      if not any(k in f for k in ("duplicate", "burst", "corrupt", "reencoded")))
        say(f"   dropping {victim}")
        os.remove(os.path.join(args.out, victim))
        written = [w for w in written if w["file"] != victim]
        files.remove(victim)

    # ---- organise into year folders --------------------------------------------------------
    # The album was a single flat directory, which made "sort by folder" and the folder facet
    # completely inert — one value, 48 photos. Nobody keeps a photo library that way; year
    # folders are the most common real arrangement and give the dimension ~15 values to work
    # with. Derived test cases move with the photograph they came from, which is also what would
    # happen on a real disk.
    say("")
    for name in sorted(f for f in os.listdir(args.out) if not f.endswith(".tsv")):
        source = os.path.join(args.out, name)
        year = "Unsorted"
        try:
            with Image.open(source) as probe:
                exif_probe = probe.getexif()
                sub = exif_probe.get_ifd(0x8769)
                stamp = str(sub.get(0x9003) or exif_probe.get(0x0132) or "")
                if len(stamp) >= 4 and stamp[:4].isdigit():
                    year = stamp[:4]
        except Exception:                                          # noqa: BLE001 - corrupt fixture
            pass

        folder = os.path.join(args.out, year)
        os.makedirs(folder, exist_ok=True)
        shutil.move(source, os.path.join(folder, name))
        for record in written:
            if record["file"] == name:
                record["folder"] = year

    folders = sorted(d for d in os.listdir(args.out) if os.path.isdir(os.path.join(args.out, d)))
    say(f"organised into {len(folders)} folders: {', '.join(folders)}")

    # ---- manifest and attribution --------------------------------------------------------------
    with open(os.path.join(args.out, "MANIFEST.tsv"), "w", encoding="utf-8") as fh:
        fh.write("# PixelFlux test album — real photographs from Wikimedia Commons\n")
        fh.write(f"# {len(files)} files. Rebuild: python tools/fetch_test_album.py --clean\n")
        fh.write("# file\tdetail\n\n")
        for w in written:
            bits = [w["subject"], f"{w['w']}x{w['h']}", f"{w['kb']}KB"]
            if w["camera"]:
                bits.append(w["camera"])
            else:
                bits.append("no-camera")
            if w["gps"]:
                bits.append("GPS")
            if w["date"]:
                bits.append(w["date"][:10].replace(":", "-"))
            fh.write(f"{w['file']}\t{' | '.join(bits)}\n")
        fh.write("\n")
        for line in notes:
            fh.write(line + "\n")

    with open(os.path.join(args.out, "ATTRIBUTION.tsv"), "w", encoding="utf-8") as fh:
        fh.write("# Source, author and licence for every downloaded photograph.\n")
        fh.write("# file\tcommons title\tauthor\tlicence\tsource\n")
        for w in written:
            fh.write(f"{w['file']}\t{w['title']}\t{w['author']}\t{w['licence']}\t{w['source']}\n")

    total_mb = sum(
        os.path.getsize(os.path.join(root, f))
        for root, _, names in os.walk(args.out)
        for f in names if not f.endswith(".tsv")) / 1024 / 1024
    with_gps = sum(1 for w in written if w["gps"])
    cameras = len({w["camera"] for w in written if w["camera"]})
    print(f"\n{len(files)} files, {total_mb:.0f} MB")
    rated = sum(1 for w in written if w.get("rating"))
    keyworded = sum(1 for w in written if w.get("keywords"))
    say(f"{with_gps} with GPS, {cameras} distinct camera bodies")
    say(f"{rated} rated, {keyworded} keyworded, {len(folders)} folders")
    say(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
