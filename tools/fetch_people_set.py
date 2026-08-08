#!/usr/bin/env python3
"""Build a labelled same-person corpus for calibrating face recognition.

WHY THIS IS SEPARATE FROM THE ALBUM
-----------------------------------
testdata/album is a curated 50-photo library that stands in for somebody's real archive: varied
subjects, real EXIF, a couple of adversarial files. It is deliberately not full of the same
person over and over, because real libraries are not.

That makes it useless for the one question recognition has to answer: is this the same person as
that? The only repeat in the album is a re-encode of one file, which is pixel-identical content
and proves nothing — matching it tests JPEG, not recognition.

So this fetches a second, much smaller corpus whose entire purpose is ground truth: several
photographs of each of a handful of people, taken at different times, in different lighting, at
different angles. The label is in the filename, which is what lets a test say "these two should
match and these two should not" without anybody hand-annotating anything.

WHOSE FACES
-----------
Public figures with large freely-licensed photo sets on Wikimedia Commons. Two reasons, and the
second matters more: their categories reliably contain many photographs of one identifiable
person, and they are people whose likeness is already published under a licence that permits
this. No private individual's face ends up in a test fixture.

Nothing here is redistributed. It is local test data, and ATTRIBUTION.tsv records author,
licence, and source for every file.

USAGE
-----
    python tools/fetch_people_set.py --out testdata/people --clean
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
import urllib.error
import urllib.parse
import urllib.request

from PIL import Image

API = "https://commons.wikimedia.org/w/api.php"
UA = "PixelFlux-testdata/0.1 (local photo-manager test corpus; https://github.com/local/pixelflux)"
MAX_EDGE = 1600

# Commons categories, each holding many photographs of one person. Chosen for set size and for
# variety within the set — a category of ten near-identical press-conference shots would make
# recognition look better than it is.
#
# Four people rather than two: with two, a threshold that simply says "yes" to everything scores
# 50%. Four makes the twelve cross-person pairs outnumber the same-person ones, which is the
# ratio a real library has and the one a threshold has to survive.
PEOPLE = [
    ("Ursula von der Leyen", "vonderleyen"),
    ("Barack Obama", "obama"),
    ("Emmanuel Macron", "macron"),
    ("Justin Trudeau", "trudeau"),
    ("Kamala Harris", "harris"),
    ("Sanna Marin", "marin"),
]

# Staged, not final. Candidates are downloaded to _raw/ and only the ones that survive being
# looked at get promoted, so this is generous on purpose.
PER_PERSON = 14


def say(message: str) -> None:
    """Print without dying on a filename the console codepage cannot represent."""
    sys.stdout.write(message.encode("utf-8", "replace").decode("utf-8", "replace") + "\n")
    sys.stdout.flush()


# The hand-verified keep-list.
#
# Everything above this line is retrieval, and retrieval cannot establish identity. A category
# name is an assertion; the detector only says "there is a face here"; neither can tell Justin
# Trudeau from his mother, and one staged candidate under his name is exactly that. So the
# candidates were looked at, a contact sheet at a time, and these are the ones a person
# confirmed. Ground truth nobody checked is not ground truth.
#
# Recorded here rather than left as whatever the fetch happened to return, so the corpus is
# reproducible: rerunning rebuilds the same set instead of a new set of the same size. If
# Commons reorders a category the indices drift, which is why --stage keeps the raw candidates
# and the contact sheet has to be looked at again before this list is edited.
#
# Chosen for spread, not for count. Eight frames of one studio sitting would make recognition
# look far better than it is, so where a person's set was mostly a single session only one frame
# of it is kept and the event photographs are preferred.
# Single-subject photographs only. Group shots were on this list at first and had to come off:
# the label is per FILE, and a test can only turn that into a label per FACE by taking the
# largest one. In a photograph of Kamala Harris handing someone a diploma, the largest face is
# the graduate — so the corpus asserted that a stranger was Harris, and the measured recall for
# her was zero. That was the ground truth failing, not the model, and the two are indistinguishable
# from the outside unless the corpus refuses the ambiguity.
CURATED = {
    "vonderleyen": [0, 1, 2, 3, 4, 6],
    "obama":       [0, 3],
    "harris":      [0, 5],
    "macron":      [8, 9, 10],

    # One photograph, deliberately. A person with a single image contributes no same-person
    # pair, but they are still a distractor: one more face the threshold has to rule out. A
    # corpus of only matchable people flatters any threshold.
    "marin":       [6],
}


def promote(out: str) -> int:
    """Copy the hand-verified candidates out of _raw into the corpus proper."""
    promoted = 0

    for label, indices in CURATED.items():
        for index in indices:
            source = os.path.join(out, "_raw", label, f"{label}_{index:02d}.jpg")

            if not os.path.exists(source):
                say(f"  ! {label}_{index:02d} is on the keep-list but was not fetched")
                continue

            shutil.copy2(source, os.path.join(out, f"{label}_{index:02d}.jpg"))
            promoted += 1

    return promoted


def api(params: dict) -> dict:
    params = {**params, "format": "json", "formatversion": "2"}
    url = f"{API}?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def download(url: str, attempts: int = 5) -> bytes:
    """Fetch bytes, backing off when Commons rate-limits us."""
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


def usable(info: dict) -> bool:
    """A reasonably sized photograph, portrait-ish, not a scan or a montage."""
    if info.get("mime") not in ("image/jpeg", "image/png"):
        return False

    width, height = info.get("width", 0), info.get("height", 0)
    if width < 800 or height < 800:
        return False

    # A very wide frame is usually a group shot or a stage panorama, where the face is a few
    # dozen pixels across and recognition would be measuring noise.
    if width > height * 1.6:
        return False

    return 80_000 <= info.get("size", 0) <= 25_000_000


def subcategories(category: str, limit: int = 25) -> list[str]:
    """Direct subcategories of a category."""
    try:
        data = api({
            "action": "query",
            "list": "categorymembers",
            "cmtitle": f"Category:{category}",
            "cmtype": "subcat",
            "cmlimit": limit,
        })
    except Exception as exc:                                       # noqa: BLE001 - network tool
        say(f"  ! subcats {category}: {exc}")
        return []

    return [m["title"].removeprefix("Category:")
            for m in data.get("query", {}).get("categorymembers", [])]


def files_in(category: str, limit: int = 60) -> list[dict]:
    """Photographs that are direct file members of a category."""
    try:
        data = api({
            "action": "query",
            "generator": "categorymembers",
            "gcmtitle": f"Category:{category}",
            "gcmtype": "file",
            "gcmlimit": limit,
            "prop": "imageinfo",
            "iiprop": "url|size|mime|extmetadata",
            "iiurlwidth": MAX_EDGE,
        })
    except Exception as exc:                                       # noqa: BLE001 - network tool
        say(f"  ! files {category}: {exc}")
        return []

    return data.get("query", {}).get("pages", [])


def rank_subcategory(name: str) -> int:
    """How likely a subcategory is to hold plain photographs of the person.

    Commons organises a public figure's category into a hub of themes, and most of those themes
    are not photographs of the person at all: "in art", "Demonstrations and protests against",
    "Signatures of", a category for their official car. The first version of this script took
    subcategories in API order and came back with fourteen Merkel candidates containing no
    Merkel — protest banners, a pencil drawing, a puppet, and a photograph of a building.

    So subcategories are ranked rather than taken as they come, and the obviously wrong ones are
    excluded outright. "Portraits of X" is the jackpot: one person, one face, many lighting
    conditions, which is exactly the shape a recognition corpus wants.
    """
    lower = name.lower()

    if re.search(r"in art|protest|demonstration|signature|video|caricature|statue|"
                 r"monument|graffiti|by location|cabinet|stamp", lower):
        return -1

    # "X in 2019" beats "Portraits of X", which is not the ranking you would guess. On Commons
    # "Portraits of" means portraiture in the art-historical sense: it collects painted
    # portraits, busts, and statues alongside photographs. Merkel's portraits category is
    # mostly artwork, and taking it first produced a corpus of oil paintings. A year category
    # is unambiguously photographs of an event.
    if re.search(r"in (19|20)\d\d", lower):
        return 3

    if lower.startswith("portraits of"):
        return 2

    # "X with national leaders", "X and Y" — real photographs, but group shots where the face is
    # small and another person's face is just as prominent. Usable, but last.
    return 1


def subcategories(category: str, limit: int = 50) -> list[str]:
    """Direct subcategories of a category, best first."""
    try:
        data = api({
            "action": "query",
            "list": "categorymembers",
            "cmtitle": f"Category:{category}",
            "cmtype": "subcat",
            "cmlimit": limit,
        })
    except Exception as exc:                                       # noqa: BLE001 - network tool
        say(f"  ! subcats {category}: {exc}")
        return []

    names = [m["title"].removeprefix("Category:")
             for m in data.get("query", {}).get("categorymembers", [])]

    ranked = [(rank_subcategory(n), n) for n in names]
    return [n for rank, n in sorted(ranked, key=lambda r: -r[0]) if rank > 0]


def files_in(category: str, limit: int = 60) -> list[dict]:
    """Photographs that are direct file members of a category."""
    try:
        data = api({
            "action": "query",
            "generator": "categorymembers",
            "gcmtitle": f"Category:{category}",
            "gcmtype": "file",
            "gcmlimit": limit,
            "prop": "imageinfo",
            "iiprop": "url|size|mime|extmetadata",
            "iiurlwidth": MAX_EDGE,
        })
    except Exception as exc:                                       # noqa: BLE001 - network tool
        say(f"  ! files {category}: {exc}")
        return []

    return data.get("query", {}).get("pages", [])


def candidates(category: str, want: int) -> list[dict]:
    """Photographs of one person, gathered from the most promising subcategories first.

    Category membership only — never a name search. A search for "Angela Merkel" returns every
    file that merely mentions her, including a photograph of a house she once lived in.

    Category membership is not proof either: a category holds group shots, and "Portraits of"
    occasionally holds a painting. So candidates are staged and looked at before any of them is
    promoted to ground truth. This function's job is to make that review short, not to skip it.
    """
    picked: list[dict] = []
    subs = subcategories(category)

    # Two levels. "X by year" is itself a hub of "X in 2013", "X in 2014" — the photographs are
    # one further down, which is where they always are for anyone well documented.
    for sub in list(subs):
        if sub.lower().endswith("by year"):
            subs.extend(subcategories(sub)[:6])
            time.sleep(0.3)

    # Subcategories first, the parent category last and only as a fallback. The parent is the
    # hub: its own direct files are the leftovers that did not fit any theme, which for a
    # well-documented public figure means protest placards and photographs of buildings. Taking
    # the parent first filled the entire quota with those and never reached "Portraits of".
    for source in [*subs, category]:
        if len(picked) >= want:
            break

        picked += _pick(files_in(source, 40), want - len(picked),
                        already={p["_title"] for p in picked})
        time.sleep(0.4)

    return picked[:want]


def _pick(pages: list[dict], want: int, already: set[str] | None = None) -> list[dict]:
    """Filter candidate pages down to usable photographs."""
    picked: list[dict] = []
    seen = set(already or ())

    for page in pages:
        if not page.get("imageinfo"):
            continue

        info = page["imageinfo"][0]
        if not usable(info):
            continue

        title = page["title"]
        if title in seen:
            continue

        # Skip anything whose title says it is not a plain photograph of the person.
        if re.search(r"signature|logo|coat of arms|map|diagram|svg|poster|plaque|banner"
                     r"|graffiti|street|building|memorial|stamp|letter|document",
                     title, re.IGNORECASE):
            continue

        info["_title"] = title
        seen.add(title)
        picked.append(info)

        if len(picked) >= want:
            break

    return picked


def store(raw: bytes, path: str) -> tuple[int, int]:
    """Downscale and write, preserving EXIF where the source has it."""
    image = Image.open(io.BytesIO(raw))
    exif = image.info.get("exif")

    if image.mode not in ("RGB", "L"):
        image = image.convert("RGB")

    if max(image.size) > MAX_EDGE:
        scale = MAX_EDGE / max(image.size)
        image = image.resize(
            (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
            Image.LANCZOS)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    image.save(path, "JPEG", quality=88, **({"exif": exif} if exif else {}))
    return image.size


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join("testdata", "people"))
    ap.add_argument("--clean", action="store_true")
    ap.add_argument("--stage", action="store_true",
                    help="fetch candidates only; do not promote the keep-list")
    args = ap.parse_args()

    if args.clean and os.path.isdir(args.out):
        shutil.rmtree(args.out)

    os.makedirs(args.out, exist_ok=True)
    attribution = [("file", "author", "licence", "source")]
    written = 0

    for category, label in PEOPLE:
        say(f"{category}")
        found = candidates(category, PER_PERSON)

        if len(found) < 2:
            # Fewer than two is not a shortfall, it is a hole: a person with one photograph
            # contributes no same-person pair at all, which is the only thing this corpus is for.
            say(f"  ! only {len(found)} usable photographs — this person proves nothing")

        for index, info in enumerate(found):
            path = os.path.join(args.out, "_raw", label, f"{label}_{index:02d}.jpg")

            try:
                raw = download(info.get("thumburl") or info["url"])
                width, height = store(raw, path)
            except Exception as exc:                               # noqa: BLE001 - network tool
                say(f"  ! {info['_title']}: {exc}")
                continue

            meta = info.get("extmetadata") or {}
            author = re.sub(r"<[^>]+>", "", (meta.get("Artist") or {}).get("value", "")).strip()
            licence = (meta.get("LicenseShortName") or {}).get("value", "")

            attribution.append((f"{label}/{os.path.basename(path)}", author or "unknown",
                                licence or "unknown", info.get("descriptionurl", "")))

            say(f"  {os.path.basename(path):<22} {width}x{height}")
            written += 1
            time.sleep(0.9)                                        # pace, or Commons 429s

    with open(os.path.join(args.out, "ATTRIBUTION.tsv"), "w", encoding="utf-8", newline="") as fh:
        for row in attribution:
            fh.write("\t".join(row) + "\n")

    say(f"\n{written} photographs of {len(PEOPLE)} people in {args.out}")
    if args.stage:
        say("staged only; review the contact sheet, then rerun without --stage to promote")
        return 0

    promoted = promote(args.out)
    say(f"{promoted} hand-verified photographs promoted to {args.out}")
    return 0 if promoted else 1


if __name__ == "__main__":
    raise SystemExit(main())
