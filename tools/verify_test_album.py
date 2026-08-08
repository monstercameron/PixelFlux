#!/usr/bin/env python3
"""Verify the test album actually has the properties the tests depend on.

A fixture corpus that quietly fails to carry EXIF, or whose "duplicates" differ by a byte, is
worse than no corpus: every test built on it passes for the wrong reason. This script is the
check on the fetcher, and it asserts the specific claims fetch_test_album.py makes rather than
eyeballing the output.

One of these assertions exists because of a bug it would have caught. An earlier synthetic
generator assigned GPS locations with `i % 8` while assigning file formats with the same period,
so two of its eight locations could never receive a fix — the corpus advertised eight countries
and shipped six, and a bounding-box search for Colombia came back empty looking like a product
bug. Counting DISTINCT values, not just files that have one, is the check that catches that
class of mistake.
"""

from __future__ import annotations

import collections
import hashlib
import os
import sys

from PIL import Image

ALBUM = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "testdata", "album")

GPS_IFD = 0x8825
EXIF_IFD = 0x8769


def dms_to_deg(dms, ref) -> float:
    d, m, s = (float(x) for x in dms)
    val = d + m / 60 + s / 3600
    return -val if ref in ("S", "W") else val


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    files = sorted(f for f in os.listdir(ALBUM) if not f.endswith(".tsv"))
    print(f"ALBUM {ALBUM}\nFILES {len(files)}\n")

    formats = collections.Counter()
    subjects = collections.Counter()
    cameras = collections.Counter()
    years = collections.Counter()
    places: set[tuple[float, float]] = set()
    aspects = collections.Counter()
    hashes: dict[str, list[str]] = collections.defaultdict(list)
    corrupt: list[str] = []
    no_exif = 0
    total_bytes = 0

    for name in files:
        path = os.path.join(ALBUM, name)
        total_bytes += os.path.getsize(path)
        hashes[sha256(path)].append(name)

        # Subject label is the second underscore-delimited segment for downloaded photos.
        parts = name.split("_")
        if len(parts) >= 3:
            subjects[parts[1]] += 1

        try:
            with Image.open(path) as img:
                img.load()
                formats[img.format] += 1
                ratio = img.width / img.height
                aspects["landscape" if ratio > 1.15 else
                        "portrait" if ratio < 0.87 else "square"] += 1

                exif = img.getexif()
                if not exif:
                    no_exif += 1
                    continue

                if model := exif.get(0x0110):
                    cameras[str(model).strip()] += 1

                sub = exif.get_ifd(EXIF_IFD)
                if when := sub.get(0x9003):
                    years[str(when)[:4]] += 1

                gps = exif.get_ifd(GPS_IFD)
                if gps and 2 in gps and 4 in gps:
                    places.add((
                        round(dms_to_deg(gps[2], gps.get(1, "N")), 2),
                        round(dms_to_deg(gps[4], gps.get(3, "E")), 2)))
        except Exception as exc:                                   # noqa: BLE001 - that's the point
            corrupt.append(f"{name}: {type(exc).__name__}")

    print("FORMATS   ", dict(formats))
    print("ASPECTS   ", dict(aspects))
    print("SUBJECTS  ", dict(subjects))
    print("YEARS     ", dict(sorted(years.items())))
    print(f"CAMERAS    {len(cameras)} distinct bodies")
    for m, c in cameras.most_common(8):
        print(f"           {c:>2}x {m}")
    print(f"PLACES     {len(places)} distinct GPS fixes")
    print(f"NO EXIF    {no_exif} files")
    print(f"SIZE       {total_bytes/1024/1024:.0f} MB total, "
          f"{total_bytes/max(len(files),1)/1024:.0f} KB average")
    print(f"UNREADABLE {corrupt or 'none'}")

    dupes = {h: n for h, n in hashes.items() if len(n) > 1}
    print(f"\nEXACT DUPLICATE GROUPS ({len(dupes)}):")
    for h, names in dupes.items():
        print(f"  {h[:12]}  {names}")

    burst = sorted(n for n in files if "burst" in n)
    print(f"\nBURST FRAMES: {len(burst)}")

    print("\n--- ASSERTIONS ---")
    ok = True

    def check(label: str, cond: bool, detail: str = "") -> None:
        nonlocal ok
        ok &= cond
        print(f"  [{'PASS' if cond else 'FAIL'}] {label}{('  ' + detail) if detail else ''}")

    check("50 files", len(files) == 50, f"got {len(files)}")
    check("real photographs, not drawings",
          total_bytes / max(len(files), 1) > 200 * 1024,
          f"{total_bytes/max(len(files),1)/1024:.0f} KB average")
    check(">=3 container formats", len(formats) >= 3, str(sorted(formats)))
    check(">=8 distinct subjects", len(subjects) >= 8, f"{len(subjects)}: {sorted(subjects)}")
    check("faces present", subjects.get("face", 0) >= 3, f"{subjects.get('face', 0)}")
    check("cars present", subjects.get("car", 0) >= 3, f"{subjects.get('car', 0)}")
    check("doors present", subjects.get("door", 0) >= 3, f"{subjects.get('door', 0)}")
    check("shoes present", subjects.get("shoes", 0) >= 2, f"{subjects.get('shoes', 0)}")
    check(">=15 camera bodies", len(cameras) >= 15, f"got {len(cameras)}")
    # DISTINCT fixes, not files-with-a-fix. See the module docstring.
    check(">=15 distinct GPS fixes", len(places) >= 15, f"got {len(places)}")
    check("dates span >=5 years", len(years) >= 5, str(sorted(years)))
    check("some files lack EXIF", no_exif >= 3, f"{no_exif} files")
    check("2 exact-duplicate groups", len(dupes) == 2, f"{len(dupes)} groups")
    check("3 burst frames", len(burst) == 3, f"{len(burst)}")
    check("exactly 1 unreadable file", len(corrupt) == 1, str(corrupt))
    check("attribution recorded",
          os.path.exists(os.path.join(ALBUM, "ATTRIBUTION.tsv")))

    print("\nRESULT:", "OK" if ok else "PROBLEMS FOUND")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
