#!/usr/bin/env python3
"""Build the offline gazetteer PixelFlux ships for reverse geocoding.

WHY THIS IS A BUILD STEP AND NOT AN API CALL
--------------------------------------------
Every reverse-geocoding service is a network call, and PixelFlux makes none — the content policy
in index.html blocks outbound requests outright, and the whole premise of the application is
that your photo library never leaves your machine. Sending the GPS coordinates of someone's
photographs to a third party in order to print "London" under them would quietly undo that.

So the lookup table ships inside the binary. This script turns GeoNames' public data into a
compact form small enough to embed, and is run when the data needs refreshing — not at runtime,
and not during an ordinary build.

WHAT IT PRODUCES
----------------
`src/PixelFlux.Core/Geo/gazetteer.bin`, a little-endian binary blob:

    magic     4 bytes   "PFGZ"
    version   1 x int32
    countries 1 x int32 count, then per country: 2-char code + length-prefixed UTF-8 name
    cities    1 x int32 count, then per city:
                  float32 latitude
                  float32 longitude
                  uint16  country index
                  uint8   name byte length
                  bytes   UTF-8 name

Fixed-width numerics up front mean the loader can slurp the whole thing without parsing text,
which matters because it is read on every application start.

SOURCE AND LICENCE
------------------
GeoNames (https://geonames.org), CC BY 4.0. Attribution is written to gazetteer-ATTRIBUTION.txt
next to the data and must ship with the application.

USAGE
-----
    python tools/build_gazetteer.py [--population 15000]
"""

from __future__ import annotations

import argparse
import io
import os
import struct
import sys
import urllib.request
import zipfile

BASE = "https://download.geonames.org/export/dump"
UA = "PixelFlux-gazetteer/0.1 (local photo manager; offline reverse geocoding)"

OUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "src", "PixelFlux.Core", "Geo")

MAGIC = b"PFGZ"
VERSION = 1


def say(message: str) -> None:
    sys.stdout.write(message.encode("utf-8", "replace").decode("utf-8", "replace") + "\n")
    sys.stdout.flush()


def fetch(name: str) -> bytes:
    request = urllib.request.Request(f"{BASE}/{name}", headers={"User-Agent": UA})
    with urllib.request.urlopen(request, timeout=180) as response:
        return response.read()


def load_countries() -> dict[str, str]:
    """ISO-3166 alpha-2 code to English country name."""
    raw = fetch("countryInfo.txt").decode("utf-8")
    countries: dict[str, str] = {}

    for line in raw.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) > 4 and len(parts[0]) == 2:
            countries[parts[0]] = parts[4]

    return countries


def load_cities(population: int) -> list[tuple[float, float, str, str]]:
    """(lat, lon, name, country code) for populated places above a threshold."""
    archive = fetch(f"cities{population}.zip")
    with zipfile.ZipFile(io.BytesIO(archive)) as zf:
        raw = zf.read(f"cities{population}.txt").decode("utf-8")

    cities = []
    for line in raw.splitlines():
        parts = line.split("\t")
        if len(parts) < 9:
            continue

        try:
            latitude = float(parts[4])
            longitude = float(parts[5])
        except ValueError:
            continue

        # Column 1 is the ASCII-folded name; column 2 is the UTF-8 one. The UTF-8 name is what
        # people actually call the place, and the app renders it in whatever script it belongs
        # to — "München", not "Muenchen".
        name = parts[1].strip()
        country = parts[8].strip()

        if name and len(country) == 2:
            cities.append((latitude, longitude, name, country))

    return cities


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the PixelFlux offline gazetteer.")
    ap.add_argument("--population", type=int, default=15000,
                    choices=[500, 1000, 5000, 15000],
                    help="GeoNames population threshold. Lower means better coverage and a "
                         "bigger file; 15000 is ~25k places and about 700 KB.")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    say("fetching countryInfo.txt")
    countries = load_countries()
    say(f"  {len(countries)} countries")

    say(f"fetching cities{args.population}.zip")
    cities = load_cities(args.population)
    say(f"  {len(cities)} populated places")

    if len(cities) < 1000:
        say("!! implausibly few cities; refusing to write a broken gazetteer")
        return 1

    # Only keep countries that a city actually references, so the country table stays honest
    # about what the data can resolve to.
    used = sorted({c[3] for c in cities if c[3] in countries})
    index = {code: i for i, code in enumerate(used)}

    buffer = io.BytesIO()
    buffer.write(MAGIC)
    buffer.write(struct.pack("<i", VERSION))

    buffer.write(struct.pack("<i", len(used)))
    for code in used:
        name = countries[code].encode("utf-8")
        buffer.write(code.encode("ascii"))
        buffer.write(struct.pack("<B", min(len(name), 255)))
        buffer.write(name[:255])

    resolvable = [c for c in cities if c[3] in index]
    buffer.write(struct.pack("<i", len(resolvable)))

    for latitude, longitude, name, country in resolvable:
        encoded = name.encode("utf-8")[:255]
        buffer.write(struct.pack("<ffHB", latitude, longitude, index[country], len(encoded)))
        buffer.write(encoded)

    path = os.path.join(OUT_DIR, "gazetteer.bin")
    with open(path, "wb") as fh:
        fh.write(buffer.getvalue())

    attribution = os.path.join(OUT_DIR, "gazetteer-ATTRIBUTION.txt")
    with open(attribution, "w", encoding="utf-8") as fh:
        fh.write(
            "PixelFlux offline gazetteer\n"
            "===========================\n\n"
            "Place names and coordinates are derived from the GeoNames geographical database.\n\n"
            "  Source:  https://www.geonames.org/\n"
            f"  Dataset: cities{args.population}.txt and countryInfo.txt\n"
            "  Licence: Creative Commons Attribution 4.0 (CC BY 4.0)\n"
            "           https://creativecommons.org/licenses/by/4.0/\n\n"
            f"Contains {len(resolvable)} populated places across {len(used)} countries.\n"
            "Rebuild with: python tools/build_gazetteer.py\n\n"
            "This attribution must ship with any distribution of PixelFlux.\n")

    say(f"\nwrote {path}  ({os.path.getsize(path) / 1024:.0f} KB)")
    say(f"      {len(resolvable)} places, {len(used)} countries")
    say(f"      attribution -> {attribution}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
