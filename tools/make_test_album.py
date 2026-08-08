#!/usr/bin/env python3
"""Generate PixelFlux's 50-image test album.

WHY THIS EXISTS
---------------
PixelFlux is a photo manager, and almost every interesting bug in a photo manager is a
*corpus* bug rather than a logic bug: the duplicate detector that never sees a duplicate, the
EXIF parser that only ever meets one camera, the date filter that was only tested inside a
single year, the slideshow that has never had to avoid a burst sequence. A handful of ad-hoc
JPEGs will not surface any of those.

So this corpus is built to be adversarial on purpose. It contains, by construction:

  * four container formats (JPEG, PNG, WebP, TIFF)
  * four aspect classes (landscape, portrait, square, panorama)
  * EXIF from seven distinct camera bodies, including phones and a drone
  * capture dates spread across 2019-2026, deliberately out of order on disk
  * GPS fixes in six countries, so "photos from Colombia in 2025" is answerable
  * two byte-identical duplicates under different filenames
  * a three-frame burst of near-duplicates that differ by a few pixels
  * files with no EXIF at all
  * one deliberately truncated file

Every one of those is a test case that the ingestion pipeline, the hash pair, the metadata
reader, and the search layer have to survive.

DETERMINISM
-----------
Everything is driven by a fixed seed and fixed timestamps. The same invocation produces
byte-identical output, which is what allows the content-hash tests to assert against known
digests rather than merely "two runs agreed". Do not introduce time.time() or an unseeded
random here.

USAGE
-----
    python tools/make_test_album.py [--out testdata/album] [--seed 20260807]
"""

from __future__ import annotations

import argparse
import math
import os
import random
import shutil
from dataclasses import dataclass, field

from PIL import Image, ImageDraw, ImageFilter
from PIL.TiffImagePlugin import IFDRational

# --------------------------------------------------------------------------------------
# EXIF tag numbers. Spelled out rather than imported so the intent is readable at the call
# site — PIL's ExifTags maps are keyed the other way round and make this harder to follow.
# --------------------------------------------------------------------------------------
TAG_MAKE = 0x010F
TAG_MODEL = 0x0110
TAG_ORIENTATION = 0x0112
TAG_SOFTWARE = 0x0131
TAG_DATETIME = 0x0132
TAG_EXIF_IFD = 0x8769
TAG_GPS_IFD = 0x8825
TAG_EXPOSURE_TIME = 0x829A
TAG_FNUMBER = 0x829D
TAG_ISO = 0x8827
TAG_DATETIME_ORIGINAL = 0x9003
TAG_DATETIME_DIGITIZED = 0x9004
TAG_FOCAL_LENGTH = 0x920A
TAG_LENS_MODEL = 0xA434
TAG_PIXEL_X = 0xA002
TAG_PIXEL_Y = 0xA003

GPS_LAT_REF, GPS_LAT, GPS_LON_REF, GPS_LON, GPS_ALT_REF, GPS_ALT = 1, 2, 3, 4, 5, 6


# --------------------------------------------------------------------------------------
# Camera bodies and places. Real-world values, because a search test that asks for
# "pictures taken with my Canon" should be matching a string a real file would carry.
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Camera:
    make: str
    model: str
    lens: str
    iso_range: tuple[int, int]
    fnums: tuple[float, ...]
    focals: tuple[int, ...]


CAMERAS = [
    Camera("Canon", "Canon EOS R6 Mark II", "RF24-70mm F2.8 L IS USM", (100, 6400), (2.8, 4.0, 5.6), (24, 35, 50, 70)),
    Camera("NIKON CORPORATION", "NIKON Z 7II", "NIKKOR Z 14-30mm f/4 S", (64, 3200), (4.0, 5.6, 8.0), (14, 20, 24, 30)),
    Camera("Sony", "ILCE-7M4", "FE 85mm F1.8", (100, 12800), (1.8, 2.8, 4.0), (85,)),
    Camera("Apple", "iPhone 15 Pro", "iPhone 15 Pro back triple camera 6.765mm f/1.78", (32, 2000), (1.78, 2.2), (6, 13, 24)),
    Camera("Google", "Pixel 9 Pro", "Pixel 9 Pro back camera 6.9mm f/1.68", (50, 3000), (1.68, 2.2), (6, 20)),
    Camera("FUJIFILM", "X-T5", "XF16-55mmF2.8 R LM WR", (125, 6400), (2.8, 4.0), (16, 23, 35, 55)),
    Camera("DJI", "FC7303", "DJI Mini 4 Pro", (100, 1600), (1.7,), (6,)),
]


@dataclass(frozen=True)
class Place:
    name: str
    country: str
    lat: float
    lon: float
    alt: int


PLACES = [
    Place("Kingston", "Jamaica", 17.9714, -76.7931, 9),
    Place("Negril", "Jamaica", 18.2683, -78.3475, 3),
    Place("Fort Lauderdale", "United States", 26.1224, -80.1373, 2),
    Place("Cartagena", "Colombia", 10.3910, -75.4794, 4),
    Place("Medellin", "Colombia", 6.2442, -75.5812, 1495),
    Place("Kyoto", "Japan", 35.0116, 135.7681, 56),
    Place("Reykjavik", "Iceland", 64.1466, -21.9426, 61),
    Place("Lisbon", "Portugal", 38.7223, -9.1393, 100),
]


def _rational(value: float, denom: int = 1000) -> IFDRational:
    """Convert a float to the rational form EXIF stores numbers in."""
    return IFDRational(round(value * denom), denom)


def _dms(value: float) -> tuple[IFDRational, IFDRational, IFDRational]:
    """Convert signed decimal degrees to the (deg, min, sec) rationals EXIF GPS uses."""
    value = abs(value)
    deg = int(value)
    minutes_full = (value - deg) * 60
    minutes = int(minutes_full)
    seconds = (minutes_full - minutes) * 60
    return IFDRational(deg, 1), IFDRational(minutes, 1), IFDRational(round(seconds * 1000), 1000)


def build_exif(
    camera: Camera,
    place: Place | None,
    when: str,
    size: tuple[int, int],
    rng: random.Random,
    orientation: int = 1,
) -> Image.Exif:
    """Assemble a plausible EXIF block.

    ``when`` is an EXIF-format timestamp (``YYYY:MM:DD HH:MM:SS``). It is passed in rather
    than generated here so the caller controls the date spread — the corpus needs dates that
    are deliberately unordered relative to filenames, and that ordering is a property of the
    album, not of any single photo.
    """
    exif = Image.Exif()
    exif[TAG_MAKE] = camera.make
    exif[TAG_MODEL] = camera.model
    exif[TAG_SOFTWARE] = "PixelFlux fixture generator"
    exif[TAG_DATETIME] = when
    exif[TAG_ORIENTATION] = orientation

    sub = exif.get_ifd(TAG_EXIF_IFD)
    sub[TAG_DATETIME_ORIGINAL] = when
    sub[TAG_DATETIME_DIGITIZED] = when
    sub[TAG_ISO] = rng.choice([camera.iso_range[0], 200, 400, 800, camera.iso_range[1]])
    sub[TAG_FNUMBER] = _rational(rng.choice(camera.fnums), 100)
    sub[TAG_EXPOSURE_TIME] = IFDRational(1, rng.choice([60, 125, 250, 500, 1000]))
    sub[TAG_FOCAL_LENGTH] = _rational(float(rng.choice(camera.focals)), 10)
    sub[TAG_LENS_MODEL] = camera.lens
    sub[TAG_PIXEL_X] = size[0]
    sub[TAG_PIXEL_Y] = size[1]

    if place is not None:
        gps = exif.get_ifd(TAG_GPS_IFD)
        gps[GPS_LAT_REF] = "N" if place.lat >= 0 else "S"
        gps[GPS_LAT] = _dms(place.lat)
        gps[GPS_LON_REF] = "E" if place.lon >= 0 else "W"
        gps[GPS_LON] = _dms(place.lon)
        gps[GPS_ALT_REF] = 0
        gps[GPS_ALT] = IFDRational(max(place.alt, 0), 1)

    return exif


# --------------------------------------------------------------------------------------
# Drawing primitives.
#
# These are intentionally crude — the corpus is testing plumbing, not model accuracy — but
# each scene has to be *visually distinct* from the others, otherwise the perceptual-hash
# tests cannot tell a genuine near-duplicate from two different photos and the whole dedup
# test becomes meaningless. Distinctness is the requirement; beauty is not.
# --------------------------------------------------------------------------------------
def vgrad(img: Image.Image, top: tuple[int, int, int], bottom: tuple[int, int, int], y0: int = 0, y1: int | None = None) -> None:
    """Paint a vertical linear gradient over a band of the image."""
    w, h = img.size
    y1 = h if y1 is None else y1
    span = max(y1 - y0, 1)
    d = ImageDraw.Draw(img)
    for y in range(y0, y1):
        t = (y - y0) / span
        d.line([(0, y), (w, y)], fill=tuple(int(top[i] + (bottom[i] - top[i]) * t) for i in range(3)))


def grain(img: Image.Image, rng: random.Random, amount: int = 6, density: float = 0.02) -> None:
    """Sprinkle per-pixel noise so encoders produce realistic, non-degenerate file sizes.

    Without this, flat synthetic images compress to a few kilobytes and every JPEG quality
    setting yields nearly the same file — which would make file-size-dependent behaviour
    (proxy sizing, thumbnail budgets) untestable.
    """
    px = img.load()
    w, h = img.size
    for _ in range(int(w * h * density)):
        x, y = rng.randrange(w), rng.randrange(h)
        r, g, b = px[x, y][:3]
        n = rng.randint(-amount, amount)
        px[x, y] = (max(0, min(255, r + n)), max(0, min(255, g + n)), max(0, min(255, b + n)))


def ridge(d: ImageDraw.ImageDraw, w: int, base_y: int, height: int, colour, rng: random.Random, steps: int = 14) -> None:
    """Draw a jagged mountain/hill silhouette across the full width."""
    pts = [(0, base_y)]
    for i in range(steps + 1):
        x = int(w * i / steps)
        y = base_y - int(height * (0.35 + 0.65 * rng.random()) * math.sin(math.pi * i / steps) ** 0.6)
        pts.append((x, y))
    pts.append((w, base_y))
    d.polygon(pts, fill=colour)


def blobs(d: ImageDraw.ImageDraw, rng: random.Random, count: int, box, radius, colour_fn) -> None:
    """Scatter filled ellipses inside a bounding box — foliage, crowds, pebbles, stars."""
    x0, y0, x1, y1 = box
    for _ in range(count):
        cx, cy = rng.randint(x0, x1), rng.randint(y0, y1)
        r = rng.randint(*radius)
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=colour_fn(rng))


def jitter(colour: tuple[int, int, int], rng: random.Random, amount: int = 18) -> tuple[int, int, int]:
    """Nudge a colour so repeated elements do not look stamped."""
    return tuple(max(0, min(255, c + rng.randint(-amount, amount))) for c in colour)


# --------------------------------------------------------------------------------------
# Scenes. Each returns the semantic tags a correct tagger ought to produce, which doubles as
# the expected-value table for search tests.
# --------------------------------------------------------------------------------------
def scene_beach(img, rng, sunset: bool):
    w, h = img.size
    horizon = int(h * 0.52)
    if sunset:
        vgrad(img, (250, 140, 70), (255, 215, 150), 0, horizon)
        sea, sand = (150, 90, 70), (215, 180, 135)
    else:
        vgrad(img, (95, 165, 230), (185, 220, 245), 0, horizon)
        sea, sand = (40, 115, 165), (232, 212, 165)
    d = ImageDraw.Draw(img)
    sun_y = horizon - 30 if sunset else int(h * 0.15)
    d.ellipse([w * 0.62, sun_y - 55, w * 0.62 + 110, sun_y + 55], fill=(255, 240, 190))
    d.rectangle([0, horizon, w, int(h * 0.72)], fill=sea)
    for i in range(18):  # wave crests
        y = horizon + rng.randint(4, int(h * 0.19))
        x = rng.randint(0, w)
        d.line([(x, y), (x + rng.randint(30, 90), y)], fill=jitter((225, 235, 240), rng), width=3)
    d.rectangle([0, int(h * 0.72), w, h], fill=sand)
    ux = int(w * 0.28)
    d.polygon([(ux, int(h * 0.62)), (ux - 105, int(h * 0.75)), (ux + 105, int(h * 0.75))], fill=(205, 55, 55))
    d.rectangle([ux - 5, int(h * 0.75), ux + 5, int(h * 0.92)], fill=(95, 72, 50))
    return ["beach", "sea", "sand", "umbrella", "sunset" if sunset else "daylight", "outdoor", "vacation"]


def scene_mountains(img, rng, snowy: bool):
    w, h = img.size
    vgrad(img, (60, 110, 180), (200, 225, 245), 0, int(h * 0.6))
    d = ImageDraw.Draw(img)
    ridge(d, w, int(h * 0.68), int(h * 0.42), (70, 85, 105), rng)
    ridge(d, w, int(h * 0.76), int(h * 0.34), (95, 112, 130), rng)
    if snowy:
        ridge(d, w, int(h * 0.62), int(h * 0.30), (235, 240, 248), rng, steps=9)
    d.rectangle([0, int(h * 0.76), w, h], fill=(58, 92, 62) if not snowy else (225, 232, 240))
    return ["mountain", "landscape", "outdoor", "snow" if snowy else "forest", "sky", "hiking"]


def scene_city_night(img, rng):
    w, h = img.size
    vgrad(img, (8, 10, 30), (55, 40, 75), 0, int(h * 0.7))
    d = ImageDraw.Draw(img)
    blobs(d, rng, 90, (0, 0, w, int(h * 0.45)), (1, 2), lambda r: (255, 255, jitter((240,), r)[0]))
    x = 0
    while x < w:  # skyline of towers with lit windows
        bw = rng.randint(40, 95)
        bh = rng.randint(int(h * 0.22), int(h * 0.55))
        top = h - bh
        d.rectangle([x, top, x + bw, h], fill=jitter((25, 28, 42), rng, 10))
        for wy in range(top + 12, h - 10, 22):
            for wx in range(x + 8, x + bw - 10, 16):
                if rng.random() < 0.45:
                    d.rectangle([wx, wy, wx + 7, wy + 11], fill=jitter((255, 210, 120), rng, 30))
        x += bw + rng.randint(4, 14)
    return ["city", "night", "skyline", "buildings", "urban", "lights"]


def scene_forest(img, rng, autumn: bool):
    w, h = img.size
    vgrad(img, (150, 195, 220), (215, 230, 235), 0, int(h * 0.5))
    d = ImageDraw.Draw(img)
    d.rectangle([0, int(h * 0.72), w, h], fill=(72, 92, 55) if not autumn else (120, 92, 48))
    palette = [(190, 95, 35), (205, 140, 40), (150, 60, 30)] if autumn else [(40, 95, 45), (55, 120, 55), (30, 78, 38)]
    for i in range(14):  # trunks back-to-front for a sense of depth
        tx = rng.randint(-20, w)
        tw = rng.randint(14, 34)
        d.rectangle([tx, int(h * 0.35), tx + tw, int(h * 0.80)], fill=jitter((78, 58, 40), rng))
        blobs(d, rng, 9, (tx - 60, int(h * 0.18), tx + 60, int(h * 0.45)), (28, 55), lambda r: jitter(rng.choice(palette), r))
    return ["forest", "trees", "nature", "outdoor", "autumn" if autumn else "green", "woods"]


def scene_desert(img, rng):
    w, h = img.size
    vgrad(img, (240, 190, 120), (250, 230, 190), 0, int(h * 0.45))
    d = ImageDraw.Draw(img)
    d.ellipse([w * 0.7, h * 0.10, w * 0.7 + 90, h * 0.10 + 90], fill=(255, 250, 220))
    for i, shade in enumerate([(215, 170, 110), (232, 192, 132), (245, 214, 160)]):
        ridge(d, w, int(h * (0.55 + i * 0.14)), int(h * 0.18), shade, rng, steps=6)
    return ["desert", "sand", "dunes", "hot", "outdoor", "arid"]


def scene_lake(img, rng):
    w, h = img.size
    mid = int(h * 0.5)
    vgrad(img, (110, 165, 215), (215, 230, 240), 0, mid)
    d = ImageDraw.Draw(img)
    ridge(d, w, mid, int(h * 0.26), (78, 100, 88), rng, steps=8)
    # reflection: the same scene flipped and dimmed
    top = img.crop((0, 0, w, mid)).transpose(Image.FLIP_TOP_BOTTOM).point(lambda v: int(v * 0.72))
    img.paste(top, (0, mid))
    for y in range(mid, h, 9):
        d.line([(0, y), (w, y)], fill=(150, 180, 200), width=1)
    return ["lake", "water", "reflection", "landscape", "calm", "outdoor"]


def scene_dog(img, rng):
    w, h = img.size
    vgrad(img, (140, 195, 240), (205, 230, 245), 0, int(h * 0.5))
    d = ImageDraw.Draw(img)
    d.rectangle([0, int(h * 0.5), w, h], fill=(92, 155, 78))
    blobs(d, rng, 200, (0, int(h * 0.5), w, h), (2, 5), lambda r: jitter((78, 140, 66), r))
    coat = rng.choice([(150, 100, 55), (60, 55, 52), (225, 215, 195)])
    cx, cy = int(w * 0.45), int(h * 0.66)
    d.ellipse([cx - 110, cy - 45, cx + 90, cy + 55], fill=coat)          # body
    d.ellipse([cx + 60, cy - 90, cx + 165, cy + 5], fill=coat)           # head
    d.ellipse([cx + 145, cy - 55, cx + 185, cy - 18], fill=(38, 30, 26)) # snout
    d.ellipse([cx + 100, cy - 70, cx + 116, cy - 54], fill=(20, 18, 16)) # eye
    d.polygon([(cx + 70, cy - 88), (cx + 60, cy - 135), (cx + 105, cy - 96)], fill=jitter(coat, rng, 25))
    for lx in (cx - 85, cx - 30, cx + 20, cx + 62):
        d.rectangle([lx, cy + 40, lx + 20, cy + 105], fill=jitter(coat, rng, 15))
    d.line([cx - 108, cy - 20, cx - 165, cy - 75], fill=coat, width=17)
    return ["dog", "pet", "animal", "grass", "outdoor", "park"]


def scene_cat(img, rng):
    w, h = img.size
    img.paste((198, 178, 155), (0, 0, w, h))
    d = ImageDraw.Draw(img)
    d.rectangle([0, int(h * 0.7), w, h], fill=(140, 100, 70))            # floor
    d.rectangle([int(w * 0.08), int(h * 0.08), int(w * 0.42), int(h * 0.52)], fill=(225, 235, 245))  # window
    coat = rng.choice([(90, 88, 92), (220, 170, 90), (245, 245, 245)])
    cx, cy = int(w * 0.62), int(h * 0.60)
    d.ellipse([cx - 95, cy - 40, cx + 95, cy + 90], fill=coat)
    d.ellipse([cx - 58, cy - 130, cx + 58, cy - 18], fill=coat)
    d.polygon([(cx - 55, cy - 118), (cx - 68, cy - 178), (cx - 14, cy - 132)], fill=coat)
    d.polygon([(cx + 55, cy - 118), (cx + 68, cy - 178), (cx + 14, cy - 132)], fill=coat)
    for ex in (-26, 26):
        d.ellipse([cx + ex - 13, cy - 88, cx + ex + 13, cy - 62], fill=(240, 220, 90))
        d.ellipse([cx + ex - 4, cy - 86, cx + ex + 4, cy - 64], fill=(15, 15, 15))
    return ["cat", "pet", "animal", "indoor", "window", "home"]


def scene_car(img, rng):
    w, h = img.size
    vgrad(img, (150, 175, 200), (225, 230, 235), 0, int(h * 0.55))
    d = ImageDraw.Draw(img)
    d.rectangle([0, int(h * 0.62), w, h], fill=(70, 70, 74))             # asphalt
    d.line([(0, int(h * 0.82)), (w, int(h * 0.82))], fill=(230, 225, 200), width=5)
    body = rng.choice([(200, 40, 40), (30, 60, 150), (245, 245, 245), (20, 20, 24)])
    bx, by = int(w * 0.12), int(h * 0.50)
    bw, bh = int(w * 0.74), int(h * 0.16)
    d.rounded_rectangle([bx, by, bx + bw, by + bh], 18, fill=body)
    d.polygon([(bx + int(bw * 0.24), by), (bx + int(bw * 0.40), by - int(bh * 0.72)),
               (bx + int(bw * 0.70), by - int(bh * 0.72)), (bx + int(bw * 0.82), by)], fill=jitter(body, rng, 20))
    d.polygon([(bx + int(bw * 0.28), by - 4), (bx + int(bw * 0.42), by - int(bh * 0.62)),
               (bx + int(bw * 0.66), by - int(bh * 0.62)), (bx + int(bw * 0.76), by - 4)], fill=(150, 190, 210))
    for wx in (bx + int(bw * 0.20), bx + int(bw * 0.76)):
        d.ellipse([wx - 34, by + bh - 30, wx + 34, by + bh + 38], fill=(22, 22, 24))
        d.ellipse([wx - 14, by + bh - 10, wx + 14, by + bh + 18], fill=(170, 172, 175))
    return ["car", "vehicle", "road", "transport", "outdoor", "street"]


def scene_traffic(img, rng):
    w, h = img.size
    vgrad(img, (170, 195, 220), (225, 232, 238), 0, int(h * 0.6))
    d = ImageDraw.Draw(img)
    d.rectangle([0, int(h * 0.7), w, h], fill=(78, 78, 82))
    bx = int(w * 0.5)
    d.rounded_rectangle([bx - 62, int(h * 0.08), bx + 62, int(h * 0.60)], 18, fill=(38, 40, 45))
    lit = rng.randrange(3)
    for i, colour in enumerate([(215, 45, 45), (225, 195, 60), (60, 185, 90)]):
        cy = int(h * 0.16) + i * int(h * 0.16)
        shade = colour if i == lit else tuple(int(c * 0.32) for c in colour)
        d.ellipse([bx - 42, cy - 42, bx + 42, cy + 42], fill=shade)
    d.rectangle([bx - 11, int(h * 0.60), bx + 11, int(h * 0.75)], fill=(60, 62, 66))
    return ["traffic light", "street", "signal", "urban", "outdoor", "road"]


def scene_food(img, rng):
    w, h = img.size
    img.paste(rng.choice([(120, 85, 60), (225, 222, 215), (40, 42, 46)]), (0, 0, w, h))
    d = ImageDraw.Draw(img)
    cx, cy = w // 2, h // 2
    r = int(min(w, h) * 0.36)
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(248, 248, 245))     # plate
    d.ellipse([cx - r + 18, cy - r + 18, cx + r - 18, cy + r - 18], outline=(225, 225, 220), width=3)
    for _ in range(rng.randint(5, 9)):  # food
        fx, fy = cx + rng.randint(-r // 2, r // 2), cy + rng.randint(-r // 2, r // 2)
        fr = rng.randint(16, 38)
        d.ellipse([fx - fr, fy - fr, fx + fr, fy + fr],
                  fill=jitter(rng.choice([(190, 70, 45), (95, 140, 55), (225, 190, 90), (140, 95, 60)]), rng))
    return ["food", "plate", "meal", "indoor", "dining", "restaurant"]


def scene_coffee(img, rng):
    w, h = img.size
    img.paste((155, 120, 85), (0, 0, w, h))
    d = ImageDraw.Draw(img)
    cx, cy = w // 2, int(h * 0.55)
    r = int(min(w, h) * 0.26)
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(250, 250, 248))
    d.ellipse([cx - r + 16, cy - r + 16, cx + r - 16, cy + r - 16], fill=(78, 48, 28))
    d.ellipse([cx - r + 40, cy - r + 40, cx + r - 40, cy + r - 40], fill=(150, 110, 75))
    d.arc([cx + r - 20, cy - 42, cx + r + 52, cy + 42], -70, 70, fill=(250, 250, 248), width=13)
    return ["coffee", "cup", "drink", "indoor", "cafe", "beverage"]


def scene_flowers(img, rng):
    w, h = img.size
    vgrad(img, (150, 200, 235), (205, 230, 245), 0, int(h * 0.42))
    d = ImageDraw.Draw(img)
    d.rectangle([0, int(h * 0.42), w, h], fill=(88, 140, 62))
    petal = rng.choice([(235, 200, 60), (215, 70, 110), (240, 130, 60), (200, 90, 210)])
    for _ in range(48):
        fx, fy = rng.randint(0, w), rng.randint(int(h * 0.45), h)
        s = rng.randint(9, 20)
        d.line([(fx, fy), (fx, fy + s * 2)], fill=(60, 110, 50), width=3)
        for a in range(0, 360, 60):
            px = fx + int(s * math.cos(math.radians(a)))
            py = fy + int(s * math.sin(math.radians(a)))
            d.ellipse([px - s // 2, py - s // 2, px + s // 2, py + s // 2], fill=jitter(petal, rng))
        d.ellipse([fx - 5, fy - 5, fx + 5, fy + 5], fill=(250, 235, 140))
    return ["flowers", "field", "nature", "spring", "outdoor", "garden"]


def scene_stars(img, rng):
    w, h = img.size
    vgrad(img, (4, 6, 22), (18, 22, 52))
    d = ImageDraw.Draw(img)
    blobs(d, rng, 420, (0, 0, w, int(h * 0.82)), (1, 2), lambda r: jitter((235, 238, 255), r, 20))
    for _ in range(22):  # brighter foreground stars
        sx, sy = rng.randint(0, w), rng.randint(0, int(h * 0.7))
        d.ellipse([sx - 3, sy - 3, sx + 3, sy + 3], fill=(255, 255, 255))
    ridge(d, w, h, int(h * 0.22), (8, 10, 14), rng, steps=10)
    return ["night sky", "stars", "astronomy", "dark", "outdoor", "milky way"]


def scene_lighthouse(img, rng):
    w, h = img.size
    vgrad(img, (70, 100, 160), (220, 180, 150), 0, int(h * 0.6))
    d = ImageDraw.Draw(img)
    d.rectangle([0, int(h * 0.68), w, h], fill=(40, 80, 110))
    d.polygon([(int(w * 0.44), int(h * 0.68)), (int(w * 0.47), int(h * 0.16)),
               (int(w * 0.55), int(h * 0.16)), (int(w * 0.58), int(h * 0.68))], fill=(245, 245, 240))
    for i in range(3):
        y = int(h * (0.26 + i * 0.14))
        d.rectangle([int(w * 0.445), y, int(w * 0.577), y + int(h * 0.06)], fill=(200, 55, 50))
    d.rectangle([int(w * 0.455), int(h * 0.10), int(w * 0.565), int(h * 0.17)], fill=(60, 65, 75))
    d.ellipse([int(w * 0.475), int(h * 0.11), int(w * 0.545), int(h * 0.16)], fill=(255, 240, 170))
    return ["lighthouse", "coast", "sea", "building", "outdoor", "landmark"]


def scene_bridge(img, rng):
    w, h = img.size
    vgrad(img, (120, 165, 215), (215, 228, 240), 0, int(h * 0.62))
    d = ImageDraw.Draw(img)
    d.rectangle([0, int(h * 0.70), w, h], fill=(58, 95, 125))
    deck = int(h * 0.58)
    d.rectangle([0, deck, w, deck + 16], fill=(120, 60, 50))
    for tx in (int(w * 0.25), int(w * 0.72)):
        d.rectangle([tx - 14, int(h * 0.18), tx + 14, deck], fill=(140, 70, 58))
    for x in range(0, w, 26):  # suspension cables
        top = int(h * 0.22 + 0.30 * h * abs(math.sin(math.pi * x / w)))
        d.line([(x, top), (x, deck)], fill=(150, 82, 70), width=2)
    return ["bridge", "river", "architecture", "outdoor", "structure", "landmark"]


def scene_boat(img, rng):
    w, h = img.size
    vgrad(img, (140, 190, 230), (215, 235, 245), 0, int(h * 0.55))
    d = ImageDraw.Draw(img)
    d.rectangle([0, int(h * 0.55), w, h], fill=(45, 105, 150))
    bx, by = int(w * 0.5), int(h * 0.62)
    d.polygon([(bx - 120, by), (bx + 120, by), (bx + 80, by + 45), (bx - 80, by + 45)], fill=(230, 230, 225))
    d.line([(bx, by), (bx, by - 190)], fill=(120, 95, 70), width=6)
    d.polygon([(bx + 6, by - 185), (bx + 6, by - 15), (bx + 120, by - 15)], fill=(250, 250, 248))
    d.polygon([(bx - 6, by - 170), (bx - 6, by - 20), (bx - 85, by - 20)], fill=(235, 90, 70))
    return ["boat", "sailboat", "sea", "water", "outdoor", "sailing"]


def scene_balloon(img, rng):
    w, h = img.size
    vgrad(img, (95, 160, 220), (205, 232, 245), 0, int(h * 0.78))
    d = ImageDraw.Draw(img)
    d.rectangle([0, int(h * 0.78), w, h], fill=(95, 145, 70))
    for _ in range(5):
        bx, by = rng.randint(int(w * 0.12), int(w * 0.88)), rng.randint(int(h * 0.12), int(h * 0.52))
        r = rng.randint(34, 62)
        for i, seg in enumerate([(220, 60, 60), (240, 200, 70), (60, 120, 200)]):
            d.pieslice([bx - r, by - r, bx + r, by + r], 180 + i * 60, 240 + i * 60, fill=seg)
        d.pieslice([bx - r, by - r, bx + r, by + int(r * 1.35)], 0, 180, fill=jitter((225, 120, 70), rng))
        d.rectangle([bx - 9, by + int(r * 1.2), bx + 9, by + int(r * 1.45)], fill=(110, 80, 55))
    return ["hot air balloon", "sky", "flight", "colorful", "outdoor", "festival"]


def scene_fireworks(img, rng):
    w, h = img.size
    vgrad(img, (6, 8, 26), (30, 26, 58))
    d = ImageDraw.Draw(img)
    for _ in range(6):
        cx, cy = rng.randint(int(w * 0.15), int(w * 0.85)), rng.randint(int(h * 0.10), int(h * 0.55))
        colour = rng.choice([(255, 190, 70), (255, 90, 120), (120, 200, 255), (170, 255, 150)])
        for a in range(0, 360, 7):
            ln = rng.randint(40, 115)
            ex = cx + int(ln * math.cos(math.radians(a)))
            ey = cy + int(ln * math.sin(math.radians(a)))
            d.line([(cx, cy), (ex, ey)], fill=jitter(colour, rng, 30), width=2)
            d.ellipse([ex - 3, ey - 3, ex + 3, ey + 3], fill=jitter(colour, rng, 20))
    ridge(d, w, h, int(h * 0.14), (10, 12, 20), rng, steps=12)
    return ["fireworks", "night", "celebration", "colorful", "outdoor", "festival"]


def scene_rain_window(img, rng):
    w, h = img.size
    vgrad(img, (95, 105, 120), (140, 148, 160))
    d = ImageDraw.Draw(img)
    for _ in range(90):  # blurred city bokeh behind the glass
        bx, by = rng.randint(0, w), rng.randint(0, h)
        r = rng.randint(8, 26)
        d.ellipse([bx - r, by - r, bx + r, by + r], fill=jitter((175, 165, 140), rng, 35))
    img.paste(img.filter(ImageFilter.GaussianBlur(7)), (0, 0))
    d = ImageDraw.Draw(img)
    for _ in range(160):  # droplets on the pane
        dx, dy = rng.randint(0, w), rng.randint(0, h)
        dl = rng.randint(6, 24)
        d.line([(dx, dy), (dx + 2, dy + dl)], fill=(215, 222, 230), width=2)
    d.rectangle([0, 0, w - 1, h - 1], outline=(60, 60, 66), width=14)
    return ["rain", "window", "indoor", "weather", "moody", "glass"]


def scene_desk(img, rng):
    w, h = img.size
    img.paste((125, 92, 62), (0, 0, w, h))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([int(w * 0.10), int(h * 0.30), int(w * 0.52), int(h * 0.78)], 6,
                        fill=rng.choice([(210, 205, 195), (60, 80, 130), (170, 60, 55)]))
    d.rounded_rectangle([int(w * 0.13), int(h * 0.33), int(w * 0.49), int(h * 0.75)], 4, fill=(245, 243, 235))
    for i in range(9):
        y = int(h * 0.38) + i * int(h * 0.04)
        d.line([(int(w * 0.16), y), (int(w * 0.16 + rng.random() * w * 0.29), y)], fill=(120, 120, 125), width=3)
    d.rectangle([int(w * 0.60), int(h * 0.34), int(w * 0.88), int(h * 0.72)], fill=(38, 40, 46))
    d.rectangle([int(w * 0.62), int(h * 0.36), int(w * 0.86), int(h * 0.66)], fill=(120, 165, 200))
    return ["desk", "book", "laptop", "indoor", "workspace", "office"]


def scene_snow(img, rng):
    w, h = img.size
    vgrad(img, (170, 190, 215), (230, 238, 248), 0, int(h * 0.55))
    d = ImageDraw.Draw(img)
    d.rectangle([0, int(h * 0.55), w, h], fill=(240, 244, 250))
    for _ in range(9):  # bare trees
        tx = rng.randint(0, w)
        d.line([(tx, int(h * 0.55)), (tx, int(h * 0.30))], fill=(70, 58, 50), width=6)
        for a in (-40, -18, 18, 40):
            d.line([(tx, int(h * 0.38)), (tx + int(60 * math.sin(math.radians(a))), int(h * 0.24))],
                   fill=(70, 58, 50), width=3)
    blobs(d, rng, 260, (0, 0, w, h), (2, 4), lambda r: (255, 255, 255))
    return ["snow", "winter", "cold", "trees", "outdoor", "landscape"]


def scene_market(img, rng):
    w, h = img.size
    img.paste((205, 195, 178), (0, 0, w, h))
    d = ImageDraw.Draw(img)
    d.rectangle([0, int(h * 0.72), w, h], fill=(140, 128, 112))
    x = 0
    while x < w:  # striped awnings over produce crates
        aw = rng.randint(110, 190)
        colour = jitter(rng.choice([(200, 70, 60), (60, 130, 180), (230, 190, 70)]), rng)
        d.polygon([(x, int(h * 0.30)), (x + aw, int(h * 0.30)), (x + aw - 14, int(h * 0.44)), (x + 14, int(h * 0.44))],
                  fill=colour)
        d.rectangle([x + 18, int(h * 0.50), x + aw - 18, int(h * 0.72)], fill=(150, 112, 72))
        blobs(d, rng, 24, (x + 26, int(h * 0.50), x + aw - 26, int(h * 0.66)), (7, 13),
              lambda r: jitter(rng.choice([(210, 80, 50), (230, 190, 60), (90, 150, 60)]), r))
        x += aw + 10
    return ["market", "stall", "shopping", "outdoor", "street", "food"]


def scene_portrait(img, rng):
    w, h = img.size
    vgrad(img, jitter((120, 130, 150), rng, 40), jitter((70, 80, 100), rng, 40))
    d = ImageDraw.Draw(img)
    skin = rng.choice([(235, 195, 165), (200, 155, 120), (150, 105, 75), (95, 65, 48)])
    cx, cy = w // 2, int(h * 0.42)
    hr = int(min(w, h) * 0.20)
    d.ellipse([cx - int(hr * 1.6), cy + hr, cx + int(hr * 1.6), h], fill=jitter((60, 70, 110), rng, 30))  # shoulders
    d.ellipse([cx - hr, cy - hr, cx + hr, cy + hr], fill=skin)
    d.chord([cx - hr, cy - int(hr * 1.15), cx + hr, cy + int(hr * 0.5)], 180, 360,
            fill=rng.choice([(35, 28, 24), (90, 60, 35), (25, 22, 20)]))  # hair
    for ex in (-hr // 2, hr // 2):
        d.ellipse([cx + ex - 11, cy - 12, cx + ex + 11, cy + 6], fill=(250, 250, 250))
        d.ellipse([cx + ex - 5, cy - 8, cx + ex + 5, cy + 2], fill=(45, 35, 28))
    d.arc([cx - 26, cy + 18, cx + 26, cy + 52], 15, 165, fill=(140, 85, 75), width=4)
    return ["portrait", "person", "face", "people", "indoor", "headshot"]


# (slug, tag list is returned by fn, fn) — the album draws from this pool.
SCENES = [
    ("beach-sunset", lambda i, r: scene_beach(i, r, True)),
    ("beach-midday", lambda i, r: scene_beach(i, r, False)),
    ("mountains-green", lambda i, r: scene_mountains(i, r, False)),
    ("mountains-snow", lambda i, r: scene_mountains(i, r, True)),
    ("city-night", scene_city_night),
    ("forest-green", lambda i, r: scene_forest(i, r, False)),
    ("forest-autumn", lambda i, r: scene_forest(i, r, True)),
    ("desert-dunes", scene_desert),
    ("lake-reflection", scene_lake),
    ("dog-park", scene_dog),
    ("cat-window", scene_cat),
    ("sports-car", scene_car),
    ("traffic-light", scene_traffic),
    ("food-plate", scene_food),
    ("coffee-cup", scene_coffee),
    ("flower-field", scene_flowers),
    ("starry-night", scene_stars),
    ("lighthouse", scene_lighthouse),
    ("bridge", scene_bridge),
    ("sailboat", scene_boat),
    ("hot-air-balloons", scene_balloon),
    ("fireworks", scene_fireworks),
    ("rainy-window", scene_rain_window),
    ("desk-workspace", scene_desk),
    ("snow-field", scene_snow),
    ("street-market", scene_market),
    ("portrait", scene_portrait),
]

# Aspect classes. A photo manager's layout code breaks on the extremes, so the corpus
# carries a true panorama and a tall portrait rather than 50 mild landscapes.
ASPECTS = [
    ("landscape", (1280, 853)),
    ("landscape", (1600, 1067)),
    ("portrait", (853, 1280)),
    ("portrait", (960, 1440)),
    ("square", (1100, 1100)),
    ("panorama", (2400, 800)),
]


@dataclass
class Plan:
    """One planned output file."""
    index: int
    slug: str
    fmt: str
    size: tuple[int, int]
    camera: Camera | None
    place: Place | None
    when: str | None
    orientation: int = 1
    note: str = ""
    tags: list[str] = field(default_factory=list)


def revisit(img: Image.Image, rng: random.Random) -> Image.Image:
    """Make a second photograph of the same subject look like a genuinely different shot.

    There are more slots in the album (44) than there are scene generators (27), so some
    subjects are photographed twice — which is exactly what a real library looks like; nobody
    owns exactly one beach photo. The problem is that a second render of the same scene comes
    out nearly pixel-identical, and a *correct* perceptual-hash implementation would rightly
    flag the pair as a duplicate. The corpus would then be punishing the code for being right.

    So a revisit is pushed far apart deliberately, using the three things that actually differ
    between two shots of the same subject: which side you stood on (mirror), the light
    (hue rotation), and how close you were (crop-zoom). All three are deterministic given the
    per-image RNG, and together they move the difference hash well clear of the duplicate
    threshold while keeping the subject recognisably the same.
    """
    img = img.transpose(Image.FLIP_LEFT_RIGHT)

    # Rotate hue in HSV. A different time of day / white balance, not a different subject.
    hsv = img.convert("HSV")
    h, s, v = hsv.split()
    shift = rng.choice([48, 96, 144, 190])
    h = h.point(lambda p, k=shift: (p + k) % 256)
    img = Image.merge("HSV", (h, s, v)).convert("RGB")

    # Step in closer: crop to ~78% about a jittered centre and scale back up.
    w, hgt = img.size
    cw, ch = int(w * 0.78), int(hgt * 0.78)
    ox = rng.randint(0, w - cw)
    oy = rng.randint(0, hgt - ch)
    return img.crop((ox, oy, ox + cw, oy + ch)).resize((w, hgt), Image.LANCZOS)


def render(plan: Plan, rng: random.Random) -> tuple[Image.Image, list[str]]:
    """Draw a scene at the planned size and return the image plus its ground-truth tags."""
    img = Image.new("RGB", plan.size, (255, 255, 255))
    fn = dict(SCENES)[plan.slug]
    tags = fn(img, rng)
    if plan.index >= len(SCENES):
        img = revisit(img, rng)
    grain(img, rng)
    return img, tags


def save(
    img: Image.Image,
    path: str,
    fmt: str,
    exif: Image.Exif | None,
    rng: random.Random,
    quality: int | None = None,
    subsampling: int | None = None,
) -> None:
    """Write the image in the requested container.

    EXIF only goes into JPEG and WebP. PNG and TIFF are left bare on purpose: real libraries
    contain plenty of files with no camera metadata (screenshots, exports, scans), and the
    ingestion path has to fall back to file timestamps for those rather than crashing or
    silently dropping the photo.

    ``quality`` and ``subsampling`` override the random choice. The burst frames need this:
    letting the RNG pick a different quality per frame added ~3 bits of perceptual-hash distance
    between them, which is not something a camera shooting a burst would ever produce — one
    camera at one setting writes all three frames identically. Left random, it pushed the burst
    spread from 3 bits to 5 and squeezed the gap against unrelated scenes down to nothing.
    """
    kwargs: dict = {}
    if fmt == "JPEG":
        kwargs.update(
            quality=quality if quality is not None else rng.choice([82, 88, 92, 95]),
            subsampling=subsampling if subsampling is not None else rng.choice([0, 2]),
            optimize=True)
        if exif is not None:
            kwargs["exif"] = exif.tobytes()
    elif fmt == "WEBP":
        kwargs.update(quality=rng.choice([80, 88, 94]), method=4)
        if exif is not None:
            kwargs["exif"] = exif.tobytes()
    elif fmt == "PNG":
        kwargs.update(optimize=True)
    elif fmt == "TIFF":
        kwargs.update(compression="tiff_lzw")

    img.save(path, fmt, **kwargs)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate the PixelFlux 50-image test album.")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                  "testdata", "album"))
    ap.add_argument("--seed", type=int, default=20260807)
    ap.add_argument("--clean", action="store_true", help="delete the output directory first")
    args = ap.parse_args()

    if args.clean and os.path.isdir(args.out):
        shutil.rmtree(args.out)
    os.makedirs(args.out, exist_ok=True)

    rng = random.Random(args.seed)

    # ---- Plan the 44 distinct photos -------------------------------------------------
    # Dates are drawn from a fixed spread and then shuffled against the filename order, so
    # "sort by capture date" and "sort by name" genuinely disagree. A corpus where they
    # happen to agree cannot catch a sort that reads the wrong field.
    years = [2019, 2021, 2022, 2023, 2024, 2025, 2026]
    dates = []
    for y in years:
        for _ in range(7):
            dates.append(f"{y}:{rng.randint(1,12):02d}:{rng.randint(1,28):02d} "
                         f"{rng.randint(6,22):02d}:{rng.randint(0,59):02d}:{rng.randint(0,59):02d}")
    rng.shuffle(dates)

    # Assigning any attribute with `i % len(...)` looks harmless and is a trap whenever two of
    # those cycles share a factor. This bit an earlier version twice in a row:
    #
    #   * format cycles with period 8, and PNG/TIFF carry no EXIF
    #   * places cycle with period 8
    #   -> places at the PNG and TIFF slots could never receive a GPS fix, so the album claimed
    #      8 countries and shipped 6. A bounding-box search for Colombia found nothing, and it
    #      read like a product bug rather than a corpus bug.
    #
    # The fix is not a cleverer modulus — that only moves the collision. Attributes that must
    # all appear are assigned from a counter that advances *only over eligible photos*, so
    # eligibility and assignment cannot correlate by construction.
    plans: list[Plan] = []
    slugs = [s for s, _ in SCENES]
    exif_seq = 0
    for i in range(44):
        slug = slugs[i % len(slugs)]
        aspect_name, size = ASPECTS[i % len(ASPECTS)]
        # Panoramas only make sense for wide scenes; give the portrait scene a tall frame.
        if slug == "portrait":
            aspect_name, size = "portrait", (960, 1440)
        fmt = ["JPEG", "JPEG", "JPEG", "JPEG", "PNG", "WEBP", "JPEG", "TIFF"][i % 8]
        has_exif = fmt in ("JPEG", "WEBP") and i % 11 != 5   # a few EXIF-less JPEGs too

        camera = place = when = None
        if has_exif:
            camera = CAMERAS[exif_seq % len(CAMERAS)]
            when = dates[i]
            # Roughly one EXIF photo in nine records no GPS fix — indoor shots, a camera with
            # location off. 9 is coprime with len(PLACES), so no single place is ever the one
            # that gets skipped.
            if exif_seq % 9 != 4:
                place = PLACES[exif_seq % len(PLACES)]
            exif_seq += 1

        plans.append(Plan(
            index=i,
            slug=slug,
            fmt=fmt,
            size=size,
            camera=camera,
            place=place,
            when=when,
            orientation=6 if i % 17 == 4 else 1,  # a couple of rotated originals
            note=aspect_name,
        ))

    manifest_lines = []
    ext = {"JPEG": "jpg", "PNG": "png", "WEBP": "webp", "TIFF": "tif"}

    written: list[tuple[str, Plan, list[str]]] = []
    for plan in plans:
        # Per-image RNG derived from the master seed keeps each scene stable even if the
        # album composition around it changes.
        irng = random.Random(args.seed + plan.index * 7919)
        img, tags = render(plan, irng)
        plan.tags = tags

        exif = None
        if plan.camera and plan.when:
            exif = build_exif(plan.camera, plan.place, plan.when, plan.size, irng, plan.orientation)

        name = f"{plan.index:03d}_{plan.slug}.{ext[plan.fmt]}"
        path = os.path.join(args.out, name)
        save(img, path, plan.fmt, exif, irng)
        written.append((name, plan, tags))

    # ---- Duplicates: byte-identical copies under different names ---------------------
    # The content hash must collapse these; the perceptual hash must too. A photo manager
    # that reports 50 unique images here is broken.
    dup_sources = [written[3][0], written[16][0]]
    for n, src in enumerate(dup_sources):
        dst = f"{44 + n:03d}_duplicate-of-{src.split('_',1)[1]}"
        shutil.copyfile(os.path.join(args.out, src), os.path.join(args.out, dst))
        manifest_lines.append(f"{dst}\tEXACT DUPLICATE of {src} — identical bytes, identical content hash")

    # ---- Burst: three near-duplicates a second apart ---------------------------------
    # Same scene, tiny shifts. Content hashes all differ; perceptual hashes must cluster.
    # This is what the slideshow's "avoid burst repetition" rule has to detect.
    # Three frames one second apart. Getting this to behave like a real burst took two
    # corrections, both of which were the fixture lying rather than the hash misbehaving:
    #
    #   1. Grain must be applied ONCE, to the shared base, before the frames diverge.
    #      Graining each frame independently put the outer frames 12 bits apart. Independent
    #      per-pixel noise flips difference-hash bits wherever neighbouring pixels are nearly
    #      equal — on a sky gradient, that is most of the frame. Real consecutive frames from
    #      one sensor share their noise; only the subject moves.
    #
    #   2. The frame offset must be a CROP WINDOW, not an affine translation. Translating
    #      leaves a black bar along the exposed edge, and once the hash downsamples to 9x8
    #      that bar poisons an entire column — 8 of the 64 comparisons — differently in each
    #      frame. Measured: shift-by-translate alone accounted for 13 of the 14 bits of drift,
    #      while a 1.5% exposure change accounted for 2. Rendering the scene oversized and
    #      cropping a shifted window keeps every frame fully populated, which is what a
    #      camera actually produces.
    #
    # With both fixed the burst sits at hamming 2-4, comfortably inside any sane threshold.
    burst_rng = random.Random(args.seed + 1_000_003)
    bw, bh, pad = 1280, 853, 6
    burst_base = Image.new("RGB", (bw + 2 * pad, bh + 2 * pad), (255, 255, 255))
    scene_dog(burst_base, random.Random(args.seed + 55))
    grain(burst_base, burst_rng)
    for n in range(3):
        dx, dy = (n - 1) * 3, (n - 1) * 2
        frame = burst_base.crop((pad + dx, pad + dy, pad + dx + bw, pad + dy + bh))
        frame = frame.point(lambda v, k=n: max(0, min(255, int(v * (1 + (k - 1) * 0.015)))))
        cam = CAMERAS[3]
        when = f"2025:07:12 16:20:{40 + n:02d}"
        exif = build_exif(cam, PLACES[2], when, frame.size, burst_rng)
        dst = f"{46 + n:03d}_burst-dog-{n + 1}.jpg"
        # Same camera, same settings, same encoder: all three frames at identical quality.
        save(frame, os.path.join(args.out, dst), "JPEG", exif, burst_rng, quality=92, subsampling=2)
        manifest_lines.append(
            f"{dst}\tBURST frame {n+1}/3 — near-duplicate, 1s apart, {cam.model}, Fort Lauderdale")

    # ---- A truncated file -------------------------------------------------------------
    # Half a JPEG. Ingestion must record it as failed and carry on, not abort the folder.
    good = os.path.join(args.out, written[7][0])
    with open(good, "rb") as fh:
        blob = fh.read()
    truncated = os.path.join(args.out, "049_corrupt-truncated.jpg")
    with open(truncated, "wb") as fh:
        fh.write(blob[: len(blob) // 2])
    manifest_lines.append("049_corrupt-truncated.jpg\tCORRUPT — first half of a valid JPEG; must fail gracefully")

    # ---- Manifest ---------------------------------------------------------------------
    header = [
        "# PixelFlux test album — generated by tools/make_test_album.py",
        f"# seed={args.seed}  files=50",
        "# Columns: filename <TAB> description",
        "",
    ]
    body = []
    for name, plan, tags in written:
        bits = [f"{plan.size[0]}x{plan.size[1]}", plan.note, plan.fmt]
        if plan.camera:
            bits.append(plan.camera.model)
        else:
            bits.append("no-EXIF")
        if plan.place:
            bits.append(f"{plan.place.name}, {plan.place.country}")
        if plan.when:
            bits.append(plan.when.split()[0].replace(":", "-"))
        if plan.orientation != 1:
            bits.append(f"orientation={plan.orientation}")
        body.append(f"{name}\t{' | '.join(bits)} | tags: {', '.join(tags)}")

    with open(os.path.join(args.out, "MANIFEST.tsv"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(header + sorted(body) + sorted(manifest_lines)) + "\n")

    total = len(os.listdir(args.out)) - 1
    print(f"Wrote {total} images + MANIFEST.tsv to {args.out}")


if __name__ == "__main__":
    main()
