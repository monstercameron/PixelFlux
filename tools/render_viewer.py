#!/usr/bin/env python3
"""Render the photo viewer in headless Edge, from real library data, and screenshot it.

WHY THIS EXISTS
---------------
The viewer is the one surface that cannot be inspected from the outside. It only appears once a
photograph is open, and opening one means clicking on a machine somebody else is using. The
result was a whole feature — mask overlay, zoom, the metadata plate — built and shipped without
anyone looking at it, and three separate visual bugs went out because of that.

This reproduces the viewer's real markup against the real app.css, with real photographs and
real masks pulled from the live library, and renders it in headless Edge. It is not a substitute
for running the app: Blazor's behaviour is not here, only its output. But it is enough to see
whether the masks land, whether the labels are legible, and whether the picture collides with
the plate — which is exactly what went wrong unseen.

USAGE
-----
    python tools/render_viewer.py [--photo <substring>] [--width 1600] [--height 1000]
"""

from __future__ import annotations

import argparse
import html
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile

EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
LIB = os.path.expandvars(r"%LOCALAPPDATA%\PixelFlux")
APP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "src", "PixelFlux.App", "wwwroot")


def file_url(path: str) -> str:
    return "file:///" + path.replace("\\", "/").replace(" ", "%20")


def hue_of(label: str) -> int:
    """Mirrors PhotoSegmentRecord.Hue so the harness colours match the app exactly."""
    h = 2166136261
    for c in label:
        h = ((h ^ ord(c)) * 16777619) & 0xFFFFFFFF
    return int(h * 0.618033988749895 % 360)


def build(photo_filter: str | None) -> tuple[str, str]:
    db = sqlite3.connect(os.path.join(LIB, "library.db"))
    db.row_factory = sqlite3.Row

    where = "AND p.file_name LIKE ?" if photo_filter else ""
    params = (f"%{photo_filter}%",) if photo_filter else ()

    row = db.execute(f"""
        SELECT p.* , COUNT(g.id) n
        FROM photos p JOIN photo_segments g ON g.photo_id = p.id
        WHERE 1=1 {where}
        GROUP BY p.id ORDER BY n DESC LIMIT 1
    """, params).fetchone()

    if row is None:
        raise SystemExit("no analysed photo matches; run `pixelflux analyze` first")

    proxy = file_url(os.path.join(LIB, "cache", row["proxy_key"].replace("/", os.sep)))
    segs = db.execute(
        "SELECT * FROM photo_segments WHERE photo_id=? ORDER BY prominence DESC", (row["id"],)
    ).fetchall()

    masks = []
    for s in segs:
        if not s["mask_key"]:
            continue
        url = file_url(os.path.join(LIB, "cache", s["mask_key"].replace("/", os.sep)))
        # An <img>, matching SegmentOverlay. The colour is baked into the file, so nothing here
        # tints it. This must stay in step with the component or the harness stops being evidence
        # — it already produced one false negative by rendering the superseded CSS-mask markup
        # against CSS that no longer supported it.
        masks.append(
            f'<div class="seg" style="left:{s["x"]*100:.3f}%;top:{s["y"]*100:.3f}%;'
            f'width:{s["w"]*100:.3f}%;height:{s["h"]*100:.3f}%;'
            f'--hue:{hue_of(s["label"])}">'
            f'<img src="{url}" alt="" draggable="false"></div>')

    # Mirrors SegmentOverlay.LabelledSegments, so the harness shows the placement the app uses
    # rather than an idealised one. If these two drift the harness stops being evidence.
    chips: list[str] = []
    placed: list[tuple[float, float]] = []

    for s in [g for g in segs if g["area"] >= 0.012][:7]:
        left = min(max((s["x"] + s["w"] / 2) * 100 - 6, 1), 72)
        top = min(max((s["y"] + s["h"] / 2) * 100, 4), 92)

        if top < 9 and left > 55:
            left = 55

        dropped = False
        while any(abs(pt - top) < 4.5 and abs(pl - left) < 22 for pt, pl in placed):
            top += 5
            if top > 92:
                dropped = True
                break

        if dropped:
            continue

        placed.append((top, left))
        chips.append(
            f'<button class="seg-chip" style="left:{left:.3f}%;top:{top:.3f}%;'
            f'--hue:{hue_of(s["label"])}">'
            f'<span class="dot"></span><span class="lbl">{html.escape(s["label"])}</span></button>')

    def plate(k, v, cls=""):
        return (f'<div class="plate-group"><span class="k">{html.escape(k)}</span>'
                f'<bdi class="v {cls}">{html.escape(str(v))}</bdi></div>')

    # Mirrors the plate's chip markup: a name that searches, and a pencil that corrects. The
    # pencil is revealed on hover in the real thing; the harness forces one visible so the
    # layout with it showing can be checked.
    chips_seen = []
    for i, s in enumerate(segs[:10]):
        shown = s["user_label"] or s["label"]
        mine = " mine" if s["user_label"] else ""
        lit = " forced" if i == 0 else ""
        chips_seen.append(
            f'<span class="tag seen{mine}{lit}" style="--hue:{hue_of(shown)}">'
            f'<button class="tagmain"><span class="dot"></span>{html.escape(shown)}</button>'
            f'<button class="tagedit">&#9998;</button></span>')

    # A keyword the user typed, with its remove control, and the add affordance.
    chips_seen.append('<span class="tag mine">holiday<button class="tagx">&#10005;</button></span>')
    chips_seen.append('<span class="tag editing" style="--hue:200">'
                      '<input type="text" value="car"></span>')

    # Enough extra keywords to overflow the strip. The whole point of the redesign is that the
    # plate's height does not move however many of these there are.
    for extra in ("holiday", "summer", "grandad's boat", "kodachrome", "reunion", "brighton",
                  "second cousin", "the good camera", "before the fire"):
        chips_seen.append(f'<span class="tag mine">{html.escape(extra)}</span>')

    seen = "".join(chips_seen)

    page = f"""<div class="viewer" role="dialog">
  <div class="viewer-stage">
    <div class="framed" style="--ar:{row["width"] / row["height"]:.5f}">
      <img src="{proxy}" alt="" />
      <div class="segs">{''.join(masks)}{''.join(chips)}</div>
    </div>
    <button class="viewer-step prev">&lsaquo;</button>
    <button class="viewer-step next">&lsaquo;</button>
    <button class="viewer-zoom">180%</button>
    <button class="viewer-masks on"><span class="glyph">&#9681;</span><span>{len(masks)}</span></button>
    <button class="viewer-close">&#10005;</button>
  </div>
  <div class="plate">
    {plate("Frame", row["file_name"])}
    {plate("Taken", row["captured_utc"][:16].replace("T", " "))}
    {plate("Camera", row["camera_model"] or "—")}
    {plate("Exposure", f'{row["focal_length_mm"] or "—"}mm  f/{row["f_number"] or "—"}')}
    {plate("Size", f'{row["width"]}x{row["height"]}')}
    {plate("Place", row["place_label"] or "—")}
    <div class="plate-group saw-group">
      <span class="k">What PixelFlux saw</span>
      <span class="saw">{seen}</span>
      <button class="tag add addpin">&#65291;</button>
    </div>
  </div>
</div>"""

    doc = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<base href="{file_url(APP)}/">
<link rel="stylesheet" href="app.css">
<style>html,body{{height:100%;margin:0}}
/* The pencil and the remove cross are hover-revealed; force one visible so the layout can be
   judged with it showing. */
.tag.forced .tagedit, .tag.mine .tagx {{ opacity: 1; }}
</style>
</head><body>{page}</body></html>"""

    return doc, row["file_name"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--photo")
    ap.add_argument("--width", type=int, default=1600)
    ap.add_argument("--height", type=int, default=1000)
    ap.add_argument("--out", default=os.path.join(tempfile.gettempdir(), "pf-viewer.png"))
    # Overrides the mask opacity for calibration. "Barely visible" is a judgement that can only
    # be made by looking, and looking at three values side by side beats guessing at one.
    ap.add_argument("--mask-opacity", type=float)
    args = ap.parse_args()

    doc, name = build(args.photo)
    if args.mask_opacity is not None:
        doc = doc.replace("</head>",
                          f"<style>.seg img{{opacity:{args.mask_opacity}}}</style></head>")
    page = os.path.join(tempfile.gettempdir(), "pf-viewer-harness.html")
    with open(page, "w", encoding="utf-8") as fh:
        fh.write(doc)

    profile = tempfile.mkdtemp(prefix="pf-edge-")
    try:
        subprocess.run([
            EDGE, "--headless=new", f"--screenshot={args.out}",
            f"--window-size={args.width},{args.height}",
            "--hide-scrollbars", "--force-device-scale-factor=1",
            "--allow-file-access-from-files",
            "--virtual-time-budget=4000",
            f"--user-data-dir={profile}",
            file_url(page),
        ], check=True, capture_output=True, timeout=120)
    finally:
        shutil.rmtree(profile, ignore_errors=True)

    print(f"{name}\n{args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
