#!/usr/bin/env python3
"""Render the faces wall in headless Edge, from real library data, and screenshot it.

WHY THIS EXISTS
---------------
Same reason as render_viewer.py: the page lives inside a WebView on a machine somebody else is
using, and shipping a whole surface without looking at it has already produced several visual
bugs that no test caught. This reproduces the wall's real markup against the real app.css, with
the real face crops the sweep wrote, and renders it at a real window size.

It is not the application. Blazor's behaviour is absent — hover, keyboard, the viewer. What it
does show is layout, legibility, and whether the crops land, which is what has gone wrong before.

Keep the markup below in step with Faces.razor. A harness that renders something the component
no longer emits is worse than no harness: it reports success about markup nobody ships.

This covers the WALL only. The person view is two clicks deep and cannot be reached from here;
it is checked against the running application instead, by launching with
PIXELFLUX_START=/faces?person=<face-id> and screenshotting the window.

USAGE
-----
    python tools/render_faces.py [--width 1500] [--height 950] [--hover N] [--lang ar]
"""

from __future__ import annotations

import argparse
import html
import os
import shutil
import sqlite3
import subprocess
import tempfile

EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
LIB = os.path.expandvars(r"%LOCALAPPDATA%\PixelFlux")
APP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "src", "PixelFlux.App", "wwwroot")

# Mirrors Strings.cs. Only the keys this page renders, and only the languages worth eyeballing:
# English for the default reading, Arabic because it is the one that reverses the layout.
COPY = {
    "en": {
        "title": "Faces", "photos": "Photos", "faces": "Faces",
        "order": "Order", "orderValue": "Most prominent",
        "confidence": "Certainty", "confidenceValue": "Everything found",
        "language": "Language", "scan": "Look again",
        "privacy": "Faces are found on this computer. Nothing is uploaded.",
        "idle": "Idle", "shown": "shown", "photoWord": "photos",
        "count": "{0} faces in {1} photos",
    },
    "ar": {
        "title": "الوجوه", "photos": "الصور", "faces": "الوجوه",
        "order": "الترتيب", "orderValue": "الأبرز",
        "confidence": "درجة التأكد", "confidenceValue": "كل ما عُثر عليه",
        "language": "اللغة", "scan": "إعادة البحث",
        "privacy": "يجري كشف الوجوه على هذا الجهاز. لا يُرفع أي شيء.",
        "idle": "خامل", "shown": "معروضة", "photoWord": "صورة",
        "count": "{0} وجهًا في {1} صورة",
    },
}


def file_url(path: str) -> str:
    return "file:///" + path.replace("\\", "/").replace(" ", "%20")


def cache_url(key: str) -> str:
    return file_url(os.path.join(LIB, "cache", key.replace("/", os.sep)))


def build(lang: str, hover: int | None) -> str:
    t = COPY[lang]
    rtl = lang == "ar"

    db = sqlite3.connect(os.path.join(LIB, "library.db"))
    db.row_factory = sqlite3.Row

    # The same query and ordering FaceStore.ListAsync issues for FaceOrder.Prominence.
    rows = db.execute("""
        SELECT f.*, p.file_name, p.thumbnail_key, p.captured_utc, p.place_label
        FROM photo_faces f JOIN photos p ON p.id = f.photo_id
        ORDER BY (SQRT(f.area) * 0.7 + f.confidence * 0.3) DESC, f.id
    """).fetchall()

    if not rows:
        raise SystemExit("no faces in the library; run `pixelflux faces` first")

    total = len(rows)
    photographs = len({r["photo_id"] for r in rows})

    counts: dict[int, int] = {}
    for r in rows:
        counts[r["photo_id"]] = counts.get(r["photo_id"], 0) + 1

    hover_photo = rows[hover]["photo_id"] if hover is not None and hover < total else None

    seen: dict[int, int] = {}
    cards = []

    for i, r in enumerate(rows):
        pid = r["photo_id"]
        seen[pid] = seen.get(pid, 0) + 1
        position, siblings = seen[pid], counts[pid]

        classes = ["facecard"]
        if hover is not None and i == hover:
            classes.append("focused")
        if hover_photo == pid and siblings > 1:
            classes.append("kin")

        if r["crop_key"]:
            img = f'<img src="{cache_url(r["crop_key"])}" alt="">'
        else:
            img = f'<img class="fallback" src="{cache_url(r["thumbnail_key"])}" alt="">'

        of = (f'<span class="of" dir="ltr">{position}<i>/</i>{siblings}</span>'
              if siblings > 1 else "")

        # A face with no vector cannot be searched for, and the card says so with one dim dot.
        uncomparable = "" if r["embedding"] else '<span class="uncomparable">·</span>'


        place = (f'<bdi class="where">{html.escape(r["place_label"])}</bdi>'
                 if r["place_label"] else "")

        cards.append(
            f'<button class="{" ".join(classes)}">{img}'
            f'<span class="sure"><i style="width:{r["confidence"] * 100:.1f}%"></i></span>'
            f'{of}{uncomparable}'
            f'<span class="whence">'
            f'<bdi class="when mono">{r["captured_utc"][:10]}</bdi>{place}</span>'
            f'</button>')

    count_line = t["count"].replace("{0}", str(total)).replace("{1}", str(photographs))

    page = f"""<div class="shell {"rtl" if rtl else ""}" dir="{"rtl" if rtl else "ltr"}">
  <header class="topbar">
    <div class="wordmark" dir="ltr"><b>Pixel</b><i>Flux</i></div>
    <nav class="viewswitch">
      <button class="vs">{t["photos"]}</button>
      <button class="vs on">{t["faces"]}<span class="pip">{total}</span></button>
    </nav>
    <span class="spacer"></span>
    <div class="tools">
      <label class="sortpick"><span class="label">{t["order"]}</span>
        <select><option>{t["orderValue"]}</option></select></label>
      <span class="tool-sep"></span>
      <label class="sortpick"><span class="label">{t["confidence"]}</span>
        <select><option>{t["confidenceValue"]}</option></select></label>
      <span class="tool-sep"></span>
      <label class="sortpick"><span class="label">{t["language"]}</span>
        <select><option>English</option></select></label>
      <button class="btn btn-primary">{t["scan"]}</button>
    </div>
  </header>

  <div class="body plain">
    <main class="sheet">
      <div class="sheet-head">
        <h1>{t["title"]}</h1>
        <span class="sub mono">{count_line}</span>
      </div>
      <div class="facewall">{"".join(cards)}</div>
    </main>
  </div>

  <footer class="statusbar">
    <span class="stat"><b>{total}</b>&nbsp;{t["faces"]}</span>
    <span class="stat"><b>{photographs}</b>&nbsp;{t["photoWord"]}</span>
    <span class="spacer"></span>
    <span class="stat quiet">{t["privacy"]}</span>
    <span class="safelight"><span class="lamp"></span><span>{t["idle"]}</span></span>
  </footer>
</div>"""

    return f"""<!DOCTYPE html><html lang="{lang}"><head><meta charset="utf-8">
<base href="{file_url(APP)}/">
<link rel="stylesheet" href="app.css">
<style>html,body{{height:100%;margin:0}}</style>
</head><body>{page}</body></html>"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=1500)
    ap.add_argument("--height", type=int, default=950)
    ap.add_argument("--lang", default="en", choices=sorted(COPY))
    ap.add_argument("--hover", type=int, help="index of the card to render as hovered/focused")
    ap.add_argument("--out", default=os.path.join(tempfile.gettempdir(), "pf-faces-page.png"))
    args = ap.parse_args()

    doc = build(args.lang, args.hover)

    # Hover and focus cannot be driven from a screenshot flag, so the harness applies the same
    # rules the pointer would. Kept next to the real ones rather than duplicating them.
    if args.hover is not None:
        doc = doc.replace("</head>", """<style>
            .facewall .facecard.focused .whence { opacity: 1; transform: none; }
            .facewall .facecard.focused img { transform: scale(1.04); }
            .facewall .facecard.focused {
                border-color: var(--select); box-shadow: 0 0 0 1px var(--select);
            }
        </style></head>""")

    page = os.path.join(tempfile.gettempdir(), "pf-faces-harness.html")
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

    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
