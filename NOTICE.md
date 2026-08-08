# Third-party components

PixelFlux is MIT licensed. It depends on the following, which are not.

## Runtime dependencies

| Component | Licence | Note |
| --- | --- | --- |
| ONNX Runtime, ONNX Runtime GenAI | MIT | Microsoft |
| **SixLabors.ImageSharp** | **Six Labors Split License** | See below — this one has a condition |
| Microsoft.Data.Sqlite, SQLitePCLRaw | MIT / Apache-2.0 | SQLite itself is public domain |
| MetadataExtractor | Apache-2.0 | EXIF reading |
| AWS SDK for .NET (S3, Core) | Apache-2.0 | Used only by the optional S3 object store |
| Microsoft.Extensions.* | MIT | |
| xUnit, coverlet | Apache-2.0 / MIT | Test-only |

### ImageSharp

ImageSharp 3.x is published under the **Six Labors Split License**, not Apache-2.0. In short: it is
free to use in open-source projects and for personal use, and a commercial licence is required for
closed-source commercial use.

PixelFlux being MIT and open source, this repository is within the free terms. **If you are
building something closed and commercial on top of it, that obligation is yours to check** — it
does not travel with the MIT grant on PixelFlux's own code. See
<https://sixlabors.com/pricing/> for the current terms.

## Data

**Reverse geocoding gazetteer** — `src/PixelFlux.Core/Geo/gazetteer.bin` is built from
[GeoNames](https://geonames.org) data, licensed **CC BY 4.0**. Attribution accompanies it in
`src/PixelFlux.Core/Geo/gazetteer-ATTRIBUTION.txt`, and `tools/build_gazetteer.py` regenerates
both. It ships as a binary blob so that resolving a photograph's coordinates to a place name never
requires a network call.

**Test photographs** — not distributed here. `tools/fetch_test_album.py` and
`tools/fetch_people_set.py` download 132 images from Wikimedia Commons under a range of Creative
Commons terms and write `testdata/*/ATTRIBUTION.tsv` naming every photographer, title, licence and
source URL. Those attribution files are committed even though the images are not.

## Fonts

| Font | Licence |
| --- | --- |
| Archivo | SIL Open Font License 1.1 |
| IBM Plex Mono | SIL Open Font License 1.1 |
| Open Sans | Apache License 2.0 |

The OFL permits bundling and redistribution with software, including commercially, provided the
fonts are not sold on their own and any modified version is not released under the reserved font
name.

## Models

Every model is a separate download under its own licence and none is distributed with this
software. One of them — the YOLO segmenter — is AGPL-3.0. See [docs/MODELS.md](docs/MODELS.md).
