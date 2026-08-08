"""Fetch the ONNX Runtime build that actually has an execution provider in it.

Why this exists
---------------
The `Microsoft.ML.OnnxRuntime` package ships a win-arm64 native with no execution provider
compiled in at all -- probe it and `GetAvailableProviders()` returns Azure and CPU, nothing else.
The packages that do have one, `.DirectML` and `.Qnn`, stopped at 1.24.4, while the vision model's
GenAI layer requires 1.28 or newer. Taken at face value that means no accelerator on this hardware
without giving up the vision model.

The way out is that Windows ML ships its own runtime, and that one is built with providers. This
script pulls `onnxruntime.dll` and `DirectML.dll` out of the Windows ML package and drops them in
`runtime/win-arm64/`, where the build copies them over the ones the package put in the output.

Measured 2026-08-08 on a Snapdragon X2: that runtime reports version 1.27.1 and
`DmlExecutionProvider`, and the managed 1.28 binding, GenAI 0.15.2 and the Qwen vision model all
load against it unchanged. The C API is backward compatible in the direction that matters.

Run it from the repository root. It writes nothing outside `runtime/`.
"""

from __future__ import annotations

import io
import pathlib
import sys
import urllib.request
import zipfile

PACKAGE = "microsoft.windows.ai.machinelearning"
VERSION = "2.4.66-preview"

# Only these two. The package is 50 MB and most of it is x64 and arm64ec copies of the same thing.
WANTED = ("onnxruntime.dll", "DirectML.dll")

RUNTIMES = ("win-arm64", "win-x64")


def main() -> int:
    root = pathlib.Path(__file__).resolve().parent.parent
    url = (
        f"https://api.nuget.org/v3-flatcontainer/{PACKAGE}/{VERSION}/"
        f"{PACKAGE}.{VERSION}.nupkg"
    )

    print(f"fetching {PACKAGE} {VERSION}")
    with urllib.request.urlopen(url, timeout=300) as response:
        payload = response.read()

    print(f"  {len(payload) / 1024 / 1024:.0f} MB")

    written = 0
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        for runtime in RUNTIMES:
            target = root / "runtime" / runtime
            for wanted in WANTED:
                entry = f"runtimes/{runtime}/native/{wanted}"
                if entry not in archive.namelist():
                    continue

                target.mkdir(parents=True, exist_ok=True)
                data = archive.read(entry)
                (target / wanted).write_bytes(data)
                print(f"  {runtime}/{wanted}  {len(data) / 1024 / 1024:.1f} MB")
                written += 1

    if written == 0:
        print("nothing extracted -- the package layout has changed", file=sys.stderr)
        return 1

    print()
    print("Done. Rebuild, then run `pixelflux accel` to confirm the provider is visible.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
