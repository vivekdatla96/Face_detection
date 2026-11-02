#!/usr/bin/env python3
"""Download a public sample portrait for local demos."""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

# OpenCV sample portrait (Lena) used widely in vision demos.
DEFAULT_URL = (
    "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/lena.jpg"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("assets/sample_face.jpg"),
        help="Where to save the sample image",
    )
    parser.add_argument("--url", default=DEFAULT_URL)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {args.url}")
    request = urllib.request.Request(
        args.url,
        headers={"User-Agent": "face-detection/0.1"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        args.output.write_bytes(response.read())
    print(f"Saved {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
