"""Command-line interface for face detection."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

from .detector import FaceDetector, detections_to_dicts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="face-detect",
        description="Detect faces in images or a webcam stream.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    image = sub.add_parser("image", help="Detect faces in one or more image files")
    image.add_argument("paths", nargs="+", type=Path, help="Input image path(s)")
    image.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for annotated images (default: alongside inputs)",
    )
    image.add_argument(
        "--json",
        action="store_true",
        help="Print detections as JSON instead of a summary",
    )
    image.add_argument("--scale-factor", type=float, default=1.1)
    image.add_argument("--min-neighbors", type=int, default=5)
    image.add_argument(
        "--min-size",
        type=int,
        nargs=2,
        metavar=("W", "H"),
        default=(30, 30),
    )
    image.add_argument("--no-save", action="store_true", help="Skip writing annotated images")

    camera = sub.add_parser("camera", help="Detect faces from a webcam")
    camera.add_argument("--device", type=int, default=0, help="Camera device index")
    camera.add_argument("--scale-factor", type=float, default=1.1)
    camera.add_argument("--min-neighbors", type=int, default=5)
    camera.add_argument(
        "--min-size",
        type=int,
        nargs=2,
        metavar=("W", "H"),
        default=(30, 30),
    )
    camera.add_argument(
        "--mirror",
        action="store_true",
        help="Mirror the preview horizontally",
    )

    return parser


def run_image(args: argparse.Namespace) -> int:
    detector = FaceDetector(
        scale_factor=args.scale_factor,
        min_neighbors=args.min_neighbors,
        min_size=tuple(args.min_size),
    )
    results = []
    exit_code = 0

    for path in args.paths:
        try:
            image, detections = detector.detect_file(path)
        except FileNotFoundError as exc:
            print(f"error: {exc}", file=sys.stderr)
            exit_code = 1
            continue

        payload = {
            "path": str(path),
            "count": len(detections),
            "faces": detections_to_dicts(detections),
        }
        results.append(payload)

        if not args.no_save:
            out_dir = args.output_dir or path.parent
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{path.stem}_faces{path.suffix}"
            annotated = detector.annotate(image, detections)
            if not cv2.imwrite(str(out_path), annotated):
                print(f"error: failed to write {out_path}", file=sys.stderr)
                exit_code = 1
            else:
                payload["annotated"] = str(out_path)

        if not args.json:
            print(f"{path}: {len(detections)} face(s)")
            if "annotated" in payload:
                print(f"  wrote {payload['annotated']}")

    if args.json:
        print(json.dumps(results, indent=2))
    return exit_code


def run_camera(args: argparse.Namespace) -> int:
    detector = FaceDetector(
        scale_factor=args.scale_factor,
        min_neighbors=args.min_neighbors,
        min_size=tuple(args.min_size),
    )
    capture = cv2.VideoCapture(args.device)
    if not capture.isOpened():
        print(f"error: could not open camera device {args.device}", file=sys.stderr)
        return 1

    print("Press q to quit.")
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                print("error: failed to read frame", file=sys.stderr)
                return 1
            if args.mirror:
                frame = cv2.flip(frame, 1)
            detections = detector.detect(frame)
            preview = detector.annotate(frame, detections)
            cv2.putText(
                preview,
                f"faces: {len(detections)}",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 220, 80),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("face-detect", preview)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "image":
        return run_image(args)
    if args.command == "camera":
        return run_camera(args)
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
