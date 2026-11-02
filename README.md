# Face Detection

Detect faces in images and webcam streams using OpenCV Haar cascades.

The cascade XML is vendored under `src/face_detection/data/` so detection works
even when OpenCV 5+ omits bundled Haar data. Runtime depends on
`opencv-python-headless` 4.x (`<5`).

## Setup

```bash
python3 -m pip install -r requirements.txt
python3 -m pip install -e .
```

## Quick start

Download a sample portrait and run detection:

```bash
python3 scripts/fetch_sample.py
python3 -m face_detection image assets/sample_face.jpg
```

JSON output:

```bash
python3 -m face_detection image assets/sample_face.jpg --json --no-save
```

Webcam (requires a display and camera):

```bash
python3 -m face_detection camera --mirror
```

## Library usage

```python
from face_detection import FaceDetector

detector = FaceDetector()
image, faces = detector.detect_file("photo.jpg")
annotated = detector.annotate(image, faces)
```

## Tests

```bash
python3 scripts/fetch_sample.py   # optional integration sample
python3 -m pytest
```

## Project layout

```
src/face_detection/   # detector + CLI + vendored Haar cascade
tests/                # unit + integration tests
scripts/              # helpers (sample download)
assets/               # local sample images (downloaded, gitignored)
```
