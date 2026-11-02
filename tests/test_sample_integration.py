"""Integration check against a real portrait when available."""

from __future__ import annotations

from pathlib import Path

import pytest

from face_detection import FaceDetector

SAMPLE = Path(__file__).resolve().parents[1] / "assets" / "sample_face.png"


@pytest.mark.skipif(not SAMPLE.is_file(), reason="sample image not downloaded")
def test_detects_face_in_sample() -> None:
    detector = FaceDetector(min_neighbors=3, min_size=(40, 40))
    _, faces = detector.detect_file(SAMPLE)
    assert len(faces) >= 1
    face = faces[0]
    assert face.width > 20 and face.height > 20
