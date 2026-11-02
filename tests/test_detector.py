"""Unit tests for FaceDetector."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from face_detection import Detection, FaceDetector
from face_detection.detector import detections_to_dicts


@pytest.fixture
def detector() -> FaceDetector:
    return FaceDetector(scale_factor=1.1, min_neighbors=3, min_size=(20, 20))


def test_empty_image_raises(detector: FaceDetector) -> None:
    with pytest.raises(ValueError):
        detector.detect(np.array([]))


def test_invalid_scale_factor() -> None:
    with pytest.raises(ValueError):
        FaceDetector(scale_factor=1.0)


def test_missing_cascade_path() -> None:
    with pytest.raises(FileNotFoundError):
        FaceDetector(cascade_path="/tmp/does-not-exist-cascade.xml")


def test_blank_image_has_no_faces(detector: FaceDetector) -> None:
    blank = np.zeros((240, 320, 3), dtype=np.uint8)
    assert detector.detect(blank) == []


def test_annotate_draws_boxes(detector: FaceDetector) -> None:
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    detections = [Detection(10, 20, 40, 50)]
    annotated = detector.annotate(image, detections)
    assert annotated.shape == image.shape
    # Rectangle pixels should differ from the blank background.
    assert not np.array_equal(annotated, image)


def test_detections_to_dicts() -> None:
    data = detections_to_dicts([Detection(1, 2, 3, 4)])
    assert data == [{"x": 1, "y": 2, "width": 3, "height": 4}]


def test_detect_file_missing(detector: FaceDetector, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        detector.detect_file(tmp_path / "missing.jpg")


def test_detect_file_roundtrip(detector: FaceDetector, tmp_path: Path) -> None:
    path = tmp_path / "blank.png"
    cv2.imwrite(str(path), np.zeros((120, 160, 3), dtype=np.uint8))
    image, faces = detector.detect_file(path)
    assert image.shape == (120, 160, 3)
    assert faces == []


def _synthetic_face_image() -> np.ndarray:
    """Build a simple oval face that Haar cascades can occasionally catch.

    Haar cascades are trained on real photos, so this is mainly an API smoke
    path. Prefer real sample images for integration checks.
    """
    image = np.full((300, 300, 3), 210, dtype=np.uint8)
    cv2.ellipse(image, (150, 150), (70, 90), 0, 0, 360, (190, 170, 150), -1)
    cv2.circle(image, (125, 130), 8, (40, 40, 40), -1)
    cv2.circle(image, (175, 130), 8, (40, 40, 40), -1)
    cv2.ellipse(image, (150, 175), (20, 10), 0, 0, 180, (80, 60, 60), 2)
    return image


def test_detect_accepts_grayscale(detector: FaceDetector) -> None:
    gray = cv2.cvtColor(_synthetic_face_image(), cv2.COLOR_BGR2GRAY)
    faces = detector.detect(gray)
    assert isinstance(faces, list)
