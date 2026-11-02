"""OpenCV Haar-cascade face detector."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np

_PACKAGE_DIR = Path(__file__).resolve().parent
_VENDORED_CASCADE = _PACKAGE_DIR / "data" / "haarcascade_frontalface_default.xml"


def _default_cascade_path() -> Path:
    """Prefer the vendored cascade; fall back to OpenCV's data dir on 4.x."""
    if _VENDORED_CASCADE.is_file():
        return _VENDORED_CASCADE
    opencv_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    return opencv_path


@dataclass(frozen=True)
class Detection:
    """Axis-aligned face box in pixel coordinates."""

    x: int
    y: int
    width: int
    height: int

    @property
    def as_tuple(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.width, self.height

    @property
    def center(self) -> tuple[int, int]:
        return self.x + self.width // 2, self.y + self.height // 2


class FaceDetector:
    """Detect frontal faces using OpenCV's built-in Haar cascade."""

    def __init__(
        self,
        *,
        scale_factor: float = 1.1,
        min_neighbors: int = 5,
        min_size: tuple[int, int] = (30, 30),
        cascade_path: str | Path | None = None,
    ) -> None:
        if scale_factor <= 1.0:
            raise ValueError("scale_factor must be greater than 1.0")
        if min_neighbors < 1:
            raise ValueError("min_neighbors must be at least 1")

        path = Path(cascade_path) if cascade_path else _default_cascade_path()
        if not path.is_file():
            raise FileNotFoundError(f"Haar cascade not found: {path}")

        cascade = cv2.CascadeClassifier(str(path))
        if cascade.empty():
            raise RuntimeError(f"Failed to load Haar cascade: {path}")

        self._cascade = cascade
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size

    def detect(self, image: np.ndarray) -> list[Detection]:
        """Return face boxes for a BGR or grayscale image."""
        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            raise ValueError("image must be a non-empty numpy array")

        gray = self._to_gray(image)
        boxes = self._cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_size,
            flags=cv2.CASCADE_SCALE_IMAGE,
        )
        return [Detection(int(x), int(y), int(w), int(h)) for x, y, w, h in boxes]

    def detect_file(self, path: str | Path) -> tuple[np.ndarray, list[Detection]]:
        """Load an image from disk and detect faces."""
        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        return image, self.detect(image)

    def annotate(
        self,
        image: np.ndarray,
        detections: Sequence[Detection],
        *,
        color: tuple[int, int, int] = (0, 220, 80),
        thickness: int = 2,
        label: bool = True,
    ) -> np.ndarray:
        """Draw face boxes onto a copy of the image."""
        annotated = image.copy()
        for index, det in enumerate(detections, start=1):
            x, y, w, h = det.as_tuple
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, thickness)
            if label:
                text = f"face {index}"
                cv2.putText(
                    annotated,
                    text,
                    (x, max(0, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    1,
                    cv2.LINE_AA,
                )
        return annotated

    @staticmethod
    def _to_gray(image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            return image
        if image.ndim == 3 and image.shape[2] == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if image.ndim == 3 and image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        raise ValueError(f"Unsupported image shape: {image.shape}")


def detections_to_dicts(detections: Iterable[Detection]) -> list[dict[str, int]]:
    return [
        {"x": d.x, "y": d.y, "width": d.width, "height": d.height}
        for d in detections
    ]
