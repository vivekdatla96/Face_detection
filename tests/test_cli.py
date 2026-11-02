"""CLI smoke tests."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from face_detection.cli import main


def test_image_command_json(tmp_path: Path) -> None:
    path = tmp_path / "blank.png"
    cv2.imwrite(str(path), np.zeros((80, 80, 3), dtype=np.uint8))
    code = main(["image", str(path), "--json", "--no-save"])
    assert code == 0


def test_image_command_writes_annotated(tmp_path: Path, capsys) -> None:
    path = tmp_path / "blank.png"
    cv2.imwrite(str(path), np.zeros((80, 80, 3), dtype=np.uint8))
    code = main(["image", str(path), "--json"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload[0]["count"] == 0
    assert Path(payload[0]["annotated"]).is_file()


def test_missing_image_returns_error() -> None:
    code = main(["image", "/tmp/definitely-missing-face-image.jpg", "--no-save"])
    assert code == 1
