from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import cv2


VideoInput = Union[int, str]


def normalize_source(source: Union[int, str]) -> VideoInput:
    """Convert numeric camera strings to camera indices."""
    if isinstance(source, str) and source.isdigit():
        return int(source)
    return source


@dataclass
class VideoSource:
    source: VideoInput = 0

    def __post_init__(self) -> None:
        self.source = normalize_source(self.source)
        self.capture: Optional[cv2.VideoCapture] = None

    def open(self) -> None:
        self.capture = cv2.VideoCapture(self.source)
        if not self.capture.isOpened():
            raise RuntimeError(f"Impossible d'ouvrir la source video : {self.source}")

    def read(self):
        if self.capture is None:
            self.open()
        ok, frame = self.capture.read()
        if not ok:
            return None
        return frame

    def release(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None

    def __enter__(self) -> "VideoSource":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()


def list_available_cameras(max_index: int = 5) -> list[int]:
    cameras: list[int] = []
    for index in range(max_index + 1):
        capture = cv2.VideoCapture(index)
        if capture.isOpened():
            cameras.append(index)
        capture.release()
    return cameras
