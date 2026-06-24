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
    width: int | None = 640
    height: int | None = 480
    fps: int | None = 15
    buffer_size: int = 1

    def __post_init__(self) -> None:
        self.source = normalize_source(self.source)
        self.capture: Optional[cv2.VideoCapture] = None

    def open(self) -> None:
        if isinstance(self.source, int):
            self.capture = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)
            if not self.capture.isOpened():
                self.capture = cv2.VideoCapture(self.source)
            self.capture.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)
            if self.width:
                self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            if self.height:
                self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            if self.fps:
                self.capture.set(cv2.CAP_PROP_FPS, self.fps)
        else:
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
