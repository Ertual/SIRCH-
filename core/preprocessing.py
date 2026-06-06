from __future__ import annotations

from collections import deque
from typing import Deque, Optional

import cv2
import numpy as np
from tensorflow.keras.applications.efficientnet import preprocess_input

from config import IMG_SIZE, N_FRAMES


def resize_frame(frame: np.ndarray, img_size: int = IMG_SIZE) -> np.ndarray:
    return cv2.resize(frame, (img_size, img_size))


def preprocess_frame(frame: np.ndarray, img_size: int = IMG_SIZE) -> np.ndarray:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = resize_frame(rgb, img_size)
    normalized = resized.astype(np.float32)
    return preprocess_input(normalized)


def build_sequence(frames: list[np.ndarray]) -> np.ndarray:
    if len(frames) != N_FRAMES:
        raise ValueError(f"Une sequence doit contenir exactement {N_FRAMES} frames.")
    return np.expand_dims(np.stack(frames, axis=0), axis=0)


class FrameSequenceBuffer:
    def __init__(self, n_frames: int = N_FRAMES, img_size: int = IMG_SIZE) -> None:
        self.n_frames = n_frames
        self.img_size = img_size
        self.frames: Deque[np.ndarray] = deque(maxlen=n_frames)

    def add_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        self.frames.append(preprocess_frame(frame, self.img_size))
        if len(self.frames) < self.n_frames:
            return None
        return np.expand_dims(np.stack(list(self.frames), axis=0), axis=0)

    def clear(self) -> None:
        self.frames.clear()

    def is_ready(self) -> bool:
        return len(self.frames) == self.n_frames
