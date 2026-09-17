from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "outputs" / "inspection"
DATASETS = Path(r"C:\SIRCH_ENV\datasets")
CASES = (
    ("FP", "RWF-2000", "train/NonFight/rwf_train_nonfight_0542.avi"),
    ("FP", "RWF-2000", "val/NonFight/rwf_val_nonfight_0034.avi"),
    ("FP", "RWF-2000", "train/NonFight/rwf_train_nonfight_0057.avi"),
    ("FP", "RLVS", "NonViolence/NV_40.mp4"),
    ("FP", "RLVS", "NonViolence/NV_14.mp4"),
    ("FP", "RWF-2000", "train/NonFight/rwf_train_nonfight_0449.avi"),
    ("FN", "RWF-2000", "train/Fight/rwf_train_fight_0444.avi"),
    ("FN", "RWF-2000", "train/Fight/rwf_train_fight_0376.avi"),
    ("FN", "RWF-2000", "val/Fight/rwf_val_fight_0006.avi"),
    ("FN", "RLVS", "Violence/V_233.mp4"),
    ("FN", "RLVS", "Violence/V_671.mp4"),
    ("FN", "RLVS", "Violence/V_634.mp4"),
)
FRAME_WIDTH = 300
FRAME_HEIGHT = 169


def path_for(dataset: str, relative_path: str) -> Path:
    if dataset == "RWF-2000":
        return DATASETS / dataset / relative_path
    return DATASETS / "RLVS" / "Real Life Violence Dataset" / relative_path


def tile(error_type: str, dataset: str, relative_path: str) -> np.ndarray:
    path = path_for(dataset, relative_path)
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open {path}")
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    try:
        for fraction in (0.10, 0.50, 0.90):
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(max(0, count - 1) * fraction))
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"Cannot read {path} at {fraction}")
            frames.append(cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT)))
    finally:
        capture.release()
    triptych = np.concatenate(frames, axis=1)
    label = np.full((34, triptych.shape[1], 3), (245, 245, 245), dtype=np.uint8)
    title = f"{error_type} | {dataset} | {Path(relative_path).name} | 10%, 50%, 90%"
    cv2.putText(label, title, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (25, 25, 25), 1, cv2.LINE_AA)
    return np.concatenate((label, triptych), axis=0)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for i in range(0, len(CASES), 4):
        tiles = [tile(*case) for case in CASES[i : i + 4]]
        sheet = np.concatenate(tiles, axis=0)
        destination = OUT / f"error_inspection_{i // 4 + 1}.jpg"
        if not cv2.imwrite(str(destination), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90]):
            raise RuntimeError(f"Cannot write {destination}")
        print(destination)


if __name__ == "__main__":
    main()
