from __future__ import annotations

import csv
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
DATASETS_ROOT = Path(r"C:\SIRCH_ENV\datasets")
POSITIONS = tuple(float(value) for value in np.linspace(0.05, 0.95, 9))


def frames(path: Path) -> list[object]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Video illisible: {path}")
    count = max(1, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    output = []
    for position in POSITIONS:
        capture.set(cv2.CAP_PROP_POS_FRAMES, round((count - 1) * position))
        ok, frame = capture.read()
        if not ok:
            capture.release()
            raise RuntimeError(f"Frame illisible: {path} a {position:.0%}")
        output.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    return output


def main() -> None:
    with (OUTPUT_DIR / "candidate_pairs.csv").open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise RuntimeError(f"Une paire candidate attendue, {len(rows)} trouvees")
    row = rows[0]
    paths = [
        DATASETS_ROOT / "RLVS" / "Real Life Violence Dataset" / row["relative_path_a"],
        PROJECT_ROOT / "datasets" / row["relative_path_b"],
    ]
    images = [frames(path) for path in paths]
    fig, axes = plt.subplots(2, len(POSITIONS), figsize=(18, 4.8), squeeze=False)
    for line, sampled in enumerate(images):
        for column, (position, image) in enumerate(zip(POSITIONS, sampled)):
            axis = axes[line][column]
            axis.imshow(image)
            axis.axis("off")
            if line == 0:
                axis.set_title(f"{position:.0%}", fontsize=8)
        axes[line][0].set_ylabel(
            ("train\n" if line == 0 else "hard-negative\n") + paths[line].name,
            rotation=0,
            ha="right",
            va="center",
            fontsize=8,
        )
    fig.suptitle(
        "Unique pHash candidate: isolated frame similarity, no ordered match",
        fontsize=12,
    )
    fig.tight_layout(rect=(0.09, 0, 1, 0.94))
    fig.savefig(
        OUTPUT_DIR / "candidate_pair_visual_review.jpg",
        dpi=140,
        pil_kwargs={"quality": 90, "optimize": True},
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
