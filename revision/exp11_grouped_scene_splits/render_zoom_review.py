from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REVIEW_CSV = Path(__file__).resolve().parent / "outputs" / "conflict_video_review.csv"
DATASETS_ROOT = Path(r"C:\SIRCH_ENV\datasets")
DEFAULT_OUTPUT_DIR = Path(os.environ.get("TEMP", ".")) / "sirch_conflict_zoom"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("review_ids", nargs="+")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def read_rows() -> dict[str, dict[str, str]]:
    with REVIEW_CSV.open(newline="", encoding="utf-8-sig") as handle:
        return {row["review_id"]: row for row in csv.DictReader(handle)}


def extract_frames(path: Path, count: int = 24) -> list[object]:
    capture = cv2.VideoCapture(str(path))
    frame_count = max(1, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    frames = []
    for index in range(count):
        position = (index + 0.5) / count
        capture.set(cv2.CAP_PROP_POS_FRAMES, round((frame_count - 1) * position))
        ok, frame = capture.read()
        if not ok:
            capture.release()
            raise RuntimeError(f"Frame illisible: {path} a {position:.1%}")
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    return frames


def render(row: dict[str, str], output_dir: Path) -> None:
    path = DATASETS_ROOT / "RWF-2000" / row["relative_path"]
    frames = extract_frames(path)
    fig, axes = plt.subplots(4, 6, figsize=(18, 10), squeeze=False)
    for index, frame in enumerate(frames):
        axis = axes[index // 6][index % 6]
        axis.imshow(frame)
        axis.axis("off")
        axis.set_title(f"{(index + 0.5) / len(frames):.0%}", fontsize=8)
    fig.suptitle(
        f"{row['review_id']} | {row['cluster_id']} | label={row['original_label']} | "
        f"{Path(row['relative_path']).name}",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_dir / f"{row['review_id']}_zoom.jpg",
        dpi=140,
        pil_kwargs={"quality": 92, "optimize": True},
    )
    plt.close(fig)


def main() -> None:
    args = parse_args()
    rows = read_rows()
    missing = [review_id for review_id in args.review_ids if review_id not in rows]
    if missing:
        raise RuntimeError(f"Identifiants inconnus: {missing}")
    for review_id in args.review_ids:
        render(rows[review_id], args.output_dir)
    print(f"Planches agrandies produites: {len(args.review_ids)}")


if __name__ == "__main__":
    main()
