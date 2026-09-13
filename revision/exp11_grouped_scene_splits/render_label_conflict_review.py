from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import cv2
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXP10_OUTPUTS = PROJECT_ROOT / "revision" / "exp10_perceptual_leakage_audit" / "outputs"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
REVIEW_DIR = OUTPUT_DIR / "label_conflict_review"
DATASETS_ROOT = Path(r"C:\SIRCH_ENV\datasets")
FRAME_POSITIONS = (0.05, 0.16, 0.27, 0.38, 0.50, 0.62, 0.73, 0.84, 0.95)
ROWS_PER_PAGE = 8


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def node_id(row: dict[str, str], suffix: str) -> str:
    return f"{row[f'split_{suffix}']}|{row[f'dataset_{suffix}']}|{row[f'relative_path_{suffix}']}"


def resolve_video(relative_path: str) -> Path:
    return DATASETS_ROOT / "RWF-2000" / relative_path


def temporal_frames(path: Path) -> list[object]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Video illisible: {path}")
    frame_count = max(1, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    frames = []
    for position in FRAME_POSITIONS:
        capture.set(cv2.CAP_PROP_POS_FRAMES, round((frame_count - 1) * position))
        ok, frame = capture.read()
        if not ok:
            capture.release()
            raise RuntimeError(f"Frame illisible a {position:.0%}: {path}")
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    return frames


def render_page(rows: list[dict[str, object]], page_index: int) -> None:
    fig, axes = plt.subplots(
        len(rows),
        len(FRAME_POSITIONS),
        figsize=(19, 2.35 * len(rows)),
        squeeze=False,
    )
    for row_index, row in enumerate(rows):
        frames = temporal_frames(resolve_video(str(row["relative_path"])))
        for frame_index, (position, frame) in enumerate(zip(FRAME_POSITIONS, frames)):
            axis = axes[row_index][frame_index]
            axis.imshow(frame)
            axis.axis("off")
            if row_index == 0:
                axis.set_title(f"{position:.0%}", fontsize=8)
        axes[row_index][0].set_ylabel(
            f"{row['review_id']} | {row['cluster_id']}\n"
            f"{row['split']} | label={row['original_label']} | "
            f"{Path(str(row['relative_path'])).name}\n"
            f"paires conflictuelles={row['conflict_pair_count']}",
            fontsize=7,
            rotation=0,
            ha="right",
            va="center",
            labelpad=8,
        )
    fig.suptitle(
        f"Revue visuelle des labels opposes - page {page_index:02d}",
        fontsize=14,
    )
    fig.tight_layout(rect=(0.11, 0, 1, 0.985))
    fig.savefig(
        REVIEW_DIR / f"conflict_review_page_{page_index:02d}.jpg",
        dpi=130,
        bbox_inches="tight",
        pil_kwargs={"quality": 88, "optimize": True},
    )
    plt.close(fig)


def main() -> None:
    findings = read_csv(EXP10_OUTPUTS / "near_duplicate_findings.csv")
    conflicts = [
        row
        for row in findings
        if row["classification"] == "probable_near_duplicate"
        and row["label_a"] != row["label_b"]
    ]
    if len(conflicts) != 90:
        raise RuntimeError(f"90 conflits attendus, {len(conflicts)} trouves")

    clusters = read_csv(EXP10_OUTPUTS / "near_duplicate_clusters.csv")
    cluster_by_node = {row["node_id"]: row["cluster_id"] for row in clusters}
    pair_counts: Counter[str] = Counter()
    videos: dict[str, dict[str, object]] = {}
    for pair_index, row in enumerate(conflicts, start=1):
        for suffix in ("a", "b"):
            identifier = node_id(row, suffix)
            pair_counts[identifier] += 1
            videos.setdefault(
                identifier,
                {
                    "node_id": identifier,
                    "cluster_id": cluster_by_node[identifier],
                    "split": row[f"split_{suffix}"],
                    "dataset": row[f"dataset_{suffix}"],
                    "relative_path": row[f"relative_path_{suffix}"],
                    "original_label": int(row[f"label_{suffix}"]),
                    "sha256": row[f"sha256_{suffix}"],
                },
            )

    ordered = sorted(
        videos.values(),
        key=lambda row: (str(row["cluster_id"]), str(row["relative_path"])),
    )
    for index, row in enumerate(ordered, start=1):
        row["review_id"] = f"V{index:03d}"
        row["conflict_pair_count"] = pair_counts[str(row["node_id"])]
        row["reviewed_label"] = ""
        row["review_decision"] = ""
        row["visual_evidence"] = ""

    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    for stale_png in REVIEW_DIR.glob("conflict_review_page_*.png"):
        stale_png.unlink()
    for page_index, offset in enumerate(range(0, len(ordered), ROWS_PER_PAGE), start=1):
        render_page(ordered[offset : offset + ROWS_PER_PAGE], page_index)

    write_csv(
        OUTPUT_DIR / "conflict_video_review.csv",
        ordered,
        [
            "review_id",
            "cluster_id",
            "node_id",
            "split",
            "dataset",
            "relative_path",
            "sha256",
            "original_label",
            "conflict_pair_count",
            "reviewed_label",
            "review_decision",
            "visual_evidence",
        ],
    )
    print(f"Paires conflictuelles: {len(conflicts)}")
    print(f"Videos uniques a revoir: {len(ordered)}")
    print(f"Pages produites: {(len(ordered) + ROWS_PER_PAGE - 1) // ROWS_PER_PAGE}")


if __name__ == "__main__":
    main()
