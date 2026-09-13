from __future__ import annotations

import csv
from pathlib import Path

import cv2
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
DATASETS_ROOT = Path(r"C:\SIRCH_ENV\datasets")


def resolve(dataset: str, relative_path: str) -> Path:
    if dataset == "RLVS":
        return DATASETS_ROOT / "RLVS" / "Real Life Violence Dataset" / relative_path
    if dataset == "RWF-2000":
        return DATASETS_ROOT / "RWF-2000" / relative_path
    if dataset == "historical_enrichment_v1":
        return PROJECT_ROOT / "datasets" / relative_path
    raise RuntimeError(dataset)


def middle_frame(path: Path):
    capture = cv2.VideoCapture(str(path))
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, count // 2))
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"Frame illisible: {path}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def render(rows: list[dict[str, str]], title: str, path: Path) -> None:
    fig, axes = plt.subplots(len(rows), 2, figsize=(12, 2.55 * len(rows)))
    if len(rows) == 1:
        axes = [axes]
    for row_index, row in enumerate(rows):
        for side, suffix in enumerate(("a", "b")):
            image = middle_frame(resolve(row[f"dataset_{suffix}"], row[f"relative_path_{suffix}"]))
            axis = axes[row_index][side]
            axis.imshow(image)
            axis.axis("off")
            axis.set_title(
                f"{row[f'split_{suffix}']} | label={row[f'label_{suffix}']} | "
                f"{Path(row[f'relative_path_{suffix}']).name}",
                fontsize=8,
            )
        axes[row_index][0].set_ylabel(
            f"pHash min={row['min_phash_hamming']}\n"
            f"ordonnees={row['ordered_matches_h6']}/{row['ordered_matches_h8']}",
            fontsize=8,
        )
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    with (OUTPUT_DIR / "near_duplicate_findings.csv").open(
        newline="", encoding="utf-8-sig"
    ) as handle:
        rows = list(csv.DictReader(handle))
    probable = [row for row in rows if row["classification"] == "probable_near_duplicate"][:8]
    possible = [row for row in rows if row["classification"] == "possible_near_duplicate"][:8]
    borderline = [row for row in rows if row["classification"] == "possible_near_duplicate"][-8:]
    render(
        probable,
        "Controle visuel - huit quasi-doublons probables les plus forts",
        OUTPUT_DIR / "review_contact_sheet_probable.png",
    )
    render(
        possible,
        "Controle visuel - huit quasi-doublons possibles les plus forts",
        OUTPUT_DIR / "review_contact_sheet_possible.png",
    )
    render(
        borderline,
        "Controle visuel - huit quasi-doublons possibles les plus limites",
        OUTPUT_DIR / "review_contact_sheet_borderline.png",
    )


if __name__ == "__main__":
    main()
