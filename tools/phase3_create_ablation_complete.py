"""Build Phase 3 H3 ablation comparison artifacts.

This script consolidates the already computed evaluations for:
- H1 EfficientNetB0 only
- Original EfficientNetB0 + LSTM
- H2 EfficientNetB0 + GRU

Outputs:
- C:\SIRCH_ENV\models\phase3\ablation_complete.csv
- C:\SIRCH_ENV\models\phase3\figures\ablation_complete.png
"""

from __future__ import annotations

import csv
from pathlib import Path


PHASE3_DIR = Path(r"C:\SIRCH_ENV\models\phase3")
MODELS_DIR = Path(r"C:\SIRCH_ENV\models")
FIGURES_DIR = PHASE3_DIR / "figures"

H1_CSV = PHASE3_DIR / "evaluation_efficientnet_only.csv"
H2_CSV = PHASE3_DIR / "evaluation_gru.csv"
ORIGINAL_ABLATION_CSV = MODELS_DIR / "ablation_results.csv"

OUTPUT_CSV = PHASE3_DIR / "ablation_complete.csv"
OUTPUT_PNG = FIGURES_DIR / "ablation_complete.png"


def read_single_row(path: Path) -> dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable: {path}")
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"Fichier vide: {path}")
    return rows[0]


def as_float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    if value == "":
        return float("nan")
    return float(value)


def as_int(row: dict[str, str], key: str) -> int:
    value = row.get(key, "")
    if value == "":
        return 0
    return int(float(value))


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def build_rows() -> list[dict[str, object]]:
    h1 = read_single_row(H1_CSV)
    h2 = read_single_row(H2_CSV)

    with ORIGINAL_ABLATION_CSV.open("r", newline="", encoding="utf-8-sig") as handle:
        original_rows = list(csv.DictReader(handle))
    original = next(
        (
            row
            for row in original_rows
            if row.get("configuration") == "efficientnet_lstm_n20"
            and str(row.get("threshold")) == "0.5"
        ),
        None,
    )
    if original is None:
        raise RuntimeError(
            "Ligne original efficientnet_lstm_n20 introuvable dans "
            f"{ORIGINAL_ABLATION_CSV}"
        )

    rows = [
        {
            "configuration": "efficientnet_only",
            "description": "EfficientNetB0 seul, sans module recurrent",
            "threshold": as_float(h1, "threshold"),
            "n_test": as_int(h1, "n_test"),
            "tn": as_int(h1, "tn"),
            "fp": as_int(h1, "fp"),
            "fn": as_int(h1, "fn"),
            "tp": as_int(h1, "tp"),
            "accuracy": as_float(h1, "accuracy"),
            "precision": as_float(h1, "precision"),
            "recall": as_float(h1, "recall"),
            "f1_score": as_float(h1, "f1_score"),
            "ms_per_frame": as_float(h1, "ms_per_frame"),
            "roc_auc": "",
            "average_precision": "",
            "model_path": h1.get("model_path", ""),
            "training_log": h1.get("training_log", ""),
            "source_csv": str(H1_CSV),
        },
        {
            "configuration": "efficientnet_lstm_original",
            "description": "EfficientNetB0 + LSTM 256, modele original epoch 4",
            "threshold": as_float(original, "threshold"),
            "n_test": as_int(original, "n_test"),
            "tn": as_int(original, "tn"),
            "fp": as_int(original, "fp"),
            "fn": as_int(original, "fn"),
            "tp": as_int(original, "tp"),
            "accuracy": as_float(original, "accuracy"),
            "precision": as_float(original, "precision"),
            "recall": as_float(original, "recall"),
            "f1_score": as_float(original, "f1"),
            "ms_per_frame": 36.87481104034016,
            "roc_auc": as_float(original, "roc_auc"),
            "average_precision": as_float(original, "average_precision"),
            "model_path": str(MODELS_DIR / "sirch_model.h5"),
            "training_log": str(MODELS_DIR / "training_log.csv"),
            "source_csv": str(ORIGINAL_ABLATION_CSV),
        },
        {
            "configuration": "efficientnet_gru",
            "description": "EfficientNetB0 + GRU 256, phase 3 H2",
            "threshold": as_float(h2, "threshold"),
            "n_test": as_int(h2, "n_test"),
            "tn": as_int(h2, "tn"),
            "fp": as_int(h2, "fp"),
            "fn": as_int(h2, "fn"),
            "tp": as_int(h2, "tp"),
            "accuracy": as_float(h2, "accuracy"),
            "precision": as_float(h2, "precision"),
            "recall": as_float(h2, "recall"),
            "f1_score": as_float(h2, "f1_score"),
            "ms_per_frame": as_float(h2, "ms_per_frame"),
            "roc_auc": "",
            "average_precision": "",
            "model_path": h2.get("model_path", ""),
            "training_log": h2.get("training_log", ""),
            "source_csv": str(H2_CSV),
        },
    ]
    return rows


def write_csv(rows: list[dict[str, object]]) -> None:
    PHASE3_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "configuration",
        "description",
        "threshold",
        "n_test",
        "tn",
        "fp",
        "fn",
        "tp",
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "ms_per_frame",
        "roc_auc",
        "average_precision",
        "model_path",
        "training_log",
        "source_csv",
    ]
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_png(rows: list[dict[str, object]]) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    labels = ["EffNet seul", "EffNet+LSTM", "EffNet+GRU"]
    metrics = ["accuracy", "precision", "recall", "f1_score"]
    metric_labels = ["Accuracy", "Precision", "Rappel", "F1-score"]

    values = np.array([[float(row[m]) for row in rows] for m in metrics])
    x = np.arange(len(labels))
    width = 0.18

    fig = plt.figure(figsize=(13, 8), constrained_layout=True)
    grid = fig.add_gridspec(2, 1, height_ratios=[2.2, 1.4])

    ax = fig.add_subplot(grid[0])
    for idx, metric_label in enumerate(metric_labels):
        ax.bar(x + (idx - 1.5) * width, values[idx] * 100, width, label=metric_label)

    ax.set_title("P3-H - Ablation complete sur le jeu de test principal (599 videos)")
    ax.set_ylabel("Score (%)")
    ax.set_ylim(75, 100)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncols=4, loc="upper center", bbox_to_anchor=(0.5, -0.08))

    ax_table = fig.add_subplot(grid[1])
    ax_table.axis("off")

    table_rows = []
    for label, row in zip(labels, rows):
        table_rows.append(
            [
                label,
                pct(float(row["accuracy"])),
                pct(float(row["precision"])),
                pct(float(row["recall"])),
                pct(float(row["f1_score"])),
                f'{float(row["ms_per_frame"]):.2f}',
                f'{int(row["tn"])}/{int(row["fp"])}/{int(row["fn"])}/{int(row["tp"])}',
            ]
        )

    table = ax_table.table(
        cellText=table_rows,
        colLabels=[
            "Modele",
            "Accuracy",
            "Precision",
            "Rappel",
            "F1",
            "ms/frame",
            "TN/FP/FN/TP",
        ],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    fig.savefig(OUTPUT_PNG, dpi=180)
    plt.close(fig)


def main() -> None:
    rows = build_rows()
    write_csv(rows)
    write_png(rows)

    print(f"CSV sauvegarde : {OUTPUT_CSV}")
    print(f"PNG sauvegarde : {OUTPUT_PNG}")
    for row in rows:
        print(
            f"{row['configuration']}: "
            f"accuracy={float(row['accuracy']):.4f}, "
            f"precision={float(row['precision']):.4f}, "
            f"recall={float(row['recall']):.4f}, "
            f"f1={float(row['f1_score']):.4f}, "
            f"ms/frame={float(row['ms_per_frame']):.2f}"
        )


if __name__ == "__main__":
    main()
