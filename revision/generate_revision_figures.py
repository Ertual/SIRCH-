from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, roc_curve


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REVISION_ROOT = PROJECT_ROOT / "revision"
BOOTSTRAP_ITERATIONS = 10_000
BOOTSTRAP_SEED = 42

COLORS = {
    "blue": "#24506f",
    "green": "#2d7d65",
    "red": "#b4473c",
    "gold": "#b98522",
    "purple": "#72558a",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#30343b",
            "axes.labelcolor": "#20242a",
            "axes.titleweight": "bold",
            "font.size": 10,
            "legend.frameon": False,
            "grid.alpha": 0.22,
            "grid.color": "#737b84",
            "savefig.dpi": 200,
        }
    )


def plot_threshold_selection(
    grid_path: Path,
    selected_path: Path,
    output_path: Path,
    title: str,
) -> None:
    rows = read_csv(grid_path)
    selected = json.loads(selected_path.read_text(encoding="utf-8"))["selected"]
    figure, axis = plt.subplots(figsize=(8.6, 5.2))
    colors = [COLORS["blue"], COLORS["green"], COLORS["red"], COLORS["gold"], COLORS["purple"]]
    k_values = sorted({int(row["k"]) for row in rows})
    for color, k_value in zip(colors, k_values):
        subset = sorted(
            (row for row in rows if int(row["k"]) == k_value),
            key=lambda row: float(row["threshold"]),
        )
        axis.plot(
            [float(row["threshold"]) for row in subset],
            [float(row["f1"]) for row in subset],
            marker="o",
            markersize=3.5,
            linewidth=1.7,
            color=color,
            label=f"K={k_value}",
        )
    axis.scatter(
        [float(selected["threshold"])],
        [float(selected["f1"])],
        s=150,
        marker="*",
        color="#111111",
        zorder=5,
        label=(
            f"Selection: theta={float(selected['threshold']):.2f}, "
            f"K={int(selected['k'])}"
        ),
    )
    axis.set_xlabel("Seuil theta")
    axis.set_ylabel("F1 validation")
    axis.set_title(title)
    axis.set_xlim(0.28, 0.92)
    axis.set_ylim(0.0, 1.02)
    axis.grid(True)
    axis.legend(ncol=2, loc="lower left")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def training_rows(log_path: Path, recovery_path: Path | None = None) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    if recovery_path is not None and recovery_path.exists():
        recovery = json.loads(recovery_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "epoch": float(recovery["epoch"]),
                "accuracy": float(recovery["accuracy_displayed"]),
                "loss": float(recovery["loss_displayed"]),
                "precision": float(recovery["precision_displayed"]),
                "recall": float(recovery["recall_displayed"]),
                "val_accuracy": float("nan"),
                "val_loss": float(recovery["val_loss"]),
                "val_precision": float("nan"),
                "val_recall": float("nan"),
            }
        )
    for row in read_csv(log_path):
        rows.append(
            {
                "epoch": float(int(row["epoch"]) + 1),
                **{
                    key: float(row[key])
                    for key in (
                        "accuracy",
                        "loss",
                        "precision",
                        "recall",
                        "val_accuracy",
                        "val_loss",
                        "val_precision",
                        "val_recall",
                    )
                },
            }
        )
    return rows


def plot_training(
    log_path: Path,
    output_path: Path,
    title: str,
    best_epoch: int,
    weights_sha256: str,
    recovery_path: Path | None = None,
) -> None:
    rows = training_rows(log_path, recovery_path)
    epochs = np.asarray([row["epoch"] for row in rows])
    figure, axes = plt.subplots(2, 2, figsize=(11.2, 7.6), sharex=True)
    panels = (
        ("loss", "val_loss", "Loss", "Binary cross-entropy"),
        ("accuracy", "val_accuracy", "Accuracy", "Accuracy"),
        ("precision", "val_precision", "Precision", "Precision"),
        ("recall", "val_recall", "Rappel", "Rappel"),
    )
    for axis, (train_key, validation_key, panel_title, ylabel) in zip(axes.flat, panels):
        axis.plot(
            epochs,
            [row[train_key] for row in rows],
            color=COLORS["blue"],
            marker="o",
            markersize=3.5,
            label="Entrainement",
        )
        axis.plot(
            epochs,
            [row[validation_key] for row in rows],
            color=COLORS["red"],
            marker="s",
            markersize=3.5,
            label="Validation",
        )
        axis.axvline(best_epoch, color=COLORS["green"], linestyle="--", linewidth=1.4)
        axis.set_title(panel_title)
        axis.set_ylabel(ylabel)
        axis.grid(True)
        axis.legend(loc="best")
    for axis in axes[-1]:
        axis.set_xlabel("Epoch")
    figure.suptitle(f"{title}\nMeilleurs poids restaures: epoch {best_epoch}", fontsize=14)
    figure.text(
        0.5,
        0.01,
        f"SHA-256 des meilleurs poids: {weights_sha256}",
        ha="center",
        fontsize=8,
        color="#4b535b",
    )
    figure.tight_layout(rect=(0, 0.035, 1, 0.94))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)

    best_row = next(row for row in rows if int(row["epoch"]) == best_epoch)
    final_row = rows[-1]
    result = {
        "status": "training_complete",
        "training_log": str(log_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "epochs_completed": int(final_row["epoch"]),
        "best_epoch": best_epoch,
        "best_val_loss": best_row["val_loss"],
        "early_stopping_restored_best_weights": True,
        "best_weights_sha256": weights_sha256,
        "final_epoch_metrics": {
            key: final_row[key]
            for key in ("accuracy", "loss", "precision", "recall", "val_accuracy", "val_loss")
        },
    }
    result_path = output_path.parent / "training_result.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    summary = [
        f"# {title}",
        "",
        f"- Epochs termines : {int(final_row['epoch'])}",
        f"- Meilleur epoch restaure : {best_epoch}",
        f"- Meilleure val_loss : {best_row['val_loss']:.6f}",
        f"- Accuracy finale journalisee : {final_row['accuracy']:.4f}",
        f"- Val_accuracy finale journalisee : {final_row['val_accuracy']:.4f}",
        f"- SHA-256 des meilleurs poids : `{weights_sha256}`",
        "",
        "Le fichier de poids lui-meme reste hors Git; son empreinte verrouille l'artefact utilise.",
    ]
    (output_path.parent / "training_summary.md").write_text(
        "\n".join(summary) + "\n", encoding="utf-8"
    )


def video_score(row: dict[str, str]) -> float:
    if "video_score" in row:
        return float(row["video_score"])
    candidates = [key for key in row if key.startswith("max_rolling_mean_k")]
    if len(candidates) != 1:
        raise RuntimeError("Colonne de score video absente ou ambigue.")
    return float(row[candidates[0]])


def roc_data(predictions_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    rows = read_csv(predictions_path)
    truth = np.asarray([int(row["label"]) for row in rows], dtype=np.int8)
    scores = np.asarray([video_score(row) for row in rows], dtype=float)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(truth, scores)
    return false_positive_rate, true_positive_rate, thresholds, float(
        auc(false_positive_rate, true_positive_rate)
    )


def plot_roc(
    predictions_path: Path,
    output_path: Path,
    title: str,
    color: str,
) -> float:
    fpr, tpr, thresholds, area = roc_data(predictions_path)
    rows = [
        {
            "false_positive_rate": f"{x:.10f}",
            "true_positive_rate": f"{y:.10f}",
            "threshold": "inf" if np.isinf(threshold) else f"{threshold:.10f}",
        }
        for x, y, threshold in zip(fpr, tpr, thresholds)
    ]
    write_csv(output_path.with_suffix(".csv"), rows)
    figure, axis = plt.subplots(figsize=(6.4, 5.5))
    axis.plot(fpr, tpr, color=color, linewidth=2.2, label=f"ROC-AUC = {area:.4f}")
    axis.plot([0, 1], [0, 1], color="#737b84", linestyle="--", linewidth=1.2)
    axis.set_xlabel("Taux de faux positifs")
    axis.set_ylabel("Taux de vrais positifs")
    axis.set_title(title)
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1.02)
    axis.grid(True)
    axis.legend(loc="lower right")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)
    return area


def plot_combined_roc(
    specifications: list[tuple[str, Path, str]], output_path: Path
) -> None:
    figure, axis = plt.subplots(figsize=(7.4, 6.2))
    for label, path, color in specifications:
        fpr, tpr, _, area = roc_data(path)
        axis.plot(fpr, tpr, color=color, linewidth=2, label=f"{label} (AUC={area:.4f})")
    axis.plot([0, 1], [0, 1], color="#737b84", linestyle="--", linewidth=1.2)
    axis.set_xlabel("Taux de faux positifs")
    axis.set_ylabel("Taux de vrais positifs")
    axis.set_title("Courbes ROC du factoriel 2x2 - test propre")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1.02)
    axis.grid(True)
    axis.legend(loc="lower right")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def plot_factorial(
    specifications: dict[str, Path],
    factorial_result_path: Path,
    output_path: Path,
) -> None:
    correctness: dict[str, np.ndarray] = {}
    for key, path in specifications.items():
        rows = read_csv(path)
        correctness[key] = np.asarray([int(row["correct"]) for row in rows], dtype=float)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = {
        key: np.empty(BOOTSTRAP_ITERATIONS, dtype=float) for key in specifications
    }
    n = len(next(iter(correctness.values())))
    for iteration in range(BOOTSTRAP_ITERATIONS):
        indices = rng.integers(0, n, size=n)
        for key in specifications:
            samples[key][iteration] = float(np.mean(correctness[key][indices]))

    rows = []
    intervals = {}
    for key in specifications:
        estimate = float(np.mean(correctness[key]))
        lower, upper = np.percentile(samples[key], [2.5, 97.5])
        intervals[key] = (estimate, float(lower), float(upper))
        architecture, enrichment = key.split("_", maxsplit=1)
        rows.append(
            {
                "cell": key,
                "architecture": architecture.upper(),
                "enrichment": enrichment,
                "accuracy": f"{estimate:.10f}",
                "ci95_lower": f"{lower:.10f}",
                "ci95_upper": f"{upper:.10f}",
                "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
                "seed": BOOTSTRAP_SEED,
            }
        )
    write_csv(output_path.with_suffix(".csv"), rows)

    factorial = json.loads(factorial_result_path.read_text(encoding="utf-8"))
    interaction = factorial["primary_interaction_test"]
    figure, axis = plt.subplots(figsize=(7.6, 5.6))
    x = np.arange(2)
    for architecture, color, marker in (
        ("lstm", COLORS["blue"], "o"),
        ("gru", COLORS["red"], "s"),
    ):
        keys = [f"{architecture}_original", f"{architecture}_augmented"]
        estimates = np.asarray([intervals[key][0] for key in keys])
        lower = np.asarray([intervals[key][1] for key in keys])
        upper = np.asarray([intervals[key][2] for key in keys])
        axis.errorbar(
            x,
            estimates,
            yerr=np.vstack([estimates - lower, upper - estimates]),
            color=color,
            marker=marker,
            markersize=7,
            linewidth=2.2,
            capsize=5,
            label=architecture.upper(),
        )
    axis.set_xticks(x, labels=["Original", "Enrichi"])
    axis.set_ylabel("Accuracy sur le test propre")
    axis.set_title("Factoriel 2x2 : architecture x enrichissement")
    axis.set_ylim(0.82, 0.95)
    axis.grid(axis="y")
    axis.legend(loc="best")
    axis.text(
        0.02,
        0.03,
        (
            f"Interaction = {interaction['estimate'] * 100:+.2f} points\n"
            f"IC95 % [{interaction['ci95_lower'] * 100:+.2f}; "
            f"{interaction['ci95_upper'] * 100:+.2f}], p={interaction['exact_sign_flip_two_sided_p_value']:.3f}"
        ),
        transform=axis.transAxes,
        va="bottom",
        bbox={"boxstyle": "square,pad=0.4", "facecolor": "white", "edgecolor": "#9aa1a8"},
    )
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    configure_style()
    generated: list[Path] = []

    threshold_specs = (
        (
            REVISION_ROOT / "exp02_validation_threshold_k" / "outputs",
            "LSTM original - selection de theta/K sur validation",
        ),
        (
            REVISION_ROOT / "exp05_augmented_evaluation" / "outputs",
            "LSTM enrichi - selection de theta/K sur validation",
        ),
        (
            REVISION_ROOT / "exp08_gru_factorial_evaluation" / "outputs" / "gru_original",
            "GRU original - selection de theta/K sur validation",
        ),
        (
            REVISION_ROOT / "exp08_gru_factorial_evaluation" / "outputs" / "gru_augmented",
            "GRU enrichi - selection de theta/K sur validation",
        ),
    )
    for directory, title in threshold_specs:
        output = directory / "threshold_selection_curve.png"
        plot_threshold_selection(
            directory / "selection_grid.csv",
            directory / "selected_config.json",
            output,
            title,
        )
        generated.append(output)

    training_specs = (
        {
            "directory": REVISION_ROOT / "exp04_augmented_clean" / "outputs",
            "log": "training_log_lstm_augmented_clean.csv",
            "title": "Entrainement LSTM enrichi propre",
            "best_epoch": 6,
            "weights_sha256": "4ccfe5bb3838cd829fa5e4eaba0332128d14e52380509eee2d4083c321aeaa13",
            "recovery": "epoch_001_recovery.json",
        },
        {
            "directory": REVISION_ROOT / "exp07_gru_augmented_clean" / "outputs",
            "log": "training_log_gru_augmented_clean.csv",
            "title": "Entrainement GRU enrichi propre",
            "best_epoch": 5,
            "weights_sha256": "f7d3a87697d6b6b1ca36d46cdf7c1af6a499557b3e60dccc06078ff26a15956e",
            "recovery": None,
        },
    )
    for spec in training_specs:
        directory = spec["directory"]
        output = directory / "training_curves.png"
        recovery = directory / spec["recovery"] if spec["recovery"] else None
        plot_training(
            directory / spec["log"],
            output,
            spec["title"],
            spec["best_epoch"],
            spec["weights_sha256"],
            recovery,
        )
        generated.extend(
            [output, directory / "training_result.json", directory / "training_summary.md"]
        )

    roc_specs = [
        (
            "LSTM original",
            REVISION_ROOT / "exp02_validation_threshold_k" / "outputs" / "final_test_locked" / "video_predictions.csv",
            REVISION_ROOT / "exp02_validation_threshold_k" / "outputs" / "final_test_locked" / "roc_curve.png",
            COLORS["blue"],
        ),
        (
            "LSTM enrichi",
            REVISION_ROOT / "exp05_augmented_evaluation" / "outputs" / "final_test_locked" / "video_predictions.csv",
            REVISION_ROOT / "exp05_augmented_evaluation" / "outputs" / "final_test_locked" / "roc_curve.png",
            COLORS["green"],
        ),
        (
            "GRU original",
            REVISION_ROOT / "exp08_gru_factorial_evaluation" / "outputs" / "final_test_locked" / "gru_original" / "video_predictions.csv",
            REVISION_ROOT / "exp08_gru_factorial_evaluation" / "outputs" / "final_test_locked" / "gru_original" / "roc_curve.png",
            COLORS["red"],
        ),
        (
            "GRU enrichi",
            REVISION_ROOT / "exp08_gru_factorial_evaluation" / "outputs" / "final_test_locked" / "gru_augmented" / "video_predictions.csv",
            REVISION_ROOT / "exp08_gru_factorial_evaluation" / "outputs" / "final_test_locked" / "gru_augmented" / "roc_curve.png",
            COLORS["gold"],
        ),
    ]
    combined_specs = []
    for label, predictions, output, color in roc_specs:
        plot_roc(predictions, output, f"Courbe ROC - {label}", color)
        generated.extend([output, output.with_suffix(".csv")])
        combined_specs.append((label, predictions, color))

    factorial_dir = (
        REVISION_ROOT
        / "exp08_gru_factorial_evaluation"
        / "outputs"
        / "factorial_2x2"
    )
    combined_roc = factorial_dir / "roc_curves_2x2.png"
    plot_combined_roc(combined_specs, combined_roc)
    generated.append(combined_roc)

    factorial_predictions = {
        "lstm_original": roc_specs[0][1],
        "lstm_augmented": roc_specs[1][1],
        "gru_original": roc_specs[2][1],
        "gru_augmented": roc_specs[3][1],
    }
    factorial_figure = factorial_dir / "factorial_accuracy_interaction.png"
    plot_factorial(
        factorial_predictions,
        factorial_dir / "factorial_results.json",
        factorial_figure,
    )
    generated.extend([factorial_figure, factorial_figure.with_suffix(".csv")])

    generated_by_experiment: dict[str, list[Path]] = {}
    for path in generated:
        experiment = path.relative_to(REVISION_ROOT).parts[0]
        generated_by_experiment.setdefault(experiment, []).append(path)

    for experiment, paths in generated_by_experiment.items():
        experiment_manifest = REVISION_ROOT / experiment / "outputs" / "figure_manifest.json"
        experiment_manifest.write_text(
            json.dumps(
                {
                    "status": "complete",
                    "generator": "revision/generate_revision_figures.py",
                    "files": [
                        {
                            "path": str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                            "sha256": sha256_file(path),
                            "bytes": path.stat().st_size,
                        }
                        for path in paths
                    ],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        generated.append(experiment_manifest)

    manifest_rows = [
        {
            "path": str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for path in generated
    ]
    manifest_path = REVISION_ROOT / "figure_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "status": "complete",
                "generator": "revision/generate_revision_figures.py",
                "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
                "bootstrap_seed": BOOTSTRAP_SEED,
                "files": manifest_rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"generated": len(generated), "manifest": str(manifest_path)}, indent=2))


if __name__ == "__main__":
    main()
