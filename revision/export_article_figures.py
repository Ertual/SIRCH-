from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
REVISION = ROOT / "revision"
TRAIN = REVISION / "exp13_grouped_retraining" / "outputs"
EVAL = REVISION / "exp14_grouped_factorial_evaluation" / "outputs"
OUT = REVISION / "figures_article"
MODELS = (
    ("lstm_original_grouped", "LSTM / original", "#25577a"),
    ("lstm_augmented_grouped", "LSTM / enriched", "#268073"),
    ("gru_original_grouped", "GRU / original", "#b37824"),
    ("gru_augmented_grouped", "GRU / enriched", "#b34c53"),
)
K_COLORS = {1: "#25577a", 3: "#268073", 5: "#b37824", 7: "#b34c53", 10: "#705e91"}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def save_figure(figure: plt.Figure, stem: str) -> list[Path]:
    outputs = [OUT / f"{stem}.png", OUT / f"{stem}.svg"]
    figure.savefig(outputs[0], dpi=300, bbox_inches="tight", facecolor="white")
    figure.savefig(outputs[1], bbox_inches="tight", facecolor="white")
    outputs[1].write_bytes(outputs[1].read_bytes().replace(b"\r\n", b"\n"))
    plt.close(figure)
    return outputs


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#dce2e6",
            "grid.linewidth": 0.65,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "svg.fonttype": "none",
        }
    )


def verify_inputs() -> tuple[dict, dict, list[Path]]:
    lock_path = EVAL / "locked_protocol.json"
    factorial_path = EVAL / "factorial_2x2" / "factorial_results.json"
    lock = read_json(lock_path)
    factorial = read_json(factorial_path)
    assert lock["status"] == "locked_before_test"
    assert lock["validation_count"] == 597
    assert lock["test_manifest_not_read_before_lock"] is True
    assert factorial["same_597_test_rows_verified_by_sha256"] is True
    sources = [lock_path, factorial_path]

    for name, _, _ in MODELS:
        config_path = EVAL / "validation_selection" / name / "selected_config.json"
        grid_path = EVAL / "validation_selection" / name / "selection_grid.csv"
        log_path = TRAIN / name / f"training_log_{name}.csv"
        result_path = TRAIN / name / "training_result.json"
        roc_path = EVAL / "final_test_locked" / name / "roc_curve.csv"
        metrics_path = EVAL / "final_test_locked" / name / "final_test_metrics.json"
        sources.extend((config_path, grid_path, log_path, result_path, roc_path, metrics_path))

        locked = lock["selected_configurations"][name]
        selected = read_json(config_path)
        assert digest(config_path) == locked["selected_config_sha256"]
        assert selected["status"] == "selected_on_grouped_validation_only"
        assert selected["test_manifest_read"] is False
        assert selected["n_validation"] == 597
        assert float(selected["selected"]["threshold"]) == float(locked["theta"])
        assert int(selected["selected"]["k"]) == int(locked["k"])
        grid = read_csv(grid_path)
        assert len(grid) == 65 and {int(row["n_validation"]) for row in grid} == {597}
        assert sum(int(row["selected"]) for row in grid) == 1
        assert read_json(result_path)["status"] == "complete"
        assert factorial["cells"][name]["n"] == 597
        assert len(read_csv(roc_path)) > 2
        assert read_json(metrics_path)["metrics"]["n"] == 597
    return lock, factorial, sources


def architecture_box(ax, x, y, width, height, number, title, detail, color) -> None:
    box = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.008,rounding_size=0.01",
        linewidth=1.4, edgecolor=color, facecolor="white",
    )
    ax.add_patch(box)
    ax.text(x + 0.014, y + height - 0.045, f"{number}  {title}", color=color,
            weight="bold", fontsize=11, va="top")
    ax.text(x + 0.014, y + height - 0.095, detail, color="#26343d",
            fontsize=9, va="top", linespacing=1.35)


def arrow(ax, start, end, color="#52626c", dashed=False) -> None:
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=13, linewidth=1.7,
        color=color, linestyle="--" if dashed else "-",
        connectionstyle="arc3,rad=0", shrinkA=0, shrinkB=0,
    ))


def plot_architecture() -> list[Path]:
    figure, ax = plt.subplots(figsize=(14, 7.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.04, 0.96, "Figure 1. SIRCH system architecture", fontsize=19,
            weight="bold", color="#1d303a", va="top")
    ax.text(0.04, 0.91, "Seven operational modules; spatial and temporal stages share module 3",
            fontsize=10, color="#596771", va="top")

    architecture_box(ax, 0.04, 0.60, 0.16, 0.22, "1", "Acquisition",
                     "USB / IP camera\nor video file", "#25577a")
    architecture_box(ax, 0.25, 0.60, 0.18, 0.22, "2", "Preprocessing",
                     "BGR to RGB; 224 x 224\n20-frame buffer", "#268073")
    architecture_box(ax, 0.48, 0.57, 0.25, 0.28, "3", "Spatiotemporal model",
                     "EfficientNet-B0 (frozen)\n1280 features / frame\nLSTM 256 + sigmoid score", "#705e91")
    architecture_box(ax, 0.78, 0.60, 0.18, 0.22, "4", "Decision",
                     "Mean of last K scores\nAlert if mean >= theta", "#b37824")
    architecture_box(ax, 0.70, 0.17, 0.26, 0.22, "5", "Alerts",
                     "Sound / email / Telegram\n30 s cooldown", "#b34c53")
    architecture_box(ax, 0.38, 0.17, 0.25, 0.22, "6", "Incident log",
                     "SQLite: time, score,\nscreenshot", "#268073")
    architecture_box(ax, 0.04, 0.17, 0.27, 0.22, "7", "Operator dashboard",
                     "Flask: history, settings,\nCSV export", "#25577a")

    arrow(ax, (0.20, 0.71), (0.25, 0.71))
    arrow(ax, (0.43, 0.71), (0.48, 0.71))
    arrow(ax, (0.73, 0.71), (0.78, 0.71))
    arrow(ax, (0.87, 0.60), (0.83, 0.39))
    ax.plot([0.82, 0.82, 0.51], [0.60, 0.49, 0.49], color="#52626c", lw=1.7)
    arrow(ax, (0.51, 0.49), (0.51, 0.39))
    arrow(ax, (0.38, 0.28), (0.31, 0.28))
    ax.text(0.48, 0.45, "trigger + capture", fontsize=8.5, color="#596771",
            ha="center", va="top")
    ax.plot([0.17, 0.17, 0.975, 0.975], [0.17, 0.07, 0.07, 0.71],
            color="#858f97", lw=1.3, ls="--")
    arrow(ax, (0.975, 0.71), (0.96, 0.71), color="#858f97", dashed=True)
    ax.text(0.54, 0.085, "configurable theta / K", fontsize=8.5,
            color="#66727b", ha="center", va="bottom")
    return save_figure(figure, "figure_1_architecture")


def plot_thresholds(lock: dict) -> list[Path]:
    figure, axes = plt.subplots(2, 2, figsize=(12, 8.6), sharex=True, sharey=True)
    for ax, (name, label, _) in zip(axes.flat, MODELS):
        grid = read_csv(EVAL / "validation_selection" / name / "selection_grid.csv")
        selected = read_json(EVAL / "validation_selection" / name / "selected_config.json")["selected"]
        for k, color in K_COLORS.items():
            rows = sorted((row for row in grid if int(row["k"]) == k),
                          key=lambda row: float(row["threshold"]))
            ax.plot([float(row["threshold"]) for row in rows],
                    [float(row["f1"]) for row in rows], marker="o", markersize=2.8,
                    linewidth=1.5, color=color, label=f"K={k}")
        ax.scatter([float(selected["threshold"])], [float(selected["f1"])],
                   s=145, marker="*", color="#111111", zorder=5)
        ax.set_title(f"{label}  |  locked theta={lock['selected_configurations'][name]['theta']:.2f}, "
                     f"K={lock['selected_configurations'][name]['k']}", fontsize=11)
        ax.set_xlim(0.29, 0.91)
        ax.set_ylim(0.74, 0.90)
        ax.set_xticks(np.arange(0.30, 0.91, 0.10))
    figure.supxlabel("Decision threshold (theta)")
    figure.supylabel("F1 on grouped validation")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=5, frameon=False,
                  bbox_to_anchor=(0.5, 0.955))
    figure.suptitle("Joint theta / K selection - validation only (n=597)",
                    fontsize=16, weight="bold", y=1.0)
    figure.tight_layout(rect=(0.02, 0.02, 1, 0.91))
    return save_figure(figure, "figure_2_f1_vs_theta")


def plot_training() -> list[Path]:
    figure, axes = plt.subplots(4, 2, figsize=(12, 12.5), sharex=True)
    for row_index, (name, label, _) in enumerate(MODELS):
        rows = read_csv(TRAIN / name / f"training_log_{name}.csv")
        result = read_json(TRAIN / name / "training_result.json")
        epochs = [int(row["epoch"]) + 1 for row in rows]
        for col_index, (train_key, val_key, ylabel) in enumerate((
            ("loss", "val_loss", "Binary cross-entropy"),
            ("accuracy", "val_accuracy", "Accuracy"),
        )):
            ax = axes[row_index, col_index]
            ax.plot(epochs, [float(row[train_key]) for row in rows],
                    marker="o", ms=3, lw=1.6, color="#25577a", label="Training")
            ax.plot(epochs, [float(row[val_key]) for row in rows],
                    marker="s", ms=3, lw=1.6, color="#b34c53", label="Validation")
            ax.axvline(int(result["best_epoch"]), ls="--", lw=1.2,
                       color="#58676f", label="Best val. loss")
            ax.set_ylabel(ylabel)
            ax.set_xlim(0.7, 10.3)
            ax.set_xticks(range(1, 11))
            if col_index == 0:
                ax.set_title(f"{label}  |  best epoch {result['best_epoch']}",
                             fontsize=11, loc="left")
            else:
                ax.set_title("Accuracy", fontsize=10, loc="left")
            if row_index == 3:
                ax.set_xlabel("Epoch")
    axes[0, 0].legend(loc="upper right", fontsize=8, frameon=False)
    figure.suptitle("Grouped-scene retraining - four factorial configurations",
                    fontsize=16, weight="bold", y=0.995)
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    outputs = save_figure(figure, "figure_3_training_four_models")
    for name, _, _ in MODELS:
        source = TRAIN / name / "training_curves.png"
        target = OUT / f"training_{name}.png"
        shutil.copy2(source, target)
        outputs.append(target)
    return outputs


def plot_roc(factorial: dict) -> list[Path]:
    figure, ax = plt.subplots(figsize=(7.4, 6.5))
    for name, label, color in MODELS:
        rows = read_csv(EVAL / "final_test_locked" / name / "roc_curve.csv")
        fpr = np.asarray([float(row["false_positive_rate"]) for row in rows])
        tpr = np.asarray([float(row["true_positive_rate"]) for row in rows])
        area = factorial["cells"][name]["roc_auc"]
        assert abs(float(np.trapz(tpr, fpr)) - area) < 0.002
        ax.step(fpr, tpr, where="post", color=color, lw=2.0,
                label=f"{label}  |  AUC {area:.3f}")
    ax.plot([0, 1], [0, 1], ls="--", lw=1.1, color="#8a949a")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    ax.set_xlabel("False-positive rate")
    ax.set_ylabel("True-positive rate")
    ax.set_title("ROC curves - locked grouped test (n=597)", weight="bold", fontsize=14)
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    figure.tight_layout()
    return save_figure(figure, "figure_4_roc_overlay")


def plot_factorial(factorial: dict) -> list[Path]:
    figure, axes = plt.subplots(1, 3, figsize=(13.5, 4.7))
    panels = (("accuracy", "Accuracy", (84, 90)),
              ("f1", "F1", (84, 90)),
              ("recall", "Violence recall", (82, 96)))
    for ax, (metric, title, limits) in zip(axes, panels):
        for arch, color, marker in (("lstm", "#25577a", "o"),
                                    ("gru", "#b34c53", "s")):
            keys = (f"{arch}_original_grouped", f"{arch}_augmented_grouped")
            values = [100 * factorial["cells"][key][metric] for key in keys]
            ax.plot([0, 1], values, marker=marker, ms=7, lw=2.2,
                    color=color, label=arch.upper())
        ax.set_xticks([0, 1], ["Original", "Enriched"])
        ax.set_xlim(-0.16, 1.16)
        ax.set_ylim(*limits)
        ax.set_ylabel("Percent")
        ax.set_title(title, fontsize=11, weight="bold")
    axes[0].legend(loc="lower left", frameon=False)
    interaction = factorial["primary_interaction_test"]
    figure.suptitle("Architecture x enrichment - same locked test set (n=597)",
                    fontsize=15, weight="bold", y=1.04)
    figure.text(0.5, -0.035,
                "Primary accuracy interaction: "
                f"{100 * interaction['estimate']:+.2f} pp, "
                f"95% CI [{100 * interaction['ci95_lower']:+.2f}, "
                f"{100 * interaction['ci95_upper']:+.2f}], "
                f"paired sign-flip p={interaction['exact_sign_flip_two_sided_p_value']:.3f}",
                ha="center", fontsize=9, color="#35444c")
    figure.tight_layout()
    return save_figure(figure, "figure_5_factorial_2x2")


def plot_interaction_intervals(factorial: dict) -> list[Path]:
    metrics = (("accuracy", "Accuracy (primary)"), ("f1", "F1"),
               ("recall", "Recall"), ("specificity", "Specificity"))
    intervals = factorial["paired_bootstrap"]["effect_intervals"]
    values = [intervals[metric]["interaction"] for metric, _ in metrics]
    figure, ax = plt.subplots(figsize=(8.4, 5.1))
    for index, item in enumerate(values):
        estimate = 100 * item["estimate"]
        lower = 100 * item["ci95_lower"]
        upper = 100 * item["ci95_upper"]
        ax.errorbar(index, estimate, yerr=[[estimate - lower], [upper - estimate]],
                    fmt="o", capsize=5, ms=7, lw=1.9,
                    color="#25577a" if index == 0 else "#6d7780")
    ax.axhline(0, color="#b34c53", ls="--", lw=1.2)
    ax.set_xticks(range(len(metrics)), [label for _, label in metrics])
    ax.set_ylabel("Difference-in-differences (percentage points)")
    ax.set_title("Factorial interaction - paired bootstrap 95% CI",
                 fontsize=14, weight="bold")
    ax.set_xlim(-0.4, 3.4)
    figure.tight_layout()
    return save_figure(figure, "figure_6_interaction_ci95")


def write_readme(lock: dict) -> Path:
    lines = [
        "# Article figures - grouped-scene revision",
        "",
        "These exports use the grouped-scene protocol. The four theta/K choices were "
        "selected on 597 validation videos and locked before the 597-video test was read. "
        "No figure in this directory uses the older, ungrouped 599-video test curves.",
        "",
        "| File | Article caption / interpretation |",
        "|---|---|",
        "| `figure_1_architecture` | Seven operational SIRCH modules; the spatial "
        "EfficientNet-B0 and temporal LSTM stages share module 3. The dashboard's "
        "theta/K settings feed the decision module. |",
        "| `figure_2_f1_vs_theta` | Joint theta/K F1 sweep on grouped validation "
        "only (n=597); stars mark locked choices. |",
        "| `figure_3_training_four_models` | Training and validation loss/accuracy for "
        "all four grouped-scene retrainings; dashed lines mark best validation-loss epochs. |",
        "| `figure_4_roc_overlay` | Four ROC curves on the same, once-evaluated "
        "grouped test (n=597). AUC is threshold-independent. |",
        "| `figure_5_factorial_2x2` | Accuracy, F1 and violence recall for the "
        "LSTM/GRU x original/enriched design on the grouped test. |",
        "| `figure_6_interaction_ci95` | Paired-bootstrap 95% intervals for the "
        "difference-in-differences; accuracy is the prespecified primary scale, "
        "other metrics are descriptive. |",
        "",
        "Each numbered figure has both 300-dpi PNG and editable SVG versions. "
        "Four `training_*.png` files retain the original per-model two-panel plots.",
        "",
        "## Locked configurations",
        "",
        "| Model | theta | K | Best epoch |",
        "|---|---:|---:|---:|",
    ]
    for name, label, _ in MODELS:
        chosen = lock["selected_configurations"][name]
        best = lock["models"][name]["best_epoch"]
        lines.append(f"| {label} | {chosen['theta']:.2f} | {chosen['k']} | {best} |")
    lines += [
        "",
        "Source data: `revision/exp13_grouped_retraining/outputs/` and "
        "`revision/exp14_grouped_factorial_evaluation/outputs/`. "
        "`manifest.json` records SHA-256 provenance for every input and export.",
        "",
        "Figure 1 follows the current `main.py` flow: acquisition, preprocessing, "
        "inference, decision, alerts, incident log, dashboard. It corrects the "
        "reviewer's six-versus-seven-step mismatch. The deployed model in `main.py` "
        "is LSTM; the GRU is an experimental alternative in the factorial study.",
        "",
    ]
    path = OUT / "README.md"
    path.write_bytes("\n".join(lines).encode("utf-8"))
    return path


def main() -> None:
    lock, factorial, sources = verify_inputs()
    OUT.mkdir(parents=True, exist_ok=True)
    style()
    outputs = []
    outputs += plot_architecture()
    outputs += plot_thresholds(lock)
    outputs += plot_training()
    outputs += plot_roc(factorial)
    outputs += plot_factorial(factorial)
    outputs += plot_interaction_intervals(factorial)
    outputs.append(write_readme(lock))

    manifest = {
        "protocol": "grouped_scene_locked",
        "validation_videos": 597,
        "test_videos": 597,
        "generator": relative(Path(__file__)),
        "inputs": [{"path": relative(path), "sha256": digest(path)} for path in sources],
        "outputs": [{"path": relative(path), "sha256": digest(path), "bytes": path.stat().st_size}
                    for path in outputs],
    }
    manifest_path = OUT / "manifest.json"
    manifest_path.write_bytes((json.dumps(manifest, indent=2) + "\n").encode("utf-8"))
    print(f"Exported {len(outputs)} files to {OUT}")


if __name__ == "__main__":
    main()
