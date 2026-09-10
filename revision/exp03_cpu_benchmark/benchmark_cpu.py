from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import matplotlib
import numpy as np
import psutil
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input

matplotlib.use("Agg")
import matplotlib.pyplot as plt


N_FRAMES = 20
IMG_SIZE = 224
SEED = 42
VIDEOS_PER_CLASS = 50
WARMUP_SEQUENCES = 10
REPEATS = 3


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    exp01_outputs = project_root / "revision" / "exp01_manifests_sha256" / "outputs"
    parser = argparse.ArgumentParser(description="Reproducible SIRCH CPU benchmark.")
    parser.add_argument(
        "--datasets-root",
        type=Path,
        default=Path(os.environ.get("SIRCH_DATASETS_DIR", r"C:\SIRCH_ENV\datasets")),
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path(r"C:\SIRCH_ENV\models\sirch_model.h5"),
    )
    parser.add_argument(
        "--validation-manifest",
        type=Path,
        default=exp01_outputs / "validation_manifest.csv",
    )
    parser.add_argument(
        "--manifest-seal", type=Path, default=exp01_outputs / "manifest_seal.json"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path(__file__).resolve().parent / "outputs"
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(
    path: Path, seal_path: Path
) -> tuple[list[dict[str, str]], str, dict[str, object]]:
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    if seal.get("status") != "sealed_clean":
        raise RuntimeError(f"Le sceau n'est pas propre: {seal.get('status')}")
    if int(seal.get("cross_split_sha256_duplicate_groups", -1)) != 0:
        raise RuntimeError("Le sceau signale encore des doublons inter-splits.")
    expected_hash = seal["manifest_sha256"]["validation"]
    actual_hash = sha256_file(path)
    if actual_hash != expected_hash:
        raise RuntimeError("Le manifeste de validation ne correspond pas au sceau.")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    expected_count = int(seal["counts"]["validation"]["total"])
    if len(rows) != expected_count or any(row["split"] != "validation" for row in rows):
        raise RuntimeError(
            f"Le benchmark attend les {expected_count} lignes de validation scellees."
        )
    return rows, actual_hash, seal


def resolve_video_path(row: dict[str, str], datasets_root: Path) -> Path:
    if row["dataset"] == "RLVS":
        root = datasets_root / "RLVS" / "Real Life Violence Dataset"
    elif row["dataset"] == "RWF-2000":
        root = datasets_root / "RWF-2000"
    else:
        raise RuntimeError(f"Dataset inattendu: {row['dataset']}")
    return root / Path(row["relative_path"])


def select_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    generator = random.Random(SEED)
    selected: list[dict[str, str]] = []
    for label in (0, 1):
        candidates = [row for row in rows if int(row["label"]) == label]
        selected.extend(generator.sample(candidates, VIDEOS_PER_CLASS))
    generator.shuffle(selected)
    return selected


def load_sequence(path: Path) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Video illisible: {path}")
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, max(total - 1, 0), N_FRAMES).astype(int)
    frames: list[np.ndarray] = []
    last_frame: np.ndarray | None = None
    for frame_index in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = capture.read()
        if not ok:
            if last_frame is None:
                frame = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
            else:
                frame = last_frame.copy()
        last_frame = frame
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE)).astype(np.float32)
        frames.append(preprocess_input(resized))
    capture.release()
    return np.expand_dims(np.stack(frames, axis=0), axis=0)


def load_inference(model_path: Path):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from core.inference import ViolenceInference

    inference = ViolenceInference(str(model_path))
    start = time.perf_counter()
    inference.load()
    load_seconds = time.perf_counter() - start
    return inference, load_seconds


def percentile(values: list[float], value: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), value))


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "n": len(values),
        "mean_ms_per_sequence": statistics.fmean(values),
        "std_ms_per_sequence": statistics.pstdev(values),
        "p50_ms_per_sequence": percentile(values, 50),
        "p95_ms_per_sequence": percentile(values, 95),
        "p99_ms_per_sequence": percentile(values, 99),
        "min_ms_per_sequence": min(values),
        "max_ms_per_sequence": max(values),
        "mean_ms_per_frame": statistics.fmean(values) / N_FRAMES,
        "p50_ms_per_frame": percentile(values, 50) / N_FRAMES,
        "p95_ms_per_frame": percentile(values, 95) / N_FRAMES,
        "p99_ms_per_frame": percentile(values, 99) / N_FRAMES,
    }


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def make_figure(summary_rows: list[dict[str, object]], output_path: Path) -> None:
    pooled = [row for row in summary_rows if row["scope"] == "pooled"]
    labels = ["P50", "P95", "P99"]
    x = np.arange(len(labels))
    width = 0.34
    pure = next(row for row in pooled if row["protocol"] == "pure_inference")
    end_to_end = next(row for row in pooled if row["protocol"] == "end_to_end")
    pure_values = [pure[f"{label.lower()}_ms_per_frame"] for label in labels]
    end_values = [end_to_end[f"{label.lower()}_ms_per_frame"] for label in labels]
    figure, axis = plt.subplots(figsize=(8, 4.8), dpi=160)
    axis.bar(x - width / 2, pure_values, width, label="Inference pure", color="#234574")
    axis.bar(x + width / 2, end_values, width, label="Bout en bout", color="#1f8f55")
    axis.set_xticks(x, labels)
    axis.set_ylabel("Latence normalisee (ms/frame)")
    axis.set_title("SIRCH original - latence CPU, 300 observations")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows, manifest_hash, manifest_seal = load_manifest(
        args.validation_manifest, args.manifest_seal
    )
    selected = select_rows(rows)
    paths = [resolve_video_path(row, args.datasets_root) for row in selected]
    for row, path in zip(selected, paths):
        if not path.exists():
            raise FileNotFoundError(path)
        if sha256_file(path) != row["sha256"]:
            raise RuntimeError(f"Hash video different du manifeste: {path}")

    inference, model_load_seconds = load_inference(args.model_path)
    model_hash = sha256_file(args.model_path)
    print(
        f"CPU benchmark: {len(selected)} videos x {REPEATS} repetitions | "
        f"warm-up={WARMUP_SEQUENCES}",
        flush=True,
    )
    for row, path in list(zip(selected, paths))[:WARMUP_SEQUENCES]:
        sequence = load_sequence(path)
        inference.predict(sequence)
    print("Warm-up termine.", flush=True)

    raw_rows: list[dict[str, object]] = []
    for repeat in range(1, REPEATS + 1):
        order = list(range(len(selected)))
        random.Random(SEED + repeat).shuffle(order)
        for position, selected_index in enumerate(order, start=1):
            row = selected[selected_index]
            path = paths[selected_index]
            total_start = time.perf_counter()
            sequence = load_sequence(path)
            inference_start = time.perf_counter()
            score = inference.predict(sequence)
            inference_ms = (time.perf_counter() - inference_start) * 1000
            end_to_end_ms = (time.perf_counter() - total_start) * 1000
            raw_rows.append(
                {
                    "repeat": repeat,
                    "position": position,
                    "dataset": row["dataset"],
                    "relative_path": row["relative_path"],
                    "video_sha256": row["sha256"],
                    "label": row["label"],
                    "score": f"{score:.10f}",
                    "pure_inference_ms_per_sequence": f"{inference_ms:.6f}",
                    "pure_inference_ms_per_frame": f"{inference_ms / N_FRAMES:.6f}",
                    "end_to_end_ms_per_sequence": f"{end_to_end_ms:.6f}",
                    "end_to_end_ms_per_frame": f"{end_to_end_ms / N_FRAMES:.6f}",
                }
            )
            if position % 20 == 0:
                print(f"Repetition {repeat}/{REPEATS}: {position}/{len(order)}", flush=True)

    raw_fields = list(raw_rows[0].keys())
    write_csv(args.output_dir / "latency_raw.csv", raw_fields, raw_rows)
    summary_rows: list[dict[str, object]] = []
    for protocol, field in (
        ("pure_inference", "pure_inference_ms_per_sequence"),
        ("end_to_end", "end_to_end_ms_per_sequence"),
    ):
        for repeat in range(1, REPEATS + 1):
            values = [
                float(row[field]) for row in raw_rows if int(row["repeat"]) == repeat
            ]
            summary_rows.append(
                {"protocol": protocol, "scope": f"repeat_{repeat}", **summarize(values)}
            )
        pooled_values = [float(row[field]) for row in raw_rows]
        summary_rows.append(
            {"protocol": protocol, "scope": "pooled", **summarize(pooled_values)}
        )
    summary_fields = list(summary_rows[0].keys())
    write_csv(args.output_dir / "latency_summary.csv", summary_fields, summary_rows)

    memory = psutil.virtual_memory()
    environment = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "cpu_only_requested": True,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "tensorflow_visible_gpus": [
            device.name for device in tf.config.get_visible_devices("GPU")
        ],
        "platform": platform.platform(),
        "python": platform.python_version(),
        "tensorflow": tf.__version__,
        "opencv": cv2.__version__,
        "processor": platform.processor(),
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores": psutil.cpu_count(logical=True),
        "ram_total_gb": round(memory.total / (1024**3), 3),
        "model_path": str(args.model_path),
        "model_sha256": model_hash,
        "model_load_seconds": model_load_seconds,
        "validation_manifest_sha256": manifest_hash,
        "manifest_status": manifest_seal["status"],
        "cross_split_sha256_duplicate_groups": manifest_seal[
            "cross_split_sha256_duplicate_groups"
        ],
        "test_manifest_read": False,
        "n_frames": N_FRAMES,
        "batch_size": 1,
        "warmup_sequences": WARMUP_SEQUENCES,
        "videos": len(selected),
        "videos_per_class": VIDEOS_PER_CLASS,
        "repeats": REPEATS,
        "observations": len(raw_rows),
        "seed": SEED,
    }
    (args.output_dir / "benchmark_environment.json").write_text(
        json.dumps(environment, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    make_figure(summary_rows, args.output_dir / "latency_percentiles.png")

    pure = next(
        row
        for row in summary_rows
        if row["protocol"] == "pure_inference" and row["scope"] == "pooled"
    )
    end_to_end = next(
        row
        for row in summary_rows
        if row["protocol"] == "end_to_end" and row["scope"] == "pooled"
    )
    pure_repeats = [
        row
        for row in summary_rows
        if row["protocol"] == "pure_inference" and row["scope"].startswith("repeat_")
    ]
    end_to_end_repeats = [
        row
        for row in summary_rows
        if row["protocol"] == "end_to_end" and row["scope"].startswith("repeat_")
    ]
    old_pure = 28.87
    old_end_to_end = 47.33
    pure_repeat_means = ", ".join(
        f"{float(row['mean_ms_per_frame']):.2f}" for row in pure_repeats
    )
    end_to_end_repeat_means = ", ".join(
        f"{float(row['mean_ms_per_frame']):.2f}" for row in end_to_end_repeats
    )
    summary = [
        "# Benchmark CPU SIRCH original",
        "",
        f"Observations : {len(raw_rows)} apres {WARMUP_SEQUENCES} warm-ups.",
        "",
        "| Protocole | Moyenne ms/frame | P50 | P95 | P99 |",
        "|---|---:|---:|---:|---:|",
        f"| Inference pure | {pure['mean_ms_per_frame']:.2f} | {pure['p50_ms_per_frame']:.2f} | {pure['p95_ms_per_frame']:.2f} | {pure['p99_ms_per_frame']:.2f} |",
        f"| Bout en bout | {end_to_end['mean_ms_per_frame']:.2f} | {end_to_end['p50_ms_per_frame']:.2f} | {end_to_end['p95_ms_per_frame']:.2f} | {end_to_end['p99_ms_per_frame']:.2f} |",
        "",
        "Le chargement du modele et les controles SHA-256 sont exclus des latences.",
        "Le jeu de test principal n'a pas ete lu.",
        "",
        "## Ecart avec les chiffres publies auparavant",
        "",
        "| Mesure | Ancien protocole | Nouveau protocole | Ecart |",
        "|---|---:|---:|---:|",
        f"| Inference pure | {old_pure:.2f} | {pure['mean_ms_per_frame']:.2f} | +{pure['mean_ms_per_frame'] - old_pure:.2f} ms/frame ({(pure['mean_ms_per_frame'] / old_pure - 1) * 100:.1f} %) |",
        f"| Bout en bout | {old_end_to_end:.2f} | {end_to_end['mean_ms_per_frame']:.2f} | +{end_to_end['mean_ms_per_frame'] - old_end_to_end:.2f} ms/frame ({(end_to_end['mean_ms_per_frame'] / old_end_to_end - 1) * 100:.1f} %) |",
        "",
        "L'ancien benchmark (`phase3_results/model_complexity.txt`) utilisait seulement 5 videos RLVS non violentes, soit 5 mesures de sequence et 100 frames au total, en un seul passage. Le nouveau benchmark utilise 100 videos de validation equilibrees (50 violentes et 50 non violentes, RLVS et RWF-2000), trois passages et 300 mesures.",
        "",
        "Le warm-up n'explique pas l'ecart : il etait deja exclu de l'ancienne mesure et les 10 warm-ups du nouveau protocole sont egalement exclus. Le nombre de 300 mesures ne ralentit pas mathematiquement une prediction ; il expose mieux la variabilite et la charge soutenue du CPU.",
        "",
        f"La variation entre repetitions le confirme : inference pure {pure_repeat_means} ms/frame ; bout en bout {end_to_end_repeat_means} ms/frame. Les passages deviennent ici legerement plus rapides, ce qui est compatible avec la stabilisation des caches TensorFlow, systeme et disque. Le pipeline complet ajoute aussi la variabilite des codecs, de la lecture disque et du redimensionnement sur un corpus plus heterogene.",
        "",
        "Les deux chiffres ne sont donc pas directement comparables. L'ancien resultat est une petite mesure ponctuelle favorable ; le nouveau resultat, avec distribution P50/P95/P99 et mesures brutes, est la reference reproductible a retenir.",
        "",
        "## Debit reel soutenable et portee du terme temps reel",
        "",
        f"La moyenne bout en bout de {end_to_end['mean_ms_per_frame']:.2f} ms/frame correspond a environ {1000 / end_to_end['mean_ms_per_frame']:.2f} frames/s, alors qu'un flux de 25 ou 30 fps impose respectivement 40,00 ou 33,33 ms/frame. Le pipeline CPU mesure ne peut donc pas traiter exhaustivement un flux continu a 25-30 fps sans perte ni retard.",
        "",
        f"La latence normalisee ne doit pas etre confondue avec la duree d'un appel du modele : une prediction sur 20 frames prend en moyenne {pure['mean_ms_per_sequence']:.2f} ms en inference pure. Avec `--infer-every 5`, un flux de 25 fps demanderait 5 predictions/s, soit au plus 200 ms par appel, et un flux de 30 fps en demanderait 6, soit 166,67 ms par appel. La mesure d'environ {pure['mean_ms_per_sequence'] / 1000:.2f} s par appel reste trop lente, meme avec cet espacement.",
        "",
        "Dans `main.py`, acquisition, pretraitement, prediction et affichage sont synchrones dans un seul thread. Pendant `model.predict`, aucune frame n'est lue. L'application ne possede ni file d'attente explicite, ni thread de capture, ni mecanisme explicite de saut de frames. Elle demande seulement `CAP_PROP_BUFFERSIZE=1` a OpenCV sans verifier que le pilote l'accepte. Si ce tampon est respecte, les images arrivees pendant l'inference sont abandonnees ou remplacees par la plus recente ; s'il est ignore, le tampon du pilote peut accumuler des images et produire du retard. `--infer-every 5` espace les predictions, mais ne resout pas le blocage pendant une prediction.",
        "",
        "La revendication exacte est donc : demonstration interactive ou quasi temps reel avec echantillonnage/perte possible de frames sur ce CPU, et non traitement exhaustif garanti a 25-30 fps. Une revendication de temps reel plein debit exige une mesure directe de la boucle camera et une optimisation ou une architecture asynchrone validant au plus 40 ms/frame a 25 fps (33,33 ms/frame a 30 fps).",
    ]
    (args.output_dir / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(json.dumps({"environment": environment, "summary": summary_rows}, indent=2), flush=True)


if __name__ == "__main__":
    main()
