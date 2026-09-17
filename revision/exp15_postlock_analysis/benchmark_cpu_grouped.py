from __future__ import annotations

import csv
import gc
import hashlib
import json
import os
import platform
import random
import statistics
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import keras
import numpy as np
import psutil
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "outputs"
VALIDATION = ROOT / "revision" / "exp11_grouped_scene_splits" / "outputs" / "validation_manifest_grouped.csv"
TEST_METRICS = ROOT / "revision" / "exp14_grouped_factorial_evaluation" / "outputs" / "final_test_locked"
DATASETS = Path(os.environ.get("SIRCH_DATASETS_DIR", r"C:\SIRCH_ENV\datasets"))
MODELS = (
    "lstm_original_grouped", "lstm_augmented_grouped",
    "gru_original_grouped", "gru_augmented_grouped",
)
VALIDATION_SHA256 = "d4b3f4a41c4d118a99abd70e88943f7886970695353369259ec3ea506e3964e5"
N_FRAMES = 20
IMG_SIZE = 224
BATCH_SIZE = 1
WARMUP = 5
VIDEOS_PER_CLASS = 5
REPEATS = 2
SEED = 42


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def video_path(row: dict[str, str]) -> Path:
    if row["dataset"] == "RLVS":
        return DATASETS / "RLVS" / "Real Life Violence Dataset" / row["relative_path"]
    if row["dataset"] == "RWF-2000":
        return DATASETS / "RWF-2000" / row["relative_path"]
    raise RuntimeError(f"Unknown dataset: {row['dataset']}")


def load_sequence(path: Path) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Unreadable video: {path}")
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, max(total - 1, 0), N_FRAMES).astype(int)
    frames = []
    last = None
    try:
        for index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = capture.read()
            if not ok:
                frame = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8) if last is None else last.copy()
            last = frame
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(preprocess_input(cv2.resize(rgb, (IMG_SIZE, IMG_SIZE)).astype(np.float32)))
    finally:
        capture.release()
    return np.expand_dims(np.stack(frames), axis=0)


class ResourceSampler:
    def __init__(self) -> None:
        self.process = psutil.Process()
        self.rows: list[dict[str, float]] = []
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        self.process.cpu_percent(interval=None)
        psutil.cpu_percent(interval=None)
        previous_thread_cpu = {
            thread.id: thread.user_time + thread.system_time
            for thread in self.process.threads()
        }
        while not self.stop.wait(0.5):
            try:
                current_thread_cpu = {
                    thread.id: thread.user_time + thread.system_time
                    for thread in self.process.threads()
                }
                cpu_active_threads = sum(
                    cpu_time > previous_thread_cpu.get(thread_id, 0)
                    for thread_id, cpu_time in current_thread_cpu.items()
                )
                previous_thread_cpu = current_thread_cpu
                self.rows.append({
                    "process_cpu_pct": self.process.cpu_percent(interval=None),
                    "system_cpu_pct": psutil.cpu_percent(interval=None),
                    "rss_mb": self.process.memory_info().rss / (1024 * 1024),
                    "process_threads": self.process.num_threads(),
                    "cpu_active_threads_in_sample": cpu_active_threads,
                })
            except psutil.Error:
                break

    def __enter__(self) -> ResourceSampler:
        self.thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop.set()
        self.thread.join()


def processor_name() -> str:
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
            return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
    except (ImportError, OSError):
        return platform.processor()


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean_ms": float(array.mean()),
        "median_ms": float(np.percentile(array, 50)),
        "p95_ms": float(np.percentile(array, 95)),
        "std_ms": float(array.std()),
    }


def main() -> None:
    if sha256_file(VALIDATION) != VALIDATION_SHA256:
        raise RuntimeError("Grouped validation manifest seal changed")
    if tf.config.list_physical_devices("GPU"):
        raise RuntimeError("GPU visible: CPU-only benchmark required")
    tf.config.threading.set_intra_op_parallelism_threads(8)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    rows = read_csv(VALIDATION)
    if len(rows) != 597:
        raise RuntimeError("Unexpected validation size")
    rng = random.Random(SEED)
    selected = []
    for label in (0, 1):
        selected.extend(rng.sample([row for row in rows if int(row["label"]) == label], VIDEOS_PER_CLASS))
    rng.shuffle(selected)
    for row in selected:
        path = video_path(row)
        if not path.exists() or sha256_file(path) != row["sha256"]:
            raise RuntimeError(f"Sample file missing or changed: {path}")
    OUT.mkdir(parents=True, exist_ok=True)
    hardware = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "cpu_model": processor_name(),
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_processors": psutil.cpu_count(logical=True),
        "ram_total_gb": psutil.virtual_memory().total / (1024**3),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "tensorflow": tf.__version__,
        "keras": keras.__version__,
        "opencv": cv2.__version__,
        "numpy": np.__version__,
        "psutil": psutil.__version__,
        "resource_sampling_interval_seconds": 0.5,
        "active_thread_definition": "OS threads with increased CPU time since previous 0.5 s sample; not simultaneous execution",
        "tensorflow_intra_op_threads_configured": tf.config.threading.get_intra_op_parallelism_threads(),
        "tensorflow_inter_op_threads_configured": tf.config.threading.get_inter_op_parallelism_threads(),
        "opencv_threads_configured": cv2.getNumThreads(),
        "batch_size_clips": BATCH_SIZE,
        "frames_per_clip": N_FRAMES,
        "warmup_clips_per_model": WARMUP,
        "videos": len(selected),
        "videos_per_class": VIDEOS_PER_CLASS,
        "repeats": REPEATS,
        "observations_per_model": len(selected) * REPEATS,
        "manifest": str(VALIDATION),
        "manifest_sha256": VALIDATION_SHA256,
        "test_manifest_read": False,
        "sample": [{"dataset": r["dataset"], "relative_path": r["relative_path"], "sha256": r["sha256"]} for r in selected],
    }
    with (OUT / "cpu_benchmark_environment.json").open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(hardware, handle, indent=2)
        handle.write("\n")
    raw = []
    summaries = []
    for model_key in MODELS:
        metric = json.loads((TEST_METRICS / model_key / "final_test_metrics.json").read_text(encoding="utf-8"))
        path = Path(metric["model"]["model_path"])
        if sha256_file(path) != metric["model"]["model_sha256"]:
            raise RuntimeError(f"Locked model hash changed: {model_key}")
        print(f"Loading {model_key}", flush=True)
        load_start = time.perf_counter()
        model = tf.keras.models.load_model(path, compile=False)
        load_seconds = time.perf_counter() - load_start
        if tuple(model.input_shape) != (None, 20, 224, 224, 3):
            raise RuntimeError(f"Unexpected model input: {model_key} {model.input_shape}")
        for row in selected[:WARMUP]:
            model.predict(load_sequence(video_path(row)), batch_size=BATCH_SIZE, verbose=0)
        process = psutil.Process()
        rss_before_mb = process.memory_info().rss / (1024 * 1024)
        model_raw = []
        with ResourceSampler() as sampler:
            for repeat in range(1, REPEATS + 1):
                order = list(range(len(selected)))
                random.Random(SEED + repeat).shuffle(order)
                for position, index in enumerate(order, start=1):
                    row = selected[index]
                    start = time.perf_counter()
                    sequence = load_sequence(video_path(row))
                    infer_start = time.perf_counter()
                    score = float(model.predict(sequence, batch_size=BATCH_SIZE, verbose=0).ravel()[0])
                    infer_ms = (time.perf_counter() - infer_start) * 1000
                    total_ms = (time.perf_counter() - start) * 1000
                    model_raw.append({
                        "model": model_key, "repeat": repeat, "position": position,
                        "dataset": row["dataset"], "relative_path": row["relative_path"],
                        "video_sha256": row["sha256"], "score": f"{score:.9f}",
                        "inference_ms_per_clip": f"{infer_ms:.6f}",
                        "decode_preprocess_predict_ms_per_clip": f"{total_ms:.6f}",
                    })
                print(f"{model_key}: repeat {repeat}/{REPEATS} complete", flush=True)
        samples = sampler.rows
        if len(samples) < 3:
            raise RuntimeError(f"Too few resource observations for {model_key}")
        infer = summarize([float(r["inference_ms_per_clip"]) for r in model_raw])
        total = summarize([float(r["decode_preprocess_predict_ms_per_clip"]) for r in model_raw])
        summaries.append({
            "model": model_key, "model_sha256": metric["model"]["model_sha256"],
            "n": len(model_raw), "load_seconds_excluded": load_seconds,
            "inference_mean_ms_per_clip": infer["mean_ms"],
            "inference_p50_ms_per_clip": infer["median_ms"],
            "inference_p95_ms_per_clip": infer["p95_ms"],
            "inference_std_ms_per_clip": infer["std_ms"],
            "inference_clips_per_second": 1000 / infer["mean_ms"],
            "inference_normalized_fps": N_FRAMES * 1000 / infer["mean_ms"],
            "end_to_end_mean_ms_per_clip": total["mean_ms"],
            "end_to_end_p50_ms_per_clip": total["median_ms"],
            "end_to_end_p95_ms_per_clip": total["p95_ms"],
            "end_to_end_std_ms_per_clip": total["std_ms"],
            "end_to_end_clips_per_second": 1000 / total["mean_ms"],
            "end_to_end_normalized_fps": N_FRAMES * 1000 / total["mean_ms"],
            "resource_samples": len(samples),
            "process_cpu_pct_mean": statistics.fmean(s["process_cpu_pct"] for s in samples),
            "process_cpu_pct_peak": max(s["process_cpu_pct"] for s in samples),
            "system_cpu_pct_mean": statistics.fmean(s["system_cpu_pct"] for s in samples),
            "system_cpu_pct_peak": max(s["system_cpu_pct"] for s in samples),
            "rss_mb_before_measurement": rss_before_mb,
            "rss_mb_mean": statistics.fmean(s["rss_mb"] for s in samples),
            "rss_mb_peak": max(s["rss_mb"] for s in samples),
            "process_threads_mean": statistics.fmean(s["process_threads"] for s in samples),
            "process_threads_peak": max(s["process_threads"] for s in samples),
            "cpu_active_threads_mean": statistics.fmean(s["cpu_active_threads_in_sample"] for s in samples),
            "cpu_active_threads_peak": max(s["cpu_active_threads_in_sample"] for s in samples),
        })
        raw.extend(model_raw)
        del model
        tf.keras.backend.clear_session()
        gc.collect()
    write_csv(OUT / "cpu_benchmark_raw.csv", raw)
    write_csv(OUT / "cpu_benchmark_summary.csv", summaries)
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == "__main__":
    main()
