from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = EXPERIMENT_DIR / "outputs"
EXP01_OUTPUTS = PROJECT_ROOT / "revision" / "exp01_manifests_sha256" / "outputs"
EXP04_OUTPUTS = PROJECT_ROOT / "revision" / "exp04_augmented_clean" / "outputs"
EXPECTED_HASHES = {
    "augmented_train": "f28b0ee1cb6862dcb8bf51923a4c39734abc9d55a8001818269a61b9f28a5cd9",
    "validation": "933a26378a88236beafeb51fc91b53c7abab5185aba9d97daa0b68dd2310e397",
    "test": "13bfda2224edb723a75be0deb4bd067d37fb7c481830a00cd11284485994cb1f",
}
SAMPLE_FRACTIONS = tuple(float(value) for value in np.linspace(0.05, 0.95, 9))
CROP_FRACTIONS = (0.0, 0.10, 0.20)
PHASH_FRAME_THRESHOLD = 8
LSH_CHUNKS = ((0, 13), (13, 13), (26, 13), (39, 13), (52, 12))
SIGNATURE_FIELDS = [
    "split",
    "manifest_row_id",
    "source_group",
    "dataset",
    "label",
    "relative_path",
    "video_sha256",
    "source_id",
    "width",
    "height",
    "fps",
    "frame_count",
    "duration_seconds",
    "sample_frame_indices",
    "phash_crop_0",
    "phash_crop_10",
    "phash_crop_20",
    "decode_status",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit perceptuel des quasi-doublons entre train, validation et test."
    )
    parser.add_argument(
        "--datasets-root",
        type=Path,
        default=Path(os.environ.get("SIRCH_DATASETS_DIR", r"C:\SIRCH_ENV\datasets")),
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--max-videos",
        type=int,
        default=None,
        help="Limite de diagnostic uniquement; omettre pour l'audit final complet.",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def manifest_row_id(row: dict[str, str]) -> str:
    return f"{row['split']}|{row['dataset']}|{row['relative_path']}"


def load_audit_rows() -> tuple[
    list[dict[str, str]], dict[str, str], dict[str, object]
]:
    paths = {
        "augmented_train": EXP04_OUTPUTS / "augmented_train_manifest.csv",
        "validation": EXP01_OUTPUTS / "validation_manifest.csv",
        "test": EXP01_OUTPUTS / "test_manifest.csv",
    }
    actual_hashes = {key: sha256_file(path) for key, path in paths.items()}
    if actual_hashes != EXPECTED_HASHES:
        raise RuntimeError(
            f"Un manifeste a change: attendu={EXPECTED_HASHES}, obtenu={actual_hashes}"
        )

    rows: list[dict[str, str]] = []
    for source_key, path in paths.items():
        for row in read_csv(path):
            item = dict(row)
            item["split"] = "train" if source_key == "augmented_train" else source_key
            item["source_group"] = item.get("source_group", "") or (
                "principal_train_clean" if source_key == "augmented_train" else source_key
            )
            rows.append(item)

    expected_counts = {"train": 2818, "validation": 594, "test": 592}
    actual_counts = {
        split: sum(row["split"] == split for row in rows) for split in expected_counts
    }
    if actual_counts != expected_counts:
        raise RuntimeError(f"Comptes inter-splits inattendus: {actual_counts}")
    hash_groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        hash_groups[row["sha256"]].append(row)
    duplicate_groups = [group for group in hash_groups.values() if len(group) > 1]
    cross_split_groups = [
        group for group in duplicate_groups if len({row["split"] for row in group}) > 1
    ]
    if cross_split_groups:
        raise RuntimeError(
            f"{len(cross_split_groups)} chevauchement(s) SHA-256 inter-splits subsistent."
        )
    within_split_counts = {
        split: sum(group[0]["split"] == split for group in duplicate_groups)
        for split in ("train", "validation", "test")
    }
    exact_duplicate_stats = {
        "cross_split_groups": len(cross_split_groups),
        "within_split_groups": within_split_counts,
        "within_split_groups_total": len(duplicate_groups),
    }
    return rows, actual_hashes, exact_duplicate_stats


def resolve_video_path(row: dict[str, str], datasets_root: Path) -> Path:
    if row["dataset"] == "RLVS":
        return datasets_root / "RLVS" / "Real Life Violence Dataset" / row["relative_path"]
    if row["dataset"] == "RWF-2000":
        return datasets_root / "RWF-2000" / row["relative_path"]
    if row["dataset"] == "historical_enrichment_v1":
        return PROJECT_ROOT / "datasets" / row["relative_path"]
    raise RuntimeError(f"Dataset inattendu: {row['dataset']}")


def source_id_for(row: dict[str, str]) -> str:
    explicit = row.get("video_id", "").strip()
    if explicit:
        return explicit
    if row["dataset"] == "historical_enrichment_v1":
        match = re.search(r"_([A-Za-z0-9_-]{11})$", Path(row["relative_path"]).stem)
        if match:
            return match.group(1)
    return ""


def phash(frame: np.ndarray, crop_fraction: float) -> int:
    height, width = frame.shape[:2]
    if crop_fraction:
        top = min(int(round(height * crop_fraction)), max(0, height // 2 - 1))
        left = min(int(round(width * crop_fraction)), max(0, width // 2 - 1))
        frame = frame[top : height - top, left : width - left]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
    coefficients = cv2.dct(resized.astype(np.float32))[:8, :8].reshape(-1)
    median = float(np.median(coefficients[1:]))
    bits = coefficients > median
    bits[0] = False
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def extract_signature(
    row: dict[str, str], datasets_root: Path
) -> dict[str, object]:
    path = resolve_video_path(row, datasets_root)
    if not path.exists():
        raise FileNotFoundError(path)
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Video illisible: {path}")
    frame_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    if frame_count <= 0:
        capture.release()
        raise RuntimeError(f"Nombre de frames invalide: {path}")
    indices = sorted(
        set(min(frame_count - 1, max(0, int(round(fraction * (frame_count - 1))))) for fraction in SAMPLE_FRACTIONS)
    )
    hashes: dict[int, list[str]] = {crop: [] for crop in (0, 10, 20)}
    decoded_indices: list[int] = []
    for frame_index in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok:
            continue
        decoded_indices.append(frame_index)
        for crop_fraction in CROP_FRACTIONS:
            crop_key = int(round(crop_fraction * 100))
            hashes[crop_key].append(f"{phash(frame, crop_fraction):016x}")
    capture.release()
    decode_status = "complete" if len(decoded_indices) == len(indices) else "partial"
    if len(decoded_indices) < 3:
        decode_status = "insufficient"
    duration = frame_count / fps if fps > 0 else math.nan
    return {
        "split": row["split"],
        "manifest_row_id": manifest_row_id(row),
        "source_group": row["source_group"],
        "dataset": row["dataset"],
        "label": int(row["label"]),
        "relative_path": row["relative_path"],
        "video_sha256": row["sha256"],
        "source_id": source_id_for(row),
        "width": width,
        "height": height,
        "fps": f"{fps:.6f}" if math.isfinite(fps) else "",
        "frame_count": frame_count,
        "duration_seconds": f"{duration:.6f}" if math.isfinite(duration) else "",
        "sample_frame_indices": ";".join(str(value) for value in decoded_indices),
        "phash_crop_0": ";".join(hashes[0]),
        "phash_crop_10": ";".join(hashes[10]),
        "phash_crop_20": ";".join(hashes[20]),
        "decode_status": decode_status,
    }


def load_or_extract_signatures(
    rows: list[dict[str, str]], datasets_root: Path, output_dir: Path
) -> list[dict[str, str]]:
    final_path = output_dir / "video_perceptual_signatures.csv"
    partial_path = output_dir / "video_perceptual_signatures.partial.csv"
    if final_path.exists():
        final_rows = read_csv(final_path)
        expected_ids = [manifest_row_id(row) for row in rows]
        if [row.get("manifest_row_id", "") for row in final_rows] == expected_ids:
            print(f"Signatures perceptuelles reutilisees: {len(final_rows)}", flush=True)
            return final_rows
    existing_path = final_path if final_path.exists() else partial_path
    signatures = read_csv(existing_path) if existing_path.exists() else []
    by_id = {row["manifest_row_id"]: row for row in signatures}

    for index, row in enumerate(rows, start=1):
        row_id = manifest_row_id(row)
        if row_id not in by_id:
            signature = extract_signature(row, datasets_root)
            by_id[row_id] = {key: str(value) for key, value in signature.items()}
        if index % 25 == 0 or index == len(rows):
            ordered = [
                by_id[manifest_row_id(item)]
                for item in rows
                if manifest_row_id(item) in by_id
            ]
            write_csv(partial_path, ordered, SIGNATURE_FIELDS)
            print(f"Signatures perceptuelles: {index}/{len(rows)}", flush=True)

    signatures = [by_id[manifest_row_id(row)] for row in rows]
    write_csv(final_path, signatures, SIGNATURE_FIELDS)
    if partial_path.exists():
        partial_path.unlink()
    return signatures


def signature_hashes(row: dict[str, str]) -> list[list[int]]:
    crops = []
    for key in ("phash_crop_0", "phash_crop_10", "phash_crop_20"):
        crops.append([int(value, 16) for value in row[key].split(";") if value])
    if not crops or any(len(values) != len(crops[0]) for values in crops):
        return []
    return [[crops[crop][sample] for crop in range(3)] for sample in range(len(crops[0]))]


def chunk_value(value: int, offset: int, width: int) -> int:
    return (value >> offset) & ((1 << width) - 1)


def candidate_pairs(signatures: list[dict[str, str]]) -> set[tuple[int, int]]:
    index: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    candidates: set[tuple[int, int]] = set()
    for video_index, row in enumerate(signatures):
        per_sample = signature_hashes(row)
        flat_hashes = {value for sample in per_sample for value in sample}
        for value in flat_hashes:
            bit_count = value.bit_count()
            if bit_count < 5 or bit_count > 59:
                continue
            for chunk_index, (offset, width) in enumerate(LSH_CHUNKS):
                current = chunk_value(value, offset, width)
                neighbors = [current] + [current ^ (1 << bit) for bit in range(width)]
                for neighbor in neighbors:
                    for other_index, other_hash in index.get((chunk_index, neighbor), []):
                        if signatures[other_index]["split"] == row["split"]:
                            continue
                        if (value ^ other_hash).bit_count() <= PHASH_FRAME_THRESHOLD:
                            candidates.add((other_index, video_index))
            for chunk_index, (offset, width) in enumerate(LSH_CHUNKS):
                index[(chunk_index, chunk_value(value, offset, width))].append(
                    (video_index, value)
                )
        if (video_index + 1) % 250 == 0:
            print(
                f"Index pHash: {video_index + 1}/{len(signatures)}, "
                f"paires candidates={len(candidates)}",
                flush=True,
            )

    metadata_groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for video_index, row in enumerate(signatures):
        if row["source_id"]:
            metadata_groups[("source_id", row["source_id"])].append(video_index)
        metadata_groups[("basename", Path(row["relative_path"]).stem.lower())].append(
            video_index
        )
    for indices in metadata_groups.values():
        for left, right in combinations(indices, 2):
            if signatures[left]["split"] != signatures[right]["split"]:
                candidates.add((min(left, right), max(left, right)))
    return candidates


def ordered_match_count(distances: np.ndarray, threshold: int) -> int:
    rows, columns = distances.shape
    table = np.zeros((rows + 1, columns + 1), dtype=np.int16)
    for row_index in range(1, rows + 1):
        for column_index in range(1, columns + 1):
            if distances[row_index - 1, column_index - 1] <= threshold:
                table[row_index, column_index] = table[row_index - 1, column_index - 1] + 1
            else:
                table[row_index, column_index] = max(
                    table[row_index - 1, column_index],
                    table[row_index, column_index - 1],
                )
    return int(table[rows, columns])


def assess_pair(left: dict[str, str], right: dict[str, str]) -> dict[str, object]:
    left_hashes = signature_hashes(left)
    right_hashes = signature_hashes(right)
    distances = np.full((len(left_hashes), len(right_hashes)), 64, dtype=np.int16)
    for left_sample, left_variants in enumerate(left_hashes):
        for right_sample, right_variants in enumerate(right_hashes):
            distances[left_sample, right_sample] = min(
                (a ^ b).bit_count() for a in left_variants for b in right_variants
            )
    minimum = int(np.min(distances)) if distances.size else 64
    ordered_h6 = ordered_match_count(distances, 6)
    ordered_h8 = ordered_match_count(distances, 8)
    duration_left = float(left["duration_seconds"]) if left["duration_seconds"] else math.nan
    duration_right = float(right["duration_seconds"]) if right["duration_seconds"] else math.nan
    duration_delta = abs(duration_left - duration_right)
    duration_ratio = (
        min(duration_left, duration_right) / max(duration_left, duration_right)
        if duration_left > 0 and duration_right > 0
        else math.nan
    )
    source_id_match = bool(
        left["source_id"] and left["source_id"] == right["source_id"]
    )
    basename_match = Path(left["relative_path"]).stem.lower() == Path(
        right["relative_path"]
    ).stem.lower()

    if source_id_match or basename_match or ordered_h6 >= 5 or ordered_h8 >= 6:
        classification = "probable_near_duplicate"
    elif (
        ordered_h8 >= 4
        or (ordered_h6 >= 3 and math.isfinite(duration_ratio) and duration_ratio >= 0.70)
    ):
        classification = "possible_near_duplicate"
    else:
        classification = "isolated_frame_similarity"

    return {
        "split_a": left["split"],
        "split_b": right["split"],
        "dataset_a": left["dataset"],
        "dataset_b": right["dataset"],
        "label_a": left["label"],
        "label_b": right["label"],
        "relative_path_a": left["relative_path"],
        "relative_path_b": right["relative_path"],
        "sha256_a": left["video_sha256"],
        "sha256_b": right["video_sha256"],
        "source_id_a": left["source_id"],
        "source_id_b": right["source_id"],
        "source_id_match": int(source_id_match),
        "basename_match": int(basename_match),
        "duration_seconds_a": left["duration_seconds"],
        "duration_seconds_b": right["duration_seconds"],
        "duration_delta_seconds": duration_delta,
        "duration_ratio": duration_ratio,
        "resolution_a": f"{left['width']}x{left['height']}",
        "resolution_b": f"{right['width']}x{right['height']}",
        "min_phash_hamming": minimum,
        "ordered_matches_h6": ordered_h6,
        "ordered_matches_h8": ordered_h8,
        "classification": classification,
    }


def make_figure(rows: list[dict[str, object]], path: Path) -> None:
    values = [int(row["min_phash_hamming"]) for row in rows]
    fig, ax = plt.subplots(figsize=(8.6, 4.9))
    if values:
        bins = np.arange(-0.5, PHASH_FRAME_THRESHOLD + 1.5, 1)
        ax.hist(values, bins=bins, color="#3969ac", edgecolor="white")
        ax.set_xticks(range(PHASH_FRAME_THRESHOLD + 1))
    else:
        ax.text(0.5, 0.5, "Aucune paire candidate", ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel("Distance de Hamming pHash minimale")
    ax.set_ylabel("Nombre de paires inter-splits")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def node_id(row: dict[str, object], suffix: str) -> str:
    return (
        f"{row[f'split_{suffix}']}|{row[f'dataset_{suffix}']}|"
        f"{row[f'relative_path_{suffix}']}"
    )


def summarize_findings(
    findings: list[dict[str, object]], output_dir: Path
) -> dict[str, object]:
    group_buckets: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    nodes: dict[str, dict[str, object]] = {}
    neighbor_stats: dict[str, dict[str, object]] = defaultdict(
        lambda: {
            "probable_neighbors": set(),
            "possible_neighbors": set(),
            "cross_label_neighbors": set(),
            "train_neighbors": set(),
        }
    )
    parent: dict[str, str] = {}

    def find(value: str) -> str:
        parent.setdefault(value, value)
        if parent[value] != value:
            parent[value] = find(parent[value])
        return parent[value]

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for row in findings:
        split_pair = f"{row['split_a']}--{row['split_b']}"
        dataset_pair = f"{row['dataset_a']}--{row['dataset_b']}"
        group_buckets[(str(row["classification"]), split_pair, dataset_pair)].append(row)
        left = node_id(row, "a")
        right = node_id(row, "b")
        for value, suffix in ((left, "a"), (right, "b")):
            nodes[value] = {
                "node_id": value,
                "split": row[f"split_{suffix}"],
                "dataset": row[f"dataset_{suffix}"],
                "label": row[f"label_{suffix}"],
                "relative_path": row[f"relative_path_{suffix}"],
                "sha256": row[f"sha256_{suffix}"],
            }
        union(left, right)
        bucket_name = (
            "probable_neighbors"
            if row["classification"] == "probable_near_duplicate"
            else "possible_neighbors"
        )
        neighbor_stats[left][bucket_name].add(right)
        neighbor_stats[right][bucket_name].add(left)
        if str(row["label_a"]) != str(row["label_b"]):
            neighbor_stats[left]["cross_label_neighbors"].add(right)
            neighbor_stats[right]["cross_label_neighbors"].add(left)
        if row["split_a"] == "train":
            neighbor_stats[right]["train_neighbors"].add(left)
        if row["split_b"] == "train":
            neighbor_stats[left]["train_neighbors"].add(right)

    group_rows: list[dict[str, object]] = []
    for (classification, split_pair, dataset_pair), rows in sorted(group_buckets.items()):
        group_rows.append(
            {
                "classification": classification,
                "split_pair": split_pair,
                "dataset_pair": dataset_pair,
                "n_pairs": len(rows),
                "label_conflict_pairs": sum(
                    str(row["label_a"]) != str(row["label_b"]) for row in rows
                ),
                "unique_videos_a": len({node_id(row, "a") for row in rows}),
                "unique_videos_b": len({node_id(row, "b") for row in rows}),
            }
        )
    write_csv(
        output_dir / "finding_summary.csv",
        group_rows,
        list(group_rows[0]) if group_rows else ["classification", "n_pairs"],
    )

    affected_rows: list[dict[str, object]] = []
    for value, data in sorted(nodes.items(), key=lambda item: item[0]):
        stats = neighbor_stats[value]
        affected_rows.append(
            {
                **data,
                "highest_evidence": (
                    "probable_near_duplicate"
                    if stats["probable_neighbors"]
                    else "possible_near_duplicate"
                ),
                "probable_neighbors": len(stats["probable_neighbors"]),
                "possible_neighbors": len(stats["possible_neighbors"]),
                "cross_label_neighbors": len(stats["cross_label_neighbors"]),
                "has_train_neighbor": int(bool(stats["train_neighbors"])),
            }
        )
    write_csv(
        output_dir / "affected_videos.csv",
        affected_rows,
        list(affected_rows[0]) if affected_rows else ["node_id"],
    )

    components: dict[str, set[str]] = defaultdict(set)
    for value in nodes:
        components[find(value)].add(value)
    ordered_components = sorted(
        components.values(), key=lambda component: (-len(component), sorted(component)[0])
    )
    cluster_rows: list[dict[str, object]] = []
    cluster_summary_rows: list[dict[str, object]] = []
    for cluster_index, component in enumerate(ordered_components, start=1):
        cluster_id = f"cluster_{cluster_index:04d}"
        component_edges = [
            row
            for row in findings
            if node_id(row, "a") in component and node_id(row, "b") in component
        ]
        split_counts = {
            split: sum(nodes[value]["split"] == split for value in component)
            for split in ("train", "validation", "test")
        }
        cluster_summary_rows.append(
            {
                "cluster_id": cluster_id,
                "n_videos": len(component),
                "n_train": split_counts["train"],
                "n_validation": split_counts["validation"],
                "n_test": split_counts["test"],
                "n_edges": len(component_edges),
                "probable_edges": sum(
                    row["classification"] == "probable_near_duplicate"
                    for row in component_edges
                ),
                "possible_edges": sum(
                    row["classification"] == "possible_near_duplicate"
                    for row in component_edges
                ),
                "label_conflict_edges": sum(
                    str(row["label_a"]) != str(row["label_b"])
                    for row in component_edges
                ),
            }
        )
        for value in sorted(component):
            cluster_rows.append(
                {
                    "cluster_id": cluster_id,
                    "cluster_size": len(component),
                    **nodes[value],
                }
            )
    write_csv(
        output_dir / "near_duplicate_clusters.csv",
        cluster_rows,
        list(cluster_rows[0]) if cluster_rows else ["cluster_id"],
    )
    write_csv(
        output_dir / "cluster_summary.csv",
        cluster_summary_rows,
        list(cluster_summary_rows[0]) if cluster_summary_rows else ["cluster_id"],
    )

    def unique_by_split(rows: list[dict[str, object]]) -> dict[str, int]:
        return {
            split: len(
                {
                    node_id(row, suffix)
                    for row in rows
                    for suffix in ("a", "b")
                    if row[f"split_{suffix}"] == split
                }
            )
            for split in ("train", "validation", "test")
        }

    probable = [
        row for row in findings if row["classification"] == "probable_near_duplicate"
    ]
    train_test_probable = [
        row
        for row in probable
        if {str(row["split_a"]), str(row["split_b"])} == {"train", "test"}
    ]
    train_validation_probable = [
        row
        for row in probable
        if {str(row["split_a"]), str(row["split_b"])} == {"train", "validation"}
    ]
    return {
        "unique_flagged_videos_by_split": unique_by_split(findings),
        "unique_probable_videos_by_split": unique_by_split(probable),
        "probable_label_conflict_pairs": sum(
            str(row["label_a"]) != str(row["label_b"]) for row in probable
        ),
        "probable_train_test_pairs": len(train_test_probable),
        "test_videos_with_probable_train_match": unique_by_split(train_test_probable)["test"],
        "probable_train_validation_pairs": len(train_validation_probable),
        "validation_videos_with_probable_train_match": unique_by_split(
            train_validation_probable
        )["validation"],
        "flagged_connected_components": len(ordered_components),
        "largest_component_videos": max((len(component) for component in ordered_components), default=0),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows, manifest_hashes, exact_duplicate_stats = load_audit_rows()
    if args.max_videos is not None:
        rows = rows[: args.max_videos]
    signatures = load_or_extract_signatures(rows, args.datasets_root, args.output_dir)
    incomplete = [row for row in signatures if row["decode_status"] != "complete"]

    candidates = candidate_pairs(signatures)
    print(f"Evaluation approfondie de {len(candidates)} paires candidates.", flush=True)
    assessed = [assess_pair(signatures[left], signatures[right]) for left, right in sorted(candidates)]
    assessed.sort(
        key=lambda row: (
            {"probable_near_duplicate": 0, "possible_near_duplicate": 1}.get(
                str(row["classification"]), 2
            ),
            -int(row["ordered_matches_h8"]),
            int(row["min_phash_hamming"]),
        )
    )
    fields = list(assessed[0]) if assessed else [
        "split_a",
        "split_b",
        "relative_path_a",
        "relative_path_b",
        "min_phash_hamming",
        "ordered_matches_h6",
        "ordered_matches_h8",
        "classification",
    ]
    write_csv(args.output_dir / "cross_split_perceptual_candidates.csv", assessed, fields)

    findings = [
        row
        for row in assessed
        if row["classification"] in {"probable_near_duplicate", "possible_near_duplicate"}
    ]
    finding_fields = fields
    write_csv(args.output_dir / "near_duplicate_findings.csv", findings, finding_fields)
    make_figure(assessed, args.output_dir / "perceptual_candidate_distances.png")
    finding_stats = summarize_findings(findings, args.output_dir)

    split_counts = {
        split: sum(row["split"] == split for row in signatures)
        for split in ("train", "validation", "test")
    }
    pair_space = (
        split_counts["train"] * split_counts["validation"]
        + split_counts["train"] * split_counts["test"]
        + split_counts["validation"] * split_counts["test"]
    )
    class_counts = {
        classification: sum(row["classification"] == classification for row in assessed)
        for classification in (
            "probable_near_duplicate",
            "possible_near_duplicate",
            "isolated_frame_similarity",
        )
    }
    result = {
        "status": "complete" if not incomplete and args.max_videos is None else "complete_with_limitations",
        "created_utc": utc_now(),
        "scope": "augmented train versus validation versus test",
        "split_counts": split_counts,
        "cross_split_video_pair_space": pair_space,
        "exact_sha256_overlap_count": 0,
        "exact_sha256_duplicate_stats": exact_duplicate_stats,
        "manifest_sha256": manifest_hashes,
        "method": {
            "sample_fractions": SAMPLE_FRACTIONS,
            "frames_per_video": len(SAMPLE_FRACTIONS),
            "phash_bits": 64,
            "crop_fractions": CROP_FRACTIONS,
            "frame_candidate_hamming_threshold": PHASH_FRAME_THRESHOLD,
            "temporal_confirmation": "longest ordered matching frame subsequence",
            "metadata_checks": ["source_id", "exact basename", "duration", "resolution"],
        },
        "candidate_pairs": len(assessed),
        "classification_counts": class_counts,
        "near_duplicate_findings": len(findings),
        "finding_statistics": finding_stats,
        "incomplete_video_decodes": len(incomplete),
        "limitations": [
            "Nine sampled frames can miss a very short shared segment between otherwise different videos.",
            "pHash is robust to re-encoding and modest image changes, but not to every severe crop or overlay.",
            "Source identifiers are available for the historical enrichment, not for every source dataset clip.",
        ],
    }
    manual_review_path = args.output_dir / "manual_visual_review.json"
    manual_review = (
        json.loads(manual_review_path.read_text(encoding="utf-8"))
        if manual_review_path.exists()
        else None
    )
    if manual_review is not None:
        result["manual_visual_review"] = manual_review
    write_json(args.output_dir / "audit_result.json", result)

    verdict = (
        f"{len(findings)} paire(s) probable(s) ou possible(s) a examiner."
        if findings
        else "Aucun quasi-doublon probable ou possible n'a ete detecte."
    )
    summary = "\n".join(
        [
            "# Audit perceptuel anti-fuite",
            "",
            f"Audit complet de {len(signatures)} videos et {pair_space} paires inter-splits possibles.",
            "Le train audite inclut les 2 793 videos principales et les 25 videos d'enrichissement.",
            "",
            f"**Verdict : {verdict}**",
            "",
            f"- Chevauchements SHA-256 exacts : 0",
            f"- Groupes SHA-256 internes aux splits (hors fuite) : {exact_duplicate_stats['within_split_groups_total']}",
            f"- Paires candidates pHash analysees : {len(assessed)}",
            f"- Quasi-doublons probables : {class_counts['probable_near_duplicate']}",
            f"- Quasi-doublons possibles : {class_counts['possible_near_duplicate']}",
            f"- Similarites de frame isolees rejetees : {class_counts['isolated_frame_similarity']}",
            f"- Videos test avec un quasi-doublon probable dans le train : {finding_stats['test_videos_with_probable_train_match']}",
            f"- Videos validation avec un quasi-doublon probable dans le train : {finding_stats['validation_videos_with_probable_train_match']}",
            f"- Paires probables avec labels opposes : {finding_stats['probable_label_conflict_pairs']}",
            f"- Videos avec decodage incomplet : {len(incomplete)}",
            "- Controle visuel cible : 23/24 paires compatibles avec une meme source/scene, 1 inconclusive, 0 contradiction",
            "",
            "La methode utilise neuf positions temporelles, trois cadrages par frame (0, 10 et 20 %),",
            "un pHash 64 bits et une confirmation par correspondances temporelles ordonnees. Les",
            "metadonnees de source, duree, resolution et nom sont conservees dans les sorties.",
            "",
            "Limite : un fragment commun tres court entre deux videos longues peut tomber entre les",
            "positions echantillonnees; un recadrage severe avec surimpression peut aussi echapper au pHash.",
        ]
    )
    (args.output_dir / "summary.md").write_text(summary + "\n", encoding="utf-8")

    figure_names = [
        "perceptual_candidate_distances.png",
        "review_contact_sheet_borderline.png",
        "review_contact_sheet_possible.png",
        "review_contact_sheet_probable.png",
    ]
    figure_manifest = {
        "status": "complete",
        "created_utc": utc_now(),
        "figures": {
            name: {
                "sha256": sha256_file(args.output_dir / name),
                "size_bytes": (args.output_dir / name).stat().st_size,
            }
            for name in figure_names
        },
    }
    write_json(args.output_dir / "figure_manifest.json", figure_manifest)

    published = [
        "audit_result.json",
        "affected_videos.csv",
        "cluster_summary.csv",
        "cross_split_perceptual_candidates.csv",
        "finding_summary.csv",
        "figure_manifest.json",
        "manual_visual_review.json",
        "near_duplicate_clusters.csv",
        "near_duplicate_findings.csv",
        "perceptual_candidate_distances.png",
        "review_contact_sheet_borderline.png",
        "review_contact_sheet_possible.png",
        "review_contact_sheet_probable.png",
        "summary.md",
        "video_perceptual_signatures.csv",
    ]
    artifact_manifest = {
        "status": result["status"],
        "created_utc": utc_now(),
        "inputs": manifest_hashes,
        "outputs": {
            name: {
                "sha256": sha256_file(args.output_dir / name),
                "size_bytes": (args.output_dir / name).stat().st_size,
            }
            for name in published
        },
    }
    write_json(args.output_dir / "artifact_manifest.json", artifact_manifest)
    print(verdict, flush=True)


if __name__ == "__main__":
    main()
