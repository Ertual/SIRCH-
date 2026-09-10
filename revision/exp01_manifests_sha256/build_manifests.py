from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import os
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from sklearn.model_selection import train_test_split


VIDEO_PATTERNS = ("*.mp4", "*.avi", "*.mov", "*.mkv")
EXPECTED_MAIN_COUNTS = {"train": 2793, "validation": 599, "test": 599}
EXPECTED_HARD_NEGATIVE_COUNTS = {"sport": 15, "danse": 10, "calme": 5}


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Build sealed SIRCH manifests.")
    parser.add_argument(
        "--datasets-root",
        type=Path,
        default=Path(os.environ.get("SIRCH_DATASETS_DIR", r"C:\SIRCH_ENV\datasets")),
    )
    parser.add_argument("--project-root", type=Path, default=project_root)
    parser.add_argument(
        "--output-dir", type=Path, default=Path(__file__).resolve().parent / "outputs"
    )
    return parser.parse_args()


def collect_from_folder(folder: str, label: int) -> list[tuple[str, int]]:
    files: list[str] = []
    for pattern in VIDEO_PATTERNS:
        files.extend(glob.glob(os.path.join(folder, "**", pattern), recursive=True))
    return [(path, label) for path in files]


def collect_dataset(root: Path) -> list[tuple[str, int]]:
    samples: list[tuple[str, int]] = []
    label_rules = {
        1: ["Violence", "Fight"],
        0: ["NonViolence", "NonFight", "Non-Violence", "Non Violence"],
    }
    for label, names in label_rules.items():
        for name in names:
            for folder in glob.glob(str(root / "**" / name), recursive=True):
                if os.path.isdir(folder):
                    samples.extend(collect_from_folder(folder, label))
    unique: dict[str, int] = {}
    for path, label in samples:
        unique[path] = label
    return list(unique.items())


def reproduce_original_split(datasets_root: Path) -> tuple[dict[str, list[str]], dict[str, int]]:
    rwf_root = datasets_root / "RWF-2000"
    rlvs_root = datasets_root / "RLVS" / "Real Life Violence Dataset"
    for root in (rwf_root, rlvs_root):
        if not root.exists():
            raise FileNotFoundError(f"Dataset introuvable: {root}")

    samples: list[tuple[str, int]] = []
    for root in (rwf_root, rlvs_root):
        samples.extend(collect_dataset(root))
    samples = list(dict(samples).items())

    random.seed(42)
    random.shuffle(samples)
    paths = [path for path, _ in samples]
    labels = [label for _, label in samples]

    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        paths, labels, test_size=0.30, random_state=42, stratify=labels
    )
    validation_paths, test_paths, validation_labels, test_labels = train_test_split(
        temp_paths,
        temp_labels,
        test_size=0.50,
        random_state=42,
        stratify=temp_labels,
    )
    label_by_path = dict(
        zip(
            train_paths + validation_paths + test_paths,
            train_labels + validation_labels + test_labels,
        )
    )
    return {
        "train": train_paths,
        "validation": validation_paths,
        "test": test_paths,
    }, label_by_path


def dataset_and_relative_path(path: Path, datasets_root: Path) -> tuple[str, str]:
    rlvs_root = datasets_root / "RLVS" / "Real Life Violence Dataset"
    rwf_root = datasets_root / "RWF-2000"
    try:
        return "RLVS", path.resolve().relative_to(rlvs_root.resolve()).as_posix()
    except ValueError:
        return "RWF-2000", path.resolve().relative_to(rwf_root.resolve()).as_posix()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_reference_test(project_root: Path, test_paths: list[str]) -> None:
    reference = project_root / "phase2_results" / "metriques_avancees_scores_599.csv"
    if not reference.exists():
        raise FileNotFoundError(f"Reference de test introuvable: {reference}")
    with reference.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    published = {
        str(Path(row["video"]).resolve()).casefold()
        for row in rows
        if row.get("model") == "original"
    }
    reconstructed = {str(Path(path).resolve()).casefold() for path in test_paths}
    if published != reconstructed:
        raise RuntimeError(
            "Le test reconstruit ne correspond pas aux 599 videos publiees "
            f"(manquantes={len(published - reconstructed)}, "
            f"nouvelles={len(reconstructed - published)})."
        )


def hard_negative_paths(project_root: Path) -> dict[str, list[Path]]:
    datasets_dir = project_root / "datasets"
    folders = {
        "sport": datasets_dir / "test_sport_v2",
        "danse": datasets_dir / "test_danse_v2",
        "calme": datasets_dir / "test_calme_v2",
    }
    result: dict[str, list[Path]] = {}
    for category, folder in folders.items():
        if not folder.exists():
            raise FileNotFoundError(f"Dossier hard-negative introuvable: {folder}")
        paths = sorted(
            path
            for path in folder.rglob("*")
            if path.is_file() and path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}
        )
        expected = EXPECTED_HARD_NEGATIVE_COUNTS[category]
        if len(paths) != expected:
            raise RuntimeError(
                f"Nombre inattendu pour {category}: {len(paths)} au lieu de {expected}."
            )
        result[category] = paths
    return result


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "split",
        "split_index",
        "dataset",
        "category",
        "label",
        "label_name",
        "relative_path",
        "size_bytes",
        "sha256",
    ]
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    splits, label_by_path = reproduce_original_split(args.datasets_root)
    for split, expected in EXPECTED_MAIN_COUNTS.items():
        if len(splits[split]) != expected:
            raise RuntimeError(f"Split {split}: {len(splits[split])} au lieu de {expected}.")
    verify_reference_test(args.project_root, splits["test"])

    all_main_paths = sum(splits.values(), [])
    main_path_keys = [str(Path(path).resolve()).casefold() for path in all_main_paths]
    if len(main_path_keys) != len(set(main_path_keys)):
        raise RuntimeError("Un chemin principal apparait dans plusieurs splits.")

    rows_by_split: dict[str, list[dict[str, object]]] = defaultdict(list)
    hash_locations: dict[str, list[tuple[str, str]]] = defaultdict(list)
    total_to_hash = len(all_main_paths) + sum(EXPECTED_HARD_NEGATIVE_COUNTS.values())
    progress = 0

    for split_name in ("train", "validation", "test"):
        for index, raw_path in enumerate(splits[split_name]):
            path = Path(raw_path)
            if not path.exists():
                raise FileNotFoundError(path)
            dataset, relative_path = dataset_and_relative_path(path, args.datasets_root)
            label = int(label_by_path[raw_path])
            digest = sha256_file(path)
            row = {
                "split": split_name,
                "split_index": index,
                "dataset": dataset,
                "category": "",
                "label": label,
                "label_name": "violence" if label else "non_violence",
                "relative_path": relative_path,
                "size_bytes": path.stat().st_size,
                "sha256": digest,
            }
            rows_by_split[split_name].append(row)
            hash_locations[digest].append((split_name, f"{dataset}/{relative_path}"))
            progress += 1
            if progress % 250 == 0:
                print(f"SHA-256: {progress}/{total_to_hash}", flush=True)

    hard_rows: list[dict[str, object]] = []
    hard_paths = hard_negative_paths(args.project_root)
    for category in ("sport", "danse", "calme"):
        for path in hard_paths[category]:
            digest = sha256_file(path)
            row = {
                "split": "hard_negative_v2",
                "split_index": len(hard_rows),
                "dataset": "hard_negative_v2",
                "category": category,
                "label": 0,
                "label_name": "non_violence",
                "relative_path": f"test_{category}_v2/{path.name}",
                "size_bytes": path.stat().st_size,
                "sha256": digest,
            }
            hard_rows.append(row)
            hash_locations[digest].append(("hard_negative_v2", row["relative_path"]))
            progress += 1
    rows_by_split["hard_negative_v2"] = hard_rows
    print(f"SHA-256: {progress}/{total_to_hash}", flush=True)

    manifest_paths: dict[str, Path] = {}
    for split_name, rows in rows_by_split.items():
        manifest_path = args.output_dir / f"{split_name}_manifest.csv"
        write_csv(manifest_path, rows)
        manifest_paths[split_name] = manifest_path

    rows_by_hash: dict[str, list[dict[str, object]]] = defaultdict(list)
    for rows in rows_by_split.values():
        for row in rows:
            rows_by_hash[str(row["sha256"])].append(row)
    cross_split_duplicates = {
        digest: rows
        for digest, rows in rows_by_hash.items()
        if len({str(row["split"]) for row in rows}) > 1
    }
    duplicate_report = args.output_dir / "cross_split_duplicates.csv"
    with duplicate_report.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "duplicate_group",
            "sha256",
            "split",
            "dataset",
            "category",
            "label",
            "label_name",
            "relative_path",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for group_index, (digest, rows) in enumerate(
            sorted(cross_split_duplicates.items()), start=1
        ):
            for row in rows:
                writer.writerow(
                    {
                        "duplicate_group": group_index,
                        "sha256": digest,
                        "split": row["split"],
                        "dataset": row["dataset"],
                        "category": row["category"],
                        "label": row["label"],
                        "label_name": row["label_name"],
                        "relative_path": row["relative_path"],
                    }
                )

    hard_negative_overlap_groups = sum(
        1
        for rows in cross_split_duplicates.values()
        if "hard_negative_v2" in {str(row["split"]) for row in rows}
        and len({str(row["split"]) for row in rows}) > 1
    )
    label_conflict_groups = sum(
        1
        for rows in cross_split_duplicates.values()
        if len({int(row["label"]) for row in rows}) > 1
    )

    manifest_hashes = {name: sha256_file(path) for name, path in manifest_paths.items()}
    counts = {
        name: {
            "total": len(rows),
            "violence": sum(int(row["label"]) for row in rows),
            "non_violence": len(rows) - sum(int(row["label"]) for row in rows),
            "datasets": dict(Counter(str(row["dataset"]) for row in rows)),
            "categories": dict(
                Counter(str(row["category"]) for row in rows if row["category"])
            ),
        }
        for name, rows in rows_by_split.items()
    }
    seal = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": (
            "sealed_with_cross_split_duplicates"
            if cross_split_duplicates
            else "sealed_clean"
        ),
        "split_protocol": "train_test_split 70/15/15, stratified, seed=42",
        "test_reference_verified": True,
        "cross_split_sha256_duplicate_groups": len(cross_split_duplicates),
        "hard_negative_overlap_groups": hard_negative_overlap_groups,
        "label_conflict_groups": label_conflict_groups,
        "counts": counts,
        "manifest_sha256": manifest_hashes,
        "duplicate_report_sha256": sha256_file(duplicate_report),
    }
    seal_path = args.output_dir / "manifest_seal.json"
    seal_path.write_text(json.dumps(seal, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    summary_lines = [
        "# Resume des manifestes SIRCH",
        "",
        f"Generation UTC : {seal['created_utc']}",
        "",
        "| Ensemble | Total | Violence | Non-violence | SHA-256 du manifeste |",
        "|---|---:|---:|---:|---|",
    ]
    for name in ("train", "validation", "test", "hard_negative_v2"):
        item = counts[name]
        summary_lines.append(
            f"| {name} | {item['total']} | {item['violence']} | "
            f"{item['non_violence']} | `{manifest_hashes[name]}` |"
        )
    summary_lines.extend(
        [
            "",
            "Le test principal reconstruit correspond exactement aux 599 videos publiees.",
            f"Groupes SHA-256 traversant plusieurs ensembles : {len(cross_split_duplicates)}.",
            f"Chevauchements hard-negative v2 / corpus principal : {hard_negative_overlap_groups}.",
            f"Groupes avec etiquettes contradictoires : {label_conflict_groups}.",
            "Voir `cross_split_duplicates.csv` pour les chemins concernes.",
        ]
    )
    (args.output_dir / "summary.md").write_text(
        "\n".join(summary_lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(seal, indent=2, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
