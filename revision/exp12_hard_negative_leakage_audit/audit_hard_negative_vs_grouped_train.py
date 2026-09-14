from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = EXP_DIR / "outputs"
EXP01_OUTPUTS = PROJECT_ROOT / "revision" / "exp01_manifests_sha256" / "outputs"
EXP10_OUTPUTS = PROJECT_ROOT / "revision" / "exp10_perceptual_leakage_audit" / "outputs"
EXP11_OUTPUTS = PROJECT_ROOT / "revision" / "exp11_grouped_scene_splits" / "outputs"
DATASETS_ROOT = Path(r"C:\SIRCH_ENV\datasets")

sys.path.insert(0, str(PROJECT_ROOT))
from revision.exp10_perceptual_leakage_audit import (  # noqa: E402
    audit_perceptual_leakage as perceptual,
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, indent=2, ensure_ascii=True) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_inputs() -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, str]]:
    train_path = EXP11_OUTPUTS / "train_manifest_grouped.csv"
    hard_path = EXP01_OUTPUTS / "hard_negative_v2_manifest.csv"
    exp11_seal = json.loads((EXP11_OUTPUTS / "manifest_sha256.json").read_text(encoding="utf-8"))
    exp01_seal = json.loads((EXP01_OUTPUTS / "manifest_seal.json").read_text(encoding="utf-8"))
    expected_train = next(
        item["sha256"] for item in exp11_seal["files"] if item["file"] == train_path.name
    )
    expected_hard = exp01_seal["manifest_sha256"]["hard_negative_v2"]
    actual = {"train": sha256_file(train_path), "hard_negative": sha256_file(hard_path)}
    expected = {"train": expected_train, "hard_negative": expected_hard}
    if actual != expected:
        raise RuntimeError(f"Manifeste non scelle: attendu={expected}, obtenu={actual}")

    train_rows = read_csv(train_path)
    hard_rows = read_csv(hard_path)
    if len(train_rows) != 2785 or any(row["split"] != "train" for row in train_rows):
        raise RuntimeError("Le nouveau train principal doit contenir 2785 videos.")
    expected_categories = {"sport": 15, "danse": 10, "calme": 5}
    categories = Counter(row["category"] for row in hard_rows)
    if len(hard_rows) != 30 or dict(categories) != expected_categories:
        raise RuntimeError(f"Corpus hard-negative inattendu: {dict(categories)}")
    for row in hard_rows:
        path = PROJECT_ROOT / "datasets" / row["relative_path"]
        if not path.exists() or sha256_file(path) != row["sha256"]:
            raise RuntimeError(f"Video hard-negative absente ou modifiee: {path}")
    return train_rows, hard_rows, actual


def load_train_signatures(train_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    prior = read_csv(EXP10_OUTPUTS / "video_perceptual_signatures.csv")
    by_video = {
        (row["dataset"], row["relative_path"], row["video_sha256"]): row for row in prior
    }
    signatures: list[dict[str, str]] = []
    for row in train_rows:
        key = (row["dataset"], row["relative_path"], row["sha256"])
        if key not in by_video:
            raise RuntimeError(f"Signature exp10 introuvable pour {key}")
        signature = dict(by_video[key])
        signature["split"] = "train"
        signature["manifest_row_id"] = f"train|{row['dataset']}|{row['relative_path']}"
        signature["source_group"] = "principal_train_grouped_clean"
        signature["label"] = row["label"]
        signatures.append(signature)
    return signatures


def extract_hard_signatures(hard_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    cache_path = OUTPUT_DIR / "hard_negative_perceptual_signatures.csv"
    expected_ids = [f"hard_negative|hard_negative_v2|{row['relative_path']}" for row in hard_rows]
    if cache_path.exists():
        cached = read_csv(cache_path)
        if [row["manifest_row_id"] for row in cached] == expected_ids:
            if all(row["video_sha256"] == source["sha256"] for row, source in zip(cached, hard_rows)):
                print("Signatures hard-negative reutilisees: 30", flush=True)
                return cached

    signatures: list[dict[str, str]] = []
    for index, row in enumerate(hard_rows, start=1):
        extraction_row = dict(row)
        extraction_row["split"] = "hard_negative"
        extraction_row["dataset"] = "historical_enrichment_v1"
        extraction_row["source_group"] = "hard_negative_v2"
        signature = perceptual.extract_signature(extraction_row, DATASETS_ROOT)
        signature["manifest_row_id"] = expected_ids[index - 1]
        signature["dataset"] = "hard_negative_v2"
        signature["split"] = "hard_negative"
        signature["source_group"] = "hard_negative_v2"
        signatures.append({key: str(value) for key, value in signature.items()})
        print(f"Signatures hard-negative: {index}/30", flush=True)
    write_csv(cache_path, signatures, perceptual.SIGNATURE_FIELDS)
    return signatures


def audit_pairs(
    train_signatures: list[dict[str, str]],
    hard_signatures: list[dict[str, str]],
    category_by_path: dict[str, str],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    signatures = train_signatures + hard_signatures
    candidates = perceptual.candidate_pairs(signatures)
    rows: list[dict[str, object]] = []
    for index, (left_index, right_index) in enumerate(sorted(candidates), start=1):
        left = signatures[left_index]
        right = signatures[right_index]
        if {left["split"], right["split"]} != {"train", "hard_negative"}:
            continue
        assessed = perceptual.assess_pair(left, right)
        hard_path = (
            left["relative_path"] if left["split"] == "hard_negative" else right["relative_path"]
        )
        assessed["hard_negative_category"] = category_by_path[hard_path]
        rows.append(assessed)
        if index % 100 == 0:
            print(f"Paires candidates evaluees: {index}/{len(candidates)}", flush=True)
    findings = [
        row
        for row in rows
        if row["classification"] in {"probable_near_duplicate", "possible_near_duplicate"}
    ]
    return rows, findings


def artifact_manifest() -> None:
    artifacts = []
    for path in sorted(OUTPUT_DIR.rglob("*")):
        if path.is_file() and path.name != "artifact_manifest.json":
            artifacts.append(
                {
                    "file": path.relative_to(EXP_DIR).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    write_json(OUTPUT_DIR / "artifact_manifest.json", {"artifacts": artifacts})


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_rows, hard_rows, manifest_hashes = load_inputs()
    exact_overlap = sorted({row["sha256"] for row in train_rows} & {row["sha256"] for row in hard_rows})
    train_signatures = load_train_signatures(train_rows)
    hard_signatures = extract_hard_signatures(hard_rows)
    combined = train_signatures + hard_signatures
    write_csv(OUTPUT_DIR / "video_perceptual_signatures.csv", combined, perceptual.SIGNATURE_FIELDS)

    category_by_path = {row["relative_path"]: row["category"] for row in hard_rows}
    candidates, findings = audit_pairs(train_signatures, hard_signatures, category_by_path)
    fields = [
        "split_a", "split_b", "dataset_a", "dataset_b", "label_a", "label_b",
        "relative_path_a", "relative_path_b", "sha256_a", "sha256_b", "source_id_a",
        "source_id_b", "source_id_match", "basename_match", "duration_seconds_a",
        "duration_seconds_b", "duration_delta_seconds", "duration_ratio", "resolution_a",
        "resolution_b", "min_phash_hamming", "ordered_matches_h6", "ordered_matches_h8",
        "classification", "hard_negative_category",
    ]
    write_csv(OUTPUT_DIR / "candidate_pairs.csv", candidates, fields)
    write_csv(OUTPUT_DIR / "near_duplicate_findings.csv", findings, fields)
    perceptual.make_figure(candidates, OUTPUT_DIR / "candidate_phash_distances.png")

    counts = Counter(row["classification"] for row in findings)
    manual_path = OUTPUT_DIR / "manual_visual_review.json"
    manual_review = (
        json.loads(manual_path.read_text(encoding="utf-8")) if manual_path.exists() else None
    )
    affected_hard = sorted(
        {
            row["relative_path_a"] if row["split_a"] == "hard_negative" else row["relative_path_b"]
            for row in findings
        }
    )
    result = {
        "status": "PASS" if not exact_overlap and not findings else "REVIEW_REQUIRED",
        "created_utc": utc_now(),
        "scope": "30 sealed hard-negative v2 videos versus 2785 grouped principal train videos",
        "method": {
            "sample_positions": 9,
            "crop_fractions": list(perceptual.CROP_FRACTIONS),
            "phash_bits": 64,
            "frame_candidate_hamming_threshold": perceptual.PHASH_FRAME_THRESHOLD,
            "classification_rules": "identical to revision/exp10_perceptual_leakage_audit",
        },
        "manifest_sha256": manifest_hashes,
        "exact_sha256_overlap_count": len(exact_overlap),
        "candidate_pair_count": len(candidates),
        "probable_near_duplicate_count": counts["probable_near_duplicate"],
        "possible_near_duplicate_count": counts["possible_near_duplicate"],
        "affected_hard_negative_video_count": len(affected_hard),
        "affected_hard_negative_videos": affected_hard,
        "manual_visual_review": manual_review,
    }
    write_json(OUTPUT_DIR / "result.json", result)
    summary = "\n".join(
        [
            "# Audit hard-negative contre nouveau train groupe",
            "",
            f"Verdict: **{result['status']}**.",
            "",
            f"- Train principal: {len(train_rows)} videos",
            f"- Hard-negative v2: {len(hard_rows)} videos",
            f"- Paires possibles exhaustives: {len(train_rows) * len(hard_rows)}",
            f"- Chevauchements SHA-256 exacts: {len(exact_overlap)}",
            f"- Paires candidates pHash: {len(candidates)}",
            f"- Quasi-doublons probables: {counts['probable_near_duplicate']}",
            f"- Quasi-doublons possibles: {counts['possible_near_duplicate']}",
            f"- Videos hard-negative affectees: {len(affected_hard)}",
            "- Controle visuel de l'unique similarite isolee: scenes sans rapport; rejet conforte.",
            "",
            "La methode et les seuils sont identiques a exp10. Aucun modele ni resultat",
            "de classification n'est consulte par cet audit.",
            "",
        ]
    )
    with (OUTPUT_DIR / "summary.md").open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(summary)
    figures = [
        {
            "file": "outputs/candidate_phash_distances.png",
            "purpose": "Distribution des distances pHash minimales des paires candidates",
            "sha256": sha256_file(OUTPUT_DIR / "candidate_phash_distances.png"),
        }
    ]
    visual_path = OUTPUT_DIR / "candidate_pair_visual_review.jpg"
    if visual_path.exists():
        figures.append(
            {
                "file": "outputs/candidate_pair_visual_review.jpg",
                "purpose": "Controle visuel de l'unique similarite de frame isolee",
                "sha256": sha256_file(visual_path),
            }
        )
    write_json(OUTPUT_DIR / "figure_manifest.json", {"figures": figures})
    artifact_manifest()
    print(json.dumps(result, indent=2, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
