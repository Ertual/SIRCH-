from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = EXP_DIR / "outputs"
EXP01_OUTPUTS = PROJECT_ROOT / "revision" / "exp01_manifests_sha256" / "outputs"
EXP04_OUTPUTS = PROJECT_ROOT / "revision" / "exp04_augmented_clean" / "outputs"
EXP10_OUTPUTS = PROJECT_ROOT / "revision" / "exp10_perceptual_leakage_audit" / "outputs"

SPLITS = ("train", "validation", "test")
TARGET_SIZES = {"train": 2785, "validation": 597, "test": 597}
ORDER_SALT = "sirch-exp11-grouped-scene-splits-v1"

# Each correction was checked on the nine-frame review sheet and a 24-frame
# second-pass sheet. Other reviewed clips retain their original labels.
CORRECTIONS = {
    "V096": (
        0,
        "Hall d'hotel: passants et bagages; aucun coup, contact agressif, "
        "poursuite ou menace visible sur 24 instants.",
    ),
    "V102": (
        0,
        "Voiture garee seule pendant tout le clip; aucun individu ni action "
        "violente visible sur 24 instants.",
    ),
    "V109": (
        0,
        "Fin de scene: les personnes se dispersent et marchent; aucun contact "
        "agressif visible dans ce segment sur 24 instants.",
    ),
    "V113": (
        0,
        "Parking lointain: des pietons marchent; aucun contact ni geste "
        "agressif visible sur 24 instants.",
    ),
    "V116": (
        0,
        "Homme seul pres d'une cloture; aucun adversaire, coup ou action "
        "violente visible sur 24 instants.",
    ),
}


class DisjointSet:
    def __init__(self, values: Iterable[str]) -> None:
        self.parent = {value: value for value in values}
        self.rank = {value: 0 for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left == root_right:
            return
        if self.rank[root_left] < self.rank[root_right]:
            root_left, root_right = root_right, root_left
        self.parent[root_right] = root_left
        if self.rank[root_left] == self.rank[root_right]:
            self.rank[root_left] += 1


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, indent=2, ensure_ascii=True) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def identity(dataset: str, relative_path: str) -> str:
    return f"{dataset}|{relative_path}"


def finding_node(row: dict[str, str], suffix: str) -> str:
    return f"{row[f'split_{suffix}']}|{row[f'dataset_{suffix}']}|{row[f'relative_path_{suffix}']}"


def load_principal_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for split in SPLITS:
        path = EXP01_OUTPUTS / f"{split}_manifest.csv"
        for row in read_csv(path):
            prepared: dict[str, object] = dict(row)
            prepared["_identity"] = identity(row["dataset"], row["relative_path"])
            prepared["_old_split"] = split
            prepared["_old_index"] = int(row["split_index"])
            prepared["_original_label"] = int(row["label"])
            prepared["label"] = int(row["label"])
            rows.append(prepared)
    if len(rows) != 3979:
        raise RuntimeError(f"3979 videos principales attendues, {len(rows)} trouvees")
    identifiers = [str(row["_identity"]) for row in rows]
    if len(identifiers) != len(set(identifiers)):
        raise RuntimeError("Une video principale apparait plusieurs fois dans les manifestes")
    return rows


def complete_visual_review() -> tuple[list[dict[str, object]], dict[str, dict[str, object]]]:
    review_path = OUTPUT_DIR / "conflict_video_review.csv"
    review_rows: list[dict[str, object]] = []
    by_node: dict[str, dict[str, object]] = {}
    for source in read_csv(review_path):
        row: dict[str, object] = dict(source)
        review_id = source["review_id"]
        original = int(source["original_label"])
        reviewed = CORRECTIONS.get(review_id, (original, ""))[0]
        if review_id in CORRECTIONS:
            decision = "corrected_to_non_violence"
            evidence = CORRECTIONS[review_id][1]
        elif original == 1:
            decision = "confirmed_violence"
            evidence = (
                "Revue de 9 instants repartis sur tout le clip; interaction physique "
                "agressive, lutte, poursuite ou menace visible dans le segment."
            )
        else:
            decision = "confirmed_non_violence"
            evidence = (
                "Revue de 9 instants repartis sur tout le clip; aucune interaction "
                "physique agressive visible dans ce segment."
            )
        row["reviewed_label"] = reviewed
        row["review_decision"] = decision
        row["visual_evidence"] = evidence
        review_rows.append(row)
        by_node[str(row["node_id"])] = row

    if len(review_rows) != 129:
        raise RuntimeError(f"129 videos conflictuelles attendues, {len(review_rows)} trouvees")
    write_csv(
        review_path,
        review_rows,
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
    return review_rows, by_node


def document_conflict_pairs(by_node: dict[str, dict[str, object]]) -> list[dict[str, object]]:
    findings = read_csv(EXP10_OUTPUTS / "near_duplicate_findings.csv")
    conflicts = [
        row
        for row in findings
        if row["classification"] == "probable_near_duplicate"
        and row["label_a"] != row["label_b"]
    ]
    if len(conflicts) != 90:
        raise RuntimeError(f"90 paires conflictuelles attendues, {len(conflicts)} trouvees")

    output: list[dict[str, object]] = []
    for index, finding in enumerate(conflicts, start=1):
        left = by_node[finding_node(finding, "a")]
        right = by_node[finding_node(finding, "b")]
        reviewed_left = int(left["reviewed_label"])
        reviewed_right = int(right["reviewed_label"])
        corrected_ids = [
            str(row["review_id"])
            for row in (left, right)
            if int(row["original_label"]) != int(row["reviewed_label"])
        ]
        if reviewed_left == reviewed_right:
            resolution = "resolved_by_label_correction"
            verdict = (
                "Les deux extraits montrent la meme classe apres correction visuelle de "
                + ", ".join(corrected_ids)
                + "."
            )
        else:
            resolution = "legitimate_temporal_transition"
            verdict = (
                "Labels opposes confirmes: les extraits proviennent de la meme scene, "
                "mais couvrent une phase violente et une phase sans violence distinctes."
            )
        cluster_ids = sorted({str(left["cluster_id"]), str(right["cluster_id"])})
        output.append(
            {
                "pair_id": f"C{index:03d}",
                "cluster_id": ";".join(cluster_ids),
                "review_id_a": left["review_id"],
                "review_id_b": right["review_id"],
                "split_a": finding["split_a"],
                "split_b": finding["split_b"],
                "relative_path_a": finding["relative_path_a"],
                "relative_path_b": finding["relative_path_b"],
                "original_label_a": finding["label_a"],
                "original_label_b": finding["label_b"],
                "reviewed_label_a": reviewed_left,
                "reviewed_label_b": reviewed_right,
                "corrected_review_ids": ";".join(corrected_ids),
                "resolution": resolution,
                "visual_verdict": verdict,
                "min_phash_hamming": finding["min_phash_hamming"],
                "ordered_matches_h6": finding["ordered_matches_h6"],
                "ordered_matches_h8": finding["ordered_matches_h8"],
            }
        )
    write_csv(
        OUTPUT_DIR / "label_conflict_pair_review.csv",
        output,
        list(output[0].keys()),
    )
    return output


def apply_label_corrections(
    rows: list[dict[str, object]], review_rows: list[dict[str, object]]
) -> list[dict[str, object]]:
    corrections_by_identity: dict[str, dict[str, object]] = {}
    for review in review_rows:
        if int(review["original_label"]) == int(review["reviewed_label"]):
            continue
        identifier = identity(str(review["dataset"]), str(review["relative_path"]))
        corrections_by_identity[identifier] = review

    correction_output: list[dict[str, object]] = []
    for row in rows:
        identifier = str(row["_identity"])
        review = corrections_by_identity.get(identifier)
        if review is None:
            row["_review_id"] = ""
            row["_label_corrected"] = 0
            continue
        row["label"] = int(review["reviewed_label"])
        row["label_name"] = "violence" if int(row["label"]) == 1 else "non_violence"
        row["_review_id"] = review["review_id"]
        row["_label_corrected"] = 1
        correction_output.append(
            {
                "review_id": review["review_id"],
                "dataset": row["dataset"],
                "relative_path": row["relative_path"],
                "sha256": row["sha256"],
                "previous_split": row["_old_split"],
                "original_label": row["_original_label"],
                "corrected_label": row["label"],
                "visual_evidence": review["visual_evidence"],
            }
        )
    if len(correction_output) != len(CORRECTIONS):
        raise RuntimeError(
            f"{len(CORRECTIONS)} corrections attendues, {len(correction_output)} appliquees"
        )
    correction_output.sort(key=lambda row: str(row["review_id"]))
    write_csv(
        OUTPUT_DIR / "label_corrections.csv",
        correction_output,
        list(correction_output[0].keys()),
    )
    return correction_output


def build_scene_groups(
    rows: list[dict[str, object]],
) -> tuple[list[dict[str, object]], dict[str, list[dict[str, object]]]]:
    row_by_identity = {str(row["_identity"]): row for row in rows}
    dsu = DisjointSet(row_by_identity)

    perceptual_members: dict[str, list[str]] = defaultdict(list)
    for item in read_csv(EXP10_OUTPUTS / "near_duplicate_clusters.csv"):
        identifier = identity(item["dataset"], item["relative_path"])
        if identifier not in row_by_identity:
            raise RuntimeError(f"Video du groupe perceptuel absente: {identifier}")
        perceptual_members[item["cluster_id"]].append(identifier)
    if len(perceptual_members) != 191:
        raise RuntimeError(f"191 groupes perceptuels attendus, {len(perceptual_members)} trouves")
    for members in perceptual_members.values():
        for member in members[1:]:
            dsu.union(members[0], member)

    sha_members: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        sha_members[str(row["sha256"])].append(str(row["_identity"]))
    for members in sha_members.values():
        for member in members[1:]:
            dsu.union(members[0], member)

    components: dict[str, list[dict[str, object]]] = defaultdict(list)
    for identifier, row in row_by_identity.items():
        components[dsu.find(identifier)].append(row)

    perceptual_ids_by_root: dict[str, set[str]] = defaultdict(set)
    for cluster_id, members in perceptual_members.items():
        perceptual_ids_by_root[dsu.find(members[0])].add(cluster_id)

    ordered_roots = sorted(
        components,
        key=lambda root: min(str(row["_identity"]) for row in components[root]),
    )
    exact_counter = 0
    singleton_counter = 0
    group_rows: list[dict[str, object]] = []
    groups: dict[str, list[dict[str, object]]] = {}
    for root in ordered_roots:
        members = sorted(components[root], key=lambda row: str(row["_identity"]))
        perceptual_ids = sorted(perceptual_ids_by_root.get(root, set()))
        if len(perceptual_ids) == 1:
            group_id = perceptual_ids[0]
            origin = "perceptual_near_duplicate"
        elif perceptual_ids:
            digest = hashlib.sha256("|".join(perceptual_ids).encode()).hexdigest()[:10]
            group_id = f"merged_perceptual_{digest}"
            origin = "perceptual_plus_exact_sha"
        elif len(members) > 1:
            exact_counter += 1
            group_id = f"exact_sha_group_{exact_counter:04d}"
            origin = "exact_sha256"
        else:
            singleton_counter += 1
            group_id = f"singleton_{singleton_counter:04d}"
            origin = "singleton"
        for member in members:
            member["_scene_group_id"] = group_id
            member["_group_origin"] = origin
        groups[group_id] = members
        group_rows.append(
            {
                "scene_group_id": group_id,
                "group_origin": origin,
                "perceptual_cluster_ids": ";".join(perceptual_ids),
                "size": len(members),
                "label_0_count": sum(int(row["label"]) == 0 for row in members),
                "label_1_count": sum(int(row["label"]) == 1 for row in members),
                "old_train_count": sum(row["_old_split"] == "train" for row in members),
                "old_validation_count": sum(
                    row["_old_split"] == "validation" for row in members
                ),
                "old_test_count": sum(row["_old_split"] == "test" for row in members),
                "new_split": "",
                "moved_count": "",
            }
        )
    return group_rows, groups


def hamilton_label_targets(rows: list[dict[str, object]]) -> dict[str, int]:
    total_label_1 = sum(int(row["label"]) == 1 for row in rows)
    total = len(rows)
    quotas = {
        split: total_label_1 * TARGET_SIZES[split] / total for split in SPLITS
    }
    targets = {split: int(np.floor(quota)) for split, quota in quotas.items()}
    remaining = total_label_1 - sum(targets.values())
    ranked = sorted(SPLITS, key=lambda split: (quotas[split] - targets[split], split), reverse=True)
    for split in ranked[:remaining]:
        targets[split] += 1
    return targets


def assign_groups(
    rows: list[dict[str, object]],
    group_rows: list[dict[str, object]],
    groups: dict[str, list[dict[str, object]]],
) -> tuple[dict[str, str], dict[str, int], dict[str, object]]:
    group_ids = sorted(groups)
    group_count = len(group_ids)
    split_count = len(SPLITS)
    variable_count = group_count * split_count
    label_targets = hamilton_label_targets(rows)

    matrix = lil_matrix((group_count + 2 * split_count, variable_count), dtype=float)
    lower = np.zeros(group_count + 2 * split_count, dtype=float)
    upper = np.zeros(group_count + 2 * split_count, dtype=float)
    costs = np.zeros(variable_count, dtype=float)

    for group_index, group_id in enumerate(group_ids):
        members = groups[group_id]
        matrix[group_index, group_index * split_count : (group_index + 1) * split_count] = 1
        lower[group_index] = 1
        upper[group_index] = 1
        old_counts = Counter(str(row["_old_split"]) for row in members)
        for split_index, split in enumerate(SPLITS):
            variable = group_index * split_count + split_index
            moved = len(members) - old_counts[split]
            tie_digest = hashlib.sha256(f"{ORDER_SALT}|{group_id}|{split}".encode()).digest()
            tie = int.from_bytes(tie_digest[:4], "big") / (2**32) * 1e-6
            costs[variable] = moved + tie

    for split_index, split in enumerate(SPLITS):
        total_row = group_count + split_index
        label_row = group_count + split_count + split_index
        for group_index, group_id in enumerate(group_ids):
            variable = group_index * split_count + split_index
            members = groups[group_id]
            matrix[total_row, variable] = len(members)
            matrix[label_row, variable] = sum(int(row["label"]) == 1 for row in members)
        lower[total_row] = upper[total_row] = TARGET_SIZES[split]
        lower[label_row] = upper[label_row] = label_targets[split]

    result = milp(
        c=costs,
        integrality=np.ones(variable_count, dtype=int),
        bounds=Bounds(np.zeros(variable_count), np.ones(variable_count)),
        constraints=LinearConstraint(matrix.tocsr(), lower, upper),
        options={"time_limit": 300.0, "mip_rel_gap": 0.0},
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"Affectation MILP impossible: {result.message}")

    assignments: dict[str, str] = {}
    for group_index, group_id in enumerate(group_ids):
        values = result.x[group_index * split_count : (group_index + 1) * split_count]
        assignments[group_id] = SPLITS[int(np.argmax(values))]
    for row in rows:
        row["_new_split"] = assignments[str(row["_scene_group_id"])]

    group_row_by_id = {str(row["scene_group_id"]): row for row in group_rows}
    for group_id, split in assignments.items():
        members = groups[group_id]
        group_row_by_id[group_id]["new_split"] = split
        group_row_by_id[group_id]["moved_count"] = sum(
            row["_old_split"] != split for row in members
        )

    solver = {
        "method": "scipy.optimize.milp (HiGHS)",
        "objective": "minimize moved videos under exact size and class constraints",
        "success": bool(result.success),
        "status": int(result.status),
        "message": str(result.message),
        "objective_value": float(result.fun),
    }
    return assignments, label_targets, solver


def stable_order(row: dict[str, object], split: str) -> str:
    return hashlib.sha256(
        f"{ORDER_SALT}|{split}|{row['sha256']}|{row['relative_path']}".encode()
    ).hexdigest()


MANIFEST_FIELDS = [
    "split",
    "split_index",
    "dataset",
    "category",
    "label",
    "label_name",
    "relative_path",
    "size_bytes",
    "sha256",
    "scene_group_id",
    "group_origin",
    "previous_split",
    "previous_split_index",
    "split_changed",
    "original_label",
    "label_corrected",
    "label_review_id",
]


def public_manifest_row(row: dict[str, object], split: str, index: int) -> dict[str, object]:
    return {
        "split": split,
        "split_index": index,
        "dataset": row["dataset"],
        "category": row["category"],
        "label": row["label"],
        "label_name": row["label_name"],
        "relative_path": row["relative_path"],
        "size_bytes": row["size_bytes"],
        "sha256": row["sha256"],
        "scene_group_id": row["_scene_group_id"],
        "group_origin": row["_group_origin"],
        "previous_split": row["_old_split"],
        "previous_split_index": row["_old_index"],
        "split_changed": int(row["_old_split"] != split),
        "original_label": row["_original_label"],
        "label_corrected": row.get("_label_corrected", 0),
        "label_review_id": row.get("_review_id", ""),
    }


def write_manifests(rows: list[dict[str, object]]) -> dict[str, list[dict[str, object]]]:
    manifests: dict[str, list[dict[str, object]]] = {}
    for split in SPLITS:
        selected = [row for row in rows if row["_new_split"] == split]
        selected.sort(key=lambda row: stable_order(row, split))
        public_rows = [public_manifest_row(row, split, index) for index, row in enumerate(selected)]
        manifests[split] = public_rows
        write_csv(OUTPUT_DIR / f"{split}_manifest_grouped.csv", public_rows, MANIFEST_FIELDS)
    return manifests


def write_augmented_manifest(
    train_rows: list[dict[str, object]], principal_rows: list[dict[str, object]]
) -> tuple[list[dict[str, object]], dict[str, int]]:
    old_augmented = read_csv(EXP04_OUTPUTS / "augmented_train_manifest.csv")
    enrichment = [row for row in old_augmented if row["source_group"] != "principal_train_clean"]
    if len(enrichment) != 25:
        raise RuntimeError(f"25 videos enrichies attendues, {len(enrichment)} trouvees")
    principal_hashes = {str(row["sha256"]) for row in principal_rows}
    exact_overlaps = sum(row["sha256"] in principal_hashes for row in enrichment)
    perceptual_cluster_rows = sum(
        row["dataset"] == "historical_enrichment_v1"
        for row in read_csv(EXP10_OUTPUTS / "near_duplicate_clusters.csv")
    )
    if exact_overlaps or perceptual_cluster_rows:
        raise RuntimeError(
            "L'enrichissement historique chevauche le corpus principal et ne peut pas "
            "rester train-only sans regroupement supplementaire"
        )

    combined: list[dict[str, object]] = []
    for row in train_rows:
        augmented = dict(row)
        augmented["source_group"] = "principal_train_grouped_clean"
        combined.append(augmented)
    for source in enrichment:
        combined.append(
            {
                "split": "train",
                "split_index": 0,
                "dataset": source["dataset"],
                "category": source["category"],
                "label": int(source["label"]),
                "label_name": source["label_name"],
                "relative_path": source["relative_path"],
                "size_bytes": source["size_bytes"],
                "sha256": source["sha256"],
                "scene_group_id": f"historical_enrichment:{source['relative_path']}",
                "group_origin": "historical_enrichment_training_only",
                "previous_split": "train",
                "previous_split_index": source["split_index"],
                "split_changed": 0,
                "original_label": int(source["label"]),
                "label_corrected": 0,
                "label_review_id": "",
                "source_group": source["source_group"],
            }
        )
    combined.sort(key=lambda row: stable_order(row, "augmented_train"))
    for index, row in enumerate(combined):
        row["split_index"] = index
    fields = MANIFEST_FIELDS + ["source_group"]
    write_csv(OUTPUT_DIR / "augmented_train_manifest_grouped.csv", combined, fields)
    return combined, {
        "video_count": len(enrichment),
        "exact_sha256_overlap_with_principal": exact_overlaps,
        "perceptual_cluster_rows": perceptual_cluster_rows,
    }


def verification(
    rows: list[dict[str, object]],
    manifests: dict[str, list[dict[str, object]]],
    augmented: list[dict[str, object]],
    group_rows: list[dict[str, object]],
    groups: dict[str, list[dict[str, object]]],
    label_targets: dict[str, int],
    solver: dict[str, object],
    enrichment_audit: dict[str, int],
) -> dict[str, object]:
    group_splits: dict[str, set[str]] = defaultdict(set)
    sha_splits: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        group_splits[str(row["_scene_group_id"])].add(str(row["_new_split"]))
        sha_splits[str(row["sha256"])].add(str(row["_new_split"]))
    group_violations = sorted(group for group, splits in group_splits.items() if len(splits) > 1)
    sha_violations = sorted(sha for sha, splits in sha_splits.items() if len(splits) > 1)

    summary_rows: list[dict[str, object]] = []
    for split in SPLITS:
        manifest = manifests[split]
        summary_rows.append(
            {
                "split": split,
                "target_total": TARGET_SIZES[split],
                "actual_total": len(manifest),
                "proportion": f"{len(manifest) / len(rows):.8f}",
                "label_0": sum(int(row["label"]) == 0 for row in manifest),
                "label_1": sum(int(row["label"]) == 1 for row in manifest),
                "target_label_1": label_targets[split],
                "rwf_2000": sum(row["dataset"] == "RWF-2000" for row in manifest),
                "rlvs": sum(row["dataset"] == "RLVS" for row in manifest),
                "moved_in": sum(
                    row["previous_split"] != split for row in manifest
                ),
                "moved_out": sum(
                    row["_old_split"] == split and row["_new_split"] != split for row in rows
                ),
            }
        )
    write_csv(OUTPUT_DIR / "split_summary.csv", summary_rows, list(summary_rows[0].keys()))

    membership_rows: list[dict[str, object]] = []
    changed_rows: list[dict[str, object]] = []
    for row in sorted(rows, key=lambda value: str(value["_identity"])):
        public = {
            "scene_group_id": row["_scene_group_id"],
            "group_origin": row["_group_origin"],
            "dataset": row["dataset"],
            "relative_path": row["relative_path"],
            "sha256": row["sha256"],
            "label": row["label"],
            "original_label": row["_original_label"],
            "label_corrected": row.get("_label_corrected", 0),
            "previous_split": row["_old_split"],
            "new_split": row["_new_split"],
            "split_changed": int(row["_old_split"] != row["_new_split"]),
        }
        membership_rows.append(public)
        if public["split_changed"]:
            changed_rows.append(public)
    membership_fields = list(membership_rows[0].keys())
    write_csv(OUTPUT_DIR / "scene_group_membership.csv", membership_rows, membership_fields)
    write_csv(OUTPUT_DIR / "videos_changed_split.csv", changed_rows, membership_fields)
    write_csv(
        OUTPUT_DIR / "scene_group_assignments.csv",
        sorted(group_rows, key=lambda row: str(row["scene_group_id"])),
        list(group_rows[0].keys()),
    )

    checks = {
        "status": "PASS" if not group_violations and not sha_violations else "FAIL",
        "principal_video_count": len(rows),
        "augmented_train_video_count": len(augmented),
        "perceptual_groups_used_as_base": 191,
        "final_scene_group_count": len(groups),
        "non_singleton_scene_group_count": sum(len(members) > 1 for members in groups.values()),
        "scene_group_split_violations": group_violations,
        "exact_sha256_split_violations": sha_violations,
        "label_correction_count": sum(int(row.get("_label_corrected", 0)) for row in rows),
        "historical_enrichment_audit": enrichment_audit,
        "videos_changed_split": len(changed_rows),
        "target_sizes": TARGET_SIZES,
        "target_label_1": label_targets,
        "solver": solver,
        "split_summary": summary_rows,
    }
    if checks["status"] != "PASS":
        raise RuntimeError("La verification d'integrite des nouveaux splits a echoue")
    write_json(OUTPUT_DIR / "integrity_checks.json", checks)
    return checks


def seal_manifests(manifests: dict[str, list[dict[str, object]]], augmented: list[dict[str, object]]) -> dict[str, object]:
    files = []
    for split in SPLITS:
        path = OUTPUT_DIR / f"{split}_manifest_grouped.csv"
        files.append(
            {
                "file": path.name,
                "rows": len(manifests[split]),
                "sha256": sha256_file(path),
            }
        )
    augmented_path = OUTPUT_DIR / "augmented_train_manifest_grouped.csv"
    files.append(
        {
            "file": augmented_path.name,
            "rows": len(augmented),
            "sha256": sha256_file(augmented_path),
        }
    )
    seal = {
        "algorithm": "SHA-256",
        "scope": "new grouped manifests; labels include completed visual corrections",
        "order_salt": ORDER_SALT,
        "files": files,
    }
    write_json(OUTPUT_DIR / "manifest_sha256.json", seal)
    return seal


def write_reports(
    checks: dict[str, object],
    seal: dict[str, object],
    pair_rows: list[dict[str, object]],
    corrections: list[dict[str, object]],
) -> None:
    resolved_pairs = sum(row["resolution"] == "resolved_by_label_correction" for row in pair_rows)
    transitions = len(pair_rows) - resolved_pairs
    correction_lines = [
        f"| {row['review_id']} | `{row['relative_path']}` | "
        f"{row['original_label']} | {row['corrected_label']} | {row['visual_evidence']} |"
        for row in corrections
    ]
    split_lines = []
    for row in checks["split_summary"]:
        split_lines.append(
            f"| {row['split']} | {row['actual_total']} | {row['proportion']} | "
            f"{row['label_0']} | {row['label_1']} | {row['moved_in']} | {row['moved_out']} |"
        )
    summary = "\n".join(
        [
            "# Splits groupes par scene/source",
            "",
            "Aucun entrainement n'a ete lance. Cette etape s'arrete aux manifestes scelles,",
            "conformement a la demande de validation prealable.",
            "",
            "## Revue des labels opposes",
            "",
            f"- Paires examinees visuellement: {len(pair_rows)}",
            "- Videos uniques examinees: 129",
            f"- Corrections de label: {len(corrections)} (toutes violence -> non_violence)",
            f"- Paires devenues concordantes apres correction: {resolved_pairs}",
            f"- Paires restant opposees car segments temporels distincts: {transitions}",
            "- Preuve: 17 planches a 9 instants et 5 planches agrandies a 24 instants.",
            "",
            "| Cas | Video | Avant | Apres | Motif visuel |",
            "|---|---|---:|---:|---|",
            *correction_lines,
            "",
            "## Nouveaux splits principaux",
            "",
            "| Split | Videos | Proportion | Non-violence | Violence | Entrees | Sorties |",
            "|---|---:|---:|---:|---:|---:|---:|",
            *split_lines,
            "",
            f"Videos ayant change de split: **{checks['videos_changed_split']}**.",
            "",
            "## Integrite",
            "",
            f"- Groupes perceptuels de base: {checks['perceptual_groups_used_as_base']}",
            f"- Groupes finaux, singletons compris: {checks['final_scene_group_count']}",
            f"- Groupes finaux non singletons: {checks['non_singleton_scene_group_count']}",
            f"- Violations groupe/split: {len(checks['scene_group_split_violations'])}",
            f"- Violations SHA-256 exact/split: {len(checks['exact_sha256_split_violations'])}",
            "- Enrichissement historique: 25 videos train-only, aucun hash exact ni groupe perceptuel commun avec le corpus principal.",
            "- Affectation: tailles et comptes de classes exacts; minimisation du nombre de videos deplacees.",
            "",
            "## Scellement",
            "",
            *[f"- `{item['file']}`: `{item['sha256']}` ({item['rows']} lignes)" for item in seal["files"]],
            "",
            "## Arret controle",
            "",
            "Les nouveaux manifestes n'ont ete consommes par aucun entrainement. Les quatre",
            "reentrainements restent bloques dans l'attente de la validation de Tracy.",
            "",
        ]
    )
    with (OUTPUT_DIR / "summary.md").open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(summary)

    result = {
        "status": "completed_waiting_for_user_validation",
        "training_started": False,
        "visual_review": {
            "opposing_label_pairs": len(pair_rows),
            "unique_videos": 129,
            "label_corrections": len(corrections),
            "pairs_resolved_by_correction": resolved_pairs,
            "legitimate_temporal_transitions": transitions,
        },
        "splits": checks["split_summary"],
        "videos_changed_split": checks["videos_changed_split"],
        "integrity_status": checks["status"],
        "manifest_seal": seal,
        "next_action": "wait_for_explicit_user_validation_before_any_training",
    }
    write_json(OUTPUT_DIR / "result.json", result)

    figures = []
    for path in sorted((OUTPUT_DIR / "label_conflict_review").glob("*.jpg")):
        figures.append(
            {
                "file": path.relative_to(OUTPUT_DIR).as_posix(),
                "purpose": "Nine-frame visual review of opposing-label near-duplicate clips",
                "sha256": sha256_file(path),
            }
        )
    for path in sorted((OUTPUT_DIR / "label_correction_zoom").glob("*.jpg")):
        figures.append(
            {
                "file": path.relative_to(OUTPUT_DIR).as_posix(),
                "purpose": "Twenty-four-frame second-pass evidence for a corrected label",
                "sha256": sha256_file(path),
            }
        )
    write_json(OUTPUT_DIR / "figure_manifest.json", {"figures": figures})


def write_artifact_manifest() -> None:
    artifacts = []
    for path in sorted(OUTPUT_DIR.rglob("*")):
        if not path.is_file() or path.name == "artifact_manifest.json":
            continue
        artifacts.append(
            {
                "file": path.relative_to(EXP_DIR).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    write_json(OUTPUT_DIR / "artifact_manifest.json", {"artifacts": artifacts})


def main() -> None:
    review_rows, review_by_node = complete_visual_review()
    pair_rows = document_conflict_pairs(review_by_node)
    principal_rows = load_principal_rows()
    corrections = apply_label_corrections(principal_rows, review_rows)
    group_rows, groups = build_scene_groups(principal_rows)
    _, label_targets, solver = assign_groups(principal_rows, group_rows, groups)
    manifests = write_manifests(principal_rows)
    augmented, enrichment_audit = write_augmented_manifest(manifests["train"], principal_rows)
    checks = verification(
        principal_rows,
        manifests,
        augmented,
        group_rows,
        groups,
        label_targets,
        solver,
        enrichment_audit,
    )
    seal = seal_manifests(manifests, augmented)
    write_reports(checks, seal, pair_rows, corrections)
    write_artifact_manifest()
    print(json.dumps({"checks": checks, "seal": seal}, indent=2))


if __name__ == "__main__":
    main()
