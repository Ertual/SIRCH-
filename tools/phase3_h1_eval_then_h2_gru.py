"""Evaluate Phase 3 H1, then run Phase 3 H2 GRU training.

This helper keeps the original notebooks as the source of truth.  H1 is
already trained, so we execute its setup/model/evaluation cells but skip the
training cell that reloads the epoch checkpoint.  The saved H1 model is loaded
as weights into the rebuilt architecture, which avoids old/new Keras H5
deserialization differences.
"""

from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
H1_NOTEBOOK = PROJECT_ROOT / "colab" / "train_efficientnet_only.ipynb"
H2_NOTEBOOK = PROJECT_ROOT / "colab" / "train_gru.ipynb"


def iter_code_cells(notebook_path: Path):
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    for index, cell in enumerate(notebook.get("cells", []), start=1):
        if cell.get("cell_type") != "code":
            continue
        yield index, "".join(cell.get("source", []))


def exec_cell(source: str, notebook_path: Path, index: int, namespace: dict) -> None:
    exec(compile(source, f"{notebook_path}#cell-{index}", "exec"), namespace)


def evaluate_h1() -> None:
    print(f"[SIRCH] Evaluation H1 depuis {H1_NOTEBOOK}", flush=True)
    namespace = {"__name__": "__main__", "__file__": str(H1_NOTEBOOK)}

    for index, source in iter_code_cells(H1_NOTEBOOK):
        if "CELLULE 11 - Entrainement avec reprise automatique" in source:
            print("[SIRCH] Cellule H1 entrainement ignoree: H1 est deja termine.", flush=True)
            continue

        if "CELLULE 12 - Evaluation obligatoire" in source:
            model = namespace["model"]
            model_path = namespace["LOCAL_MODEL_OUTPUT"]
            print(f"[SIRCH] Chargement compatible des poids H1: {model_path}", flush=True)
            model.load_weights(model_path)

        print(f"[SIRCH] H1 cellule {index}...", flush=True)
        exec_cell(source, H1_NOTEBOOK, index, namespace)
        print(f"[SIRCH] H1 cellule {index} terminee.", flush=True)

    print("[SIRCH] Evaluation H1 terminee.", flush=True)


def run_h2_gru() -> None:
    print(f"[SIRCH] Lancement H2 GRU depuis {H2_NOTEBOOK}", flush=True)
    namespace = {"__name__": "__main__", "__file__": str(H2_NOTEBOOK)}

    for index, source in iter_code_cells(H2_NOTEBOOK):
        print(f"[SIRCH] H2 cellule {index}...", flush=True)
        exec_cell(source, H2_NOTEBOOK, index, namespace)
        print(f"[SIRCH] H2 cellule {index} terminee.", flush=True)

    print("[SIRCH] H2 GRU termine.", flush=True)


def main() -> int:
    evaluate_h1()
    run_h2_gru()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
