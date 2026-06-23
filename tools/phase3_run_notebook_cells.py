"""Run selected code cells from SIRCH Phase 3 notebooks.

The helper executes existing notebook cells without editing the notebook. It can
skip cells containing marker strings, which lets us rerun evaluation cells
without rerunning a completed training cell.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--notebook",
        action="append",
        required=True,
        help="Notebook path. Can be passed multiple times.",
    )
    parser.add_argument(
        "--skip",
        action="append",
        default=[],
        help="Marker text. Code cells containing it are skipped.",
    )
    parser.add_argument(
        "--clear-skips-between-notebooks",
        action="store_true",
        help="Apply skips only to the first notebook.",
    )
    return parser.parse_args()


def run_notebook_cells(notebook_path: Path, skip_markers: tuple[str, ...]) -> None:
    print(f"[SIRCH] Notebook: {notebook_path}", flush=True)
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    namespace = {"__name__": "__main__", "__file__": str(notebook_path)}

    for index, cell in enumerate(notebook.get("cells", []), start=1):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if any(marker in source for marker in skip_markers):
            print(f"[SIRCH] Cellule {index} ignoree volontairement.", flush=True)
            continue
        print(f"[SIRCH] Execution cellule {index}...", flush=True)
        exec(compile(source, f"{notebook_path}#cell-{index}", "exec"), namespace)
        print(f"[SIRCH] Cellule {index} terminee.", flush=True)


def main() -> int:
    args = parse_args()
    skip_markers = tuple(args.skip)
    for position, notebook in enumerate(args.notebook):
        active_skips = skip_markers
        if args.clear_skips_between_notebooks and position > 0:
            active_skips = ()
        run_notebook_cells(Path(notebook).resolve(), active_skips)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
