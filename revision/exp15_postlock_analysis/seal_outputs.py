from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
DESTINATION = OUT / "artifact_manifest.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    paths = sorted(
        path for path in HERE.rglob("*")
        if path.is_file() and path != DESTINATION and "__pycache__" not in path.parts
    )
    payload = {
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "exp15_postlock_analysis",
        "artifacts": [
            {"path": path.relative_to(HERE).as_posix(), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in paths
        ],
    }
    with DESTINATION.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    print(f"Sealed {len(paths)} files: {DESTINATION}")


if __name__ == "__main__":
    main()
