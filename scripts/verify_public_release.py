"""Verify the frozen public-release artifacts against their SHA-256 manifest.

Run from any working directory with:
    python scripts/verify_public_release.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RELEASE_DIR = ROOT / "reports" / "final-release"
MANIFEST_PATH = RELEASE_DIR / "release-manifest.json"
ARTIFACT_PATHS = {
    "pdf": ROOT / "reports",
    "presentation": ROOT / "reports",
    "phase3e_result": RELEASE_DIR,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    failures: list[str] = []

    for key, directory in ARTIFACT_PATHS.items():
        record = manifest["files"][key]
        path = directory / record["name"]
        if not path.is_file():
            failures.append(f"MISSING {path.relative_to(ROOT)}")
            continue
        if sha256(path) != record["sha256"]:
            failures.append(f"MISMATCH {path.relative_to(ROOT)}")
            continue
        print(f"OK   {path.relative_to(ROOT)}")

    if failures:
        print("\nVerification failed:")
        print("\n".join(failures))
        return 1

    print("\nPASS: frozen public-release manifest matches all listed artifacts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
