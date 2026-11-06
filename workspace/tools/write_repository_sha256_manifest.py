"""Write a portable SHA-256 manifest for repository deliverables."""
from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "REPOSITORY_FILE_MANIFEST.sha256"
SKIP_PARTS = {".git", ".paseo-local", ".zcode", "__pycache__", ".pytest_cache", ".ruff_cache", "lo_profile", "rendered"}
SKIP_NAMES = {MANIFEST.name}


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def main() -> None:
    records = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.name in SKIP_NAMES:
            continue
        if any(part in SKIP_PARTS for part in path.relative_to(ROOT).parts):
            continue
        records.append(f"{digest(path)}  {path.relative_to(ROOT).as_posix()}")
    MANIFEST.write_text("\n".join(sorted(records)) + "\n", encoding="utf-8")
    print(f"manifest_files={len(records)}")


if __name__ == "__main__":
    main()
