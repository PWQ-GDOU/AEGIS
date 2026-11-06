"""Normalize persisted path references without touching binary artifacts.

Local checkout paths become repository-relative. Paths that refer to artifacts
left on a compute server are retained under the explicit relative namespace
``remote_artifacts/`` so provenance remains visible without hard-coding hosts.
"""
from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TEXT_EXTENSIONS = {
    ".csv", ".json", ".md", ".ps1", ".py", ".sh", ".sha256", ".sum", ".tex", ".txt",
    ".yaml", ".yml", ".xml",
}
SKIP_PARTS = {".git", ".paseo-local", ".zcode", "__pycache__", ".pytest_cache", ".ruff_cache"}
SKIP_NAMES = {"normalize_repo_paths.py"}


def normalize(text: str) -> str:
    # Current H-drive checkout references.
    for prefix in (
        r"H:\项目代码\water_turbine_step30_chatgpt_handoff_20260810",
        r"H:\\项目代码\\water_turbine_step30_chatgpt_handoff_20260810",
        "H:/项目代码/water_turbine_step30_chatgpt_handoff_20260810",
    ):
        text = re.sub(re.escape(prefix) + r"[\\/]*", "", text, flags=re.IGNORECASE)
    # A few legacy results were written while the Windows console was using a
    # different code page. Their Chinese directory segment is mojibake, but
    # the repository-name suffix remains reliable.
    text = re.sub(
        r"(?i)H:(?:\\\\|\\|/).*?water_turbine_step30_chatgpt_handoff_20260810(?:\\\\|\\|/)*",
        "",
        text,
    )

    # Legacy local checkout references already represented in this repository.
    for old_root in (
        r"B:\undertaking\学业\非专业相关\比赛\挑战杯\水轮机",
        r"B:\\undertaking\\学业\\非专业相关\\比赛\\挑战杯\\水轮机",
        "B:/undertaking/学业/非专业相关/比赛/挑战杯/水轮机",
    ):
        escaped = re.escape(old_root) + r"[\\/]*"
        text = re.sub(escaped + r"workspace[\\/]", "workspace/", text, flags=re.IGNORECASE)
        text = re.sub(escaped + r"DecPelton2024[\\/]", "DecPelton2024/", text, flags=re.IGNORECASE)
        text = re.sub(escaped + r"DecPelton2026[\\/]", "DecPelton2026/", text, flags=re.IGNORECASE)
        text = re.sub(escaped, "./", text, flags=re.IGNORECASE)
    text = text.replace(
        "/mnt/b/undertaking/学业/非专业相关/比赛/挑战杯/水轮机/workspace/",
        "workspace/",
    )

    # Server-only provenance remains explicit but portable.
    text = re.sub(r"/public/home/wenquan/PWQ/", "remote_artifacts/PWQ/", text)
    text = re.sub(r"/public/home/wenquan/", "remote_artifacts/wenquan/", text)
    text = re.sub(r"/home/user/", "remote_artifacts/user/", text)
    # Historical drive-qualified references are provenance only; keep them portable.
    text = re.sub(r"(?i)C:[\\/]+Users[\\/]+20132[\\/]*", "historical_external/users/20132/", text)
    text = re.sub(r"(?i)B:[\\/]+undertaking[\\/]+", "historical_external/undertaking/", text)
    text = re.sub(r"(?i)H:[\\/]+项目代码[\\/]+", "external_workspaces/", text)
    return text


def should_scan(path: Path) -> bool:
    return (
        path.suffix.lower() in TEXT_EXTENSIONS
        and not any(part in SKIP_PARTS for part in path.parts)
    )


def main() -> None:
    changed = 0
    for path in ROOT.rglob("*"):
        if (
            not path.is_file()
            or path.name in SKIP_NAMES
            or path.name.startswith("test_")
            or not should_scan(path)
        ):
            continue
        try:
            original = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        updated = normalize(original)
        if updated != original:
            path.write_text(updated, encoding="utf-8", newline="")
            changed += 1
    print(f"normalized_files={changed}")


if __name__ == "__main__":
    main()
