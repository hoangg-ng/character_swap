"""Integrity check for the vendored contract.

``contracts/`` is duplicated in two repos, so nothing stops someone editing one copy and
forgetting the other. This records a sha256 per file in ``checksum.txt``; the backend test
suite in both repos verifies it, so an edit that was not re-synced fails CI.

What this catches: an edit in one repo that was never checksummed or copied across.
What it cannot catch on its own: someone regenerating the checksum in both repos with
different content. ``scripts/check-contracts.sh --against <other-repo>`` compares two
checkouts directly and is what CI runs when both are available.

    python contracts/python/checksum.py            # verify, exit 1 on drift
    python contracts/python/checksum.py --write    # regenerate after an intended change
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

CONTRACTS_DIR = Path(__file__).resolve().parent.parent
MANIFEST = CONTRACTS_DIR / "checksum.txt"
MANIFEST_NAME = MANIFEST.name

#: Not part of the contract: caches and the manifest's own line.
EXCLUDED_DIRS = {"__pycache__", ".pytest_cache", "node_modules"}
EXCLUDED_SUFFIXES = {".pyc"}


def contract_files() -> list[Path]:
    """Every contract file, in a stable order, relative to ``contracts/``."""
    files = [
        path
        for path in CONTRACTS_DIR.rglob("*")
        if path.is_file()
        and path.name != MANIFEST_NAME
        and path.suffix not in EXCLUDED_SUFFIXES
        and not EXCLUDED_DIRS & set(path.relative_to(CONTRACTS_DIR).parts)
    ]
    return sorted(files, key=lambda p: p.relative_to(CONTRACTS_DIR).as_posix())


def file_digest(path: Path) -> str:
    """Hash file bytes with CRLF normalised, so a Windows checkout is not "drift"."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def manifest_lines() -> list[str]:
    return [
        f"{file_digest(path)}  {path.relative_to(CONTRACTS_DIR).as_posix()}"
        for path in contract_files()
    ]


def manifest_text() -> str:
    lines = manifest_lines()
    combined = hashlib.sha256("\n".join(lines).encode()).hexdigest()
    header = [
        "# sha256 of every file in contracts/ except this one. Regenerate with:",
        "#   python contracts/python/checksum.py --write",
        f"# contract: {(CONTRACTS_DIR / 'VERSION').read_text().strip()}",
        f"# combined: {combined}",
    ]
    return "\n".join([*header, *lines]) + "\n"


def combined_digest() -> str:
    """One hash standing for the whole contract; what two repos compare."""
    return hashlib.sha256("\n".join(manifest_lines()).encode()).hexdigest()


def verify() -> list[str]:
    """Return a list of human-readable problems; empty means the copy is intact."""
    if not MANIFEST.exists():
        return [f"{MANIFEST_NAME} is missing; run with --write to create it"]

    recorded = {
        name: digest
        for digest, _, name in (
            line.partition("  ")
            for line in MANIFEST.read_text().splitlines()
            if line and not line.startswith("#")
        )
        if name
    }
    actual = {
        path.relative_to(CONTRACTS_DIR).as_posix(): file_digest(path) for path in contract_files()
    }

    problems = [
        f"not in {MANIFEST_NAME} (new file?): {name}"
        for name in sorted(actual.keys() - recorded.keys())
    ]
    problems += [f"missing from contracts/: {name}" for name in sorted(recorded.keys() - actual.keys())]
    problems += [
        f"changed since last checksum: {name}"
        for name in sorted(recorded.keys() & actual.keys())
        if recorded[name] != actual[name]
    ]
    return problems


def main(argv: list[str]) -> int:
    if "--write" in argv:
        MANIFEST.write_text(manifest_text())
        print(f"wrote {MANIFEST_NAME} ({len(contract_files())} files)")
        print(f"combined: {combined_digest()}")
        return 0

    if "--print-combined" in argv:
        print(combined_digest())
        return 0

    problems = verify()
    if problems:
        print("contract drift detected:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        print(
            "\nIf the change was intended: regenerate with --write, then copy contracts/ to the "
            "other repo with scripts/sync-contracts.sh and commit in both.",
            file=sys.stderr,
        )
        return 1

    print(f"contracts/ intact ({len(contract_files())} files, combined {combined_digest()[:12]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
