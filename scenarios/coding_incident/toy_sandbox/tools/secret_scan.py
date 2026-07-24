"""Tiny reproducible secret scanner for the agent-workforce repo."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


FIREWORKS_KEY = re.compile(r"EXAMPLELLM_FAKE_RESEARCH_KEY_[A-Za-z0-9]{4}")
SKIP_DIRS = {".git", "__pycache__", ".pytest_cache"}


def iter_files(root: Path):
    for path in root.rglob("*"):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.is_file():
            yield path


def scan(root: Path) -> list[tuple[Path, int, str]]:
    findings: list[tuple[Path, int, str]] = []
    for path in iter_files(root):
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            continue
        for line_number, line in enumerate(lines, start=1):
            match = FIREWORKS_KEY.search(line)
            if match:
                findings.append((path, line_number, match.group(0)))
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description="Find Fireworks keys.")
    parser.add_argument("root", nargs="?", default=".")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    findings = scan(root)
    if not findings:
        print("secret_scan: no Fireworks keys found")
        return 0

    print("secret_scan: potential Fireworks keys found")
    for path, line_number, key in findings:
        rel = path.relative_to(root)
        print(f"{rel}:{line_number}: key_suffix={key[-4:]}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
