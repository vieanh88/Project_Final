#!/usr/bin/env python3
"""
change_path_vivoicetxt.py

Replace Windows-style backslashes "\" with Linux/POSIX slashes "/" inside one
or more text files.

Example:
    python change_path_vivoicetxt.py \
      /workspace/Project_Final/TTS_StyleTTS2/fine-tune/data_pipeline/prepare_vivoice/output/wavs_to_delete.txt

Multiple files:
    python change_path_vivoicetxt.py file1.txt file2.txt

Dry-run:
    python change_path_vivoicetxt.py file.txt --dry-run
"""

from __future__ import annotations

import argparse
from pathlib import Path
from datetime import datetime


def make_backup_path(path: Path, suffix: str = ".bak_windows_paths") -> Path:
    """
    Return a backup path without overwriting an existing backup.

    For example:
      data.txt -> data.txt.bak_windows_paths
      if exists -> data.txt.bak_windows_paths.20260520_183000
    """
    backup = path.with_name(path.name + suffix)
    if not backup.exists():
        return backup

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return path.with_name(path.name + suffix + "." + timestamp)


def fix_file(
    file_path: Path,
    *,
    encoding: str = "utf-8",
    backup: bool = True,
    dry_run: bool = False,
) -> bool:
    """
    Replace all "\\" characters with "/" in file_path.

    Returns:
        True if file content needed changes, False otherwise.
    """
    if not file_path.exists():
        print(f"SKIP: not found: {file_path}")
        return False

    if not file_path.is_file():
        print(f"SKIP: not a file: {file_path}")
        return False

    text = file_path.read_text(encoding=encoding, errors="replace")
    num_backslashes = text.count("\\")

    if num_backslashes == 0:
        print(f"OK: no backslash found: {file_path}")
        return False

    new_text = text.replace("\\", "/")

    print(f"FOUND: {num_backslashes} backslash(es): {file_path}")

    if dry_run:
        print(f"DRY-RUN: no file written: {file_path}")
        return True

    if backup:
        backup_path = make_backup_path(file_path)
        backup_path.write_text(text, encoding=encoding)
        print(f"BACKUP: {backup_path}")

    file_path.write_text(new_text, encoding=encoding)
    print(f"FIXED : {file_path}")
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Replace Windows-style backslashes "\\" with Linux/POSIX slashes "/" '
            "inside one or more text files."
        )
    )
    parser.add_argument(
        "files",
        nargs="+",
        type=Path,
        help=(
            "Path(s) to text file(s) to fix, for example: "
            "/workspace/Project_Final/TTS_StyleTTS2/fine-tune/"
            "data_pipeline/prepare_vivoice/output/wavs_to_delete.txt"
        ),
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not create .bak_windows_paths backup file before overwriting.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report what would be changed; do not write anything.",
    )
    parser.add_argument(
        "--encoding",
        default="utf-8",
        help="File encoding. Default: utf-8",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    changed = 0
    for file_path in args.files:
        if fix_file(
            file_path,
            encoding=args.encoding,
            backup=not args.no_backup,
            dry_run=args.dry_run,
        ):
            changed += 1

    print("-" * 80)
    print(f"DONE: {changed}/{len(args.files)} file(s) needed changes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
