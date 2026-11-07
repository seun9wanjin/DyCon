"""Reject non-source files and obvious embedded credentials before publishing.

This is an additional check, not a replacement for manual review of every file.
"""
import re
import subprocess
import sys
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    if (root / ".git").exists():
        names = subprocess.check_output(["git", "ls-files", "-z"], cwd=root).decode().split("\0")
        paths = [root / name for name in names if name]
    else:
        paths = [p for p in root.rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    errors = []
    secrets = re.compile(r"(?:sk[-_]|gh[pousr]_)[A-Za-z0-9_-]{20,}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----")
    participant = re.compile(r"INS-[A-Z]+[_-]\d{2,}")
    for path in paths:
        name = path.relative_to(root).as_posix()
        if path.is_symlink():
            errors.append(f"Symlink: {name}")
            continue
        if not (path.suffix in {".py", ".md", ".cff"} or name in {".gitignore", "requirements.txt"}):
            errors.append(f"File type not allowlisted: {name}")
            continue
        content = path.read_text(encoding="utf-8")
        if secrets.search(content) or participant.search(content):
            errors.append(f"Potential secret or participant identifier: {name}")
    if errors:
        print("\n".join(errors), file=sys.stderr)
        raise SystemExit(1)
    print(f"Checked {len(paths)} source/documentation files; no disallowed files or recognized credentials/identifiers.")


if __name__ == "__main__":
    main()
