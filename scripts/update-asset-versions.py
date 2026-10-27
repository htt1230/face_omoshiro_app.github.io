"""Refresh local asset URL versions from their content. No build dependencies."""

import argparse
import hashlib
from pathlib import Path
import re


def update_versions(root: Path, check: bool = False) -> bool:
    html_path = root / "index.html"
    original = html_path.read_text(encoding="utf-8")
    updated = original
    for filename, attribute in (("app.js", "src"), ("style.css", "href")):
        version = hashlib.sha256((root / filename).read_bytes()).hexdigest()[:16]
        pattern = rf'{attribute}="\./{re.escape(filename)}(?:\?[^"\s]*)?"'
        updated, count = re.subn(pattern, f'{attribute}="./{filename}?v={version}"', updated)
        if count != 1:
            raise ValueError(f"Expected exactly one {filename} reference, found {count}")
    if updated == original:
        print("Asset versions are current.")
        return True
    if check:
        print("Asset versions are stale. Run: python3 scripts/update-asset-versions.py")
        return False
    html_path.write_text(updated, encoding="utf-8")
    print("Updated asset versions in index.html.")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Check without writing")
    args = parser.parse_args()
    raise SystemExit(0 if update_versions(Path(__file__).resolve().parents[1], args.check) else 1)
