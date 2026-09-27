"""创建或核对迁移文件的 SHA-256 清单（不包含缓存和训练输出）。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "transfer_manifest.json"
SKIP_DIRS = {".git", ".idea", ".pytest_cache", ".venv", "__pycache__", "runs"}
SKIP_NAMES = {MANIFEST.name}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def files_to_transfer():
    for current, directories, filenames in os.walk(ROOT):
        directories[:] = sorted(name for name in directories if name not in SKIP_DIRS)
        for filename in sorted(filenames):
            if filename in SKIP_NAMES or filename.endswith((".pyc", ".part")):
                continue
            path = Path(current) / filename
            yield path.relative_to(ROOT).as_posix(), path


def create() -> None:
    records = []
    for relative, path in files_to_transfer():
        records.append({"path": relative, "bytes": path.stat().st_size, "sha256": digest(path)})
        print(f"已记录: {relative}", flush=True)
    MANIFEST.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"清单: {MANIFEST}，共 {len(records)} 个文件", flush=True)


def verify() -> None:
    records = json.loads(MANIFEST.read_text(encoding="utf-8"))
    expected = {record["path"]: record for record in records}
    issues = []
    for relative, record in expected.items():
        path = ROOT / relative
        if not path.is_file():
            issues.append(f"缺失: {relative}")
        elif path.stat().st_size != record["bytes"] or digest(path) != record["sha256"]:
            issues.append(f"内容不同: {relative}")
    if issues:
        raise SystemExit("\n".join(issues))
    print(f"核对通过：{len(records)} 个文件", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("create", "verify"))
    args = parser.parse_args()
    if args.action == "create":
        create()
    else:
        verify()


if __name__ == "__main__":
    main()
