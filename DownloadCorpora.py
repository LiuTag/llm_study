"""下载 CS336 官方指定的 TinyStories V2 和 OpenWebText 子集原始文件。"""

from __future__ import annotations

import argparse
import gzip
import time
from pathlib import Path
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parent
FILES = {
    "tinystories": (
        ("TinyStoriesV2-GPT4-train.txt", "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt"),
        ("TinyStoriesV2-GPT4-valid.txt", "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt"),
    ),
    "owt": (
        ("owt_train.txt.gz", "https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz"),
        ("owt_valid.txt.gz", "https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz"),
    ),
}


def download(url: str, destination: Path) -> None:
    if destination.is_file():
        print(f"已存在，跳过: {destination}", flush=True)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    for attempt in range(1, 6):
        try:
            download_once(url, partial, destination)
            return
        except OSError as error:
            if attempt == 5:
                raise
            print(f"连接中断（第 {attempt} 次），保留断点稍后重试: {error}", flush=True)
            time.sleep(5)


def download_once(url: str, partial: Path, destination: Path) -> None:
    existing = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": "mini-gpt-data-prep/1.0", "Accept-Encoding": "identity"}
    if existing:
        headers["Range"] = f"bytes={existing}-"
    with urlopen(Request(url, headers=headers), timeout=120) as response:
        resumed = response.status == 206
        if existing and not resumed:
            raise IOError(f"服务器未支持断点续传，未覆盖已有文件: {partial}")
        expected = response.headers.get("Content-Length")
        expected_remaining = int(expected) if expected is not None else None
        transferred = 0
        next_report = 100 * 1024 * 1024
        with partial.open("ab" if resumed else "wb") as output:
            while chunk := response.read(8 * 1024 * 1024):
                output.write(chunk)
                transferred += len(chunk)
                if transferred >= next_report:
                    print(f"{destination.name}: {existing + transferred:,} bytes", flush=True)
                    next_report += 100 * 1024 * 1024
    if expected_remaining is not None and transferred != expected_remaining:
        raise IOError(f"下载不完整: {destination.name}，期望 {expected_remaining}，实际 {transferred}")
    partial.replace(destination)
    print(f"下载完成: {destination} ({destination.stat().st_size:,} bytes)", flush=True)


def extract_gzip(source: Path) -> None:
    destination = source.with_suffix("")
    if destination.is_file():
        print(f"已存在，跳过解压: {destination}", flush=True)
        return
    partial = destination.with_name(destination.name + ".part")
    with gzip.open(source, "rb") as compressed, partial.open("wb") as output:
        while chunk := compressed.read(8 * 1024 * 1024):
            output.write(chunk)
    partial.replace(destination)
    print(f"解压完成: {destination} ({destination.stat().st_size:,} bytes)", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=("tinystories", "owt", "all"))
    parser.add_argument("--extract-owt", action="store_true", help="下载后解压 OWT .gz 文件")
    args = parser.parse_args()
    selected = FILES if args.dataset == "all" else {args.dataset: FILES[args.dataset]}
    for dataset, entries in selected.items():
        for filename, url in entries:
            destination = ROOT / "data" / "raw" / dataset / filename
            download(url, destination)
            if dataset == "owt" and args.extract_owt:
                extract_gzip(destination)


if __name__ == "__main__":
    main()
