"""把文本语料转换为本项目训练循环使用的 uint16 token 文件。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from BPE import BPETokenizer
from TokenizeData import tokenize_text_file


ROOT = Path(__file__).resolve().parent
FIXTURES = ROOT / "assignment1-basics" / "tests" / "fixtures"
END_OF_TEXT = "<|endoftext|>"


def gpt2_byte_decoder() -> dict[str, int]:
    visible = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    codepoints = visible[:]
    shifted = 0
    for value in range(256):
        if value not in visible:
            codepoints.append(256 + shifted)
            visible.append(value)
            shifted += 1
    return {chr(codepoint): value for value, codepoint in zip(visible, codepoints)}


def load_gpt2_tokenizer() -> BPETokenizer:
    """从随项目携带的官方 GPT-2 词表和 merges 构造本项目的分词器。"""
    decoder = gpt2_byte_decoder()

    def original_bytes(encoded: str) -> bytes:
        return bytes(decoder[character] for character in encoded)

    vocab_json = json.loads((FIXTURES / "gpt2_vocab.json").read_text(encoding="utf-8"))
    vocab = {int(token_id): original_bytes(token) for token, token_id in vocab_json.items()}
    merges: list[tuple[bytes, bytes]] = []
    with (FIXTURES / "gpt2_merges.txt").open(encoding="utf-8") as source:
        for line in source:
            parts = line.strip().split()
            if len(parts) == 2:
                merges.append((original_bytes(parts[0]), original_bytes(parts[1])))
    return BPETokenizer(vocab, merges, [END_OF_TEXT])


def prepare_pair(train_text: Path, val_text: Path, output_dir: Path) -> dict[str, object]:
    for path in (train_text, val_text):
        if not path.is_file():
            raise FileNotFoundError(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = load_gpt2_tokenizer()
    train_bin = output_dir / "train.bin"
    val_bin = output_dir / "val.bin"
    train_count = tokenize_text_file(train_text, train_bin, tokenizer)
    val_count = tokenize_text_file(val_text, val_bin, tokenizer)
    info: dict[str, object] = {
        "tokenizer": "GPT-2 fixture vocab/merges; project BPETokenizer",
        "vocab_size": len(tokenizer.vocab),
        "dtype": "uint16 little-endian",
        "train_text": str(train_text),
        "val_text": str(val_text),
        "train_tokens": train_count,
        "val_tokens": val_count,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return info


def make_smoke_texts(output_dir: Path) -> tuple[Path, Path]:
    """从随仓库携带的小样本中取不重叠故事，仅用于流程验证。"""
    sample = (FIXTURES / "tinystories_sample_5M.txt").read_text(encoding="utf-8")
    stories = [part + END_OF_TEXT for part in sample.split(END_OF_TEXT)[:80]]
    if len(stories) < 80:
        raise ValueError("TinyStories 小样本不足 80 个故事")
    output_dir.mkdir(parents=True, exist_ok=True)
    train_text = output_dir / "train.txt"
    val_text = output_dir / "val.txt"
    train_text.write_text("\n".join(stories[:64]) + "\n", encoding="utf-8")
    val_text.write_text("\n".join(stories[64:]) + "\n", encoding="utf-8")
    return train_text, val_text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="用内置 TinyStories 小样本生成冒烟数据")
    parser.add_argument("--train-text", type=Path, help="UTF-8 训练文本")
    parser.add_argument("--val-text", type=Path, help="UTF-8 验证文本")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.smoke:
        if args.train_text or args.val_text:
            parser.error("--smoke 不能与 --train-text / --val-text 同时使用")
        train_text, val_text = make_smoke_texts(args.output_dir)
    else:
        if not args.train_text or not args.val_text:
            parser.error("必须同时提供 --train-text 和 --val-text")
        train_text, val_text = args.train_text, args.val_text
    print(json.dumps(prepare_pair(train_text, val_text, args.output_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
