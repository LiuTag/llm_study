"""文本语料转原始 uint16 token 文件的验收测试；可直接运行。"""

import _paths  # noqa: F401
import importlib

import numpy as np
import pytest

from BPE import BPETokenizer
from TrainingData import load_token_ids


def get_converter():
    module = importlib.import_module("TokenizeData")
    converter = getattr(module, "tokenize_text_file", None)
    assert callable(converter), "请在 TokenizeData.py 中实现 tokenize_text_file"
    return converter


def make_tokenizer(special_id: int = 256):
    vocab = {i: bytes([i]) for i in range(256)}
    vocab[special_id] = b"<|endoftext|>"
    return BPETokenizer(vocab, [], ["<|endoftext|>"])


def test_text_to_raw_uint16_matches_tokenizer_and_loader(tmp_path):
    text = "Hi 猫.\n<|endoftext|>\nHello!"
    input_path = tmp_path / "corpus.txt"
    output_path = tmp_path / "tokens.bin"
    input_path.write_text(text, encoding="utf-8")
    tokenizer = make_tokenizer()

    count = get_converter()(input_path, output_path, tokenizer)

    expected = tokenizer.encode(text)
    assert isinstance(count, int)
    assert count == len(expected)
    assert output_path.is_file()
    assert output_path.stat().st_size == 2 * count
    np.testing.assert_array_equal(load_token_ids(output_path), expected)


def test_rejects_token_ids_outside_uint16_range(tmp_path):
    input_path = tmp_path / "corpus.txt"
    output_path = tmp_path / "tokens.bin"
    input_path.write_text("<|endoftext|>", encoding="utf-8")
    tokenizer = make_tokenizer(special_id=65536)

    with pytest.raises(ValueError):
        get_converter()(input_path, output_path, tokenizer)


def test_line_stream_matches_whole_text_across_blank_lines(tmp_path):
    text = "a\n\n b"
    input_path = tmp_path / "corpus.txt"
    output_path = tmp_path / "tokens.bin"
    input_path.write_text(text, encoding="utf-8")
    vocab = {i: bytes([i]) for i in range(256)}
    vocab[256] = b"\n\n"
    tokenizer = BPETokenizer(vocab, [(b"\n", b"\n")], None)

    count = get_converter()(input_path, output_path, tokenizer)

    expected = tokenizer.encode(text)
    assert count == len(expected)
    np.testing.assert_array_equal(load_token_ids(output_path), expected)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
