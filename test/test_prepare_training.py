"""迁移数据准备工具验收；可直接执行本文件。"""

import _paths  # noqa: F401
import json

import numpy as np
import pytest
import tiktoken

from PrepareTraining import load_gpt2_tokenizer, prepare_pair


def test_gpt2_fixture_loader_matches_reference() -> None:
    tokenizer = load_gpt2_tokenizer()
    reference = tiktoken.get_encoding("gpt2")
    text = "Once upon a time, 猫\n\n<|endoftext|> Next story!"
    assert len(tokenizer.vocab) == 50257
    assert tokenizer.encode(text) == reference.encode(text, allowed_special={"<|endoftext|>"})


def test_prepare_pair_writes_compatible_bins(tmp_path) -> None:
    train_text = tmp_path / "train.txt"
    val_text = tmp_path / "val.txt"
    train_text.write_text("A little cat.\n<|endoftext|>\n", encoding="utf-8")
    val_text.write_text("A little dog.\n<|endoftext|>\n", encoding="utf-8")
    output_dir = tmp_path / "tokens"

    info = prepare_pair(train_text, val_text, output_dir)
    reference = tiktoken.get_encoding("gpt2")
    for split, text_path in (("train", train_text), ("val", val_text)):
        expected = reference.encode(text_path.read_text(encoding="utf-8"), allowed_special={"<|endoftext|>"})
        actual = np.memmap(output_dir / f"{split}.bin", mode="r", dtype="uint16")
        np.testing.assert_array_equal(actual, expected)
        assert info[f"{split}_tokens"] == len(expected)
    assert json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))["vocab_size"] == 50257


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
