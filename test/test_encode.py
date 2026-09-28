import _paths  # noqa: F401
import pytest

from BPE import BPETokenizer, train_bpe_from_counts


def test_encode_without_special_tokens():
    pretoken_frequencies = {
        (97, 98, 97, 98): 2,  # abab
        (97, 98, 97, 99): 1,  # abac
    }

    vocab, merges = train_bpe_from_counts(
        pretoken_frequencies=pretoken_frequencies,
        vocab_size=258,
    )

    tokenizer = BPETokenizer(
        vocab=vocab,
        merges=merges,
        special_tokens=None,
    )

    assert vocab[256] == b"ab"
    assert vocab[257] == b"abab"

    assert tokenizer.encode("abab") == [257]
    assert tokenizer.encode("abac") == [256, 97, 99]
    assert tokenizer.encode("你") == [228, 189, 160]
    assert tokenizer.encode("") == []


def test_encode_with_special_token():
    special_token = "<|endoftext|>"

    pretoken_frequencies = {
        (97, 98, 97, 98): 2,  # abab
        (97, 98, 97, 99): 1,  # abac
    }

    vocab, merges = train_bpe_from_counts(
        pretoken_frequencies=pretoken_frequencies,
        vocab_size=259,
        special_tokens=[special_token.encode("utf-8")],
    )

    supplied_special_tokens = [special_token]

    tokenizer = BPETokenizer(
        vocab=vocab,
        merges=merges,
        special_tokens=supplied_special_tokens,
    )

    assert vocab[256] == b"<|endoftext|>"
    assert vocab[257] == b"ab"
    assert vocab[258] == b"abab"

    assert tokenizer.encode(
        f"abab{special_token}ab"
    ) == [258, 256, 257]

    assert tokenizer.encode(
        f"{special_token}ab"
    ) == [256, 257]

    assert tokenizer.encode(
        f"ab{special_token}"
    ) == [257, 256]

    assert tokenizer.encode(
        f"{special_token}{special_token}"
    ) == [256, 256]

    assert supplied_special_tokens == [special_token]
    assert tokenizer.decode([255]) == "\ufffd"


def test_encode_iterable_preserves_multiline_merge():
    vocab = {i: bytes([i]) for i in range(256)}
    vocab[256] = b"\n\n"
    tokenizer = BPETokenizer(vocab, [(b"\n", b"\n")], None)
    text = "a\n\n b"

    assert list(tokenizer.encode_iterable(["a\n", "\n", " b"])) == tokenizer.encode(text)


def test_encode_iterable_waits_for_multistage_merge():
    vocab = {i: bytes([i]) for i in range(256)}
    vocab[256] = b"bc"
    vocab[257] = b"abc"
    tokenizer = BPETokenizer(vocab, [(b"b", b"c"), (b"a", b"bc")], None)

    assert list(tokenizer.encode_iterable(["a", "b", "c"])) == tokenizer.encode("abc")


def test_encode_iterable_preserves_split_special_token():
    special_token = "<|endoftext|>"
    vocab = {i: bytes([i]) for i in range(256)}
    vocab[256] = special_token.encode("utf-8")
    tokenizer = BPETokenizer(vocab, [], [special_token])

    assert list(tokenizer.encode_iterable(["<|", "endoftext", "|>"])) == tokenizer.encode(special_token)


def test_encode_iterable_keeps_whitespace_context_before_split_special():
    special_token = "<|endoftext|>"
    vocab = {i: bytes([i]) for i in range(256)}
    vocab[256] = b"\n\n"
    vocab[257] = special_token.encode("utf-8")
    tokenizer = BPETokenizer(vocab, [(b"\n", b"\n")], [special_token])
    text = "\n\nbba<|endoftext|><|endoftext|>c"
    chunks = ["\n\nbba<|e", "ndoftex", "t", "|><|e", "ndoft", "ex", "t|>c"]

    assert list(tokenizer.encode_iterable(chunks)) == tokenizer.encode(text)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
