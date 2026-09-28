import _paths  # noqa: F401
from copy import deepcopy

import pytest

from BPE import (
    BPETokenizer,
    count_pairs,
    merge_pair,
    pretokenize_counts,
    train_bpe_from_counts,
)


def test_count_pairs_uses_sequence_frequencies():
    sequences = {
        (97, 98, 97, 98): 2,
        (97, 98, 97, 99): 1,
    }

    assert count_pairs(sequences) == {
        (97, 98): 5,
        (98, 97): 3,
        (97, 99): 1,
    }


def test_count_pairs_ignores_short_sequences():
    assert count_pairs({(): 3, (97,): 5}) == {}


def test_merge_pair_is_left_to_right_and_non_overlapping():
    assert merge_pair((97, 98, 97, 98), (97, 98), 256) == (256, 256)
    assert merge_pair((97, 97, 97), (97, 97), 256) == (256, 97)


def test_merge_pair_handles_edges_and_does_not_modify_input():
    sequence = (97, 98, 99)
    original = sequence

    assert merge_pair(sequence, (1, 2), 256) == sequence
    assert sequence == original
    assert merge_pair((), (97, 98), 256) == ()
    assert merge_pair((97,), (97, 98), 256) == (97,)


def test_train_bpe_from_counts_toy_example():
    pretokens = {
        (97, 98, 97, 98): 2,
        (97, 98, 97, 99): 1,
    }
    original = deepcopy(pretokens)

    vocab, merges = train_bpe_from_counts(pretokens, vocab_size=258)

    assert vocab[0] == b"\x00"
    assert vocab[97] == b"a"
    assert vocab[255] == b"\xff"
    assert vocab[256] == b"ab"
    assert vocab[257] == b"abab"
    assert merges == [(b"a", b"b"), (b"ab", b"ab")]
    assert pretokens == original


def test_train_bpe_handles_special_tokens_and_early_exit():
    special = b"<|endoftext|>"
    vocab, merges = train_bpe_from_counts(
        {(97, 98, 97, 98): 2, (97, 98, 97, 99): 1},
        vocab_size=259,
        special_tokens=[special],
    )

    assert vocab[256] == special
    assert vocab[257] == b"ab"
    assert vocab[258] == b"abab"
    assert merges == [(b"a", b"b"), (b"ab", b"ab")]

    empty_vocab, empty_merges = train_bpe_from_counts({}, vocab_size=258)
    assert len(empty_vocab) == 256
    assert empty_merges == []

    short_vocab, short_merges = train_bpe_from_counts({(97,): 3}, vocab_size=258)
    assert len(short_vocab) == 256
    assert short_merges == []


def test_train_bpe_uses_deterministic_byte_tie_breaking():
    vocab, merges = train_bpe_from_counts(
        {(97, 98): 1, (99, 100): 1},
        vocab_size=257,
    )

    assert vocab[256] == b"cd"
    assert merges == [(b"c", b"d")]


def test_pretokenize_counts_protects_special_tokens():
    special = "<|endoftext|>"

    assert pretokenize_counts(
        f"abab{special}abab",
        [special],
    ) == {(97, 98, 97, 98): 2}

    assert pretokenize_counts(f"{special}{special}", [special]) == {}
    assert pretokenize_counts("你", []) == {
        tuple("你".encode("utf-8")): 1,
    }


def test_pretokenize_counts_prefers_longest_overlapping_special_token():
    assert pretokenize_counts(
        "<a>bX",
        ["<a>", "<a>b"],
    ) == {(88,): 1}


def test_tokenizer_encode_decode_with_special_token():
    special = "<|endoftext|>"
    vocab, merges = train_bpe_from_counts(
        {(97, 98, 97, 98): 2, (97, 98, 97, 99): 1},
        vocab_size=259,
        special_tokens=[special.encode("utf-8")],
    )
    tokenizer = BPETokenizer(vocab, merges, [special])

    text = f"abab{special}ab"
    ids = tokenizer.encode(text)

    assert ids == [258, 256, 257]
    assert tokenizer.decode(ids) == text
    assert tokenizer.encode("") == []


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
