import regex as re
from collections.abc import Iterable, Iterator

# 模块级预编译的 pretoken 正则
_PRETOKEN_PATTERN = re.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+|"""
    r""" ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


def count_pairs(
    sequences: dict[tuple[int, ...], int],
) -> dict[tuple[int, int], int]:
    """统计序列中所有相邻对的数量"""
    result: dict[tuple[int, int], int] = {}
    for seq, freq in sequences.items():
        for j in range(1, len(seq)):
            key = (seq[j - 1], seq[j])
            result[key] = result.get(key, 0) + freq
    return result


def merge_pair(
    sequence: tuple[int, ...],
    pair: tuple[int, int],
    new_token: int,
) -> tuple[int, ...]:
    """按照指定对合并相邻对（非重叠）"""
    if len(sequence) <= 1:
        return sequence
    left, right = pair
    result = []
    i = 0
    n = len(sequence)
    while i < n:
        if i + 1 < n and sequence[i] == left and sequence[i + 1] == right:
            result.append(new_token)
            i += 2
        else:
            result.append(sequence[i])
            i += 1
    return tuple(result)


def train_bpe_from_counts(
    pretoken_frequencies: dict[tuple[int, ...], int],
    vocab_size: int,
    special_tokens: list[bytes] | None = None,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """按照指定的词表大小进行合并相邻项"""
    special_tokens_count = 0 if special_tokens is None else len(special_tokens)
    if vocab_size < special_tokens_count + 256:
        raise ValueError("词表数太小")

    vocab_dict: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    merge_list: list[tuple[bytes, bytes]] = []
    if special_tokens_count > 0:
        for st in special_tokens:
            vocab_dict[len(vocab_dict)] = st

    frequencies = pretoken_frequencies
    while vocab_size > len(vocab_dict):
        pairs = count_pairs(frequencies)
        if not pairs:
            break

        # 选择频率最高、平局时按 (left_bytes, right_bytes) 字典序更大者
        best_pair = None
        best_key = None
        for p, cnt in pairs.items():
            key = (cnt, vocab_dict[p[0]], vocab_dict[p[1]])
            if best_key is None or key > best_key:
                best_key = key
                best_pair = p

        new_token = len(vocab_dict)
        new_frequencies: dict[tuple[int, ...], int] = {}
        for seq, freq in frequencies.items():
            merged = merge_pair(seq, best_pair, new_token)
            new_frequencies[merged] = new_frequencies.get(merged, 0) + freq

        vocab_dict[new_token] = vocab_dict[best_pair[0]] + vocab_dict[best_pair[1]]
        merge_list.append((vocab_dict[best_pair[0]], vocab_dict[best_pair[1]]))
        frequencies = new_frequencies

    return vocab_dict, merge_list


def pretokenize_counts(
    text: str,
    special_tokens: list[str] | None = None,
) -> dict[tuple[int, ...], int]:
    """预先分词"""
    if not special_tokens:
        words = [text]
    else:
        sort_special_tokens = sorted(special_tokens, key=len, reverse=True)
        special_str = '|'.join(re.escape(sep) for sep in sort_special_tokens)
        words = [w for w in re.split(special_str, text) if w]

    result: dict[tuple[int, ...], int] = {}
    for word in words:
        for match in _PRETOKEN_PATTERN.findall(word):
            token = tuple(match.encode("utf-8"))
            result[token] = result.get(token, 0) + 1
    return result


class BPETokenizer:

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None,
    ):
        self.vocab = vocab
        self.special_tokens = (
            [] if special_tokens is None
            else sorted(special_tokens, key=len, reverse=True)
        )

        self.bytes2token = {v: k for k, v in vocab.items()}

        # (left_id, right_id) -> (rank, new_id)，用于 O(1) 查询优先级与结果
        self.merges: dict[tuple[int, int], tuple[int, int]] = {}
        for rank, (left_bytes, right_bytes) in enumerate(merges):
            left_id = self.bytes2token[left_bytes]
            right_id = self.bytes2token[right_bytes]
            new_id = self.bytes2token[left_bytes + right_bytes]
            self.merges[(left_id, right_id)] = (rank, new_id)

        # 特殊 token 映射与集合（O(1) 判定）
        self.specialtoken2id: dict[bytes, int] = {}
        self._special_set = set(self.special_tokens)
        for token in self.special_tokens:
            tokenbytes = token.encode("utf-8")
            if tokenbytes not in self.bytes2token:
                raise ValueError("存在特殊token不在词表中")
            self.specialtoken2id[tokenbytes] = self.bytes2token[tokenbytes]

        # 预编译特殊 token 分割正则
        if self.special_tokens:
            special_str = '|'.join(re.escape(sep) for sep in self.special_tokens)
            self._special_split_re = re.compile('(' + special_str + ')')
        else:
            self._special_split_re = None

        # 单字节 -> id 的查找表，避免反复 bytes([b])
        self._byte_to_id = [self.bytes2token[bytes([i])] for i in range(256)]

        # 编码缓存
        self._cache: dict[str, list[int]] = {}

    # ---------- 内部 ----------

    def _apply_bpe(self, tokens: list[int]) -> list[int]:
        """标准 BPE：反复合并当前序列中优先级最高（rank 最小）的相邻对"""
        merges = self.merges
        while len(tokens) > 1:
            min_rank = -1
            best_idx = -1
            best_new_id = -1
            prev = tokens[0]
            for i in range(1, len(tokens)):
                cur = tokens[i]
                entry = merges.get((prev, cur))
                if entry is not None:
                    rank, new_id = entry
                    if min_rank == -1 or rank < min_rank:
                        min_rank = rank
                        best_idx = i - 1
                        best_new_id = new_id
                prev = cur
            if best_idx == -1:
                break
            tokens[best_idx:best_idx + 2] = [best_new_id]
        return tokens

    # ---------- 对外接口 ----------

    def encode(self, text: str) -> list[int]:
        if self._special_split_re is not None:
            words = [w for w in self._special_split_re.split(text) if w]
        else:
            words = [text]

        result: list[int] = []
        cache = self._cache
        byte_to_id = self._byte_to_id
        special_set = self._special_set
        special_ids = self.specialtoken2id
        pretoken_re = _PRETOKEN_PATTERN

        for word in words:
            if word in special_set:
                result.append(special_ids[word.encode("utf-8")])
                continue

            cached = cache.get(word)
            if cached is not None:
                result.extend(cached)
                continue

            word_tokens: list[int] = []
            for match in pretoken_re.findall(word):
                sub = cache.get(match)
                if sub is None:
                    ids = [byte_to_id[b] for b in match.encode("utf-8")]
                    self._apply_bpe(ids)
                    cache[match] = ids
                    sub = ids
                word_tokens.extend(sub)

            cache[word] = word_tokens
            result.extend(word_tokens)

        return result

    def decode(self, tokens: list[int]) -> str:
        return b"".join(self.vocab[t] for t in tokens).decode(
            "utf-8", errors="replace"
        )

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for s in iterable:
            yield from self.encode(s)