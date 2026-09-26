import regex as re
from collections.abc import Iterable,Iterator

def count_pairs(
    sequences: dict[tuple[int, ...], int],
    ) -> dict[tuple[int, int], int]:
    """统计序列中所有相邻对的数量"""
    result = {}
    if len(sequences) <= 0:
        return {}
    for i in sequences:
        if len(i) <= 1:
            continue
        for j in range(1,len(i)):
            key = i[j-1:j+1]
            result[key] = result.get(key,0) + sequences[i]
    return result

def merge_pair(
    sequence: tuple[int, ...],pair: tuple[int, int],
    new_token: int,
    ) -> tuple[int, ...]:
    """按照指定对合并相邻对"""
    if len(sequence) <= 1:
        return sequence
    result = []
    list_now = []
    for i in sequence:
        list_now.append(i)
        if len(list_now) == 2:
            if tuple(list_now) == pair:
                result.append(new_token)
                list_now.clear()
            else:
                result.append(list_now.pop(0))
    if len(list_now) != 0:
        result.append(list_now.pop())
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
    vocab_dict = {i:bytes([i]) for i in range(256)}
    merge_list = []
    if special_tokens_count > 0:
        for i in special_tokens:
            vocab_dict[len(vocab_dict)] = i
    frequencies = pretoken_frequencies
    while vocab_size > len(vocab_dict):
        pairs = count_pairs(frequencies)
        if(len(pairs) == 0):
            break
        pair = max(pairs,key=lambda i:(pairs[i],vocab_dict[i[0]],vocab_dict[i[1]]))
        new_frequencies = {}
        token = len(vocab_dict)
        for i in {k:v for k,v in frequencies.items() if v > 0}:
            seq_merged = merge_pair(i,pair,token)
            new_frequencies[seq_merged] = new_frequencies.get(seq_merged,0) + frequencies[i]
        vocab_dict[token] = vocab_dict[pair[0]] + vocab_dict[pair[1]]
        merge_list.append((vocab_dict[pair[0]],vocab_dict[pair[1]]))
        frequencies = new_frequencies
    return vocab_dict,merge_list

def pretokenize_counts(
    text: str,
    special_tokens: list[str] | None = None,
    ) -> dict[tuple[int, ...], int]:
    """预先分词"""
    words = []
    if special_tokens is None or len(special_tokens) == 0:
        words.append(text)
    else:
        sort_special_tokens = sorted(special_tokens,key=len,reverse=True)
        special_str =  '|'.join(re.escape(sep) for sep in sort_special_tokens)
        words = [word for word in re.split(special_str, text) if word]
    result = {}
    PRETOKEN_PATTERN = (
                        r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+|"""
                        r""" ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
                        )
    for word in words:
        matches = re.findall(PRETOKEN_PATTERN,word)
        for i in matches:
            token = [x for x in i.encode("utf-8")]
            result[tuple(token)] = result.get(tuple(token),0) + 1
    return result
 

class BPETokenizer:

    def __init__(self,vocab:dict[int, bytes],merges:list[tuple[bytes, bytes]],
                 special_tokens: list[str] | None):
        self.vocab = vocab
        self.merges = {v:k for k,v in enumerate(merges)}
        self.special_tokens = [] if special_tokens is None else sorted(special_tokens,key=len,reverse=True)

        self.bytes2token = {v:k for k,v in self.vocab.items()}

        self.specialtoken2id = {}
        for token in self.special_tokens:
            tokenbytes = token.encode("utf-8")
            if  tokenbytes not in self.bytes2token:
                raise ValueError("存在特殊token不在词表中")
            else:
                self.specialtoken2id[tokenbytes] = self.bytes2token[token.encode("utf-8")]
        

    def encode(self,text:str)->list[int]:
        words = []
        if len(self.special_tokens) == 0:
            words.append(text)
        else:
            special_str =  '|'.join(re.escape(sep) for sep in self.special_tokens)
            special_str = '(' + special_str + ')'
            words = [word for word in re.split(special_str, text) if word]
        result = []
        PRETOKEN_PATTERN = (
                        r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+|"""
                        r""" ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
                        )
        for word in words:
            if word not in self.special_tokens:
                matches = re.findall(PRETOKEN_PATTERN,word)
                for i in matches:
                    bytes_sequence = i.encode("utf-8")
                    token = tuple([self.bytes2token[bytes_sequence[x:x+1]] for x in range(len(bytes_sequence))])
                    for i in self.merges:
                        left_token = self.bytes2token[i[0]]
                        right_token = self.bytes2token[i[1]]
                        new_token = self.bytes2token[i[0]+i[1]]
                        token = merge_pair(token,(left_token,right_token),new_token)
                    result += list(token)
            else:
                result.append(self.specialtoken2id[word.encode("utf-8")])

        return result


    def decode(self,tokens:list[int])->str:
        result = b"".join([self.vocab[token] for token in tokens]).decode("utf-8",errors='replace')
        return result

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for s in iterable:
            result = self.encode(s)
            for i in result:
                yield i