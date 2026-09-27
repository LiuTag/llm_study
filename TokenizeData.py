import os
from BPE import BPETokenizer

def tokenize_text_file(
    input_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    tokenizer: BPETokenizer,
) -> int:
    token_count = 0
    with open(input_path, "r", encoding="utf-8") as read_f:
        result = tokenizer.encode_iterable(read_f)
        with open(output_path,"wb") as write_f:
            buf = bytearray()
            for i in result:
                if i < 0 or i > 65535:
                    raise ValueError("数值过大")
                buf += i.to_bytes(2, "little")
                token_count += 1
                if len(buf) >= 1 << 20: 
                    write_f.write(buf)
                    buf.clear()
            if buf:  
                write_f.write(buf)
    return token_count
