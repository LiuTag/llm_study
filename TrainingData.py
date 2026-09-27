import os
import numpy as np

def load_token_ids(path: str | os.PathLike[str]) -> np.memmap:
    fp = np.memmap(path,mode="r",dtype="uint16")
    return fp
