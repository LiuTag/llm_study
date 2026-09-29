import os
from typing import IO, BinaryIO
import torch
import tempfile

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
) -> None:

    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration
        }, out)

def save_checkpoint_atomic(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike[str],
) -> None:
    target_path = os.path.abspath(out)
    target_dir = os.path.dirname(target_path)
    os.makedirs(target_dir, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=os.path.basename(target_path) + ".",
        suffix=".tmp",
        dir=target_dir,
    )
    try:
        with os.fdopen(fd, "wb") as f:
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "iteration": iteration
                }, f)
            f.flush()          
            os.fsync(f.fileno()) 

        os.replace(tmp_path, target_path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise

def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    ckpt = torch.load(src,weights_only=True,map_location="cpu")

    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt["iteration"]