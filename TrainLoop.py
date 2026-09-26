import numpy.typing as npt
import torch


def evaluate_loss(
    model: torch.nn.Module,
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str,
    eval_batches: int,
) -> float:
    raise NotImplementedError


def train_loop(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_data: npt.NDArray,
    val_data: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str,
    max_steps: int,
    eval_interval: int,
    eval_batches: int,
) -> list[dict[str, int | float]]:
    raise NotImplementedError