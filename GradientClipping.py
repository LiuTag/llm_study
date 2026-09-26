from collections.abc import Iterable

import torch


def gradient_clipping(
    parameters: Iterable[torch.nn.Parameter],
    max_l2_norm: float,
) -> None:
    params = [i for i in parameters if i.grad is not None]
    if len(params) > 0:
        l2_norm = torch.sqrt(sum([torch.sum(i.grad**2) for i in params]))
        if l2_norm > 0:
            ratio = max_l2_norm / l2_norm
            if ratio < 1.0:
                for param in params:
                    param.grad.mul_(ratio)