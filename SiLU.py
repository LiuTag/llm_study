import torch

def silu(x:torch.Tensor)->torch.Tensor:
    return torch.sigmoid(x) * x
