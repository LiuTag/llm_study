"""silu激活函数"""

import torch

def silu(x:torch.Tensor)->torch.Tensor:
    return torch.sigmoid(x) * x
