import torch
from torch import nn

class RotaryPositionalEmbedding(nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device=None,
    ):
        super().__init__()
        self.theta = theta
        if d_k <= 0 or d_k % 2 != 0:
            raise ValueError("d_k must be a positive even integer")
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        frequencies = theta ** (-torch.arange(0, d_k, 2, device=device) / d_k)
        positions = torch.arange(0,max_seq_len,device=device)
        angles = positions[:,None]  * frequencies[None,:]
        self.register_buffer("cos", torch.cos(angles), persistent=False)
        self.register_buffer("sin", torch.sin(angles), persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor,
    ) -> torch.Tensor:
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]  
        cos = self.cos[token_positions].to(dtype=x.dtype)
        sin = self.sin[token_positions].to(dtype=x.dtype)
        while cos.ndim < x.ndim:
            cos = cos.unsqueeze(-3)
            sin = sin.unsqueeze(-3)
        new_even = x_even * cos - x_odd * sin
        new_odd  = x_even * sin + x_odd * cos
        return torch.stack([new_even, new_odd], dim=-1).flatten(-2)