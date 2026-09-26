import torch
from Attention import MultiHeadSelfAttention
from SwiGLU import SwiGLU
from RMSNorm import RMSNorm


class TransformerBlock(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
    ):
        super().__init__()
        self.ln1 = RMSNorm(d_model=d_model)
        self.ln2 = RMSNorm(d_model=d_model)
        self.attn = MultiHeadSelfAttention(d_model=d_model,
                                           num_heads=num_heads,
                                           max_seq_len=max_seq_len,
                                           theta=theta)
        self.ffn = SwiGLU(d_model=d_model,d_ff=d_ff)

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        u = x + self.attn(self.ln1(x),token_positions)
        y = u + self.ffn(self.ln2(u))
        return y
        