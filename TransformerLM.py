import torch
from torch import nn
from TransformerBlock import TransformerBlock
from Embedding import Embedding
from Embedding import Linear
from RMSNorm import RMSNorm


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
    ) -> None:
        super().__init__()
        self.token_embeddings = Embedding(vocab_size,d_model)
        self.layers = nn.ModuleList([TransformerBlock(d_model=d_model,
                                                   num_heads=num_heads,
                                                   d_ff=d_ff,
                                                   max_seq_len=context_length,
                                                   theta=rope_theta) for i in range(num_layers)])
        self.lm_head = Linear(d_model,vocab_size)
        self.ln_final = RMSNorm(d_model=d_model)

    def forward(
        self,
        token_ids: torch.Tensor,
        token_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        result = self.token_embeddings(token_ids)
        for i in self.layers:
            result = i(result,token_positions)
        return self.lm_head(self.ln_final(result))