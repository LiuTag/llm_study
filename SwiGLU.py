from Embedding import Linear
from torch import nn
import torch

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        # 注册三个独立的自定义 Linear 子模块
        super().__init__()
        self.w1 = Linear(d_model, d_ff)
        self.w3 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.w1(x)
        b = self.w3(x)
        return self.w2(torch.sigmoid(a) * a * b) 