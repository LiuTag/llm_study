from torch import nn
import torch

class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device=None,
        dtype=None,
    ):
        # 初始化模块、保存 eps、创建可训练的 weight
        super().__init__()
        self.eps = eps
        self.dtype = dtype
        self.d_model = d_model
        self.weight = nn.Parameter(torch.ones((d_model,),dtype=dtype,device=device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 根据上面的公式实现
        x_float32 = x.to(dtype=torch.float32)
        x_2 = x_float32 * x_float32
        r = torch.sqrt(torch.mean(x_2,dim=-1,keepdim=True) + self.eps)
        y = x_float32 / r * self.weight
        y = y.to(dtype=x.dtype)
        return y


