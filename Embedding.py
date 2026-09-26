from torch import nn
import torch
import math

class Embedding(nn.Module):
    def __init__(self,num_embeddings:int,embedding_dim:int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn((num_embeddings,embedding_dim),dtype=torch.float32))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self,x:torch.Tensor):
        has_native = torch.any(x < 0)
        if has_native:
            raise ValueError("数组中索引不能有负值")
        result = self.weight[x]
        return result
        

class Linear(nn.Module):
    def __init__(self,in_features:int,out_features:int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn((out_features,in_features),dtype=torch.float32))
        std = math.sqrt(2.0 / (in_features + out_features))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3.0*std, b=3.0*std)

    def forward(self,x:torch.Tensor
                )-> torch.Tensor:
        return x @ self.weight.T