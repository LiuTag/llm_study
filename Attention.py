import torch
import math
from Embedding import Linear
from RoPE import RotaryPositionalEmbedding

def softmax(
    x: torch.Tensor,
    dim: int,
    ) -> torch.Tensor:
    if x.dtype in (torch.float16, torch.bfloat16):
        compute_dtype = torch.float32
    else:
        compute_dtype = x.dtype

    temp_x = x.to(dtype=compute_dtype)
    max_value = torch.max(temp_x,dim=dim,keepdim=True).values
    y = torch.exp(temp_x - max_value)
    return (y / torch.sum(y,dim=dim,keepdim=True)).to(dtype=x.dtype)


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
    """
    q: (..., query_length, d_k)
    k: (..., key_length, d_k)
    v: (..., key_length, d_v)

    mask:
        可广播到 (..., query_length, key_length)
        True 表示允许关注
        False 表示屏蔽

    returns:
        (..., query_length, d_v)
    """

    d_k = q.shape[-1]
    qk = (q @ k.transpose(-1,-2)) / math.sqrt(d_k)
    if mask is not None:
        qk = qk.masked_fill(mask == False,-torch.inf)
    attention = softmax(qk,dim=-1) @ v
    return attention


class MultiHeadSelfAttention(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int,
        theta: float,
        use_rope: bool = True,
    ):
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError("嵌入向量维度不能被头数整除")
        
        self.q_proj = Linear(d_model,d_model)
        self.k_proj = Linear(d_model,d_model)
        self.v_proj = Linear(d_model,d_model)
        self.output_proj = Linear(d_model,d_model)
        self.rope = RotaryPositionalEmbedding(theta=theta,d_k=d_model//num_heads,max_seq_len=max_seq_len) if use_rope else None
        self.num_heads = num_heads


    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        sequence_length = x.shape[-2]
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        shape = q.shape
        new_shape = list(shape[:-1]) + [self.num_heads,shape[-1]//self.num_heads]
        q = q.reshape(new_shape)
        k = k.reshape(new_shape)
        v = v.reshape(new_shape)
        q = q.transpose(-3,-2)
        k = k.transpose(-3,-2)
        v = v.transpose(-3,-2)

        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(0,sequence_length,1,dtype=torch.int64,device=x.device)
            q_r = self.rope(q,token_positions)
            k_r = self.rope(k,token_positions)
        else:
            q_r = q
            k_r = k

        mask = torch.tril(torch.ones(sequence_length, sequence_length, dtype=torch.bool,device=x.device),diagonal=0)

        attention = scaled_dot_product_attention(q_r,k_r,v,mask)
        attention = attention.transpose(-2,-3)
        attention = attention.reshape(shape)

        mix_attention = self.output_proj(attention)

        return mix_attention
        
        

        