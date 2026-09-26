from collections.abc import Iterable
from typing import Callable

import torch


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter] | Iterable[dict],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ) -> None:
        super().__init__(params=params,defaults={
            "lr":lr,
            "betas":betas,
            "eps":eps,
            "weight_decay":weight_decay
        })


    def step(
        self,
        closure: Callable[[], torch.Tensor] | None = None,
    ) -> torch.Tensor | None:
        if closure is not None:
            raise ValueError("未处理closure")
        
        for group in self.param_groups:
            lr = group["lr"]
            betas = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                grad = p.grad
                if grad is None:
                    continue
                with torch.no_grad():
                    state = self.state[p]
                    state["step"] = state.get("step",0) + 1
                    m_t = betas[0] * state.get("exp_avg",torch.tensor(0,device=p.device,dtype=p.dtype)) + (1.0 - betas[0]) * grad
                    v_t = betas[1] * state.get("exp_avg_sq",torch.tensor(0,device=p.device,dtype=p.dtype)) + (1.0 - betas[1]) * grad ** 2
                    m_t_d = m_t / (1.0 - betas[0] ** state["step"])
                    v_t_d = v_t / (1.0 - betas[1] ** state["step"])
                    u_t = m_t_d / (torch.sqrt(v_t_d) + eps)
                    update = -lr * (u_t + weight_decay * p)
                    p.add_(update) 
                    state["exp_avg"] = m_t
                    state["exp_avg_sq"] = v_t
