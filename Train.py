import torch
from CrossEntropy import cross_entropy
from GradientClipping import gradient_clipping

def train_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    input_ids: torch.Tensor,
    target_ids: torch.Tensor,
    *,
    max_grad_norm: float | None = None,
) -> torch.Tensor:
    model.train()
    optimizer.zero_grad()
    logits = model(input_ids)
    loss = cross_entropy(logits=logits,targets=target_ids)
    loss.backward()
    if max_grad_norm is not None:
        gradient_clipping(model.parameters(),max_grad_norm)
    optimizer.step()
    return loss.detach()