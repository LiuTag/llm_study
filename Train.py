import torch
from CrossEntropy import cross_entropy


def train_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    input_ids: torch.Tensor,
    target_ids: torch.Tensor,
) -> torch.Tensor:
    model.train()
    optimizer.zero_grad()
    logits = model(input_ids)
    loss = cross_entropy(logits=logits,targets=target_ids)
    loss.backward()
    optimizer.step()
    return loss.detach()