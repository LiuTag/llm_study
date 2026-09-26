import torch
import math


def cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    new_logits = logits.to(dtype=torch.float64)
    max_logit = torch.max(new_logits,dim=-1,keepdim=True).values
    num = math.prod(list(targets.shape))
    loss = max_logit.squeeze(dim=-1) + \
            torch.log(torch.sum(torch.exp(new_logits-max_logit),dim=-1)) - \
            new_logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    average_loss = (torch.sum(loss).squeeze()/num).to(dtype=logits.dtype)
    return average_loss