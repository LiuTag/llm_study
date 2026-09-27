import numpy.typing as npt
import torch
from Train import train_step
from Data import get_batch
from CrossEntropy import cross_entropy


def evaluate_loss(
    model: torch.nn.Module,
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str,
    eval_batches: int,
) -> float:
    if eval_batches <= 0:
        raise ValueError("批次必须为正整数")
    origin_training = model.training
    if origin_training:
        model.eval()
    with torch.no_grad():
        loss_count = 0.0
        for i in range(eval_batches):
            origin_ids,target_ids = get_batch(dataset,batch_size,context_length,device)
            logits = model(origin_ids)
            loss = cross_entropy(logits=logits,targets=target_ids)
            loss_count += loss.detach().sum().item()
    if origin_training:
        model.train()
    return loss_count / eval_batches



def train_loop(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_data: npt.NDArray,
    val_data: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str,
    max_steps: int,
    eval_interval: int,
    eval_batches: int,
) -> list[dict[str, int | float]]:
    result = [
        {
            "step":0,
            "train_loss":evaluate_loss(model,train_data,batch_size,context_length,device,eval_batches),
            "val_loss":evaluate_loss(model,val_data,batch_size,context_length,device,eval_batches),
        }
    ]

    for i in range(1,1 + max_steps):
        train_origin_data,train_target_data = get_batch(train_data,batch_size,context_length,device)
        train_step(model,optimizer,train_origin_data,train_target_data)

        if i % eval_interval == 0 or i == max_steps:
            val_loss = evaluate_loss(model,val_data,batch_size,context_length,device,eval_batches)
            train_loss = evaluate_loss(model,train_data,batch_size,context_length,device,eval_batches)
            result.append({
                "step":i,
                "train_loss":train_loss,
                "val_loss":val_loss,
            })
    return result
