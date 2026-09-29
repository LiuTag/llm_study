import numpy.typing as npt
import torch
import os
from collections.abc import Callable

from Train import train_step
from Data import get_batch
from CrossEntropy import cross_entropy


MetricRecord = dict[str, int | float]

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
    *,
    start_step: int = 0,
    on_evaluation: Callable[[MetricRecord], None] | None = None,
    on_step_end: Callable[[int], None] | None = None,
    max_grad_norm: float | None = None,
    lr_scheduler: Callable[[int], float] | None = None,
) -> list[MetricRecord]:

    result = []

    if start_step == 0:
        record = {
                    "step":start_step,
                    "train_loss":evaluate_loss(model,train_data,batch_size,context_length,device,eval_batches),
                    "val_loss":evaluate_loss(model,val_data,batch_size,context_length,device,eval_batches),
                }
        if on_evaluation is not None:
            on_evaluation(record)
        result.append(record)

    for i in range(1 + start_step,1 + max_steps):
        train_origin_data,train_target_data = get_batch(train_data,batch_size,context_length,device)


        if lr_scheduler is not None:
            lr = lr_scheduler(i)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

        train_step(model,optimizer,train_origin_data,train_target_data,max_grad_norm=max_grad_norm)

        if on_step_end is not None:
            on_step_end(i)

        if i % eval_interval == 0 or i == max_steps:
            val_loss = evaluate_loss(model,val_data,batch_size,context_length,device,eval_batches)
            train_loss = evaluate_loss(model,train_data,batch_size,context_length,device,eval_batches)
            record = {
                "step": i,
                "train_loss":train_loss,
                "val_loss":val_loss,
            }
            result.append(record)
            if on_evaluation is not None:
                on_evaluation(record)
    return result
