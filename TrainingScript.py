import os
import json
from collections import deque
from TrainingData import load_token_ids
from TrainLoop import train_loop,MetricRecord
from TransformerLM import TransformerLM
from AdamW import AdamW
from Checkpoint import save_checkpoint_atomic,load_checkpoint
from LearningRateSchedule import get_lr_cosine_schedule

def run_training(
    train_path: str | os.PathLike[str],
    val_path: str | os.PathLike[str],
    metrics_path: str | os.PathLike[str],
    checkpoint_path: str | os.PathLike[str],
    *,
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    batch_size: int,
    max_steps: int,
    eval_interval: int,
    eval_batches: int,
    learning_rate: float,
    device: str,
    checkpoint_interval: int | None = None,
    resume_from: str | os.PathLike[str] | None = None,
    max_grad_norm: float | None = None,
    min_learning_rate: float = 0.0,
    warmup_iters: int = 0,
    cosine_cycle_iters: int | None = None,
) -> list[MetricRecord]:
    train_tokens = load_token_ids(train_path)
    val_tokens = load_token_ids(val_path)
    model = TransformerLM(vocab_size,context_length,d_model,num_layers,num_heads,d_ff,rope_theta).to(device=device)
    optimizer = AdamW(model.parameters(),lr=learning_rate)

    start_step = 0
    if resume_from is not None:
        start_step = load_checkpoint(resume_from,model=model,optimizer=optimizer)

    def on_step_end(it:int)->None:
        if checkpoint_interval is not None and it % checkpoint_interval == 0:
            save_checkpoint_atomic(model,optimizer,it,checkpoint_path)

    def on_evaluation(record:MetricRecord)->None:
        '''写入最新记录'''
        with open(metrics_path, "a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def lr_scheduler(it:int)->float:
        return get_lr_cosine_schedule(it,learning_rate,min_learning_rate,warmup_iters,cosine_cycle_iters)

    '''处理之前已存在日志，去除日志中错误信息'''
    def deal_exist_log()->list[MetricRecord]:
        if os.path.exists(metrics_path) and os.path.getsize(metrics_path) > 0:
            with open(metrics_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            if lines and len(lines) > 0:
                need_remove = False
                new_lines = []
                for i in lines:
                    try:
                        last_record = json.loads(i)
                        if last_record["step"] > start_step or start_step == 0:
                            need_remove = True
                        else:
                            new_lines.append(i)
                    except json.JSONDecodeError:
                        need_remove = True
                if need_remove:
                    with open(metrics_path, "w", encoding="utf-8") as f:
                        f.writelines(new_lines)
                return [json.loads(i) for i in new_lines]
        return []
       

    before_log = deal_exist_log()

    train_log = train_loop(model,optimizer,train_tokens,val_tokens,batch_size,context_length,
                           device,max_steps,eval_interval,eval_batches,
                start_step=start_step,
                on_evaluation=on_evaluation,
                on_step_end=on_step_end,
                max_grad_norm=max_grad_norm,
                lr_scheduler=lr_scheduler if cosine_cycle_iters is not None else None)
    
    save_checkpoint_atomic(model,optimizer,max_steps,checkpoint_path)
    return before_log + train_log
