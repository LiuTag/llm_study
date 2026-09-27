import os
import json
from TrainingData import load_token_ids
from TrainLoop import train_loop
from TransformerLM import TransformerLM
from AdamW import AdamW
from Checkpoint import save_checkpoint

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
) -> list[dict[str, int | float]]:
    train_tokens = load_token_ids(train_path)
    val_tokens = load_token_ids(val_path)
    model = TransformerLM(vocab_size,context_length,d_model,num_layers,num_heads,d_ff,rope_theta).to(device=device)
    optimizer = AdamW(model.parameters(),lr=learning_rate)
    train_log = train_loop(model,optimizer,train_tokens,val_tokens,batch_size,context_length,device,max_steps,
               eval_interval,eval_batches)
    with open(metrics_path, "w", encoding="utf-8", newline="\n") as f:
        for record in train_log:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    save_checkpoint(model,optimizer,max_steps,checkpoint_path)
    return train_log
