"""训练期间落盘、定期 checkpoint 和断点恢复验收；可直接执行本文件。"""

import _paths  # noqa: F401
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

import TrainingScript
import RunTraining
import Checkpoint
from AdamW import AdamW
from TrainLoop import train_loop


class TinyLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(16, 8)
        self.head = nn.Linear(8, 16)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.head(self.embedding(input_ids))


def training_kwargs(tmp_path, max_steps: int) -> dict:
    train_path = tmp_path / "train.bin"
    val_path = tmp_path / "val.bin"
    (np.arange(64, dtype=np.uint16) % 16).tofile(train_path)
    (np.arange(64, 128, dtype=np.uint16) % 16).tofile(val_path)
    return {
        "train_path": train_path,
        "val_path": val_path,
        "metrics_path": tmp_path / "metrics.jsonl",
        "checkpoint_path": tmp_path / "checkpoint.pt",
        "vocab_size": 16,
        "context_length": 4,
        "d_model": 8,
        "num_layers": 1,
        "num_heads": 2,
        "d_ff": 16,
        "rope_theta": 10000.0,
        "batch_size": 2,
        "max_steps": max_steps,
        "eval_interval": 1,
        "eval_batches": 1,
        "learning_rate": 0.01,
        "device": "cpu",
    }


def read_metrics(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_train_loop_resume_uses_absolute_steps_and_calls_hooks() -> None:
    torch.manual_seed(31)
    model = TinyLM()
    optimizer = AdamW(model.parameters(), lr=0.01)
    dataset = np.arange(64, dtype=np.uint16) % 16
    evaluations: list[int] = []
    completed_steps: list[int] = []

    records = train_loop(
        model, optimizer, dataset, dataset,
        batch_size=2, context_length=4, device="cpu",
        max_steps=5, eval_interval=2, eval_batches=1,
        start_step=2,
        on_evaluation=lambda record: evaluations.append(record["step"]),
        on_step_end=completed_steps.append,
    )

    assert [record["step"] for record in records] == [4, 5]
    assert evaluations == [4, 5]
    assert completed_steps == [3, 4, 5]
    assert all(state["step"] == 3 for state in optimizer.state.values())


def test_metrics_and_periodic_checkpoint_survive_an_interruption(tmp_path, monkeypatch) -> None:
    paths = training_kwargs(tmp_path, max_steps=3)

    def interrupted_loop(*args, start_step, on_evaluation, on_step_end, **kwargs):
        assert start_step == 0
        on_evaluation({"step": 0, "train_loss": 1.0, "val_loss": 1.1})
        assert [row["step"] for row in read_metrics(paths["metrics_path"])] == [0]
        on_step_end(1)
        on_step_end(2)
        saved = torch.load(paths["checkpoint_path"], map_location="cpu", weights_only=True)
        assert saved["iteration"] == 2
        raise RuntimeError("模拟训练中断")

    monkeypatch.setattr(TrainingScript, "train_loop", interrupted_loop)
    with pytest.raises(RuntimeError, match="模拟训练中断"):
        TrainingScript.run_training(**paths, checkpoint_interval=2)

    assert [row["step"] for row in read_metrics(paths["metrics_path"])] == [0]
    saved = torch.load(paths["checkpoint_path"], map_location="cpu", weights_only=True)
    assert saved["iteration"] == 2


def test_resume_trims_log_ahead_of_checkpoint_and_continues(tmp_path, monkeypatch) -> None:
    paths = training_kwargs(tmp_path, max_steps=4)
    original_train_loop = TrainingScript.train_loop

    def interrupt_after_step_three_is_logged(*args, on_evaluation, **kwargs):
        def log_then_interrupt(record):
            on_evaluation(record)
            if record["step"] == 3:
                raise RuntimeError("模拟第 3 步日志落盘后中断")

        return original_train_loop(
            *args, on_evaluation=log_then_interrupt, **kwargs
        )

    monkeypatch.setattr(
        TrainingScript, "train_loop", interrupt_after_step_three_is_logged
    )
    torch.manual_seed(37)
    with pytest.raises(RuntimeError, match="第 3 步日志落盘后中断"):
        TrainingScript.run_training(**paths, checkpoint_interval=2)

    # 第 2 步已保存 checkpoint；第 3 步已评估并写入日志，但还未到下次保存点。
    assert [row["step"] for row in read_metrics(paths["metrics_path"])] == [0, 1, 2, 3]
    assert paths["checkpoint_path"].is_file(), "第 2 步应已保存 checkpoint"
    interrupted_checkpoint = torch.load(
        paths["checkpoint_path"], map_location="cpu", weights_only=True
    )
    assert interrupted_checkpoint["iteration"] == 2
    assert interrupted_checkpoint["optimizer"]["state"]
    assert all(
        state["step"] == 2
        for state in interrupted_checkpoint["optimizer"]["state"].values()
    )

    monkeypatch.setattr(TrainingScript, "train_loop", original_train_loop)
    records = TrainingScript.run_training(
        **paths, checkpoint_interval=2, resume_from=paths["checkpoint_path"]
    )

    logged = read_metrics(paths["metrics_path"])
    assert [row["step"] for row in logged] == [0, 1, 2, 3, 4]
    assert logged == records
    assert all(row["train_loss"] >= 0 for row in logged)
    saved = torch.load(paths["checkpoint_path"], map_location="cpu", weights_only=True)
    assert saved["iteration"] == 4
    assert all(state["step"] == 4 for state in saved["optimizer"]["state"].values())


def test_cli_resolves_resume_path_and_forwards_checkpoint_interval(tmp_path, monkeypatch) -> None:
    data = tmp_path / "data"
    data.mkdir()
    (np.arange(32, dtype=np.uint16) % 16).tofile(data / "train.bin")
    (np.arange(32, dtype=np.uint16) % 16).tofile(data / "val.bin")
    config = training_kwargs(tmp_path, max_steps=4)
    for name in ("train_path", "val_path"):
        config[name] = f"data/{name.removesuffix('_path')}.bin"
    config["metrics_path"] = "runs/metrics.jsonl"
    config["checkpoint_path"] = "runs/checkpoint.pt"
    config["resume_from"] = "runs/previous.pt"
    config["checkpoint_interval"] = 2
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    received = {}

    def fake_run_training(**kwargs):
        received.update(kwargs)
        return [{"step": 4, "train_loss": 1.0, "val_loss": 1.1}]

    monkeypatch.setattr(RunTraining, "ROOT", tmp_path)
    monkeypatch.setattr(RunTraining, "run_training", fake_run_training)
    monkeypatch.setattr(sys, "argv", ["RunTraining.py", "--config", str(config_path)])
    RunTraining.main()

    assert received["checkpoint_interval"] == 2
    assert received["resume_from"] == tmp_path / "runs" / "previous.pt"
    assert received["checkpoint_path"] == tmp_path / "runs" / "checkpoint.pt"


def test_resume_rejects_max_steps_that_would_not_advance(tmp_path) -> None:
    """续训时 max_steps 必须严格大于 checkpoint 记录的步数，否则一步都不会训练。"""
    paths = training_kwargs(tmp_path, max_steps=2)
    torch.manual_seed(37)
    TrainingScript.run_training(**paths, checkpoint_interval=1)

    for bad_steps in (2, 1):
        with pytest.raises(ValueError, match="max_steps"):
            TrainingScript.run_training(
                **{**paths, "max_steps": bad_steps},
                resume_from=paths["checkpoint_path"],
            )

    records = TrainingScript.run_training(
        **{**paths, "max_steps": 3}, resume_from=paths["checkpoint_path"]
    )
    assert records[-1]["step"] == 3


def test_atomic_checkpoint_preserves_previous_file_if_write_fails(tmp_path, monkeypatch) -> None:
    model = nn.Linear(3, 2)
    optimizer = AdamW(model.parameters())
    path = tmp_path / "checkpoint.pt"
    Checkpoint.save_checkpoint(model, optimizer, iteration=2, out=path)
    Checkpoint.save_checkpoint_atomic(model, optimizer, iteration=3, out=path)
    assert torch.load(path, map_location="cpu", weights_only=True)["iteration"] == 3

    def broken_save(_payload, out):
        if hasattr(out, "write"):
            out.write(b"incomplete")
        else:
            Path(out).write_bytes(b"incomplete")
        raise OSError("模拟保存中断")

    monkeypatch.setattr(Checkpoint.torch, "save", broken_save)
    with pytest.raises(OSError, match="模拟保存中断"):
        Checkpoint.save_checkpoint_atomic(model, optimizer, iteration=4, out=path)

    saved = torch.load(path, map_location="cpu", weights_only=True)
    assert saved["iteration"] == 3


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
