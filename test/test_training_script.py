"""最小训练入口验收测试；可直接执行本文件。"""

import _paths  # noqa: F401
import importlib
import json
import math

import numpy as np
import pytest
import torch


def get_runner():
    module = importlib.import_module("TrainingScript")
    runner = getattr(module, "run_training", None)
    assert callable(runner), "请在 TrainingScript.py 中实现 run_training"
    return runner


def write_tokens(path, first, last):
    np.arange(first, last, dtype=np.uint16).tofile(path)


def test_run_training_writes_metrics_and_resumable_checkpoint(tmp_path):
    train_path = tmp_path / "train.bin"
    val_path = tmp_path / "val.bin"
    metrics_path = tmp_path / "metrics.jsonl"
    checkpoint_path = tmp_path / "checkpoint.pt"
    write_tokens(train_path, 0, 30)
    write_tokens(val_path, 30, 60)

    torch.manual_seed(7)
    records = get_runner()(
        train_path=train_path,
        val_path=val_path,
        metrics_path=metrics_path,
        checkpoint_path=checkpoint_path,
        vocab_size=64,
        context_length=4,
        d_model=8,
        num_layers=1,
        num_heads=2,
        d_ff=16,
        rope_theta=10000.0,
        batch_size=2,
        max_steps=2,
        eval_interval=1,
        eval_batches=1,
        learning_rate=0.01,
        device="cpu",
    )

    assert [row["step"] for row in records] == [0, 1, 2]
    assert all(set(row) == {"step", "train_loss", "val_loss"} for row in records)
    assert all(
        isinstance(row["train_loss"], float)
        and isinstance(row["val_loss"], float)
        and math.isfinite(row["train_loss"])
        and math.isfinite(row["val_loss"])
        for row in records
    )

    assert metrics_path.is_file()
    logged = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines()]
    assert logged == records

    assert checkpoint_path.is_file()
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    assert saved["iteration"] == 2
    assert saved["model"]
    assert saved["optimizer"]["state"]
    assert saved["optimizer"]["param_groups"][0]["lr"] == pytest.approx(0.01)


def test_zero_steps_only_records_baseline_and_saves_iteration_zero(tmp_path):
    train_path = tmp_path / "train.bin"
    val_path = tmp_path / "val.bin"
    metrics_path = tmp_path / "metrics.jsonl"
    checkpoint_path = tmp_path / "checkpoint.pt"
    write_tokens(train_path, 0, 30)
    write_tokens(val_path, 30, 60)

    records = get_runner()(
        train_path=train_path,
        val_path=val_path,
        metrics_path=metrics_path,
        checkpoint_path=checkpoint_path,
        vocab_size=64,
        context_length=4,
        d_model=8,
        num_layers=1,
        num_heads=2,
        d_ff=16,
        rope_theta=10000.0,
        batch_size=2,
        max_steps=0,
        eval_interval=1,
        eval_batches=1,
        learning_rate=0.01,
        device="cpu",
    )

    assert [row["step"] for row in records] == [0]
    logged = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines()]
    assert logged == records
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    assert saved["iteration"] == 0
    assert not saved["optimizer"]["state"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
