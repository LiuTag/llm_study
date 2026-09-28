"""外层训练循环验收测试；可用 python test_train_loop.py 直接运行。"""

import _paths  # noqa: F401
from copy import deepcopy
import importlib
import importlib.util

import numpy as np
import pytest
import torch
from torch import nn

from AdamW import AdamW
from CrossEntropy import cross_entropy
from Data import get_batch
from Train import train_step
from TransformerLM import TransformerLM


def get_evaluate_loss():
    assert importlib.util.find_spec("TrainLoop") is not None, "请先创建 TrainLoop.py"
    module = importlib.import_module("TrainLoop")
    evaluate_loss = getattr(module, "evaluate_loss", None)
    assert callable(evaluate_loss), "请先在 TrainLoop.py 中实现 evaluate_loss"
    return evaluate_loss


def get_train_loop():
    assert importlib.util.find_spec("TrainLoop") is not None, "请先创建 TrainLoop.py"
    module = importlib.import_module("TrainLoop")
    train_loop = getattr(module, "train_loop", None)
    assert callable(train_loop), "请先在 TrainLoop.py 中实现 train_loop"
    return train_loop


class TinyLM(nn.Module):
    def __init__(self, vocab_size: int = 11, d_model: int = 12) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.head = nn.Linear(d_model, vocab_size)
        self.forward_modes: list[tuple[bool, bool]] = []

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        self.forward_modes.append((self.training, torch.is_grad_enabled()))
        return self.head(self.embedding(input_ids))


def test_evaluate_loss_averages_batches_without_gradients_or_parameter_changes():
    evaluate_loss = get_evaluate_loss()
    torch.manual_seed(1)
    model = TinyLM()
    model.train()
    dataset = (np.arange(60, dtype=np.uint16) % 11).astype(np.uint16)
    before = [parameter.detach().clone() for parameter in model.parameters()]
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)

    torch.manual_seed(17)
    actual = evaluate_loss(
        model, dataset, batch_size=3, context_length=4,
        device="cpu", eval_batches=4,
    )

    assert isinstance(actual, float)
    assert np.isfinite(actual)
    assert model.training
    assert model.forward_modes == [(False, False)] * 4
    for previous, parameter in zip(before, model.parameters()):
        assert torch.equal(previous, parameter)
        assert torch.equal(parameter.grad, torch.ones_like(parameter))

    torch.manual_seed(17)
    with torch.no_grad():
        expected_losses = []
        for _ in range(4):
            x, y = get_batch(dataset, 3, 4, "cpu")
            expected_losses.append(cross_entropy(model(x), y).item())
    assert actual == pytest.approx(sum(expected_losses) / 4, abs=1e-6)


def test_evaluate_loss_restores_an_initial_eval_mode():
    evaluate_loss = get_evaluate_loss()
    model = TinyLM()
    model.eval()
    dataset = np.arange(20, dtype=np.uint16) % 11

    evaluate_loss(model, dataset, 2, 3, "cpu", 2)

    assert not model.training
    assert model.forward_modes == [(False, False)] * 2


def test_train_loop_records_initial_periodic_and_final_evaluations():
    train_loop = get_train_loop()
    torch.manual_seed(2)
    model = TinyLM()
    reference_model = deepcopy(model)
    optimizer = AdamW(model.parameters(), lr=0.03, weight_decay=0.0)
    reference_optimizer = AdamW(
        reference_model.parameters(), lr=0.03, weight_decay=0.0
    )
    train_data = np.array([0, 1, 2, 3, 4], dtype=np.uint16)
    val_data = np.array([5, 6, 7, 8, 9], dtype=np.uint16)
    train_x = torch.tensor([[0, 1, 2, 3]])
    train_y = torch.tensor([[1, 2, 3, 4]])
    val_x = torch.tensor([[5, 6, 7, 8]])
    val_y = torch.tensor([[6, 7, 8, 9]])
    initial_train_loss = cross_entropy(model(train_x), train_y).item()
    initial_val_loss = cross_entropy(model(val_x), val_y).item()

    records = train_loop(
        model, optimizer, train_data, val_data,
        batch_size=1, context_length=4, device="cpu",
        max_steps=5, eval_interval=2, eval_batches=2,
    )

    assert isinstance(records, list)
    assert [record["step"] for record in records] == [0, 2, 4, 5]
    assert all(set(record) == {"step", "train_loss", "val_loss"} for record in records)
    assert records[0]["train_loss"] == pytest.approx(initial_train_loss, abs=1e-6)
    assert records[0]["val_loss"] == pytest.approx(initial_val_loss, abs=1e-6)
    assert all(
        isinstance(record["train_loss"], float)
        and isinstance(record["val_loss"], float)
        and np.isfinite(record["train_loss"])
        and np.isfinite(record["val_loss"])
        for record in records
    )
    assert records[-1]["train_loss"] < records[0]["train_loss"]
    assert model.training

    for _ in range(5):
        train_step(reference_model, reference_optimizer, train_x, train_y)
    for actual, expected in zip(model.parameters(), reference_model.parameters()):
        assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)
        assert optimizer.state[actual]["step"] == 5


def test_train_loop_zero_steps_only_evaluates_initial_model():
    train_loop = get_train_loop()
    model = TinyLM()
    optimizer = AdamW(model.parameters())
    before = [parameter.detach().clone() for parameter in model.parameters()]
    data = np.array([0, 1, 2, 3, 4], dtype=np.uint16)

    records = train_loop(
        model, optimizer, data, data,
        batch_size=1, context_length=4, device="cpu",
        max_steps=0, eval_interval=2, eval_batches=1,
    )

    assert [record["step"] for record in records] == [0]
    assert all(
        torch.equal(previous, parameter)
        for previous, parameter in zip(before, model.parameters())
    )
    assert not optimizer.state


def test_train_loop_connects_real_transformer_to_data_and_optimizer():
    train_loop = get_train_loop()
    torch.manual_seed(3)
    model = TransformerLM(19, 8, 8, 1, 2, 16, 10000.0)
    optimizer = AdamW(model.parameters(), lr=0.01)
    train_data = np.arange(30, dtype=np.uint16) % 19
    val_data = np.arange(30, 60, dtype=np.uint16) % 19

    records = train_loop(
        model, optimizer, train_data, val_data,
        batch_size=2, context_length=4, device="cpu",
        max_steps=2, eval_interval=2, eval_batches=1,
    )

    assert [record["step"] for record in records] == [0, 2]
    assert all(np.isfinite(record["val_loss"]) for record in records)
    assert any(state.get("step") == 2 for state in optimizer.state.values())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
