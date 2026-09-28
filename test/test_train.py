"""最小训练单步验收测试；可用 python test_train.py 直接运行。"""

import _paths  # noqa: F401
from copy import deepcopy
import importlib
import importlib.util

import pytest
import torch
from torch import nn

from AdamW import AdamW
from CrossEntropy import cross_entropy
from TransformerLM import TransformerLM


def get_train_step_function():
    assert importlib.util.find_spec("Train") is not None, "请先创建 Train.py"
    function = getattr(importlib.import_module("Train"), "train_step", None)
    assert callable(function), "请先在 Train.py 中实现 train_step"
    return function


class ToyLM(nn.Module):
    def __init__(self, vocab_size: int = 11, d_model: int = 12) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.head(self.embedding(input_ids))


def test_train_step_matches_manual_forward_backward_and_adamw_update():
    train_step = get_train_step_function()
    torch.manual_seed(0)
    model = ToyLM()
    reference_model = deepcopy(model)
    optimizer = AdamW(model.parameters(), lr=0.02, weight_decay=0.01)
    reference_optimizer = AdamW(
        reference_model.parameters(), lr=0.02, weight_decay=0.01
    )
    x = torch.tensor([[0, 1, 2], [3, 4, 5]])
    y = torch.tensor([[1, 2, 3], [4, 5, 6]])
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)

    actual_loss = train_step(model, optimizer, x, y)

    reference_model.train()
    reference_optimizer.zero_grad(set_to_none=True)
    expected_loss = cross_entropy(reference_model(x), y)
    expected_loss.backward()
    reference_optimizer.step()

    assert isinstance(actual_loss, torch.Tensor)
    assert actual_loss.shape == torch.Size([])
    assert not actual_loss.requires_grad
    assert torch.allclose(actual_loss, expected_loss.detach(), atol=1e-6)
    for actual, expected in zip(model.parameters(), reference_model.parameters()):
        assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_train_step_updates_the_real_transformer_lm():
    train_step = get_train_step_function()
    torch.manual_seed(1)
    model = TransformerLM(
        vocab_size=19,
        context_length=8,
        d_model=8,
        num_layers=1,
        num_heads=2,
        d_ff=16,
        rope_theta=10000.0,
    )
    model.eval()
    optimizer = AdamW(model.parameters(), lr=0.01)
    x = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    y = torch.tensor([[2, 3, 4, 5], [6, 7, 8, 9]])
    before = [parameter.detach().clone() for parameter in model.parameters()]

    loss = train_step(model, optimizer, x, y)

    assert model.training
    assert torch.isfinite(loss)
    assert any(
        not torch.equal(previous, current)
        for previous, current in zip(before, model.parameters())
    )
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_train_step_can_overfit_a_fixed_toy_batch():
    train_step = get_train_step_function()
    torch.manual_seed(2)
    model = ToyLM()
    optimizer = AdamW(model.parameters(), lr=0.03, weight_decay=0.0)
    x = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]])
    y = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    initial_loss = cross_entropy(model(x), y).item()

    for _ in range(30):
        train_step(model, optimizer, x, y)

    final_loss = cross_entropy(model(x), y).item()
    assert final_loss < initial_loss * 0.5


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
