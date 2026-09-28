"""AdamW 的验收测试；可用 python test_adamw.py 直接运行。"""

import _paths  # noqa: F401
from copy import deepcopy
import importlib
import importlib.util

import pytest
import torch


def make_optimizer(params, **kwargs):
    specification = importlib.util.find_spec("AdamW")
    assert specification is not None, "请先在 AdamW.py 中实现 AdamW"
    optimizer_class = getattr(importlib.import_module("AdamW"), "AdamW", None)
    assert optimizer_class is not None, "请先在 AdamW.py 中实现 AdamW 类"
    return optimizer_class(params, **kwargs)


def test_adamw_is_a_registered_torch_optimizer():
    parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
    optimizer = make_optimizer([parameter], lr=0.01)

    assert isinstance(optimizer, torch.optim.Optimizer)
    assert len(optimizer.param_groups) == 1
    assert optimizer.param_groups[0]["params"][0] is parameter
    assert isinstance(optimizer.state_dict(), dict)


def test_adamw_first_step_matches_pytorch_with_bias_correction():
    parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0], dtype=torch.float64))
    reference_parameter = torch.nn.Parameter(parameter.detach().clone())
    kwargs = dict(lr=0.1, betas=(0.8, 0.9), eps=1e-8, weight_decay=0.05)
    optimizer = make_optimizer([parameter], **kwargs)
    reference = torch.optim.AdamW([reference_parameter], **kwargs)
    gradient = torch.tensor([0.2, -0.4], dtype=torch.float64)
    parameter.grad = gradient.clone()
    reference_parameter.grad = gradient.clone()

    optimizer.step()
    reference.step()

    assert torch.allclose(parameter, reference_parameter, atol=1e-10, rtol=1e-10)


def test_adamw_multiple_steps_match_pytorch_with_changing_gradients():
    parameter = torch.nn.Parameter(torch.tensor([2.0, -1.0], dtype=torch.float64))
    reference_parameter = torch.nn.Parameter(parameter.detach().clone())
    kwargs = dict(lr=0.03, betas=(0.6, 0.8), eps=1e-5, weight_decay=0.1)
    optimizer = make_optimizer([parameter], **kwargs)
    reference = torch.optim.AdamW([reference_parameter], **kwargs)
    gradients = [(0.3, -0.5), (-0.2, -0.1), (0.4, 0.2)]

    for values in gradients:
        gradient = torch.tensor(values, dtype=torch.float64)
        parameter.grad = gradient.clone()
        reference_parameter.grad = gradient.clone()
        optimizer.step()
        reference.step()
        assert torch.allclose(
            parameter, reference_parameter, atol=1e-10, rtol=1e-10
        )


def test_adamw_applies_decoupled_decay_with_zero_gradient():
    parameter = torch.nn.Parameter(torch.tensor([2.0, -3.0], dtype=torch.float64))
    optimizer = make_optimizer(
        [parameter], lr=0.1, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.2
    )
    original = parameter.detach().clone()
    parameter.grad = torch.zeros_like(parameter)

    optimizer.step()

    expected = original * (1.0 - 0.1 * 0.2)
    assert torch.allclose(parameter, expected, atol=1e-12, rtol=1e-12)


def test_adamw_skips_none_gradient_and_tracks_steps_per_parameter():
    first = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float64))
    second = torch.nn.Parameter(torch.tensor([2.0], dtype=torch.float64))
    reference_first = torch.nn.Parameter(first.detach().clone())
    reference_second = torch.nn.Parameter(second.detach().clone())
    kwargs = dict(lr=0.1, betas=(0.5, 0.7), eps=1e-8, weight_decay=0.2)
    optimizer = make_optimizer([first, second], **kwargs)
    reference = torch.optim.AdamW([reference_first, reference_second], **kwargs)

    first.grad = torch.tensor([0.4], dtype=torch.float64)
    reference_first.grad = first.grad.clone()
    optimizer.step()
    reference.step()
    assert torch.equal(second, torch.tensor([2.0], dtype=torch.float64))

    first.grad = torch.tensor([-0.2], dtype=torch.float64)
    second.grad = torch.tensor([0.3], dtype=torch.float64)
    reference_first.grad = first.grad.clone()
    reference_second.grad = second.grad.clone()
    optimizer.step()
    reference.step()

    assert torch.allclose(first, reference_first, atol=1e-10, rtol=1e-10)
    assert torch.allclose(second, reference_second, atol=1e-10, rtol=1e-10)


def test_adamw_respects_parameter_group_hyperparameters():
    first = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float64))
    second = torch.nn.Parameter(torch.tensor([-2.0], dtype=torch.float64))
    reference_first = torch.nn.Parameter(first.detach().clone())
    reference_second = torch.nn.Parameter(second.detach().clone())
    groups = [
        {"params": [first], "lr": 0.01, "weight_decay": 0.0},
        {"params": [second], "lr": 0.2, "weight_decay": 0.3},
    ]
    reference_groups = [
        {"params": [reference_first], "lr": 0.01, "weight_decay": 0.0},
        {"params": [reference_second], "lr": 0.2, "weight_decay": 0.3},
    ]
    optimizer = make_optimizer(groups, betas=(0.8, 0.9), eps=1e-8)
    reference = torch.optim.AdamW(reference_groups, betas=(0.8, 0.9), eps=1e-8)
    first.grad = torch.tensor([0.5], dtype=torch.float64)
    second.grad = torch.tensor([-0.25], dtype=torch.float64)
    reference_first.grad = first.grad.clone()
    reference_second.grad = second.grad.clone()

    optimizer.step()
    reference.step()

    assert torch.allclose(first, reference_first, atol=1e-10, rtol=1e-10)
    assert torch.allclose(second, reference_second, atol=1e-10, rtol=1e-10)


def test_adamw_state_dict_restores_continuation():
    parameter = torch.nn.Parameter(torch.tensor([2.0, -1.0], dtype=torch.float64))
    kwargs = dict(lr=0.02, betas=(0.7, 0.8), eps=1e-8, weight_decay=0.05)
    optimizer = make_optimizer([parameter], **kwargs)
    parameter.grad = torch.tensor([0.4, -0.3], dtype=torch.float64)
    optimizer.step()

    resumed_parameter = torch.nn.Parameter(parameter.detach().clone())
    resumed_optimizer = make_optimizer([resumed_parameter], **kwargs)
    resumed_optimizer.load_state_dict(deepcopy(optimizer.state_dict()))
    next_gradient = torch.tensor([-0.1, 0.2], dtype=torch.float64)
    parameter.grad = next_gradient.clone()
    resumed_parameter.grad = next_gradient.clone()

    optimizer.step()
    resumed_optimizer.step()

    assert torch.allclose(
        parameter, resumed_parameter, atol=1e-10, rtol=1e-10
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
