"""梯度全局范数裁剪验收测试；可用 python test_gradient_clipping.py 直接运行。"""

import _paths  # noqa: F401
import importlib
import importlib.util

import pytest
import torch


def get_gradient_clipping():
    assert importlib.util.find_spec("GradientClipping") is not None, (
        "请先创建 GradientClipping.py"
    )
    function = getattr(
        importlib.import_module("GradientClipping"), "gradient_clipping", None
    )
    assert callable(function), (
        "请先在 GradientClipping.py 中实现 gradient_clipping"
    )
    return function


def test_gradient_clipping_uses_one_global_norm_for_all_parameters():
    gradient_clipping = get_gradient_clipping()
    first = torch.nn.Parameter(torch.tensor([10.0, 20.0], dtype=torch.float64))
    second = torch.nn.Parameter(torch.tensor([-3.0], dtype=torch.float64))
    original_first = first.detach().clone()
    original_second = second.detach().clone()
    first.grad = torch.tensor([3.0, 4.0], dtype=torch.float64)
    second.grad = torch.tensor([12.0], dtype=torch.float64)
    first_grad_object = first.grad
    second_grad_object = second.grad

    result = gradient_clipping([first, second], max_l2_norm=6.5)

    assert result is None
    assert first.grad is first_grad_object
    assert second.grad is second_grad_object
    assert torch.allclose(first.grad, torch.tensor([1.5, 2.0], dtype=torch.float64))
    assert torch.allclose(second.grad, torch.tensor([6.0], dtype=torch.float64))
    assert torch.equal(first, original_first)
    assert torch.equal(second, original_second)


def test_gradient_clipping_matches_pytorch_and_skips_none_gradients():
    gradient_clipping = get_gradient_clipping()
    torch.manual_seed(0)
    parameters = [
        torch.nn.Parameter(torch.randn(5, 5, dtype=torch.float64))
        for _ in range(4)
    ]
    reference = [
        torch.nn.Parameter(parameter.detach().clone())
        for parameter in parameters
    ]
    for parameter, expected_parameter in zip(parameters[:-1], reference[:-1]):
        grad = torch.randn_like(parameter)
        parameter.grad = grad.clone()
        expected_parameter.grad = grad.clone()
    assert parameters[-1].grad is None

    gradient_clipping(parameters, max_l2_norm=0.1)
    torch.nn.utils.clip_grad_norm_(reference, max_norm=0.1)

    for parameter, expected_parameter in zip(parameters[:-1], reference[:-1]):
        assert torch.allclose(
            parameter.grad, expected_parameter.grad, atol=1e-6, rtol=1e-6
        )
    assert parameters[-1].grad is None


def test_gradient_clipping_leaves_small_and_zero_gradients_unchanged():
    gradient_clipping = get_gradient_clipping()
    first = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    second = torch.nn.Parameter(torch.tensor([3.0]))
    first.grad = torch.tensor([0.1, 0.2])
    second.grad = torch.zeros_like(second)
    initial_first = first.grad.clone()
    initial_second = second.grad.clone()

    gradient_clipping([first, second], max_l2_norm=1.0)

    assert torch.equal(first.grad, initial_first)
    assert torch.equal(second.grad, initial_second)


def test_gradient_clipping_accepts_a_generator_and_empty_gradients():
    gradient_clipping = get_gradient_clipping()
    first = torch.nn.Parameter(torch.tensor([0.0]))
    second = torch.nn.Parameter(torch.tensor([0.0]))
    first.grad = torch.tensor([4.0])

    gradient_clipping((parameter for parameter in (first, second)), 2.0)
    assert torch.allclose(first.grad, torch.tensor([2.0]), atol=1e-6)
    assert second.grad is None
    assert gradient_clipping([], 2.0) is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
