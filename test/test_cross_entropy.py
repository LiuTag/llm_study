"""Cross-Entropy 的验收测试；可用 python test_cross_entropy.py 直接运行。"""

import _paths  # noqa: F401
import importlib
import importlib.util
import math

import pytest
import torch
import torch.nn.functional as F


def get_cross_entropy():
    specification = importlib.util.find_spec("CrossEntropy")
    assert specification is not None, (
        "请先在 CrossEntropy.py 中实现 cross_entropy"
    )
    function = getattr(
        importlib.import_module("CrossEntropy"),
        "cross_entropy",
        None,
    )
    assert callable(function), (
        "请先在 CrossEntropy.py 中实现 cross_entropy(logits, targets)"
    )
    return function


def test_cross_entropy_uniform_logits_equal_log_vocab_size():
    cross_entropy = get_cross_entropy()
    logits = torch.zeros(2, 3)
    targets = torch.tensor([0, 2], dtype=torch.long)

    loss = cross_entropy(logits, targets)

    assert loss.shape == torch.Size([])
    assert torch.allclose(loss, torch.tensor(math.log(3)), atol=1e-6)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cross_entropy_matches_pytorch_for_batched_classification(dtype):
    cross_entropy = get_cross_entropy()
    logits = torch.tensor(
        [[2.0, -1.0, 0.5, 3.0], [-2.0, 4.0, 1.0, 0.0], [0.2, 0.4, -0.3, 0.1]],
        dtype=dtype,
    )
    targets = torch.tensor([3, 0, 1], dtype=torch.long)

    actual = cross_entropy(logits, targets)
    expected = F.cross_entropy(logits, targets)

    assert actual.shape == torch.Size([])
    assert actual.dtype == dtype
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_cross_entropy_accepts_language_model_logits_and_averages_all_positions():
    cross_entropy = get_cross_entropy()
    torch.manual_seed(0)
    logits = torch.randn(2, 5, 7)
    targets = torch.tensor(
        [[1, 2, 3, 4, 5], [6, 0, 3, 2, 1]],
        dtype=torch.long,
    )

    actual = cross_entropy(logits, targets)
    expected = F.cross_entropy(logits.reshape(-1, 7), targets.reshape(-1))

    assert actual.shape == torch.Size([])
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_cross_entropy_is_stable_for_large_positive_and_negative_logits():
    cross_entropy = get_cross_entropy()
    logits = torch.tensor(
        [[10000.0, 10001.0, 9999.0], [-10000.0, -10001.0, -9999.0]]
    )
    targets = torch.tensor([1, 2], dtype=torch.long)

    actual = cross_entropy(logits, targets)
    expected = F.cross_entropy(logits, targets)

    assert torch.isfinite(actual)
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_cross_entropy_is_invariant_to_per_position_logit_shifts():
    cross_entropy = get_cross_entropy()
    logits = torch.tensor(
        [[[0.2, -1.5, 3.0], [4.0, 0.0, -2.0]],
         [[-3.0, 0.8, 1.2], [0.1, 0.3, 0.7]]],
        dtype=torch.float64,
    )
    targets = torch.tensor([[2, 0], [1, 2]], dtype=torch.long)
    shifts = torch.tensor([[1e8, -1e8], [1e8, -1e8]])[..., None]

    original = cross_entropy(logits, targets)
    shifted = cross_entropy(logits + shifts, targets)

    assert torch.allclose(original, shifted, atol=1e-7, rtol=1e-7)


def test_cross_entropy_gradient_matches_pytorch():
    cross_entropy = get_cross_entropy()
    torch.manual_seed(1)
    values = torch.randn(2, 3, 5, dtype=torch.float64)
    targets = torch.tensor([[0, 1, 4], [3, 2, 0]], dtype=torch.long)
    actual_logits = values.clone().requires_grad_()
    reference_logits = values.clone().requires_grad_()

    actual_loss = cross_entropy(actual_logits, targets)
    reference_loss = F.cross_entropy(
        reference_logits.reshape(-1, 5), targets.reshape(-1)
    )
    actual_loss.backward()
    reference_loss.backward()

    assert actual_logits.grad is not None
    assert torch.isfinite(actual_logits.grad).all()
    assert torch.allclose(
        actual_logits.grad,
        reference_logits.grad,
        atol=1e-8,
        rtol=1e-8,
    )


def test_cross_entropy_does_not_modify_logits_or_targets():
    cross_entropy = get_cross_entropy()
    logits = torch.tensor([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
    targets = torch.tensor([2, 0], dtype=torch.long)
    original_logits = logits.clone()
    original_targets = targets.clone()

    cross_entropy(logits, targets)

    assert torch.equal(logits, original_logits)
    assert torch.equal(targets, original_targets)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
