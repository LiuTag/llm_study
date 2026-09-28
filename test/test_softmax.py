import _paths  # noqa: F401
import pytest
import torch

from Attention import softmax


def test_softmax_matches_pytorch():
    torch.manual_seed(0)
    x = torch.randn(2, 3, 4)

    actual = softmax(x, dim=-1)
    expected = torch.softmax(x, dim=-1)

    assert actual.shape == x.shape
    assert torch.all(actual >= 0)
    assert torch.allclose(actual.sum(dim=-1), torch.ones(2, 3), atol=1e-6)
    assert torch.allclose(actual, expected, atol=1e-6)


@pytest.mark.parametrize("dim", [0, 1, -1])
def test_softmax_normalizes_the_requested_dimension(dim):
    x = torch.randn(2, 3, 4)
    output = softmax(x, dim=dim)

    expected_sum = torch.ones_like(output.sum(dim=dim))

    assert output.shape == x.shape
    assert torch.allclose(output.sum(dim=dim), expected_sum, atol=1e-6)


def test_softmax_is_stable_for_large_logits():
    x = torch.tensor([1000.0, 1001.0, 1002.0])
    output = softmax(x, dim=-1)

    assert torch.isfinite(output).all()
    assert torch.allclose(output.sum(), torch.tensor(1.0))
    assert torch.allclose(output, torch.softmax(x, dim=-1), atol=1e-6)


def test_softmax_float32_moderate_translation_invariance():
    torch.manual_seed(0)
    x = torch.randn(2, 3, 4, dtype=torch.float32)

    assert torch.allclose(
        softmax(x, dim=-1),
        softmax(x + 10.0, dim=-1),
        atol=1e-6,
        rtol=1e-5,
    )


def test_softmax_float64_large_translation_invariance():
    torch.manual_seed(0)
    x = torch.randn(2, 3, 4, dtype=torch.float64)

    assert torch.allclose(
        softmax(x, dim=-1),
        softmax(x + 1e8, dim=-1),
        atol=1e-8,
        rtol=1e-8,
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_softmax_preserves_low_precision_dtype(dtype):
    x = torch.randn(2, 3, 4, dtype=dtype)
    output = softmax(x, dim=-1)

    assert output.dtype == dtype
    assert torch.isfinite(output).all()
    assert torch.allclose(
        output.float(),
        torch.softmax(x.float(), dim=-1),
        atol=2e-3,
        rtol=2e-3,
    )


def test_softmax_supports_gradients():
    x = torch.randn(2, 3, 4, requires_grad=True)

    softmax(x, dim=-1).square().sum().backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
