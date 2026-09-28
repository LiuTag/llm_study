import _paths  # noqa: F401
import pytest
import torch

from RMSNorm import RMSNorm


def test_rmsnorm_matches_reference_formula():
    eps = 1e-5
    layer = RMSNorm(d_model=4, eps=eps)
    weight = torch.tensor([1.0, 0.5, 2.0, -1.0])
    with torch.no_grad():
        layer.weight.copy_(weight)

    x = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0],
            [-2.0, 0.0, 1.0, 3.0],
        ]
    )
    expected = x / torch.sqrt(x.square().mean(dim=-1, keepdim=True) + eps)
    expected = expected * weight

    actual = layer(x)

    assert actual.shape == x.shape
    assert torch.allclose(actual, expected, atol=1e-6)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_rmsnorm_preserves_low_precision_dtype(dtype):
    layer = RMSNorm(d_model=4)
    x = torch.randn(2, 3, 4, dtype=dtype)

    output = layer(x)

    assert output.dtype == dtype
    assert torch.isfinite(output).all()


def test_rmsnorm_handles_zero_input_and_gradients():
    layer = RMSNorm(d_model=4)
    x = torch.zeros(2, 4, requires_grad=True)

    output = layer(x)
    output.sum().backward()

    assert torch.equal(output, torch.zeros_like(output))
    assert x.grad is not None
    assert layer.weight.grad is not None
    assert torch.isfinite(x.grad).all()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
