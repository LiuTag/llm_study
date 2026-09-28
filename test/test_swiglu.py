import _paths  # noqa: F401
import pytest
import torch

from SwiGLU import SwiGLU


def test_swiglu_matches_reference_formula():
    layer = SwiGLU(d_model=3, d_ff=4)
    w1 = torch.tensor(
        [
            [1.0, 0.0, -1.0],
            [0.5, 1.0, 0.0],
            [0.0, -1.0, 1.0],
            [1.0, 1.0, 1.0],
        ]
    )
    w3 = torch.tensor(
        [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [-1.0, 1.0, 0.0],
            [0.5, 0.5, 0.5],
        ]
    )
    w2 = torch.tensor(
        [
            [1.0, 0.0, 0.5, -1.0],
            [0.0, 1.0, -0.5, 0.5],
            [0.5, -1.0, 1.0, 0.0],
        ]
    )
    with torch.no_grad():
        layer.w1.weight.copy_(w1)
        layer.w3.weight.copy_(w3)
        layer.w2.weight.copy_(w2)

    x = torch.tensor(
        [
            [[1.0, 2.0, -1.0], [0.0, 1.0, 1.0]],
            [[-1.0, 0.5, 2.0], [2.0, -1.0, 0.0]],
        ]
    )

    a = x @ w1.T
    b = x @ w3.T
    expected = (torch.sigmoid(a) * a * b) @ w2.T
    actual = layer(x)

    assert actual.shape == x.shape
    assert torch.allclose(actual, expected, atol=1e-6)


def test_swiglu_supports_gradients():
    layer = SwiGLU(d_model=4, d_ff=8)
    x = torch.randn(2, 3, 4, requires_grad=True)

    output = layer(x)
    output.sum().backward()

    assert output.shape == x.shape
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    for parameter in layer.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
