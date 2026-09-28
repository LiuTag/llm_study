import _paths  # noqa: F401
import pytest
import torch

from Embedding import Linear


def test_linear_matches_matrix_multiplication():
    layer = Linear(in_features=3, out_features=2)
    weight = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [-1.0, 0.5, 2.0],
        ]
    )
    with torch.no_grad():
        layer.weight.copy_(weight)

    x = torch.tensor(
        [
            [[1.0, 0.0, 2.0], [0.0, 1.0, 1.0]],
            [[2.0, -1.0, 0.0], [1.0, 1.0, 1.0]],
        ]
    )

    actual = layer(x)
    expected = x @ weight.T

    assert actual.shape == (2, 2, 2)
    assert torch.equal(actual, expected)


def test_linear_has_no_bias_and_supports_gradients():
    layer = Linear(in_features=4, out_features=5)
    x = torch.randn(2, 3, 4, requires_grad=True)

    output = layer(x)
    output.sum().backward()

    assert not hasattr(layer, "bias")
    assert output.shape == (2, 3, 5)
    assert x.grad is not None
    assert layer.weight.grad is not None
    assert torch.isfinite(x.grad).all()
    assert torch.isfinite(layer.weight.grad).all()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
