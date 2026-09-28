import _paths  # noqa: F401
import pytest
import torch

from RoPE import RotaryPositionalEmbedding


def test_rope_preserves_shape_and_position_zero():
    rope = RotaryPositionalEmbedding(theta=10000.0, d_k=4, max_seq_len=16)
    x = torch.randn(2, 5, 4)
    positions = torch.arange(5)

    output = rope(x, positions)

    assert output.shape == x.shape
    assert torch.allclose(output[:, 0], x[:, 0], atol=1e-6)


def test_rope_preserves_each_two_dimensional_norm():
    rope = RotaryPositionalEmbedding(theta=10000.0, d_k=8, max_seq_len=16)
    x = torch.randn(2, 3, 5, 8)

    output = rope(x, torch.arange(5))
    before = x.reshape(2, 3, 5, 4, 2).square().sum(dim=-1)
    after = output.reshape(2, 3, 5, 4, 2).square().sum(dim=-1)

    assert torch.allclose(before, after, atol=1e-5)


def test_rope_broadcasts_batch_positions_across_heads():
    rope = RotaryPositionalEmbedding(theta=10000.0, d_k=4, max_seq_len=16)
    x = torch.randn(2, 3, 5, 4)
    positions = torch.arange(5).repeat(2, 1)

    output = rope(x, positions)

    assert output.shape == x.shape
    assert torch.allclose(output[:, :, 0], x[:, :, 0], atol=1e-6)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_rope_preserves_low_precision_dtype(dtype):
    rope = RotaryPositionalEmbedding(theta=10000.0, d_k=4, max_seq_len=16).to(dtype=dtype)
    x = torch.randn(2, 5, 4, dtype=dtype)

    output = rope(x, torch.arange(5))

    assert output.dtype == dtype
    assert torch.isfinite(output).all()
    assert torch.allclose(output[:, 0], x[:, 0], atol=2e-3)


def test_rope_has_buffers_but_no_parameters_or_persistent_state():
    rope = RotaryPositionalEmbedding(theta=10000.0, d_k=4, max_seq_len=16)

    assert set(dict(rope.named_buffers())) == {"cos", "sin"}
    assert list(rope.parameters()) == []
    assert list(rope.state_dict()) == []


def test_rope_supports_gradients():
    rope = RotaryPositionalEmbedding(theta=10000.0, d_k=4, max_seq_len=16)
    x = torch.randn(2, 5, 4, requires_grad=True)

    rope(x, torch.arange(5)).sum().backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


@pytest.mark.parametrize("d_k", [0, -2, 3, 5])
def test_rope_rejects_non_positive_or_odd_dimensions(d_k):
    with pytest.raises(ValueError):
        RotaryPositionalEmbedding(theta=10000.0, d_k=d_k, max_seq_len=16)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
