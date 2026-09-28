import _paths  # noqa: F401
import pytest
import torch

from Attention import scaled_dot_product_attention


def reference_scaled_dot_product_attention(q, k, v, mask=None):
    d_k = q.shape[-1]
    scores = q @ k.transpose(-2, -1)
    scores = scores / (d_k**0.5)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    return torch.softmax(scores, dim=-1) @ v


def test_scaled_dot_product_attention_uniform_scores_average_values():
    q = torch.zeros(1, 2, 4)
    k = torch.zeros(1, 3, 4)
    v = torch.tensor(
        [
            [
                [1.0, 0.0],
                [0.0, 2.0],
                [3.0, 4.0],
            ]
        ]
    )

    output = scaled_dot_product_attention(q, k, v)
    expected = v.mean(dim=-2, keepdim=True).expand(1, 2, 2)

    assert output.shape == (1, 2, 2)
    assert torch.allclose(output, expected, atol=1e-6)


def test_scaled_dot_product_attention_boolean_mask_semantics():
    q = torch.zeros(1, 2, 4)
    k = torch.zeros(1, 3, 4)
    v = torch.tensor(
        [
            [
                [1.0, 0.0],
                [0.0, 2.0],
                [3.0, 4.0],
            ]
        ]
    )
    mask = torch.tensor(
        [
            [True, False, False],
            [False, True, True],
        ]
    )

    output = scaled_dot_product_attention(q, k, v, mask)
    expected = torch.stack(
        [
            v[0, 0],
            (v[0, 1] + v[0, 2]) / 2,
        ]
    ).unsqueeze(0)

    assert torch.allclose(output, expected, atol=1e-6)


def test_scaled_dot_product_attention_supports_different_sequence_lengths_and_dv():
    torch.manual_seed(0)
    q = torch.randn(2, 4, 8)
    k = torch.randn(2, 6, 8)
    v = torch.randn(2, 6, 5)

    output = scaled_dot_product_attention(q, k, v)
    expected = reference_scaled_dot_product_attention(q, k, v)

    assert output.shape == (2, 4, 5)
    assert torch.allclose(output, expected, atol=1e-6)


def test_scaled_dot_product_attention_supports_multihead_input_and_broadcast_mask():
    torch.manual_seed(1)
    batch_size, num_heads, sequence_length, head_dim = 2, 3, 5, 4
    q = torch.randn(batch_size, num_heads, sequence_length, head_dim)
    k = torch.randn(batch_size, num_heads, sequence_length, head_dim)
    v = torch.randn(batch_size, num_heads, sequence_length, head_dim)
    causal_mask = torch.tril(
        torch.ones(sequence_length, sequence_length, dtype=torch.bool)
    )

    output = scaled_dot_product_attention(q, k, v, causal_mask)
    expected = reference_scaled_dot_product_attention(q, k, v, causal_mask)

    assert output.shape == q.shape
    assert torch.isfinite(output).all()
    assert torch.allclose(output, expected, atol=1e-5)


def test_scaled_dot_product_attention_causal_mask_blocks_future_values():
    torch.manual_seed(2)
    sequence_length, head_dim = 5, 4
    q = torch.randn(1, sequence_length, head_dim)
    k = torch.randn(1, sequence_length, head_dim)
    v = torch.randn(1, sequence_length, head_dim)
    changed_v = v.clone()
    changed_v[:, 1:] += 1000.0
    causal_mask = torch.tril(
        torch.ones(sequence_length, sequence_length, dtype=torch.bool)
    )

    original_output = scaled_dot_product_attention(q, k, v, causal_mask)
    changed_output = scaled_dot_product_attention(q, k, changed_v, causal_mask)

    assert torch.allclose(
        original_output[:, 0],
        changed_output[:, 0],
        atol=1e-6,
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_scaled_dot_product_attention_preserves_low_precision_dtype(dtype):
    q = torch.randn(2, 3, 4, dtype=dtype)
    k = torch.randn(2, 3, 4, dtype=dtype)
    v = torch.randn(2, 3, 6, dtype=dtype)

    output = scaled_dot_product_attention(q, k, v)

    assert output.shape == (2, 3, 6)
    assert output.dtype == dtype
    assert torch.isfinite(output).all()


def test_scaled_dot_product_attention_supports_gradients():
    q = torch.randn(2, 3, 4, requires_grad=True)
    k = torch.randn(2, 5, 4, requires_grad=True)
    v = torch.randn(2, 5, 6, requires_grad=True)

    output = scaled_dot_product_attention(q, k, v)
    output.square().sum().backward()

    for tensor in (q, k, v):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
