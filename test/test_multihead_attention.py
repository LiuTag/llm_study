import _paths  # noqa: F401
import math

import pytest
import torch

import Attention


def make_attention(
    d_model: int = 8,
    num_heads: int = 2,
    max_seq_len: int = 32,
    theta: float = 10000.0,
):
    attention_class = getattr(Attention, "MultiHeadSelfAttention", None)
    assert attention_class is not None, (
        "请先在 Attention.py 中实现 MultiHeadSelfAttention"
    )
    return attention_class(
        d_model=d_model,
        num_heads=num_heads,
        max_seq_len=max_seq_len,
        theta=theta,
    )


def reference_rope(
    x: torch.Tensor,
    token_positions: torch.Tensor,
    theta: float,
) -> torch.Tensor:
    head_dim = x.shape[-1]
    frequencies = theta ** (
        -torch.arange(0, head_dim, 2, device=x.device, dtype=x.dtype)
        / head_dim
    )
    angles = token_positions.to(dtype=x.dtype)[..., None] * frequencies
    cos = torch.cos(angles)
    sin = torch.sin(angles)

    while cos.ndim < x.ndim:
        cos = cos.unsqueeze(-3)
        sin = sin.unsqueeze(-3)

    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    rotated_even = x_even * cos - x_odd * sin
    rotated_odd = x_even * sin + x_odd * cos
    return torch.stack((rotated_even, rotated_odd), dim=-1).flatten(-2)


def reference_multihead_self_attention(
    x: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    v_weight: torch.Tensor,
    output_weight: torch.Tensor,
    num_heads: int,
    theta: float,
    token_positions: torch.Tensor,
) -> torch.Tensor:
    sequence_length = x.shape[-2]
    d_model = x.shape[-1]
    head_dim = d_model // num_heads
    leading_shape = x.shape[:-2]

    def project_and_split(weight: torch.Tensor) -> torch.Tensor:
        projected = x @ weight.transpose(-1, -2)
        projected = projected.reshape(
            *leading_shape,
            sequence_length,
            num_heads,
            head_dim,
        )
        return projected.transpose(-3, -2)

    q = reference_rope(project_and_split(q_weight), token_positions, theta)
    k = reference_rope(project_and_split(k_weight), token_positions, theta)
    v = project_and_split(v_weight)

    scores = q @ k.transpose(-1, -2)
    scores = scores / math.sqrt(head_dim)
    causal_mask = torch.tril(
        torch.ones(
            sequence_length,
            sequence_length,
            dtype=torch.bool,
            device=x.device,
        )
    )
    scores = scores.masked_fill(~causal_mask, float("-inf"))
    per_head_output = torch.softmax(scores, dim=-1) @ v

    merged = per_head_output.transpose(-3, -2).reshape(
        *leading_shape,
        sequence_length,
        d_model,
    )
    return merged @ output_weight.transpose(-1, -2)


def test_multihead_attention_has_four_full_projection_matrices():
    d_model = 8
    attention = make_attention(d_model=d_model, num_heads=2)

    expected_shape = (d_model, d_model)
    assert attention.q_proj.weight.shape == expected_shape
    assert attention.k_proj.weight.shape == expected_shape
    assert attention.v_proj.weight.shape == expected_shape
    assert attention.output_proj.weight.shape == expected_shape
    assert sum(parameter.numel() for parameter in attention.parameters()) == 4 * d_model**2


def test_multihead_attention_rejects_incompatible_dimensions():
    attention_class = getattr(Attention, "MultiHeadSelfAttention", None)
    assert attention_class is not None, (
        "请先在 Attention.py 中实现 MultiHeadSelfAttention"
    )

    with pytest.raises(ValueError):
        attention_class(
            d_model=10,
            num_heads=3,
            max_seq_len=16,
            theta=10000.0,
        )

    with pytest.raises(ValueError):
        attention_class(
            d_model=6,
            num_heads=2,
            max_seq_len=16,
            theta=10000.0,
        )


def test_multihead_attention_preserves_arbitrary_leading_dimensions():
    attention = make_attention(d_model=8, num_heads=2)
    x = torch.randn(2, 3, 5, 8)

    output = attention(x)

    assert output.shape == x.shape
    assert torch.isfinite(output).all()


def test_multihead_attention_matches_independent_reference():
    torch.manual_seed(0)
    d_model = 8
    num_heads = 2
    theta = 10000.0
    attention = make_attention(
        d_model=d_model,
        num_heads=num_heads,
        max_seq_len=32,
        theta=theta,
    )

    with torch.no_grad():
        for parameter in attention.parameters():
            parameter.copy_(torch.randn_like(parameter) * 0.2)

    x = torch.randn(2, 4, d_model)
    token_positions = torch.tensor(
        [
            [0, 1, 3, 6],
            [2, 4, 5, 9],
        ]
    )

    actual = attention(x, token_positions)
    expected = reference_multihead_self_attention(
        x=x,
        q_weight=attention.q_proj.weight,
        k_weight=attention.k_proj.weight,
        v_weight=attention.v_proj.weight,
        output_weight=attention.output_proj.weight,
        num_heads=num_heads,
        theta=theta,
        token_positions=token_positions,
    )

    assert actual.shape == x.shape
    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_multihead_attention_default_positions_match_explicit_positions():
    torch.manual_seed(1)
    attention = make_attention(d_model=8, num_heads=2)
    x = torch.randn(2, 5, 8)
    positions = torch.arange(x.shape[-2], device=x.device)

    default_output = attention(x)
    explicit_output = attention(x, positions)

    assert torch.allclose(default_output, explicit_output, atol=1e-6, rtol=1e-6)


def test_multihead_attention_is_causal():
    torch.manual_seed(2)
    attention = make_attention(d_model=8, num_heads=2)
    x = torch.randn(2, 6, 8)
    changed_x = x.clone()
    changed_x[:, 3:] += 1000.0

    original_output = attention(x)
    changed_output = attention(changed_x)

    assert torch.allclose(
        original_output[:, :3],
        changed_output[:, :3],
        atol=1e-5,
        rtol=1e-5,
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_multihead_attention_preserves_low_precision_dtype(dtype):
    attention = make_attention(d_model=8, num_heads=2).to(dtype=dtype)
    x = torch.randn(2, 4, 8, dtype=dtype)

    output = attention(x)

    assert output.shape == x.shape
    assert output.dtype == dtype
    assert torch.isfinite(output).all()


def test_multihead_attention_supports_gradients():
    torch.manual_seed(3)
    attention = make_attention(d_model=8, num_heads=2)
    x = torch.randn(2, 5, 8, requires_grad=True)

    output = attention(x)
    output.square().mean().backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    for name in ("q_proj", "k_proj", "v_proj", "output_proj"):
        gradient = getattr(attention, name).weight.grad
        assert gradient is not None
        assert torch.isfinite(gradient).all()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
