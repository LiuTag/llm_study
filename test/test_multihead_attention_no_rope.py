"""不带 RoPE 的因果多头注意力验收测试；可直接执行此文件。"""

import _paths  # noqa: F401
import math

import pytest
import torch
import torch.nn.functional as F

from Attention import MultiHeadSelfAttention


def reference_no_rope_attention(
    x: torch.Tensor,
    num_heads: int,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    v_weight: torch.Tensor,
    output_weight: torch.Tensor,
) -> torch.Tensor:
    d_model = x.shape[-1]
    sequence_length = x.shape[-2]
    head_dim = d_model // num_heads
    leading_shape = x.shape[:-2]

    def split_heads(weight: torch.Tensor) -> torch.Tensor:
        projected = x @ weight.transpose(-1, -2)
        return projected.reshape(
            *leading_shape, sequence_length, num_heads, head_dim
        ).transpose(-3, -2)

    q = split_heads(q_weight)
    k = split_heads(k_weight)
    v = split_heads(v_weight)
    scores = q @ k.transpose(-1, -2) / math.sqrt(head_dim)
    causal_mask = torch.ones(
        sequence_length, sequence_length, dtype=torch.bool, device=x.device
    ).tril()
    scores = scores.masked_fill(~causal_mask, -torch.inf)
    values = F.softmax(scores, dim=-1) @ v
    combined = values.transpose(-3, -2).reshape(
        *leading_shape, sequence_length, d_model
    )
    return combined @ output_weight.transpose(-1, -2)


def test_no_rope_matches_independent_causal_multihead_reference():
    torch.manual_seed(0)
    x = torch.randn(2, 3, 5, 8)
    q_weight = torch.randn(8, 8) / 8
    k_weight = torch.randn(8, 8) / 8
    v_weight = torch.randn(8, 8) / 8
    output_weight = torch.randn(8, 8) / 8
    module = MultiHeadSelfAttention(
        d_model=8, num_heads=2, max_seq_len=5,
        theta=10000.0, use_rope=False,
    )
    with torch.no_grad():
        module.q_proj.weight.copy_(q_weight)
        module.k_proj.weight.copy_(k_weight)
        module.v_proj.weight.copy_(v_weight)
        module.output_proj.weight.copy_(output_weight)

    actual = module(x)
    expected = reference_no_rope_attention(
        x, 2, q_weight, k_weight, v_weight, output_weight
    )

    assert actual.shape == x.shape
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)


def test_no_rope_output_does_not_depend_on_token_positions():
    torch.manual_seed(1)
    module = MultiHeadSelfAttention(
        d_model=8, num_heads=2, max_seq_len=12,
        theta=10000.0, use_rope=False,
    )
    x = torch.randn(2, 5, 8)
    first_positions = torch.arange(5)
    different_positions = torch.tensor([6, 7, 8, 9, 10])

    default_output = module(x)
    first_output = module(x, first_positions)
    different_output = module(x, different_positions)

    assert torch.allclose(default_output, first_output, atol=1e-6, rtol=1e-6)
    assert torch.allclose(default_output, different_output, atol=1e-6, rtol=1e-6)


def test_no_rope_still_cannot_read_future_tokens():
    torch.manual_seed(2)
    module = MultiHeadSelfAttention(
        d_model=8, num_heads=2, max_seq_len=6,
        theta=10000.0, use_rope=False,
    )
    x = torch.randn(2, 6, 8)
    changed = x.clone()
    changed[:, 3:] = torch.randn_like(changed[:, 3:])

    original_output = module(x)
    changed_output = module(changed)

    assert torch.allclose(
        original_output[:, :3], changed_output[:, :3], atol=1e-6, rtol=1e-6
    )


def test_no_rope_backpropagates_through_all_four_projection_matrices():
    torch.manual_seed(3)
    module = MultiHeadSelfAttention(
        d_model=8, num_heads=2, max_seq_len=4,
        theta=10000.0, use_rope=False,
    )
    x = torch.randn(2, 4, 8, requires_grad=True)

    module(x).square().mean().backward()

    assert x.grad is not None and torch.isfinite(x.grad).all()
    for projection in (
        module.q_proj, module.k_proj, module.v_proj, module.output_proj
    ):
        assert projection.weight.grad is not None
        assert torch.isfinite(projection.weight.grad).all()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
