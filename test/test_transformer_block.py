"""Transformer Block 的验收测试；可用 python test_transformer_block.py 直接运行。"""

import _paths  # noqa: F401
import importlib
import importlib.util

import pytest
import torch

from Attention import MultiHeadSelfAttention
from RMSNorm import RMSNorm
from SwiGLU import SwiGLU


def make_block(
    d_model: int = 8,
    num_heads: int = 2,
    d_ff: int = 24,
    max_seq_len: int = 32,
    theta: float = 10000.0,
):
    specification = importlib.util.find_spec("TransformerBlock")
    assert specification is not None, (
        "请先在 TransformerBlock.py 中实现 TransformerBlock"
    )
    block_class = getattr(
        importlib.import_module("TransformerBlock"),
        "TransformerBlock",
        None,
    )
    assert block_class is not None, (
        "请先在 TransformerBlock.py 中实现 TransformerBlock 类"
    )
    return block_class(
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        max_seq_len=max_seq_len,
        theta=theta,
    )


def test_transformer_block_has_independent_registered_submodules():
    block = make_block()

    assert isinstance(block, torch.nn.Module)
    assert isinstance(block.ln1, RMSNorm)
    assert isinstance(block.attn, MultiHeadSelfAttention)
    assert isinstance(block.ln2, RMSNorm)
    assert isinstance(block.ffn, SwiGLU)
    assert block.ln1 is not block.ln2
    assert block.ln1.weight.data_ptr() != block.ln2.weight.data_ptr()
    assert block.ln1.weight.shape == (8,)
    assert block.ln2.weight.shape == (8,)
    assert {"ln1.weight", "ln2.weight"}.issubset(block.state_dict())


def test_transformer_block_matches_prenorm_residual_formula_with_positions():
    torch.manual_seed(0)
    block = make_block()
    x = torch.randn(2, 4, 8)
    positions = torch.tensor([[0, 1, 3, 6], [2, 4, 5, 9]])

    actual = block(x, positions)
    attention_residual = x + block.attn(block.ln1(x), positions)
    expected = attention_residual + block.ffn(block.ln2(attention_residual))

    assert actual.shape == x.shape
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_transformer_block_default_positions_match_explicit_positions():
    torch.manual_seed(1)
    block = make_block()
    x = torch.randn(2, 5, 8)
    positions = torch.arange(x.shape[-2], device=x.device)

    assert torch.allclose(
        block(x),
        block(x, positions),
        atol=1e-6,
        rtol=1e-6,
    )


@pytest.mark.parametrize("shape", [(5, 8), (2, 3, 5, 8)])
def test_transformer_block_preserves_shape_with_leading_dimensions(shape):
    block = make_block()
    x = torch.randn(*shape)

    output = block(x)

    assert output.shape == x.shape
    assert torch.isfinite(output).all()


def test_transformer_block_returns_input_when_both_branches_are_zero():
    torch.manual_seed(2)
    block = make_block()
    x = torch.randn(2, 5, 8)
    with torch.no_grad():
        block.attn.output_proj.weight.zero_()
        block.ffn.w2.weight.zero_()

    output = block(x)

    assert torch.equal(output, x)


def test_transformer_block_cannot_read_future_tokens():
    torch.manual_seed(3)
    block = make_block()
    x = torch.randn(2, 6, 8)
    changed_x = x.clone()
    changed_x[:, 3:] += 100.0

    assert torch.allclose(
        block(x)[:, :3],
        block(changed_x)[:, :3],
        atol=1e-5,
        rtol=1e-5,
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_transformer_block_preserves_low_precision_dtype(dtype):
    block = make_block().to(dtype=dtype)
    x = torch.randn(2, 4, 8, dtype=dtype)

    output = block(x)

    assert output.shape == x.shape
    assert output.dtype == dtype
    assert torch.isfinite(output).all()


def test_transformer_block_backpropagates_through_both_branches():
    torch.manual_seed(4)
    block = make_block()
    x = torch.randn(2, 5, 8, requires_grad=True)

    block(x).square().mean().backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    for name, parameter in block.named_parameters():
        assert parameter.grad is not None, f"{name} 没有梯度"
        assert torch.isfinite(parameter.grad).all(), f"{name} 的梯度不是有限值"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
