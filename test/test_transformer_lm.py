"""Transformer LM 的验收测试；可用 python test_transformer_lm.py 直接运行。"""

import _paths  # noqa: F401
import importlib
import importlib.util

import pytest
import torch

from Embedding import Embedding, Linear
from RMSNorm import RMSNorm
from TransformerBlock import TransformerBlock


def make_model(
    vocab_size: int = 31,
    context_length: int = 16,
    d_model: int = 8,
    num_layers: int = 2,
    num_heads: int = 2,
    d_ff: int = 24,
    rope_theta: float = 10000.0,
):
    specification = importlib.util.find_spec("TransformerLM")
    assert specification is not None, (
        "请先在 TransformerLM.py 中实现 TransformerLM"
    )
    model_class = getattr(
        importlib.import_module("TransformerLM"),
        "TransformerLM",
        None,
    )
    assert model_class is not None, (
        "请先在 TransformerLM.py 中实现 TransformerLM 类"
    )
    return model_class(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=rope_theta,
    )


def test_transformer_lm_registers_independent_layers_and_components():
    model = make_model(num_layers=3)

    assert isinstance(model, torch.nn.Module)
    assert isinstance(model.token_embeddings, Embedding)
    assert isinstance(model.layers, torch.nn.ModuleList)
    assert len(model.layers) == 3
    assert all(isinstance(block, TransformerBlock) for block in model.layers)
    assert len({id(block) for block in model.layers}) == 3
    assert len({block.ln1.weight.data_ptr() for block in model.layers}) == 3
    assert isinstance(model.ln_final, RMSNorm)
    assert isinstance(model.lm_head, Linear)
    assert model.token_embeddings.weight.shape == (31, 8)
    assert model.lm_head.weight.shape == (31, 8)
    state = model.state_dict()
    assert "token_embeddings.weight" in state
    assert "layers.0.ln1.weight" in state
    assert "layers.2.ln1.weight" in state
    assert "ln_final.weight" in state
    assert "lm_head.weight" in state


@pytest.mark.parametrize("batch_size,seq_len", [(1, 1), (2, 5), (3, 7)])
def test_transformer_lm_maps_token_ids_to_vocab_logits(batch_size, seq_len):
    model = make_model()
    token_ids = torch.randint(0, 31, (batch_size, seq_len))

    logits = model(token_ids)

    assert logits.shape == (batch_size, seq_len, 31)
    assert logits.is_floating_point()
    assert torch.isfinite(logits).all()


def test_transformer_lm_matches_full_forward_formula_with_positions():
    torch.manual_seed(0)
    model = make_model()
    token_ids = torch.randint(0, 31, (2, 4))
    positions = torch.tensor([[0, 1, 3, 5], [2, 4, 6, 8]])

    actual = model(token_ids, positions)
    hidden = model.token_embeddings(token_ids)
    for block in model.layers:
        hidden = block(hidden, positions)
    expected = model.lm_head(model.ln_final(hidden))

    assert actual.shape == (2, 4, 31)
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_transformer_lm_default_positions_match_explicit_positions():
    torch.manual_seed(1)
    model = make_model()
    token_ids = torch.randint(0, 31, (2, 5))
    positions = torch.arange(5, device=token_ids.device)

    assert torch.allclose(
        model(token_ids),
        model(token_ids, positions),
        atol=1e-6,
        rtol=1e-6,
    )


def test_transformer_lm_cannot_read_future_tokens():
    torch.manual_seed(2)
    model = make_model()
    token_ids = torch.randint(0, 31, (2, 6))
    changed_ids = token_ids.clone()
    changed_ids[:, 3:] = (changed_ids[:, 3:] + 1) % 31

    assert torch.allclose(
        model(token_ids)[:, :3],
        model(changed_ids)[:, :3],
        atol=1e-5,
        rtol=1e-5,
    )


def test_transformer_lm_backpropagates_to_every_layer():
    torch.manual_seed(3)
    model = make_model()
    token_ids = torch.randint(0, 31, (2, 5))

    model(token_ids).square().mean().backward()

    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, f"{name} 没有梯度"
        assert torch.isfinite(parameter.grad).all(), f"{name} 的梯度不是有限值"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
