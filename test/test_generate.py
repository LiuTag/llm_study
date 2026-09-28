"""文本生成验收测试；可直接执行本文件。"""

import _paths  # noqa: F401
import importlib

import pytest
import torch
from torch import nn

from BPE import BPETokenizer


def get_generate_text():
    module = importlib.import_module("Generate")
    generate_text = getattr(module, "generate_text", None)
    assert callable(generate_text), "请在 Generate.py 中实现 generate_text"
    return generate_text


def make_tokenizer():
    vocab = {i: bytes([i]) for i in range(256)}
    vocab[256] = b"<|endoftext|>"
    return BPETokenizer(vocab, [], ["<|endoftext|>"])


class ScriptedLM(nn.Module):
    def __init__(self, next_ids):
        super().__init__()
        self.next_ids = next_ids
        self.inputs = []
        self.forward_modes = []
        self.dummy = nn.Parameter(torch.zeros(()))

    def forward(self, input_ids):
        self.inputs.append(input_ids.detach().clone())
        self.forward_modes.append((self.training, torch.is_grad_enabled()))
        logits = torch.full(
            (input_ids.shape[0], input_ids.shape[1], 257),
            -1000.0,
            device=input_ids.device,
        )
        logits[:, -1, self.next_ids[len(self.inputs) - 1]] = 0.0
        return logits


class TwoChoiceLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(()))

    def forward(self, input_ids):
        logits = torch.full(
            (input_ids.shape[0], input_ids.shape[1], 257),
            -1000.0,
            device=input_ids.device,
        )
        logits[:, -1, ord("b")] = 2.0
        logits[:, -1, ord("c")] = 1.0
        return logits


def test_generate_stops_at_eos_without_emitting_eos_text():
    model = ScriptedLM([ord("b"), 256, ord("c")])
    model.train()
    result = get_generate_text()(
        model, make_tokenizer(), "a",
        context_length=8, max_new_tokens=5,
        temperature=1.0, top_p=1.0,
        eos_token_id=256, device="cpu",
    )

    assert result == "ab"
    assert len(model.inputs) == 2
    assert model.forward_modes == [(False, False), (False, False)]
    assert model.training


def test_generate_crops_model_input_but_preserves_full_prompt():
    model = ScriptedLM([ord("x"), ord("y")])
    result = get_generate_text()(
        model, make_tokenizer(), "abcd",
        context_length=3, max_new_tokens=2,
        temperature=1.0, top_p=1.0,
        eos_token_id=None, device="cpu",
    )

    assert result == "abcdxy"
    assert [tuple(x.shape) for x in model.inputs] == [(1, 3), (1, 3)]
    assert model.inputs[0].tolist() == [[ord("b"), ord("c"), ord("d")]]
    assert model.inputs[1].tolist() == [[ord("c"), ord("d"), ord("x")]]


def test_zero_new_tokens_returns_prompt_without_model_call():
    model = ScriptedLM([])
    result = get_generate_text()(
        model, make_tokenizer(), "abc",
        context_length=3, max_new_tokens=0,
        temperature=1.0, top_p=1.0,
        eos_token_id=256, device="cpu",
    )

    assert result == "abc"
    assert model.inputs == []


def test_top_p_excludes_tokens_beyond_nucleus():
    model = TwoChoiceLM()
    tokenizer = make_tokenizer()
    generate_text = get_generate_text()

    for seed in range(30):
        torch.manual_seed(seed)
        result = generate_text(
            model, tokenizer, "a",
            context_length=4, max_new_tokens=1,
            temperature=1.0, top_p=0.5,
            eos_token_id=None, device="cpu",
        )
        assert result == "ab"


def test_temperature_changes_sampling_concentration():
    model = TwoChoiceLM()
    tokenizer = make_tokenizer()
    generate_text = get_generate_text()
    results = {}

    for temperature in (0.05, 10.0):
        sampled = []
        for seed in range(40):
            torch.manual_seed(seed)
            sampled.append(generate_text(
                model, tokenizer, "a",
                context_length=4, max_new_tokens=1,
                temperature=temperature, top_p=1.0,
                eos_token_id=None, device="cpu",
            ))
        results[temperature] = sampled

    assert all(text == "ab" for text in results[0.05])
    assert sum(text == "ac" for text in results[10.0]) >= 5


def test_top_p_one_preserves_temperature_softmax_probabilities(monkeypatch):
    captured = []

    def capture_multinomial(probabilities, num_samples, replacement=False):
        captured.append(probabilities.detach().clone())
        return torch.zeros(num_samples, dtype=torch.long, device=probabilities.device)

    monkeypatch.setattr(torch, "multinomial", capture_multinomial)
    result = get_generate_text()(
        TwoChoiceLM(), make_tokenizer(), "a",
        context_length=4, max_new_tokens=1,
        temperature=1.0, top_p=1.0,
        eos_token_id=None, device="cpu",
    )

    expected = torch.softmax(torch.tensor([2.0, 1.0]), dim=0)
    assert result == "ab"
    assert len(captured) == 1
    torch.testing.assert_close(captured[0][:2], expected, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize(
    "overrides",
    [
        {"context_length": 0},
        {"max_new_tokens": -1},
        {"temperature": 0.0},
        {"top_p": 1.1},
    ],
)
def test_rejects_invalid_generation_controls(overrides):
    options = {
        "context_length": 4,
        "max_new_tokens": 1,
        "temperature": 1.0,
        "top_p": 1.0,
    }
    options.update(overrides)
    with pytest.raises(ValueError):
        get_generate_text()(
            TwoChoiceLM(), make_tokenizer(), "a",
            eos_token_id=None, device="cpu", **options,
        )


def test_restores_training_mode_when_model_raises():
    class FailingLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy = nn.Parameter(torch.zeros(()))

        def forward(self, input_ids):
            assert not self.training
            assert not torch.is_grad_enabled()
            raise RuntimeError("模拟推理失败")

    model = FailingLM()
    model.train()
    with pytest.raises(RuntimeError, match="模拟推理失败"):
        get_generate_text()(
            model, make_tokenizer(), "a",
            context_length=4, max_new_tokens=1,
            temperature=1.0, top_p=1.0,
            eos_token_id=None, device="cpu",
        )
    assert model.training


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
