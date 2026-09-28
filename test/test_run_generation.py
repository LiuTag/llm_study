"""训练 checkpoint 到文本生成的命令行验收；可直接执行本文件。"""

import _paths  # noqa: F401
import json
import sys

import pytest
import torch

import RunGeneration
from TransformerLM import TransformerLM


def test_cli_loads_checkpoint_and_generates_text(tmp_path, monkeypatch, capsys) -> None:
    config = {
        "checkpoint_path": "runs/final.pt",
        "vocab_size": 50257,
        "context_length": 16,
        "d_model": 8,
        "num_layers": 1,
        "num_heads": 2,
        "d_ff": 16,
        "rope_theta": 10000.0,
        "device": "cpu",
        "seed": 7,
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    checkpoint_path = tmp_path / "runs" / "final.pt"
    checkpoint_path.parent.mkdir()
    model = TransformerLM(50257, 16, 8, 1, 2, 16, 10000.0)
    torch.save({"model": model.state_dict(), "iteration": 7}, checkpoint_path)

    monkeypatch.setattr(RunGeneration, "ROOT", tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        ["RunGeneration.py", "--config", "config.json", "--prompt", "Once upon a time",
         "--max-new-tokens", "1", "--device", "cpu"],
    )
    RunGeneration.main()

    output = capsys.readouterr().out.splitlines()
    assert output[0] == "checkpoint step: 7"
    assert output[1].startswith("Once upon a time")


def test_cli_reports_missing_checkpoint(tmp_path, monkeypatch, capsys) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"checkpoint_path": "missing.pt"}), encoding="utf-8")
    monkeypatch.setattr(RunGeneration, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["RunGeneration.py", "--config", "config.json"])

    with pytest.raises(SystemExit) as exc:
        RunGeneration.main()

    assert exc.value.code == 2
    assert "找不到 checkpoint" in capsys.readouterr().err


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
