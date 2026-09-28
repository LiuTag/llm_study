"""checkpoint 保存与恢复验收测试；可直接运行此 Python 文件。"""

import _paths  # noqa: F401
import importlib
import importlib.util
import io

import pytest
import torch

from AdamW import AdamW


def get_checkpoint_functions():
    assert importlib.util.find_spec("Checkpoint") is not None, (
        "请先创建 Checkpoint.py"
    )
    module = importlib.import_module("Checkpoint")
    save = getattr(module, "save_checkpoint", None)
    load = getattr(module, "load_checkpoint", None)
    assert callable(save), "请在 Checkpoint.py 中实现 save_checkpoint"
    assert callable(load), "请在 Checkpoint.py 中实现 load_checkpoint"
    return save, load


def make_model_and_optimizer():
    model = torch.nn.Linear(3, 2)
    optimizer = AdamW(model.parameters(), lr=0.03, weight_decay=0.02)
    return model, optimizer


def train_one_step(model, optimizer):
    x = torch.tensor([[1.0, -2.0, 0.5], [0.2, 0.3, -0.4]])
    target = torch.tensor([[0.1, 0.9], [-0.3, 0.7]])
    optimizer.zero_grad()
    loss = torch.nn.functional.mse_loss(model(x), target)
    loss.backward()
    optimizer.step()


def assert_optimizer_states_equal(actual, expected):
    actual_state = actual.state_dict()
    expected_state = expected.state_dict()
    assert actual_state["param_groups"] == expected_state["param_groups"]
    assert actual_state["state"].keys() == expected_state["state"].keys()
    for parameter_id, actual_values in actual_state["state"].items():
        expected_values = expected_state["state"][parameter_id]
        assert actual_values.keys() == expected_values.keys()
        for name, value in actual_values.items():
            expected_value = expected_values[name]
            if torch.is_tensor(value):
                torch.testing.assert_close(value, expected_value)
            else:
                assert value == expected_value


def test_path_roundtrip_restores_model_optimizer_and_iteration(tmp_path):
    save_checkpoint, load_checkpoint = get_checkpoint_functions()
    torch.manual_seed(13)
    model, optimizer = make_model_and_optimizer()
    train_one_step(model, optimizer)
    train_one_step(model, optimizer)
    checkpoint_path = tmp_path / "training.pt"

    save_checkpoint(model, optimizer, iteration=7, out=checkpoint_path)

    restored_model, restored_optimizer = make_model_and_optimizer()
    loaded_iteration = load_checkpoint(
        src=checkpoint_path,
        model=restored_model,
        optimizer=restored_optimizer,
    )
    assert loaded_iteration == 7
    for original, restored in zip(model.parameters(), restored_model.parameters()):
        torch.testing.assert_close(restored, original)
    assert_optimizer_states_equal(restored_optimizer, optimizer)

    train_one_step(model, optimizer)
    train_one_step(restored_model, restored_optimizer)
    for original, restored in zip(model.parameters(), restored_model.parameters()):
        torch.testing.assert_close(restored, original)


def test_binary_file_object_roundtrip_without_prior_optimizer_steps():
    save_checkpoint, load_checkpoint = get_checkpoint_functions()
    torch.manual_seed(29)
    model, optimizer = make_model_and_optimizer()
    buffer = io.BytesIO()

    save_checkpoint(model, optimizer, iteration=0, out=buffer)
    buffer.seek(0)

    restored_model, restored_optimizer = make_model_and_optimizer()
    loaded_iteration = load_checkpoint(
        src=buffer,
        model=restored_model,
        optimizer=restored_optimizer,
    )
    assert loaded_iteration == 0
    for original, restored in zip(model.parameters(), restored_model.parameters()):
        torch.testing.assert_close(restored, original)
    assert_optimizer_states_equal(restored_optimizer, optimizer)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
