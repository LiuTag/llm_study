"""训练批次采样验收测试；可用 python test_data.py 直接运行。"""

import _paths  # noqa: F401
import importlib
import importlib.util

import numpy as np
import pytest
import torch


def get_batch_function():
    assert importlib.util.find_spec("Data") is not None, "请先创建 Data.py"
    function = getattr(importlib.import_module("Data"), "get_batch", None)
    assert callable(function), "请先在 Data.py 中实现 get_batch"
    return function


def test_get_batch_returns_shifted_contiguous_long_tensors():
    get_batch = get_batch_function()
    dataset = np.arange(50, dtype=np.uint16)
    original = dataset.copy()

    x, y = get_batch(dataset, batch_size=32, context_length=5, device="cpu")

    assert x.shape == y.shape == (32, 5)
    assert x.dtype == y.dtype == torch.long
    assert x.device.type == y.device.type == "cpu"
    for row in range(32):
        start = int(x[row, 0])
        assert 0 <= start <= len(dataset) - 5 - 1
        assert torch.equal(x[row], torch.arange(start, start + 5))
        assert torch.equal(y[row], torch.arange(start + 1, start + 6))
    assert np.array_equal(dataset, original)


def test_get_batch_can_use_the_only_valid_window():
    get_batch = get_batch_function()
    dataset = np.array([10, 11, 12, 13, 14], dtype=np.uint16)

    x, y = get_batch(dataset, batch_size=3, context_length=4, device="cpu")

    expected_x = torch.tensor([[10, 11, 12, 13]] * 3)
    expected_y = torch.tensor([[11, 12, 13, 14]] * 3)
    assert torch.equal(x, expected_x)
    assert torch.equal(y, expected_y)


def test_get_batch_samples_more_than_one_start_position():
    get_batch = get_batch_function()
    dataset = np.arange(100, dtype=np.int64)

    x, y = get_batch(dataset, batch_size=128, context_length=4, device="cpu")

    assert x[:, 0].unique().numel() > 1
    assert torch.equal(y[:, :-1], x[:, 1:])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
