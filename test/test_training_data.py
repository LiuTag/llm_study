"""原始 uint16 token 文件的内存映射读取验收测试。"""

import _paths  # noqa: F401
import importlib

import numpy as np
import pytest

from Data import get_batch


def get_loader():
    module = importlib.import_module("TrainingData")
    loader = getattr(module, "load_token_ids", None)
    assert callable(loader), "请在 TrainingData.py 中实现 load_token_ids"
    return loader


def test_load_token_ids_returns_read_only_uint16_memmap(tmp_path):
    expected = np.array([0, 1, 255, 256, 1023, 65535], dtype=np.uint16)
    path = tmp_path / "tokens.bin"
    expected.tofile(path)

    actual = get_loader()(path)

    assert isinstance(actual, np.memmap)
    assert actual.dtype == np.dtype("uint16")
    assert actual.ndim == 1
    assert not actual.flags.writeable
    np.testing.assert_array_equal(actual, expected)


def test_memmapped_tokens_work_with_get_batch(tmp_path):
    expected = np.arange(100, dtype=np.uint16)
    path = tmp_path / "tokens.bin"
    expected.tofile(path)
    dataset = get_loader()(path)

    torch_x, torch_y = get_batch(dataset, batch_size=4, context_length=8, device="cpu")

    assert tuple(torch_x.shape) == (4, 8)
    assert tuple(torch_y.shape) == (4, 8)
    x = torch_x.numpy()
    y = torch_y.numpy()
    for input_row, target_row in zip(x, y):
        start = int(input_row[0])
        np.testing.assert_array_equal(input_row, expected[start:start + 8])
        np.testing.assert_array_equal(target_row, expected[start + 1:start + 9])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
