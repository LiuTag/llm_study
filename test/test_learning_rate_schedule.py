"""余弦学习率计划验收测试；可直接运行此 Python 文件。"""

import _paths  # noqa: F401
import importlib
import importlib.util

import pytest


def get_schedule():
    assert importlib.util.find_spec("LearningRateSchedule") is not None, (
        "请先创建 LearningRateSchedule.py"
    )
    function = getattr(
        importlib.import_module("LearningRateSchedule"),
        "get_lr_cosine_schedule",
        None,
    )
    assert callable(function), (
        "请在 LearningRateSchedule.py 中实现 get_lr_cosine_schedule"
    )
    return function


@pytest.mark.parametrize(
    ("it", "expected"),
    [(0, 0.0), (1, 0.225), (3, 0.675), (4, 0.9)],
)
def test_linear_warmup_reaches_max_at_boundary(it, expected):
    schedule = get_schedule()
    actual = schedule(
        it=it,
        max_learning_rate=0.9,
        min_learning_rate=0.1,
        warmup_iters=4,
        cosine_cycle_iters=12,
    )
    assert actual == pytest.approx(expected)


@pytest.mark.parametrize(
    ("it", "expected"),
    [(4, 0.9), (8, 0.5), (12, 0.1), (13, 0.1), (100, 0.1)],
)
def test_cosine_midpoint_endpoint_and_floor(it, expected):
    schedule = get_schedule()
    actual = schedule(
        it=it,
        max_learning_rate=0.9,
        min_learning_rate=0.1,
        warmup_iters=4,
        cosine_cycle_iters=12,
    )
    assert actual == pytest.approx(expected)


def test_matches_official_non_midpoint_checkpoints():
    schedule = get_schedule()
    expected = {
        8: 0.9887175604818206,
        14: 0.55,
        20: 0.11128243951817937,
        21: 0.1,
    }
    for it, learning_rate in expected.items():
        actual = schedule(
            it=it,
            max_learning_rate=1.0,
            min_learning_rate=0.1,
            warmup_iters=7,
            cosine_cycle_iters=21,
        )
        assert actual == pytest.approx(learning_rate)


def test_zero_warmup_starts_at_maximum_without_dividing_by_zero():
    schedule = get_schedule()
    expected = {0: 0.8, 5: 0.5, 10: 0.2, 11: 0.2}
    for it, learning_rate in expected.items():
        actual = schedule(
            it=it,
            max_learning_rate=0.8,
            min_learning_rate=0.2,
            warmup_iters=0,
            cosine_cycle_iters=10,
        )
        assert actual == pytest.approx(learning_rate)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
