"""读取 JSON 配置，在本机（建议 GPU 台式机）运行本项目训练入口。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from TrainingData import load_token_ids
from TrainingScript import run_training


ROOT = Path(__file__).resolve().parent
PATH_KEYS = ("train_path", "val_path", "metrics_path", "checkpoint_path")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="训练 JSON 配置文件")
    parser.add_argument("--device", choices=("cpu", "cuda"), help="临时覆盖配置中的运行设备")
    parser.add_argument("--probe-steps", type=int, help="只跑指定步数测速，输出另存为 *_probe 文件")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.device is not None:
        config["device"] = args.device
    for key in PATH_KEYS:
        config[key] = ROOT / config[key]
    if args.probe_steps is not None:
        if args.probe_steps <= 0:
            parser.error("--probe-steps 必须为正整数")
        config["max_steps"] = args.probe_steps
        for key in ("metrics_path", "checkpoint_path"):
            path = config[key]
            config[key] = path.with_name(path.stem + "_probe" + path.suffix)
    if config["device"] == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA 不可用；请在 GPU 电脑上检查驱动及 CUDA 版 PyTorch")
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    context_length = config["context_length"]
    vocab_size = config["vocab_size"]
    for key in ("train_path", "val_path"):
        path = config[key]
        if not path.is_file():
            raise FileNotFoundError(f"缺少 token 文件: {path}")
        tokens = load_token_ids(path)
        if len(tokens) <= context_length:
            raise ValueError(f"{path} 的 token 数不足 context_length + 1")
        if int(np.max(tokens)) >= vocab_size:
            raise ValueError(f"{path} 中存在超出 vocab_size={vocab_size} 的 token ID")
        print(f"{key}: {path} ({len(tokens):,} tokens)", flush=True)

    config["metrics_path"].parent.mkdir(parents=True, exist_ok=True)
    config["checkpoint_path"].parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(config.get("seed", 42))
    config.pop("seed", None)
    print(f"开始训练: {config['max_steps']} steps", flush=True)
    started = perf_counter()
    records = run_training(**config)
    elapsed = perf_counter() - started
    print(f"完成: {elapsed:.1f}s，含验证平均 {elapsed / config['max_steps']:.3f}s/step; 最终记录: {records[-1]}", flush=True)
    print(f"指标: {config['metrics_path']}", flush=True)
    print(f"Checkpoint: {config['checkpoint_path']}", flush=True)


if __name__ == "__main__":
    main()
