"""推理命令行入口：从训练 checkpoint 加载模型并生成文本。"""

import argparse
import json
import random
import sys
from pathlib import Path

import torch

from Generate import generate_text
from PrepareTraining import load_gpt2_tokenizer
from TransformerLM import TransformerLM


# 项目根目录 = 本脚本所在目录，用于解析所有相对路径
ROOT = Path(__file__).resolve().parent


def resolve_path(p: str | Path) -> Path:
    """相对路径 -> 相对项目根目录；绝对路径原样返回。"""
    path = Path(p)
    if path.is_absolute():
        return path
    return (ROOT / path).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="推理程序：从训练 checkpoint 加载模型并生成文本",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/tinystories.json",
        help="模型结构 / 设备 / 默认 checkpoint 路径的配置文件；"
             "相对路径基于项目根目录解析（默认: config/tinystories.json）",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="要加载的 .pt 文件路径（只读，不用于保存）；"
             "默认取配置文件中的 checkpoint_path",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Once upon a time",
        help="生成的起始文本；包含空格时请用引号包裹（默认: 'Once upon a time'）",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=120,
        help="最多新增的 token 数，须 >= 0（默认: 120）",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="采样温度，须 > 0（默认: 1.0）",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="核采样阈值，须在 (0, 1] 区间内（默认: 0.9）",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda"],
        help="运行设备；默认取配置文件中的 device",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="随机数种子；默认取配置文件中的 seed，缺失时用 42",
    )

    args = parser.parse_args()

    # ---- 参数合法性校验 ----
    if args.max_new_tokens < 0:
        parser.error("--max-new-tokens 必须 >= 0")
    if args.temperature <= 0:
        parser.error("--temperature 必须 > 0")
    if not (0.0 < args.top_p <= 1.0):
        parser.error("--top-p 必须在 (0, 1] 区间内")

    return args


def load_config(config_path: Path) -> dict:
    if not config_path.is_file():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    return json.loads(config_path.read_text(encoding="utf-8"))


def resolve_device(cli_device: str | None, cfg_device: str | None) -> str:
    """命令行优先；其次配置文件；都没有则自动探测。指定 cuda 不可用时明确报错。"""
    device = cli_device or cfg_device
    if device is None:
        return "cuda" if torch.cuda.is_available() else "cpu"

    if device not in ("cpu", "cuda"):
        raise ValueError(f"未知设备: {device!r}，只能是 'cpu' 或 'cuda'")

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "指定使用 CUDA，但当前环境没有可用的 CUDA 设备。"
            "可用 `--device cpu` 强制使用 CPU。"
        )
    return device


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()

    # 1) 配置文件（相对项目根目录解析）
    config_path = resolve_path(args.config)
    config = load_config(config_path)

    # 2) checkpoint 路径：命令行优先，回落到配置
    checkpoint_arg = args.checkpoint or config.get("checkpoint_path")
    if not checkpoint_arg:
        raise ValueError(
            f"未提供 --checkpoint，且配置文件 {config_path} 中也没有 'checkpoint_path' 字段。"
        )
    checkpoint_path = resolve_path(checkpoint_arg)
    if not checkpoint_path.is_file():
        print(f"找不到 checkpoint: {checkpoint_path}", file=sys.stderr)
        raise SystemExit(2)

    # 3) 设备 / 随机种子
    device = resolve_device(args.device, config.get("device"))
    seed = args.seed if args.seed is not None else config.get("seed", 42)
    set_seed(seed)

    # 4) 构建模型（结构必须与训练配置一致）
    model = TransformerLM(
        vocab_size=config["vocab_size"],
        context_length=config["context_length"],
        d_model=config["d_model"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
        rope_theta=config["rope_theta"],
    )

    # 5) 加载 checkpoint：只取 model 权重；iteration 单独读取打印
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if "model" not in checkpoint:
        raise KeyError(
            f"checkpoint {checkpoint_path} 中找不到 'model' 字段，"
            f"实际包含: {list(checkpoint.keys())}"
        )
    iteration = checkpoint.get("iteration", checkpoint.get("iter", "unknown"))

    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()

    # 6) 分词器
    tokenizer = load_gpt2_tokenizer()
    eos_token_id = tokenizer.specialtoken2id[b"<|endoftext|>"]

    # 7) 生成
    text = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        context_length=config["context_length"],
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        eos_token_id=eos_token_id,
        device=device,
    )

    # 8) 输出
    print(f"checkpoint step: {iteration}")
    print(text)


if __name__ == "__main__":
    main()
