"""CLI 入口：准备缓存、训练、评估与 CPU 最小冒烟

模块: train/run.py
依赖: config, data.training_cache, train.engine
读取配置: 无（经 load_config 统一加载）
对外接口: 命令行
用法:
    .venv/Scripts/python train/run.py --mode prepare
    .venv/Scripts/python train/run.py --mode train
    .venv/Scripts/python train/run.py --mode eval --checkpoint output/training/best.pt
    .venv/Scripts/python train/run.py --env config/train_smoke.yaml --mode smoke
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import load_config
from data.training_cache import prepare_training_cache
from train.engine import evaluate_model, smoke_test, train_model


def main() -> None:
    parser = argparse.ArgumentParser(description="流场残差 Transformer 训练系统")
    parser.add_argument("--mode", required=True, choices=("prepare", "train", "eval", "smoke"))
    parser.add_argument("--env", default=None, help="环境覆盖 yaml（如 config/train_smoke.yaml）")
    parser.add_argument("--checkpoint", default=None, help="恢复或评估使用的 checkpoint")
    args = parser.parse_args()
    cfg = load_config(env_path=args.env)
    if args.mode == "prepare":
        prepare_training_cache(cfg)
    elif args.mode == "train":
        train_model(cfg, args.checkpoint)
    elif args.mode == "eval":
        evaluate_model(cfg, args.checkpoint)
    else:
        smoke_test(cfg)


if __name__ == "__main__":
    main()
