"""CLI 入口：准备缓存、训练、微调、评估、翼型优化与 CPU 最小冒烟

模块: train/run.py
依赖: config, data.training_cache, train.airfoil_optimization, train.engine, train.fine_tuning
读取配置: 无（经 load_config 统一加载）
对外接口: 命令行
用法:
    .venv/Scripts/python train/run.py --mode prepare
    .venv/Scripts/python train/run.py --mode train
    .venv/Scripts/python train/run.py --env config/finetune.yaml --mode finetune --checkpoint output/training/best.pt
    .venv/Scripts/python train/run.py --mode eval --checkpoint output/training/best.pt
    .venv/Scripts/python train/run.py --mode optimize --checkpoint output/training/best.pt
    .venv/Scripts/python train/run.py --env config/train_smoke.yaml --mode smoke
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import load_config
from data.training_cache import prepare_training_cache
from train.airfoil_optimization import optimize_airfoil
from train.engine import evaluate_model, smoke_test, train_model
from train.fine_tuning import fine_tune_model


def main() -> None:
    parser = argparse.ArgumentParser(description="流场残差 Transformer 训练系统")
    parser.add_argument(
        "--mode", required=True,
        choices=("prepare", "train", "finetune", "eval", "optimize", "smoke"))
    parser.add_argument("--env", default=None, help="环境覆盖 yaml（如 config/train_smoke.yaml）")
    parser.add_argument(
        "--checkpoint", default=None,
        help="续训/评估/优化 checkpoint，微调时为预训练权重")
    args = parser.parse_args()
    cfg = load_config(env_path=args.env)
    if args.mode == "prepare":
        prepare_training_cache(cfg)
    elif args.mode == "train":
        train_model(cfg, args.checkpoint)
    elif args.mode == "finetune":
        if args.checkpoint is None:
            parser.error("--mode finetune 必须通过 --checkpoint 指定预训练权重")
        fine_tune_model(cfg, args.checkpoint)
    elif args.mode == "eval":
        evaluate_model(cfg, args.checkpoint)
    elif args.mode == "optimize":
        optimize_airfoil(cfg, args.checkpoint)
    else:
        smoke_test(cfg)


if __name__ == "__main__":
    main()
