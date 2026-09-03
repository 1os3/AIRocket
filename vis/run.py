"""CLI 入口：流场样本、训练曲线、模型推理与优化翼型 LBM 复核（留在 vis 包根）

模块: vis/run.py
依赖: config, vis.flowfield, vis.inference, vis.optimization_lbm, vis.training_curves
读取配置: 无（经 load_config 统一加载）
对外接口: 命令行
用法:
    python -m vis.run                                       # 按 config/default.yaml 渲染正式库
    python -m vis.run --env config/smoke.yaml               # 渲染冒烟库
    python -m vis.run --env config/smoke.yaml --indices 0 1  # 只渲染指定编号的样本
    python -m vis.run --training-curves                      # 渲染训练损失与梯度曲线
    python -m vis.run --inference                            # 随机初始化模型推理图
    python -m vis.run --inference output/training/best.pt    # 检查点模型推理图
    python -m vis.run --optimization-lbm                     # LBM 复核默认优化结果
    python -m vis.run --optimization-lbm output/optimization/result.json
说明: 渲染数量/面板/色图/输出目录均在 config 的 vis 段调整；--indices 支持任意编号
      （含断点空洞），不存在的编号跳过并告警。
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # 允许从任意目录启动

from config import load_config
from vis.flowfield import render_samples
from vis.inference import render_model_inference
from vis.optimization_lbm import render_optimization_lbm_evaluation
from vis.training_curves import render_training_curves


def main() -> None:
    parser = argparse.ArgumentParser(description="流场样本、训练曲线与模型推理可视化")
    parser.add_argument("--env", default=None, help="环境覆盖 yaml（如 config/smoke.yaml）")
    target = parser.add_mutually_exclusive_group()
    target.add_argument("--training-curves", action="store_true",
                        help="读取训练日志并渲染 loss 与梯度诊断曲线")
    target.add_argument("--inference", nargs="?", const="", default=None,
                        metavar="CHECKPOINT",
                        help="渲染模型推理；不附路径用随机初始化，附路径则加载训练检查点")
    target.add_argument("--optimization-lbm", nargs="?", const="", default=None,
                        metavar="RESULT_JSON",
                        help="用 LBM 复核初始/最优翼型；可附 optimization result.json 路径")
    parser.add_argument("--indices", type=int, nargs="+", default=None,
                        help="指定样本编号（可多个）；适用于流场样本和模型推理")
    args = parser.parse_args()
    if (args.training_curves or args.optimization_lbm is not None) and args.indices is not None:
        parser.error("--indices 不适用于 --training-curves/--optimization-lbm")
    cfg = load_config(env_path=args.env)
    if args.training_curves:
        render_training_curves(cfg)
    elif args.inference is not None:
        render_model_inference(cfg, checkpoint=args.inference or None, indices=args.indices)
    elif args.optimization_lbm is not None:
        render_optimization_lbm_evaluation(
            cfg, result_path=args.optimization_lbm or None)
    else:
        render_samples(cfg, indices=args.indices)


if __name__ == "__main__":
    main()
