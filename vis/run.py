"""CLI 入口：流场可视化（从 LMDB 渲染 PNG，留在 vis 包根）

模块: vis/run.py
依赖: config, vis.flowfield
读取配置: 无（经 load_config 统一加载）
对外接口: 命令行
用法:
    python -m vis.run                                       # 按 config/default.yaml 渲染正式库
    python -m vis.run --env config/smoke.yaml               # 渲染冒烟库
    python -m vis.run --env config/smoke.yaml --indices 0 1  # 只渲染指定编号的样本
说明: 渲染数量/面板/色图/输出目录均在 config 的 vis 段调整；--indices 支持任意编号
      （含断点空洞），不存在的编号跳过并告警。
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # 允许从任意目录启动

from config import load_config
from vis.flowfield import render_samples


def main() -> None:
    parser = argparse.ArgumentParser(description="流场可视化")
    parser.add_argument("--env", default=None, help="环境覆盖 yaml（如 config/smoke.yaml）")
    parser.add_argument("--indices", type=int, nargs="+", default=None,
                        help="指定样本编号（可多个）；缺省渲染 config vis.num_samples 决定的数量")
    args = parser.parse_args()
    render_samples(load_config(env_path=args.env), indices=args.indices)


if __name__ == "__main__":
    main()
