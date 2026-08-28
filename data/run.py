"""CLI 入口：2D 流场仿真数据集端到端生成（留在 data 包根）

模块: data/run.py
依赖: config, data.collector
读取配置: 无（经 load_config 统一加载）
对外接口: 命令行
用法:
    .venv/Scripts/python data/run.py                          # 按 config/default.yaml 正式采集
    .venv/Scripts/python data/run.py --env config/smoke.yaml  # 小规模冒烟（CPU 开发调试用）
    .venv/Scripts/python data/run.py --env config/smoke.yaml --read  # 校验读取接口与元数据
说明: 断点续采无需参数——中断后原命令重跑即可自动跳过已入库样本。
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # 允许从任意目录启动

from config import load_config
from data.collector import collect


def main() -> None:
    parser = argparse.ArgumentParser(description="LBM 流场数据集生成")
    parser.add_argument("--env", default=None, help="环境覆盖 yaml（如 config/smoke.yaml）")
    parser.add_argument("--read", action="store_true", help="采集后抽查读取接口与样本元数据")
    args = parser.parse_args()
    cfg = load_config(env_path=args.env)
    collect(cfg)
    if args.read:
        from data.storage import FlowFieldDataset
        ds = FlowFieldDataset(cfg.storage.path)
        assert len(ds) > 0, "库内无样本（可能全部未收敛），无内容可读"
        sample = ds[0]
        print(f"[read] 库内样本数={len(ds)} meta.seed={ds.meta['seed']} "
              f"样本0: index={sample['index']} seed={sample['seed']} "
              f"Re={sample['params']['reynolds']:.1f} rho.shape={tuple(sample['fields']['rho'].shape)}")


if __name__ == "__main__":
    main()
