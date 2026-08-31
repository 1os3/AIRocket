from pathlib import Path


def check_training_log(path: Path) -> None:
    # 校验对象: render_training_curves 的训练指标日志 —— 主日志必须存在且非空
    assert path.is_file() and path.stat().st_size > 0, (
        f"训练日志不存在或为空：{path}；先运行 train/run.py --mode train")
