"""配置加载入口：读 yaml → 合并环境覆盖 → 构造 Config

模块: config/__init__.py
依赖: config.schema
读取配置: 全部（本模块是配置的唯一入口）
对外接口:
    - load_config(env_path=None, base_path=None) -> Config
"""

from pathlib import Path

import yaml

from config.schema import Config, config_from_dict

__all__ = ["Config", "load_config"]

_DEFAULT_PATH = Path(__file__).with_name("default.yaml")


def _deep_merge(base: dict, override: dict) -> dict:
    """递归合并：环境 yaml 只需写想覆盖的叶子。"""
    merged = dict(base)
    for k, v in override.items():
        merged[k] = _deep_merge(base[k], v) if isinstance(base.get(k), dict) and isinstance(v, dict) else v
    return merged


def load_config(env_path: str | None = None, base_path: str | None = None) -> Config:
    """加载配置。

    参数:
        env_path: 可选环境覆盖 yaml（如 config/smoke.yaml），只写需要覆盖的键
        base_path: 可选默认 yaml 路径，默认取 config/default.yaml
    返回:
        校验通过的 Config 对象（只读，实现文件不得修改）
    说明:
        storage.path / vis.out_dir 的相对路径一律锚定到项目根（config/ 的上级），
        与启动目录无关，保证 CLI 从任何 cwd 启动行为一致。
    """
    base = Path(base_path or _DEFAULT_PATH)
    with open(base, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if env_path is not None:
        with open(env_path, encoding="utf-8") as f:
            raw = _deep_merge(raw, yaml.safe_load(f) or {})
    root = base.resolve().parent.parent
    for section, key in (("storage", "path"), ("vis", "out_dir"),
                         ("training_data", "cache_path"), ("training", "output_dir"),
                         ("optimization", "checkpoint"), ("optimization", "output_dir")):
        p = Path(raw[section][key])
        raw[section][key] = str(p if p.is_absolute() else root / p)
    return config_from_dict(raw)
