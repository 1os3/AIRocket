"""LMDB 存储后端：稳态流场 + 参数元数据的写入与快速随机读取

模块: data/storage/storage.py
依赖: lmdb, torch, data.storage.checks.storage_checks
读取配置: storage.path, storage.map_size_gb, seed, version
对外接口:
    - FlowFieldWriter: existing_indices(), existing_seeds(), write(plan, fields, extra), write_meta(), close()
    - FlowFieldDataset: torch Dataset，按位随机读取；get_by_index(sample_index) 精确读取
说明:
    - 键设计：b"sample/%08d" 存样本（前缀扫描即得断点），b"meta/info" 存全局元信息。
    - 值用 pickle 序列化 numpy 数组（float32），读取零拷贝解码为 torch 张量。
    - 只存收敛稳态场（rho/ux/uy/p/mask），不存瞬态；非收敛样本由上层直接丢弃。
"""

import pickle
from datetime import datetime, timezone
from pathlib import Path

import lmdb
import torch

from data.storage.checks.storage_checks import check_fields

__all__ = ["FlowFieldWriter", "FlowFieldDataset"]

_SAMPLE_PREFIX = b"sample/"
_META_INFO = b"meta/info"


class FlowFieldWriter:
    """断点续采友好的 LMDB 写入器：打开即可查询已有样本与已用种子。"""

    def __init__(self, cfg):
        Path(cfg.storage.path).mkdir(parents=True, exist_ok=True)  # lmdb 不自动建目录
        self._env = lmdb.open(cfg.storage.path, map_size=cfg.storage.map_size_gb * 2 ** 30,
                              subdir=True, create=True)
        self._cfg = cfg

    def existing_indices(self) -> set:
        """扫描 sample/ 前缀，返回库中已存在的样本索引集合（断点依据）。"""
        with self._env.begin() as txn, txn.cursor() as cur:
            return {int(k[len(_SAMPLE_PREFIX):]) for k, _ in cur.iternext()
                    if k.startswith(_SAMPLE_PREFIX)}

    def existing_seeds(self) -> set:
        """返回库中全部样本的派生种子，供续采时冲突检测（防配置变更后种子复用）。"""
        with self._env.begin() as txn:
            get = lambda i: txn.get(f"{_SAMPLE_PREFIX.decode()}{i:08d}".encode())
            return {pickle.loads(raw)["seed"] for i in self.existing_indices()
                    if (raw := get(i)) is not None}

    def write_meta(self) -> None:
        """写入全局元信息：版本、主种子、完整配置快照、torch/lmdb 版本、创建时间。"""
        cfg = self._cfg
        info = {
            "version": cfg.version, "seed": cfg.seed,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "torch_version": torch.__version__, "lmdb_version": lmdb.version(),
            "config": {k: getattr(cfg, k).__dict__ if hasattr(getattr(cfg, k), "__dict__")
                       else getattr(cfg, k)
                       for k in ("device", "grid", "solver", "airfoil", "sampler", "storage")},
        }
        with self._env.begin(write=True) as txn:
            txn.put(_META_INFO, pickle.dumps(info))

    def write(self, plan, fields: dict, extra: dict) -> None:
        """写入一个收敛样本。

        参数:
            plan: SamplePlan（提供 index/seed/采样参数）
            fields: dict(rho, ux, uy, p, mask) 的 CPU 张量
            extra: 格子参数与运行信息（u_lb, tau, nu, steps, reynolds_lattice 等）
        """
        check_fields(fields, self._cfg.grid)
        record = {"index": plan.index, "seed": plan.seed,
                  "params": {**{k: getattr(plan, k) for k in
                                ("reynolds", "mach", "aoa_deg", "naca_m", "naca_p", "naca_t")},
                             **extra},
                  "fields": {k: v.numpy() for k, v in fields.items()}}
        with self._env.begin(write=True) as txn:
            # 校验对象: 写入键 —— 同 index 重复写入说明续采逻辑失效，直接报错而非覆盖
            key = f"{_SAMPLE_PREFIX.decode()}{plan.index:08d}".encode()
            assert txn.get(key) is None, f"样本 {plan.index} 已存在，禁止覆盖写入"
            txn.put(key, pickle.dumps(record, protocol=4))

    def close(self) -> None:
        self._env.sync()
        self._env.close()


class FlowFieldDataset(torch.utils.data.Dataset):
    """只读随机访问：__getitem__ 按位置取（容忍断点造成的索引空洞）。"""

    def __init__(self, path: str):
        # 库不存在时给出可操作提示，而非裸 lmdb.Error（常见于忘了 --env 或尚未采集）
        if not Path(path).exists():
            raise FileNotFoundError(
                f"LMDB 库不存在: {path}\n"
                "  → 尚未采集：先运行 data/run.py 生成数据集；\n"
                "  → 或库在别的路径：用 --env 指定对应配置（如 config/smoke.yaml）。")
        self._env = lmdb.open(path, readonly=True, lock=False, subdir=True, readahead=False)
        with self._env.begin() as txn, txn.cursor() as cur:
            self._keys = sorted(k for k, _ in cur.iternext() if k.startswith(_SAMPLE_PREFIX))
            raw = txn.get(_META_INFO)
        self.meta = pickle.loads(raw) if raw else None

    def __len__(self) -> int:
        return len(self._keys)

    def _decode(self, key: bytes) -> dict:
        with self._env.begin() as txn:
            record = pickle.loads(txn.get(key))
        return {**record, "fields": {k: torch.from_numpy(v) for k, v in record["fields"].items()}}

    def __getitem__(self, i: int) -> dict:
        return self._decode(self._keys[i])

    def get_by_index(self, sample_index: int) -> dict:
        """按样本索引精确读取；不存在返回 None。"""
        key = f"{_SAMPLE_PREFIX.decode()}{sample_index:08d}".encode()
        with self._env.begin() as txn:
            raw = txn.get(key)
        if raw is None:
            return None
        record = pickle.loads(raw)
        return {**record, "fields": {k: torch.from_numpy(v) for k, v in record["fields"].items()}}
