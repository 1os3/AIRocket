"""LMDB 存储后端：一个样本一个独立子库，稳态流场 + 参数元数据的写入与随机读取

模块: data/storage/storage.py
依赖: lmdb, torch, data.storage.checks.storage_checks
读取配置: storage.path, storage.map_size_mb, seed, version
对外接口:
    - FlowFieldWriter: existing_indices(), existing_seeds(), write(plan, fields, extra), write_meta(), close()
    - FlowFieldDataset: torch Dataset，按位随机读取；indices/get_by_index()/close() 精确访问
说明:
    - 目录结构：storage.path 为数据集根目录；
      meta.lmdb/ 存全局信息（版本/配置快照/种子注册表 seed/%08d），
      sample_%08d.lmdb/ 为单样本子库，含 record（流场+参数元数据）与 meta（自包含，
      便于按样本搬运/分片，单库即完整样本）。
    - 值用 pickle 序列化 numpy 数组（float32），读取解码为 torch 张量。
    - 只存收敛稳态场（rho/ux/uy/p/mask），不存瞬态；非收敛样本由上层直接丢弃。
"""

import pickle
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

import lmdb
import torch

from data.storage.checks.storage_checks import check_fields

__all__ = ["FlowFieldWriter", "FlowFieldDataset"]

_META_DIR = "meta.lmdb"
_INFO_KEY = b"info"
_RECORD_KEY = b"record"
_META_KEY = b"meta"
_SAMPLE_GLOB = "sample_*.lmdb"


def _sample_dir(root: Path, index: int) -> Path:
    return root / f"sample_{index:08d}.lmdb"


def _global_meta(cfg) -> dict:
    """全局元信息：版本、主种子、完整配置快照、torch/lmdb 版本、创建时间。"""
    return {
        "version": cfg.version, "seed": cfg.seed,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "torch_version": torch.__version__, "lmdb_version": lmdb.version(),
        "config": {k: getattr(cfg, k).__dict__ if hasattr(getattr(cfg, k), "__dict__")
                   else getattr(cfg, k)
                   for k in ("device", "grid", "solver", "airfoil", "sampler", "storage")},
    }


class FlowFieldWriter:
    """断点续采友好的写入器：打开即可查询已有样本编号与已用种子。"""

    def __init__(self, cfg):
        self._cfg = cfg
        self._root = Path(cfg.storage.path)
        self._root.mkdir(parents=True, exist_ok=True)
        meta_dir = self._root / _META_DIR
        meta_dir.mkdir(exist_ok=True)  # lmdb 不自动建目录
        self._meta_env = lmdb.open(str(meta_dir), map_size=16 * 2 ** 20, subdir=True)

    def existing_indices(self) -> set:
        """扫描根目录的 sample_*.lmdb 子库，返回已存在样本编号集合（断点依据）。"""
        return {int(p.stem.removeprefix("sample_")) for p in self._root.glob(_SAMPLE_GLOB)}

    def existing_seeds(self) -> set:
        """从种子注册表返回全部已用派生种子（续采冲突检测，免逐库打开）。"""
        with self._meta_env.begin() as txn, txn.cursor() as cur:
            return {pickle.loads(v) for k, v in cur.iternext() if k.startswith(b"seed/")} \
                if cur.set_range(b"seed/") else set()

    def write_meta(self) -> None:
        """写入/更新全局元信息（每次采集启动时刷新配置快照）。"""
        with self._meta_env.begin(write=True) as txn:
            txn.put(_INFO_KEY, pickle.dumps(_global_meta(self._cfg)))

    def write(self, plan, fields: dict, extra: dict) -> None:
        """写入一个收敛样本为独立子库（record + meta，自包含）。

        参数:
            plan: SamplePlan（提供 index/seed/采样参数）
            fields: dict(rho, ux, uy, p, mask) 的 CPU 张量
            extra: 格子参数与运行信息（u_lb, tau, nu, steps, reynolds_lattice 等）
        """
        check_fields(fields, self._cfg.grid)
        sample_dir = _sample_dir(self._root, plan.index)
        # 校验对象: 写入目标 —— 同编号子库已存在说明续采逻辑失效，报错而非覆盖
        assert not sample_dir.exists(), f"样本 {plan.index} 已存在，禁止覆盖写入"
        record = {"index": plan.index, "seed": plan.seed,
                  "params": {**{k: getattr(plan, k) for k in
                                ("reynolds", "mach", "aoa_deg", "naca_m", "naca_p", "naca_t")},
                             **extra},
                  "fields": {k: v.numpy() for k, v in fields.items()}}
        sample_dir.mkdir()
        env = lmdb.open(str(sample_dir), map_size=self._cfg.storage.map_size_mb * 2 ** 20,
                        subdir=True)
        try:
            with env.begin(write=True) as txn:
                txn.put(_RECORD_KEY, pickle.dumps(record, protocol=4))
                txn.put(_META_KEY, pickle.dumps(_global_meta(self._cfg)))
        finally:
            env.sync()
            env.close()
        with self._meta_env.begin(write=True) as txn:  # 登记种子，供续采冲突检测
            txn.put(f"seed/{plan.index:08d}".encode(), pickle.dumps(plan.seed))

    def close(self) -> None:
        self._meta_env.sync()
        self._meta_env.close()


class FlowFieldDataset(torch.utils.data.Dataset):
    """只读随机访问：扫描根目录子库，__getitem__ 按位置取（容忍编号空洞）。"""

    def __init__(self, path: str, reader_cache_size: int = 0):
        root = Path(path)
        if not root.exists():
            raise FileNotFoundError(
                f"数据集根目录不存在: {root}\n"
                "  → 尚未采集：先运行 data/run.py 生成数据集；\n"
                "  → 或库在别的路径：用 --env 指定对应配置（如 config/smoke.yaml）。")
        self._root = root
        self._dirs = sorted(root.glob(_SAMPLE_GLOB))
        assert reader_cache_size >= 0, "reader_cache_size 必须 >= 0"
        self._reader_cache_size = reader_cache_size
        self._envs = OrderedDict()
        meta_env_path = root / _META_DIR
        self.meta = None
        if meta_env_path.exists():
            env = lmdb.open(str(meta_env_path), readonly=True, lock=False, subdir=True)
            try:
                with env.begin() as txn:
                    raw = txn.get(_INFO_KEY)
                self.meta = pickle.loads(raw) if raw else None
            finally:
                env.close()

    def __len__(self) -> int:
        return len(self._dirs)

    @property
    def indices(self) -> list:
        """返回按目录排序的稳定样本编号，不打开样本数据页。"""
        return [int(path.stem.removeprefix("sample_")) for path in self._dirs]

    def _decode(self, sample_dir: Path) -> dict:
        key = str(sample_dir)
        temporary = self._reader_cache_size == 0
        env = self._envs.get(key) if not temporary else None
        if env is None:
            env = lmdb.open(key, readonly=True, lock=False, subdir=True, readahead=False)
        if not temporary:
            self._envs[key] = env
            self._envs.move_to_end(key)
            while len(self._envs) > self._reader_cache_size:
                _, stale = self._envs.popitem(last=False)
                stale.close()
        with env.begin() as txn:
            record = pickle.loads(txn.get(_RECORD_KEY))
            raw_meta = txn.get(_META_KEY)
        if temporary:
            env.close()
        return {**record,
                "fields": {k: torch.from_numpy(v) for k, v in record["fields"].items()},
                "sample_meta": pickle.loads(raw_meta) if raw_meta else None}

    def __getitem__(self, i: int) -> dict:
        return self._decode(self._dirs[i])

    def get_by_index(self, sample_index: int) -> dict:
        """按样本编号精确读取；不存在返回 None。"""
        sample_dir = _sample_dir(self._root, sample_index)
        return self._decode(sample_dir) if sample_dir.exists() else None

    def close(self) -> None:
        """关闭本进程 LRU 中仍保留的只读 LMDB 句柄。"""
        envs = getattr(self, "_envs", None)
        while envs:
            _, env = envs.popitem(last=False)
            env.close()

    def __del__(self):
        self.close()
