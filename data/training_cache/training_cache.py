"""把原始稳态 LMDB 预处理为可直接训练的独立单库缓存

模块: data/training_cache/training_cache.py
依赖: lmdb, numpy, torch, data.airfoil, data.potential_initializer, data.storage
读取配置: seed, device, grid.*, airfoil.*, solver.potential_*, storage.path,
          training_data.*
对外接口:
    - prepare_training_cache(cfg) -> dict
    - TrainingFlowDataset: 按 train/val/test/all 划分读取缓存
说明: 原始 LMDB 永不修改；缓存记录使用 float32 并逐条 zlib 无损压缩。
"""

import hashlib
import json
import os
import pickle
import zlib
from pathlib import Path
from types import SimpleNamespace

import lmdb
import numpy as np
import torch

from data.airfoil import build_airfoil_geometry, build_airfoil_polygon
from data.potential_initializer import build_potential_initial
from data.storage import FlowFieldDataset
from data.training_cache.checks import check_source_sample

__all__ = ["prepare_training_cache", "TrainingFlowDataset"]

_CACHE_VERSION = "1.0"
_FINGERPRINT_KEY = b"meta/fingerprint"
_MANIFEST_KEY = b"meta/manifest"
_READER_PROCESS_ID = os.getpid()
_READER_ENVIRONMENTS = {}


def _record_key(index: int) -> bytes:
    return f"sample/{index:08d}".encode()


def _encode(record: dict, level: int) -> bytes:
    return zlib.compress(pickle.dumps(record, protocol=4), level)


def _decode(raw: bytes) -> dict:
    return pickle.loads(zlib.decompress(raw))


def _reader_environment(path: str):
    global _READER_PROCESS_ID
    process_id = os.getpid()
    if process_id != _READER_PROCESS_ID:
        # fork 后丢弃父进程继承的 Python 引用；每个 worker 必须自行打开 LMDB。
        _READER_ENVIRONMENTS.clear()
        _READER_PROCESS_ID = process_id
    key = str(Path(path).resolve())
    if key not in _READER_ENVIRONMENTS:
        _READER_ENVIRONMENTS[key] = lmdb.open(
            key, readonly=True, lock=False, readahead=False, subdir=False)
    return _READER_ENVIRONMENTS[key]


def _fingerprint(cfg, indices: list[int], source_meta: dict | None) -> str:
    payload = {
        "cache_version": _CACHE_VERSION,
        "source": str(Path(cfg.storage.path).resolve()),
        "indices": indices,
        "grid": cfg.grid.__dict__,
        # SDF 分块只改变性能且已要求逐位等价，不应使已有缓存失效。
        "airfoil": {"n_points": cfg.airfoil.n_points},
        "potential": {key: getattr(cfg.solver, key) for key in
                      ("potential_panels", "potential_blend", "potential_speed_limit")},
        "source_meta": source_meta,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _split_indices(indices: list[int], ratios: list[float], seed: int) -> dict:
    ranked = sorted(indices, key=lambda index: hashlib.blake2b(
        f"{seed}:{index}".encode(), digest_size=8).digest())
    total = len(ranked)
    train_count = max(1, round(total * ratios[0])) if total else 0
    val_count = round(total * ratios[1]) if total >= 3 else 0
    train_count = min(train_count, total - val_count)
    return {
        "train": ranked[:train_count],
        "val": ranked[train_count:train_count + val_count],
        "test": ranked[train_count + val_count:],
    }


def _boundary_grid(cfg, plan, device: torch.device) -> torch.Tensor:
    polygon = build_airfoil_polygon(
        cfg, plan.naca_m, plan.naca_p, plan.naca_t, plan.aoa_deg, device).float()
    x = (polygon[:, 0] * cfg.grid.chord + cfg.grid.x_le) * (2.0 / cfg.grid.nx) - 1.0
    y = (polygon[:, 1] * cfg.grid.chord + cfg.grid.y_center) * (2.0 / cfg.grid.ny) - 1.0
    return torch.stack([x, y], dim=1)


def _sample_config(cfg, sample: dict):
    """用样本自带快照恢复几何坐标；训练配置只能提供缺失的旧字段。"""
    meta = sample.get("sample_meta") or {}
    snapshot = meta.get("config") or {}
    grid = SimpleNamespace(**snapshot.get("grid", cfg.grid.__dict__))
    stored_airfoil = snapshot.get("airfoil", {})
    airfoil = SimpleNamespace(**{
        key: stored_airfoil.get(key, getattr(cfg.airfoil, key)) for key in
        ("n_points", "sdf_chunk_cpu", "sdf_chunk_cuda")
    })
    stored_solver = snapshot.get("solver", {})
    solver = SimpleNamespace(**{
        key: stored_solver.get(key, getattr(cfg.solver, key)) for key in
        ("potential_panels", "potential_blend", "potential_speed_limit")
    })
    return SimpleNamespace(grid=grid, airfoil=airfoil, solver=solver)


def _prepare_record(cfg, sample: dict, device: torch.device) -> dict:
    check_source_sample(sample, cfg.grid)
    params = sample["params"]
    plan = SimpleNamespace(**params)
    geometry_cfg = _sample_config(cfg, sample)
    assert geometry_cfg.grid.nx == cfg.grid.nx and geometry_cfg.grid.ny == cfg.grid.ny, (
        f"样本 {sample['index']} 网格尺寸与模型配置不一致")
    mask, sdf = build_airfoil_geometry(
        geometry_cfg, plan.naca_m, plan.naca_p, plan.naca_t, plan.aoa_deg, device)
    stored_mask = sample["fields"]["mask"].to(device)
    assert torch.equal(mask, stored_mask), f"样本 {sample['index']} 的重建翼型 mask 与 GT 不一致"
    u_lb = torch.tensor([params["u_lb"]], dtype=torch.float32, device=device)
    potential = build_potential_initial(geometry_cfg, [plan], u_lb, mask.unsqueeze(0), device)
    ux0, uy0 = potential["ux"][0], potential["uy"][0]
    p0 = 0.5 * (u_lb[0].square() - ux0.square() - uy0.square())
    fluid = ~mask
    p0 = p0 - p0.masked_fill(~fluid, 0.0).sum() / fluid.sum().clamp_min(1)
    scale_u = u_lb[0]
    scale_p = scale_u.square()
    fields = {key: value.to(device=device, dtype=torch.float32)
              for key, value in sample["fields"].items() if key in ("ux", "uy", "p")}
    inputs = torch.stack([
        (sdf.float() / float(geometry_cfg.grid.chord)).clamp(-1.0, 1.0),
        ux0 / scale_u, uy0 / scale_u, p0 / scale_p,
    ])
    target = torch.stack([
        (fields["ux"] - ux0) / scale_u,
        (fields["uy"] - uy0) / scale_u,
        (fields["p"] - p0) / scale_p,
    ])
    reynolds = params.get("reynolds_lattice", params["reynolds"])
    conditions = torch.tensor(
        [np.log(reynolds), np.log(params["nu"]), params["u_lb"]], dtype=torch.float32)
    return {
        "index": int(sample["index"]),
        "inputs": inputs.cpu().numpy(),
        "target": target.cpu().numpy(),
        "mask": mask.cpu().numpy(),
        "conditions": conditions.numpy(),
        "boundary": _boundary_grid(geometry_cfg, plan, device).cpu().numpy(),
        "u_lb": float(params["u_lb"]),
        "nu": float(params["nu"]),
        "chord": float(params.get("chord", geometry_cfg.grid.chord)),
    }


def _statistics(env, splits: dict) -> tuple[list, list, list]:
    condition_values, target_energy = [], torch.zeros(3, dtype=torch.float64)
    target_count = 0
    with env.begin() as txn:
        for index in splits["train"]:
            record = _decode(txn.get(_record_key(index)))
            condition_values.append(torch.from_numpy(record["conditions"]).double())
            target = torch.from_numpy(record["target"]).double()
            fluid = ~torch.from_numpy(record["mask"])
            target_energy += target.square().masked_fill(~fluid.unsqueeze(0), 0.0).sum((1, 2))
            target_count += int(fluid.sum())
    conditions = torch.stack(condition_values)
    condition_mean = conditions.mean(0)
    condition_std = conditions.std(0, unbiased=False).clamp_min(1.0e-6)
    target_rms = (target_energy / max(target_count, 1)).sqrt().clamp_min(1.0e-6)
    return condition_mean.tolist(), condition_std.tolist(), target_rms.tolist()


def prepare_training_cache(cfg) -> dict:
    """从 cfg.storage.path 构建独立缓存，重复执行会跳过已完成样本。"""
    source = FlowFieldDataset(cfg.storage.path, cfg.storage.reader_cache_size)
    assert len(source) > 0, "原始 LMDB 无样本，无法准备训练缓存"
    indices = source.indices
    first_sample = source[0]
    fingerprint = _fingerprint(cfg, indices, first_sample.get("sample_meta"))
    root = Path(cfg.training_data.cache_path)
    root.mkdir(parents=True, exist_ok=True)
    map_size = max(1 << 30, len(indices) * cfg.grid.nx * cfg.grid.ny * 40)
    env = lmdb.open(str(root / "data.lmdb"), map_size=map_size, subdir=False)
    with env.begin(write=True) as txn:
        existing = txn.get(_FINGERPRINT_KEY)
        if existing is not None and existing.decode() != fingerprint:
            cursor = txn.cursor()
            has_records = cursor.set_range(b"sample/") and cursor.key().startswith(b"sample/")
            assert not has_records, (
                "训练缓存与当前源数据/几何配置不匹配，请改用新的 training_data.cache_path")
        txn.put(_FINGERPRINT_KEY, fingerprint.encode())
    device = torch.device("cuda" if cfg.device == "auto" and torch.cuda.is_available() else
                          (cfg.device if cfg.device != "auto" else "cpu"))
    if device.type == "cpu":
        torch.set_num_threads(cfg.training_data.prepare_cpu_threads)
    written = 0
    for start in range(0, len(source), cfg.training_data.prepare_batch_size):
        batch_indices = range(start, min(start + cfg.training_data.prepare_batch_size, len(source)))
        for position in batch_indices:
            index = indices[position]
            with env.begin() as txn:
                exists = txn.get(_record_key(index)) is not None
            if exists:
                continue
            record = _prepare_record(cfg, source[position], device)
            with env.begin(write=True) as txn:
                txn.put(_record_key(index), _encode(record, cfg.training_data.compression_level))
            written += 1
            print(f"[prepare] {position + 1}/{len(source)} 样本ID {index} 已缓存")
    splits = _split_indices(indices, cfg.training_data.split, cfg.training_data.split_seed)
    condition_mean, condition_std, target_rms = _statistics(env, splits)
    manifest = {
        "version": _CACHE_VERSION,
        "fingerprint": fingerprint,
        "count": len(indices),
        "splits": splits,
        "condition_mean": condition_mean,
        "condition_std": condition_std,
        "target_rms": target_rms,
    }
    with env.begin(write=True) as txn:
        txn.put(_MANIFEST_KEY, pickle.dumps(manifest, protocol=4))
    env.sync()
    env.close()
    with open(root / "manifest.json", "w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)
    source.close()
    print(f"[prepare] 完成：总数={len(indices)} 新增={written} 划分="
          f"{ {key: len(value) for key, value in splits.items()} }")
    return manifest


class TrainingFlowDataset(torch.utils.data.Dataset):
    """从集中缓存读取模型输入；LMDB 句柄在各 DataLoader 进程内惰性创建。"""

    def __init__(self, cfg, split: str = "train"):
        assert split in ("train", "val", "test", "all"), "split 必须为 train|val|test|all"
        self._path = str(Path(cfg.training_data.cache_path) / "data.lmdb")
        if not Path(self._path).exists():
            raise FileNotFoundError("训练缓存不存在，请先运行 train/run.py --mode prepare")
        env = lmdb.open(self._path, readonly=True, lock=False, subdir=False)
        with env.begin() as txn:
            raw = txn.get(_MANIFEST_KEY)
        env.close()
        assert raw is not None, "训练缓存尚未完成，无 manifest"
        self.manifest = pickle.loads(raw)
        all_indices = sum(self.manifest["splits"].values(), [])
        self.indices = all_indices if split == "all" else self.manifest["splits"][split]
        self._condition_mean = torch.tensor(self.manifest["condition_mean"], dtype=torch.float32)
        self._condition_std = torch.tensor(self.manifest["condition_std"], dtype=torch.float32)
        self._target_rms = torch.tensor(self.manifest["target_rms"], dtype=torch.float32)
        self._env = None
        self._env_process_id = None

    def _environment(self):
        process_id = os.getpid()
        if self._env is None or self._env_process_id != process_id:
            self._env = _reader_environment(self._path)
            self._env_process_id = process_id
        return self._env

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int) -> dict:
        index = self.indices[position]
        with self._environment().begin() as txn:
            record = _decode(txn.get(_record_key(index)))
        target = torch.from_numpy(record["target"]).float()
        return {
            "index": torch.tensor(record["index"]),
            "inputs": torch.from_numpy(record["inputs"]).float(),
            "target": target / self._target_rms.view(3, 1, 1),
            "target_scale": self._target_rms.clone(),
            "mask": torch.from_numpy(record["mask"]),
            "conditions": (torch.from_numpy(record["conditions"]).float() - self._condition_mean)
                          / self._condition_std,
            "boundary": torch.from_numpy(record["boundary"]).float(),
            "u_lb": torch.tensor(record["u_lb"], dtype=torch.float32),
            "nu": torch.tensor(record["nu"], dtype=torch.float32),
            "chord": torch.tensor(record["chord"], dtype=torch.float32),
        }

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_env"] = None
        state["_env_process_id"] = None
        return state
