"""配置类型定义与加载期校验（配置约束的单一来源）

模块: config/schema.py
依赖: 标准库 dataclasses
读取配置: 全部（本文件定义其类型与约束）
对外接口:
    - Grid / Solver / Airfoil / Sampler / Storage / Config: 配置 dataclass
    - config_from_dict(raw: dict) -> Config
说明: 约束校验只写在这里（规范 §7.1 唯一例外），运行期 _checks.py 不重复。
"""

from dataclasses import dataclass, field


def _range_check(name: str, r: list) -> None:
    # 校验对象: 所有 [lo, hi] 区间型配置项 —— 必须二元且 lo < hi
    assert isinstance(r, list) and len(r) == 2 and r[0] < r[1], (
        f"{name} 必须为 [lo, hi] 且 lo < hi，得到 {r}"
    )


@dataclass(frozen=True)
class Grid:
    nx: int
    ny: int
    chord: int
    x_le: int
    y_center: int

    def __post_init__(self):
        # 校验对象: grid.* —— 分辨率与弦长为正，翼型须完整落在域内
        assert self.nx > 0 and self.ny > 0 and self.chord > 0, "grid 分辨率/弦长必须为正"
        assert 0 < self.x_le and self.x_le + self.chord < self.nx, "翼型流向越界"
        assert 0 < self.y_center < self.ny, "grid.y_center 必须落在域内"


@dataclass(frozen=True)
class Solver:
    batch_size: int
    boundary: str
    max_steps: int
    check_interval: int
    conv_tol: float
    float64: bool
    mrt_s: list = field(default_factory=list)

    def __post_init__(self):
        # 校验对象: solver.* —— 批大小/步数/间隔为正、容差为正、mrt_s 恰为 7 项且落在 (0,2)
        assert self.batch_size > 0, "solver.batch_size 必须 > 0"
        assert self.boundary in ("bounce_back", "bouzidi"), (
            f"solver.boundary 仅支持 bounce_back|bouzidi，得到 {self.boundary}")
        assert self.max_steps > 0 and self.check_interval > 0, "solver 步数配置必须为正"
        assert self.conv_tol > 0, "solver.conv_tol 必须 > 0"
        assert len(self.mrt_s) == 7 and all(0.0 < s < 2.0 for s in self.mrt_s), (
            "solver.mrt_s 需 7 项且均在 (0,2)（Lallemand-Luo 稳定域）"
        )


@dataclass(frozen=True)
class Airfoil:
    n_points: int

    def __post_init__(self):
        # 校验对象: airfoil.n_points —— 至少 8 点才能勾勒厚度分布
        assert self.n_points >= 8, "airfoil.n_points 必须 >= 8"


@dataclass(frozen=True)
class Sampler:
    num_samples: int
    method: str
    reynolds: list
    mach: list
    aoa_deg: list
    naca_m: list
    naca_p: list
    naca_t: list
    tau_min: float
    u_lb_max: float
    u_lb_fixed: float | None = None

    def __post_init__(self):
        # 校验对象: sampler.* —— 样本数非负、方法枚举、各采样区间为合法 [lo,hi]
        assert self.num_samples >= 0, "sampler.num_samples 必须 >= 0"
        assert self.method in ("lhs", "random"), f"sampler.method 仅支持 lhs|random，得到 {self.method}"
        for name in ("reynolds", "mach", "aoa_deg", "naca_m", "naca_p", "naca_t"):
            _range_check(f"sampler.{name}", getattr(self, name))
        # 校验对象: sampler.naca_p —— 弯度位置须在 (0,1) 内，否则弯度线公式奇异
        assert self.naca_p[0] > 0.0 and self.naca_p[1] < 1.0, "sampler.naca_p 必须在 (0,1)"
        # 校验对象: sampler.tau_min / u_lb_max —— tau 须 > 0.5（粘性为正），U 上限须为正
        assert self.tau_min > 0.5, "sampler.tau_min 必须 > 0.5"
        assert self.u_lb_max > 0.0, "sampler.u_lb_max 必须 > 0"
        # 校验对象: sampler.u_lb_fixed —— 固定入口速度须满足低马赫假设
        assert self.u_lb_fixed is None or 0.0 < self.u_lb_fixed < 0.5, (
            "sampler.u_lb_fixed 须在 (0, 0.5) 或为 null")


@dataclass(frozen=True)
class Storage:
    path: str
    map_size_mb: int

    def __post_init__(self):
        # 校验对象: storage.map_size_mb —— 单样本子库地址空间，须为正
        assert self.map_size_mb > 0, "storage.map_size_mb 必须 > 0"


@dataclass(frozen=True)
class Vis:
    out_dir: str
    cmap: str
    dpi: int
    num_samples: int
    fields: list

    def __post_init__(self):
        # 校验对象: vis.* —— dpi 正、num_samples 非负、fields 为可渲染面板子集
        assert self.dpi > 0, "vis.dpi 必须 > 0"
        assert self.num_samples >= 0, "vis.num_samples 必须 >= 0"
        allowed = {"ux", "uy", "p", "rho", "speed"}
        assert self.fields and set(self.fields) <= allowed, (
            f"vis.fields 仅支持 {sorted(allowed)} 的非空子集，得到 {self.fields}")


@dataclass(frozen=True)
class Config:
    seed: int
    version: str
    device: str
    grid: Grid
    solver: Solver
    airfoil: Airfoil
    sampler: Sampler
    storage: Storage
    vis: Vis


def config_from_dict(raw: dict) -> Config:
    """把 yaml 原始 dict 构造成带校验的 Config 对象。

    参数:
        raw: load_config 合并环境覆盖后的完整 dict
    返回:
        Config；任何非法字段在构造期抛出 AssertionError
    """
    # 校验对象: 顶层键 —— 防止 yaml 笔误产生被静默忽略的多余配置段
    known = {"seed", "version", "device", "grid", "solver", "airfoil", "sampler", "storage", "vis"}
    extra = set(raw) - known
    assert not extra, f"配置存在未知顶层键: {extra}"
    assert raw["device"] in ("auto", "cpu", "cuda"), "device 仅支持 auto|cpu|cuda"
    return Config(
        seed=int(raw["seed"]),
        version=str(raw["version"]),
        device=str(raw["device"]),
        grid=Grid(**raw["grid"]),
        solver=Solver(**raw["solver"]),
        airfoil=Airfoil(**raw["airfoil"]),
        sampler=Sampler(**raw["sampler"]),
        storage=Storage(**raw["storage"]),
        vis=Vis(**raw["vis"]),
    )
