"""配置类型定义与加载期校验（配置约束的单一来源）

模块: config/schema.py
依赖: 标准库 dataclasses
读取配置: 全部（本文件定义其类型与约束）
对外接口:
    - Grid / Solver / Airfoil / Sampler / Storage / Vis: 数据生成配置 dataclass
    - TrainingData / Model / Training / Loss / Evaluation: 模型训练配置 dataclass
    - Config / config_from_dict(raw): 完整配置及构造入口
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
    grid_sequence: bool = False
    grid_sequence_scale: int = 2
    grid_sequence_conv_tol: float = 1.0e-5
    grid_sequence_policy: str = "auto"
    torch_compile: bool = False
    steady_hits: int = 1
    precondition_gamma: float = 1.0
    initializer: str = "uniform"
    potential_panels: int = 64
    potential_blend: float = 0.5
    potential_speed_limit: float = 1.5
    sample_continuation: bool = False
    continuation_bank_size: int = 64

    def __post_init__(self):
        # 校验对象: solver.* —— 批大小/步数/间隔为正、容差为正、mrt_s 恰为 7 项且落在 (0,2)
        assert self.batch_size > 0, "solver.batch_size 必须 > 0"
        assert self.boundary in ("bounce_back", "bouzidi"), (
            f"solver.boundary 仅支持 bounce_back|bouzidi，得到 {self.boundary}")
        assert self.max_steps > 0 and self.check_interval > 0, "solver 步数配置必须为正"
        assert self.conv_tol > 0, "solver.conv_tol 必须 > 0"
        assert self.grid_sequence_scale >= 2, "solver.grid_sequence_scale 必须 >= 2"
        assert self.grid_sequence_conv_tol > 0, "solver.grid_sequence_conv_tol 必须 > 0"
        assert self.grid_sequence_policy in ("auto", "always"), (
            "solver.grid_sequence_policy 仅支持 auto|always")
        assert self.steady_hits >= 1, "solver.steady_hits 必须 >= 1"
        assert 0.0 < self.precondition_gamma <= 1.0, (
            "solver.precondition_gamma 必须在 (0,1]；1 表示关闭预条件")
        assert self.initializer in ("uniform", "potential"), (
            "solver.initializer 仅支持 uniform|potential")
        assert self.potential_panels >= 16 and self.potential_panels % 2 == 0, (
            "solver.potential_panels 必须是 >=16 的偶数")
        assert 0.0 <= self.potential_blend <= 1.0, (
            "solver.potential_blend 必须在 [0,1]")
        assert self.potential_speed_limit >= 1.0, (
            "solver.potential_speed_limit 必须 >= 1")
        assert self.continuation_bank_size >= 1, (
            "solver.continuation_bank_size 必须 >= 1")
        assert len(self.mrt_s) == 7 and all(0.0 < s < 2.0 for s in self.mrt_s), (
            "solver.mrt_s 需 7 项且均在 (0,2)（Lallemand-Luo 稳定域）"
        )


@dataclass(frozen=True)
class Airfoil:
    n_points: int
    sdf_chunk_cpu: int
    sdf_chunk_cuda: int

    def __post_init__(self):
        # 校验对象: airfoil.n_points —— 至少 8 点才能勾勒厚度分布
        assert self.n_points >= 8, "airfoil.n_points 必须 >= 8"
        assert self.sdf_chunk_cpu > 0 and self.sdf_chunk_cuda > 0, "SDF 分块必须 > 0"


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
    reader_cache_size: int

    def __post_init__(self):
        # 校验对象: storage.map_size_mb —— 单样本子库地址空间，须为正
        assert self.map_size_mb > 0, "storage.map_size_mb 必须 > 0"
        assert self.reader_cache_size >= 0, "storage.reader_cache_size 必须 >= 0"


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
class TrainingData:
    cache_path: str
    split: list
    split_seed: int
    compression_level: int
    prepare_batch_size: int
    prepare_cpu_threads: int

    def __post_init__(self):
        # 校验对象: training_data.* —— 划分覆盖全集，压缩级别与准备批量合法
        assert len(self.split) == 3 and all(0.0 <= x <= 1.0 for x in self.split), (
            "training_data.split 必须为三个 [0,1] 比例")
        assert abs(sum(self.split) - 1.0) < 1.0e-8, "training_data.split 之和必须为 1"
        assert 0 <= self.compression_level <= 9, "compression_level 必须在 [0,9]"
        assert self.prepare_batch_size > 0, "prepare_batch_size 必须 > 0"
        assert self.prepare_cpu_threads > 0, "prepare_cpu_threads 必须 > 0"


@dataclass(frozen=True)
class Model:
    input_channels: int
    output_channels: int
    patch_size: int
    dim: int
    depth: int
    heads: int
    ffn_hidden: int
    decoder_channels: list
    norm_eps: float
    dropout: float

    def __post_init__(self):
        # 校验对象: model.* —— 注意力、SwiGLU 与三级像素洗牌的维度必须可整除
        assert self.input_channels > 0 and self.output_channels > 0, "模型通道数必须 > 0"
        assert self.patch_size > 0 and self.dim > 0 and self.depth > 0, "模型尺寸必须 > 0"
        assert self.dim % self.heads == 0, "model.dim 必须能被 heads 整除"
        assert self.dim % 4 == 0, "二维正余弦位置编码要求 model.dim 能被 4 整除"
        assert self.ffn_hidden % 2 == 0, "SwiGLU 要求 ffn_hidden 为偶数"
        assert len(self.decoder_channels) == 3 and all(x > 0 for x in self.decoder_channels), (
            "decoder_channels 必须是三个正整数")
        assert self.norm_eps > 0.0 and 0.0 <= self.dropout < 1.0, "Norm/dropout 配置非法"


@dataclass(frozen=True)
class Training:
    epochs: int
    batch_size: int
    gradient_accumulation: int
    num_workers: int
    lr: float
    min_lr: float
    beta1: float
    beta2: float
    weight_decay: float
    warmup_ratio: float
    grad_clip: float
    amp_dtype: str
    float32_matmul_precision: str
    torch_compile: bool
    gradient_monitor: bool
    gradient_small_threshold: float
    output_dir: str
    checkpoint_every: int
    log_every: int
    max_steps: int | None
    smoke_min_improvement: float

    def __post_init__(self):
        # 校验对象: training.* —— 优化器、精度与循环参数必须处于有效范围
        assert self.epochs > 0 and self.batch_size > 0 and self.gradient_accumulation > 0, (
            "训练轮数、批量和梯度累积必须 > 0")
        assert self.num_workers >= 0 and self.lr > 0.0 and 0.0 <= self.min_lr <= self.lr, (
            "训练 worker/学习率配置非法")
        assert 0.0 < self.beta1 < 1.0 and 0.0 < self.beta2 < 1.0, "AdamW beta 必须在 (0,1)"
        assert self.weight_decay >= 0.0 and 0.0 <= self.warmup_ratio < 1.0, (
            "weight_decay/warmup_ratio 配置非法")
        assert self.grad_clip > 0.0 and self.amp_dtype in ("auto", "bfloat16", "float16"), (
            "grad_clip/amp_dtype 配置非法")
        assert self.float32_matmul_precision in ("highest", "high", "medium"), (
            "float32_matmul_precision 仅支持 highest|high|medium")
        assert isinstance(self.gradient_monitor, bool) and self.gradient_small_threshold > 0.0, (
            "gradient_monitor 必须为布尔值且 gradient_small_threshold 必须 > 0")
        assert self.checkpoint_every > 0 and self.log_every > 0, "保存与日志间隔必须 > 0"
        assert self.max_steps is None or self.max_steps > 0, "max_steps 必须 > 0 或为 null"
        assert 0.0 < self.smoke_min_improvement < 1.0, "smoke_min_improvement 必须在 (0,1)"


@dataclass(frozen=True)
class Loss:
    huber_delta: float
    data_weight: float
    gradient_weight: float
    divergence_weight: float
    momentum_weight: float
    boundary_weight: float
    physics_warmup_ratio: float

    def __post_init__(self):
        # 校验对象: loss.* —— 损失尺度为正、权重非负、升权比例合法
        assert self.huber_delta > 0.0, "loss.huber_delta 必须 > 0"
        weights = (self.data_weight, self.gradient_weight, self.divergence_weight,
                   self.momentum_weight, self.boundary_weight)
        assert all(x >= 0.0 for x in weights), "损失权重不得为负"
        assert 0.0 <= self.physics_warmup_ratio <= 1.0, "physics_warmup_ratio 必须在 [0,1]"


@dataclass(frozen=True)
class Evaluation:
    num_visualizations: int

    def __post_init__(self):
        # 校验对象: evaluation.num_visualizations —— 可为 0（关闭渲染）
        assert self.num_visualizations >= 0, "num_visualizations 必须 >= 0"


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
    training_data: TrainingData
    model: Model
    training: Training
    loss: Loss
    evaluation: Evaluation


def config_from_dict(raw: dict) -> Config:
    """把 yaml 原始 dict 构造成带校验的 Config 对象。

    参数:
        raw: load_config 合并环境覆盖后的完整 dict
    返回:
        Config；任何非法字段在构造期抛出 AssertionError
    """
    # 校验对象: 顶层键 —— 防止 yaml 笔误产生被静默忽略的多余配置段
    known = {"seed", "version", "device", "grid", "solver", "airfoil", "sampler", "storage", "vis",
             "training_data", "model", "training", "loss", "evaluation"}
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
        training_data=TrainingData(**raw["training_data"]),
        model=Model(**raw["model"]),
        training=Training(**raw["training"]),
        loss=Loss(**raw["loss"]),
        evaluation=Evaluation(**raw["evaluation"]),
    )
