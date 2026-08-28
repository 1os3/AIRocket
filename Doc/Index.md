# 文档与源文件索引

## Doc

- `Doc/开发规范.md` — 项目强制开发约定（配置外置、校验下沉、文件头、中文注释）
- `Doc/Index.md` — 本索引，全项目文档与源文件单一导航入口

## config

- `config/__init__.py` — 配置加载入口：读 yaml → 合并环境覆盖 → 构造 Config
- `config/default.yaml` — 数据集生成全部可调参数（唯一数据来源）
- `config/schema.py` — 配置类型定义与加载期校验（配置约束的单一来源）
- `config/smoke.yaml` — 冒烟环境覆盖：小网格少步数，CPU 开发平台端到端验证用

## data

- `data/__init__.py` — 数据包声明（各子模块见对应文件夹）
- `data/lbm_solver/__init__.py` — LBM 求解模块重导出
- `data/lbm_solver/lbm_solver.py` — D2Q9 多松弛时间（MRT）格子玻尔兹曼批量求解器，全流程在 GPU 张量上完成
- `data/lbm_solver/checks/lbm_solver_checks.py` — lbm_solver 入参校验
- `data/airfoil/__init__.py` — 翼型模块重导出
- `data/airfoil/airfoil.py` — 参数化翼型生成与 GPU 栅格化：NACA 四位数族 → 反弹边界掩码与有符号距离场
- `data/airfoil/checks/airfoil_checks.py` — airfoil 入参校验
- `data/sampler/__init__.py` — 采样模块重导出
- `data/sampler/sampler.py` — 工况参数采样器：拉丁超立方/随机采样 + 确定性种子派生
- `data/sampler/checks/sampler_checks.py` — sampler 输出校验
- `data/storage/__init__.py` — 存储模块重导出
- `data/storage/storage.py` — LMDB 存储后端：一个样本一个独立子库，稳态流场 + 参数元数据的写入与随机读取
- `data/storage/checks/storage_checks.py` — storage 入参校验
- `data/collector/__init__.py` — 采集模块重导出
- `data/collector/collector.py` — 采集编排：断点续采 + 分批并行求解 + 入库的端到端流程
- `data/run.py` — CLI 入口：2D 流场仿真数据集端到端生成（留在 data 包根）

## vis

- `vis/__init__.py` — 可视化包声明（各子模块见对应文件夹）
- `vis/flowfield/__init__.py` — 流场可视化模块重导出
- `vis/flowfield/flowfield.py` — 流场可视化：从 LMDB 读取稳态样本并渲染多面板 PNG
- `vis/flowfield/checks/flowfield_checks.py` — flowfield 渲染输入校验
- `vis/run.py` — CLI 入口：流场可视化（从 LMDB 渲染 PNG，留在 vis 包根）
