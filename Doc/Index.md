# 文档与源文件索引

## Doc

- `Doc/开发规范.md` — 项目强制开发约定（配置外置、校验下沉、文件头、中文注释）
- `Doc/Index.md` — 本索引，全项目文档与源文件单一导航入口
- `Doc/稳态加速.md` — 严格稳态采集的算法、基准、误差与后续加速路线
- `Doc/模型训练.md` — 流场残差 Transformer、训练缓存、物理损失与精度边界说明

## config

- `config/__init__.py` — 配置加载入口：读 yaml → 合并环境覆盖 → 构造 Config
- `config/default.yaml` — 数据生成、模型与训练全部可调参数（唯一数据来源）
- `config/finetune.yaml` — 预训练模型的小学习率与翼面邻域侧重微调覆盖配置
- `config/schema.py` — 配置类型定义与加载期校验（配置约束的单一来源）
- `config/train_smoke.yaml` — 正式模型结构的 CPU 最小冒烟与单样本过拟合覆盖

## data

- `data/__init__.py` — 数据包声明（各子模块见对应文件夹）
- `data/lbm_solver/__init__.py` — LBM 求解模块重导出
- `data/lbm_solver/lbm_solver.py` — D2Q9 多松弛时间（MRT）格子玻尔兹曼批量求解器，全流程在 GPU 张量上完成
- `data/lbm_solver/checks/lbm_solver_checks.py` — lbm_solver 入参校验
- `data/airfoil/__init__.py` — 翼型模块重导出
- `data/airfoil/airfoil.py` — 参数化翼型生成与 GPU 栅格化：NACA 四位数族 → 反弹边界掩码与有符号距离场
- `data/airfoil/checks/airfoil_checks.py` — airfoil 入参校验
- `data/potential_initializer/__init__.py` — 无粘源面元初值模块重导出
- `data/potential_initializer/potential_initializer.py` — 无粘源面元近似：为黏性 LBM 构造满足翼面不穿透的初始速度场
- `data/potential_initializer/checks/potential_initializer_checks.py` — potential_initializer 入参校验
- `data/sampler/__init__.py` — 采样模块重导出
- `data/sampler/sampler.py` — 工况参数采样器：拉丁超立方/随机采样 + 确定性种子派生
- `data/sampler/checks/sampler_checks.py` — sampler 输出校验
- `data/storage/__init__.py` — 存储模块重导出
- `data/storage/storage.py` — LMDB 存储后端：一个样本一个独立子库，稳态流场 + 参数元数据的写入与随机读取
- `data/storage/checks/storage_checks.py` — storage 入参校验
- `data/collector/__init__.py` — 采集模块重导出
- `data/collector/collector.py` — 采集编排：断点续采 + 分批并行求解 + 入库的端到端流程
- `data/collector/checks/__init__.py` — 采集编排校验重导出
- `data/collector/checks/collector_checks.py` — 校验批次粗细网格步数记录与启用语义
- `data/run.py` — CLI 入口：2D 流场仿真数据集端到端生成（留在 data 包根）
- `data/training_cache/__init__.py` — 训练缓存模块重导出
- `data/training_cache/training_cache.py` — 原始 LMDB 到独立训练缓存的精确几何与残差预处理
- `data/training_cache/checks/training_cache_checks.py` — 训练缓存源样本校验

## model

- `model/__init__.py` — 模型包声明，公开流场残差 Transformer
- `model/flow_transformer/__init__.py` — 流场残差 Transformer 模块重导出
- `model/flow_transformer/flow_transformer.py` — 12 层静态注意力残差 Transformer 与三级 FP32 PixelShuffle 解码器
- `model/flow_transformer/checks/flow_transformer_checks.py` — 流场残差 Transformer 输入校验

## train

- `train/__init__.py` — 训练包声明
- `train/run.py` — CLI 入口：准备缓存、训练、微调、评估与 CPU 最小冒烟
- `train/losses/__init__.py` — 流场损失模块重导出
- `train/losses/losses.py` — 监督、翼面邻域侧重、稳态 Navier–Stokes 与精确边界物理损失
- `train/losses/checks/losses_checks.py` — 流场损失输入校验
- `train/engine/__init__.py` — 训练引擎模块重导出
- `train/engine/engine.py` — 单卡训练、微调、评估、断点恢复与 CPU 最小过拟合验收
- `train/engine/checks/engine_checks.py` — 训练数据划分非空校验
- `train/fine_tuning/__init__.py` — 预训练模型微调模块重导出
- `train/fine_tuning/fine_tuning.py` — 兼容参数部分加载与独立优化状态的模型微调入口
- `train/fine_tuning/checks/fine_tuning_checks.py` — 预训练检查点与兼容参数校验

## vis

- `vis/__init__.py` — 可视化包声明（各子模块见对应文件夹）
- `vis/flowfield/__init__.py` — 流场可视化模块重导出
- `vis/flowfield/flowfield.py` — 流场可视化：从 LMDB 读取稳态样本并渲染多面板 PNG
- `vis/flowfield/checks/flowfield_checks.py` — flowfield 渲染输入校验
- `vis/training_curves/__init__.py` — 训练曲线可视化模块重导出
- `vis/training_curves/training_curves.py` — 训练损失、验证损失与梯度健康状态的四面板曲线
- `vis/training_curves/checks/training_curves_checks.py` — 训练曲线主日志存在性校验
- `vis/inference/__init__.py` — 模型推理可视化模块重导出
- `vis/inference/inference.py` — 用训练检查点或确定性随机初始化模型渲染流场推理对比图
- `vis/inference/checks/__init__.py` — 模型推理可视化校验重导出
- `vis/inference/checks/inference_checks.py` — 模型推理样本与检查点兼容性校验
- `vis/run.py` — CLI 入口：流场样本、训练曲线与模型推理可视化（留在 vis 包根）
