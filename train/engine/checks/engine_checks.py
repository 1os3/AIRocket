def check_dataset(dataset, split: str) -> None:
    # 校验对象: train/evaluate 使用的数据集 —— 选定划分必须包含至少一个样本
    assert len(dataset) > 0, f"训练缓存的 {split} 划分为空"


def check_training_initialization(checkpoint, pretrained_checkpoint) -> None:
    # 校验对象: train_model 的初始化来源 —— 断点恢复与预训练初始化语义互斥
    assert checkpoint is None or pretrained_checkpoint is None, (
        "checkpoint 断点恢复与 pretrained_checkpoint 微调初始化不能同时使用")
