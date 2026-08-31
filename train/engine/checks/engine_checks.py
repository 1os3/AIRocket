def check_dataset(dataset, split: str) -> None:
    # 校验对象: train/evaluate 使用的数据集 —— 选定划分必须包含至少一个样本
    assert len(dataset) > 0, f"训练缓存的 {split} 划分为空"
