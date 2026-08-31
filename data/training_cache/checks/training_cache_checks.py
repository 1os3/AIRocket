import torch


def check_source_sample(sample: dict, grid) -> None:
    # 校验对象: prepare_training_cache 的源样本 —— 必备字段、参数与网格形状必须完整
    assert "fields" in sample and "params" in sample, "源样本缺少 fields/params"
    missing_fields = {"ux", "uy", "p", "mask"} - set(sample["fields"])
    missing_params = {"naca_m", "naca_p", "naca_t", "aoa_deg", "u_lb", "nu"} - set(sample["params"])
    assert not missing_fields, f"源样本缺少场: {missing_fields}"
    assert not missing_params, f"源样本缺少参数: {missing_params}"
    assert all(tuple(sample["fields"][key].shape) == (grid.ny, grid.nx)
               for key in ("ux", "uy", "p", "mask")), "源样本网格形状不一致"
    assert sample["fields"]["mask"].dtype == torch.bool, "源样本 mask 必须为 bool"
