import torch

_REQUIRED = ("rho", "ux", "uy", "p", "mask")


def check_fields(fields, grid) -> None:
    # 校验对象: FlowFieldWriter.write 入参 fields —— 键齐全、形状与 grid 一致、dtype 正确
    missing = set(_REQUIRED) - set(fields)
    assert not missing, f"fields 缺少字段: {missing}"
    for k in _REQUIRED:
        assert tuple(fields[k].shape) == (grid.ny, grid.nx), (
            f"fields[{k}] 形状 {tuple(fields[k].shape)} 与 grid ({grid.ny}, {grid.nx}) 不符"
        )
    assert fields["mask"].dtype == torch.bool, "fields[mask] 须为 bool"
