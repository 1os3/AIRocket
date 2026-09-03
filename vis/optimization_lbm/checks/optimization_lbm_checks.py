from pathlib import Path

import torch

from data.lbm_solver import CS2


def check_lbm_evaluation_path(result_path: str | Path) -> None:
    # 校验对象: LBM 复核输入 result_path —— 必须是优化器产生的完整 JSON 文件
    assert Path(result_path).is_file(), f"优化结果不存在：{result_path}"


def check_lbm_evaluation_result(result: dict, cfg) -> None:
    # 校验对象: LBM 复核输入 result —— 必须包含工况、参数与目标函数定义
    required = {"flow", "parameters", "best", "objective"}
    assert required <= set(result), f"优化结果缺少字段：{sorted(required - set(result))}"
    # 校验对象: result.flow —— LBM 必须能按优化时的同一实际工况求解
    flow = result["flow"]
    assert {"u_lb", "reynolds", "aoa_deg"} <= set(flow), "优化结果 flow 字段不完整"
    flow_values = torch.tensor(
        [flow["u_lb"], flow["reynolds"], flow["aoa_deg"]], dtype=torch.float64)
    assert bool(torch.isfinite(flow_values).all()), "优化结果 flow 含 NaN/Inf"
    assert 0.0 < flow["u_lb"] < 0.5 and flow["reynolds"] > 0.0, (
        "优化结果的 u_lb/reynolds 非法")
    tau = 0.5 + flow["u_lb"] * cfg.grid.chord / flow["reynolds"] / CS2
    assert tau >= cfg.sampler.tau_min, (
        f"目标工况 tau={tau:.6g} 小于 sampler.tau_min={cfg.sampler.tau_min}；"
        "collector 会改变实际 Re，无法同工况复核，请降低 tau_min 或调整工况")
    # 校验对象: result.parameters/result.best —— 初始和最优 NACA 参数必须齐全且有限
    names = ("naca_m", "naca_p", "naca_t")
    assert all(name in result["parameters"] and name in result["best"] for name in names), (
        "优化结果缺少初始或最优 NACA 参数")
    assert all(isinstance(result["parameters"][name], dict)
               and "initial" in result["parameters"][name] for name in names), (
        "优化结果 parameters 缺少 initial")
    values = [result["parameters"][name]["initial"] for name in names]
    values += [result["best"][name] for name in names]
    assert bool(torch.isfinite(torch.tensor(values, dtype=torch.float64)).all()), (
        "优化结果的 NACA 参数含 NaN/Inf")
    # 校验对象: result.objective —— 必须足以复现优化阶段的目标函数
    objective = result["objective"]
    objective_keys = {"mode", "target_lift", "lift_weight", "drag_weight", "drag_epsilon"}
    assert objective_keys <= set(objective), "优化结果 objective 字段不完整"
    assert objective["mode"] in {"maximize_lift", "minimize_drag", "maximize_lift_to_drag",
                                 "target_lift_min_drag"}, "优化结果 objective.mode 非法"
    objective_values = torch.tensor([
        objective["target_lift"], objective["lift_weight"],
        objective["drag_weight"], objective["drag_epsilon"],
    ], dtype=torch.float64)
    assert bool(torch.isfinite(objective_values).all()), "优化结果 objective 含 NaN/Inf"
    assert (objective["lift_weight"] > 0.0 and objective["drag_weight"] > 0.0
            and objective["drag_epsilon"] > 0.0), "优化结果 objective 权重/稳定项必须 > 0"


def check_lbm_solver_output(out: dict, count: int, cfg) -> None:
    # 校验对象: solve_plans 的 LBM 输出 —— 每个对比案例都必须有场量、mask 与收敛记录
    required = {"rho", "ux", "uy", "p", "mask", "steps", "coarse_steps",
                "total_steps", "converged"}
    assert required <= set(out), f"LBM 输出缺少字段：{sorted(required - set(out))}"
    expected = (count, cfg.grid.ny, cfg.grid.nx)
    assert all(tuple(out[name].shape) == expected
               for name in ("rho", "ux", "uy", "p", "mask")), (
        f"LBM 场量形状必须为 {expected}")
    assert all(len(out[name]) == count
               for name in ("steps", "coarse_steps", "total_steps", "converged")), (
        "LBM 收敛记录数量与案例数不一致")
