def check_render_inputs(dataset, vis_cfg) -> None:
    # 校验对象: render_samples 的数据源 —— 库内须有样本，否则无可渲染内容
    assert len(dataset) > 0, "LMDB 库内无样本，先运行 run.py 采集"
    # 校验对象: 样本字段完整性 —— 渲染所需的面板数据必须存在
    missing = {"ux", "uy", "p", "rho", "mask"} - set(dataset[0]["fields"])
    assert not missing, f"样本缺少字段 {missing}，库格式与当前代码版本不符"
