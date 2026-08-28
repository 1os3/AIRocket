def check_naca_params(m, p, t) -> None:
    # 校验对象: naca4_polygon 入参 m/p/t —— 四位数翼型参数的物理合法域
    assert 0.0 <= m <= 0.1, f"最大弯度 m={m} 须在 [0, 0.1]"
    assert 0.0 < p < 1.0, f"弯度位置 p={p} 须在 (0, 1)，否则弯度线公式奇异"
    assert 0.0 < t <= 0.3, f"最大厚度 t={t} 须在 (0, 0.3]"
