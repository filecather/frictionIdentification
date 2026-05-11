"""
两步分层 Stribeck 参数辨识

第 1 步：高速区线性回归 → Tc, Tv
    当 |v| >> vs 时，exp(-(v/vs)^2) ≈ 0
    模型退化为: T ≈ Tc * sign(v) + Tv * v

第 2 步：对数线性回归 → Ts, vs
    残差 y = |T_target - Tc*sign(v) - Tv*v| = (Ts-Tc) * exp(-v^2/vs^2)
    取对数: ln(y) = ln(Ts-Tc) - (1/vs^2) * v^2
    对 v^2 做线性回归即可解析求解

第 3 步（可选）: PSO 精调，以解析解为初始种子优化
"""

import numpy as np
import csv
import os
import json

DATA_FILE = "velocity_friction_2d.csv"
TRUE_PARAMS = np.array([0.50, 0.80, 0.30, 0.15])


def load_data(filepath: str) -> tuple:
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        v_list, t_list = [], []
        for row in reader:
            v_list.append(float(row[0]))
            t_list.append(float(row[1]))
    return np.array(v_list), np.array(t_list)


def step1_linear_regression(v: np.ndarray, t: np.ndarray, threshold: float = 0.6):
    """
    高速区线性回归求 Tc, Tv
    模型: T = Tc * sign(v) + Tv * v
    """
    mask = np.abs(v) > threshold
    v_high = v[mask]
    t_high = t[mask]

    # 设计矩阵: [sign(v), v]
    X = np.column_stack([np.sign(v_high), v_high])
    params, residuals, rank, s = np.linalg.lstsq(X, t_high, rcond=None)
    Tc, Tv = params

    # 拟合质量
    t_pred = X @ params
    r2 = 1 - np.sum((t_high - t_pred) ** 2) / np.sum((t_high - np.mean(t_high)) ** 2)

    return Tc, Tv, r2, np.sum(mask)


def step2_log_linear_regression(v: np.ndarray, t: np.ndarray, Tc: float, Tv: float):
    """
    对数线性回归求 Ts, vs
    y = |T - Tc*sign(v) - Tv*v| = (Ts-Tc) * exp(-v^2/vs^2)
    ln(y) = ln(Ts-Tc) - (1/vs^2) * v^2
    """
    # 正速度残差
    mask_pos = v > 0
    y_pos = t[mask_pos] - Tc - Tv * v[mask_pos]

    # 负速度残差
    mask_neg = v < 0
    y_neg = -(t[mask_neg] + Tc - Tv * v[mask_neg])

    # 合并有效数据 (y > 0)
    y_valid = np.concatenate([y_pos[y_pos > 1e-6], y_neg[y_neg > 1e-6]])
    v_sq_valid = np.concatenate([
        v[mask_pos][y_pos > 1e-6] ** 2,
        v[mask_neg][y_neg > 1e-6] ** 2
    ])

    if len(y_valid) < 10:
        raise ValueError("有效低速数据不足，无法拟合 Ts/vs")

    log_y = np.log(y_valid)
    X = np.column_stack([np.ones(len(log_y)), v_sq_valid])
    params, residuals, rank, s = np.linalg.lstsq(X, log_y, rcond=None)
    intercept, slope = params

    # 解析求解
    vs = np.sqrt(-1.0 / slope) if slope < 0 else 0.1
    Ts = Tc + np.exp(intercept)

    # 拟合质量
    log_y_pred = X @ params
    r2 = 1 - np.sum((log_y - log_y_pred) ** 2) / np.sum((log_y - np.mean(log_y)) ** 2)

    return Ts, vs, r2, len(y_valid)


def compute_fitness(v: np.ndarray, t: np.ndarray, params: np.ndarray) -> float:
    """计算 MSE"""
    Tc, Ts, vs, Tv = params
    sign_approx = np.tanh(v / 1e-3)
    stribeck_term = (Ts - Tc) * np.exp(-(v / vs) ** 2)
    t_pred = (Tc + stribeck_term) * sign_approx + Tv * v
    return np.mean((t_pred - t) ** 2)


def step3_pso_refinement(v: np.ndarray, t: np.ndarray, init_params: np.ndarray,
                         n_particles: int = 50, max_iter: int = 100):
    """
    以解析解为种子，在小范围内 PSO 精调
    """
    np.random.seed(42)
    DIM = 4

    # 在解析解附近设置搜索边界（±30%）
    LB = np.maximum(init_params * 0.5, np.array([0.01, 0.01, 0.01, 0.001]))
    UB = init_params * 1.5
    UB[1] = max(UB[1], init_params[1] + 0.5)  # Ts 上限放宽

    v_max = 0.2 * (UB - LB)

    # 初始化：以解析解为中心，加入少量随机扰动
    positions = np.random.uniform(LB, UB, size=(n_particles, DIM))
    positions[0] = init_params.copy()  # 第一个粒子为解析解
    velocities = np.random.uniform(-v_max, v_max, size=(n_particles, DIM))

    pBest = positions.copy()
    pBest_fit = np.array([compute_fitness(v, t, p) for p in positions])

    gBest_idx = np.argmin(pBest_fit)
    gBest = positions[gBest_idx].copy()
    gBest_fit = pBest_fit[gBest_idx]

    W, c1, c2 = 0.8, 1.5, 1.5

    print("\n--- PSO 精调 ---")
    print(f"{'Iter':>5} | {'Tc':>8} {'Ts':>8} {'vs':>8} {'Tv':>8} | {'MSE':>10}")
    print("-" * 55)

    for iteration in range(max_iter):
        for i in range(n_particles):
            fit = compute_fitness(v, t, positions[i])
            if fit < pBest_fit[i]:
                pBest_fit[i] = fit
                pBest[i] = positions[i].copy()

        best_idx = np.argmin(pBest_fit)
        if pBest_fit[best_idx] < gBest_fit:
            gBest_fit = pBest_fit[best_idx]
            gBest = pBest[best_idx].copy()

        if (iteration + 1) % 20 == 1 or iteration == max_iter - 1:
            print(f"{iteration+1:>5} | {gBest[0]:>8.4f} {gBest[1]:>8.4f} "
                  f"{gBest[2]:>8.4f} {gBest[3]:>8.4f} | {gBest_fit:>10.6f}")

        r1 = np.random.rand(n_particles, DIM)
        r2 = np.random.rand(n_particles, DIM)
        velocities = (W * velocities
                      + c1 * r1 * (pBest - positions)
                      + c2 * r2 * (gBest - positions))
        velocities = np.clip(velocities, -v_max, v_max)
        positions += velocities
        positions = np.clip(positions, LB, UB)

    return gBest, gBest_fit


def print_result(label: str, est: np.ndarray, true: np.ndarray):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"{'参数':>6} | {'真实值':>10} | {'辨识值':>10} | {'绝对误差':>10} | {'相对误差':>10}")
    print("-" * 60)
    names = ["Tc", "Ts", "vs", "Tv"]
    for i, name in enumerate(names):
        abs_err = abs(est[i] - true[i])
        rel_err = abs((est[i] - true[i]) / (true[i] + 1e-8)) * 100
        print(f"{name:>6} | {true[i]:>10.4f} | {est[i]:>10.4f} | {abs_err:>10.4f} | {rel_err:>10.2f}%")

    mse = compute_fitness(v_data, t_target, est)
    t_pred = (est[0] + (est[1]-est[0])*np.exp(-(v_data/est[2])**2)) * np.tanh(v_data/1e-3) + est[3]*v_data
    r2 = 1 - np.sum((t_target - t_pred)**2) / np.sum((t_target - np.mean(t_target))**2)
    print(f"\n  MSE = {mse:.6f}")
    print(f"  R²  = {r2:.6f}")


def main():
    print("=" * 60)
    print("    两步分层 Stribeck 参数辨识")
    print("=" * 60)

    global v_data, t_target
    v_data, t_target = load_data(DATA_FILE)
    print(f"\n数据样本数 : {len(v_data)}")
    print(f"速度范围   : [{v_data.min():.4f}, {v_data.max():.4f}]")

    # ========== 第 1 步：高速区线性回归 ==========
    print("\n" + "-" * 60)
    print("  第 1 步：高速区线性回归 (|v| > 0.6) → Tc, Tv")
    print("-" * 60)

    Tc_est, Tv_est, r2_1, n_high = step1_linear_regression(v_data, t_target, threshold=0.6)
    print(f"  使用数据点: {n_high}")
    print(f"  拟合 R²   : {r2_1:.6f}")
    print(f"  Tc (解析) : {Tc_est:.4f}  (真实: {TRUE_PARAMS[0]:.4f})")
    print(f"  Tv (解析) : {Tv_est:.4f}  (真实: {TRUE_PARAMS[3]:.4f})")

    # ========== 第 2 步：对数线性回归 ==========
    print("\n" + "-" * 60)
    print("  第 2 步：对数线性回归 → Ts, vs")
    print("-" * 60)

    Ts_est, vs_est, r2_2, n_low = step2_log_linear_regression(v_data, t_target, Tc_est, Tv_est)
    print(f"  使用数据点: {n_low}")
    print(f"  拟合 R²   : {r2_2:.6f}")
    print(f"  Ts (解析) : {Ts_est:.4f}  (真实: {TRUE_PARAMS[1]:.4f})")
    print(f"  vs (解析) : {vs_est:.4f}  (真实: {TRUE_PARAMS[2]:.4f})")

    # 解析结果
    analytic_params = np.array([Tc_est, Ts_est, vs_est, Tv_est])
    print_result("解析解辨识结果", analytic_params, TRUE_PARAMS)

    # ========== 第 3 步：PSO 精调 ==========
    refined, refined_fit = step3_pso_refinement(v_data, t_target, analytic_params)
    print_result("PSO 精调结果", refined, TRUE_PARAMS)

    # 保存结果
    result = {
        "true_params": TRUE_PARAMS.tolist(),
        "analytic_params": analytic_params.tolist(),
        "refined_params": refined.tolist(),
        "analytic_mse": float(compute_fitness(v_data, t_target, analytic_params)),
        "refined_mse": float(refined_fit),
    }
    with open("hierarchical_result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n✅ 结果已保存: {os.path.abspath('hierarchical_result.json')}")


if __name__ == "__main__":
    main()
