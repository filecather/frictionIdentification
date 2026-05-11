"""
粒子群优化（PSO）辨识 Stribeck 摩擦参数
配合 pso_init.py 使用

辨识参数（4维）:
    x[0] = Tc  (Coulomb 摩擦转矩)
    x[1] = Ts  (最大静摩擦转矩)
    x[2] = vs  (Stribeck 速度)
    x[3] = Tv  (粘滞阻尼系数)

目标: 最小化预测摩擦转矩与目标摩擦转矩的均方误差 (MSE)
"""

import numpy as np
import csv
import os
import pso_init

# ==================== 用户可调参数 ====================
DATA_FILE = "velocity_friction_2d.csv"   # 输入: 速度-摩擦转矩二维数据

# PSO 超参数
MAX_ITER = 200           # 最大迭代次数
W = 0.8                  # 惯性权重
c1 = 1.5                 # 个体学习因子
c2 = 1.5                 # 社会学习因子

# 搜索边界 [Tc, Ts, vs, Tv]
LB = np.array([0.0, 0.0, 0.01, 0.0])
UB = np.array([2.0, 1.5, 2.0, 1.0])   # Ts 上限从 3.0 降至 1.5，改善收敛

# 真实参数（用于对比评估）
TRUE_PARAMS = np.array([0.50, 0.80, 0.30, 0.15])

# 初始化参数
N = 100
DIM = 4
SEED = 42
# ====================================================


def load_data(filepath: str) -> tuple:
    """加载速度-摩擦转矩二维数据"""
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        headers = next(reader)
        v_list, t_list = [], []
        for row in reader:
            v_list.append(float(row[0]))
            t_list.append(float(row[1]))
    return np.array(v_list), np.array(t_list)


def stribeck_predict(v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stribeck 模型预测（连续近似版本，便于数值优化）
    T = [Tc + (Ts - Tc) * exp(-(v/vs)^2)] * tanh(v/eps) + Tv * v
    """
    Tc, Ts, vs, Tv = params
    # 连续近似 sign(v)，eps 越小越接近理想 sign
    eps = 1e-3
    sign_approx = np.tanh(v / eps)
    stribeck_term = (Ts - Tc) * np.exp(-(v / vs) ** 2)
    T = (Tc + stribeck_term) * sign_approx + Tv * v
    return T


def fitness(params: np.ndarray, v_data: np.ndarray, t_target: np.ndarray) -> float:
    """适应度函数: MSE + 约束惩罚"""
    T_pred = stribeck_predict(v_data, params)
    mse = np.mean((T_pred - t_target) ** 2)

    Tc, Ts, vs, Tv = params
    penalty = 0.0

    # 约束: Ts >= Tc
    if Ts < Tc:
        penalty += 1e6 * (Tc - Ts) ** 2
    # 约束: vs > 0.01
    if vs <= 0.01:
        penalty += 1e6 * (0.01 - vs) ** 2
    # 约束: Tc >= 0
    if Tc < 0:
        penalty += 1e6 * abs(Tc)
    # 约束: Tv >= 0
    if Tv < 0:
        penalty += 1e6 * abs(Tv)

    return mse + penalty


def compute_param_error(est: np.ndarray, true: np.ndarray) -> dict:
    """计算参数估计误差"""
    abs_err = np.abs(est - true)
    rel_err = np.abs((est - true) / (true + 1e-8)) * 100
    return {"abs": abs_err, "rel": rel_err}


def main():
    print("=" * 70)
    print("    PSO 辨识 Stribeck 摩擦参数")
    print("=" * 70)

    # 1. 加载数据
    if not os.path.exists(DATA_FILE):
        print(f"\n错误: 找不到数据文件 {DATA_FILE}")
        print("请先运行 derive_friction_torque.py 生成二维数据。")
        return

    v_data, t_target = load_data(DATA_FILE)
    print(f"\n数据样本数 : {len(v_data)}")
    print(f"速度范围   : [{v_data.min():.4f}, {v_data.max():.4f}]")
    print(f"转矩范围   : [{t_target.min():.4f}, {t_target.max():.4f}]")

    # 2. 初始化粒子群（调用 pso_init）
    swarm = pso_init.init_swarm(N, DIM, LB, UB, seed=SEED)
    pso_init.print_init_state(swarm)

    print("\n" + "=" * 70)
    print("                     开始 PSO 迭代")
    print("=" * 70)

    # 表头
    print(f"\n{'Iter':>5} | {'Tc':>8} {'Ts':>8} {'vs':>8} {'Tv':>8} | {'MSE':>10} | {'Status'}")
    print("-" * 70)

    # 3. PSO 迭代
    for iteration in range(MAX_ITER):
        # 评估所有粒子
        for i in range(swarm["N"]):
            fit = fitness(swarm["positions"][i], v_data, t_target)

            # 更新个体最优
            if fit < swarm["pBest_fitness"][i]:
                swarm["pBest_fitness"][i] = fit
                swarm["pBest"][i] = swarm["positions"][i].copy()

        # 更新全局最优
        best_idx = np.argmin(swarm["pBest_fitness"])
        if swarm["pBest_fitness"][best_idx] < swarm["gBest_fitness"]:
            swarm["gBest_fitness"] = swarm["pBest_fitness"][best_idx]
            swarm["gBest"] = swarm["pBest"][best_idx].copy()

        # 打印当前辨识情况
        g = swarm["gBest"]
        mse_val = swarm["gBest_fitness"]
        status = "↓" if iteration > 0 else "init"
        print(f"{iteration+1:>5} | {g[0]:>8.4f} {g[1]:>8.4f} {g[2]:>8.4f} {g[3]:>8.4f} | {mse_val:>10.6f} | {status}")

        # 每 20 轮打印一次详细统计
        if (iteration + 1) % 20 == 0:
            pos_min = swarm["positions"].min(axis=0)
            pos_max = swarm["positions"].max(axis=0)
            print(f"       种群范围: Tc=[{pos_min[0]:.3f},{pos_max[0]:.3f}] "
                  f"Ts=[{pos_min[1]:.3f},{pos_max[1]:.3f}] "
                  f"vs=[{pos_min[2]:.3f},{pos_max[2]:.3f}] "
                  f"Tv=[{pos_min[3]:.3f},{pos_max[3]:.3f}]")

        # 更新速度与位置
        r1 = np.random.rand(N, DIM)
        r2 = np.random.rand(N, DIM)

        swarm["velocities"] = (
            W * swarm["velocities"]
            + c1 * r1 * (swarm["pBest"] - swarm["positions"])
            + c2 * r2 * (swarm["gBest"] - swarm["positions"])
        )
        swarm["velocities"] = np.clip(
            swarm["velocities"], -swarm["v_max"], swarm["v_max"]
        )

        swarm["positions"] += swarm["velocities"]
        swarm["positions"] = np.clip(
            swarm["positions"], swarm["LB"], swarm["UB"]
        )

    # 4. 最终结果
    print("\n" + "=" * 70)
    print("                     辨识结果")
    print("=" * 70)

    est = swarm["gBest"]
    err = compute_param_error(est, TRUE_PARAMS)

    print(f"\n{'参数':>6} | {'真实值':>10} | {'辨识值':>10} | {'绝对误差':>10} | {'相对误差':>10}")
    print("-" * 60)
    param_names = ["Tc", "Ts", "vs", "Tv"]
    for i, name in enumerate(param_names):
        print(f"{name:>6} | {TRUE_PARAMS[i]:>10.4f} | {est[i]:>10.4f} | "
              f"{err['abs'][i]:>10.4f} | {err['rel'][i]:>10.2f}%")

    # 最终预测效果
    T_pred_final = stribeck_predict(v_data, est)
    final_mse = np.mean((T_pred_final - t_target) ** 2)
    final_rmse = np.sqrt(final_mse)
    r2 = 1 - np.sum((t_target - T_pred_final) ** 2) / np.sum((t_target - np.mean(t_target)) ** 2)

    print(f"\n预测指标:")
    print(f"  MSE  = {final_mse:.6f}")
    print(f"  RMSE = {final_rmse:.6f}")
    print(f"  R²   = {r2:.6f}")

    # 保存辨识结果
    RESULT_FILE = "stribeck_identified_params.json"
    import json
    result = {
        "true_params": TRUE_PARAMS.tolist(),
        "identified_params": est.tolist(),
        "param_names": param_names,
        "abs_error": err["abs"].tolist(),
        "rel_error_percent": err["rel"].tolist(),
        "mse": float(final_mse),
        "rmse": float(final_rmse),
        "r2": float(r2),
        "pso_config": {
            "N": N, "MAX_ITER": MAX_ITER, "W": W, "c1": c1, "c2": c2,
            "LB": LB.tolist(), "UB": UB.tolist(),
        },
    }
    with open(RESULT_FILE, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n✅ 辨识结果已保存: {os.path.abspath(RESULT_FILE)}")


if __name__ == "__main__":
    main()
