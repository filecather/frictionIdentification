"""
固定 Tc 和 Tv，用网格搜索 + 梯度下降精确拟合 Ts 和 vs
纯 numpy 实现，无需 scipy
"""
import numpy as np
import csv

# 加载数据
with open("velocity_friction_2d.csv", "r", encoding="utf-8") as f:
    reader = csv.reader(f)
    next(reader)
    v_data, t_target = [], []
    for row in reader:
        v_data.append(float(row[0]))
        t_target.append(float(row[1]))
v_all = np.array(v_data)
t_all = np.array(t_target)

# 只使用过渡区数据拟合 Ts/vs (|v| < 0.8)，排除高速区无信息数据
mask = np.abs(v_all) < 0.8
v = v_all[mask]
t = t_all[mask]
print(f"总样本: {len(v_all)}，过渡区样本 (|v|<0.8): {len(v)}")

# 固定已辨识准确的 Tc 和 Tv
Tc_fixed = 0.4801
Tv_fixed = 0.1712
TRUE_PARAMS = [0.50, 0.80, 0.30, 0.15]


def predict(v, Ts, vs):
    sign_approx = np.tanh(v / 1e-3)
    stribeck_term = (Ts - Tc_fixed) * np.exp(-(v / vs) ** 2)
    return (Tc_fixed + stribeck_term) * sign_approx + Tv_fixed * v


def mse(v, t, Ts, vs):
    return np.mean((predict(v, Ts, vs) - t) ** 2)


# ========== 第 1 阶段：粗网格搜索 ==========
print("阶段 1: 粗网格搜索...")
Ts_grid = np.linspace(0.3, 1.2, 200)
vs_grid = np.linspace(0.05, 0.8, 200)

best_mse = np.inf
best_Ts, best_vs = 0.7, 0.4

for Ts in Ts_grid:
    # 向量化计算 vs 维度
    for vs in vs_grid:
        err = mse(v, t, Ts, vs)
        if err < best_mse:
            best_mse = err
            best_Ts, best_vs = Ts, vs

print(f"  粗网格最优: Ts={best_Ts:.4f}, vs={best_vs:.4f}, MSE={best_mse:.6f}")

# ========== 第 2 阶段：细网格搜索 ==========
print("\n阶段 2: 细网格搜索...")
Ts_range = 0.15
vs_range = 0.15
Ts_grid = np.linspace(max(0.1, best_Ts - Ts_range), best_Ts + Ts_range, 300)
vs_grid = np.linspace(max(0.02, best_vs - vs_range), best_vs + vs_range, 300)

for Ts in Ts_grid:
    for vs in vs_grid:
        err = mse(v, t, Ts, vs)
        if err < best_mse:
            best_mse = err
            best_Ts, best_vs = Ts, vs

print(f"  细网格最优: Ts={best_Ts:.4f}, vs={best_vs:.4f}, MSE={best_mse:.6f}")

# ========== 第 3 阶段：梯度下降精调 ==========
print("\n阶段 3: 梯度下降精调...")


def grad_mse(v, t, Ts, vs, h=1e-5):
    """数值梯度"""
    f = mse(v, t, Ts, vs)
    dTs = (mse(v, t, Ts + h, vs) - f) / h
    dvs = (mse(v, t, Ts, vs + h) - f) / h
    return np.array([dTs, dvs])


Ts, vs = best_Ts, best_vs
lr = 0.001
for i in range(5000):
    g = grad_mse(v, t, Ts, vs)
    Ts -= lr * g[0]
    vs -= lr * g[1]
    # 边界保护
    Ts = max(0.1, Ts)
    vs = max(0.02, vs)
    if i % 1000 == 0:
        err = mse(v, t, Ts, vs)
        print(f"  iter {i:>5}: Ts={Ts:.5f}, vs={vs:.5f}, MSE={err:.8f}")

final_mse = mse(v, t, Ts, vs)
print(f"\n  梯度下降最终: Ts={Ts:.5f}, vs={vs:.5f}, MSE={final_mse:.8f}")

# ========== 结果汇总 ==========
final_params = [Tc_fixed, Ts, vs, Tv_fixed]

print(f"\n{'='*60}")
print("  最终辨识参数汇总")
print(f"{'='*60}")
print(f"{'参数':>6} | {'真实值':>10} | {'辨识值':>10} | {'绝对误差':>10} | {'相对误差':>10}")
print("-" * 60)
for i, name in enumerate(["Tc", "Ts", "vs", "Tv"]):
    abs_err = abs(final_params[i] - TRUE_PARAMS[i])
    rel_err = abs(final_params[i] - TRUE_PARAMS[i]) / TRUE_PARAMS[i] * 100
    print(f"{name:>6} | {TRUE_PARAMS[i]:>10.4f} | {final_params[i]:>10.4f} | {abs_err:>10.4f} | {rel_err:>10.2f}%")

# 预测质量
t_pred = predict(v, Ts, vs)
r2 = 1 - np.sum((t - t_pred) ** 2) / np.sum((t - np.mean(t)) ** 2)
print(f"\n  MSE = {final_mse:.6f}")
print(f"  RMSE = {np.sqrt(final_mse):.6f}")
print(f"  R²   = {r2:.6f}")
