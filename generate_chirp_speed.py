"""
生成连续扫频速度样本：多频正弦叠加，覆盖 [-0.2, 3.0] 无间隙
确保 v ≈ vs=0.3 的过渡区有充足数据

关键设计：
  1. 多频正弦叠加（频率不可公约）产生准周期速度覆盖
  2. 同时保存原始 velocity，避免后续从 position 数值微分反推时的过零点符号翻转
  3. 零电流偏置 (i_bias=0)，消除反推时的系统性偏移
"""

import numpy as np
import csv

np.random.seed(42)

# ==================== 物理参数 ====================
count, dt = 2000, 0.05          # 样本数，采样间隔 (s)
t = np.arange(count) * dt       # 时间轴
TRUE_PARAMS = np.array([0.5, 0.8, 0.3, 0.15])  # 真实 Stribeck 参数 [Tc, Ts, vs, Tv]
# =================================================


def stribeck_continuous(v, params):
    """
    连续近似 Stribeck 摩擦模型
    使用 tanh(v/eps) 代替 sign(v)，避免数值优化时的不连续问题
    """
    Tc, Ts, vs, Tv = params
    eps = 1e-3  # 过渡宽度，越小越接近理想 sign(v)
    sign_approx = np.tanh(v / eps)
    stribeck_term = (Ts - Tc) * np.exp(-(v / vs) ** 2)
    return (Tc + stribeck_term) * sign_approx + Tv * v


def generate_current(torque, cfg):
    """根据转矩生成电流 = K_t * torque + bias + noise"""
    K_t = cfg.get("K_t", 2.0)
    i_bias = cfg.get("i_bias", 0.0)
    snr_db = cfg.get("snr_db", None)

    I_base = K_t * torque + i_bias

    if snr_db is not None and np.mean(I_base ** 2) > 0:
        signal_power = np.mean(I_base ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        i_noise_std = np.sqrt(noise_power)
    else:
        i_noise_std = cfg.get("i_noise_std", 0.08)

    I_noise = np.random.normal(0, i_noise_std, len(torque))
    I = I_base + I_noise
    return I, I_base, I_noise


def save_csv(data_dict, filepath, columns):
    """保存字典数据到 CSV"""
    rows = zip(*[data_dict[c] for c in columns])
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        writer.writerows(rows)


# ========== 1. 生成多频正弦速度曲线 ==========
# 频率不可公约 → 准周期运动 → 速度分布更均匀
velocity = (
    1.2                                    # 直流偏置
    + 1.5 * np.sin(0.02 * np.pi * t)       # 基频，周期 100s
    + 0.35 * np.sin(0.06 * np.pi * t + 0.5)  # 3倍频，相位偏移
    + 0.12 * np.sin(0.14 * np.pi * t + 1.2)  # 7倍频，小振幅
    + np.random.normal(0, 0.015, count)       # 微量噪声
)
velocity = np.clip(velocity, -0.5, 3.5)

# ========== 2. 梯形积分得位置 ==========
position = np.zeros_like(velocity)
for i in range(1, len(velocity)):
    position[i] = position[i - 1] + 0.5 * (velocity[i - 1] + velocity[i]) * dt

# ========== 3. 计算真实摩擦转矩 ==========
torque = stribeck_continuous(velocity, TRUE_PARAMS)

# ========== 4. 生成电流（零偏置，高 SNR） ==========
current_cfg = {"K_t": 2.0, "i_bias": 0.00, "snr_db": 50}
current, _, _ = generate_current(torque, current_cfg)

# ========== 5. 保存样本（关键：同时保存原始 velocity） ==========
# 不从 position 反推 velocity，避免 np.gradient 在过零点引入符号翻转
data = {
    "time": np.round(t, 4),
    "position": np.round(position, 6),
    "velocity": np.round(velocity, 6),
    "current": np.round(current, 6),
}
save_csv(data, "stribeck_samples.csv", ["time", "position", "velocity", "current"])

# ========== 6. 输出统计 ==========
print(f"速度范围: [{velocity.min():.4f}, {velocity.max():.4f}]")
print(f"转矩范围: [{torque.min():.4f}, {torque.max():.4f}]")

bins = [-0.5, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.8, 1.5, 2.0, 3.5]
counts, _ = np.histogram(velocity, bins=bins)
print("\n各速度区间数据点数:")
for i in range(len(bins) - 1):
    print(f"  [{bins[i]:>4.1f}, {bins[i+1]:>4.1f}]: {counts[i]:>4d} 点")
print("\nSaved: stribeck_samples.csv")
