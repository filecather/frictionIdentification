"""
根据三维样本 (time, position, current) 反推目标摩擦转矩

物理模型:
    电磁力矩  Te = K_t * I
    运动方程  Te = T_friction + J * α
    → 目标摩擦转矩  T_friction = K_t * I - J * α

输出: 速度-摩擦转矩 二维数据表格 (velocity, target_friction_torque)
"""

import numpy as np
import csv
import os

# ==================== 物理参数 ====================
J = 3.9e-4              # 转动惯量 (kg·m²)
K_t = 2.0               # 转矩-电流系数 (N·m / A)，与生成时保持一致
DT = 0.05               # 采样间隔 (s)，需与样本一致

INPUT_FILE = "stribeck_samples.csv"     # 输入三维样本
OUTPUT_FILE = "velocity_friction_2d.csv" # 输出二维表格
# =================================================


def load_csv(filepath: str) -> dict:
    """读取 CSV 返回列字典"""
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        headers = next(reader)
        cols = {h: [] for h in headers}
        for row in reader:
            for h, v in zip(headers, row):
                cols[h].append(float(v))
    return {k: np.array(v) for k, v in cols.items()}


def main():
    print("=" * 60)
    print("     反推目标摩擦转矩 —— 速度 vs 摩擦力矩 二维表格")
    print("=" * 60)

    # 1. 加载三维样本
    if not os.path.exists(INPUT_FILE):
        print(f"\n错误: 找不到输入文件 {INPUT_FILE}")
        print("请先运行 stribeck_current_generator.py 生成样本。")
        return

    data = load_csv(INPUT_FILE)
    time_arr = data["time"]
    position = data["position"]
    current = data["current"]
    n = len(time_arr)

    # 2. 计算时间间隔（若样本自带，优先用实际间隔）
    dt = float(np.mean(np.diff(time_arr))) if n > 1 else DT
    print(f"\n样本数量    : {n}")
    print(f"采样间隔 dt : {dt:.4f} s")
    print(f"转动惯量 J  : {J:.2e} kg·m²")
    print(f"转矩系数 K_t: {K_t:.2f} N·m/A")

    # 3. 加载原始速度，直接数值微分求加速度（避免 position→velocity 反推的符号误差）
    if "velocity" in data:
        velocity = data["velocity"]
    else:
        velocity = np.gradient(position, dt)  # 兼容旧数据
    acceleration = np.gradient(velocity, dt)

    # 4. 电磁力矩 = current / K_t
    #    （生成脚本中 current = K_t * torque，故 torque = current / K_t）
    T_elec = current / K_t

    # 5. 目标摩擦转矩 = 电磁力矩 - J * α
    T_friction_target = T_elec - J * acceleration

    # 6. 保存二维表格 (velocity, target_friction_torque)
    rows = []
    for i in range(n):
        rows.append([
            round(float(velocity[i]), 6),
            round(float(T_friction_target[i]), 6),
        ])

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["velocity", "target_friction_torque"])
        writer.writerows(rows)

    print(f"\n✅ 二维表格已保存: {os.path.abspath(OUTPUT_FILE)}")
    print(f"   文件大小      : {os.path.getsize(OUTPUT_FILE) / 1024:.1f} KB")

    # 7. 统计摘要
    print("\n【速度-摩擦转矩 统计摘要】")
    print(f"{'指标':>8} | {'velocity':>12} {'friction_torque':>16}")
    print("-" * 42)
    stats = [("count", len), ("mean", np.mean), ("std", np.std),
             ("min", np.min), ("max", np.max)]
    for label, fn in stats:
        v_val = fn(velocity)
        t_val = fn(T_friction_target)
        if label == "count":
            print(f"{label:>8} | {int(v_val):>12} {int(t_val):>16}")
        else:
            print(f"{label:>8} | {v_val:>12.4f} {t_val:>16.4f}")

    # 8. 前 10 行示例
    print("\n【前 10 条数据】")
    print(f"{'velocity':>12} | {'target_friction_torque':>22}")
    print("-" * 40)
    for r in rows[:10]:
        print(f"{r[0]:>12.6f} | {r[1]:>22.6f}")

    # 9. 附加：保存含中间量的完整分析表
    FULL_FILE = "full_dynamics_analysis.csv"
    with open(FULL_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "position", "velocity", "acceleration",
                         "current", "T_elec", "T_friction_target"])
        for i in range(n):
            writer.writerow([
                round(float(time_arr[i]), 4),
                round(float(position[i]), 6),
                round(float(velocity[i]), 6),
                round(float(acceleration[i]), 6),
                round(float(current[i]), 6),
                round(float(T_elec[i]), 6),
                round(float(T_friction_target[i]), 6),
            ])
    print(f"\n✅ 完整动力学分析表: {os.path.abspath(FULL_FILE)}")
    print("   (含 time, position, velocity, acceleration, current, Te, Tf)")


if __name__ == "__main__":
    main()
