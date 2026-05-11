"""
Stribeck 参数辨识 —— 一键运行管道

完整工作流：
  1. 生成多频正弦速度样本（覆盖近零速~高速区）
  2. 根据运动方程反推目标摩擦转矩
  3. 两步分层辨识：高速区线性回归 → 对数线性回归 → PSO 精调
  4. 固定 Tc/Tv，网格搜索精调 Ts/vs

使用方式:
    python run_pipeline.py
"""

import subprocess
import sys


def run_script(name):
    """运行子脚本并实时打印输出"""
    print(f"\n{'='*70}")
    print(f"  >>> 执行: {name}")
    print(f"{'='*70}")
    result = subprocess.run([sys.executable, name], capture_output=False, text=True)
    if result.returncode != 0:
        print(f"[ERROR] {name} 执行失败，退出码: {result.returncode}")
        sys.exit(1)
    return result


def main():
    print("=" * 70)
    print("  Stribeck 摩擦参数辨识 —— 完整工作流")
    print("=" * 70)

    # ---------- 步骤 1: 生成样本 ----------
    # 多频正弦叠加产生 [-0.3, 2.7] 连续速度覆盖
    # 关键：直接保存原始 velocity，避免后续数值微分反推时的过零点符号翻转
    run_script("generate_chirp_speed.py")

    # ---------- 步骤 2: 反推目标摩擦转矩 ----------
    # 运动方程: T_elec = T_friction + J * α
    # → T_friction = current/K_t - J * α
    # 关键：直接加载原始 velocity 求加速度，不用 position 反推
    run_script("derive_friction_torque.py")

    # ---------- 步骤 3: 两步分层辨识 ----------
    # 第 1 步：高速区(|v|>0.6)线性回归 → Tc, Tv
    # 第 2 步：对数线性回归 → Ts, vs（初值）
    # 第 3 步：PSO 精调，以解析解为种子
    run_script("hierarchical_identify.py")

    # ---------- 步骤 4: 网格搜索精调 Ts/vs ----------
    # 固定已精确辨识的 Tc/Tv，在过渡区(|v|<0.8)做网格搜索+梯度下降
    # 消除 Ts/vs 强耦合导致的参数漂移
    run_script("refine_vs_ts.py")

    print("\n" + "=" * 70)
    print("  ✅ 全部步骤执行完毕")
    print("=" * 70)
    print("\n  输出文件:")
    print("    stribeck_samples.csv          — 原始三维样本")
    print("    velocity_friction_2d.csv      — 速度-摩擦转矩二维数据")
    print("    full_dynamics_analysis.csv    — 完整动力学中间量")
    print("    hierarchical_result.json      — 分层辨识结果")
    print("=" * 70)


if __name__ == "__main__":
    main()
