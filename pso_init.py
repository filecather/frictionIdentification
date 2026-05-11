"""
粒子群优化（PSO）初始化模块
支持独立运行，也支持被 import 用于其他 PSO 任务
"""

import numpy as np


def init_swarm(N: int, DIM: int, LB: np.ndarray, UB: np.ndarray, seed: int = 42) -> dict:
    """
    初始化粒子群

    参数:
        N: 粒子数量
        DIM: 搜索维度
        LB: 各维度下限 (shape: (DIM,))
        UB: 各维度上限 (shape: (DIM,))
        seed: 随机种子

    返回:
        dict: 包含粒子群状态的字典
    """
    np.random.seed(seed)

    v_max = 0.2 * (UB - LB)

    positions = np.random.uniform(LB, UB, size=(N, DIM))
    velocities = np.random.uniform(-v_max, v_max, size=(N, DIM))

    pBest = positions.copy()
    pBest_fitness = np.full(N, np.inf)

    gBest = positions[0].copy()
    gBest_fitness = np.inf

    return {
        "N": N,
        "DIM": DIM,
        "LB": LB,
        "UB": UB,
        "v_max": v_max,
        "positions": positions,
        "velocities": velocities,
        "pBest": pBest,
        "pBest_fitness": pBest_fitness,
        "gBest": gBest,
        "gBest_fitness": gBest_fitness,
    }


def print_init_state(swarm: dict):
    """打印初始化状态（调试用）"""
    print("=" * 60)
    print(f"  PSO 粒子初始化完成  |  N = {swarm['N']}  |  DIM = {swarm['DIM']}")
    print("=" * 60)
    print(f"\n  positions  shape: {swarm['positions'].shape}")
    print(f"  velocities shape: {swarm['velocities'].shape}")
    print(f"  pBest      shape: {swarm['pBest'].shape}")

    print("\n--- 搜索边界 ---")
    for d in range(swarm['DIM']):
        print(f"  维度 {d+1}: [{swarm['LB'][d]:.4f}, {swarm['UB'][d]:.4f}]")

    print(f"\n--- 速度上限 V_max (20% 范围) ---")
    print(f"  {swarm['v_max']}")

    print("\n--- 前 5 个粒子位置示例 ---")
    for i in range(min(5, swarm['N'])):
        print(f"  粒子 {i+1:3d}: {swarm['positions'][i]}")


# ==================== 独立运行入口 ====================
if __name__ == "__main__":
    # 默认参数（Stribeck 辨识场景）
    N = 100
    DIM = 4
    LB = np.array([0.0, 0.0, 0.01, 0.0])   # [Tc, Ts, vs, Tv] 下限
    UB = np.array([2.0, 3.0, 2.0, 1.0])    # [Tc, Ts, vs, Tv] 上限

    swarm = init_swarm(N, DIM, LB, UB, seed=42)
    print_init_state(swarm)
