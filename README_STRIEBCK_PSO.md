# Stribeck 摩擦参数 PSO 辨识 —— 完整工作流

## 1. 项目概述

本项目实现了一套完整的 **Stribeck 摩擦参数辨识系统**，基于粒子群优化（PSO）与分层解析算法，从电机驱动的时间-位置-电流三维样本中，自动辨识出 Coulomb 摩擦 $T_c$、静摩擦 $T_s$、Stribeck 速度 $v_s$ 和粘滞阻尼系数 $T_v$ 四个关键参数。

**核心流程：**

```
生成速度-位置-电流样本 → 运动方程反推摩擦转矩 → 两步分层辨识 → 网格精调
```

**最终辨识精度（典型值）：**

| 参数 | 真实值 | 辨识值 | 误差 |
|------|--------|--------|------|
| $T_c$ | 0.5000 | **0.5008** | **0.16%** |
| $T_s$ | 0.8000 | **0.7981** | **0.23%** |
| $v_s$ | 0.3000 | **0.3092** | **3.06%** |
| $T_v$ | 0.1500 | **0.1497** | **0.22%** |

---

## 2. 物理模型

### 2.1 Stribeck 摩擦模型

摩擦转矩与速度的非线性关系：

$$T_{friction}(v) = \underbrace{\left[T_c + (T_s - T_c) \cdot e^{-(v/v_s)^2}\right] \cdot \text{sign}(v)}_{\text{非线性摩擦}} + \underbrace{T_v \cdot v}_{\text{粘滞阻尼}}$$

| 参数 | 物理意义 | 单位 |
|------|---------|------|
| $T_c$ | Coulomb 摩擦转矩（动摩擦） | N·m |
| $T_s$ | 最大静摩擦转矩 | N·m |
| $v_s$ | Stribeck 特征速度 | rad/s |
| $T_v$ | 粘滞阻尼系数 | N·m·s/rad |

**数值优化中的连续近似：**

理想 `sign(v)` 在 $v=0$ 处不连续，导致梯度优化困难。采用 `tanh(v/ε)` 近似：

```python
eps = 1e-3                          # 过渡宽度
sign_approx = np.tanh(v / eps)      # 连续可导的 sign 近似
```

### 2.2 电机运动方程

电磁力矩、摩擦转矩与惯性力矩的平衡：

$$T_{elec} = T_{friction} + J \cdot \alpha$$

其中：
- $T_{elec} = K_t \cdot I$：电流产生的电磁力矩
- $J = 3.9 \times 10^{-4}$ kg·m²：转动惯量
- $K_t = 2.0$ N·m/A：转矩-电流系数
- $\alpha = \dot{v}$：角加速度

**反推公式：**

$$T_{friction}^{target} = \frac{I}{K_t} - J \cdot \alpha$$

---

## 3. 完整工作流

### 3.1 数据生成 (`generate_chirp_speed.py`)

#### 目标
生成覆盖近零速区到高速区的连续速度样本，确保 Stribeck 过渡段有充足数据。

#### 速度曲线设计
采用**多频正弦叠加**（频率不可公约），产生准周期覆盖：

```python
velocity = (
    1.2                                    # 直流偏置
    + 1.5 * np.sin(0.02 * np.pi * t)       # 基频，周期 100s
    + 0.35 * np.sin(0.06 * np.pi * t + 0.5)  # 3倍频
    + 0.12 * np.sin(0.14 * np.pi * t + 1.2)  # 7倍频
    + np.random.normal(0, 0.015, count)       # 微量噪声
)
```

**速度分布特点：**
- 范围：$[-0.33, 2.74]$ rad/s
- 过零区（$|v| < 0.2$）：约 663 点
- 高速区（$v > 2.0$）：约 782 点

#### 关键设计决策

**① 直接保存原始 `velocity`**

```python
data = {
    "time": ..., "position": ..., 
    "velocity": velocity,        # ← 关键：保存原始速度
    "current": ...
}
```

**为什么？** 若只保存 `position`，后续用 `np.gradient(position)` 反推速度时，会在过零点附近产生**符号翻转**（详见第 5 节）。

**② 零电流偏置 `i_bias = 0.0`**

```python
current_cfg = {"K_t": 2.0, "i_bias": 0.00, "snr_db": 50}
```

若保留偏置，反推时会产生 $T_{bias} = i_{bias} / K_t$ 的系统性偏移，污染辨识结果。

**③ 高信噪比 `snr_db = 50`**

电流噪声经 $K_t$ 缩放后，对应摩擦转矩噪声 std ≈ 0.0016 N·m，远小于 Stribeck 过渡段的特征变化（~0.08 N·m）。

---

### 3.2 摩擦转矩反推 (`derive_friction_torque.py`)

#### 输入
`stribeck_samples.csv`（time, position, **velocity**, current）

#### 核心代码

```python
# 1. 直接加载原始 velocity（避免从 position 反推）
if "velocity" in data:
    velocity = data["velocity"]
else:
    velocity = np.gradient(position, dt)   # 兼容旧数据（不推荐）

# 2. 数值微分求加速度
acceleration = np.gradient(velocity, dt)

# 3. 电磁力矩 = current / K_t
T_elec = current / K_t

# 4. 目标摩擦转矩 = 电磁力矩 - J * α
T_friction_target = T_elec - J * acceleration
```

#### 输出
- `velocity_friction_2d.csv`：二维辨识数据 (velocity, target_friction_torque)
- `full_dynamics_analysis.csv`：完整中间量（time, position, velocity, acceleration, current, Te, Tf）

---

### 3.3 PSO 初始化 (`pso_init.py`)

模块化设计，支持独立运行或被 import：

```python
import pso_init

# 初始化粒子群
swarm = pso_init.init_swarm(
    N=100, DIM=4,
    LB=np.array([0.0, 0.0, 0.01, 0.0]),   # [Tc, Ts, vs, Tv] 下限
    UB=np.array([2.0, 1.5, 2.0, 1.0]),    # 上限
    seed=42
)

# 返回字典包含：positions, velocities, pBest, gBest, v_max, LB, UB
```

**搜索边界设置原则：**
- $T_c \in [0, 2.0]$：Coulomb 摩擦非负
- $T_s \in [0, 1.5]$：静摩擦上限（避免与 $T_c$ 耦合导致不可辨识）
- $v_s \in [0.01, 2.0]$：Stribeck 速度必须为正
- $T_v \in [0, 1.0]$：粘滞系数非负

---

### 3.4 两步分层辨识 (`hierarchical_identify.py`)

#### 核心思想
Stribeck 模型的 4 个参数在有限速度范围内**强耦合**，同时优化会导致参数漂移。采用**分层解耦策略**：

```
高速区线性回归 ──→ Tc, Tv（精确）
        ↓
对数线性回归 ────→ Ts, vs（初值）
        ↓
PSO 精调 ────────→ 四参数联合优化
```

#### 第 1 步：高速区线性回归 → $T_c$, $T_v$

**物理依据：** 当 $|v| \gg v_s$ 时，$e^{-(v/v_s)^2} \to 0$，模型退化为直线：

$$T \approx T_c \cdot \text{sign}(v) + T_v \cdot v$$

**代码实现：**

```python
# 选取 |v| > threshold 的高速数据点
mask = np.abs(v) > 0.6

# 设计矩阵: [sign(v), v]
X = np.column_stack([np.sign(v_high), v_high])
params, _, _, _ = np.linalg.lstsq(X, t_high, rcond=None)
Tc, Tv = params
```

**阈值选择：** $threshold = 0.6 \approx 2 \times v_s^{true}$，确保 Stribeck 项衰减至 < 2%。

**典型结果：** $T_c = 0.5008$ (误差 0.16%), $T_v = 0.1497$ (误差 0.22%)

#### 第 2 步：对数线性回归 → $T_s$, $v_s$

**数学推导：**

残差中的 Stribeck 项：

$$y = |T_{target} - T_c \cdot \text{sign}(v) - T_v \cdot v| = (T_s - T_c) \cdot e^{-v^2/v_s^2}$$

取对数：

$$\ln(y) = \underbrace{\ln(T_s - T_c)}_{截距} - \underbrace{\frac{1}{v_s^2}}_{斜率} \cdot v^2$$

对 $v^2$ 做线性回归即可解析求解：

```python
# 计算残差
y_pos = t[v>0] - Tc - Tv * v[v>0]
y_neg = -(t[v<0] + Tc - Tv * v[v<0])

# 合并有效数据 (y > 0)
y_valid = np.concatenate([y_pos[y_pos>1e-6], y_neg[y_neg>1e-6]])
v_sq = np.concatenate([v[v>0][y_pos>1e-6]**2, v[v<0][y_neg>1e-6]**2])

# 对数线性回归
log_y = np.log(y_valid)
X = np.column_stack([np.ones(len(log_y)), v_sq])
params, _, _, _ = np.linalg.lstsq(X, log_y, rcond=None)
intercept, slope = params

vs = np.sqrt(-1.0 / slope)          # 从斜率解析求解
Ts = Tc + np.exp(intercept)         # 从截距解析求解
```

**局限性：** 对数回归受噪声影响大，R² 通常只有 0.6~0.7，只能提供**初值**。

#### 第 3 步：PSO 精调

以解析解为种子，在小范围内（±30%）做 PSO 精调。但由于 Ts/vs 耦合，PSO 精调往往会把原本精确的 Tc/Tv 也拉偏。

---

### 3.5 网格搜索精调 (`refine_vs_ts.py`)

**关键洞察：** Tc 和 Tv 已由高速区线性回归精确确定，不应再被 PSO 扰动。

**策略：**
1. 固定 $T_c = 0.5008$, $T_v = 0.1497$（来自第 1 步解析解）
2. 只优化 $T_s$ 和 $v_s$ 两个参数
3. 在过渡区（$|v| < 0.8$）进行，排除高速区无信息数据

**算法：**

```python
# 1. 粗网格搜索（200×200）
for Ts in np.linspace(0.3, 1.2, 200):
    for vs in np.linspace(0.05, 0.8, 200):
        err = mse(v, t, Ts, vs)
        # 记录最优

# 2. 细网格搜索（以粗网格最优为中心，±15%）
...

# 3. 梯度下降精调（5000 迭代）
for i in range(5000):
    g = numerical_gradient(v, t, Ts, vs)
    Ts -= lr * g[0]
    vs -= lr * g[1]
```

**结果：** $T_s = 0.7981$ (误差 0.23%), $v_s = 0.3092$ (误差 3.06%)

---

## 4. 关键陷阱与解决方案

### 陷阱 1：$K_t$ 双重应用

**现象：** 辨识结果整体偏大 4 倍，$T_c$ 压到搜索边界。

**原因：** 生成脚本中 `current = K_t * torque`，反推时错误地写为 `T_elec = K_t * current`，导致 $T_{elec} = K_t^2 \cdot torque$。

**修复：**

```python
# 正确：电流→转矩是除法
T_elec = current / K_t

# 错误（已修复）
# T_elec = K_t * current   ← 会导致 K_t² 放大
```

### 陷阱 2：过零点速度符号翻转 ★★★

**现象：** 全数据 MSE 从 6×10⁻⁶ 恶化到 0.019，vs 辨识误差 > 80%。

**根因：** `np.gradient(position)` 对梯形积分的位置做中心差分，在速度过零点附近产生平滑相位偏移，导致反推的 velocity **符号翻转**。

**案例：**
- 真实速度：`v = -0.008`（负）
- 反推速度：`v = +0.007`（正）
- Stribeck 输出差异：`T(-0.008) ≈ -0.8` vs `T(+0.007) ≈ +0.8`，误差 **1.6**

**修复：** 生成样本时直接保存原始 velocity，反推时直接加载：

```python
# generate_chirp_speed.py
data = {
    "time": ..., "position": ...,
    "velocity": velocity,      # ← 关键：保存原始速度
    "current": ...
}

# derive_friction_torque.py
if "velocity" in data:
    velocity = data["velocity"]     # ← 直接加载
else:
    velocity = np.gradient(position, dt)   # 旧数据兼容（不推荐）
```

### 陷阱 3：$T_s$ / $v_s$ 参数耦合

**现象：** 单独优化时，$T_s$ 降低 ↔ $v_s$ 增大，多种组合产生相似的 MSE。

**物理解释：** 在 $v \gg v_s$ 区域，$e^{-(v/v_s)^2} \approx 0$，模型退化为 $T \approx T_c + T_v \cdot v$，$T_s$ 和 $v_s$ 对该区域无贡献。

**对策：**
1. **扩展速度范围**：同时包含 $|v| \ll v_s$（近零速）和 $|v| \gg v_s$（高速）
2. **固定 Tc/Tv 后单独优化 Ts/vs**：消除耦合自由度
3. **使用非线性最小二乘**（如 grid search + gradient descent）替代线性近似

### 陷阱 4：电流偏置 $i_{bias}$

**现象：** 辨识结果整体偏移，残差均值非零。

**原因：** 生成时 `current = K_t * torque + i_bias`，反推时未扣除偏置，导致 $T_{target} = torque + i_{bias}/K_t$。

**修复：** 生成时设 `i_bias = 0.0`。

---

## 5. 文件清单

| 文件 | 功能 |
|------|------|
| `run_pipeline.py` | **一键运行脚本**，顺序执行完整工作流 |
| `generate_chirp_speed.py` | 生成多频正弦速度样本（含原始 velocity） |
| `derive_friction_torque.py` | 根据运动方程反推目标摩擦转矩 |
| `pso_init.py` | PSO 粒子初始化模块（可独立运行或 import） |
| `hierarchical_identify.py` | 两步分层辨识（解析解 + PSO 精调） |
| `refine_vs_ts.py` | 固定 Tc/Tv，网格搜索精调 Ts/vs |
| `README_STRIEBCK_PSO.md` | 本说明文档 |

**输出文件：**

| 文件 | 内容 |
|------|------|
| `stribeck_samples.csv` | 原始三维样本 (time, position, velocity, current) |
| `velocity_friction_2d.csv` | 二维辨识数据 (velocity, target_friction_torque) |
| `full_dynamics_analysis.csv` | 完整中间量（Te, α, Tf 等） |
| `hierarchical_result.json` | 辨识结果（JSON 格式） |

---

## 6. 使用说明

### 方式一：一键运行（推荐）

```bash
python run_pipeline.py
```

### 方式二：分步运行

```bash
# 步骤 1: 生成样本
python generate_chirp_speed.py

# 步骤 2: 反推摩擦转矩
python derive_friction_torque.py

# 步骤 3: 分层辨识
python hierarchical_identify.py

# 步骤 4: 精调 Ts/vs
python refine_vs_ts.py
```

### 方式三：单独使用 PSO 初始化模块

```python
import pso_init

swarm = pso_init.init_swarm(
    N=100, DIM=4,
    LB=np.array([0.0, 0.0, 0.01, 0.0]),
    UB=np.array([2.0, 1.5, 2.0, 1.0]),
    seed=42
)
```

---

## 7. 算法原理图解

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  generate_chirp │ ──→ │ derive_friction │ ──→ │  hierarchical   │
│    _speed.py     │     │   _torque.py    │     │   _identify.py  │
│                 │     │                 │     │                 │
│ 多频正弦速度    │     │ T = I/Kt - J·α  │     │ ① 高速区回归   │
│ + 零偏置电流    │     │ + 保存 velocity │     │    → Tc, Tv     │
│ + 保存原始 v    │     │                 │     │ ② 对数回归     │
│                 │     │                 │     │    → Ts, vs(初) │
└─────────────────┘     └─────────────────┘     │ ③ PSO 精调     │
                                                  └────────┬────────┘
                                                           │
                                                  ┌────────▼────────┐
                                                  │  refine_vs_ts   │
                                                  │     .py         │
                                                  │ 固定 Tc/Tv      │
                                                  │ 网格搜索        │
                                                  │ + 梯度下降      │
                                                  │ → Ts, vs(精)    │
                                                  └─────────────────┘
```

---

## 8. 扩展与改进

### 8.1 提高 $v_s$ 辨识精度

当前 $v_s$ 误差约 3%，若需 < 1%：
- 增加 $|v| < 0.1$ 区间的数据密度（目前约 265 点）
- 使用 chirp 扫频信号，速度连续穿越 $v_s$ 附近
- 采用加权最小二乘，给低速区更高权重

### 8.2 在线辨识

将 `refine_vs_ts.py` 中的网格搜索替换为 **递推最小二乘 (RLS)** 或 **扩展卡尔曼滤波 (EKF)**，实现参数的实时更新。

### 8.3 温度补偿

Stribeck 参数随温度变化。可增加温度传感器，建立 $T_c(T), T_s(T), v_s(T), T_v(T)$ 的 Arrhenius 模型。

---

*文档版本: 2026-05-09*
