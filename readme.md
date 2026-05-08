# Math in RL Visual

> A native Python visual reinforcement-learning study framework for connecting math, code, policy change, value change, and training results in GridWorld.

## Language

- [中文说明](#中文说明)
- [English README](#english-readme)

<details open>
<summary id="中文说明">中文说明</summary>

## 为什么写这个项目

这个项目基于赵世钰老师《强化学习的数学原理》里的 GridWorld 环境，用来辅助强化学习算法学习。
它希望把数学推导、代码实现和算法过程更直观地对应起来，帮助学习者观察价值更新、策略变化和训练结果。

## 这个项目现在能做什么

项目当前基于原生 Python、NumPy、PyQt6 和 TensorBoard 构建，已经完成这些核心能力：

- 手写 GridWorld 环境，不依赖 Gym
- 主训练网格显示环境布局和 reward
- 策略概率可视化，能看到动作概率的动态变化
- 状态价值图显示当前策略下的 `V^π(s)`
- 公式区展示当前更新公式、变量和数值代入
- 算法指针直接高亮真实核心算法源码
- 支持 TensorBoard 实时观察训练指标
- 支持在 `main.py` 中切换核心算法和参数配置

当前这份工作区已经把书中主线算法扩展到 18 个核心函数入口，覆盖：

- Dynamic Programming
- Monte Carlo
- TD control
- Function approximation
- Policy gradient / actor-critic
- Native NumPy teaching versions of DQN and deterministic actor-critic

## 运行视频




https://github.com/user-attachments/assets/fc999b72-8a74-4f34-b75d-a07ffba51816




## 如何运行

推荐使用项目约定的 Conda 环境：

```bash
conda activate MathInRL
pip install -r requirements.txt
python main.py
```

如果你已经在兼容的 Python 环境里，也可以直接：

```bash
pip install -r requirements.txt
python main.py
```

## 算法写在哪里

核心算法函数写在：

```text
algorithms/
```


## 在哪里切换训练算法和参数

训练算法和训练参数主要在 `main.py` 顶部切换。

当前入口形式是：

```python
from algorithms import *

CORE_ALGORITHM = mc_basic
ALGORITHM_NAME = CORE_ALGORITHM.__name__
STARTUP_MODE = "restart"
```

常用训练参数集中在：

```python
ALGORITHM_CONFIG = RLAlgorithmConfig(...)
```

也就是说，日常切换算法时，优先改：

- `CORE_ALGORITHM`
- `ALGORITHM_CONFIG`

## 环境参数在哪里改

环境同样在 `main.py` 里创建：

```python
env = GridWorld(
    r_boundary=-10.0,
    r_forbidden=-30.0,
    r_target=10,
    r_other=-5,
    forbidden_blocks=False,
)
```

如果只是想调 reward 或阻挡逻辑，通常改这里就够了。
如果要换地图，再去改 `envs/grid_world.py` 或从 `main.py` 传入自定义 `layout`。

## 主要目录

```text
main.py                       算法选择、训练参数、环境参数入口
algorithms/                   书中风格核心算法纯函数
core/                        算法适配、训练调度、共享参数
envs/                        手写环境
ui/                          主界面和可视化
utils/                       session、日志、TensorBoard 等辅助功能
```

</details>

<details>
<summary id="english-readme">English README</summary>

## Why This Project Exists

This project is based on the GridWorld environment from Zhao Shiyu's *Mathematical Principles of Reinforcement Learning*.
Its purpose is to help learners study reinforcement learning by making the math, the code, and the algorithm process easier to line up visually.

## What The Project Can Do Now

The project is currently built with native Python, NumPy, PyQt6, and TensorBoard, and already provides:

- a hand-written GridWorld environment without Gym
- a main grid showing layout and rewards
- policy-probability visualization
- a state-value view showing `V^π(s)`
- a formula panel for the current update
- a source-code pointer on the real core algorithm
- TensorBoard-based training metric monitoring
- algorithm and parameter switching from `main.py`

This workspace currently exposes 18 core algorithm entries across:

- Dynamic Programming
- Monte Carlo
- TD control
- Function approximation
- Policy gradient / actor-critic
- Native NumPy teaching versions of DQN and deterministic actor-critic

## Demo Video



https://github.com/user-attachments/assets/d823af37-37c2-47f4-bcd5-6993d6432635



## Run

Recommended Conda environment:

```bash
conda activate MathInRL
pip install -r requirements.txt
python main.py
```

Or a compatible Python environment:

```bash
pip install -r requirements.txt
python main.py
```

## Where To Write Algorithms

Core algorithm functions live in:

```text
algorithms/
```


## Where To Switch Algorithms And Parameters

Algorithm selection and training parameters are mainly changed at the top of `main.py`.

The current entry pattern is:

```python
from algorithms import *

CORE_ALGORITHM = mc_basic
ALGORITHM_NAME = CORE_ALGORITHM.__name__
STARTUP_MODE = "restart"
```

Training parameters are mainly grouped in:

```python
ALGORITHM_CONFIG = RLAlgorithmConfig(...)
```

So for normal use, the main things to change are:

- `CORE_ALGORITHM`
- `ALGORITHM_CONFIG`

## Where To Change Environment Parameters

The environment is also created in `main.py`:

```python
env = GridWorld(
    r_boundary=-10.0,
    r_forbidden=-30.0,
    r_target=10,
    r_other=-5,
    forbidden_blocks=False,
)
```

If you only want to change rewards or blocking behavior, editing this call is usually enough.
If you want a different map, edit `envs/grid_world.py` or pass a custom `layout` from `main.py`.

## Main Folders

```text
main.py                       algorithm selection, training config, environment config
algorithms/                   book-style pure core algorithm functions
core/                         adaptation, training flow, shared parameters
envs/                         hand-written environments
ui/                           main interface and visualization
utils/                        session, logging, TensorBoard helpers
```

</details>
