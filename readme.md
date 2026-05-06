# Math in RL Visual

> A visual reinforcement learning study framework for connecting formulas, code execution, policy changes and TensorBoard metrics.

## Language

- [中文说明](#中文说明)
- [English README](#english-readme)

<details open>
<summary id="中文说明">中文说明</summary>

## 为什么写这个项目

这个项目基于赵世钰老师的《强化学习的数学原理》中的 GridWorld 环境，主要是为了辅助强化学习算法的学习。

希望通过这个项目，把算法运行过程中的价值更新、策略变化和训练结果更直观地展示出来，帮助后来学习的同学在理解数学推导的同时，也能更容易看懂代码实现和算法过程。

## 运行视频

> 这里预留代码运行视频位置，后续自行添加。

<!--
后续可以在这里放视频链接，例如：

https://your-video-link
-->

## 项目构建概览

项目使用原生 Python 编写，主要依赖 NumPy、PyQt6 和 TensorBoard。

- `NumPy`: 负责 Q 表、策略概率、线性函数近似权重和特征计算。
- `PyQt6`: 负责 GridWorld、策略概率图、状态价值图、公式区和算法指针界面。
- `TensorBoard`: 负责训练指标实时观测。
- 手写 `GridWorld`: 不依赖 Gym，方便完全对齐书中环境和教学展示。

当前已经完成的功能：

- 基于书中 GridWorld 的原生环境实现。
- 主训练框显示环境 grid 和 reward 数值。
- 策略概率图用箭头显示每个动作概率，箭头长度随概率实时变化。
- 右侧状态价值图显示当前策略下的 `V^pi(s)`。
- 公式区显示每次 TD 更新的公式、变量和代入计算。
- 算法指针自动解析核心算法函数源码，并高亮当前展示行。
- 训练指标写入 `runs/<algorithm_name>`，点击开始训练后自动打开 TensorBoard。
- 支持“重新运行”和“继续上次”，并保存/恢复 checkpoint。
- 目前优先稳定演示 Algorithm 8.2 Sarsa with function approximation。

## 安装与运行

推荐使用项目对应的 Conda 环境：

```bash
conda activate MathInRL
pip install -r requirements.txt
python main.py
```

如果你不用 Conda，也可以在自己的 Python 环境中安装依赖后运行：

```bash
pip install -r requirements.txt
python main.py
```

程序启动后，在界面点击“开始”。训练开始时，TensorBoard 会等待服务端就绪后自动打开。

## 代码应该改哪里

日常使用时，主要改三个地方：

- 写算法：改 `algorithms/`。
- 选择算法和训练参数：改 `main.py` 顶部配置。
- 改环境奖励和布局：优先改 `main.py` 里的 `GridWorld(...)` 创建参数；需要换地图时再改 `envs/grid_world.py` 或在 `main.py` 传入 `layout`。

框架层的 `core/`、`ui/`、`utils/` 主要负责适配、训练、显示、保存和日志，不应该为了某一个具体算法临时改 UI 或训练循环。

## 写自己的算法

核心算法函数放在 `algorithms/` 目录里。当前示例是：

```text
algorithms/sarsa_value_function.py
```

当前算法函数：

```python
def sarsa_fa(w, s0, pi, q_hat, grad_q_hat, step,
             alpha=0.01, gamma=0.9, epsilon=0.1,
             episodes=500, max_steps=100):
    ...
    return w
```

写新算法时，尽量保持书中伪代码或笔记里的纯函数风格。算法文件里只写算法本身，例如采样、TD error、权重更新、策略选择，不写 PyQt、TensorBoard、checkpoint、Info Dict 或 UI 字段。

如果你的新算法仍然符合某个已有函数签名，框架会自动复用已有适配器。如果是全新的算法族，再到 `core/algorithm_adapters.py` 中新增签名识别和适配逻辑。

## 在 main.py 中选择训练算法

训练算法入口在 `main.py` 顶部：

```python
from algorithms.sarsa_value_function import sarsa_fa

CORE_ALGORITHM = sarsa_fa
ALGORITHM_NAME = CORE_ALGORITHM.__name__
STARTUP_MODE = "restart"
```

切换算法时，一般只需要：

1. 从 `algorithms/` 导入新的算法函数。
2. 把 `CORE_ALGORITHM` 改成新的函数。

例如：

```python
from algorithms.my_algorithm import my_algorithm

CORE_ALGORITHM = my_algorithm
ALGORITHM_NAME = CORE_ALGORITHM.__name__
```

`ALGORITHM_NAME` 默认从函数名自动生成，对应的 TensorBoard 日志会写到：

```text
runs/<algorithm_name>
```

checkpoint 会写到：

```text
sessions/<algorithm_name>/checkpoint.pkl
```

## 在 main.py 中改训练参数

训练参数集中在 `main.py` 的 `ALGORITHM_CONFIG`：

```python
ALGORITHM_CONFIG = RLAlgorithmConfig(
    alpha=0.02,
    gamma=0.9,
    epsilon=0.05,
    behavior_policy="epsilon_greedy",
    auto_refresh_policy=False,
    softmax_temperature=1.0,
    value_function="linear",
    feature_kind="one_hot",
    feature_order=0,
    weight_init="zeros",
)
```

常用字段：

- `alpha`: 学习率。
- `gamma`: 折扣因子。
- `epsilon`: epsilon-greedy 的探索概率。
- `behavior_policy`: 行为策略，可选 `"greedy"`、`"epsilon_greedy"`、`"softmax"`。
- `softmax_temperature`: 只有选择 softmax 策略时才主要使用。
- `value_function`: 价值函数形式，可选 `"table"`、`"linear"`。
- `feature_kind`: 函数近似特征，可选 `"one_hot"`、`"fourier"`。
- `feature_order`: Fourier 特征阶数。
- `weight_init`: 权重初始化方式，可选 `"zeros"`、`"normal"`。

当前默认使用 `value_function="linear"` 和 `feature_kind="one_hot"`，是为了更稳定地观察 Algorithm 8.2 的学习路径。

## 在 main.py 中改环境参数

环境创建在 `main.py` 的 `main()` 函数里：

```python
env = GridWorld(
    r_boundary=-10.0,
    r_forbidden=-10.0,
    r_target=1.0,
    r_other=0.0,
    forbidden_blocks=False,
)
```

字段含义：

- `r_boundary`: 撞到边界时的 reward。
- `r_forbidden`: 进入 forbidden 格子时的 reward。
- `r_target`: 到达 target 格子时的 reward。
- `r_other`: 普通格子移动 reward。
- `forbidden_blocks`: forbidden 格子是否作为障碍阻挡移动。

如果只是调奖励，改这里即可。

如果要换地图，有两种方式：

1. 在 `envs/grid_world.py` 中修改 `GridWorld.DEFAULT_LAYOUT`。
2. 在 `main.py` 中创建自己的 `layout`，然后传给 `GridWorld(layout=layout, ...)`。

布局编码：

```text
0 = 普通格子
1 = forbidden 格子
2 = target 格子
```

## 主要目录

```text
main.py                       算法选择、训练参数、环境参数的主要入口
algorithms/                   自己写的书中风格核心算法函数
core/algorithm_adapters.py    把纯算法函数适配到训练和可视化协议
core/rl_parameters.py         参数、Q/V 表、策略概率、函数近似特征
core/table_agent.py           把算法包装成可训练、可保存、可展示的 Agent
core/engine.py                训练线程、暂停继续、速度控制、定期保存
envs/grid_world.py            手写 GridWorld 环境
ui/                           主界面、策略图、价值图、公式区、算法指针
utils/session.py              重新运行、继续上次、checkpoint 和日志清理
utils/logger.py               TensorBoard 标量写入
utils/tensorboard.py          TensorBoard 启动和自动打开
```

</details>

<details>
<summary id="english-readme">English README</summary>

## Why This Project Exists

This project is based on the GridWorld environment from Zhao Shiyu's *Mathematical Principles of Reinforcement Learning*. Its main purpose is to support learning reinforcement learning algorithms.

It visualizes value updates, policy changes and training results during algorithm execution, so learners can connect the mathematical derivations with code implementation and the actual algorithm process more easily.

## Demo Video

> Reserved for a running-code video. I will add it later.

<!--
Future video link:

https://your-video-link
-->

## Build Overview

The project is written in native Python and mainly uses NumPy, PyQt6 and TensorBoard.

- `NumPy`: Q tables, policy probabilities, linear weights and features.
- `PyQt6`: GridWorld, policy view, value view, formula panel and source-code pointer.
- `TensorBoard`: live training metrics.
- Hand-written `GridWorld`: no Gym dependency, easier to align with the book and teaching needs.

Current features:

- Native GridWorld implementation based on the book example.
- Main training grid with environment cells and reward values.
- Policy probability arrows whose lengths change with action probabilities.
- Current-policy state-value view showing `V^pi(s)`.
- Formula panel with TD update, variables and concrete numeric substitution.
- Source-code pointer generated from the real core algorithm function.
- Metrics written to `runs/<algorithm_name>` and opened in TensorBoard after training starts.
- Restart, continue, checkpoint save and checkpoint restore.
- Current focus: stable visualization of Algorithm 8.2 Sarsa with function approximation.

## Install And Run

Recommended Conda environment:

```bash
conda activate MathInRL
pip install -r requirements.txt
python main.py
```

Regular Python environment:

```bash
pip install -r requirements.txt
python main.py
```

After the app starts, click "开始". TensorBoard opens automatically after the server is ready.

## What To Edit

For daily use, there are three main places to edit:

- Write algorithms in `algorithms/`.
- Select the algorithm and training parameters at the top of `main.py`.
- Change reward and environment parameters in the `GridWorld(...)` call in `main.py`; change the layout in `envs/grid_world.py` or pass `layout` from `main.py`.

The framework layers in `core/`, `ui/` and `utils/` handle adaptation, training, display, saving and logging. They should not need one-off changes for a single algorithm.

## Write Your Own Algorithm

Core algorithm functions live in `algorithms/`. The current example is:

```text
algorithms/sarsa_value_function.py
```

Current algorithm function:

```python
def sarsa_fa(w, s0, pi, q_hat, grad_q_hat, step,
             alpha=0.01, gamma=0.9, epsilon=0.1,
             episodes=500, max_steps=100):
    ...
    return w
```

New algorithms should stay close to book pseudocode or personal notes. Algorithm files should contain the algorithm itself: sampling, TD error, weight updates and policy choices. They should not contain PyQt, TensorBoard, checkpoint, Info Dict or UI fields.

If the function signature matches an existing adapter family, the framework reuses it automatically. For a new algorithm family, add signature matching and adaptation in `core/algorithm_adapters.py`.

## Select The Training Algorithm In main.py

The algorithm entry is at the top of `main.py`:

```python
from algorithms.sarsa_value_function import sarsa_fa

CORE_ALGORITHM = sarsa_fa
ALGORITHM_NAME = CORE_ALGORITHM.__name__
STARTUP_MODE = "restart"
```

To switch algorithms, usually:

1. Import the new function from `algorithms/`.
2. Set `CORE_ALGORITHM` to that function.

Example:

```python
from algorithms.my_algorithm import my_algorithm

CORE_ALGORITHM = my_algorithm
ALGORITHM_NAME = CORE_ALGORITHM.__name__
```

TensorBoard logs go to:

```text
runs/<algorithm_name>
```

Checkpoints go to:

```text
sessions/<algorithm_name>/checkpoint.pkl
```

## Change Training Parameters In main.py

Training parameters are in `ALGORITHM_CONFIG`:

```python
ALGORITHM_CONFIG = RLAlgorithmConfig(
    alpha=0.02,
    gamma=0.9,
    epsilon=0.05,
    behavior_policy="epsilon_greedy",
    auto_refresh_policy=False,
    softmax_temperature=1.0,
    value_function="linear",
    feature_kind="one_hot",
    feature_order=0,
    weight_init="zeros",
)
```

Common fields:

- `alpha`: learning rate.
- `gamma`: discount factor.
- `epsilon`: exploration probability for epsilon-greedy.
- `behavior_policy`: `"greedy"`, `"epsilon_greedy"` or `"softmax"`.
- `softmax_temperature`: mainly used when softmax policy is selected.
- `value_function`: `"table"` or `"linear"`.
- `feature_kind`: `"one_hot"` or `"fourier"`.
- `feature_order`: Fourier feature order.
- `weight_init`: `"zeros"` or `"normal"`.

The default uses `value_function="linear"` and `feature_kind="one_hot"` for a stable Algorithm 8.2 learning path.

## Change Environment Parameters In main.py

The environment is created inside `main()`:

```python
env = GridWorld(
    r_boundary=-10.0,
    r_forbidden=-10.0,
    r_target=1.0,
    r_other=0.0,
    forbidden_blocks=False,
)
```

Fields:

- `r_boundary`: reward for hitting the boundary.
- `r_forbidden`: reward for entering a forbidden cell.
- `r_target`: reward for reaching the target.
- `r_other`: reward for ordinary moves.
- `forbidden_blocks`: whether forbidden cells block movement.

For reward changes, edit this call.

For layout changes, either:

1. Edit `GridWorld.DEFAULT_LAYOUT` in `envs/grid_world.py`.
2. Build a custom `layout` in `main.py` and pass it as `GridWorld(layout=layout, ...)`.

Layout encoding:

```text
0 = normal cell
1 = forbidden cell
2 = target cell
```

## Main Folders

```text
main.py                       main entry for algorithm, training and environment config
algorithms/                   book-style core algorithm functions
core/algorithm_adapters.py    adapters from pure algorithms to training/visualization
core/rl_parameters.py         config, Q/V tables, policy probabilities and features
core/table_agent.py           wraps algorithms as trainable, saveable Agents
core/engine.py                training thread, pause/resume, speed control and autosave
envs/grid_world.py            hand-written GridWorld environment
ui/                           main window, policy view, value view, formula and pointer
utils/session.py              restart, continue, checkpoint and log cleanup
utils/logger.py               TensorBoard scalar writer
utils/tensorboard.py          TensorBoard launcher and browser opener
```

</details>
