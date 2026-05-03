# Math in RL Visual

面向《强化学习的数学原理》的原生 Python 教学框架。核心目标是：
环境、训练引擎、UI、TensorBoard、会话保存都由框架负责；`algorithms/`
只保存书中伪代码级别的核心强化学习算法函数。

## 目录职责

- `main.py`: 主入口和算法热插拔选择区。
- `algorithms/`: 只放核心算法函数，例如 `q_learning_step()`、`sarsa_step()`。
- `core/rl_parameters.py`: 统一参数包，覆盖 DP、MC、TD、SARSA、Q-learning、n-step、eligibility trace、Dyna 等常见表格算法参数。
- `core/table_agent.py`: 通用表格 Agent 适配器，把核心算法函数接入训练和 UI。
- `core/engine.py`: PyQt6 `QThread` 训练循环、暂停/继续、checkpoint 保存。
- `envs/grid_world.py`: 手写 5x5 GridWorld。
- `ui/`: 主环境框、策略概率十字图、最优状态值图、公式和算法代码指针。
- `utils/session.py`: 重新运行/继续上次、checkpoint 和日志清理。
- `utils/logger.py`: TensorBoard 标量记录。
- `utils/hardware.py`: CPU/GPU 检测和后端建议。

## 切换算法

通常只改 `main.py` 顶部的三行：

```python
from algorithms.greedy_q_learning import q_learning_step

ALGORITHM_NAME = "greedy_q_learning"
CORE_ALGORITHM = q_learning_step
ALGORITHM_CONFIG = RLAlgorithmConfig(alpha=0.2, gamma=0.9, epsilon=0.1)
```

新增算法时，在 `algorithms/` 中写一个函数：

```python
def my_algorithm_step(ctx: RLAlgorithmContext, transition: RLTransition) -> RLStepResult:
    ...
```

框架会自动解析这个函数源码，并在 UI 的算法指针框中逐行高亮。

## 会话

- `重新运行`: 清空当前算法的 Q 表、策略、checkpoint、TensorBoard 日志和 UI 显示。
- `继续上次`: 从 `sessions/<algorithm_name>/checkpoint.pkl` 恢复训练。
- 训练中会周期性保存，关闭窗口时也会保存。

## 运行

```bash
pip install -r requirements.txt
python main.py
```

运行后会尝试打开 TensorBoard:

```bash
http://localhost:6006
```
