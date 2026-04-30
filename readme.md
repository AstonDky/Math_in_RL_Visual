# Math in RL Visual

一个面向《强化学习的数学原理》学习过程的原生 Python 教学框架。第一阶段目标是把“环境、算法、训练引擎、UI、监控记录”解耦，让后续算法能通过统一接口热插拔。

## 当前骨架

- `core/env_base.py`: 环境抽象接口，统一 `reset()` 与 `step()`。
- `core/agent_base.py`: 算法抽象接口和标准 `InfoDict` 协议。
- `core/engine.py`: 基于 `QThread` 的异步训练循环。
- `envs/grid_world.py`: prompt 中指定的 5x5 GridWorld。
- `algorithms/dummy_agent.py`: 均匀随机策略，用于打通数据流。
- `algorithms/greedy_td_agent.py`: Greedy TD / Q-learning 风格测试算法。
- `algorithms/on_policy_q_learning_agent.py`: on-policy Q-learning / SARSA 风格测试算法。
- `ui/main_window.py`: 主窗口、播放暂停、速度滑块与数据分发。
- `ui/grid_painter.py`: 网格、智能体位置、策略概率绿线绘制。
- `ui/monitor_panel.py`: 公式计算、奖励 r 网格、TensorBoard 说明和算法指针。
- `ui/metric_plot.py`: 备用 PyQt 原生曲线组件，当前主界面不使用。
- `ui/reward_grid.py`: Figure 4.4(b) 风格奖励网格。
- `ui/algorithm_pointer.py`: 伪代码高亮指针。
- `core/algorithm_trace.py`: 自动解析算法核心函数源码，生成算法指针内容。
- `utils/logger.py`: TensorBoard 标量记录器。

## 算法热插拔位置

只需要修改 `main.py` 中的 `build_agent()`：

```python
def build_agent(env: GridWorld) -> AgentBase:
    return OnPolicyQLearningAgent(
        num_states=env.num_states,
        num_actions=env.num_actions,
        reward_map=env.reward_map(),
    )
```

后续新增算法时，只要继承 `AgentBase` 并返回标准 `InfoDict`，UI 和训练引擎不需要改。
若希望算法指针自动显示核心代码，把真正的更新逻辑放进一个独立函数，
再在 `InfoDict["algorithm_trace"]` 中调用 `build_algorithm_trace(self._core_algorithm)`。

## 运行

```bash
pip install -r requirements.txt
python main.py
```

运行 `main.py` 时会尝试同步启动 TensorBoard 并打开：

```bash
http://localhost:6006
```

如果环境里没有安装 TensorBoard，PyQt 主界面仍会正常启动；安装
`requirements.txt` 后再次运行即可。
