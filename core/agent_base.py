"""算法基类与 Info Dict 协议。

本项目最重要的架构约束在这里：
UI、训练引擎、TensorBoard 记录器都不允许读取算法的私有变量。
算法每完成一次更新，必须通过标准 Info Dict 主动“汇报”可视化与监控数据。
这样切换算法时，UI 不需要知道具体算法是 TD、MC、DP 还是 Q-learning。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, NotRequired, TypedDict

import numpy as np
from numpy.typing import NDArray

from core.env_base import Action, State


class MathLog(TypedDict):
    """单步数学计算日志，用于 UI 展示公式、变量与中间结果。"""

    formula: str
    calculation: NotRequired[str]
    variables: dict[str, Any]


class InfoDict(TypedDict):
    """算法向外部系统返回的唯一标准数据通道。

    必填字段:
        policy_probs: 形状为 [num_states, num_actions] 的策略概率矩阵。
        math_log: 当前更新公式和参与计算的变量。
        metrics: 可写入 TensorBoard 的标量指标。

    可选字段:
        step: 全局训练步编号。
        episode: 当前 episode 编号。
        state/action/reward/next_state/done: 当前交互样本，方便 UI 显示流程。
    """

    policy_probs: NDArray[np.float64]
    math_log: MathLog
    metrics: dict[str, float]
    step: NotRequired[int]
    episode: NotRequired[int]
    state: NotRequired[State]
    action: NotRequired[Action]
    reward: NotRequired[float]
    next_state: NotRequired[State]
    updated_state: NotRequired[State]
    done: NotRequired[bool]
    env_extra: NotRequired[dict[str, Any]]
    state_values: NotRequired[NDArray[np.float64]]
    value_table: NotRequired[NDArray[np.float64]]
    reward_map: NotRequired[NDArray[np.float64]]
    algorithm_trace: NotRequired[dict[str, Any]]
    algorithm_name: NotRequired[str]


class AgentBase(ABC):
    """所有强化学习算法的抽象基类。

    新算法只需要继承这个类，并实现 :meth:`act` 与 :meth:`update`。
    只要返回的 Info Dict 满足协议，主函数、训练引擎和 UI 就无需改动。
    """

    def __init__(self, num_states: int, num_actions: int) -> None:
        self.num_states = num_states
        self.num_actions = num_actions

    @abstractmethod
    def act(self, state: State) -> Action:
        """根据当前策略选择动作。"""

    @abstractmethod
    def update(
        self,
        state: State,
        action: Action,
        reward: float,
        next_state: State,
        done: bool,
        step: int,
        episode: int,
    ) -> InfoDict:
        """执行一次算法更新，并返回标准 Info Dict。"""

    def reset(self) -> None:
        """在新 episode 开始时重置算法内部临时状态。

        不是每种算法都需要 episode 级缓存，因此这里提供空实现。
        """

    def reset_training_state(self) -> None:
        """清空训练得到的长期状态，用于“重新运行”。"""

    def state_dict(self) -> dict[str, Any]:
        """导出算法状态，用于保存 checkpoint。"""

        return {}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """加载算法状态，用于“继续上一次运行”。"""

        _ = state

    def validate_info(self, info: InfoDict) -> None:
        """在训练引擎边界检查 Info Dict，尽早暴露算法实现错误。"""

        probs = info["policy_probs"]
        expected_shape = (self.num_states, self.num_actions)

        if probs.shape != expected_shape:
            raise ValueError(
                "policy_probs 的形状必须是 "
                f"{expected_shape}，当前得到 {probs.shape}。"
            )

        row_sums = probs.sum(axis=1)
        if not np.allclose(row_sums, 1.0):
            raise ValueError("policy_probs 每一行的动作概率之和必须为 1。")
