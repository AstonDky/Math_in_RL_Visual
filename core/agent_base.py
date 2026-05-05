"""算法基类与 Info Dict 协议。

算法更新后的可视化数据统一从 Info Dict 流出。UI、训练引擎和
TensorBoard 只依赖这个结构，不读取具体算法的内部变量。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, NotRequired, TypedDict

import numpy as np
from numpy.typing import NDArray

from core.env_base import Action, State


class MathLog(TypedDict):
    """单步更新的公式、变量和代入计算。"""

    formula: str
    calculation: NotRequired[str]
    variables: dict[str, Any]


class InfoDict(TypedDict):
    """一次算法更新的标准展示数据。

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
    """所有强化学习 Agent 的共同接口。"""

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
        """重置 episode 内状态。"""

    def reset_training_state(self) -> None:
        """清空长期训练状态。"""

    def state_dict(self) -> dict[str, Any]:
        """导出可保存的算法状态。"""

        return {}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """恢复保存过的算法状态。"""

        _ = state

    def validate_info(self, info: InfoDict) -> None:
        """检查 Info Dict 的基本形状和概率归一化。"""

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
