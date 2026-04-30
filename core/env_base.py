"""环境基类。

这个模块只定义“环境应该提供什么能力”，不绑定任何具体算法或 UI。
后续无论添加 GridWorld、随机游走、赌博机还是更复杂的手写环境，
都应继承 :class:`EnvBase` 并遵守这里的接口。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


State = int
Action = int


@dataclass(frozen=True, slots=True)
class StepResult:
    """环境执行一步后的标准结果。

    Attributes:
        state: 执行动作前的状态编号。
        action: 算法选择的动作编号。
        reward: 环境反馈的即时奖励。
        next_state: 动作执行后的状态编号。
        done: 当前 episode 是否结束。
        extra: 环境特有的额外信息，例如坐标、碰壁标记等。
    """

    state: State
    action: Action
    reward: float
    next_state: State
    done: bool
    extra: dict[str, Any]


class EnvBase(ABC):
    """所有手写强化学习环境必须继承的抽象基类。"""

    @property
    @abstractmethod
    def num_states(self) -> int:
        """返回离散状态总数。"""

    @property
    @abstractmethod
    def num_actions(self) -> int:
        """返回离散动作总数。"""

    @property
    @abstractmethod
    def current_state(self) -> State:
        """返回环境当前状态编号。"""

    @abstractmethod
    def reset(self, seed: int | None = None) -> State:
        """重置环境并返回初始状态。"""

    @abstractmethod
    def step(self, action: Action) -> StepResult:
        """在环境中执行一个动作。"""
