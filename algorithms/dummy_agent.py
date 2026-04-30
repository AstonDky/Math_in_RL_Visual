"""用于验证框架数据流的随机策略算法。"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from core.agent_base import AgentBase, InfoDict
from core.env_base import Action, State
from envs.grid_world import ACTION_NAMES


class DummyAgent(AgentBase):
    """最小可运行算法：每个状态下均匀随机选择动作。

    它不学习，只负责稳定地产生标准 Info Dict。
    因此 UI、训练引擎和 TensorBoard 先能完整跑通，后续算法可直接替换。
    """

    def __init__(
        self,
        num_states: int,
        num_actions: int,
        seed: int | None = 42,
    ) -> None:
        super().__init__(num_states=num_states, num_actions=num_actions)
        self._rng = np.random.default_rng(seed)
        self._policy_probs = np.full(
            (num_states, num_actions),
            fill_value=1.0 / num_actions,
            dtype=float,
        )

    @property
    def policy_probs(self) -> NDArray[np.float64]:
        """返回策略副本，避免外部对象误改算法内部矩阵。"""

        return self._policy_probs.copy()

    def act(self, state: State) -> Action:
        probs = self._policy_probs[state]
        return int(self._rng.choice(self.num_actions, p=probs))

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
        action_name = ACTION_NAMES.get(action, str(action))
        info: InfoDict = {
            "policy_probs": self.policy_probs,
            "math_log": {
                "formula": r"\pi(a|s)=\frac{1}{|\mathcal{A}|}",
                "variables": {
                    "s": state,
                    "a": action_name,
                    "r": reward,
                    "s_next": next_state,
                    "done": done,
                    "probability": float(self._policy_probs[state, action]),
                },
            },
            "metrics": {
                "reward": float(reward),
                "episode": float(episode),
                "action": float(action),
            },
            "step": step,
            "episode": episode,
            "state": state,
            "action": action,
            "reward": reward,
            "next_state": next_state,
            "done": done,
        }
        self.validate_info(info)
        return info
