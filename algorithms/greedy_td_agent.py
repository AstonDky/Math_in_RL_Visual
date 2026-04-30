"""Greedy TD 控制算法。

这里实现的是教学版 Q-learning / greedy TD control：

    Q(s,a) <- Q(s,a) + alpha * [r + gamma max_a' Q(s',a') - Q(s,a)]

算法每一步都返回标准 Info Dict，因此 UI 不需要读取 Q 表等私有变量。
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from core.agent_base import AgentBase, InfoDict
from core.env_base import Action, State
from envs.grid_world import ACTION_NAMES


TransitionModel = Callable[[State, Action], tuple[State, float, bool, dict[str, object]]]


class GreedyTdAgent(AgentBase):
    """带少量柔化概率的贪心 TD 控制算法。

    `policy_probs` 不直接做成 0/1，是为了让 Figure 7.5 风格的概率绿线
    更容易观察：最佳动作最长，其余动作仍保留很短的线段。
    """

    algorithm_lines = [
        "1. 读取当前状态 s",
        "2. 按 greedy 策略选择动作 a = argmax Q(s,a)",
        "3. 与环境交互，得到 r 和 s'",
        "4. 计算 TD 误差 delta = r + gamma max Q(s',a') - Q(s,a)",
        "5. 更新 Q(s,a)，并重算策略 pi(a|s)",
    ]

    def __init__(
        self,
        num_states: int,
        num_actions: int,
        transition_model: TransitionModel,
        reward_map: NDArray[np.float64],
        alpha: float = 0.18,
        gamma: float = 0.92,
        epsilon: float = 0.1,
        greedy_prob: float = 0.82,
        seed: int | None = 7,
    ) -> None:
        super().__init__(num_states=num_states, num_actions=num_actions)
        self.transition_model = transition_model
        self.reward_map = reward_map
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.greedy_prob = greedy_prob
        self._rng = np.random.default_rng(seed)
        self._q = np.zeros((num_states, num_actions), dtype=float)
        self._policy_probs = np.full(
            (num_states, num_actions),
            fill_value=1.0 / num_actions,
            dtype=float,
        )
        self._recompute_policy()

    def act(self, state: State) -> Action:
        """按当前策略采样动作。

        因为策略已经高度偏向 greedy 动作，所以这里既能体现贪心，
        又能保留一点探索，避免早期全零 Q 表时卡在单一路径。
        """

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
        old_q = float(self._q[state, action])
        bootstrap = 0.0 if done else float(np.max(self._q[next_state]))
        td_target = reward + self.gamma * bootstrap
        td_error = td_target - old_q
        new_q = old_q + self.alpha * td_error
        self._q[state, action] = new_q
        self._recompute_policy()

        action_name = ACTION_NAMES.get(action, str(action))
        best_next_action = int(np.argmax(self._q[next_state]))

        info: InfoDict = {
            "policy_probs": self._policy_probs.copy(),
            "value_table": self._q.copy(),
            "reward_map": self.reward_map.copy(),
            "math_log": {
                "formula": (
                    "Q(s,a) <- Q(s,a) + alpha "
                    "[r + gamma max Q(s',a') - Q(s,a)]"
                ),
                "calculation": (
                    f"Q({state},{action_name}) <- {old_q:.3f} + "
                    f"{self.alpha:.2f} * ({reward:.3f} + {self.gamma:.2f} * "
                    f"{bootstrap:.3f} - {old_q:.3f}) = {new_q:.3f}"
                ),
                "variables": {
                    "s": state,
                    "a": action_name,
                    "r": round(float(reward), 3),
                    "s_next": next_state,
                    "max_Q_next": round(bootstrap, 3),
                    "best_next_action": ACTION_NAMES.get(
                        best_next_action,
                        str(best_next_action),
                    ),
                    "td_target": round(float(td_target), 3),
                    "td_error": round(float(td_error), 3),
                    "Q_old": round(old_q, 3),
                    "Q_new": round(float(new_q), 3),
                },
            },
            "metrics": {
                "rollout/reward": float(reward),
                "train/td_error": float(td_error),
                "train/loss": float(td_error * td_error),
                "train/max_q": float(np.max(self._q)),
            },
            "algorithm_trace": {
                "title": "Greedy TD Control",
                "lines": self.algorithm_lines,
                "current_line": 5,
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

    def _recompute_policy(self) -> None:
        """根据 Q 表重算柔化 greedy 策略。"""

        low_prob = (1.0 - self.greedy_prob) / (self.num_actions - 1)
        self._policy_probs.fill(low_prob)

        for state in range(self.num_states):
            values = self._q[state]
            max_value = np.max(values)
            best_actions = np.flatnonzero(np.isclose(values, max_value))
            best_action = int(self._rng.choice(best_actions))
            self._policy_probs[state, best_action] = self.greedy_prob
