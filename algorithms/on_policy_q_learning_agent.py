"""贪心 Q-learning 教学算法。

这里采用表格型 Q-learning。TD 目标使用下一状态的最大动作价值：

    Q(s,a) <- Q(s,a) + alpha [r + gamma max_a Q(s',a) - Q(s,a)]

策略显示使用 tie-aware greedy：如果四个动作 Q 值相等，就四方向等长；
训练后某个方向更优，它对应的线段才会变长。
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from core.agent_base import AgentBase, InfoDict
from core.algorithm_trace import build_algorithm_trace
from core.env_base import Action, State
from envs.grid_world import ACTION_NAMES


class GreedyQLearningAgent(AgentBase):
    """贪心 Q-learning。"""

    def __init__(
        self,
        num_states: int,
        num_actions: int,
        reward_map: NDArray[np.float64],
        alpha: float = 0.2,
        gamma: float = 0.9,
        epsilon: float = 0.1,
        seed: int | None = 13,

    ) -> None:
        super().__init__(num_states=num_states, num_actions=num_actions)
        self.reward_map = reward_map
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        self._q = np.zeros((num_states, num_actions), dtype=float)
        self._policy_probs = np.full(
            (num_states, num_actions),
            fill_value=1.0 / num_actions,
            dtype=float,
        )
        self._refresh_policy()

    def reset_training_state(self) -> None:
        """清空 Q 表、策略和随机数状态。"""

        self._rng = np.random.default_rng(self.seed)
        self._q.fill(0.0)
        self._policy_probs.fill(1.0 / self.num_actions)
        self._refresh_policy()

    def state_dict(self) -> dict[str, object]:
        return {
            "q": self._q.copy(),
            "policy_probs": self._policy_probs.copy(),
            "rng_state": self._rng.bit_generator.state,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "seed": self.seed,
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        q = np.asarray(state["q"], dtype=float)
        policy_probs = np.asarray(state["policy_probs"], dtype=float)
        if q.shape != self._q.shape:
            raise ValueError(f"Q 表形状不匹配：{q.shape} != {self._q.shape}")
        if policy_probs.shape != self._policy_probs.shape:
            raise ValueError(
                "策略概率形状不匹配："
                f"{policy_probs.shape} != {self._policy_probs.shape}"
            )

        self._q = q.copy()
        self._policy_probs = policy_probs.copy()
        self.alpha = float(state.get("alpha", self.alpha))
        self.gamma = float(state.get("gamma", self.gamma))
        self.epsilon = float(state.get("epsilon", self.epsilon))
        self.seed = state.get("seed", self.seed)
        self._rng = np.random.default_rng()
        rng_state = state.get("rng_state")
        if rng_state is not None:
            self._rng.bit_generator.state = rng_state

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
        core = self._core_algorithm(state, action, reward, next_state, done)
        next_action = int(core["next_action"])
        old_q = float(core["old_q"])
        next_q = float(core["next_q"])
        td_target = float(core["td_target"])
        td_error = float(core["td_error"])
        new_q = float(core["new_q"])

        action_name = ACTION_NAMES.get(action, str(action))
        next_action_name = ACTION_NAMES.get(next_action, str(next_action))

        info: InfoDict = {
            "policy_probs": self._policy_probs.copy(),
            "value_table": self._q.copy(),
            "reward_map": self.reward_map.copy(),
            "math_log": {
                "formula": (
                    "Q(s,a) <- Q(s,a) + alpha "
                    "[r + gamma max_a Q(s',a) - Q(s,a)]"
                ),
                "calculation": (
                    f"Q({state},{action_name}) <- {old_q:.3f} + "
                    f"{self.alpha:.2f} * ({reward:.3f} + {self.gamma:.2f} * "
                    f"max_Q({next_state}) - {old_q:.3f}) "
                    f"= {new_q:.3f}"
                ),
                "variables": {
                    "s": state,
                    "a": action_name,
                    "r": round(float(reward), 3),
                    "s_next": next_state,
                    "a_next": next_action_name,
                    "Q_old": round(old_q, 3),
                    "max_Q_next": round(next_q, 3),
                    "td_target": round(float(td_target), 3),
                    "td_error": round(float(td_error), 3),
                    "Q_new": round(float(new_q), 3),
                },
            },
            "metrics": {
                "rollout/reward": float(reward),
                "train/loss": float(td_error * td_error),
                "train/td_error": float(td_error),
                "train/q_value": float(new_q),
            },
            "algorithm_trace": {
                **build_algorithm_trace(
                    self._core_algorithm,
                    title="运行中的核心算法函数: _core_algorithm",
                ),
            },
            "step": step,
            "episode": episode,
            "state": state,
            "action": action,
            "reward": reward,
            "next_state": next_state,
            "updated_state": state,
            "done": done,
        }
        self.validate_info(info)
        return info

    def _core_algorithm(
        self,
        state: State,
        action: Action,
        reward: float,
        next_state: State,
        done: bool,
    ) -> dict[str, float | int]:
        """只放真正需要在 UI 中展示的 greedy Q-learning 核心代码。"""

        old_q = float(self._q[state, action])
        next_q = 0.0 if done else float(np.max(self._q[next_state]))
        td_target = reward + self.gamma * next_q
        td_error = td_target - old_q
        new_q = old_q + self.alpha * td_error
        self._q[state, action] = new_q
        self._refresh_policy()

        return {
            "next_action": int(np.argmax(self._q[next_state])),
            "old_q": old_q,
            "next_q": next_q,
            "td_target": td_target,
            "td_error": td_error,
            "new_q": new_q,
        }

    def _refresh_policy(self) -> None:
        self._policy_probs.fill(0.0)
        for state in range(self.num_states):
            values = self._q[state]
            best_value = np.max(values)
            best_actions = np.flatnonzero(np.isclose(values, best_value))

            greedy_bonus = (1.0 - self.epsilon) / len(best_actions)
            self._policy_probs[state, best_actions] += greedy_bonus


            # probability = 1.0 / len(best_actions)
            # self._policy_probs[state, best_actions] = probability


OnPolicyQLearningAgent = GreedyQLearningAgent
