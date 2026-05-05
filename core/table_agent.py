"""通用表格型算法适配器。

`algorithms/` 中只放核心算法函数；这个适配器负责把函数接入训练循环、
Info Dict、TensorBoard、UI、保存和恢复。
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
from numpy.typing import NDArray

from core.agent_base import AgentBase, InfoDict
from core.algorithm_trace import build_algorithm_trace
from core.env_base import Action, State
from core.rl_parameters import (
    CoreAlgorithm,
    RLAlgorithmConfig,
    RLAlgorithmContext,
    RLTables,
    RLTransition,
)


class TableAgent(AgentBase):
    """把一个核心 RL 函数包装成可训练、可视化、可保存的 Agent。"""

    def __init__(
        self,
        num_states: int,
        num_actions: int,
        reward_map: NDArray[np.float64],
        algorithm: CoreAlgorithm,
        config: RLAlgorithmConfig,
        algorithm_name: str,
        state_shape: tuple[int, int] | None = None,
    ) -> None:
        super().__init__(num_states=num_states, num_actions=num_actions)
        self.reward_map = reward_map
        self.algorithm = algorithm
        self._initial_config = copy.deepcopy(config)
        self.config = copy.deepcopy(config)
        self.algorithm_name = algorithm_name
        self.state_shape = state_shape
        self._rng = np.random.default_rng(self.config.seed)
        self._next_action: Action | None = None
        self._next_action_state: State | None = None
        self._algorithm_trace = build_algorithm_trace(
            self.algorithm,
            title=f"运行中的核心算法函数: {self.algorithm.__name__}",
        )
        self.tables = RLTables.create(
            num_states=num_states,
            num_actions=num_actions,
            config=self.config,
            rng=self._rng,
            state_shape=self.state_shape,
            initial_policy=self.config.initial_policy,
        )
        self.context = RLAlgorithmContext(
            num_states=num_states,
            num_actions=num_actions,
            config=self.config,
            tables=self.tables,
            rng=self._rng,
        )
        if self.config.initial_policy is None:
            self.context.refresh_all_policies()

    def act(self, state: State) -> Action:
        if self._next_action_state == state and self._next_action is not None:
            action = self._next_action
            self._next_action = None
            self._next_action_state = None
            return action
        return self.context.sample_action(state)

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
        transition = RLTransition(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            step=step,
            episode=episode,
        )
        result = self.algorithm(self.context, transition)
        self._next_action = result.next_action
        self._next_action_state = next_state if result.next_action is not None else None
        if self.config.auto_refresh_policy:
            self.context.refresh_policy_state(result.updated_state)

        info: InfoDict = {
            "policy_probs": self.tables.policy.copy(),
            "state_values": self.tables.v.copy(),
            "value_table": self.tables.q.copy(),
            "reward_map": self.reward_map.copy(),
            "math_log": {
                "formula": result.formula,
                "calculation": result.calculation,
                "variables": result.variables,
            },
            "metrics": result.metrics,
            "algorithm_trace": self._algorithm_trace,
            "algorithm_name": self.algorithm_name,
            "step": step,
            "episode": episode,
            "state": state,
            "action": action,
            "reward": reward,
            "next_state": next_state,
            "updated_state": result.updated_state,
            "done": done,
        }
        self.validate_info(info)
        return info

    def reset(self) -> None:
        """清空 episode 级临时动作缓存。

        对 SARSA 这类 on-policy 算法，上一时刻采样出的 ``a_next`` 只属于
        当前 episode；进入新 episode 后必须丢弃，不能把旧动作带到新的起点。
        """

        self._next_action = None
        self._next_action_state = None

    def reset_training_state(self) -> None:
        self.config = copy.deepcopy(self._initial_config)
        self._rng = np.random.default_rng(self.config.seed)
        self._next_action = None
        self._next_action_state = None
        self.tables = RLTables.create(
            num_states=self.num_states,
            num_actions=self.num_actions,
            config=self.config,
            rng=self._rng,
            state_shape=self.state_shape,
            initial_policy=self.config.initial_policy,
        )
        self.context = RLAlgorithmContext(
            num_states=self.num_states,
            num_actions=self.num_actions,
            config=self.config,
            tables=self.tables,
            rng=self._rng,
        )
        if self.config.initial_policy is None:
            self.context.refresh_all_policies()

    def state_dict(self) -> dict[str, Any]:
        return {
            "algorithm_name": self.algorithm_name,
            "config": self.config,
            "rng_state": self._rng.bit_generator.state,
            "q": self.tables.q.copy(),
            "v": self.tables.v.copy(),
            "policy": self.tables.policy.copy(),
            "w": self.tables.w.copy(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.config = state.get("config", self.config)
        self._rng = np.random.default_rng()
        rng_state = state.get("rng_state")
        if rng_state is not None:
            self._rng.bit_generator.state = rng_state

        self.tables = RLTables.create(
            num_states=self.num_states,
            num_actions=self.num_actions,
            config=self.config,
            rng=self._rng,
            state_shape=self.state_shape,
            initial_policy=self.config.initial_policy,
        )
        self.tables.q = np.asarray(state["q"], dtype=float).copy()
        self.tables.v = np.asarray(state["v"], dtype=float).copy()
        self.tables.policy = np.asarray(state["policy"], dtype=float).copy()
        self.tables.w = np.asarray(state.get("w", self.tables.w), dtype=float).copy()

        self.context = RLAlgorithmContext(
            num_states=self.num_states,
            num_actions=self.num_actions,
            config=self.config,
            tables=self.tables,
            rng=self._rng,
        )
        self.context.sync_q_from_w()
        self._next_action = None
        self._next_action_state = None
