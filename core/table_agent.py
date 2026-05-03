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
    ) -> None:
        super().__init__(num_states=num_states, num_actions=num_actions)
        self.reward_map = reward_map
        self.algorithm = algorithm
        self._initial_config = copy.deepcopy(config)
        self.config = copy.deepcopy(config)
        self.algorithm_name = algorithm_name
        self._rng = np.random.default_rng(config.seed)
        self.tables = RLTables.create(
            num_states=num_states,
            num_actions=num_actions,
            initial_policy=config.initial_policy,
        )
        self.context = RLAlgorithmContext(
            num_states=num_states,
            num_actions=num_actions,
            config=config,
            tables=self.tables,
            rng=self._rng,
        )
        if config.initial_policy is None:
            self.context.refresh_all_policies()

    def act(self, state: State) -> Action:
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
        if self.config.auto_refresh_policy:
            self.context.refresh_policy_state(result.updated_state)
            self.context.decay_epsilon()

        info: InfoDict = {
            "policy_probs": self.tables.policy.copy(),
            "value_table": self.tables.q.copy(),
            "reward_map": self.reward_map.copy(),
            "math_log": {
                "formula": result.formula,
                "calculation": result.calculation,
                "variables": result.variables,
            },
            "metrics": result.metrics,
            "algorithm_trace": build_algorithm_trace(
                self.algorithm,
                title=f"运行中的核心算法函数: {self.algorithm.__name__}",
            ),
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

    def reset_training_state(self) -> None:
        self.config = copy.deepcopy(self._initial_config)
        self._rng = np.random.default_rng(self.config.seed)
        self.tables = RLTables.create(
            num_states=self.num_states,
            num_actions=self.num_actions,
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
            "returns_sum": self.tables.returns_sum.copy(),
            "returns_count": self.tables.returns_count.copy(),
            "eligibility": self.tables.eligibility.copy(),
            "visit_count": self.tables.visit_count.copy(),
            "model_next_state": self.tables.model_next_state.copy(),
            "model_reward": self.tables.model_reward.copy(),
            "episode_buffer": list(self.tables.episode_buffer),
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
            initial_policy=self.config.initial_policy,
        )
        self.tables.q = np.asarray(state["q"], dtype=float).copy()
        self.tables.v = np.asarray(state["v"], dtype=float).copy()
        self.tables.policy = np.asarray(state["policy"], dtype=float).copy()
        self.tables.returns_sum = np.asarray(state["returns_sum"], dtype=float).copy()
        self.tables.returns_count = np.asarray(
            state["returns_count"],
            dtype=float,
        ).copy()
        self.tables.eligibility = np.asarray(state["eligibility"], dtype=float).copy()
        self.tables.visit_count = np.asarray(state["visit_count"], dtype=float).copy()
        self.tables.model_next_state = np.asarray(
            state["model_next_state"],
            dtype=int,
        ).copy()
        self.tables.model_reward = np.asarray(state["model_reward"], dtype=float).copy()

        self.context = RLAlgorithmContext(
            num_states=self.num_states,
            num_actions=self.num_actions,
            config=self.config,
            tables=self.tables,
            rng=self._rng,
        )
