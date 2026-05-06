"""把核心算法函数接入训练框架的 Agent。"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
from numpy.typing import NDArray

from core.agent_base import AgentBase, EpisodeBatch, InfoDict
from core.algorithm_adapters import AdaptedAlgorithm
from core.algorithm_trace import build_algorithm_trace
from core.env_base import Action, EnvBase, State
from core.rl_parameters import (
    RLAlgorithmContext,
    RLAlgorithmConfig,
    RLStepEvent,
    RLTables,
    RLTransition,
)


class TableAgent(AgentBase):
    """面向离散状态动作空间的通用 Agent 包装器。"""

    def __init__(
        self,
        num_states: int,
        num_actions: int,
        reward_map: NDArray[np.float64],
        algorithm: AdaptedAlgorithm,
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
            self.algorithm.display_function,
            title=f"运行中的核心算法函数: {self.algorithm.name}",
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
        if self.training_mode() != "transition":
            raise RuntimeError("episode 自主训练模式下不应通过 act() 驱动算法。")
        if self._next_action_state == state and self._next_action is not None:
            action = self._next_action
            self._next_action = None
            self._next_action_state = None
            return action
        return self.context.sample_action(state)

    def training_mode(self) -> str:
        return self.algorithm.mode

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
        if self.algorithm.transition_runner is None:
            raise RuntimeError("当前算法没有 transition 驱动的适配器。")
        transition = RLTransition(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            step=step,
            episode=episode,
        )
        result = self.algorithm.transition_runner(self.context, transition)
        self._next_action = result.next_action
        self._next_action_state = next_state if result.next_action is not None else None
        if self.config.auto_refresh_policy:
            self.context.refresh_policy_state(result.updated_state)
        return self._build_info(
            event=RLStepEvent(
                transition=transition,
                result=result,
                env_extra={},
            )
        )

    def run_episode(
        self,
        env: EnvBase,
        step: int,
        episode: int,
        max_steps: int,
    ) -> EpisodeBatch:
        if self.algorithm.episode_runner is None:
            raise RuntimeError("当前算法没有 episode 驱动的适配器。")
        batch = self.algorithm.episode_runner(
            self.context,
            env,
            step,
            episode,
            max_steps,
        )
        infos = [self._build_info(event) for event in batch.updates]
        self._next_action = None
        self._next_action_state = None
        return EpisodeBatch(
            infos=infos,
            episode_reward=batch.episode_reward,
            episode_steps=batch.episode_steps,
        )

    def _build_info(self, event: RLStepEvent) -> InfoDict:
        transition = event.transition
        result = event.result
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
            "step": transition.step,
            "episode": transition.episode,
            "state": transition.state,
            "action": transition.action,
            "reward": transition.reward,
            "next_state": transition.next_state,
            "updated_state": result.updated_state,
            "done": transition.done,
            "env_extra": event.env_extra,
        }
        self.validate_info(info)
        return info

    def reset(self) -> None:
        """清空 episode 内复用的动作缓存。"""

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
            "theta": self.tables.theta.copy(),
            "extras": copy.deepcopy(self.tables.extras),
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
        self.tables.theta = np.asarray(
            state.get("theta", self.tables.theta),
            dtype=float,
        ).copy()
        self.tables.extras = copy.deepcopy(state.get("extras", {}))

        self.context = RLAlgorithmContext(
            num_states=self.num_states,
            num_actions=self.num_actions,
            config=self.config,
            tables=self.tables,
            rng=self._rng,
        )
        self.context.sync_q_from_w()
        self.context.sync_state_values()
        self._next_action = None
        self._next_action_state = None
