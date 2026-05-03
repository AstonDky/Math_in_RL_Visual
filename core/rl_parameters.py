"""强化学习通用参数、状态表和算法函数协议。

这里集中放“书中算法常见会用到的参数”。算法函数只关心这些结构，
不用知道 UI、QThread、TensorBoard 或环境实现细节。
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Deque, Literal

import numpy as np
from numpy.typing import NDArray

from core.env_base import Action, State


PolicyKind = Literal["greedy", "epsilon_greedy", "softmax", "custom"]


@dataclass(slots=True)
class RLAlgorithmConfig:
    """表格型 RL 算法的通用超参数。

    字段覆盖 DP、MC、TD、SARSA、Q-learning、Expected SARSA、n-step、
    eligibility trace、Dyna 等书中常见算法所需的主要参数。
    未被当前算法使用的参数会保留在配置中，方便以后切换算法。
    """

    alpha: float = 0.2
    gamma: float = 0.9
    epsilon: float = 0.1
    epsilon_min: float = 0.01
    epsilon_decay: float = 1.0
    temperature: float = 1.0
    theta: float = 1e-6
    lambda_: float = 0.0
    n_step: int = 1
    planning_steps: int = 0
    max_iterations: int = 10_000
    first_visit: bool = True
    exploring_starts: bool = False
    ordinary_importance_sampling: bool = True
    behavior_policy: PolicyKind = "epsilon_greedy"
    target_policy: PolicyKind = "greedy"
    auto_refresh_policy: bool = True
    initial_policy: NDArray[np.float64] | None = None
    seed: int | None = 13


@dataclass(slots=True)
class RLTransition:
    """一次环境交互样本。"""

    state: State
    action: Action
    reward: float
    next_state: State
    done: bool
    step: int
    episode: int


@dataclass(slots=True)
class RLStepResult:
    """核心算法函数的标准返回值。"""

    formula: str
    calculation: str
    variables: dict[str, Any]
    metrics: dict[str, float]
    updated_state: State


@dataclass(slots=True)
class RLTables:
    """算法共享状态表。"""

    q: NDArray[np.float64]
    v: NDArray[np.float64]
    policy: NDArray[np.float64]
    returns_sum: NDArray[np.float64]
    returns_count: NDArray[np.float64]
    eligibility: NDArray[np.float64]
    visit_count: NDArray[np.float64]
    model_next_state: NDArray[np.int_]
    model_reward: NDArray[np.float64]
    episode_buffer: Deque[RLTransition] = field(default_factory=deque)

    @classmethod
    def create(
        cls,
        num_states: int,
        num_actions: int,
        initial_policy: NDArray[np.float64] | None = None,
    ) -> "RLTables":
        q = np.zeros((num_states, num_actions), dtype=float)
        v = np.zeros(num_states, dtype=float)
        policy = _build_initial_policy(num_states, num_actions, initial_policy)
        return cls(
            q=q,
            v=v,
            policy=policy,
            returns_sum=np.zeros((num_states, num_actions), dtype=float),
            returns_count=np.zeros((num_states, num_actions), dtype=float),
            eligibility=np.zeros((num_states, num_actions), dtype=float),
            visit_count=np.zeros((num_states, num_actions), dtype=float),
            model_next_state=np.full((num_states, num_actions), -1, dtype=int),
            model_reward=np.zeros((num_states, num_actions), dtype=float),
        )


@dataclass(slots=True)
class RLAlgorithmContext:
    """传给核心算法函数的上下文。"""

    num_states: int
    num_actions: int
    config: RLAlgorithmConfig
    tables: RLTables
    rng: np.random.Generator

    def sample_action(self, state: State) -> Action:
        probs = self.tables.policy[state].astype(float, copy=True)
        total = float(probs.sum())
        if total <= 0.0:
            probs.fill(1.0 / self.num_actions)
        else:
            probs /= total
        return int(self.rng.choice(self.num_actions, p=probs))

    def refresh_policy_state(self, state: State) -> None:
        self.tables.policy[state] = make_policy_probs(
            q_values=self.tables.q[state],
            policy_kind=self.config.behavior_policy,
            epsilon=self.config.epsilon,
            temperature=self.config.temperature,
        )

    def refresh_all_policies(self) -> None:
        for state in range(self.num_states):
            self.refresh_policy_state(state)

    def decay_epsilon(self) -> None:
        self.config.epsilon = max(
            self.config.epsilon_min,
            self.config.epsilon * self.config.epsilon_decay,
        )


CoreAlgorithm = Callable[[RLAlgorithmContext, RLTransition], RLStepResult]


def make_policy_probs(
    q_values: NDArray[np.float64],
    policy_kind: PolicyKind,
    epsilon: float,
    temperature: float,
) -> NDArray[np.float64]:
    """根据 Q 值生成一行策略概率。"""

    num_actions = q_values.shape[0]

    if policy_kind == "custom":
        raise ValueError("custom policy 必须通过 config.initial_policy 提供。")

    if policy_kind == "softmax":
        scaled = q_values / max(temperature, 1e-8)
        shifted = scaled - np.max(scaled)
        exp_values = np.exp(shifted)
        return exp_values / exp_values.sum()

    best_value = np.max(q_values)
    best_actions = np.flatnonzero(np.isclose(q_values, best_value))

    if policy_kind == "greedy":
        probs = np.zeros(num_actions, dtype=float)
        probs[best_actions] = 1.0 / len(best_actions)
        return probs

    if policy_kind == "epsilon_greedy":
        probs = np.full(num_actions, epsilon / num_actions, dtype=float)
        probs[best_actions] += (1.0 - epsilon) / len(best_actions)
        return probs

    raise ValueError(f"未知 policy_kind: {policy_kind}")


def _build_initial_policy(
    num_states: int,
    num_actions: int,
    initial_policy: NDArray[np.float64] | None,
) -> NDArray[np.float64]:
    if initial_policy is None:
        return np.full((num_states, num_actions), 1.0 / num_actions, dtype=float)

    policy = np.asarray(initial_policy, dtype=float)
    expected_shape = (num_states, num_actions)
    if policy.shape != expected_shape:
        raise ValueError(f"initial_policy 形状必须为 {expected_shape}。")

    row_sums = policy.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0):
        raise ValueError("initial_policy 每一行概率和必须大于 0。")
    return policy / row_sums
