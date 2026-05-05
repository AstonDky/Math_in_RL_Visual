"""强化学习通用参数、状态表和算法函数协议。

这里集中放“书中算法常见会用到的参数”。算法函数只关心这些结构，
不用知道 UI、QThread、TensorBoard 或环境实现细节。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from core.env_base import Action, State


PolicyKind = Literal["greedy", "epsilon_greedy", "softmax"]
ValueFunctionKind = Literal["table", "linear"]
FeatureKind = Literal["one_hot", "fourier"]
WeightInitKind = Literal["zeros", "normal"]


@dataclass(slots=True)
class RLAlgorithmConfig:
    """当前书中算法可视化需要的最小通用参数。"""

    alpha: float = 0.2
    gamma: float = 0.9
    epsilon: float = 0.1
    behavior_policy: PolicyKind = "epsilon_greedy"
    auto_refresh_policy: bool = True
    initial_policy: NDArray[np.float64] | None = None
    seed: int | None = 13
    softmax_temperature: float = 1.0
    value_function: ValueFunctionKind = "table"
    feature_kind: FeatureKind = "one_hot"
    feature_order: int = 0
    weight_init: WeightInitKind = "zeros"
    weight_mean: float = 0.0
    weight_std: float = 1.0


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
    next_action: Action | None = None


@dataclass(slots=True)
class RLTables:
    """算法共享状态表。"""

    q: NDArray[np.float64]
    v: NDArray[np.float64]
    policy: NDArray[np.float64]
    w: NDArray[np.float64]
    features: NDArray[np.float64]

    @classmethod
    def create(
        cls,
        num_states: int,
        num_actions: int,
        config: RLAlgorithmConfig,
        rng: np.random.Generator,
        state_shape: tuple[int, int] | None = None,
        initial_policy: NDArray[np.float64] | None = None,
    ) -> "RLTables":
        q = np.zeros((num_states, num_actions), dtype=float)
        policy = _build_initial_policy(num_states, num_actions, initial_policy)
        features = build_action_value_features(
            num_states=num_states,
            num_actions=num_actions,
            feature_kind=config.feature_kind,
            feature_order=config.feature_order,
            state_shape=state_shape,
        )
        w = _build_initial_w(
            feature_dim=features.shape[-1],
            config=config,
            rng=rng,
        )
        if config.value_function == "linear":
            q = np.tensordot(features, w, axes=([2], [0]))
        v = make_state_values(q, policy)
        return cls(
            q=q,
            v=v,
            policy=policy,
            w=w,
            features=features,
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

    def q_hat(self, state: State, action: Action) -> float:
        if self.config.value_function == "linear":
            return float(np.dot(self.tables.features[state, action], self.tables.w))
        return float(self.tables.q[state, action])

    def grad_q_hat(self, state: State, action: Action) -> NDArray[np.float64]:
        return self.tables.features[state, action].copy()

    def sync_q_from_w(self) -> None:
        if self.config.value_function != "linear":
            return
        self.tables.q = np.tensordot(self.tables.features, self.tables.w, axes=([2], [0]))
        self.sync_state_values()

    def sync_state_value(self, state: State) -> None:
        self.tables.v[state] = make_state_value(
            q_values=self.tables.q[state],
            policy_probs=self.tables.policy[state],
        )

    def sync_state_values(self) -> None:
        self.tables.v = make_state_values(
            q_table=self.tables.q,
            policy_probs=self.tables.policy,
        )

    def refresh_policy_state(self, state: State) -> None:
        self.tables.policy[state] = make_policy_probs(
            q_values=self.tables.q[state],
            policy_kind=self.config.behavior_policy,
            epsilon=self.config.epsilon,
            softmax_temperature=self.config.softmax_temperature,
        )
        self.sync_state_value(state)

    def refresh_all_policies(self) -> None:
        for state in range(self.num_states):
            self.refresh_policy_state(state)
        self.sync_state_values()


CoreAlgorithm = Callable[[RLAlgorithmContext, RLTransition], RLStepResult]


def make_policy_probs(
    q_values: NDArray[np.float64],
    policy_kind: PolicyKind,
    epsilon: float,
    softmax_temperature: float = 1.0,
) -> NDArray[np.float64]:
    """根据 Q 值生成一行策略概率。"""

    num_actions = q_values.shape[0]

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

    if policy_kind == "softmax":
        temperature = max(float(softmax_temperature), 1e-8)
        logits = (q_values - best_value) / temperature
        weights = np.exp(logits)
        total = float(weights.sum())
        if total <= 0.0:
            return np.full(num_actions, 1.0 / num_actions, dtype=float)
        return weights / total

    raise ValueError(f"未知 policy_kind: {policy_kind}")


def make_state_value(
    q_values: NDArray[np.float64],
    policy_probs: NDArray[np.float64],
) -> float:
    """Return V^pi(s)=sum_a pi(a|s) Q(s,a) for one state."""

    probs = np.asarray(policy_probs, dtype=float)
    probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
    probs = np.clip(probs, 0.0, None)
    total = float(probs.sum())
    if total <= 0.0:
        probs = np.full_like(probs, 1.0 / probs.shape[0], dtype=float)
    else:
        probs = probs / total
    return float(np.dot(probs, np.asarray(q_values, dtype=float)))


def make_state_values(
    q_table: NDArray[np.float64],
    policy_probs: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return current-policy state values for every state."""

    q_table = np.asarray(q_table, dtype=float)
    policy_probs = np.asarray(policy_probs, dtype=float)
    return np.array(
        [
            make_state_value(q_values=q_table[state], policy_probs=policy_probs[state])
            for state in range(q_table.shape[0])
        ],
        dtype=float,
    )


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


def build_action_value_features(
    num_states: int,
    num_actions: int,
    feature_kind: FeatureKind,
    feature_order: int,
    state_shape: tuple[int, int] | None,
) -> NDArray[np.float64]:
    """Build phi(s,a) for action-value function approximation."""

    if feature_kind == "one_hot":
        features = np.zeros((num_states, num_actions, num_states * num_actions))
        for state in range(num_states):
            for action in range(num_actions):
                features[state, action, state * num_actions + action] = 1.0
        return features

    if feature_kind != "fourier":
        raise ValueError(f"未知 feature_kind: {feature_kind}")

    if state_shape is None:
        raise ValueError("fourier 特征需要提供 state_shape=(rows, cols)。")
    if feature_order < 0:
        raise ValueError("feature_order 必须大于等于 0。")

    rows, cols = state_shape
    coefficients = np.array(
        [
            (row_order, col_order)
            for row_order in range(feature_order + 1)
            for col_order in range(feature_order + 1)
        ],
        dtype=float,
    )
    state_feature_dim = coefficients.shape[0]
    feature_dim = state_feature_dim * num_actions
    features = np.zeros((num_states, num_actions, feature_dim), dtype=float)

    row_denominator = max(1, rows - 1)
    col_denominator = max(1, cols - 1)
    for state in range(num_states):
        row, col = divmod(state, cols)
        normalized_location = np.array(
            [
                -1.0 + 2.0 * row / row_denominator,
                -1.0 + 2.0 * col / col_denominator,
            ],
            dtype=float,
        )
        phi_s = np.cos(np.pi * coefficients @ normalized_location)
        for action in range(num_actions):
            start = action * state_feature_dim
            stop = start + state_feature_dim
            features[state, action, start:stop] = phi_s

    return features


def _build_initial_w(
    feature_dim: int,
    config: RLAlgorithmConfig,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    if config.weight_init == "zeros":
        return np.zeros(feature_dim, dtype=float)
    if config.weight_init == "normal":
        return rng.normal(
            loc=config.weight_mean,
            scale=config.weight_std,
            size=feature_dim,
        ).astype(float)
    raise ValueError(f"未知 weight_init: {config.weight_init}")
