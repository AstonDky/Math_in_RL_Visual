"""书中风格算法函数的框架适配层。"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

import numpy as np

from core.env_base import Action, State
from core.rl_parameters import (
    CoreAlgorithm,
    RLAlgorithmContext,
    RLStepResult,
    RLTransition,
    make_policy_probs,
)


BookAlgorithm = Callable[..., Any]
_SARSA_FA_REQUIRED_PARAMS = ("w", "s0", "pi", "q_hat", "grad_q_hat", "step")
_SARSA_FA_SUPPORTED_OPTIONAL_PARAMS = {
    "alpha",
    "gamma",
    "epsilon",
    "episodes",
    "max_steps",
}


def adapt_algorithm(book_algorithm: BookAlgorithm) -> CoreAlgorithm:
    """按函数签名选择适配器。"""

    if _matches_sarsa_fa_signature(book_algorithm):
        return adapt_sarsa_fa(book_algorithm)

    signature = inspect.signature(book_algorithm)
    raise ValueError(
        "No framework adapter matches algorithm "
        f"{book_algorithm.__name__}{signature}. "
        "Add a new signature matcher in core/algorithm_adapters.py instead of "
        "changing engine/UI/session code."
    )


def adapt_sarsa_fa(book_algorithm: BookAlgorithm) -> CoreAlgorithm:
    """适配 ``sarsa_fa(w, s0, pi, q_hat, grad_q_hat, step, ...)`` 族函数。

    适配器把当前 transition 包装成算法看到的 ``step``，并运行一次可展示的
    权重更新。环境推进、UI 数据和 checkpoint 仍由框架层处理。
    """

    parameter_names = tuple(inspect.signature(book_algorithm).parameters)

    def core_algorithm(
        ctx: RLAlgorithmContext,
        transition: RLTransition,
    ) -> RLStepResult:
        w_t = ctx.tables.w.copy()
        first_action_used = False
        sampled_next_action: Action | None = None

        def pi(
            state: State,
            w: np.ndarray,
            epsilon: float | None = None,
        ) -> Action:
            nonlocal first_action_used, sampled_next_action
            if not first_action_used and state == transition.state:
                first_action_used = True
                return transition.action

            effective_epsilon = (
                ctx.config.epsilon if epsilon is None else float(epsilon)
            )
            q_values = _q_values_from_w(ctx, state, w)
            probs = make_policy_probs(
                q_values,
                ctx.config.behavior_policy,
                effective_epsilon,
                ctx.config.softmax_temperature,
            )
            action = int(ctx.rng.choice(ctx.num_actions, p=probs))
            sampled_next_action = action
            return action

        def q_hat(state: State, action: Action, w: np.ndarray) -> float:
            return float(np.dot(ctx.tables.features[state, action], w))

        def grad_q_hat(state: State, action: Action, w: np.ndarray) -> np.ndarray:
            _ = w
            return ctx.tables.features[state, action].copy()

        def step(state: State, action: Action) -> tuple[State, float, bool]:
            _ = state, action
            return transition.next_state, transition.reward, transition.done

        q_hat_t = q_hat(transition.state, transition.action, w_t)
        call_kwargs = _build_supported_kwargs(ctx, parameter_names)
        w_next = book_algorithm(
            w_t,
            transition.state,
            pi,
            q_hat,
            grad_q_hat,
            step,
            **call_kwargs,
        )
        q_hat_t_plus_1 = (
            0.0
            if transition.done or sampled_next_action is None
            else q_hat(transition.next_state, sampled_next_action, w_t)
        )
        td_error = (
            transition.reward
            + ctx.config.gamma * q_hat_t_plus_1
            - q_hat_t
        )

        ctx.tables.w = np.asarray(w_next, dtype=float)
        ctx.sync_q_from_w()
        ctx.refresh_all_policies()
        pi_t_plus_1 = ctx.tables.policy[transition.state].copy()
        q_hat_after = ctx.q_hat(transition.state, transition.action)
        state_value_after = ctx.tables.v[transition.state]

        return RLStepResult(
            formula=(
                "w = w + alpha * [r + gamma*q_hat(s_next,a_next,w) "
                "- q_hat(s,a,w)] * grad_q_hat(s,a,w)"
            ),
            calculation=(
                f"delta = {transition.reward:.3f} + {ctx.config.gamma:.3f} * "
                f"{q_hat_t_plus_1:.3f} - {q_hat_t:.3f} = {td_error:.3f}; "
                f"q_hat({transition.state},{transition.action}) -> "
                f"{q_hat_after:.3f}"
            ),
            variables={
                "s": transition.state,
                "a": transition.action,
                "r": round(float(transition.reward), 3),
                "s_next": transition.next_state,
                "a_next": sampled_next_action,
                "q_hat": round(float(q_hat_t), 3),
                "q_hat_next": round(float(q_hat_t_plus_1), 3),
                "delta": round(float(td_error), 3),
                "w_norm": round(float(np.linalg.norm(ctx.tables.w)), 3),
                "epsilon": round(float(ctx.config.epsilon), 3),
                "pi(a|s)": np.round(pi_t_plus_1, 3).tolist(),
                "V_pi(s)": round(float(state_value_after), 3),
            },
            metrics={
                "rollout/reward": float(transition.reward),
                "train/loss": float(td_error * td_error),
                "train/td_error": float(td_error),
                "train/q_value": float(q_hat_after),
                "train/state_value": float(state_value_after),
                "train/w_norm": float(np.linalg.norm(ctx.tables.w)),
            },
            updated_state=transition.state,
            next_action=sampled_next_action,
        )

    core_algorithm.__name__ = book_algorithm.__name__
    core_algorithm.__doc__ = book_algorithm.__doc__
    core_algorithm.__wrapped__ = book_algorithm
    return core_algorithm


def _matches_sarsa_fa_signature(book_algorithm: BookAlgorithm) -> bool:
    signature = inspect.signature(book_algorithm)
    parameters = signature.parameters
    names = tuple(parameters)
    if names[: len(_SARSA_FA_REQUIRED_PARAMS)] != _SARSA_FA_REQUIRED_PARAMS:
        return False

    for name, parameter in list(parameters.items())[len(_SARSA_FA_REQUIRED_PARAMS) :]:
        if (
            name not in _SARSA_FA_SUPPORTED_OPTIONAL_PARAMS
            and parameter.default is inspect.Signature.empty
        ):
            return False
    return True


def _build_supported_kwargs(
    ctx: RLAlgorithmContext,
    parameter_names: tuple[str, ...],
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if "alpha" in parameter_names:
        kwargs["alpha"] = ctx.config.alpha
    if "gamma" in parameter_names:
        kwargs["gamma"] = ctx.config.gamma
    if "epsilon" in parameter_names:
        kwargs["epsilon"] = ctx.config.epsilon
    if "episodes" in parameter_names:
        kwargs["episodes"] = 1
    if "max_steps" in parameter_names:
        kwargs["max_steps"] = 1
    return kwargs


def _q_values_from_w(
    ctx: RLAlgorithmContext,
    state: State,
    w: np.ndarray,
) -> np.ndarray:
    return np.array(
        [
            np.dot(ctx.tables.features[state, action], w)
            for action in range(ctx.num_actions)
        ],
        dtype=float,
    )
