"""Adapters that connect book-style algorithms to the visual framework."""

from __future__ import annotations

import numpy as np

from core.env_base import Action, State
from core.rl_parameters import (
    RLAlgorithmContext,
    RLStepResult,
    RLTransition,
    make_policy_probs,
)


def adapt_sarsa_fa(book_algorithm):
    """Wrap ``sarsa_fa(w, s0, pi, q_hat, grad_q_hat, step, ...)``.

    The engine already owns environment stepping so it can animate the GridWorld.
    This adapter therefore replays the current transition as the algorithm's
    ``step`` function and runs the book algorithm for exactly one update.
    """

    def core_algorithm(
        ctx: RLAlgorithmContext,
        transition: RLTransition,
    ) -> RLStepResult:
        w_t = ctx.tables.w.copy()
        first_action_used = False
        sampled_next_action: Action | None = None

        def pi(state: State, w: np.ndarray) -> Action:
            nonlocal first_action_used, sampled_next_action
            if not first_action_used and state == transition.state:
                first_action_used = True
                return transition.action

            q_values = _q_values_from_w(ctx, state, w)
            probs = make_policy_probs(
                q_values,
                ctx.config.behavior_policy,
                ctx.config.epsilon,
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
        w_next = book_algorithm(
            w_t,
            transition.state,
            pi,
            q_hat,
            grad_q_hat,
            step,
            ctx.config.alpha,
            ctx.config.gamma,
            1,
            1,
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
        ctx.refresh_policy_state(transition.state)
        pi_t_plus_1 = ctx.tables.policy[transition.state].copy()
        q_hat_after = ctx.q_hat(transition.state, transition.action)

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
                "pi(a|s)": np.round(pi_t_plus_1, 3).tolist(),
            },
            metrics={
                "rollout/reward": float(transition.reward),
                "train/loss": float(td_error * td_error),
                "train/td_error": float(td_error),
                "train/q_value": float(q_hat_after),
                "train/w_norm": float(np.linalg.norm(ctx.tables.w)),
            },
            updated_state=transition.state,
            next_action=sampled_next_action,
        )

    core_algorithm.__name__ = book_algorithm.__name__
    core_algorithm.__doc__ = book_algorithm.__doc__
    core_algorithm.__wrapped__ = book_algorithm
    return core_algorithm


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
