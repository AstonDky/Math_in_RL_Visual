"""书中 Q-learning 核心算法函数。"""

from __future__ import annotations

import numpy as np

from core.rl_parameters import RLAlgorithmContext, RLStepResult, RLTransition


def q_learning_step(
    ctx: RLAlgorithmContext,
    transition: RLTransition,
) -> RLStepResult:
    """Q-learning 单步更新。只放书中伪代码对应的核心计算。"""

    old_q = float(ctx.tables.q[transition.state, transition.action])
    next_q = 0.0 if transition.done else float(np.max(ctx.tables.q[transition.next_state]))
    td_target = transition.reward + ctx.config.gamma * next_q
    td_error = td_target - old_q
    new_q = old_q + ctx.config.alpha * td_error
    ctx.tables.q[transition.state, transition.action] = new_q
    ctx.tables.v[transition.state] = float(np.max(ctx.tables.q[transition.state]))

    return RLStepResult(
        formula="Q(s,a) <- Q(s,a) + alpha [r + gamma max_a Q(s',a) - Q(s,a)]",
        calculation=(
            f"Q({transition.state},{transition.action}) <- {old_q:.3f} + "
            f"{ctx.config.alpha:.3f} * ({transition.reward:.3f} + "
            f"{ctx.config.gamma:.3f} * {next_q:.3f} - {old_q:.3f}) = "
            f"{new_q:.3f}"
        ),
        variables={
            "s": transition.state,
            "a": transition.action,
            "r": round(float(transition.reward), 3),
            "s_next": transition.next_state,
            "Q_old": round(old_q, 3),
            "max_Q_next": round(next_q, 3),
            "td_target": round(float(td_target), 3),
            "td_error": round(float(td_error), 3),
            "Q_new": round(float(new_q), 3),
            "epsilon": round(float(ctx.config.epsilon), 4),
        },
        metrics={
            "rollout/reward": float(transition.reward),
            "train/loss": float(td_error * td_error),
            "train/td_error": float(td_error),
            "train/q_value": float(new_q),
        },
        updated_state=transition.state,
    )

