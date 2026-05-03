"""书中 SARSA / on-policy Q-learning 核心算法函数。"""

from __future__ import annotations

from core.rl_parameters import RLAlgorithmContext, RLStepResult, RLTransition


def sarsa_step(
    ctx: RLAlgorithmContext,
    transition: RLTransition,
) -> RLStepResult:
    """SARSA 单步更新。只放书中伪代码对应的核心计算。"""

    next_action = ctx.sample_action(transition.next_state)
    old_q = float(ctx.tables.q[transition.state, transition.action])
    next_q = (
        0.0
        if transition.done
        else float(ctx.tables.q[transition.next_state, next_action])
    )
    td_target = transition.reward + ctx.config.gamma * next_q
    td_error = td_target - old_q
    new_q = old_q + ctx.config.alpha * td_error
    ctx.tables.q[transition.state, transition.action] = new_q
    ctx.tables.v[transition.state] = float(ctx.tables.q[transition.state].max())

    return RLStepResult(
        formula="Q(s,a) <- Q(s,a) + alpha [r + gamma Q(s',a') - Q(s,a)]",
        calculation=(
            f"Q({transition.state},{transition.action}) <- {old_q:.3f} + "
            f"{ctx.config.alpha:.3f} * ({transition.reward:.3f} + "
            f"{ctx.config.gamma:.3f} * Q({transition.next_state},{next_action}) "
            f"- {old_q:.3f}) = {new_q:.3f}"
        ),
        variables={
            "s": transition.state,
            "a": transition.action,
            "r": round(float(transition.reward), 3),
            "s_next": transition.next_state,
            "a_next": next_action,
            "Q_old": round(old_q, 3),
            "Q_next": round(next_q, 3),
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

