from __future__ import annotations

import numpy as np

__all__ = ["policy_iteration"]


def policy_iteration(
    env,
    v,
    policy,
    model,
    gamma=0.9,
    iterations=1,
    eval_iterations=20,
    threshold=1e-8,
):
    for _ in range(iterations):
        for _ in range(eval_iterations):
            v_old = v.copy()
            for s in env.states:
                value = 0.0
                for a in env.actions:
                    s_next, r, done = model.transition(s, a)
                    target = r if done else r + gamma * v_old[s_next]
                    value += policy[s, a] * target
                v[s] = value
            if np.linalg.norm(v - v_old) <= threshold:
                break

        policy_old = policy.copy()
        for s in env.states:
            q_values = np.zeros(env.num_actions, dtype=float)
            for a in env.actions:
                s_next, r, done = model.transition(s, a)
                q_values[a] = r if done else r + gamma * v[s_next]
            best_value = np.max(q_values)
            best_actions = np.flatnonzero(np.isclose(q_values, best_value))
            policy[s] = 0.0
            policy[s, best_actions] = 1.0 / len(best_actions)

        if np.array_equal(policy, policy_old):
            break

    return v, policy
