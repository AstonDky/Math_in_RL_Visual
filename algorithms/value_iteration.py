from __future__ import annotations

import numpy as np

__all__ = ["value_iteration"]


def value_iteration(
    env,
    v,
    policy,
    model,
    gamma=0.9,
    iterations=1,
    threshold=1e-8,
):
    for _ in range(iterations):
        v_old = v.copy()

        for s in env.states:
            q_values = np.zeros(env.num_actions, dtype=float)
            for a in env.actions:
                s_next, r, done = model.transition(s, a)
                q_values[a] = r if done else r + gamma * v_old[s_next]

            best_value = np.max(q_values)
            best_actions = np.flatnonzero(np.isclose(q_values, best_value))
            policy[s] = 0.0
            policy[s, best_actions] = 1.0 / len(best_actions)
            v[s] = best_value

        if np.linalg.norm(v - v_old) <= threshold:
            break

    return v, policy
