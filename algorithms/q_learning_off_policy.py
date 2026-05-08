from __future__ import annotations

import numpy as np

__all__ = ["q_learning_off_policy"]


def q_learning_off_policy(
    env,
    q,
    policy,
    alpha=0.1,
    gamma=0.9,
    episodes=1,
    max_steps=100,
):
    for _ in range(episodes):
        s = env.reset()

        for _ in range(max_steps):
            a = int(np.random.choice(env.num_actions))
            s_next, r, done = env.step(a)
            target = r if done else r + gamma * np.max(q[s_next])
            td_error = target - q[s, a]
            q[s, a] = q[s, a] + alpha * td_error

            best_value = np.max(q[s])
            best_actions = np.flatnonzero(np.isclose(q[s], best_value))
            policy[s] = 0.0
            policy[s, best_actions] = 1.0 / len(best_actions)

            if done:
                break
            s = s_next

    return q, policy
