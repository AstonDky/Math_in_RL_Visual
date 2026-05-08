from __future__ import annotations

import numpy as np

__all__ = ["q_learning_on_policy"]


def _set_epsilon_greedy(policy, q, s, epsilon):
    n_actions = policy.shape[1]
    best_value = np.max(q[s])
    best_actions = np.flatnonzero(np.isclose(q[s], best_value))
    policy[s] = epsilon / n_actions
    policy[s, best_actions] += (1.0 - epsilon) / len(best_actions)


def q_learning_on_policy(
    env,
    q,
    policy,
    sample_from_policy,
    alpha=0.1,
    gamma=0.9,
    epsilon=0.1,
    episodes=1,
    max_steps=100,
):
    for _ in range(episodes):
        s = env.reset()

        for _ in range(max_steps):
            _set_epsilon_greedy(policy, q, s, epsilon)
            a = sample_from_policy(s, policy)
            s_next, r, done = env.step(a)
            target = r if done else r + gamma * np.max(q[s_next])
            td_error = target - q[s, a]
            q[s, a] = q[s, a] + alpha * td_error
            _set_epsilon_greedy(policy, q, s, epsilon)
            if done:
                break
            s = s_next

    return q, policy
