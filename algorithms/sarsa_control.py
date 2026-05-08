from __future__ import annotations

import numpy as np

__all__ = ["sarsa_control"]


def _set_epsilon_greedy(policy, q, s, epsilon):
    n_actions = policy.shape[1]
    best_value = np.max(q[s])
    best_actions = np.flatnonzero(np.isclose(q[s], best_value))
    policy[s] = epsilon / n_actions
    policy[s, best_actions] += (1.0 - epsilon) / len(best_actions)


def sarsa_control(
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
        _set_epsilon_greedy(policy, q, s, epsilon)
        a = sample_from_policy(s, policy)

        for _ in range(max_steps):
            s_next, r, done = env.step(a)
            if done:
                td_error = r - q[s, a]
                q[s, a] = q[s, a] + alpha * td_error
                _set_epsilon_greedy(policy, q, s, epsilon)
                break

            _set_epsilon_greedy(policy, q, s_next, epsilon)
            a_next = sample_from_policy(s_next, policy)
            td_error = r + gamma * q[s_next, a_next] - q[s, a]
            q[s, a] = q[s, a] + alpha * td_error
            _set_epsilon_greedy(policy, q, s, epsilon)
            s = s_next
            a = a_next

    return q, policy
