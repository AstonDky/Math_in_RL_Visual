from __future__ import annotations

import numpy as np

__all__ = ["q_learning_fa"]


def _set_epsilon_greedy(policy, env, q_hat, w, s, epsilon):
    q_values = np.array([q_hat(s, a, w) for a in env.actions], dtype=float)
    best_value = np.max(q_values)
    best_actions = np.flatnonzero(np.isclose(q_values, best_value))
    policy[s] = epsilon / env.num_actions
    policy[s, best_actions] += (1.0 - epsilon) / len(best_actions)


def q_learning_fa(
    env,
    w,
    policy,
    sample_from_policy,
    q_hat,
    grad_q_hat,
    alpha=0.01,
    gamma=0.9,
    epsilon=0.1,
    episodes=1,
    max_steps=100,
):
    for _ in range(episodes):
        s = env.reset()

        for _ in range(max_steps):
            _set_epsilon_greedy(policy, env, q_hat, w, s, epsilon)
            a = sample_from_policy(s, policy)
            s_next, r, done = env.step(a)
            if done:
                target = r
            else:
                q_next = max(q_hat(s_next, a_next, w) for a_next in env.actions)
                target = r + gamma * q_next
            delta = target - q_hat(s, a, w)
            w = w + alpha * delta * grad_q_hat(s, a, w)
            _set_epsilon_greedy(policy, env, q_hat, w, s, epsilon)
            if done:
                break
            s = s_next

    return w, policy
