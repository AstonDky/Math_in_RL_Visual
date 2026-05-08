from __future__ import annotations

import numpy as np

__all__ = ["mc_epsilon_greedy"]


def _set_epsilon_greedy(policy, q, s, epsilon):
    n_actions = policy.shape[1]
    best_value = np.max(q[s])
    best_actions = np.flatnonzero(np.isclose(q[s], best_value))
    policy[s] = epsilon / n_actions
    policy[s, best_actions] += (1.0 - epsilon) / len(best_actions)


def mc_epsilon_greedy(
    env,
    q,
    policy,
    returns,
    num,
    gamma=0.9,
    epsilon=0.1,
    episodes=1,
    max_steps=100,
):
    for _ in range(episodes):
        s = env.reset()
        probs = policy[s] / np.sum(policy[s])
        a = int(np.random.choice(env.num_actions, p=probs))
        episode = []

        for _ in range(max_steps):
            s_next, r, done = env.step(a)
            episode.append((s, a, r))
            if done:
                break
            probs = policy[s_next] / np.sum(policy[s_next])
            a = int(np.random.choice(env.num_actions, p=probs))
            s = s_next

        g = 0.0
        for s, a, r in reversed(episode):
            g = gamma * g + r
            returns[s, a] += g
            num[s, a] += 1.0
            q[s, a] = returns[s, a] / num[s, a]
            _set_epsilon_greedy(policy, q, s, epsilon)

    return q, policy, returns, num
