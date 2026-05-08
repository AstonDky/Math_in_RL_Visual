from __future__ import annotations

import numpy as np

__all__ = ["reinforce"]


def reinforce(
    env,
    theta,
    sample_from_policy,
    grad_log_policy,
    alpha=0.01,
    gamma=0.9,
    episodes=1,
    max_steps=100,
):
    for _ in range(episodes):
        s = env.reset()
        episode = []

        for _ in range(max_steps):
            a = sample_from_policy(s, theta)
            s_next, r, done = env.step(a)
            episode.append((s, a, r))
            if done:
                break
            s = s_next

        g = 0.0
        returns = []
        for _, _, r in reversed(episode):
            g = gamma * g + r
            returns.append(g)
        returns.reverse()

        for (s, a, _), return_t in zip(episode, returns):
            theta = theta + alpha * grad_log_policy(s, a, theta) * return_t

    return theta
