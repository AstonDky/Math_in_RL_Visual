from __future__ import annotations

import numpy as np

__all__ = ["mc_exploring_starts"]


def mc_exploring_starts(
    env,
    q,
    policy,
    returns,
    num,
    gamma=0.9,
    episodes=1,
    max_steps=100,
):
    for _ in range(episodes):
        s = int(np.random.choice(env.num_states))
        a = int(np.random.choice(env.num_actions))
        episode = []

        for _ in range(max_steps):
            s_next, r, done = env.transition(s, a)
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

            best_value = np.max(q[s])
            best_actions = np.flatnonzero(np.isclose(q[s], best_value))
            policy[s] = 0.0
            policy[s, best_actions] = 1.0 / len(best_actions)

    return q, policy, returns, num
