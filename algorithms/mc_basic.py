from __future__ import annotations

import numpy as np

__all__ = ["mc_basic"]


def _rollout_from(env, policy, s0, a0, gamma, max_steps):
    episode = []
    s = s0
    a = a0
    for _ in range(max_steps):
        s_next, r, done = env.transition(s, a)
        episode.append((s, a, r))
        if done:
            break
        probs = policy[s_next] / np.sum(policy[s_next])
        a = int(np.random.choice(env.num_actions, p=probs))
        s = s_next
    return episode


def mc_basic(
    env,
    q,
    policy,
    returns,
    num,
    gamma=0.9,
    iterations=1,
    max_steps=100,
):
    for _ in range(iterations):
        for s in env.states:
            for a in env.actions:
                episode = _rollout_from(env, policy, s, a, gamma, max_steps)
                g = 0.0
                for state, action, r in reversed(episode):
                    g = gamma * g + r
                    if state == s and action == a:
                        returns[state, action] += g
                        num[state, action] += 1.0
                        q[state, action] = returns[state, action] / num[state, action]
                        break

        for s in env.states:
            best_value = np.max(q[s])
            best_actions = np.flatnonzero(np.isclose(q[s], best_value))
            policy[s] = 0.0
            policy[s, best_actions] = 1.0 / len(best_actions)

    return q, policy, returns, num
