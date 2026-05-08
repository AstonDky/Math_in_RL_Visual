from __future__ import annotations

import numpy as np

__all__ = ["td_value_fa"]


def td_value_fa(
    env,
    v_w,
    policy,
    v_hat,
    grad_v_hat,
    alpha=0.1,
    gamma=0.9,
    episodes=1,
    max_steps=100,
):
    for _ in range(episodes):
        s = env.reset()

        for _ in range(max_steps):
            probs = policy[s] / np.sum(policy[s])
            a = int(np.random.choice(env.num_actions, p=probs))
            s_next, r, done = env.step(a)
            next_value = 0.0 if done else v_hat(s_next, v_w)
            delta = r + gamma * next_value - v_hat(s, v_w)
            v_w = v_w + alpha * delta * grad_v_hat(s, v_w)
            if done:
                break
            s = s_next

    return v_w
