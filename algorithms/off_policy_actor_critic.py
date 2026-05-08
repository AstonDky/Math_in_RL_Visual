from __future__ import annotations

import numpy as np

__all__ = ["off_policy_actor_critic"]


def off_policy_actor_critic(
    env,
    theta,
    v,
    policy_probs,
    grad_log_policy,
    grad_v_hat,
    alpha_theta=0.01,
    alpha_w=0.1,
    gamma=0.9,
    episodes=1,
    max_steps=100,
):
    for _ in range(episodes):
        s = env.reset()

        for _ in range(max_steps):
            beta = np.full(env.num_actions, 1.0 / env.num_actions, dtype=float)
            a = int(np.random.choice(env.num_actions, p=beta))
            s_next, r, done = env.step(a)
            target_probs = policy_probs(s, theta)
            rho = target_probs[a] / max(beta[a], 1e-12)
            next_value = 0.0 if done else v[s_next]
            delta = r + gamma * next_value - v[s]
            theta = theta + alpha_theta * rho * delta * grad_log_policy(s, a, theta)
            v = v + alpha_w * rho * delta * grad_v_hat(s, v)
            if done:
                break
            s = s_next

    return theta, v
