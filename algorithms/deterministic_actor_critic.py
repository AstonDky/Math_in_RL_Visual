from __future__ import annotations

import numpy as np

__all__ = ["deterministic_actor_critic"]


def deterministic_actor_critic(
    env,
    theta,
    q,
    grad_q,
    alpha_theta=0.01,
    alpha_w=0.1,
    gamma=0.9,
    episodes=1,
    max_steps=100,
):
    for _ in range(episodes):
        s = env.reset()

        for _ in range(max_steps):
            a = int(np.random.choice(env.num_actions))
            s_next, r, done = env.step(a)
            mu_next = int(np.argmax(theta[s_next]))
            q_next = 0.0 if done else q[s_next, mu_next]
            delta = r + gamma * q_next - q[s, a]
            q = q + alpha_w * delta * grad_q(s, a, q)

            mu = int(np.argmax(theta[s]))
            theta_grad = np.zeros_like(theta)
            theta_grad[s, mu] = 1.0
            theta = theta + alpha_theta * q[s, mu] * theta_grad

            if done:
                break
            s = s_next

    return theta, q
