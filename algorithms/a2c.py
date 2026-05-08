from __future__ import annotations

__all__ = ["a2c"]
import numpy as np

def softmax(x):
    x = x - np.max(x)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)


def policy_probs(s, theta):
    return softmax(theta[s])


def sample_from_policy(s, theta):
    probs = policy_probs(s, theta)
    return np.random.choice(len(probs), p=probs)


def a2c(
    env,
    theta,
    v,
    sample_from_policy,
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
            a = sample_from_policy(s, theta)
            s_next, r, done = env.step(a)
            next_value = 0.0 if done else v[s_next]
            delta = r + gamma * next_value - v[s]
            theta = theta + alpha_theta * delta * grad_log_policy(s, a, theta)
            v = v + alpha_w * delta * grad_v_hat(s, v)
            if done:
                break
            s = s_next

    return theta, v
