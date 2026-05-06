from __future__ import annotations

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


def q(s, a, w):
    return w[s, a]


def grad_log_policy(s, a, theta):
    grad = np.zeros_like(theta)
    probs = policy_probs(s, theta)

    grad[s, :] = -probs
    grad[s, a] += 1

    return grad


def grad_q(s, a, w):
    grad = np.zeros_like(w)
    grad[s, a] = 1
    return grad


def QAC(env, theta, w, alpha_theta, alpha_w, gamma, num_episodes):
    for _ in range(num_episodes):
        s = env.reset()
        a = sample_from_policy(s, theta)

        done = False
        while not done:
            s_next, r, done = env.step(a)

            if done:
                q_next = 0
                a_next = None
            else:
                a_next = sample_from_policy(s_next, theta)
                q_next = q(s_next, a_next, w)

            q_sa = q(s, a, w)

            theta = theta + alpha_theta * grad_log_policy(s, a, theta) * q_sa

            td_error = r + gamma * q_next - q_sa
            w = w + alpha_w * td_error * grad_q(s, a, w)

            s = s_next
            a = a_next

    return theta, w