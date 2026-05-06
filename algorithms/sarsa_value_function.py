"""Algorithm 8.2: Sarsa with linear action-value approximation.

The function keeps the book-style training loop. The framework supplies
``pi``, ``q_hat``, ``grad_q_hat`` and ``step`` when it runs the algorithm
inside the visual trainer.
"""

from __future__ import annotations

import numpy as np

__all__ = ["sarsa_fa"]


def sarsa_fa(
    w: np.ndarray,
    s0: int,
    pi,
    q_hat,
    grad_q_hat,
    step,
    alpha: float = 0.01,
    gamma: float = 0.9,
    epsilon: float = 0.1,
    episodes: int = 500,
    max_steps: int = 100,
) -> np.ndarray:
    for _ in range(episodes):
        s = s0
        a = pi(s, w, epsilon)

        for _ in range(max_steps):
            s_next, r, done = step(s, a)

            if done:
                delta = r - q_hat(s, a, w)
                w = w + alpha * delta * grad_q_hat(s, a, w)
                break

            a_next = pi(s_next, w, epsilon)

            delta = r + gamma * q_hat(s_next, a_next, w) - q_hat(s, a, w)
            w = w + alpha * delta * grad_q_hat(s, a, w)

            s = s_next
            a = a_next

    return w
