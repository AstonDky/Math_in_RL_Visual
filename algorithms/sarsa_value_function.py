"""Algorithm 8.2: Sarsa with function approximation.

This file intentionally contains only the book-style algorithm function. The
framework supplies ``pi``, ``q_hat``, ``grad_q_hat`` and ``step`` through an
adapter, then records the intermediate update for UI/TensorBoard.
"""

from __future__ import annotations

import numpy as np


def sarsa_fa(
    w: np.ndarray,
    s0: int,
    pi,
    q_hat,
    grad_q_hat,
    step,
    alpha: float = 0.01,
    gamma: float = 0.9,
    episodes: int = 500,
    max_steps: int = 100,
) -> np.ndarray:
    for _ in range(episodes):
        s = s0
        a = pi(s, w)

        for _ in range(max_steps):
            s_next, r, done = step(s, a)

            if done:
                delta = r - q_hat(s, a, w)
                w = w + alpha * delta * grad_q_hat(s, a, w)
                break

            a_next = pi(s_next, w)

            delta = r + gamma * q_hat(s_next, a_next, w) - q_hat(s, a, w)
            w = w + alpha * delta * grad_q_hat(s, a, w)

            s = s_next
            a = a_next

    return w
