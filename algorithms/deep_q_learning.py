from __future__ import annotations

import numpy as np

__all__ = ["deep_q_learning"]


def _features(env, state, action):
    x = np.zeros(env.num_states + env.num_actions, dtype=float)
    x[state] = 1.0
    x[env.num_states + action] = 1.0
    return x


def _forward(x, w1, b1, w2, b2):
    z1 = x @ w1 + b1
    h = np.maximum(0.0, z1)
    y = float(h @ w2 + b2)
    return y, z1, h


def _q(env, state, action, w1, b1, w2, b2):
    y, _, _ = _forward(_features(env, state, action), w1, b1, w2, b2)
    return y


def deep_q_learning(
    env,
    w1,
    b1,
    w2,
    b2,
    target_w1,
    target_b1,
    target_w2,
    target_b2,
    replay_buffer,
    alpha=0.001,
    gamma=0.9,
    batch_size=32,
    target_update_interval=10,
    iterations=1,
    max_steps=100,
):
    for iteration in range(iterations):
        s = env.reset()
        for _ in range(max_steps):
            a = int(np.random.choice(env.num_actions))
            s_next, r, done = env.step(a)
            replay_buffer.append((s, a, r, s_next, done))
            if done:
                break
            s = s_next

        if len(replay_buffer) == 0:
            continue

        batch_count = min(batch_size, len(replay_buffer))
        sample_indices = np.random.choice(len(replay_buffer), size=batch_count, replace=False)
        for idx in sample_indices:
            s, a, r, s_next, done = replay_buffer[int(idx)]
            if done:
                y_target = r
            else:
                next_values = [
                    _q(env, s_next, a_next, target_w1, target_b1, target_w2, target_b2)
                    for a_next in env.actions
                ]
                y_target = r + gamma * max(next_values)

            x = _features(env, s, a)
            y, z1, h = _forward(x, w1, b1, w2, b2)
            error = y_target - y
            relu_grad = (z1 > 0.0).astype(float)

            old_w2 = w2.copy()
            w2 = w2 + alpha * error * h
            b2 = b2 + alpha * error
            hidden_delta = error * old_w2 * relu_grad
            w1 = w1 + alpha * np.outer(x, hidden_delta)
            b1 = b1 + alpha * hidden_delta

        if (iteration + 1) % target_update_interval == 0:
            target_w1 = w1.copy()
            target_b1 = b1.copy()
            target_w2 = w2.copy()
            target_b2 = b2

    return w1, b1, w2, b2, target_w1, target_b1, target_w2, target_b2, replay_buffer
