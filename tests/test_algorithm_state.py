"""Short regressions for algorithm state and checkpoint restoration."""

import unittest

import numpy as np

import algorithms
from core.algorithm_adapters import adapt_algorithm
from core.rl_parameters import RLAlgorithmConfig, policy_from_preferences
from core.table_agent import TableAgent
from envs.grid_world import GridWorld


def make_agent(function, **config_overrides):
    env = GridWorld(layout=np.array([[0, 2]]), r_boundary=-1.0, r_target=2.0)
    config = RLAlgorithmConfig(
        alpha=0.1, max_steps=2, hidden_units=4, batch_size=2,
        target_update_interval=2, **config_overrides,
    )
    agent = TableAgent(
        env.num_states, env.num_actions, env.reward_map(),
        adapt_algorithm(function), config, function.__name__,
        state_shape=env.layout.shape,
    )
    return env, agent


class AlgorithmStateTests(unittest.TestCase):
    def test_tabular_q_update_survives_visual_sync(self):
        env, agent = make_agent(algorithms.q_learning_off_policy)
        batch = agent.run_episode(env, 0, 0, 1)
        sample = batch.infos[0]
        self.assertAlmostEqual(
            agent.tables.q[sample["state"], sample["action"]],
            agent.config.alpha * sample["reward"],
        )
        self.assertGreater(np.linalg.norm(agent.tables.q), 0.0)

    def test_direct_state_values_survive_policy_updates_and_restore(self):
        for function in (
            algorithms.value_iteration, algorithms.policy_iteration,
            algorithms.truncated_policy_iteration, algorithms.a2c,
            algorithms.off_policy_actor_critic, algorithms.td_value_fa,
        ):
            with self.subTest(algorithm=function.__name__):
                env, agent = make_agent(function)
                agent.run_episode(env, 0, 0, 2)
                values = agent.tables.v.copy()
                self.assertGreater(np.linalg.norm(values), 0.0)
                agent.context.rebuild_visual_state({"policy"})
                np.testing.assert_array_equal(agent.tables.v, values)
                _, restored = make_agent(function)
                restored.load_state_dict(agent.state_dict())
                np.testing.assert_array_equal(restored.tables.v, values)

    def test_value_iteration_keeps_bellman_result(self):
        env, agent = make_agent(algorithms.value_iteration)
        agent.run_episode(env, 0, 0, 2)
        self.assertEqual(agent.tables.v[0], 2.0)

    def test_actor_policy_survives_critic_only_updates(self):
        env, agent = make_agent(algorithms.QAC)
        agent.run_episode(env, 0, 0, 2)
        agent.context.rebuild_visual_state({"w"})
        np.testing.assert_allclose(
            agent.tables.policy, policy_from_preferences(agent.tables.theta),
        )

    def test_dqn_target_starts_as_online_copy(self):
        _, agent = make_agent(algorithms.deep_q_learning)
        for name in ("w1", "b1", "w2", "b2"):
            np.testing.assert_array_equal(
                agent.context.get_runtime_state("target_" + name),
                agent.context.get_runtime_state(name),
            )

    def test_dqn_target_interval_survives_calls_and_restore(self):
        env, agent = make_agent(algorithms.deep_q_learning)
        agent.run_episode(env, 0, 0, 2)
        self.assertEqual(agent.tables.extras["iteration_offset"], 1)
        self.assertGreater(np.linalg.norm(agent.tables.q), 0.0)
        self.assertFalse(np.array_equal(
            agent.tables.extras["w1"], agent.tables.extras["target_w1"],
        ))
        restored_env, restored = make_agent(algorithms.deep_q_learning)
        restored.load_state_dict(agent.state_dict())
        agent.run_episode(env, 2, 1, 2)
        restored.run_episode(restored_env, 2, 1, 2)
        self.assertEqual(restored.tables.extras["iteration_offset"], 2)
        for name in ("w1", "b1", "w2", "b2"):
            np.testing.assert_array_equal(
                restored.tables.extras[name], restored.tables.extras["target_" + name],
            )
            np.testing.assert_array_equal(
                restored.tables.extras[name], agent.tables.extras[name],
            )
        np.testing.assert_array_equal(restored.tables.q, agent.tables.q)

    def test_all_exported_algorithms_run_short_finite_batches(self):
        self.assertEqual(len(algorithms.__all__), 18)
        for name in algorithms.__all__:
            with self.subTest(algorithm=name):
                value_function = "linear" if name in {"sarsa_fa", "q_learning_fa"} else "table"
                env, agent = make_agent(getattr(algorithms, name), value_function=value_function)
                batch = agent.run_episode(env, 0, 0, 2)
                self.assertTrue(batch.infos)
                for values in (agent.tables.q, agent.tables.v, agent.tables.policy,
                               agent.tables.w, agent.tables.theta):
                    self.assertTrue(np.all(np.isfinite(values)))
                np.testing.assert_allclose(agent.tables.policy.sum(axis=1), 1.0)


if __name__ == "__main__":
    unittest.main()
