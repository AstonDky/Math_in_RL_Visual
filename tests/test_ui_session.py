"""Regression checks for preserving training sessions across UI actions."""

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QSettings

from algorithms import q_learning_off_policy
from core.algorithm_adapters import adapt_algorithm
from core.engine import TrainingEngine
from core.rl_parameters import RLAlgorithmConfig
from core.table_agent import TableAgent
from envs.grid_world import GridWorld
from ui.main_window import MainWindow
from utils.logger import TensorBoardLogger
from utils.session import TrainingSessionManager


class WindowSessionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.env = GridWorld()
        self.agent = TableAgent(
            num_states=self.env.num_states,
            num_actions=self.env.num_actions,
            reward_map=self.env.reward_map(),
            algorithm=adapt_algorithm(q_learning_off_policy),
            config=RLAlgorithmConfig(seed=7),
            algorithm_name="q_learning_off_policy",
            state_shape=self.env.layout.shape,
        )
        self.sessions = TrainingSessionManager(
            "q_learning_off_policy", root / "sessions", root / "runs"
        )
        self.agent.tables.q.fill(1.25)
        self.agent.tables.w.fill(1.25)
        self.sessions.save(self.agent, step=37, episode=4)
        self.engine = TrainingEngine(
            env=self.env,
            agent=self.agent,
            session_manager=self.sessions,
            logger=TensorBoardLogger(self.sessions.log_dir),
        )
        self.engine.set_progress(37, 4)
        self.window = MainWindow(
            self.env, self.engine,
            settings=QSettings(str(root / "ui.ini"), QSettings.Format.IniFormat),
        )

    def tearDown(self):
        self.window.close()
        self.window.deleteLater()
        self.app.processEvents()
        self.temp.cleanup()

    def assert_start_preserves_session(self):
        expected_q = self.agent.tables.q.copy()
        expected_progress = self.engine.progress()
        with patch.object(self.engine, "start") as start:
            self.window.play_button.click()
        start.assert_called_once()
        np.testing.assert_array_equal(self.agent.tables.q, expected_q)
        self.assertEqual(self.engine.progress(), expected_progress)
        self.assertIsNotNone(self.engine.logger)

    def test_start_preserves_prepared_startup_checkpoint(self):
        self.assert_start_preserves_session()
        self.assertTrue(np.any(self.agent.tables.q))

    def test_continue_then_start_preserves_loaded_progress(self):
        self.agent.tables.q.fill(0)
        self.engine.set_progress(0, 0)
        self.window.continue_button.click()
        self.assertEqual(self.engine.progress(), (37, 4))
        self.assertTrue(np.any(self.agent.tables.q))
        self.assert_start_preserves_session()

    def test_stop_then_start_reopens_logger_without_reset(self):
        self.window.stop_button.click()
        self.assertIsNone(self.engine.logger)
        self.assert_start_preserves_session()

    def test_close_preserves_current_and_other_algorithm_logs(self):
        paths = [
            self.sessions.log_dir / "retained.txt",
            self.sessions.runs_dir / "other_algorithm" / "retained.txt",
        ]
        for path in paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("training evidence", encoding="utf-8")
        self.window.close()
        for path in paths:
            self.assertEqual(path.read_text(encoding="utf-8"), "training evidence")
        self.assertEqual(self.sessions.load(self.agent), (37, 4))

    def test_explicit_restart_still_resets_current_session(self):
        self.window.restart_button.click()
        self.assertEqual(self.engine.progress(), (0, 0))
        np.testing.assert_array_equal(self.agent.tables.q, 0)


if __name__ == "__main__":
    unittest.main()
