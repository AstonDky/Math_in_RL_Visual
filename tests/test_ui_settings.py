"""Presentation settings persist without changing algorithm state or user files."""

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PyQt6.QtCore import QSettings
from PyQt6.QtWidgets import QApplication, QDialogButtonBox

from algorithms import q_learning_off_policy
from core.algorithm_adapters import adapt_algorithm
from core.engine import TrainingEngine
from core.rl_parameters import RLAlgorithmConfig
from core.table_agent import TableAgent
from envs.grid_world import GridWorld
from ui.main_window import MainWindow
from ui.settings import InterfaceSettings
from ui.theme import THEMES


class InterfaceSettingsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.settings_path = Path(self.temp.name) / "interface.ini"
        self.store = QSettings(str(self.settings_path), QSettings.Format.IniFormat)
        self.env = GridWorld()
        self.agent = TableAgent(
            num_states=self.env.num_states, num_actions=self.env.num_actions,
            reward_map=self.env.reward_map(), algorithm=adapt_algorithm(q_learning_off_policy),
            config=RLAlgorithmConfig(seed=7), algorithm_name="q_learning_off_policy",
            state_shape=self.env.layout.shape,
        )
        self.engine = TrainingEngine(self.env, self.agent, delay_ms=12.5)
        self.window = MainWindow(self.env, self.engine, settings=self.store)

    def tearDown(self):
        self.window.close()
        self.window.deleteLater()
        self.app.processEvents()
        self.temp.cleanup()

    def open_settings(self):
        self.window.settings_button.click()
        return self.window._settings_dialog

    def test_defaults_keep_engine_interval_and_light_theme(self):
        self.assertEqual(self.window.preferences, InterfaceSettings(delay_ms=12.5))
        self.assertEqual(self.engine.delay_ms, 12.5)
        self.assertFalse(self.settings_path.exists())

    def test_apply_persists_and_themes_every_painter_without_reset(self):
        self.agent.tables.q.fill(2.5)
        self.engine.set_progress(17, 3)
        dialog = self.open_settings()
        dialog.theme_input.setCurrentIndex(dialog.theme_input.findData("dark"))
        dialog.font_input.setValue(12)
        dialog.interval_input.setValue(46.25)
        dialog.tensorboard_input.setChecked(False)
        dialog.buttons.button(QDialogButtonBox.StandardButton.Apply).click()
        expected = InterfaceSettings("dark", 12, 46.25, False)
        self.assertEqual(InterfaceSettings.load(self.store), expected)
        self.assertEqual(self.window.interval_input.value(), 46.25)
        self.assertEqual(self.engine.delay_ms, 46.25)
        self.assertEqual(self.engine.progress(), (17, 3))
        np.testing.assert_array_equal(self.agent.tables.q, 2.5)
        for painter in (
            self.window.grid_painter, self.window.monitor_panel.reward_grid,
            self.window.monitor_panel.policy_painter,
        ):
            self.assertEqual(painter.theme, THEMES["dark"])
        reopened_store = QSettings(str(self.settings_path), QSettings.Format.IniFormat)
        self.assertEqual(InterfaceSettings.load(reopened_store), expected)

    def test_cancel_discards_draft_and_reopen_uses_current_values(self):
        dialog = self.open_settings()
        dialog.font_input.setValue(14)
        dialog.theme_input.setCurrentIndex(1)
        dialog.buttons.button(QDialogButtonBox.StandardButton.Cancel).click()
        self.assertEqual(self.window.preferences, InterfaceSettings(delay_ms=12.5))
        self.assertEqual(self.open_settings().preferences(), self.window.preferences)

    def test_main_interval_control_persists_and_updates_settings_dialog(self):
        dialog = self.open_settings()
        self.window.interval_input.setValue(71.5)
        self.assertEqual(dialog.interval_input.value(), 71.5)
        self.assertEqual(InterfaceSettings.load(self.store).delay_ms, 71.5)
        self.assertEqual(self.engine.delay_ms, 71.5)

    def test_restore_defaults_is_a_draft_until_applied_and_preserves_progress(self):
        self.window._apply_preferences(InterfaceSettings("dark", 14, 25, False))
        self.engine.set_progress(99, 8)
        self.agent.tables.q.fill(3)
        dialog = self.open_settings()
        dialog.buttons.button(QDialogButtonBox.StandardButton.RestoreDefaults).click()
        self.assertEqual(dialog.preferences(), InterfaceSettings())
        self.assertEqual(self.window.preferences.theme, "dark")
        dialog.buttons.button(QDialogButtonBox.StandardButton.Ok).click()
        self.assertEqual(self.window.preferences, InterfaceSettings())
        self.assertEqual(self.engine.progress(), (99, 8))
        np.testing.assert_array_equal(self.agent.tables.q, 3)

    def test_auto_tensorboard_setting_controls_start(self):
        tensorboard = Mock()
        self.window.tensorboard = tensorboard
        self.window._apply_preferences(InterfaceSettings(auto_tensorboard=False))
        with patch.object(self.engine, "start"):
            self.window.play_button.click()
        tensorboard.start.assert_not_called()
        self.window._apply_preferences(InterfaceSettings(auto_tensorboard=True))
        with patch.object(self.engine, "start"):
            self.window.play_button.click()
        tensorboard.start.assert_called_once()

    def test_window_and_splitters_restore_from_same_store(self):
        self.window.show()
        self.window.resize(1100, 760)
        self.app.processEvents()
        self.window.main_splitter.setSizes([360, 680])
        self.window.monitor_panel.inspector_splitter.setSizes([200, 140, 150])
        self.window.close()
        main_state = self.store.value("window/main_splitter")
        detail_state = self.store.value("window/inspector_splitter")
        self.assertTrue(main_state)
        self.assertTrue(detail_state)
        replacement = MainWindow(self.env, self.engine, settings=self.store)
        self.assertEqual(replacement.main_splitter.saveState(), main_state)
        self.assertEqual(replacement.monitor_panel.inspector_splitter.saveState(), detail_state)
        replacement.close()
        replacement.deleteLater()

    def test_invalid_stored_values_use_safe_interface_defaults(self):
        self.store.setValue("interface/theme", "unknown")
        self.store.setValue("interface/font_size", "oops")
        self.store.setValue("interface/delay_ms", "nan")
        self.assertEqual(InterfaceSettings.load(self.store), InterfaceSettings())


if __name__ == "__main__":
    unittest.main()
