"""Main application window."""

from __future__ import annotations

from dataclasses import replace

from PyQt6.QtCore import QByteArray, QSettings, Qt
from PyQt6.QtGui import QFont, QGuiApplication
from PyQt6.QtWidgets import (
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QStatusBar,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from core.engine import TrainingEngine
from envs.grid_world import GridWorld
from ui.grid_painter import GridPainter
from ui.monitor_panel import MonitorPanel
from ui.settings import InterfaceSettings, SettingsDialog
from ui.theme import THEMES, palette_for, stylesheet
from utils.logger import TensorBoardLogger
from utils.tensorboard import TensorBoardLauncher


class MainWindow(QMainWindow):
    """Composes the controls, GridWorld view, and monitoring panels."""

    def __init__(
        self,
        env: GridWorld,
        engine: TrainingEngine,
        tensorboard: TensorBoardLauncher | None = None,
        settings: QSettings | None = None,
    ) -> None:
        super().__init__()
        self.env = env
        self.engine = engine
        self.tensorboard = tensorboard
        self.settings_store = settings if settings is not None else QSettings(
            "MathInRLVisual", "MathInRLVisual"
        )
        self.preferences = InterfaceSettings.load(self.settings_store, engine.delay_ms)
        self._settings_dialog: SettingsDialog | None = None
        self._logger_settings = None
        if engine.logger is not None:
            self._logger_settings = {
                "log_interval_steps": engine.logger.log_interval_steps,
                "flush_interval_steps": engine.logger.flush_interval_steps,
                "flush_interval_seconds": engine.logger.flush_interval_seconds,
            }

        self.setWindowTitle("Math in RL Visual")
        self.resize(1480, 930)
        self.setMinimumSize(1000, 700)

        self.grid_painter = GridPainter(layout=env.layout)
        self.grid_painter.set_reward_map(env.reward_map())
        self.monitor_panel = MonitorPanel(layout=env.layout)
        if engine.logger is not None and tensorboard is not None:
            self.monitor_panel.set_tensorboard_cadence(
                log_interval_steps=engine.logger.log_interval_steps,
                flush_interval_steps=engine.logger.flush_interval_steps,
                flush_interval_seconds=engine.logger.flush_interval_seconds,
                reload_interval_seconds=tensorboard.reload_interval_seconds,
            )
        self.play_button = QPushButton("开始")
        self.pause_button = QPushButton("暂停")
        self.stop_button = QPushButton("停止")
        self.restart_button = QPushButton("重新运行")
        self.continue_button = QPushButton("继续上次")
        self.settings_button = QPushButton("设置")
        self.play_button.setObjectName("primaryAction")
        self.play_button.setToolTip("开始或继续当前训练，保留已学习的参数 (F5)")
        self.play_button.setShortcut("F5")
        self.pause_button.setToolTip("暂停训练，保留当前进度 (F6)")
        self.pause_button.setShortcut("F6")
        self.stop_button.setToolTip("保存并停止当前训练 (F7)")
        self.stop_button.setShortcut("F7")
        self.restart_button.setToolTip("清空当前算法的训练进度，准备重新运行")
        self.continue_button.setToolTip("加载当前算法上次保存的训练进度")
        self.settings_button.setToolTip("设置主题、字号、显示间隔和 TensorBoard (Ctrl+,)")
        self.settings_button.setShortcut("Ctrl+,")
        self.step_label = QLabel("训练步  —")
        self.state_label = QLabel("状态  —    奖励  —")
        self.state_label.setWordWrap(True)
        self.interval_label = QLabel("显示间隔")
        self.interval_input = QDoubleSpinBox()
        self.interval_input.setDecimals(3)
        self.interval_input.setRange(0.0, 1000.0)
        self.interval_input.setSingleStep(1.0)
        self.interval_input.setSuffix(" ms")
        self.interval_input.setValue(self.preferences.delay_ms)
        self.interval_input.setToolTip("每次训练更新的等待时间；0 ms 为快速训练")
        self.interval_label.setBuddy(self.interval_input)
        self.interval_input.setKeyboardTracking(False)

        self._build_layout()
        self._connect_signals()
        self._apply_preferences(self.preferences, persist=False)
        self._restore_layout()

    def closeEvent(self, event) -> None:
        self.settings_store.setValue("window/geometry", self.saveGeometry())
        self.settings_store.setValue("window/main_splitter", self.main_splitter.saveState())
        self.settings_store.setValue(
            "window/inspector_splitter", self.monitor_panel.inspector_splitter.saveState()
        )
        self.settings_store.sync()
        self._stop_all()
        super().closeEvent(event)

    def _build_layout(self) -> None:
        title = QLabel("Math in RL Visual")
        title.setObjectName("appTitle")
        self.algorithm_label = QLabel(str(self.engine.agent.algorithm_name))
        self.algorithm_label.setObjectName("algorithmName")
        self.algorithm_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        header = QHBoxLayout()
        header.addWidget(title)
        header.addSpacing(14)
        header.addWidget(self.algorithm_label)
        header.addStretch()
        header.addWidget(self.settings_button)

        controls = QHBoxLayout()
        controls.setSpacing(8)
        controls.addWidget(self.play_button)
        controls.addWidget(self.pause_button)
        controls.addWidget(self.stop_button)
        controls.addWidget(self.restart_button)
        controls.addWidget(self.continue_button)
        controls.addStretch()
        controls.addWidget(self.interval_label)
        controls.addWidget(self.interval_input)

        environment_title = QLabel("GridWorld 环境")
        environment_title.setObjectName("sectionTitle")
        self.environment_note = QLabel(
            f"{self.env.layout.shape[0]} × {self.env.layout.shape[1]} 网格  ·  "
            f"{self.env.num_actions} 个动作  ·  格内数字为即时奖励"
        )
        self.environment_note.setObjectName("muted")
        self.environment_note.setWordWrap(True)
        left_layout = QVBoxLayout()
        left_layout.setContentsMargins(18, 18, 18, 16)
        left_layout.setSpacing(10)
        left_layout.addWidget(environment_title)
        left_layout.addWidget(self.environment_note)
        left_layout.addWidget(self.grid_painter, stretch=1)
        self.legend_label = QLabel()
        self.legend_label.setWordWrap(True)
        left_layout.addWidget(self.legend_label)
        left_layout.addSpacing(8)
        left_layout.addWidget(self.step_label)
        left_layout.addWidget(self.state_label)
        environment_panel = QFrame()
        environment_panel.setObjectName("environmentPanel")
        environment_panel.setLayout(left_layout)
        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.main_splitter.setChildrenCollapsible(False)
        self.main_splitter.setHandleWidth(14)
        self.main_splitter.addWidget(environment_panel)
        self.main_splitter.addWidget(self.monitor_panel)
        self.main_splitter.setStretchFactor(0, 2)
        self.main_splitter.setStretchFactor(1, 3)
        self.main_splitter.setSizes([560, 850])

        root_layout = QVBoxLayout()
        root_layout.setContentsMargins(20, 14, 20, 8)
        root_layout.setSpacing(12)
        root_layout.addLayout(header)
        root_layout.addLayout(controls)
        root_layout.addWidget(self.main_splitter, 1)

        central = QWidget()
        central.setObjectName("appRoot")
        central.setLayout(root_layout)
        self.setCentralWidget(central)
        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("就绪 · 开始训练后可检查价值、策略与每次更新的计算")

    def _restore_layout(self) -> None:
        geometry = self.settings_store.value("window/geometry")
        if isinstance(geometry, QByteArray):
            self.restoreGeometry(geometry)
        screen = self.screen() or QGuiApplication.primaryScreen()
        if screen is not None:
            available = screen.availableGeometry()
            self.resize(min(self.width(), available.width()), min(self.height(), available.height()))
            frame = self.frameGeometry()
            frame.moveLeft(max(available.left(), min(frame.left(), available.right() - frame.width() + 1)))
            frame.moveTop(max(available.top(), min(frame.top(), available.bottom() - frame.height() + 1)))
            self.move(frame.topLeft())
        for key, splitter in (
            ("main_splitter", self.main_splitter),
            ("inspector_splitter", self.monitor_panel.inspector_splitter),
        ):
            state = self.settings_store.value(f"window/{key}")
            if isinstance(state, QByteArray):
                splitter.restoreState(state)

    def _apply_preferences(self, preferences: InterfaceSettings, persist: bool = True) -> None:
        self.preferences = preferences
        theme = THEMES[preferences.theme]
        font = QFont(self.font())
        font.setPointSize(preferences.font_size)
        self.setFont(font)
        self.setPalette(palette_for(theme))
        self.setStyleSheet(stylesheet(theme, preferences.font_size))
        self.interval_input.setFixedWidth(115 + (preferences.font_size - 10) * 10)
        self.grid_painter.set_theme(theme)
        self.monitor_panel.set_theme(theme, preferences.font_size)
        self.legend_label.setText(
            f'<span style="color:{theme.accent}; font-weight:600;">蓝框：当前位置</span>'
            f'　<span style="color:{theme.text};">琥珀：禁区　青绿：目标</span>'
        )
        self.interval_input.blockSignals(True)
        self.interval_input.setValue(preferences.delay_ms)
        self.interval_input.blockSignals(False)
        self.engine.set_delay(preferences.delay_ms)
        if persist:
            preferences.save(self.settings_store)

    def _open_settings(self) -> None:
        if self._settings_dialog is None:
            self._settings_dialog = SettingsDialog(
                self.preferences, self._apply_preferences, self
            )
        else:
            self._settings_dialog.set_preferences(self.preferences)
        self._settings_dialog.open()

    def _connect_signals(self) -> None:
        self.play_button.clicked.connect(self._start_or_resume)
        self.pause_button.clicked.connect(self.engine.pause)
        self.stop_button.clicked.connect(self._stop_all)
        self.restart_button.clicked.connect(self._restart_run)
        self.continue_button.clicked.connect(self._continue_run)
        self.interval_input.valueChanged.connect(self._on_interval_changed)
        self.settings_button.clicked.connect(self._open_settings)

        self.engine.info_ready.connect(self._on_info_ready)
        self.engine.status_changed.connect(self.statusBar().showMessage)
        self.engine.episode_finished.connect(self._on_episode_finished)

    def _start_or_resume(self) -> None:
        if self.engine.isRunning():
            if self.tensorboard is not None and self.preferences.auto_tensorboard:
                self.tensorboard.start()
            self.engine.resume()
            return

        # Startup and "continue" have already prepared the current session.
        # Only the explicit restart action should discard learned state.
        if self.engine.logger is None and self.engine.session_manager is not None:
            self.engine.set_logger(
                TensorBoardLogger(
                    log_dir=self.engine.session_manager.log_dir,
                    **(self._logger_settings or {}),
                )
            )
        if self.tensorboard is not None and self.preferences.auto_tensorboard:
            self.tensorboard.start()
        if not self.engine.isRunning():
            self.engine.start()

    def _stop_all(self) -> None:
        self.engine.stop()
        self.engine.set_logger(None)
        if self.tensorboard is not None:
            self.tensorboard.stop()
        self.statusBar().showMessage("已停止")

    def _restart_run(self) -> None:
        self._prepare_fresh_run(open_tensorboard=None)

    def _continue_run(self) -> None:
        loaded = self.engine.continue_session()
        if loaded:
            if self.tensorboard is not None and self.engine.session_manager is not None:
                self.tensorboard.set_log_dir(self.engine.session_manager.log_dir)
            self._clear_training_views()

    def _on_interval_changed(self, value: float) -> None:
        self.preferences = replace(self.preferences, delay_ms=float(value))
        self.engine.set_delay(float(value))
        self.preferences.save(self.settings_store)
        if self._settings_dialog is not None:
            self._settings_dialog.interval_input.setValue(float(value))

    def _on_info_ready(self, info: dict) -> None:
        current_state = int(info.get("next_state", self.env.current_state))
        self.grid_painter.update_view(
            current_state=current_state,
        )
        self.monitor_panel.update_info(info)

        self.step_label.setText(f"训练步  {info.get('step', '-')}")
        self.state_label.setText(
            f"状态  {info.get('state', '-')} → {info.get('next_state', '-')}    "
            f"奖励  {info.get('reward', '-')}"
        )

    def _on_episode_finished(self, episode: int, reward: float) -> None:
        self.statusBar().showMessage(
            f"episode {episode} finished, reward={reward:.2f}"
        )

    def _prepare_fresh_run(self, open_tensorboard: bool | None) -> None:
        reopen_tensorboard = False
        if self.tensorboard is not None:
            reopen_tensorboard = self.tensorboard.is_running()
            self.tensorboard.stop()

        self.engine.start_fresh_session()

        if self.engine.logger is None and self.engine.session_manager is not None:
            logger_settings = self._logger_settings or {}
            self.engine.set_logger(
                TensorBoardLogger(
                    log_dir=self.engine.session_manager.log_dir,
                    **logger_settings,
                )
            )
        if self.tensorboard is not None and self.engine.session_manager is not None:
            self.tensorboard.set_log_dir(self.engine.session_manager.log_dir)
            should_open = reopen_tensorboard if open_tensorboard is None else open_tensorboard
            if should_open:
                self.tensorboard.start()

        self._clear_training_views()

    def _clear_training_views(self) -> None:
        state = self.env.reset()
        self.grid_painter.set_current_state(state)
        self.monitor_panel.clear()
        self.step_label.setText("训练步  —")
        self.state_label.setText("状态  —    奖励  —")
