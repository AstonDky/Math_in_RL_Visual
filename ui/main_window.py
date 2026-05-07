"""应用主窗口。"""

from __future__ import annotations

from PyQt6.QtWidgets import (
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QStatusBar,
    QStyle,
    QVBoxLayout,
    QWidget,
)

from core.engine import TrainingEngine
from envs.grid_world import GridWorld
from ui.grid_painter import GridPainter
from ui.monitor_panel import MonitorPanel
from utils.tensorboard import TensorBoardLauncher


class MainWindow(QMainWindow):
    """组合控制区、GridWorld 可视化区和监控区。"""

    def __init__(
        self,
        env: GridWorld,
        engine: TrainingEngine,
        tensorboard: TensorBoardLauncher | None = None,
    ) -> None:
        super().__init__()
        self.env = env
        self.engine = engine
        self.tensorboard = tensorboard

        self.setWindowTitle("Math in RL Visual")
        self.resize(1320, 780)

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
        self.play_button.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay)
        )
        self.pause_button.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPause)
        )
        self.stop_button.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_MediaStop)
        )
        self.step_label = QLabel("step: -")
        self.state_label = QLabel("state: -")
        self.interval_label = QLabel("间隔 (ms)")
        self.interval_input = QDoubleSpinBox()
        self.interval_input.setDecimals(3)
        self.interval_input.setRange(0.0, 1000.0)
        self.interval_input.setSingleStep(1.0)
        self.interval_input.setSuffix(" ms")
        self.interval_input.setValue(engine.delay_ms)
        self.interval_input.setKeyboardTracking(False)

        self._build_layout()
        self._connect_signals()

    def closeEvent(self, event) -> None:
        self.engine.stop()
        if self.tensorboard is not None:
            self.tensorboard.stop()
        super().closeEvent(event)

    def _build_layout(self) -> None:
        controls = QHBoxLayout()
        controls.addWidget(self.play_button)
        controls.addWidget(self.pause_button)
        controls.addWidget(self.stop_button)
        controls.addWidget(self.restart_button)
        controls.addWidget(self.continue_button)
        controls.addSpacing(20)
        controls.addWidget(self.interval_label)
        controls.addWidget(self.interval_input)

        left_layout = QVBoxLayout()
        left_layout.addLayout(controls)
        left_layout.addWidget(self.grid_painter, stretch=1)

        summary_layout = QHBoxLayout()
        summary_layout.addWidget(self.step_label)
        summary_layout.addWidget(self.state_label)
        summary_layout.addStretch()
        left_layout.addLayout(summary_layout)

        root_layout = QHBoxLayout()
        root_layout.addLayout(left_layout, stretch=5)
        root_layout.addWidget(self.monitor_panel, stretch=6)

        central = QWidget()
        central.setLayout(root_layout)
        self.setCentralWidget(central)
        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("ready")

    def _connect_signals(self) -> None:
        self.play_button.clicked.connect(self._start_or_resume)
        self.pause_button.clicked.connect(self.engine.pause)
        self.stop_button.clicked.connect(self.engine.stop)
        self.restart_button.clicked.connect(self._restart_run)
        self.continue_button.clicked.connect(self._continue_run)
        self.interval_input.valueChanged.connect(self._on_interval_changed)

        self.engine.info_ready.connect(self._on_info_ready)
        self.engine.status_changed.connect(self.statusBar().showMessage)
        self.engine.episode_finished.connect(self._on_episode_finished)

    def _start_or_resume(self) -> None:
        if self.tensorboard is not None:
            self.tensorboard.start()
        if not self.engine.isRunning():
            self.engine.start()
        else:
            self.engine.resume()

    def _restart_run(self) -> None:
        reopen_tensorboard = False
        if self.tensorboard is not None:
            reopen_tensorboard = self.tensorboard.is_running()
            self.tensorboard.stop()
        self.engine.restart_session()
        if self.tensorboard is not None and self.engine.session_manager is not None:
            self.tensorboard.set_log_dir(self.engine.session_manager.log_dir)
            if reopen_tensorboard:
                self.tensorboard.start()
        self._clear_training_views()

    def _continue_run(self) -> None:
        loaded = self.engine.continue_session()
        if loaded:
            if self.tensorboard is not None and self.engine.session_manager is not None:
                self.tensorboard.set_log_dir(self.engine.session_manager.log_dir)
            self._clear_training_views()

    def _on_interval_changed(self, value: float) -> None:
        self.engine.set_delay(float(value))

    def _on_info_ready(self, info: dict) -> None:
        current_state = int(info.get("next_state", self.env.current_state))
        self.grid_painter.update_view(
            current_state=current_state,
        )
        self.monitor_panel.update_info(info)

        self.step_label.setText(f"step: {info.get('step', '-')}")
        self.state_label.setText(
            "state: "
            f"{info.get('state', '-')} -> {info.get('next_state', '-')}, "
            f"reward: {info.get('reward', '-')}"
        )

    def _on_episode_finished(self, episode: int, reward: float) -> None:
        self.statusBar().showMessage(
            f"episode {episode} finished, reward={reward:.2f}"
        )

    def _clear_training_views(self) -> None:
        state = self.env.reset()
        self.grid_painter.set_current_state(state)
        self.monitor_panel.clear()
        self.step_label.setText("step: -")
        self.state_label.setText("state: -")
