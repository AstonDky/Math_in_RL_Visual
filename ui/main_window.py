"""应用主窗口。"""

from __future__ import annotations

import math

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSlider,
    QStatusBar,
    QStyle,
    QVBoxLayout,
    QWidget,
)

from core.engine import TrainingEngine
from envs.grid_world import GridWorld
from ui.grid_painter import GridPainter
from ui.monitor_panel import MonitorPanel


class MainWindow(QMainWindow):
    """组合控制区、GridWorld 可视化区和监控区。"""

    def __init__(self, env: GridWorld, engine: TrainingEngine) -> None:
        super().__init__()
        self.env = env
        self.engine = engine

        self.setWindowTitle("Math in RL Visual")
        self.resize(1320, 780)

        self.grid_painter = GridPainter(layout=env.layout)
        self.grid_painter.set_reward_map(env.reward_map())
        self.monitor_panel = MonitorPanel(layout=env.layout)
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
        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setRange(0, 1000)
        self.speed_slider.setValue(self._delay_to_slider(engine.delay_ms))
        self.speed_label = QLabel(self._speed_text(engine.delay_ms))

        self._build_layout()
        self._connect_signals()

    def closeEvent(self, event) -> None:
        self.engine.stop()
        super().closeEvent(event)

    def _build_layout(self) -> None:
        controls = QHBoxLayout()
        controls.addWidget(self.play_button)
        controls.addWidget(self.pause_button)
        controls.addWidget(self.stop_button)
        controls.addWidget(self.restart_button)
        controls.addWidget(self.continue_button)
        controls.addSpacing(20)
        controls.addWidget(self.speed_label)
        controls.addWidget(self.speed_slider, stretch=1)

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
        self.speed_slider.valueChanged.connect(self._on_speed_changed)

        self.engine.info_ready.connect(self._on_info_ready)
        self.engine.status_changed.connect(self.statusBar().showMessage)
        self.engine.episode_finished.connect(self._on_episode_finished)

    def _start_or_resume(self) -> None:
        if not self.engine.isRunning():
            self.engine.start()
        else:
            self.engine.resume()

    def _restart_run(self) -> None:
        self.engine.restart_session()
        self._clear_training_views()

    def _continue_run(self) -> None:
        loaded = self.engine.continue_session()
        if loaded:
            self._clear_training_views()

    def _on_speed_changed(self, value: int) -> None:
        delay_ms = self._slider_to_delay(value)
        self.engine.set_delay(delay_ms)
        self.speed_label.setText(self._speed_text(delay_ms))

    def _slider_to_delay(self, value: int) -> float:
        """把线性滑块映射成对数延迟。

        value=0 表示不 sleep；value>0 时最小约 0.001 ms，并逐渐增至 1000 ms。
        因此滑块左端能表达“无限接近 0”的训练间隔。
        """

        if value <= 0:
            return 0.0
        return 10 ** (-3.0 + 6.0 * value / 1000.0)

    def _delay_to_slider(self, delay_ms: float) -> int:
        if delay_ms <= 0.0:
            return 0
        return max(1, min(1000, round((math.log10(delay_ms) + 3.0) / 6.0 * 1000)))

    def _speed_text(self, value: float) -> str:
        if value <= 0.0:
            return "速度间隔: 0 ms (正常运行速度)"
        if value < 1.0:
            return f"速度间隔: {value:.4f} ms"
        return f"速度间隔: {value:.1f} ms"

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
