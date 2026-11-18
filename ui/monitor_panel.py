"""训练过程监控面板。"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ui.algorithm_pointer import AlgorithmPointer
from ui.policy_painter import PolicyPainter
from ui.reward_grid import RewardGrid
from ui.theme import Theme


class MonitorPanel(QWidget):
    """解析标准 Info Dict，并显示公式、奖励格和算法指针。"""

    def __init__(self, layout: NDArray, parent=None) -> None:
        super().__init__(parent)

        self.formula_label = QLabel("开始训练后显示本次更新的公式")
        self.formula_label.setWordWrap(True)
        self.calculation_label = QLabel("-")
        self.calculation_label.setWordWrap(True)
        self.calculation_label.setObjectName("muted")

        self.variables_view = QPlainTextEdit()
        self.variables_view.setReadOnly(True)
        self.variables_view.setMinimumHeight(50)
        self.variables_view.setAccessibleName("当前公式的数学变量")

        self.reward_grid = RewardGrid(layout=layout)
        self.policy_painter = PolicyPainter(layout=layout)
        self.algorithm_pointer = AlgorithmPointer()

        formula_box = QGroupBox("公式与变量")
        formula_layout = QVBoxLayout(formula_box)
        formula_layout.addWidget(self.formula_label)
        formula_layout.addWidget(self.calculation_label)
        formula_layout.addWidget(self.variables_view)

        reward_box = QGroupBox("状态价值 Vπ(s)")
        reward_layout = QVBoxLayout(reward_box)
        reward_layout.addWidget(self.reward_grid)

        policy_box = QGroupBox("策略概率 π(a|s)")
        policy_box.setToolTip("箭头方向对应动作，长度与动作概率成正比；圆环表示停留动作。")
        policy_layout = QVBoxLayout(policy_box)
        policy_layout.addWidget(self.policy_painter)

        top = QHBoxLayout()
        top.addWidget(reward_box, stretch=2)
        top.addWidget(policy_box, stretch=2)
        top.setContentsMargins(0, 0, 0, 0)
        top.setSpacing(12)
        plots = QWidget()
        plots.setLayout(top)
        plots.setMinimumHeight(205)

        tensorboard_box = QGroupBox("TensorBoard 日志")
        tensorboard_layout = QVBoxLayout(tensorboard_box)
        self.tensorboard_note = QLabel(
            "开始训练后记录 reward、loss、td_error 与 q_value"
        )
        self.tensorboard_note.setWordWrap(True)
        self.tensorboard_note.setObjectName("muted")
        tensorboard_layout.addWidget(self.tensorboard_note)

        self.inspector_splitter = QSplitter(Qt.Orientation.Vertical)
        self.inspector_splitter.setChildrenCollapsible(False)
        self.inspector_splitter.setHandleWidth(8)
        self.inspector_splitter.addWidget(plots)
        self.inspector_splitter.addWidget(formula_box)
        self.inspector_splitter.addWidget(self.algorithm_pointer)
        self.inspector_splitter.setSizes([280, 190, 260])
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(7)
        layout.addWidget(self.inspector_splitter, 1)
        layout.addWidget(tensorboard_box)

    def set_theme(self, theme: Theme, font_size: int) -> None:
        self.reward_grid.set_theme(theme)
        self.policy_painter.set_theme(theme)
        self.algorithm_pointer.set_theme(theme)
        self.algorithm_pointer.set_code_font(font_size)
        self.variables_view.setFont(QFont("Consolas", font_size))

    def update_info(self, info: dict) -> None:
        math_log = info.get("math_log", {})
        self.formula_label.setText(str(math_log.get("formula", "-")))
        self.calculation_label.setText(str(math_log.get("calculation", "-")))

        variables = math_log.get("variables", {})
        self.variables_view.setPlainText(self._format_dict(variables))

        state_values = info.get("state_values")
        if state_values is None:
            state_values = self._state_values_from_info(info)
        if state_values is not None:
            self.reward_grid.update_values(
                value_map=state_values,
                highlight_state=info.get("next_state"),
            )

        policy_probs = info.get("policy_probs")
        if policy_probs is not None:
            self.policy_painter.update_policy(
                policy_probs=policy_probs,
                current_state=info.get("next_state"),
                updated_state=info.get("updated_state", info.get("state")),
            )

        trace = info.get("algorithm_trace")
        if trace is not None:
            self.algorithm_pointer.update_trace(trace)

        algorithm_name = info.get("algorithm_name")
        if algorithm_name:
            self.tensorboard_note.setText(
                f"runs/{algorithm_name} · reward / loss / td_error / q_value"
            )
            cadence = getattr(self, "_tensorboard_cadence", "")
            self.tensorboard_note.setToolTip(
                "rollout/reward, rollout/episode_reward, rollout/episode_length, "
                f"train/loss, train/td_error, train/q_value\n{cadence}"
            )

    def set_tensorboard_cadence(
        self,
        log_interval_steps: int,
        flush_interval_steps: int,
        flush_interval_seconds: float,
        reload_interval_seconds: float,
    ) -> None:
        self._tensorboard_cadence = (
            "TensorBoard 节奏: "
            f"每 {log_interval_steps} 个训练 update 写入一次；"
            f"每 {flush_interval_steps} 个写入步或 "
            f"{flush_interval_seconds:g}s flush；"
            f"页面约每 {reload_interval_seconds:g}s reload。"
        )
        self.tensorboard_note.setToolTip(self._tensorboard_cadence)

    def clear(self) -> None:
        self.formula_label.setText("开始训练后显示本次更新的公式")
        self.calculation_label.setText("-")
        self.variables_view.clear()
        self.reward_grid.clear()
        self.policy_painter.clear()
        self.algorithm_pointer.clear()

    def _state_values_from_info(self, info: dict) -> NDArray | None:
        value_table = info.get("value_table")
        policy_probs = info.get("policy_probs")
        if value_table is None:
            return None

        values = np.asarray(value_table, dtype=float)
        if values.ndim == 1:
            return values
        if policy_probs is None:
            return None

        probs = np.asarray(policy_probs, dtype=float)
        probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        probs = np.clip(probs, 0.0, None)
        row_sums = probs.sum(axis=1, keepdims=True)
        probs = np.divide(
            probs,
            row_sums,
            out=np.full_like(probs, 1.0 / probs.shape[1], dtype=float),
            where=row_sums > 0.0,
        )
        return np.sum(probs * values, axis=1)

    def _format_dict(self, data: dict) -> str:
        if not data:
            return "-"
        return "\n".join(f"{key}: {value}" for key, value in data.items())
