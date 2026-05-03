"""训练过程监控面板。"""

from __future__ import annotations

from numpy.typing import NDArray
from PyQt6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ui.algorithm_pointer import AlgorithmPointer
from ui.policy_painter import PolicyPainter
from ui.reward_grid import RewardGrid


class MonitorPanel(QWidget):
    """解析标准 Info Dict，并显示公式、奖励格和算法指针。"""

    def __init__(self, layout: NDArray, parent=None) -> None:
        super().__init__(parent)

        self.formula_label = QLabel("-")
        self.formula_label.setWordWrap(True)
        self.calculation_label = QLabel("-")
        self.calculation_label.setWordWrap(True)

        self.variables_view = QTextEdit()
        self.variables_view.setReadOnly(True)
        self.variables_view.setMinimumHeight(120)

        self.reward_grid = RewardGrid(layout=layout)
        self.policy_painter = PolicyPainter(layout=layout)
        self.algorithm_pointer = AlgorithmPointer()

        formula_box = QGroupBox("TD 公式计算")
        formula_layout = QVBoxLayout(formula_box)
        formula_layout.addWidget(self.formula_label)
        formula_layout.addWidget(self.calculation_label)
        formula_layout.addWidget(self.variables_view)

        reward_box = QGroupBox("最优状态值 V(s) 网格")
        reward_layout = QVBoxLayout(reward_box)
        reward_layout.addWidget(self.reward_grid)

        policy_box = QGroupBox("策略概率十字图")
        policy_layout = QVBoxLayout(policy_box)
        policy_layout.addWidget(self.policy_painter)

        top = QHBoxLayout()
        top.addWidget(formula_box, stretch=3)
        top.addWidget(reward_box, stretch=2)
        top.addWidget(policy_box, stretch=2)

        tensorboard_box = QGroupBox("TensorBoard")
        tensorboard_layout = QVBoxLayout(tensorboard_box)
        self.tensorboard_note = QLabel(
            "训练指标已拆分写入 runs/<algorithm_name>：\n"
            "rollout/reward, train/loss, train/td_error, train/q_value"
        )
        self.tensorboard_note.setWordWrap(True)
        tensorboard_layout.addWidget(self.tensorboard_note)

        layout = QVBoxLayout(self)
        layout.addLayout(top)
        layout.addWidget(tensorboard_box)
        layout.addWidget(self.algorithm_pointer)
        layout.addStretch()

    def update_info(self, info: dict) -> None:
        math_log = info.get("math_log", {})
        self.formula_label.setText(str(math_log.get("formula", "-")))
        self.calculation_label.setText(str(math_log.get("calculation", "-")))

        variables = math_log.get("variables", {})
        self.variables_view.setPlainText(self._format_dict(variables))

        value_table = info.get("value_table")
        if value_table is not None:
            self.reward_grid.update_values(
                value_map=value_table.max(axis=1),
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
                f"训练指标已拆分写入 runs/{algorithm_name}：\n"
                "rollout/reward, rollout/episode_reward, "
                "rollout/episode_length, train/loss, "
                "train/td_error, train/q_value"
            )

    def clear(self) -> None:
        self.formula_label.setText("-")
        self.calculation_label.setText("-")
        self.variables_view.clear()
        self.reward_grid.clear()
        self.policy_painter.clear()
        self.algorithm_pointer.clear()

    def _format_dict(self, data: dict) -> str:
        if not data:
            return "-"
        return "\n".join(f"{key}: {value}" for key, value in data.items())
