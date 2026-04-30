"""GridWorld 主环境绘制组件。"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from PyQt6.QtCore import QRectF, Qt
from PyQt6.QtGui import QColor, QFont, QPainter, QPen
from PyQt6.QtWidgets import QWidget

from envs.grid_world import CellType


class GridPainter(QWidget):
    """主训练框：只绘制最纯粹的环境 GridWorld。"""

    def __init__(self, layout: NDArray[np.int_], parent=None) -> None:
        super().__init__(parent)
        self.layout = layout
        self.rows, self.cols = layout.shape
        self.current_state = 0
        self.reward_map: NDArray[np.float64] | None = None
        self.setMinimumSize(420, 420)

    def set_reward_map(self, reward_map: NDArray[np.float64]) -> None:
        self.reward_map = reward_map
        self.update()

    def update_view(
        self,
        policy_probs: NDArray[np.float64],
        current_state: int,
    ) -> None:
        _ = policy_probs
        self.current_state = current_state
        self.update()

    def paintEvent(self, event) -> None:
        _ = event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        cell_size = min(self.width() / self.cols, self.height() / self.rows)
        board_width = cell_size * self.cols
        board_height = cell_size * self.rows
        offset_x = (self.width() - board_width) / 2
        offset_y = (self.height() - board_height) / 2

        for row in range(self.rows):
            for col in range(self.cols):
                rect = QRectF(
                    offset_x + col * cell_size,
                    offset_y + row * cell_size,
                    cell_size,
                    cell_size,
                )
                state = row * self.cols + col
                self._draw_cell(painter, rect, row, col)
                self._draw_reward(painter, rect, state)
                if state == self.current_state:
                    self._draw_current_state(painter, rect)

        painter.end()

    def _draw_cell(
        self,
        painter: QPainter,
        rect: QRectF,
        row: int,
        col: int,
    ) -> None:
        cell_type = CellType(int(self.layout[row, col]))
        match cell_type:
            case CellType.FORBIDDEN:
                color = QColor("#ffe08a")
            case CellType.TARGET:
                color = QColor("#85d99e")
            case _:
                color = QColor("#ffffff")

        painter.fillRect(rect, color)
        painter.setPen(QPen(QColor("#333333"), 1.2))
        painter.drawRect(rect)

    def _draw_reward(self, painter: QPainter, rect: QRectF, state: int) -> None:
        if self.reward_map is None:
            return

        reward = float(self.reward_map[state])
        text = f"{reward:g}"
        font = QFont()
        font.setPointSize(max(11, int(rect.width() * 0.16)))
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QColor("#24292f"))
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)

    def _draw_current_state(self, painter: QPainter, rect: QRectF) -> None:
        """用红色边框标记当前状态，不再使用蓝色填充。"""

        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(QColor("#c62828"), 3))
        painter.drawRect(rect.adjusted(3, 3, -3, -3))
