"""状态值网格显示组件。"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from PyQt6.QtCore import QRectF, Qt
from PyQt6.QtGui import QColor, QFont, QPainter, QPen
from PyQt6.QtWidgets import QWidget

from envs.grid_world import CellType


class RewardGrid(QWidget):
    """显示当前策略下的状态价值 V^pi(s)。"""

    def __init__(self, layout: NDArray[np.int_], parent=None) -> None:
        super().__init__(parent)
        self.layout = layout
        self.rows, self.cols = layout.shape
        self.value_map: NDArray[np.float64] | None = None
        self.highlight_state: int | None = None
        self.setMinimumSize(220, 220)

    def update_values(
        self,
        value_map: NDArray[np.float64],
        highlight_state: int | None,
    ) -> None:
        self.value_map = value_map
        self.highlight_state = highlight_state
        self.update()

    def clear(self) -> None:
        self.value_map = None
        self.highlight_state = None
        self.update()

    def paintEvent(self, event) -> None:
        _ = event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#ffffff"))

        cell_size = min(self.width() / self.cols, self.height() / self.rows)
        board_width = cell_size * self.cols
        board_height = cell_size * self.rows
        offset_x = (self.width() - board_width) / 2
        offset_y = (self.height() - board_height) / 2

        font = QFont()
        font.setPointSize(max(8, int(cell_size * 0.16)))
        painter.setFont(font)

        for row in range(self.rows):
            for col in range(self.cols):
                state = row * self.cols + col
                rect = QRectF(
                    offset_x + col * cell_size,
                    offset_y + row * cell_size,
                    cell_size,
                    cell_size,
                )
                self._draw_cell(painter, rect, row, col, state)

        painter.end()

    def _draw_cell(
        self,
        painter: QPainter,
        rect: QRectF,
        row: int,
        col: int,
        state: int,
    ) -> None:
        cell_type = CellType(int(self.layout[row, col]))
        if cell_type == CellType.FORBIDDEN:
            color = QColor("#fff3bf")
        elif cell_type == CellType.TARGET:
            color = QColor("#c3f1cf")
        else:
            color = QColor("#ffffff")

        painter.fillRect(rect, color)
        painter.setPen(QPen(QColor("#444444"), 1))
        painter.drawRect(rect)

        if state == self.highlight_state:
            painter.setPen(QPen(QColor("#c62828"), 3))
            painter.drawRect(rect.adjusted(2, 2, -2, -2))

        if cell_type == CellType.TARGET:
            text = "V=0.00"
        elif self.value_map is None:
            text = "V=?"
        else:
            text = f"V={self.value_map[state]:.2f}"
        painter.setPen(QColor("#24292f"))
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)
