"""策略概率十字 glyph 绘制组件。"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtWidgets import QWidget

from envs.grid_world import CellType, GridAction


class PolicyPainter(QWidget):
    """用很小的中心十字线表示每个状态的动作概率。"""

    def __init__(self, layout: NDArray[np.int_], parent=None) -> None:
        super().__init__(parent)
        self.layout = layout
        self.rows, self.cols = layout.shape
        self.policy_probs: NDArray[np.float64] | None = None
        self.current_state: int | None = None
        self.updated_state: int | None = None
        self.setMinimumSize(220, 220)

    def update_policy(
        self,
        policy_probs: NDArray[np.float64],
        current_state: int | None,
        updated_state: int | None,
    ) -> None:
        self.policy_probs = np.array(policy_probs, dtype=float, copy=True)
        self.current_state = current_state
        self.updated_state = updated_state
        self.update()

    def clear(self) -> None:
        self.policy_probs = None
        self.current_state = None
        self.updated_state = None
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

        for row in range(self.rows):
            for col in range(self.cols):
                state = row * self.cols + col
                rect = QRectF(
                    offset_x + col * cell_size,
                    offset_y + row * cell_size,
                    cell_size,
                    cell_size,
                )
                self._draw_cell(painter, rect, row, col)
                self._draw_policy_glyph(painter, rect, state)
                if state == self.updated_state:
                    self._draw_updated_outline(painter, rect)
                if state == self.current_state:
                    self._draw_current_outline(painter, rect)

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
                color = QColor("#9af3ee")
            case _:
                color = QColor("#ffffff")

        painter.fillRect(rect, color)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(QColor("#333333"), 1))
        painter.drawRect(rect)

    def _draw_policy_glyph(
        self,
        painter: QPainter,
        rect: QRectF,
        state: int,
    ) -> None:
        if self.policy_probs is None:
            return

        probs = np.asarray(self.policy_probs[state], dtype=float)
        total = float(np.sum(probs))
        if total <= 0.0:
            return

        normalized = probs / total
        center = rect.center()
        unit_length = min(rect.width(), rect.height()) * 0.32
        pen = QPen(QColor("#4b7f23"), max(1.0, rect.width() * 0.018))
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)

        directions = {
            GridAction.UP: QPointF(0, -1),
            GridAction.RIGHT: QPointF(1, 0),
            GridAction.DOWN: QPointF(0, 1),
            GridAction.LEFT: QPointF(-1, 0),
        }

        for action, direction in directions.items():
            probability = float(normalized[int(action)])
            end = QPointF(
                center.x() + direction.x() * unit_length * probability,
                center.y() + direction.y() * unit_length * probability,
            )
            painter.drawLine(center, end)

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#4b7f23"))
        painter.drawEllipse(center, 1.8, 1.8)

    def _draw_current_outline(self, painter: QPainter, rect: QRectF) -> None:
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(QColor("#c62828"), 2))
        painter.drawRect(rect.adjusted(4, 4, -4, -4))

    def _draw_updated_outline(self, painter: QPainter, rect: QRectF) -> None:
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(QColor("#7b1fa2"), 2))
        painter.drawRect(rect.adjusted(1, 1, -1, -1))
