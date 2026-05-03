"""策略概率十字 glyph 绘制组件。"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QColor, QFont, QPainter, QPen
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
        self.setMinimumSize(260, 260)

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

        probabilities = self._normalized_probabilities(state)
        if probabilities is None:
            return

        center = rect.center()
        cell_size = min(rect.width(), rect.height())
        unit_length = cell_size * 0.36

        directions = {
            GridAction.UP: QPointF(0, -1),
            GridAction.RIGHT: QPointF(1, 0),
            GridAction.DOWN: QPointF(0, 1),
            GridAction.LEFT: QPointF(-1, 0),
        }

        guide_pen = QPen(QColor("#d0d7de"), max(1.0, cell_size * 0.01))
        guide_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(guide_pen)
        for direction in directions.values():
            guide_end = QPointF(
                center.x() + direction.x() * unit_length,
                center.y() + direction.y() * unit_length,
            )
            painter.drawLine(center, guide_end)

        prob_pen = QPen(QColor("#4b7f23"), max(1.4, cell_size * 0.024))
        prob_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(prob_pen)
        for action, direction in directions.items():
            probability = float(probabilities[int(action)])
            end = QPointF(
                center.x() + direction.x() * unit_length * probability,
                center.y() + direction.y() * unit_length * probability,
            )
            painter.drawLine(center, end)
            self._draw_endpoint(painter, end, probability, cell_size)

        if cell_size >= 52:
            self._draw_probability_numbers(
                painter=painter,
                center=center,
                unit_length=unit_length,
                probabilities=probabilities,
                directions=directions,
                cell_size=cell_size,
            )

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#4b7f23"))
        center_radius = max(1.6, cell_size * 0.025)
        painter.drawEllipse(center, center_radius, center_radius)

    def _normalized_probabilities(
        self,
        state: int,
    ) -> NDArray[np.float64] | None:
        if self.policy_probs is None:
            return None

        probs = np.asarray(self.policy_probs[state], dtype=float)
        probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        probs = np.clip(probs, 0.0, None)
        total = float(np.sum(probs))
        if total <= 0.0:
            return None
        return probs / total

    def _draw_endpoint(
        self,
        painter: QPainter,
        end: QPointF,
        probability: float,
        cell_size: float,
    ) -> None:
        if probability <= 0.0:
            return

        radius = max(1.2, cell_size * (0.012 + 0.018 * probability))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#4b7f23"))
        painter.drawEllipse(end, radius, radius)
        prob_pen = QPen(QColor("#4b7f23"), max(1.4, cell_size * 0.024))
        prob_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(prob_pen)

    def _draw_probability_numbers(
        self,
        painter: QPainter,
        center: QPointF,
        unit_length: float,
        probabilities: NDArray[np.float64],
        directions: dict[GridAction, QPointF],
        cell_size: float,
    ) -> None:
        font = QFont()
        font.setPointSize(max(6, int(cell_size * 0.11)))
        painter.setFont(font)
        painter.setPen(QColor("#315f1f"))

        label_size = max(14.0, cell_size * 0.22)
        for action, direction in directions.items():
            probability = float(probabilities[int(action)])
            label_center = QPointF(
                center.x() + direction.x() * (unit_length + label_size * 0.32),
                center.y() + direction.y() * (unit_length + label_size * 0.32),
            )
            label_rect = QRectF(
                label_center.x() - label_size / 2,
                label_center.y() - label_size / 2,
                label_size,
                label_size,
            )
            painter.drawText(
                label_rect,
                Qt.AlignmentFlag.AlignCenter,
                f"{probability:.2f}",
            )

    def _draw_current_outline(self, painter: QPainter, rect: QRectF) -> None:
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(QColor("#c62828"), 2))
        painter.drawRect(rect.adjusted(4, 4, -4, -4))

    def _draw_updated_outline(self, painter: QPainter, rect: QRectF) -> None:
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(QColor("#7b1fa2"), 2))
        painter.drawRect(rect.adjusted(1, 1, -1, -1))
