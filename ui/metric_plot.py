"""轻量级 TensorBoard 风格曲线面板。"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Deque

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtWidgets import QWidget


class MetricPlot(QWidget):
    """用 PyQt 原生绘图显示 reward/loss/td_error 等曲线。"""

    _colors = {
        "reward": QColor("#2e7d32"),
        "loss": QColor("#d97706"),
        "td_error": QColor("#c62828"),
        "max_q": QColor("#6d4c41"),
    }

    def __init__(self, max_points: int = 160, parent=None) -> None:
        super().__init__(parent)
        self.max_points = max_points
        self.series: dict[str, Deque[tuple[int, float]]] = defaultdict(
            lambda: deque(maxlen=max_points)
        )
        self.setMinimumHeight(190)

    def append_metrics(self, step: int, metrics: dict[str, float]) -> None:
        for name, value in metrics.items():
            self.series[name].append((step, float(value)))
        self.update()

    def paintEvent(self, event) -> None:
        _ = event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#ffffff"))

        plot_rect = QRectF(44, 18, self.width() - 62, self.height() - 54)
        self._draw_axes(painter, plot_rect)
        self._draw_series(painter, plot_rect)
        self._draw_legend(painter, plot_rect)
        painter.end()

    def _draw_axes(self, painter: QPainter, rect: QRectF) -> None:
        painter.setPen(QPen(QColor("#d0d7de"), 1))
        for i in range(5):
            y = rect.top() + rect.height() * i / 4
            painter.drawLine(QPointF(rect.left(), y), QPointF(rect.right(), y))
        painter.setPen(QPen(QColor("#57606a"), 1.2))
        painter.drawRect(rect)

    def _draw_series(self, painter: QPainter, rect: QRectF) -> None:
        visible = [item for item in self.series.items() if item[1]]
        if not visible:
            painter.setPen(QColor("#57606a"))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, "waiting for metrics")
            return

        points = [point for _, values in visible for point in values]
        min_step = min(step for step, _ in points)
        max_step = max(step for step, _ in points)
        min_value = min(value for _, value in points)
        max_value = max(value for _, value in points)
        if min_value == max_value:
            min_value -= 1.0
            max_value += 1.0
        if min_step == max_step:
            max_step += 1

        for name, values in visible:
            color = self._colors.get(name, QColor("#555555"))
            painter.setPen(QPen(color, 2.2))
            mapped = [
                self._map_point(step, value, min_step, max_step, min_value, max_value, rect)
                for step, value in values
            ]
            for start, end in zip(mapped, mapped[1:]):
                painter.drawLine(start, end)

    def _draw_legend(self, painter: QPainter, rect: QRectF) -> None:
        x = int(rect.left())
        y = int(rect.bottom() + 22)
        for name in self.series:
            color = self._colors.get(name, QColor("#555555"))
            painter.setPen(QPen(color, 3))
            painter.drawLine(x, y - 5, x + 18, y - 5)
            painter.setPen(QColor("#24292f"))
            painter.drawText(x + 24, y, name)
            x += 92

    def _map_point(
        self,
        step: int,
        value: float,
        min_step: int,
        max_step: int,
        min_value: float,
        max_value: float,
        rect: QRectF,
    ) -> QPointF:
        x_ratio = (step - min_step) / (max_step - min_step)
        y_ratio = (value - min_value) / (max_value - min_value)
        return QPointF(
            rect.left() + x_ratio * rect.width(),
            rect.bottom() - y_ratio * rect.height(),
        )
