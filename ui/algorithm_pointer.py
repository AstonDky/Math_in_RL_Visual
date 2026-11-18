"""Read-only source view with a staged reading pointer."""

from __future__ import annotations

from PyQt6.QtGui import QColor, QFont, QTextCursor, QTextFormat
from PyQt6.QtWidgets import QGroupBox, QLabel, QPlainTextEdit, QTextEdit, QVBoxLayout

from ui.theme import THEMES, Theme


class AlgorithmPointer(QGroupBox):
    """Shows real extracted source, preserving indentation and horizontal scrolling."""

    def __init__(self, parent=None) -> None:
        super().__init__("算法源码 · 阅读指针", parent)
        self.theme = THEMES["light"]
        self._lines: list[str] = []
        self._current_line = 1
        self.title_label = QLabel("开始训练后显示当前算法的真实源码")
        self.title_label.setObjectName("muted")
        self.title_label.setWordWrap(True)
        self.code_view = QPlainTextEdit()
        self.code_view.setReadOnly(True)
        self.code_view.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.code_view.setMinimumHeight(80)
        self.code_view.setAccessibleName("算法源码与当前展示行")
        self.code_view.setToolTip("按顺序展示源码，帮助对照阅读；高亮不代表实际执行行。")
        self.set_code_font(10)
        layout = QVBoxLayout(self)
        layout.setSpacing(7)
        layout.addWidget(self.title_label)
        layout.addWidget(self.code_view, 1)

    def set_code_font(self, point_size: int) -> None:
        font = QFont("Consolas", point_size)
        font.setStyleHint(QFont.StyleHint.Monospace)
        self.code_view.setFont(font)

    def set_theme(self, theme: Theme) -> None:
        self.theme = theme
        self._highlight_current_line()

    def update_trace(self, trace: dict) -> None:
        self.title_label.setText(str(trace.get("title", "Algorithm")))
        lines = [str(line) for line in trace.get("lines", [])]
        self._current_line = int(trace.get("current_line", 1))
        if lines != self._lines:
            self._lines = lines
            width = len(str(len(lines)))
            self.code_view.setPlainText("\n".join(
                f"{index:>{width}}   {line}" for index, line in enumerate(lines, 1)
            ))
        self._highlight_current_line()

    def _highlight_current_line(self) -> None:
        if not self._lines:
            self.code_view.setExtraSelections([])
            return
        index = max(0, min(len(self._lines) - 1, self._current_line - 1))
        cursor = QTextCursor(self.code_view.document().findBlockByNumber(index))
        selection = QTextEdit.ExtraSelection()
        selection.cursor = cursor
        selection.format.setBackground(QColor(self.theme.selection))
        selection.format.setForeground(QColor(self.theme.text))
        selection.format.setProperty(QTextFormat.Property.FullWidthSelection, True)
        self.code_view.setExtraSelections([selection])
        horizontal_position = self.code_view.horizontalScrollBar().value()
        self.code_view.setTextCursor(cursor)
        self.code_view.ensureCursorVisible()
        self.code_view.horizontalScrollBar().setValue(horizontal_position)

    def clear(self) -> None:
        self._lines = []
        self.title_label.setText("开始训练后显示当前算法的真实源码")
        self.code_view.clear()
        self.code_view.setExtraSelections([])
