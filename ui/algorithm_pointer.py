"""算法指针面板。"""

from __future__ import annotations

from PyQt6.QtGui import QTextCursor
from PyQt6.QtWidgets import QGroupBox, QLabel, QTextEdit, QVBoxLayout, QWidget

from core.algorithm_trace import escape_code_line


class AlgorithmPointer(QWidget):
    """显示自动解析出来的核心算法源码，并高亮当前执行行。"""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.title_label = QLabel("Algorithm")
        self.code_view = QTextEdit()
        self.code_view.setReadOnly(True)
        self.code_view.setMinimumHeight(240)

        box = QGroupBox("算法指针")
        layout = QVBoxLayout(box)
        layout.addWidget(self.title_label)
        layout.addWidget(self.code_view)

        root = QVBoxLayout(self)
        root.addWidget(box)

    def update_trace(self, trace: dict) -> None:
        title = trace.get("title", "Algorithm")
        lines = trace.get("lines", [])
        current_line = int(trace.get("current_line", 1))
        self.title_label.setText(str(title))

        html_lines = []
        for index, line in enumerate(lines, start=1):
            pointer = "&rarr;" if index == current_line else "&nbsp;&nbsp;"
            safe_line = escape_code_line(str(line))
            if index == current_line:
                html_lines.append(
                    "<tr style='background:#fff3bf; color:#111827; "
                    "font-weight:600;'>"
                    f"<td style='width:26px; padding:3px;'>{pointer}</td>"
                    f"<td style='width:36px; padding:3px; color:#57606a;'>"
                    f"{index}</td>"
                    f"<td style='padding:3px;'>{safe_line}</td>"
                    "</tr>"
                )
            else:
                html_lines.append(
                    "<tr style='color:#374151;'>"
                    f"<td style='width:26px; padding:3px;'>{pointer}</td>"
                    f"<td style='width:36px; padding:3px; color:#8c959f;'>"
                    f"{index}</td>"
                    f"<td style='padding:3px;'>{safe_line}</td>"
                    "</tr>"
                )

        self.code_view.setHtml(
            "<div style='font-family:Consolas, Menlo, monospace; "
            "font-size:13px;'>"
            "<table cellspacing='0' cellpadding='0' width='100%'>"
            f"{''.join(html_lines)}"
            "</table></div>"
        )
        self.code_view.moveCursor(QTextCursor.MoveOperation.Start)

    def clear(self) -> None:
        self.title_label.setText("Algorithm")
        self.code_view.clear()
