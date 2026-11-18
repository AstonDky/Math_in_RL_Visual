"""Shared colors for native controls and the three GridWorld views."""

from __future__ import annotations

from dataclasses import dataclass

from PyQt6.QtGui import QColor, QPalette


@dataclass(frozen=True)
class Theme:
    background: str
    surface: str
    inset: str
    text: str
    muted: str
    border: str
    grid: str
    accent: str
    accent_hover: str
    selection: str
    forbidden: str
    target: str
    updated: str
    disabled: str


THEMES = {
    "light": Theme(
        "#f2f5f8", "#ffffff", "#f7f9fc", "#1f2d3d", "#526477",
        "#d7e0e9", "#b5c3d1", "#1769aa", "#10568e", "#e2effb",
        "#f9df9e", "#b8e6d8", "#927022", "#77889a",
    ),
    "dark": Theme(
        "#111923", "#1b2734", "#15202c", "#e6edf5", "#abbccc",
        "#394b5e", "#50667b", "#70baff", "#97ccff", "#234967",
        "#71582d", "#285e56", "#e9ba62", "#8292a2",
    ),
}


def palette_for(theme: Theme) -> QPalette:
    palette = QPalette()
    for role, color in (
        (QPalette.ColorRole.Window, theme.background),
        (QPalette.ColorRole.WindowText, theme.text),
        (QPalette.ColorRole.Base, theme.surface),
        (QPalette.ColorRole.AlternateBase, theme.inset),
        (QPalette.ColorRole.Text, theme.text),
        (QPalette.ColorRole.Button, theme.surface),
        (QPalette.ColorRole.ButtonText, theme.text),
        (QPalette.ColorRole.Highlight, theme.accent),
        (QPalette.ColorRole.HighlightedText, theme.background),
        (QPalette.ColorRole.ToolTipBase, theme.surface),
        (QPalette.ColorRole.ToolTipText, theme.text),
    ):
        palette.setColor(role, QColor(color))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text,
                     QColor(theme.disabled))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText,
                     QColor(theme.disabled))
    return palette


def stylesheet(theme: Theme, font_size: int = 10) -> str:
    return f"""
        QWidget {{ color: {theme.text}; font-size: {font_size}pt; }}
        QMainWindow, QWidget#appRoot {{ background: {theme.background}; }}
        QLabel {{ background: transparent; }}
        QLabel#appTitle {{ font-size: {font_size + 5}pt; font-weight: 600; }}
        QLabel#algorithmName, QLabel#muted {{ color: {theme.muted}; }}
        QLabel#sectionTitle {{ font-weight: 600; }}
        QFrame#environmentPanel {{ background: {theme.surface};
            border: 1px solid {theme.border}; border-radius: 8px; }}
        QGroupBox {{ background: {theme.surface}; border: 1px solid {theme.border};
            border-radius: 6px; margin-top: 12px; padding: 12px 10px 9px; }}
        QGroupBox::title {{ subcontrol-origin: margin; left: 10px;
            padding: 0 5px; font-weight: 600; }}
        QPushButton {{ background: {theme.surface}; border: 1px solid {theme.border};
            border-radius: 5px; padding: 6px 11px; min-height: 20px; }}
        QPushButton:hover {{ background: {theme.selection}; border-color: {theme.accent}; }}
        QPushButton:pressed {{ background: {theme.selection}; }}
        QPushButton:focus, QAbstractSpinBox:focus, QComboBox:focus,
        QTextEdit:focus, QPlainTextEdit:focus {{ border: 2px solid {theme.accent}; }}
        QPushButton:disabled {{ color: {theme.disabled}; background: {theme.inset}; }}
        QPushButton#primaryAction {{ background: {theme.accent}; color: {theme.background};
            border-color: {theme.accent}; font-weight: 600; }}
        QPushButton#primaryAction:hover {{ background: {theme.accent_hover}; }}
        QAbstractSpinBox, QComboBox {{ background: {theme.surface};
            border: 1px solid {theme.border}; border-radius: 4px;
            padding: 5px 7px; min-height: 20px; }}
        QTextEdit, QPlainTextEdit {{ background: {theme.inset};
            border: 1px solid {theme.border}; border-radius: 4px; padding: 5px;
            selection-background-color: {theme.selection}; selection-color: {theme.text}; }}
        QCheckBox {{ spacing: 8px; }}
        QSplitter::handle {{ background: {theme.background}; }}
        QSplitter::handle:hover {{ background: {theme.border}; }}
        QStatusBar {{ background: {theme.background}; color: {theme.muted}; }}
        QStatusBar::item {{ border: none; }}
        QDialog {{ background: {theme.background}; }}
        QToolTip {{ background: {theme.surface}; color: {theme.text};
            border: 1px solid {theme.border}; padding: 5px; }}
    """
