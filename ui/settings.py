"""Persistent presentation preferences, independent of algorithm checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
import math

from PyQt6.QtCore import QSettings
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox,
    QFormLayout, QLabel, QSpinBox, QVBoxLayout,
)


@dataclass(frozen=True)
class InterfaceSettings:
    theme: str = "light"
    font_size: int = 10
    delay_ms: float = 0.0
    auto_tensorboard: bool = True

    @classmethod
    def load(cls, store: QSettings, default_delay: float = 0.0) -> InterfaceSettings:
        theme = str(store.value("interface/theme", "light"))
        try:
            font_size = int(store.value("interface/font_size", 10))
        except (ValueError, TypeError):
            font_size = 10
        try:
            delay = float(store.value("interface/delay_ms", default_delay))
        except (ValueError, TypeError):
            delay = default_delay
        auto_tb = str(store.value("interface/auto_tensorboard", "true")).lower()
        return cls(
            theme=theme if theme in ("light", "dark") else "light",
            font_size=max(9, min(14, font_size)),
            delay_ms=max(0.0, min(1000.0, delay)) if math.isfinite(delay) else 0.0,
            auto_tensorboard=auto_tb not in ("false", "0"),
        )

    def save(self, store: QSettings) -> None:
        for key, value in (
            ("theme", self.theme), ("font_size", self.font_size),
            ("delay_ms", self.delay_ms), ("auto_tensorboard", self.auto_tensorboard),
        ):
            store.setValue(f"interface/{key}", value)
        store.sync()


class SettingsDialog(QDialog):
    """Edits a draft; only Apply or OK persists it through MainWindow."""

    def __init__(self, preferences: InterfaceSettings, apply_settings, parent=None):
        super().__init__(parent)
        self.setWindowTitle("界面设置")
        self.setMinimumWidth(420)
        self._apply_settings = apply_settings
        self.theme_input = QComboBox()
        self.theme_input.setObjectName("themeInput")
        self.theme_input.addItem("浅色", "light")
        self.theme_input.addItem("深色", "dark")
        self.font_input = QSpinBox()
        self.font_input.setObjectName("fontSizeInput")
        self.font_input.setRange(9, 14)
        self.font_input.setSuffix(" pt")
        self.interval_input = QDoubleSpinBox()
        self.interval_input.setObjectName("delayInput")
        self.interval_input.setRange(0.0, 1000.0)
        self.interval_input.setDecimals(3)
        self.interval_input.setSuffix(" ms")
        self.interval_input.setKeyboardTracking(False)
        self.tensorboard_input = QCheckBox("开始训练时自动打开 TensorBoard")
        self.tensorboard_input.setObjectName("autoTensorBoardInput")

        form = QFormLayout()
        form.setSpacing(14)
        form.addRow("主题", self.theme_input)
        form.addRow("界面字号", self.font_input)
        form.addRow("训练显示间隔", self.interval_input)
        form.addRow(self.tensorboard_input)
        note = QLabel("点击“应用”或“确定”保存设置。恢复默认只影响界面设置，保留训练进度。")
        note.setObjectName("muted")
        note.setWordWrap(True)
        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Apply
            | QDialogButtonBox.StandardButton.Cancel
            | QDialogButtonBox.StandardButton.RestoreDefaults
        )
        for button, text in (
            (QDialogButtonBox.StandardButton.Ok, "确定"),
            (QDialogButtonBox.StandardButton.Apply, "应用"),
            (QDialogButtonBox.StandardButton.Cancel, "取消"),
            (QDialogButtonBox.StandardButton.RestoreDefaults, "恢复默认"),
        ):
            self.buttons.button(button).setText(text)
        self.buttons.accepted.connect(self._accept)
        self.buttons.rejected.connect(self.reject)
        self.buttons.button(QDialogButtonBox.StandardButton.Apply).clicked.connect(self.apply)
        self.buttons.button(QDialogButtonBox.StandardButton.RestoreDefaults).clicked.connect(
            lambda: self.set_preferences(InterfaceSettings())
        )
        root = QVBoxLayout(self)
        root.setContentsMargins(22, 22, 22, 18)
        root.setSpacing(22)
        root.addLayout(form)
        root.addWidget(note)
        root.addWidget(self.buttons)
        self.set_preferences(preferences)

    def set_preferences(self, preferences: InterfaceSettings) -> None:
        self.theme_input.setCurrentIndex(self.theme_input.findData(preferences.theme))
        self.font_input.setValue(preferences.font_size)
        self.interval_input.setValue(preferences.delay_ms)
        self.tensorboard_input.setChecked(preferences.auto_tensorboard)

    def preferences(self) -> InterfaceSettings:
        return InterfaceSettings(
            theme=self.theme_input.currentData(), font_size=self.font_input.value(),
            delay_ms=self.interval_input.value(),
            auto_tensorboard=self.tensorboard_input.isChecked(),
        )

    def apply(self) -> None:
        self._apply_settings(self.preferences())

    def _accept(self) -> None:
        self.apply()
        self.accept()
