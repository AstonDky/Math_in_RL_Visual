"""Record the real Qt application using isolated training data and preferences.

Run from the repository root: python tools/record_demo.py --ffmpeg /path/to/ffmpeg
Requires the normal application dependencies and an FFmpeg executable.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import QPoint, QSettings, QTimer, Qt
from PyQt6.QtGui import QColor, QFont, QFontDatabase, QImage, QPainter
from PyQt6.QtWidgets import QApplication, QDialogButtonBox

from algorithms import q_learning_off_policy
from core.algorithm_adapters import adapt_algorithm
from core.engine import TrainingEngine
from core.rl_parameters import RLAlgorithmConfig
from core.table_agent import TableAgent
from envs.grid_world import GridWorld
from ui.main_window import MainWindow
from ui.settings import InterfaceSettings
from utils.logger import TensorBoardLogger
from utils.session import TrainingSessionManager


def record(ffmpeg: str, output: Path, preview: Path, dark_preview: Path) -> None:
    fps, duration, width, height = 20, 32, 1480, 930
    for path in (output, preview, dark_preview):
        path.parent.mkdir(parents=True, exist_ok=True)
    app = QApplication([])
    app.setQuitOnLastWindowClosed(False)
    app.setStyle("Fusion")
    # Qt's offscreen plugin on Windows does not discover system fonts itself.
    if sys.platform == "win32":
        for name in ("msyh.ttc", "arial.ttf", "consola.ttf"):
            path = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / name
            if path.exists():
                QFontDatabase.addApplicationFont(str(path))
        app.setFont(QFont("Microsoft YaHei UI", 10))

    with tempfile.TemporaryDirectory(prefix="rl-demo-") as temporary:
        root = Path(temporary)
        store = QSettings(str(root / "preferences.ini"), QSettings.Format.IniFormat)
        InterfaceSettings(delay_ms=45, auto_tensorboard=False).save(store)
        env = GridWorld(r_boundary=-10, r_forbidden=-10, r_target=5, r_other=-1)
        agent = TableAgent(
            env.num_states, env.num_actions, env.reward_map(),
            adapt_algorithm(q_learning_off_policy),
            RLAlgorithmConfig(seed=7, alpha=0.1, gamma=0.9, epsilon=0.1),
            "q_learning_off_policy", env.layout.shape,
        )
        sessions = TrainingSessionManager(agent.algorithm_name, root / "sessions", root / "runs")
        logger = TensorBoardLogger(sessions.log_dir)
        engine = TrainingEngine(env, agent, logger, sessions, max_steps_per_episode=60)
        window = MainWindow(env, engine, settings=store)
        window.resize(width, height)
        window.show()
        app.processEvents()
        process = subprocess.Popen(
            [ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
             "-f", "rawvideo", "-pixel_format", "rgba",
             "-video_size", f"{width}x{height + 50}", "-framerate", str(fps),
             "-i", "pipe:0", "-an", "-c:v", "libx264", "-preset", "fast",
             "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output)],
            stdin=subprocess.PIPE,
        )
        frame_number = 0
        caption = "Ready to explore  |  Q-learning in GridWorld"
        failures: list[Exception] = []
        events: list[dict] = []
        stopped_progress = None

        def mark(text: str) -> None:
            nonlocal caption
            caption = text
            events.append({"video_seconds": frame_number / fps, "caption": text,
                           "progress": list(engine.progress())})

        def capture() -> None:
            nonlocal frame_number, stopped_progress
            try:
                second = frame_number / fps
                if frame_number == fps:
                    window.play_button.click()
                    mark("Train  |  Inspect values, action probabilities and source code")
                elif frame_number == 9 * fps:
                    window.pause_button.click()
                    mark("Pause  |  Inspect the current update")
                elif frame_number == 11 * fps:
                    window.settings_button.click()
                    mark("Preferences  |  Theme, text size, pacing and TensorBoard")
                elif frame_number == 12 * fps:
                    dialog = window._settings_dialog
                    dialog.theme_input.setCurrentIndex(dialog.theme_input.findData("dark"))
                    dialog.interval_input.setValue(25)
                elif frame_number == 14 * fps:
                    window._settings_dialog.buttons.button(QDialogButtonBox.StandardButton.Apply).click()
                    assert InterfaceSettings.load(store).theme == "dark"
                    mark("Apply  |  Display preferences are saved")
                elif frame_number == 16 * fps:
                    window._settings_dialog.buttons.button(QDialogButtonBox.StandardButton.Ok).click()
                elif frame_number == 17 * fps:
                    window.play_button.click()
                    mark("Resume  |  Continue the same training session")
                elif frame_number == 25 * fps:
                    window.stop_button.click()
                    stopped_progress = engine.progress()
                    mark("Stop  |  Save the current training checkpoint")
                elif frame_number == 26 * fps:
                    window.continue_button.click()
                    assert engine.progress() == stopped_progress
                    mark("Restore  |  Load the saved parameters and progress")
                elif frame_number == 27 * fps:
                    window.play_button.click()
                    mark("Continue  |  Train from the restored checkpoint")
                elif frame_number == 30 * fps:
                    window.stop_button.click()
                    mark("Math in RL Visual  |  Open-source learning workspace")

                if second >= duration:
                    timer.stop()
                    app.quit()
                    return
                pixels = window.grab().toImage()
                if frame_number == 8 * fps:
                    pixels.save(str(preview))
                elif frame_number == 23 * fps:
                    pixels.save(str(dark_preview))
                canvas = QImage(width, height + 50, QImage.Format.Format_RGBA8888)
                canvas.fill(QColor("#122332"))
                painter = QPainter(canvas)
                painter.drawImage(0, 0, pixels)
                dialog = window._settings_dialog
                if dialog is not None and dialog.isVisible():
                    # Compose the actual visible Qt dialog over its parent window.
                    origin = window.mapFromGlobal(dialog.mapToGlobal(QPoint(0, 0)))
                    painter.fillRect(0, 0, width, height, QColor(0, 0, 0, 65))
                    painter.drawPixmap(origin, dialog.grab())
                painter.setPen(QColor("#f2f6fa"))
                painter.setFont(QFont("Arial", 12))
                painter.drawText(22, height, width - 44, 50,
                                 Qt.AlignmentFlag.AlignVCenter, caption)
                painter.end()
                process.stdin.write(canvas.constBits().asstring(canvas.sizeInBytes()))
                frame_number += 1
            except Exception as error:
                failures.append(error)
                timer.stop()
                app.quit()

        timer = QTimer()
        timer.setTimerType(Qt.TimerType.PreciseTimer)
        timer.timeout.connect(capture)
        timer.start(round(1000 / fps))
        app.exec()
        window.close()
        process.stdin.close()
        return_code = process.wait(timeout=30)
        if failures:
            raise failures[0]
        if return_code:
            raise RuntimeError(f"FFmpeg exited with status {return_code}")
        assert frame_number == fps * duration
        assert engine.progress()[0] > 0
        assert sessions.checkpoint_path.exists()
        assert any(sessions.log_dir.iterdir())
        assert InterfaceSettings.load(store).theme == "dark"
        manifest = {
            "capture": "Actual Qt widgets and live TrainingEngine; scripted UI actions",
            "frames": frame_number, "fps": fps, "duration_seconds": duration,
            "algorithm": agent.algorithm_name, "seed": 7,
            "final_progress": list(engine.progress()), "events": events,
            "note": "Short demonstration; not a convergence or performance benchmark. User data and preferences are untouched.",
        }
        output.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"output": str(output), "frames": frame_number,
                          "steps": engine.progress()[0], "bytes": output.stat().st_size}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ffmpeg", default=shutil.which("ffmpeg"))
    parser.add_argument("--output", type=Path, default=ROOT / "docs/media/rl-demo.mp4")
    parser.add_argument("--preview", type=Path, default=ROOT / "docs/assets/app-preview.png")
    parser.add_argument("--dark-preview", type=Path, default=ROOT / "docs/assets/app-preview-dark.png")
    arguments = parser.parse_args()
    if not arguments.ffmpeg:
        parser.error("Install FFmpeg or pass --ffmpeg /path/to/ffmpeg.")
    record(arguments.ffmpeg, arguments.output, arguments.preview, arguments.dark_preview)
