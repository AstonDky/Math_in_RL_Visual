"""TensorBoard 启动器。"""

from __future__ import annotations

import socket
import subprocess
import sys
import threading
import time
import webbrowser
from importlib.util import find_spec
from os import getpid
from pathlib import Path
from typing import Callable
from urllib.error import URLError
from urllib.request import urlopen


StatusCallback = Callable[[str], None]


class TensorBoardLauncher:
    """启动 TensorBoard，并在服务就绪后打开页面。"""

    def __init__(
        self,
        log_dir: str | Path,
        preferred_port: int = 6006,
        open_timeout_seconds: float = 20.0,
        reload_interval_seconds: float = 1.0,
        status_callback: StatusCallback | None = None,
    ) -> None:
        self.log_dir = Path(log_dir)
        self.preferred_port = int(preferred_port)
        self.open_timeout_seconds = float(open_timeout_seconds)
        self.reload_interval_seconds = max(0.5, float(reload_interval_seconds))
        self.status_callback = status_callback
        self.process: subprocess.Popen | None = None
        self.port: int | None = None
        self.url: str | None = None
        self._log_file = None
        self._open_thread: threading.Thread | None = None

    def start(self) -> bool:
        if not self.is_available():
            self._emit("TensorBoard 未安装，请在 MathInRL 环境安装 requirements.txt")
            return False

        if self.process is not None and self.process.poll() is None:
            self._open_when_ready()
            return True

        self.log_dir.mkdir(parents=True, exist_ok=True)
        try:
            self.port = self._choose_port(self.preferred_port)
        except RuntimeError as error:
            self._emit(f"TensorBoard 启动失败: {error}")
            return False
        self.url = f"http://localhost:{self.port}"
        log_path = (
            self.log_dir.parent
            / f"{self.log_dir.name}_tensorboard_{getpid()}_{int(time.time())}.log"
        )
        self._log_file = log_path.open("w", encoding="utf-8")
        command = [
            sys.executable,
            "-m",
            "tensorboard.main",
            "--logdir",
            str(self.log_dir),
            "--port",
            str(self.port),
            "--reload_interval",
            f"{self.reload_interval_seconds:g}",
        ]
        try:
            self.process = subprocess.Popen(
                command,
                stdout=self._log_file,
                stderr=subprocess.STDOUT,
            )
        except OSError as error:
            self._emit(f"TensorBoard 启动失败: {error}")
            if self._log_file is not None:
                self._log_file.close()
                self._log_file = None
            return False
        self._emit(f"TensorBoard 启动中: {self.url}")
        self._open_when_ready()
        return True

    def stop(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
        self.process = None
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None

    def is_available(self) -> bool:
        return find_spec("tensorboard") is not None

    def _open_when_ready(self) -> None:
        if self.url is None:
            return
        if self._open_thread is not None and self._open_thread.is_alive():
            return
        self._open_thread = threading.Thread(target=self._wait_and_open, daemon=True)
        self._open_thread.start()

    def _wait_and_open(self) -> None:
        if self.url is None:
            return

        deadline = time.monotonic() + self.open_timeout_seconds
        while time.monotonic() < deadline:
            if self.process is not None and self.process.poll() is not None:
                self._emit("TensorBoard 启动失败，请查看 runs/*_tensorboard.log")
                return
            if self._is_http_ready(self.url):
                webbrowser.open(self.url)
                self._emit(f"TensorBoard 已打开: {self.url}")
                return
            time.sleep(0.25)

        self._emit(f"TensorBoard 未在 {self.open_timeout_seconds:.0f}s 内就绪")

    def _is_http_ready(self, url: str) -> bool:
        try:
            with urlopen(url, timeout=0.5) as response:
                return 200 <= response.status < 500
        except (OSError, URLError):
            return False

    def _choose_port(self, preferred_port: int) -> int:
        for port in range(preferred_port, preferred_port + 20):
            if self._is_port_free(port):
                return port
        raise RuntimeError("No free port found for TensorBoard.")

    def _is_port_free(self, port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.2)
            return sock.connect_ex(("127.0.0.1", port)) != 0

    def _emit(self, message: str) -> None:
        if self.status_callback is not None:
            self.status_callback(message)
