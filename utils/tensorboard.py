"""TensorBoard launcher."""

from __future__ import annotations

import os
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
    """Start TensorBoard and open the browser after the HTTP server is ready."""

    def __init__(
        self,
        log_dir: str | Path,
        preferred_port: int = 6006,
        open_timeout_seconds: float = 60.0,
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
        self._server_url: str | None = None
        self._log_file = None
        self._open_thread: threading.Thread | None = None
        self._open_generation = 0

    def start(self) -> bool:
        if not self.is_available():
            self._emit("TensorBoard is not installed in the MathInRL environment.")
            return False

        if self.is_running():
            self._open_when_ready()
            return True

        self.log_dir.mkdir(parents=True, exist_ok=True)
        try:
            self.port = self._choose_port(self.preferred_port)
        except RuntimeError as error:
            self._emit(f"TensorBoard failed to start: {error}")
            return False

        run_name = self.log_dir.name
        run_id = f"{run_name}_{int(time.time() * 1000)}"
        self._server_url = f"http://127.0.0.1:{self.port}"
        self.url = f"{self._server_url}/?reload={run_id}#timeseries"
        log_path = self.log_dir.parent / f"{run_name}_tensorboard_{getpid()}_{int(time.time())}.log"
        self._log_file = log_path.open("w", encoding="utf-8")
        command = [
            sys.executable,
            "-m",
            "tensorboard.main",
            "--logdir_spec",
            f"{run_name}:{self.log_dir.resolve()}",
            "--host",
            "127.0.0.1",
            "--port",
            str(self.port),
            "--reload_interval",
            f"{self.reload_interval_seconds:g}",
            "--window_title",
            f"TensorBoard {run_name}",
        ]
        try:
            self.process = subprocess.Popen(
                command,
                stdout=self._log_file,
                stderr=subprocess.STDOUT,
            )
        except OSError as error:
            self._emit(f"TensorBoard failed to start: {error}")
            if self._log_file is not None:
                self._log_file.close()
                self._log_file = None
            return False

        self._emit(f"TensorBoard starting: {self.url}")
        self._open_when_ready()
        return True

    def stop(self) -> None:
        self._open_generation += 1
        process = self.process
        try:
            if process is not None and process.poll() is None:
                self._terminate_process(process)
        finally:
            self.process = None
            self.port = None
            self.url = None
            self._server_url = None
            self._close_log_file()
            if self._open_thread is not None and self._open_thread.is_alive():
                self._open_thread.join(timeout=1.0)
            self._open_thread = None

    def is_running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def set_log_dir(self, log_dir: str | Path) -> None:
        """Point future TensorBoard launches at a new log directory."""

        self.stop()
        self.log_dir = Path(log_dir)
        self.port = None
        self.url = None
        self._server_url = None

    def is_available(self) -> bool:
        return (
            find_spec("tensorboard") is not None
            and find_spec("pkg_resources") is not None
        )

    def _open_when_ready(self) -> None:
        if self.url is None or self._server_url is None:
            return
        if self._open_thread is not None and self._open_thread.is_alive():
            return
        self._open_generation += 1
        self._open_thread = threading.Thread(
            target=self._wait_and_open,
            args=(self.url, self._server_url, self._open_generation),
            daemon=True,
        )
        self._open_thread.start()

    def _wait_and_open(
        self,
        browser_url: str,
        server_url: str,
        generation: int,
    ) -> None:
        deadline = time.monotonic() + self.open_timeout_seconds
        while time.monotonic() < deadline:
            if generation != self._open_generation:
                return
            if self.process is not None and self.process.poll() is not None:
                self._emit("TensorBoard failed to start; see runs/*_tensorboard.log.")
                return
            if self._is_http_ready(server_url):
                if generation != self._open_generation:
                    return
                opened = self._open_url(browser_url)
                if not opened:
                    self._emit(f"TensorBoard ready: {browser_url}")
                    return
                self._emit(f"TensorBoard opened: {browser_url}")
                return
            time.sleep(0.25)

        self._emit(f"TensorBoard was not ready within {self.open_timeout_seconds:.0f}s.")

    def _is_http_ready(self, url: str) -> bool:
        try:
            with urlopen(url, timeout=0.5) as response:
                return 200 <= response.status < 500
        except (Exception, OSError, URLError):
            return False

    def _open_url(self, url: str) -> bool:
        try:
            if webbrowser.open(url, new=2):
                return True
        except webbrowser.Error:
            pass

        if sys.platform.startswith("win"):
            try:
                os.startfile(url)  # type: ignore[attr-defined]
                return True
            except OSError:
                pass
            try:
                subprocess.Popen(
                    ["cmd", "/c", "start", "", url],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    stdin=subprocess.DEVNULL,
                    creationflags=subprocess.CREATE_NO_WINDOW,
                )
                return True
            except OSError:
                return False
        return False

    def _terminate_process(self, process: subprocess.Popen) -> None:
        if sys.platform.startswith("win"):
            try:
                subprocess.run(
                    ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    stdin=subprocess.DEVNULL,
                    timeout=5,
                    check=False,
                    creationflags=subprocess.CREATE_NO_WINDOW,
                )
            except (OSError, subprocess.TimeoutExpired):
                process.kill()
        else:
            process.terminate()

        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)

    def _close_log_file(self) -> None:
        if self._log_file is None:
            return
        try:
            self._log_file.flush()
        except OSError:
            pass
        try:
            self._log_file.close()
        finally:
            self._log_file = None

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
