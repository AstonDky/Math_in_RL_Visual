"""TensorBoard 标量记录器。"""

from __future__ import annotations

import time
from pathlib import Path


class TensorBoardLogger:
    """对 TensorBoard 标量写入做一层轻量封装。"""

    def __init__(
        self,
        log_dir: str | Path = "runs/dummy",
        log_interval_steps: int = 1,
        flush_interval_steps: int = 5,
        flush_interval_seconds: float = 1.0,
    ) -> None:
        self.log_dir = Path(log_dir)
        self.log_interval_steps = max(1, log_interval_steps)
        self.flush_interval_steps = max(1, flush_interval_steps)
        self.flush_interval_seconds = max(0.1, flush_interval_seconds)
        self._last_flush_step = -1
        self._last_flush_time = 0.0
        self._writer = self._build_writer()

    def log_metrics(
        self,
        metrics: dict[str, float],
        step: int,
        force: bool = False,
    ) -> None:
        if self._writer is None or not metrics:
            return
        if not force and step % self.log_interval_steps != 0:
            return

        from tensorboard.compat.proto.event_pb2 import Event
        from tensorboard.compat.proto.summary_pb2 import Summary

        now = time.time()
        for name, value in metrics.items():
            summary = Summary(
                value=[Summary.Value(tag=name, simple_value=float(value))]
            )
            self._writer.add_event(Event(wall_time=now, step=step, summary=summary))
        if self._should_flush(step=step, now=now, force=force):
            self._writer.flush()
            self._last_flush_step = step
            self._last_flush_time = now

    def close(self) -> None:
        if self._writer is not None:
            self._writer.flush()
            self._writer.close()

    def clone_for_log_dir(self, log_dir: str | Path) -> "TensorBoardLogger":
        return TensorBoardLogger(
            log_dir=log_dir,
            log_interval_steps=self.log_interval_steps,
            flush_interval_steps=self.flush_interval_steps,
            flush_interval_seconds=self.flush_interval_seconds,
        )

    def _build_writer(self):
        try:
            from tensorboard.summary.writer.event_file_writer import EventFileWriter
        except ImportError:
            return None
        return EventFileWriter(str(self.log_dir))

    def _should_flush(self, step: int, now: float, force: bool) -> bool:
        if force or step == 0:
            return True
        return (
            step - self._last_flush_step >= self.flush_interval_steps
            or now - self._last_flush_time >= self.flush_interval_seconds
        )
