"""TensorBoard 标量记录器。"""

from __future__ import annotations

from pathlib import Path


class TensorBoardLogger:
    """对 TensorBoard 标量写入做一层轻量封装。"""

    def __init__(
        self,
        log_dir: str | Path = "runs/dummy",
        flush_interval_steps: int = 10,
    ) -> None:
        self.log_dir = Path(log_dir)
        self.flush_interval_steps = max(1, flush_interval_steps)
        self._last_flush_step = -1
        self._writer = self._build_writer()

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        if self._writer is None:
            return

        from tensorboard.compat.proto.event_pb2 import Event
        from tensorboard.compat.proto.summary_pb2 import Summary

        for name, value in metrics.items():
            summary = Summary(
                value=[Summary.Value(tag=name, simple_value=float(value))]
            )
            self._writer.add_event(Event(step=step, summary=summary))
        if step == 0 or step - self._last_flush_step >= self.flush_interval_steps:
            self._writer.flush()
            self._last_flush_step = step

    def close(self) -> None:
        if self._writer is not None:
            self._writer.flush()
            self._writer.close()

    def _build_writer(self):
        try:
            from tensorboard.summary.writer.event_file_writer import EventFileWriter
        except ImportError:
            return None
        return EventFileWriter(str(self.log_dir))
