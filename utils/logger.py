"""TensorBoard 标量记录器。"""

from __future__ import annotations

from pathlib import Path


class TensorBoardLogger:
    """对 TensorBoard 写入逻辑做一层轻量封装。

    如果用户尚未安装 tensorboard，本类不会让训练崩溃，而是静默跳过记录。
    安装依赖后再次运行即可在 ``runs/`` 目录看到曲线。
    """

    def __init__(self, log_dir: str | Path = "runs/dummy") -> None:
        self.log_dir = Path(log_dir)
        self._writer = self._build_writer()

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        if self._writer is None:
            return

        for name, value in metrics.items():
            self._writer.add_scalar(name, value, step)
        self._writer.flush()

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()

    def _build_writer(self):
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            try:
                from tensorboard.summary.writer.event_file_writer import (
                    EventFileWriter,
                )
            except ImportError:
                return None

            class _SimpleWriter:
                def __init__(self, log_dir: Path) -> None:
                    self._writer = EventFileWriter(str(log_dir))

                def add_scalar(self, tag: str, scalar_value: float, step: int) -> None:
                    from tensorboard.compat.proto.event_pb2 import Event
                    from tensorboard.compat.proto.summary_pb2 import Summary

                    summary = Summary(
                        value=[Summary.Value(tag=tag, simple_value=scalar_value)]
                    )
                    self._writer.add_event(Event(step=step, summary=summary))

                def flush(self) -> None:
                    self._writer.flush()

                def close(self) -> None:
                    self._writer.close()

            return _SimpleWriter(self.log_dir)

        return SummaryWriter(log_dir=str(self.log_dir))
