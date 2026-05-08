"""Training session storage for checkpoints and TensorBoard logs."""

from __future__ import annotations

import pickle
import shutil
import time
from pathlib import Path
from typing import Any

from core.agent_base import AgentBase


class TrainingSessionManager:
    """Owns one algorithm's checkpoint and log directories."""

    def __init__(
        self,
        algorithm_name: str,
        root_dir: str | Path = "sessions",
        runs_dir: str | Path = "runs",
    ) -> None:
        self.algorithm_name = algorithm_name
        self.root_dir = Path(root_dir)
        self.runs_dir = Path(runs_dir)
        self.session_dir = self.root_dir / algorithm_name
        self.checkpoint_path = self.session_dir / "checkpoint.pkl"
        self.algorithm_log_root = self.runs_dir / algorithm_name
        self.log_dir = self.algorithm_log_root
        self.last_cleanup_failures: list[Path] = []

    def save(
        self,
        agent: AgentBase,
        step: int,
        episode: int,
    ) -> None:
        self.session_dir.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "algorithm_name": self.algorithm_name,
            "step": step,
            "episode": episode,
            "agent_state": agent.state_dict(),
            "log_dir": str(self.log_dir),
        }
        with self.checkpoint_path.open("wb") as file:
            pickle.dump(payload, file)

    def load(self, agent: AgentBase) -> tuple[int, int] | None:
        if not self.checkpoint_path.exists():
            return None
        with self.checkpoint_path.open("rb") as file:
            payload = pickle.load(file)
        log_dir = payload.get("log_dir")
        if log_dir:
            self.log_dir = Path(log_dir)
        agent.load_state_dict(payload["agent_state"])
        return int(payload.get("step", 0)), int(payload.get("episode", 0))

    def reset_run(self, agent: AgentBase) -> tuple[int, int]:
        """Reset model state and delete this algorithm's stored run data."""

        agent.reset_training_state()
        self.last_cleanup_failures = []
        self._delete_algorithm_tensorboard_logs()
        self._remove_path(self.session_dir)
        self._remove_path(self.algorithm_log_root)
        self.log_dir = self._new_run_log_dir()
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        return 0, 0

    def clear_logs(self) -> None:
        """Delete this algorithm's TensorBoard logs and switch to a fresh run."""

        self.last_cleanup_failures = []
        self._delete_algorithm_tensorboard_logs()
        self._remove_path(self.algorithm_log_root)
        self.log_dir = self._new_run_log_dir()
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def clear_all_runs(self) -> None:
        """Delete all runtime artifacts under the shared runs directory."""

        self.last_cleanup_failures = []
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self._clear_directory_contents(self.runs_dir)

    def _clear_directory_contents(self, directory: Path) -> None:
        if not directory.exists():
            return
        for path in directory.iterdir():
            self._remove_path(path)

    def _delete_algorithm_tensorboard_logs(self) -> None:
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.algorithm_log_root.mkdir(parents=True, exist_ok=True)
        tensorboard_logs = list(
            self.runs_dir.glob(f"{self.algorithm_name}_tensorboard*.log")
        )
        tensorboard_logs.extend(self.algorithm_log_root.glob("*_tensorboard*.log"))
        for tensorboard_log in tensorboard_logs:
            self._remove_path(tensorboard_log)

    def _remove_path(self, path: Path) -> None:
        if not path.exists():
            return
        for _ in range(5):
            try:
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
                return
            except FileNotFoundError:
                return
            except PermissionError:
                time.sleep(0.1)
        self.last_cleanup_failures.append(path)

    def _new_run_log_dir(self) -> Path:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        candidate = self.algorithm_log_root / f"run_{timestamp}"
        suffix = 1
        while candidate.exists():
            candidate = self.algorithm_log_root / f"run_{timestamp}_{suffix}"
            suffix += 1
        return candidate
