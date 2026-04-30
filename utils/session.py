"""训练会话管理。

负责把“重新运行”和“继续上一次运行”从 UI/算法中解耦出来：

- 重新运行：清理 checkpoint、TensorBoard 日志和算法内存状态。
- 继续运行：加载上次保存的算法参数与 step/episode 计数。
- 训练中：周期性保存，避免关闭窗口后丢失结果。
"""

from __future__ import annotations

import pickle
import shutil
from pathlib import Path
from typing import Any

from core.agent_base import AgentBase


class TrainingSessionManager:
    """单个算法的训练会话管理器。"""

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
        self.log_dir = self.runs_dir / algorithm_name

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
        }
        with self.checkpoint_path.open("wb") as file:
            pickle.dump(payload, file)

    def load(self, agent: AgentBase) -> tuple[int, int] | None:
        if not self.checkpoint_path.exists():
            return None
        with self.checkpoint_path.open("rb") as file:
            payload = pickle.load(file)
        agent.load_state_dict(payload["agent_state"])
        return int(payload.get("step", 0)), int(payload.get("episode", 0))

    def reset_run(self, agent: AgentBase) -> tuple[int, int]:
        """清理旧训练数据，并重置算法内存状态。"""

        agent.reset_training_state()
        if self.checkpoint_path.exists():
            self.checkpoint_path.unlink()
        if self.log_dir.exists():
            shutil.rmtree(self.log_dir)
        self.session_dir.mkdir(parents=True, exist_ok=True)
        return 0, 0

