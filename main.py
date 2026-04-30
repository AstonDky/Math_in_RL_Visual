"""Math in RL Visual 应用入口。

切换算法的核心位置只保留在 ``build_agent`` 中。
后续你实现 MonteCarloAgent、TDAgent、QLearningAgent 等算法后，只需要把
return DummyAgent(...) 改成 return NewAgent(...)，UI 和训练引擎无需修改。
"""

from __future__ import annotations

import subprocess
import sys
import webbrowser
from importlib.util import find_spec
from pathlib import Path

from PyQt6.QtWidgets import QApplication

from algorithms.on_policy_q_learning_agent import GreedyQLearningAgent
from core.agent_base import AgentBase
from core.engine import TrainingEngine
from envs.grid_world import GridWorld
from ui.main_window import MainWindow
from utils.hardware import detect_hardware
from utils.logger import TensorBoardLogger


def build_agent(env: GridWorld) -> AgentBase:
    """算法热插拔入口：当前使用 greedy Q-learning。"""

    return GreedyQLearningAgent(
        num_states=env.num_states,
        num_actions=env.num_actions,
        reward_map=env.reward_map(),
    )


def start_tensorboard(log_dir: Path, port: int = 6006) -> subprocess.Popen | None:
    """随应用启动 TensorBoard；未安装时不阻塞主界面。"""

    if find_spec("tensorboard") is None:
        return None

    command = [
        sys.executable,
        "-m",
        "tensorboard.main",
        "--logdir",
        str(log_dir),
        "--port",
        str(port),
        "--reload_interval",
        "2",
    ]
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        return None

    webbrowser.open(f"http://localhost:{port}")
    return process


def main() -> int:
    app = QApplication(sys.argv)

    log_dir = Path("runs/greedy_q_learning")
    tensorboard_process = start_tensorboard(Path("runs"))

    env = GridWorld()
    agent = build_agent(env)
    logger = TensorBoardLogger(log_dir=log_dir)
    engine = TrainingEngine(env=env, agent=agent, logger=logger)
    hardware = detect_hardware()

    window = MainWindow(env=env, engine=engine)
    window.show()
    window.statusBar().showMessage(
        f"{hardware.backend} | {hardware.note} | TensorBoard: http://localhost:6006"
    )

    exit_code = app.exec()
    if tensorboard_process is not None:
        tensorboard_process.terminate()
    logger.close()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
