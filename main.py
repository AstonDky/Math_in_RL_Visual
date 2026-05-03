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

from algorithms.sarsa_value_function import sarsa_fa
from core.agent_base import AgentBase
from core.algorithm_adapters import adapt_sarsa_fa
from core.engine import TrainingEngine
from core.rl_parameters import RLAlgorithmConfig
from core.table_agent import TableAgent
from envs.grid_world import GridWorld
from ui.main_window import MainWindow
from utils.hardware import detect_hardware
from utils.logger import TensorBoardLogger
from utils.session import TrainingSessionManager


ALGORITHM_NAME = "sarsa_value_function_8_2"
CORE_ALGORITHM = adapt_sarsa_fa(sarsa_fa)
STARTUP_MODE = "restart"  # "restart" 清空旧数据；"continue" 恢复 checkpoint。
ALGORITHM_CONFIG = RLAlgorithmConfig(
    alpha=0.001,
    gamma=0.9,
    epsilon=0.1,
    behavior_policy="epsilon_greedy",
    auto_refresh_policy=False,
    value_function="linear",
    feature_kind="fourier",
    feature_order=5,
    weight_init="normal",
)


def build_agent(env: GridWorld) -> AgentBase:
    """算法热插拔入口：只改上面的 CORE_ALGORITHM 和 ALGORITHM_CONFIG。"""

    return TableAgent(
        num_states=env.num_states,
        num_actions=env.num_actions,
        reward_map=env.reward_map(),
        algorithm=CORE_ALGORITHM,
        config=ALGORITHM_CONFIG,
        algorithm_name=ALGORITHM_NAME,
        state_shape=env.layout.shape,
    )


def start_tensorboard(log_dir: Path, port: int = 6006) -> subprocess.Popen | None:
    """随应用启动当前算法的 TensorBoard 页面。"""

    if find_spec("tensorboard") is None:
        return None

    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = (log_dir.parent / f"{log_dir.name}_tensorboard.log").open(
        "w",
        encoding="utf-8",
    )
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
    process = subprocess.Popen(command, stdout=log_file, stderr=subprocess.STDOUT)
    webbrowser.open(f"http://localhost:{port}")
    return process


def main() -> int:
    app = QApplication(sys.argv)

    env = GridWorld(
        r_boundary=-10.0,
        r_forbidden=-10.0,
        r_target=1.0,
        r_other=0.0,
        forbidden_blocks=False,
    )
    agent = build_agent(env)
    session_manager = TrainingSessionManager(ALGORITHM_NAME)
    startup_status, progress = prepare_startup_session(
        mode=STARTUP_MODE,
        session_manager=session_manager,
        agent=agent,
    )
    logger = TensorBoardLogger(log_dir=session_manager.log_dir)
    tensorboard_process = start_tensorboard(session_manager.log_dir)
    engine = TrainingEngine(
        env=env,
        agent=agent,
        logger=logger,
        session_manager=session_manager,
    )
    engine.set_progress(*progress)
    hardware = detect_hardware()

    window = MainWindow(env=env, engine=engine)
    window.show()
    tensorboard_status = (
        "TensorBoard: http://localhost:6006"
        if tensorboard_process is not None
        else "TensorBoard 未启动，请检查依赖"
    )
    window.statusBar().showMessage(
        f"{hardware.backend} | {hardware.note} | {startup_status} | {tensorboard_status}"
    )

    exit_code = app.exec()
    if tensorboard_process is not None:
        tensorboard_process.terminate()
    engine.set_logger(None)
    return exit_code


def prepare_startup_session(
    mode: str,
    session_manager: TrainingSessionManager,
    agent: AgentBase,
) -> tuple[str, tuple[int, int]]:
    """程序启动时明确选择新训练或继续上次，避免 TensorBoard 混入旧数据。"""

    if mode == "continue":
        loaded = session_manager.load(agent)
        if loaded is not None:
            return f"继续上次: step={loaded[0]}", loaded

    step, episode = session_manager.reset_run(agent)
    return "新训练: 已清理 checkpoint 和 TensorBoard 日志", (step, episode)


if __name__ == "__main__":
    raise SystemExit(main())
