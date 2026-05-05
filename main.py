"""Math in RL Visual 应用入口。"""

from __future__ import annotations

import sys

from PyQt6.QtWidgets import QApplication

from algorithms.sarsa_value_function import sarsa_fa
from core.agent_base import AgentBase
from core.algorithm_adapters import adapt_algorithm
from core.engine import TrainingEngine
from core.rl_parameters import RLAlgorithmConfig
from core.table_agent import TableAgent
from envs.grid_world import GridWorld
from ui.main_window import MainWindow
from utils.hardware import detect_hardware
from utils.logger import TensorBoardLogger
from utils.session import TrainingSessionManager
from utils.tensorboard import TensorBoardLauncher


CORE_ALGORITHM = sarsa_fa
ALGORITHM_NAME = CORE_ALGORITHM.__name__
STARTUP_MODE = "restart"  # "restart" 新训练；"continue" 读 checkpoint。
ALGORITHM_CONFIG = RLAlgorithmConfig(
    alpha=0.02,
    gamma=0.9,
    epsilon=0.05,
    behavior_policy="epsilon_greedy",
    auto_refresh_policy=False,
    softmax_temperature=1.0,
    value_function="linear",
    feature_kind="one_hot",
    feature_order=0,
    weight_init="zeros",
)
TENSORBOARD_LOG_EVERY_STEPS = 1
TENSORBOARD_FLUSH_EVERY_STEPS = 5
TENSORBOARD_FLUSH_EVERY_SECONDS = 1.0
TENSORBOARD_RELOAD_SECONDS = 1.0


def build_agent(env: GridWorld) -> AgentBase:
    """根据 main.py 顶部配置创建训练 Agent。"""

    return TableAgent(
        num_states=env.num_states,
        num_actions=env.num_actions,
        reward_map=env.reward_map(),
        algorithm=adapt_algorithm(CORE_ALGORITHM),
        config=ALGORITHM_CONFIG,
        algorithm_name=ALGORITHM_NAME,
        state_shape=env.layout.shape,
    )


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
    logger = TensorBoardLogger(
        log_dir=session_manager.log_dir,
        log_interval_steps=TENSORBOARD_LOG_EVERY_STEPS,
        flush_interval_steps=TENSORBOARD_FLUSH_EVERY_STEPS,
        flush_interval_seconds=TENSORBOARD_FLUSH_EVERY_SECONDS,
    )
    engine = TrainingEngine(
        env=env,
        agent=agent,
        logger=logger,
        session_manager=session_manager,
    )
    engine.set_progress(*progress)
    hardware = detect_hardware()
    tensorboard = TensorBoardLauncher(
        log_dir=session_manager.log_dir,
        preferred_port=6006,
        reload_interval_seconds=TENSORBOARD_RELOAD_SECONDS,
        status_callback=engine.status_changed.emit,
    )

    window = MainWindow(env=env, engine=engine, tensorboard=tensorboard)
    window.show()
    tensorboard_status = "TensorBoard 将在点击开始训练后自动打开"
    window.statusBar().showMessage(
        f"{hardware.backend} | {hardware.note} | {startup_status} | {tensorboard_status}"
    )

    exit_code = app.exec()
    tensorboard.stop()
    engine.set_logger(None)
    return exit_code


def prepare_startup_session(
    mode: str,
    session_manager: TrainingSessionManager,
    agent: AgentBase,
) -> tuple[str, tuple[int, int]]:
    """准备启动时的训练会话。"""

    if mode == "continue":
        loaded = session_manager.load(agent)
        if loaded is not None:
            return f"继续上次: step={loaded[0]}", loaded

    step, episode = session_manager.reset_run(agent)
    return "新训练: 已清理 checkpoint 和 TensorBoard 日志", (step, episode)


if __name__ == "__main__":
    raise SystemExit(main())
