"""Math in RL Visual 应用入口。

切换算法时，主函数只负责选择核心算法函数与参数配置。
框架会在 ``core.algorithm_adapters`` 中按函数签名自动选适配器；
UI、训练引擎、会话管理和保存恢复层保持不变。
"""

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
STARTUP_MODE = "restart"  # "restart" 清空旧数据；"continue" 恢复 checkpoint。
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


def build_agent(env: GridWorld) -> AgentBase:
    """算法热插拔入口：只改上面的 CORE_ALGORITHM 和 ALGORITHM_CONFIG。"""

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
    logger = TensorBoardLogger(log_dir=session_manager.log_dir)
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
    """程序启动时明确选择新训练或继续上次，避免 TensorBoard 混入旧数据。"""

    if mode == "continue":
        loaded = session_manager.load(agent)
        if loaded is not None:
            return f"继续上次: step={loaded[0]}", loaded

    step, episode = session_manager.reset_run(agent)
    return "新训练: 已清理 checkpoint 和 TensorBoard 日志", (step, episode)


if __name__ == "__main__":
    raise SystemExit(main())
