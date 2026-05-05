"""训练调度引擎。

训练循环运行在 QThread 中，通过 signal 把标准 Info Dict 发回界面。
"""

from __future__ import annotations

import gc

from PyQt6.QtCore import QMutex, QMutexLocker, QThread, pyqtSignal

from core.agent_base import AgentBase, InfoDict
from core.env_base import EnvBase
from utils.logger import TensorBoardLogger
from utils.session import TrainingSessionManager


class TrainingEngine(QThread):
    """异步训练循环。"""

    info_ready = pyqtSignal(dict)
    episode_finished = pyqtSignal(int, float)
    status_changed = pyqtSignal(str)

    def __init__(
        self,
        env: EnvBase,
        agent: AgentBase,
        logger: TensorBoardLogger | None = None,
        session_manager: TrainingSessionManager | None = None,
        delay_ms: float = 0.0,
        max_steps_per_episode: int = 200,
        save_interval_steps: int = 500,
        fast_ui_interval_steps: int = 16,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.env = env
        self.agent = agent
        self.logger = logger
        self.session_manager = session_manager
        self.delay_ms = delay_ms
        self.max_steps_per_episode = max_steps_per_episode
        self.save_interval_steps = save_interval_steps
        self.fast_ui_interval_steps = max(1, fast_ui_interval_steps)

        self._mutex = QMutex()
        self._running = False
        self._paused = True
        self._step = 0
        self._episode = 0

    def run(self) -> None:
        """QThread 入口。"""

        self._running = True
        self._paused = False
        self.status_changed.emit("running")

        state = self.env.reset()
        self.agent.reset()
        episode_reward = 0.0
        episode_steps = 0

        while self._is_running():
            if self._is_paused():
                self.msleep(40)
                continue

            action = self.agent.act(state)
            transition = self.env.step(action)
            info = self.agent.update(
                state=transition.state,
                action=transition.action,
                reward=transition.reward,
                next_state=transition.next_state,
                done=transition.done,
                step=self._step,
                episode=self._episode,
            )
            info["env_extra"] = transition.extra

            episode_reward += transition.reward
            episode_steps += 1

            if self.logger is not None:
                self.logger.log_metrics(info["metrics"], self._step)

            delay_ms = self._delay_ms()
            should_emit_info = (
                transition.done
                or self._should_emit_info(delay_ms)
            )
            if should_emit_info:
                self._emit_trace_sequence(info)
            self._step += 1

            if transition.done or episode_steps >= self.max_steps_per_episode:
                if self.logger is not None:
                    self.logger.log_metrics(
                        {
                            "rollout/episode_reward": float(episode_reward),
                            "rollout/episode_length": float(episode_steps),
                        },
                        self._episode,
                        force=True,
                    )
                self.episode_finished.emit(self._episode, episode_reward)
                self._episode += 1
                self.save_checkpoint()
                state = self.env.reset()
                self.agent.reset()
                episode_reward = 0.0
                episode_steps = 0
            else:
                state = transition.next_state

            if self._step % self.save_interval_steps == 0:
                self.save_checkpoint()

        self.save_checkpoint()
        self.status_changed.emit("stopped")

    def pause(self) -> None:
        with QMutexLocker(self._mutex):
            self._paused = True
        self.status_changed.emit("paused")

    def resume(self) -> None:
        with QMutexLocker(self._mutex):
            self._paused = False
        self.status_changed.emit("running")

    def stop(self) -> None:
        with QMutexLocker(self._mutex):
            self._running = False
        self.wait(1000)
        self.save_checkpoint()

    def restart_session(self) -> None:
        """重新开始当前算法的训练会话。"""

        if self.isRunning():
            self.stop()
        logger_template = self.logger
        if self.logger is not None:
            self.logger.close()
            self.logger = None
        if self.session_manager is not None:
            step, episode = self.session_manager.reset_run(self.agent)
            self.set_progress(step, episode)
            if logger_template is not None:
                self.logger = logger_template.clone_for_log_dir(
                    self.session_manager.log_dir
                )
            else:
                self.logger = TensorBoardLogger(log_dir=self.session_manager.log_dir)
        else:
            self.agent.reset_training_state()
            self.set_progress(0, 0)
        self.env.reset()
        gc.collect()
        self.status_changed.emit("restarted")

    def continue_session(self) -> bool:
        """从 checkpoint 恢复训练会话。"""

        if self.isRunning():
            self.stop()
        if self.session_manager is None:
            self.status_changed.emit("no session manager")
            return False
        loaded = self.session_manager.load(self.agent)
        if loaded is None:
            self.status_changed.emit("no checkpoint found")
            return False
        self.set_progress(*loaded)
        if self.logger is not None:
            self.set_logger(self.logger.clone_for_log_dir(self.session_manager.log_dir))
        else:
            self.set_logger(TensorBoardLogger(log_dir=self.session_manager.log_dir))
        self.env.reset()
        gc.collect()
        self.status_changed.emit(f"loaded checkpoint: step={loaded[0]}")
        return True

    def set_progress(self, step: int, episode: int) -> None:
        with QMutexLocker(self._mutex):
            self._step = step
            self._episode = episode

    def progress(self) -> tuple[int, int]:
        with QMutexLocker(self._mutex):
            return self._step, self._episode

    def set_logger(self, logger: TensorBoardLogger | None) -> None:
        if self.logger is not None:
            self.logger.close()
        self.logger = logger

    def save_checkpoint(self) -> None:
        if self.session_manager is None:
            return
        step, episode = self.progress()
        self.session_manager.save(self.agent, step=step, episode=episode)

    def set_delay(self, delay_ms: float) -> None:
        with QMutexLocker(self._mutex):
            self.delay_ms = delay_ms

    def _is_running(self) -> bool:
        with QMutexLocker(self._mutex):
            return self._running

    def _is_paused(self) -> bool:
        with QMutexLocker(self._mutex):
            return self._paused

    def _delay_ms(self) -> float:
        with QMutexLocker(self._mutex):
            return self.delay_ms

    def _sleep_delay(self, delay_ms: float) -> None:
        """按毫秒或亚毫秒训练间隔等待。"""

        if delay_ms <= 0.0:
            return
        if delay_ms < 1.0:
            self.usleep(max(1, int(delay_ms * 1000)))
        else:
            self.msleep(int(delay_ms))

    def _should_emit_info(self, delay_ms: float) -> bool:
        """高速训练时降低界面刷新频率。"""

        if delay_ms >= 1.0:
            return True
        return self._step % self.fast_ui_interval_steps == 0

    def _emit_trace_sequence(self, info: InfoDict) -> None:
        """把一次更新展开成算法指针的逐行展示事件。"""

        trace = info.get("algorithm_trace")
        delay_ms = self._delay_ms()
        if not trace:
            self.info_ready.emit(dict(info))
            self._sleep_delay(delay_ms)
            return

        lines = trace.get("lines", [])
        stage_count = max(1, len(lines))
        if delay_ms <= 0.0:
            staged = dict(info)
            staged_trace = dict(trace)
            staged_trace["current_line"] = stage_count
            staged["algorithm_trace"] = staged_trace
            staged["metric_step"] = info.get("step", 0)
            self.info_ready.emit(staged)
            return

        stage_delay = 0.0 if delay_ms <= 0.0 else delay_ms / stage_count

        for line_number in range(1, stage_count + 1):
            if not self._is_running() or self._is_paused():
                break

            staged = dict(info)
            staged_trace = dict(trace)
            staged_trace["current_line"] = line_number
            staged["algorithm_trace"] = staged_trace
            if line_number != stage_count:
                staged["metrics"] = {}
                staged["metric_step"] = None
            else:
                staged["metric_step"] = info.get("step", 0)
            self.info_ready.emit(staged)
            self._sleep_delay(stage_delay)
