"""训练调度引擎。

PyQt6 的 UI 线程必须保持轻量，否则窗口会卡顿。
因此训练循环放在 QThread 中运行，并通过 signal 把标准 Info Dict 发回 UI。
"""

from __future__ import annotations

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
        delay_ms: float = 300.0,
        max_steps_per_episode: int = 200,
        save_interval_steps: int = 25,
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

        self._mutex = QMutex()
        self._running = False
        self._paused = True
        self._step = 0
        self._episode = 0

    def run(self) -> None:
        """QThread 入口：循环采样、更新、发射 Info Dict。"""

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

            self._emit_trace_sequence(info)
            self._step += 1

            if transition.done or episode_steps >= self.max_steps_per_episode:
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
        """支持无限接近 0 的亚毫秒延迟。"""

        if delay_ms <= 0.0:
            return
        if delay_ms < 1.0:
            self.usleep(max(1, int(delay_ms * 1000)))
        else:
            self.msleep(int(delay_ms))

    def _emit_trace_sequence(self, info: InfoDict) -> None:
        """把一次算法更新拆成多次 UI 指针事件。

        算法计算仍然是原子完成的；这里负责教学展示，让代码指针在
        自动解析出的核心函数源码行之间移动。最后一行才携带 metrics。
        """

        trace = info.get("algorithm_trace")
        if not trace:
            self.info_ready.emit(dict(info))
            self._sleep_delay(self._delay_ms())
            return

        lines = trace.get("lines", [])
        stage_count = max(1, len(lines))
        delay_ms = self._delay_ms()
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
