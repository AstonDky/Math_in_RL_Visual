"""书中 5x5 GridWorld 环境。

格子编码:
    0: 普通区域，白色。
    1: 禁止区域，黄色，默认可进入但会得到 r_forbidden 惩罚。
    2: 目标区域，绿色，到达后 episode 结束。

默认布局来自用户 prompt:
    [00000, 01100, 00100, 01210, 01000]
"""

from __future__ import annotations

from enum import IntEnum
from typing import Final

import numpy as np
from numpy.typing import NDArray

from core.env_base import Action, EnvBase, State, StepResult


class GridAction(IntEnum):
    """GridWorld 中动作编号的统一约定。"""

    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


ACTION_NAMES: Final[dict[int, str]] = {
    GridAction.UP: "up",
    GridAction.RIGHT: "right",
    GridAction.DOWN: "down",
    GridAction.LEFT: "left",
}


class CellType(IntEnum):
    """格子类型。"""

    NORMAL = 0
    FORBIDDEN = 1
    TARGET = 2


class GridWorld(EnvBase):
    """纯手写 5x5 GridWorld。"""

    DEFAULT_LAYOUT: Final[NDArray[np.int_]] = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 1, 2, 1, 0],
            [0, 1, 0, 0, 0],
        ],
        dtype=int,
    )

    _DELTAS: Final[dict[GridAction, tuple[int, int]]] = {
        GridAction.UP: (-1, 0),
        GridAction.RIGHT: (0, 1),
        GridAction.DOWN: (1, 0),
        GridAction.LEFT: (0, -1),
    }

    def __init__(
        self,
        layout: NDArray[np.int_] | None = None,
        start_state: State | None = None,
        r_boundary: float = -1.0,
        r_forbidden: float = -1.0,
        r_target: float = 1.0,
        r_other: float = 0.0,
        forbidden_blocks: bool = False,
    ) -> None:
        self.layout = np.array(layout if layout is not None else self.DEFAULT_LAYOUT)
        self.rows, self.cols = self.layout.shape
        self.r_boundary = float(r_boundary)
        self.r_forbidden = float(r_forbidden)
        self.r_target = float(r_target)
        self.r_other = float(r_other)
        self.forbidden_blocks = forbidden_blocks

        self._normal_states = self._collect_states(CellType.NORMAL)
        self._target_states = self._collect_states(CellType.TARGET)

        if not self._target_states:
            raise ValueError("GridWorld 至少需要一个目标格子。")

        self._start_state = (
            start_state if start_state is not None else self._normal_states[0]
        )
        if self.cell_type(self._start_state) != CellType.NORMAL:
            raise ValueError("start_state 必须指向普通格子。")

        self._current_state = self._start_state

    @property
    def num_states(self) -> int:
        return self.rows * self.cols

    @property
    def num_actions(self) -> int:
        return len(GridAction)

    @property
    def current_state(self) -> State:
        return self._current_state

    @property
    def target_states(self) -> tuple[State, ...]:
        """目标状态集合，UI 可用它高亮终点。"""

        return tuple(self._target_states)

    def reset(self, seed: int | None = None) -> State:
        """重置到固定起点。

        seed 参数预留给后续随机起点或随机转移模型；当前确定性环境不使用。
        """

        _ = seed
        self._current_state = self._start_state
        return self._current_state

    def step(self, action: Action) -> StepResult:
        """执行一步状态转移。

        如果动作越界则留在原地；禁止区域是否阻挡由 forbidden_blocks 控制。
        """

        state = self._current_state
        next_state, reward, done, extra = self.transition(state, action)
        self._current_state = next_state

        return StepResult(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            extra=extra,
        )

    def transition(
        self,
        state: State,
        action: Action,
    ) -> tuple[State, float, bool, dict[str, object]]:
        """查询一次状态转移，但不改变环境当前位置。

        TD 控制算法需要估计每个动作的后继状态和奖励；这个方法提供
        只读模型查询，让算法不必偷偷读取或修改环境内部变量。
        """

        action_enum = GridAction(action)
        row, col = self.state_to_pos(state)
        delta_row, delta_col = self._DELTAS[action_enum]
        candidate = (row + delta_row, col + delta_col)

        outside_boundary = not self._is_inside_grid(candidate)
        entered_forbidden = False
        blocked = outside_boundary
        if outside_boundary:
            next_state = state
            reward = self.r_boundary
        else:
            candidate_state = self.pos_to_state(*candidate)
            entered_forbidden = self.cell_type(candidate_state) == CellType.FORBIDDEN
            if entered_forbidden and self.forbidden_blocks:
                blocked = True
                next_state = state
                reward = self.r_forbidden
            else:
                next_state = candidate_state
                if entered_forbidden:
                    reward = self.r_forbidden
                elif self.cell_type(next_state) == CellType.TARGET:
                    reward = self.r_target
                else:
                    reward = self.r_other

        done = self.cell_type(next_state) == CellType.TARGET

        return (
            next_state,
            reward,
            done,
            {
                "position": self.state_to_pos(state),
                "next_position": self.state_to_pos(next_state),
                "blocked": blocked,
                "outside_boundary": outside_boundary,
                "entered_forbidden": entered_forbidden,
                "action_name": ACTION_NAMES[action_enum],
            },
        )

    def reward_map(self) -> NDArray[np.float64]:
        """返回每个状态格子的即时奖励，用于 Figure 4.4(b) 风格显示。"""

        rewards = np.full(self.num_states, self.r_other, dtype=float)
        for state in range(self.num_states):
            cell = self.cell_type(state)
            if cell == CellType.FORBIDDEN:
                rewards[state] = self.r_forbidden
            elif cell == CellType.TARGET:
                rewards[state] = self.r_target
        return rewards

    def state_to_pos(self, state: State) -> tuple[int, int]:
        """把一维状态编号转换成二维坐标。"""

        return divmod(state, self.cols)

    def pos_to_state(self, row: int, col: int) -> State:
        """把二维坐标转换成一维状态编号。"""

        return row * self.cols + col

    def cell_type(self, state: State) -> CellType:
        """读取某个状态对应的格子类型。"""

        row, col = self.state_to_pos(state)
        return CellType(int(self.layout[row, col]))

    def _collect_states(self, cell_type: CellType) -> list[State]:
        states: list[State] = []
        for row in range(self.rows):
            for col in range(self.cols):
                if self.layout[row, col] == cell_type:
                    states.append(self.pos_to_state(row, col))
        return states

    def _is_inside_grid(self, position: tuple[int, int]) -> bool:
        row, col = position
        return 0 <= row < self.rows and 0 <= col < self.cols
