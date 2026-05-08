"""书中风格算法函数的框架适配层。"""

from __future__ import annotations

import ast
import copy
import inspect
import textwrap
from dataclasses import dataclass
from typing import Any

import numpy as np

from core.env_base import Action, EnvBase, State, StepResult
from core.rl_parameters import (
    CoreAlgorithm,
    EpisodeAlgorithm,
    RLAlgorithmContext,
    RLEpisodeResult,
    RLStepEvent,
    RLStepResult,
    RLTransition,
    TrainingMode,
    make_policy_probs,
)


BookAlgorithm = Any
_SARSA_FA_REQUIRED_PARAMS = ("w", "s0", "pi", "q_hat", "grad_q_hat", "step")
_SARSA_FA_SUPPORTED_OPTIONAL_PARAMS = {
    "alpha",
    "gamma",
    "epsilon",
    "episodes",
    "max_steps",
}
_EPISODE_COUNT_NAMES = ("num_episodes", "episodes")
_ITERATION_COUNT_NAMES = ("iterations", "num_iterations", "max_iterations", "sweeps", "num_sweeps")
_COUNT_PARAM_NAMES = _EPISODE_COUNT_NAMES + _ITERATION_COUNT_NAMES
_KNOWN_RUNTIME_STATE_NAMES = {
    "q",
    "v",
    "w",
    "theta",
    "policy",
    "e",
    "z",
    "h",
    "baseline",
    "b",
    "avg_reward",
    "average_reward",
    "reward_bar",
    "returns",
    "num",
    "counts",
    "v_w",
    "target_w",
    "replay_buffer",
    "w1",
    "b1",
    "w2",
    "b2",
    "target_w1",
    "target_b1",
    "target_w2",
    "target_b2",
}
_HELPER_PARAM_NAMES = {
    "actions",
    "env_model",
    "grad_log_policy",
    "grad_q",
    "grad_q_hat",
    "grad_v_hat",
    "mdp",
    "model",
    "num_actions",
    "num_states",
    "pi",
    "policy_probs",
    "q_hat",
    "reward",
    "sample_from_policy",
    "softmax",
    "states",
    "step",
    "transition",
    "v_hat",
}
_CONFIG_PARAM_RESOLVERS = {
    "alpha": lambda ctx: ctx.config.alpha,
    "beta": lambda ctx: ctx.config.beta,
    "gamma": lambda ctx: ctx.config.gamma,
    "epsilon": lambda ctx: ctx.config.epsilon,
    "lambda_": lambda ctx: ctx.config.lambda_,
    "n": lambda ctx: ctx.config.n_steps,
    "n_steps": lambda ctx: ctx.config.n_steps,
    "planning_steps": lambda ctx: ctx.config.planning_steps,
    "batch_size": lambda ctx: ctx.config.batch_size,
    "target_update_interval": lambda ctx: ctx.config.target_update_interval,
    "hidden_units": lambda ctx: ctx.config.hidden_units,
    "max_steps": lambda ctx: ctx.config.max_steps,
    "iterations": lambda ctx: ctx.config.iterations,
    "num_iterations": lambda ctx: ctx.config.iterations,
    "max_iterations": lambda ctx: ctx.config.iterations,
    "sweeps": lambda ctx: ctx.config.iterations,
    "num_sweeps": lambda ctx: ctx.config.iterations,
    "temperature": lambda ctx: ctx.config.temperature
    if ctx.config.temperature is not None
    else ctx.config.softmax_temperature,
    "alpha_theta": lambda ctx: ctx.config.alpha_theta
    if ctx.config.alpha_theta is not None
    else ctx.config.alpha,
    "alpha_w": lambda ctx: ctx.config.alpha_w
    if ctx.config.alpha_w is not None
    else ctx.config.alpha,
    "alpha_v": lambda ctx: ctx.config.alpha_v
    if ctx.config.alpha_v is not None
    else ctx.config.alpha,
    "seed": lambda ctx: ctx.config.seed,
}
_TRACE_LOCALS = (
    "s",
    "state",
    "state_idx",
    "a",
    "action",
    "r",
    "reward",
    "s_next",
    "next_state",
    "state_next",
    "a_next",
    "done",
    "delta",
    "td_error",
    "advantage",
    "g",
    "g_t",
    "return_t",
    "q_sa",
    "q_next",
    "loss",
)
_INSTRUMENTED_EPISODE_CACHE: dict[int, tuple[object, str]] = {}


@dataclass(slots=True)
class AdaptedAlgorithm:
    """框架内部统一的算法适配结果。"""

    mode: TrainingMode
    display_function: BookAlgorithm
    transition_runner: CoreAlgorithm | None = None
    episode_runner: EpisodeAlgorithm | None = None

    @property
    def name(self) -> str:
        return self.display_function.__name__


def adapt_algorithm(book_algorithm: BookAlgorithm) -> AdaptedAlgorithm:
    """按函数签名选择算法族适配器。"""

    if not callable(book_algorithm):
        raise TypeError(
            "CORE_ALGORITHM 必须指向核心算法函数名称，而不是模块对象。"
    )
    if _matches_sarsa_fa_signature(book_algorithm):
        return adapt_sarsa_fa_episode(book_algorithm)
    if _matches_episode_env_signature(book_algorithm):
        return adapt_episode_env_algorithm(book_algorithm)

    signature = inspect.signature(book_algorithm)
    raise ValueError(
        "No framework adapter matches algorithm "
        f"{book_algorithm.__name__}{signature}. "
        "Add a new signature matcher in core/algorithm_adapters.py instead of "
        "changing engine/UI/session code."
    )


def adapt_sarsa_fa(book_algorithm: BookAlgorithm) -> AdaptedAlgorithm:
    """适配 ``(..., step, pi, q_hat, grad_q_hat, ...)`` 风格的一步更新算法。"""

    parameter_names = tuple(inspect.signature(book_algorithm).parameters)

    def core_algorithm(
        ctx: RLAlgorithmContext,
        transition: RLTransition,
    ) -> RLStepResult:
        _ensure_linear_action_value_weights(ctx)
        w_t = ctx.tables.w.copy()
        first_action_used = False
        sampled_next_action: Action | None = None

        def pi(
            state: State,
            w: np.ndarray,
            epsilon: float | None = None,
        ) -> Action:
            nonlocal first_action_used, sampled_next_action
            if not first_action_used and state == transition.state:
                first_action_used = True
                return transition.action

            effective_epsilon = (
                ctx.config.epsilon if epsilon is None else float(epsilon)
            )
            q_values = _q_values_from_w(ctx, state, w)
            probs = make_policy_probs(
                q_values,
                ctx.config.behavior_policy,
                effective_epsilon,
                ctx.config.softmax_temperature,
            )
            action = int(ctx.rng.choice(ctx.num_actions, p=probs))
            sampled_next_action = action
            return action

        def q_hat(state: State, action: Action, w: np.ndarray) -> float:
            return float(np.dot(ctx.tables.features[state, action], w))

        def grad_q_hat(state: State, action: Action, w: np.ndarray) -> np.ndarray:
            _ = w
            return ctx.tables.features[state, action].copy()

        def step(state: State, action: Action) -> tuple[State, float, bool]:
            _ = state, action
            return transition.next_state, transition.reward, transition.done

        q_hat_t = q_hat(transition.state, transition.action, w_t)
        call_kwargs = _build_supported_kwargs(ctx, parameter_names)
        w_next = book_algorithm(
            w_t,
            transition.state,
            pi,
            q_hat,
            grad_q_hat,
            step,
            **call_kwargs,
        )
        q_hat_t_plus_1 = (
            0.0
            if transition.done or sampled_next_action is None
            else q_hat(transition.next_state, sampled_next_action, w_t)
        )
        td_error = transition.reward + ctx.config.gamma * q_hat_t_plus_1 - q_hat_t

        ctx.tables.w = np.asarray(w_next, dtype=float)
        ctx.sync_q_from_w()
        ctx.refresh_all_policies()
        policy_after = ctx.tables.policy[transition.state].copy()
        q_hat_after = ctx.q_hat(transition.state, transition.action)
        state_value_after = ctx.tables.v[transition.state]

        return RLStepResult(
            formula=(
                "w = w + alpha * [r + gamma*q_hat(s_next,a_next,w) "
                "- q_hat(s,a,w)] * grad_q_hat(s,a,w)"
            ),
            calculation=(
                f"delta = {transition.reward:.3f} + {ctx.config.gamma:.3f} * "
                f"{q_hat_t_plus_1:.3f} - {q_hat_t:.3f} = {td_error:.3f}; "
                f"q_hat({transition.state},{transition.action}) -> {q_hat_after:.3f}"
            ),
            variables={
                "s": transition.state,
                "a": transition.action,
                "r": round(float(transition.reward), 3),
                "s_next": transition.next_state,
                "a_next": sampled_next_action,
                "q_hat": round(float(q_hat_t), 3),
                "q_hat_next": round(float(q_hat_t_plus_1), 3),
                "delta": round(float(td_error), 3),
                "w_norm": round(float(np.linalg.norm(ctx.tables.w)), 3),
                "epsilon": round(float(ctx.config.epsilon), 3),
                "pi(a|s)": np.round(policy_after, 3).tolist(),
                "V_pi(s)": round(float(state_value_after), 3),
            },
            metrics={
                "rollout/reward": float(transition.reward),
                "train/loss": float(td_error * td_error),
                "train/td_error": float(td_error),
                "train/q_value": float(q_hat_after),
                "train/state_value": float(state_value_after),
                "train/w_norm": float(np.linalg.norm(ctx.tables.w)),
            },
            updated_state=transition.state,
            next_action=sampled_next_action,
        )

    return AdaptedAlgorithm(
        mode="transition",
        display_function=book_algorithm,
        transition_runner=_inherit_book_metadata(core_algorithm, book_algorithm),
    )


def adapt_sarsa_fa_episode(book_algorithm: BookAlgorithm) -> AdaptedAlgorithm:
    """Adapt Algorithm 8.2 as a book-style episode training function."""

    parameter_names = tuple(inspect.signature(book_algorithm).parameters)

    def episode_algorithm(
        ctx: RLAlgorithmContext,
        env: EnvBase,
        start_step: int,
        episode: int,
        max_steps: int,
    ) -> RLEpisodeResult:
        _ensure_linear_action_value_weights(ctx)
        tracker = _EpisodeTracker(
            ctx=ctx,
            env=env,
            start_step=start_step,
            episode=episode,
            max_steps=max_steps,
            tracked_names=("w",),
        )
        s0 = env.reset()

        def pi(
            state: State,
            w: np.ndarray,
            epsilon: float | None = None,
        ) -> Action:
            effective_epsilon = (
                ctx.config.epsilon if epsilon is None else float(epsilon)
            )
            q_values = _q_values_from_w(ctx, state, w)
            probs = make_policy_probs(
                q_values,
                ctx.config.behavior_policy,
                effective_epsilon,
                ctx.config.softmax_temperature,
            )
            return int(ctx.rng.choice(ctx.num_actions, p=probs))

        def q_hat(state: State, action: Action, w: np.ndarray) -> float:
            return float(np.dot(ctx.tables.features[state, action], w))

        def grad_q_hat(state: State, action: Action, w: np.ndarray) -> np.ndarray:
            _ = w
            return ctx.tables.features[state, action].copy()

        def step(state: State, action: Action) -> tuple[State, float, bool]:
            next_state, reward, done, extra = env.transition(int(state), int(action))
            transition = StepResult(
                state=int(state),
                action=int(action),
                reward=float(reward),
                next_state=int(next_state),
                done=bool(done),
                extra=dict(extra),
            )
            tracked_done = tracker.begin_transition(int(action), transition)
            return int(next_state), float(reward), bool(tracked_done)

        call_kwargs = _build_sarsa_fa_episode_kwargs(
            ctx=ctx,
            parameter_names=parameter_names,
            max_steps=max_steps,
        )
        instrumented_function = _load_instrumented_episode_function(
            book_algorithm,
            tracker,
        )
        instrumented_function(
            ctx.tables.w.copy(),
            s0,
            pi,
            q_hat,
            grad_q_hat,
            step,
            **call_kwargs,
        )
        return tracker.finish()

    return AdaptedAlgorithm(
        mode="episode",
        display_function=book_algorithm,
        episode_runner=_inherit_book_metadata(episode_algorithm, book_algorithm),
    )


def adapt_episode_env_algorithm(book_algorithm: BookAlgorithm) -> AdaptedAlgorithm:
    """适配 ``env`` 驱动的整段 episode 算法。"""

    signature = inspect.signature(book_algorithm)
    signature_parameters = signature.parameters
    tracked_names = tuple(
        name
        for name, parameter in signature_parameters.items()
        if _should_track_episode_state(name, parameter)
    )

    def episode_algorithm(
        ctx: RLAlgorithmContext,
        env: EnvBase,
        start_step: int,
        episode: int,
        max_steps: int,
    ) -> RLEpisodeResult:
        tracker = _EpisodeTracker(
            ctx=ctx,
            env=env,
            start_step=start_step,
            episode=episode,
            max_steps=max_steps,
            tracked_names=tracked_names,
        )
        call_kwargs = _build_episode_call_kwargs(
            ctx=ctx,
            env_proxy=tracker.env_proxy,
            signature_parameters=signature_parameters,
            max_steps=max_steps,
        )
        instrumented_function = _load_instrumented_episode_function(book_algorithm, tracker)
        if ctx.config.seed is not None:
            np.random.seed(int(ctx.config.seed) + int(episode))
        instrumented_function(**call_kwargs)
        return tracker.finish()

    return AdaptedAlgorithm(
        mode="episode",
        display_function=book_algorithm,
        episode_runner=_inherit_book_metadata(episode_algorithm, book_algorithm),
    )


class _EpisodeTracker:
    """收集 episode 风格算法的每步更新。"""

    def __init__(
        self,
        ctx: RLAlgorithmContext,
        env: EnvBase,
        start_step: int,
        episode: int,
        max_steps: int,
        tracked_names: tuple[str, ...],
    ) -> None:
        self.ctx = ctx
        self.env = env
        self.start_step = start_step
        self.episode = episode
        self.max_steps = max_steps
        self.tracked_names = tracked_names
        self.env_proxy = _EpisodeEnvProxy(self)

        self.current_snapshot = {
            name: ctx.get_runtime_state(name) for name in self.tracked_names
        }
        self.pending_before_snapshot = {
            name: _copy_value(value) for name, value in self.current_snapshot.items()
        }
        self.pending_transition: RLTransition | None = None
        self.pending_env_extra: dict[str, Any] = {}
        self.latest_locals: dict[str, Any] = {}
        self.updates: list[RLStepEvent] = []
        self.episode_reward = 0.0
        self.episode_steps = 0

    def capture(self, local_vars: dict[str, Any], line_number: int) -> None:
        for name in self.tracked_names:
            if name in local_vars:
                self.current_snapshot[name] = _copy_value(local_vars[name])

        traced_locals: dict[str, Any] = {"line": line_number}
        for name in _TRACE_LOCALS:
            if name in local_vars:
                traced_locals[name] = _copy_value(local_vars[name])
        self.latest_locals = traced_locals
        if self.pending_transition is None:
            self.finalize_model_update()

    def begin_transition(self, action: Action, transition) -> bool:
        self.finalize_pending()
        self.pending_before_snapshot = {
            name: _copy_value(value) for name, value in self.current_snapshot.items()
        }

        self.episode_reward += float(transition.reward)
        self.episode_steps += 1
        done = bool(transition.done or self.episode_steps >= self.max_steps)
        self.pending_transition = RLTransition(
            state=transition.state,
            action=int(action),
            reward=float(transition.reward),
            next_state=transition.next_state,
            done=done,
            step=self.start_step + len(self.updates),
            episode=self.episode,
        )
        self.pending_env_extra = dict(transition.extra)
        return done

    def finalize_pending(self) -> None:
        if self.pending_transition is None:
            return

        updated_names = {
            name
            for name in self.tracked_names
            if _values_changed(
                self.pending_before_snapshot.get(name),
                self.current_snapshot.get(name),
            )
        }
        for name in updated_names:
            self.ctx.set_runtime_state(name, self.current_snapshot[name])
        self.ctx.rebuild_visual_state(updated_names)

        transition = self.pending_transition
        result = RLStepResult(
            formula=_build_formula(updated_names, self.latest_locals),
            calculation=_build_calculation(
                transition=transition,
                locals_snapshot=self.latest_locals,
                ctx=self.ctx,
                updated_names=updated_names,
            ),
            variables=_build_variables(
                transition=transition,
                locals_snapshot=self.latest_locals,
                ctx=self.ctx,
                updated_names=updated_names,
            ),
            metrics=_build_metrics(
                transition=transition,
                locals_snapshot=self.latest_locals,
                ctx=self.ctx,
                updated_names=updated_names,
            ),
            updated_state=transition.state,
            next_action=_resolve_next_action(
                locals_snapshot=self.latest_locals,
                done=transition.done,
            ),
        )
        self.updates.append(
            RLStepEvent(
                transition=transition,
                result=result,
                env_extra=self.pending_env_extra,
            )
        )
        self.pending_transition = None
        self.pending_env_extra = {}

    def finalize_model_update(self) -> None:
        updated_names = {
            name
            for name in self.tracked_names
            if _values_changed(
                self.pending_before_snapshot.get(name),
                self.current_snapshot.get(name),
            )
        }
        if not updated_names:
            return

        for name in updated_names:
            self.ctx.set_runtime_state(name, self.current_snapshot[name])
        self.ctx.rebuild_visual_state(updated_names)

        state = _local_int(self.latest_locals, ("s", "state", "state_idx"), 0)
        action = _local_int(self.latest_locals, ("a", "action"), 0)
        next_state = _local_int(
            self.latest_locals,
            ("s_next", "next_state", "state_next"),
            state,
        )
        reward = _local_float(self.latest_locals, ("r", "reward"), 0.0)
        transition = RLTransition(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=False,
            step=self.start_step + len(self.updates),
            episode=self.episode,
        )
        result = RLStepResult(
            formula=_build_formula(updated_names, self.latest_locals),
            calculation=_build_calculation(
                transition=transition,
                locals_snapshot=self.latest_locals,
                ctx=self.ctx,
                updated_names=updated_names,
            ),
            variables=_build_variables(
                transition=transition,
                locals_snapshot=self.latest_locals,
                ctx=self.ctx,
                updated_names=updated_names,
            ),
            metrics=_build_metrics(
                transition=transition,
                locals_snapshot=self.latest_locals,
                ctx=self.ctx,
                updated_names=updated_names,
            ),
            updated_state=transition.state,
            next_action=_resolve_next_action(
                locals_snapshot=self.latest_locals,
                done=False,
            ),
        )
        self.updates.append(
            RLStepEvent(
                transition=transition,
                result=result,
                env_extra={},
            )
        )
        self.episode_reward += reward
        self.episode_steps += 1
        self.pending_before_snapshot = {
            name: _copy_value(value) for name, value in self.current_snapshot.items()
        }

    def finish(self) -> RLEpisodeResult:
        self.finalize_pending()
        return RLEpisodeResult(
            updates=self.updates,
            episode_reward=self.episode_reward,
            episode_steps=self.episode_steps,
        )


class _EpisodeEnvProxy:
    """把框架环境包装成书中算法常见的 ``reset/step`` 接口。"""

    def __init__(self, tracker: _EpisodeTracker) -> None:
        self._tracker = tracker

    @property
    def num_states(self) -> int:
        return self._tracker.env.num_states

    @property
    def num_actions(self) -> int:
        return self._tracker.env.num_actions

    @property
    def current_state(self) -> State:
        return self._tracker.env.current_state

    @property
    def states(self) -> tuple[State, ...]:
        return tuple(range(self.num_states))

    @property
    def actions(self) -> tuple[Action, ...]:
        return tuple(range(self.num_actions))

    @property
    def model(self) -> "_ModelView":
        return _ModelView(self._tracker.env)

    def reset(self, seed: int | None = None) -> State:
        self._tracker.finalize_pending()
        return self._tracker.env.reset(seed=seed)

    def step(self, action: Action) -> tuple[State, float, bool]:
        transition = self._tracker.env.step(int(action))
        done = self._tracker.begin_transition(int(action), transition)
        return transition.next_state, float(transition.reward), done

    def transition(
        self,
        state: State,
        action: Action,
    ) -> tuple[State, float, bool]:
        next_state, reward, done, _ = self._tracker.env.transition(int(state), int(action))
        return int(next_state), float(reward), bool(done)

    def reward(
        self,
        state: State,
        action: Action,
        next_state: State | None = None,
    ) -> float:
        model_next_state, reward, _, _ = self._tracker.env.transition(
            int(state),
            int(action),
        )
        _ = next_state, model_next_state
        return float(reward)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._tracker.env, name)


class _ModelView:
    def __init__(self, env: EnvBase) -> None:
        self._env = env

    @property
    def states(self) -> tuple[State, ...]:
        return tuple(range(self._env.num_states))

    @property
    def actions(self) -> tuple[Action, ...]:
        return tuple(range(self._env.num_actions))

    def transition(
        self,
        state: State,
        action: Action,
    ) -> tuple[State, float, bool]:
        next_state, reward, done, _ = self._env.transition(int(state), int(action))
        return int(next_state), float(reward), bool(done)

    def reward(
        self,
        state: State,
        action: Action,
        next_state: State | None = None,
    ) -> float:
        model_next_state, reward, _, _ = self._env.transition(int(state), int(action))
        _ = next_state, model_next_state
        return float(reward)


class _CaptureAssignmentsTransformer(ast.NodeTransformer):
    """在赋值语句后面插入一次 locals 捕获。"""

    def visit_Assign(self, node: ast.Assign) -> list[ast.stmt]:
        node = self.generic_visit(node)
        return [node, self._capture_statement(node.lineno)]

    def visit_AnnAssign(self, node: ast.AnnAssign) -> list[ast.stmt]:
        node = self.generic_visit(node)
        return [node, self._capture_statement(node.lineno)]

    def visit_AugAssign(self, node: ast.AugAssign) -> list[ast.stmt]:
        node = self.generic_visit(node)
        return [node, self._capture_statement(node.lineno)]

    def visit_Return(self, node: ast.Return) -> list[ast.stmt]:
        node = self.generic_visit(node)
        return [self._capture_statement(node.lineno), node]

    def _capture_statement(self, line_number: int) -> ast.Expr:
        return ast.Expr(
            value=ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id="__tracker__", ctx=ast.Load()),
                    attr="capture",
                    ctx=ast.Load(),
                ),
                args=[
                    ast.Call(
                        func=ast.Name(id="locals", ctx=ast.Load()),
                        args=[],
                        keywords=[],
                    ),
                    ast.Constant(value=line_number),
                ],
                keywords=[],
            )
        )


def _matches_sarsa_fa_signature(book_algorithm: BookAlgorithm) -> bool:
    signature = inspect.signature(book_algorithm)
    parameters = signature.parameters
    names = tuple(parameters)
    if names[: len(_SARSA_FA_REQUIRED_PARAMS)] != _SARSA_FA_REQUIRED_PARAMS:
        return False

    for name, parameter in list(parameters.items())[len(_SARSA_FA_REQUIRED_PARAMS) :]:
        if (
            name not in _SARSA_FA_SUPPORTED_OPTIONAL_PARAMS
            and parameter.default is inspect.Signature.empty
        ):
            return False
    return True


def _matches_episode_env_signature(book_algorithm: BookAlgorithm) -> bool:
    signature = inspect.signature(book_algorithm)
    names = tuple(signature.parameters)
    if not names or names[0] != "env":
        return False
    return True


def _is_episode_state_parameter(name: str) -> bool:
    if name in {"env", "s0", "max_steps"}:
        return False
    if name in _COUNT_PARAM_NAMES:
        return False
    if name in _HELPER_PARAM_NAMES:
        return False
    if name in _CONFIG_PARAM_RESOLVERS:
        return False
    return name in _KNOWN_RUNTIME_STATE_NAMES or name.isidentifier()


def _should_track_episode_state(
    name: str,
    parameter: inspect.Parameter,
) -> bool:
    if not _is_episode_state_parameter(name):
        return False
    return (
        name in _KNOWN_RUNTIME_STATE_NAMES
        or parameter.default is inspect.Signature.empty
    )


def _build_supported_kwargs(
    ctx: RLAlgorithmContext,
    parameter_names: tuple[str, ...],
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if "alpha" in parameter_names:
        kwargs["alpha"] = ctx.config.alpha
    if "gamma" in parameter_names:
        kwargs["gamma"] = ctx.config.gamma
    if "epsilon" in parameter_names:
        kwargs["epsilon"] = ctx.config.epsilon
    if "episodes" in parameter_names:
        kwargs["episodes"] = 1
    if "max_steps" in parameter_names:
        kwargs["max_steps"] = 1
    return kwargs


def _build_sarsa_fa_episode_kwargs(
    ctx: RLAlgorithmContext,
    parameter_names: tuple[str, ...],
    max_steps: int,
) -> dict[str, Any]:
    kwargs = _build_supported_kwargs(ctx, parameter_names)
    if "episodes" in parameter_names:
        kwargs["episodes"] = 1
    if "max_steps" in parameter_names:
        kwargs["max_steps"] = max_steps
    return kwargs


def _build_episode_call_kwargs(
    ctx: RLAlgorithmContext,
    env_proxy: _EpisodeEnvProxy,
    signature_parameters: dict[str, inspect.Parameter],
    max_steps: int,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    for name, parameter in signature_parameters.items():
        if name == "env":
            kwargs[name] = env_proxy
            continue
        if name in _EPISODE_COUNT_NAMES:
            kwargs[name] = 1
            continue
        if name in _ITERATION_COUNT_NAMES:
            kwargs[name] = ctx.config.iterations
            continue
        if name == "max_steps":
            kwargs[name] = max_steps
            continue
        if name in _HELPER_PARAM_NAMES:
            kwargs[name] = _build_episode_helper(ctx, env_proxy, name)
            continue
        if name in _CONFIG_PARAM_RESOLVERS:
            kwargs[name] = _CONFIG_PARAM_RESOLVERS[name](ctx)
            continue
        if _should_track_episode_state(name, parameter):
            kwargs[name] = ctx.get_runtime_state(name)
            continue
        if parameter.default is not inspect.Signature.empty:
            continue
        raise ValueError(
            f"无法为参数 `{name}` 自动注入运行时对象。"
        )
    return kwargs


def _build_episode_helper(
    ctx: RLAlgorithmContext,
    env_proxy: _EpisodeEnvProxy,
    name: str,
) -> Any:
    if name == "states":
        return env_proxy.states
    if name == "actions":
        return env_proxy.actions
    if name == "num_states":
        return ctx.num_states
    if name == "num_actions":
        return ctx.num_actions
    if name in {"model", "mdp", "env_model"}:
        return env_proxy.model
    if name == "step":
        return env_proxy.step
    if name == "transition":
        return env_proxy.transition
    if name == "reward":
        return env_proxy.reward
    if name == "softmax":
        return _softmax
    if name == "policy_probs":
        return lambda state, params=None: _policy_probs_for(ctx, state, params)
    if name in {"pi", "sample_from_policy"}:
        return lambda state, params=None: int(
            ctx.rng.choice(
                ctx.num_actions,
                p=_policy_probs_for(ctx, state, params),
            )
        )
    if name in {"q", "q_hat"}:
        return lambda state, action=None, values=None: _q_value(
            ctx,
            state,
            action,
            values,
        )
    if name in {"grad_q", "grad_q_hat"}:
        return lambda state, action, values=None: _grad_q_value(
            ctx,
            state,
            action,
            values,
        )
    if name == "v_hat":
        return lambda state, values=None: _v_value(ctx, state, values)
    if name == "grad_v_hat":
        return lambda state, values=None: _grad_v_value(ctx, state, values)
    if name == "grad_log_policy":
        return lambda state, action, params=None: _grad_log_policy(
            ctx,
            state,
            action,
            params,
        )
    raise ValueError(f"Unsupported helper parameter: {name}")


def _policy_probs_for(
    ctx: RLAlgorithmContext,
    state: State,
    params: Any = None,
) -> np.ndarray:
    if params is None:
        return ctx.tables.policy[int(state)].astype(float, copy=True)

    values = np.asarray(params, dtype=float)
    if values.ndim == 2:
        row = values[int(state)]
    elif values.ndim == 1 and values.shape[0] == ctx.num_actions:
        row = values
    else:
        row = ctx.tables.policy[int(state)]

    row = np.asarray(row, dtype=float)
    if np.all(row >= 0.0) and np.isclose(float(row.sum()), 1.0):
        return row / float(row.sum())
    return _softmax(row, temperature=ctx.temperature())


def _q_value(
    ctx: RLAlgorithmContext,
    state: State,
    action: Action | None = None,
    values: Any = None,
) -> float | np.ndarray:
    table = ctx.tables.q if values is None else np.asarray(values, dtype=float)
    state = int(state)

    if table.ndim == 2:
        row = table[state]
        if action is None:
            return row.astype(float, copy=True)
        return float(row[int(action)])

    if action is None:
        if table.shape[0] == ctx.num_states:
            return float(table[state])
        return np.array(
            [
                np.dot(ctx.tables.features[state, action_idx], table)
                for action_idx in range(ctx.num_actions)
            ],
            dtype=float,
        )
    return float(np.dot(ctx.tables.features[state, int(action)], table))


def _grad_q_value(
    ctx: RLAlgorithmContext,
    state: State,
    action: Action,
    values: Any = None,
) -> np.ndarray:
    reference = ctx.tables.w if values is None else np.asarray(values, dtype=float)
    if reference.ndim == 2:
        grad = np.zeros_like(reference, dtype=float)
        grad[int(state), int(action)] = 1.0
        return grad
    if reference.shape[0] == ctx.tables.features.shape[-1]:
        return ctx.tables.features[int(state), int(action)].copy()
    grad = np.zeros_like(reference, dtype=float)
    grad[int(state)] = 1.0
    return grad


def _v_value(
    ctx: RLAlgorithmContext,
    state: State,
    values: Any = None,
) -> float:
    if values is None:
        return float(ctx.tables.v[int(state)])
    table = np.asarray(values, dtype=float)
    if table.ndim == 2:
        return float(np.dot(ctx.tables.policy[int(state)], table[int(state)]))
    return float(table[int(state)])


def _grad_v_value(
    ctx: RLAlgorithmContext,
    state: State,
    values: Any = None,
) -> np.ndarray:
    reference = ctx.tables.v if values is None else np.asarray(values, dtype=float)
    grad = np.zeros_like(reference, dtype=float)
    grad[int(state)] = 1.0
    return grad


def _grad_log_policy(
    ctx: RLAlgorithmContext,
    state: State,
    action: Action,
    params: Any = None,
) -> np.ndarray:
    reference = ctx.tables.theta if params is None else np.asarray(params, dtype=float)
    grad = np.zeros_like(reference, dtype=float)
    probs = _policy_probs_for(ctx, state, reference)
    grad[int(state), :] = -probs
    grad[int(state), int(action)] += 1.0
    return grad


def _softmax(
    values: Any,
    temperature: float = 1.0,
) -> np.ndarray:
    logits = np.asarray(values, dtype=float)
    scaled = logits / max(float(temperature), 1e-8)
    scaled = scaled - np.max(scaled)
    weights = np.exp(scaled)
    total = float(weights.sum())
    if total <= 0.0:
        return np.full_like(weights, 1.0 / weights.size, dtype=float)
    return weights / total


def _load_instrumented_episode_function(book_algorithm: BookAlgorithm, tracker: _EpisodeTracker):
    cache_key = id(book_algorithm)
    cached = _INSTRUMENTED_EPISODE_CACHE.get(cache_key)
    if cached is None:
        source = textwrap.dedent(inspect.getsource(book_algorithm))
        tree = ast.parse(source)
        function_node = _first_function_node(tree)
        function_node.decorator_list = []
        transformer = _CaptureAssignmentsTransformer()
        transformed = transformer.visit(tree)
        ast.fix_missing_locations(transformed)
        compiled = compile(
            transformed,
            filename=inspect.getsourcefile(book_algorithm) or "<algorithm>",
            mode="exec",
        )
        cached = (compiled, function_node.name)
        _INSTRUMENTED_EPISODE_CACHE[cache_key] = cached

    compiled, function_name = cached
    namespace = dict(book_algorithm.__globals__)
    namespace["__builtins__"] = __builtins__
    namespace["__tracker__"] = tracker
    exec(compiled, namespace)
    return namespace[function_name]


def _build_formula(updated_names: set[str], locals_snapshot: dict[str, Any]) -> str:
    if "td_error" in locals_snapshot:
        driver = "td_error"
    elif "delta" in locals_snapshot:
        driver = "delta"
    elif "advantage" in locals_snapshot:
        driver = "advantage"
    elif "g" in locals_snapshot or "return_t" in locals_snapshot:
        driver = "return"
    else:
        driver = None

    if not updated_names:
        return "state advanced"
    tracked = ", ".join(sorted(updated_names))
    if driver is None:
        return f"{tracked} updated"
    return f"{tracked} updated using {driver}"


def _build_calculation(
    transition: RLTransition,
    locals_snapshot: dict[str, Any],
    ctx: RLAlgorithmContext,
    updated_names: set[str],
) -> str:
    parts = [f"r = {transition.reward:.3f}"]
    for name in ("q_sa", "q_next", "delta", "td_error", "advantage", "g", "return_t"):
        if name in locals_snapshot and _is_number(locals_snapshot[name]):
            parts.append(f"{name} = {float(locals_snapshot[name]):.3f}")
    for name in sorted(updated_names):
        value = ctx.get_runtime_state(name)
        if isinstance(value, np.ndarray):
            parts.append(f"||{name}|| = {float(np.linalg.norm(value)):.3f}")
        elif _is_number(value):
            parts.append(f"{name} = {float(value):.3f}")
    return "; ".join(parts)


def _build_variables(
    transition: RLTransition,
    locals_snapshot: dict[str, Any],
    ctx: RLAlgorithmContext,
    updated_names: set[str],
) -> dict[str, Any]:
    variables: dict[str, Any] = {
        "s": transition.state,
        "a": transition.action,
        "r": round(float(transition.reward), 3),
        "s_next": transition.next_state,
        "done": transition.done,
        "pi(a|s)": np.round(ctx.tables.policy[transition.state], 3).tolist(),
        "V_pi(s)": round(float(ctx.tables.v[transition.state]), 3),
    }
    for name in ("a_next", "delta", "td_error", "advantage", "q_sa", "q_next", "g", "return_t"):
        if name in locals_snapshot:
            value = locals_snapshot[name]
            variables[name] = round(float(value), 3) if _is_number(value) else value
    for name in sorted(updated_names):
        value = ctx.get_runtime_state(name)
        if isinstance(value, np.ndarray):
            variables[f"{name}_norm"] = round(float(np.linalg.norm(value)), 3)
        elif _is_number(value):
            variables[name] = round(float(value), 3)
    return variables


def _build_metrics(
    transition: RLTransition,
    locals_snapshot: dict[str, Any],
    ctx: RLAlgorithmContext,
    updated_names: set[str],
) -> dict[str, float]:
    metrics = {
        "rollout/reward": float(transition.reward),
        "train/state_value": float(ctx.tables.v[transition.state]),
    }
    if ctx.tables.q.shape == (ctx.num_states, ctx.num_actions):
        metrics["train/q_value"] = float(ctx.tables.q[transition.state, transition.action])
    for name in ("loss", "delta", "td_error", "advantage"):
        if name in locals_snapshot and _is_number(locals_snapshot[name]):
            metric_name = "train/loss" if name == "loss" else "train/td_error"
            metrics[metric_name] = float(locals_snapshot[name])
            break
    for name in sorted(updated_names):
        value = ctx.get_runtime_state(name)
        if isinstance(value, np.ndarray):
            metrics[f"train/{name}_norm"] = float(np.linalg.norm(value))
        elif _is_number(value):
            metrics[f"train/{name}"] = float(value)
    return metrics


def _resolve_next_action(
    locals_snapshot: dict[str, Any],
    done: bool,
) -> Action | None:
    if done:
        return None
    for name in ("a_next", "next_action", "a"):
        if name in locals_snapshot:
            try:
                return int(locals_snapshot[name])
            except (TypeError, ValueError):
                continue
    return None


def _q_values_from_w(
    ctx: RLAlgorithmContext,
    state: State,
    w: np.ndarray,
) -> np.ndarray:
    return np.array(
        [
            np.dot(ctx.tables.features[state, action], w)
            for action in range(ctx.num_actions)
        ],
        dtype=float,
    )


def _ensure_linear_action_value_weights(ctx: RLAlgorithmContext) -> None:
    """Keep Algorithm 8.2 on a linear action-value parameter vector."""

    expected_dim = int(ctx.tables.features.shape[-1])
    w = np.asarray(ctx.tables.w, dtype=float)
    if w.ndim == 1 and w.shape[0] == expected_dim:
        return

    q_values = np.asarray(ctx.tables.q, dtype=float)
    if q_values.shape != (ctx.num_states, ctx.num_actions):
        q_values = np.zeros((ctx.num_states, ctx.num_actions), dtype=float)

    linear_w = np.zeros(expected_dim, dtype=float)
    for state in range(ctx.num_states):
        for action in range(ctx.num_actions):
            feature = ctx.tables.features[state, action]
            active = np.flatnonzero(feature)
            if active.size == 1 and np.isclose(feature[active[0]], 1.0):
                linear_w[active[0]] = q_values[state, action]

    ctx.config.value_function = "linear"
    ctx.tables.w = linear_w
    ctx.sync_q_from_w()
    ctx.refresh_all_policies()


def _inherit_book_metadata(wrapper, book_algorithm):
    wrapper.__name__ = book_algorithm.__name__
    wrapper.__doc__ = book_algorithm.__doc__
    wrapper.__wrapped__ = book_algorithm
    return wrapper


def _copy_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.copy()
    if _is_number(value) or value is None:
        return value
    return copy.deepcopy(value)


def _values_changed(before: Any, after: Any) -> bool:
    if isinstance(before, np.ndarray) or isinstance(after, np.ndarray):
        return not np.array_equal(np.asarray(before), np.asarray(after))
    return before != after


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float, np.floating, np.integer))


def _local_int(
    values: dict[str, Any],
    names: tuple[str, ...],
    default: int,
) -> int:
    for name in names:
        if name in values:
            try:
                return int(values[name])
            except (TypeError, ValueError):
                continue
    return default


def _local_float(
    values: dict[str, Any],
    names: tuple[str, ...],
    default: float,
) -> float:
    for name in names:
        if name in values and _is_number(values[name]):
            return float(values[name])
    return default


def _first_function_node(tree: ast.Module) -> ast.FunctionDef | ast.AsyncFunctionDef:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            return node
    raise ValueError("book_algorithm 必须是普通函数。")
