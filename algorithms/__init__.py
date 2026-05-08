"""核心强化学习算法函数包。

这个包只放书中伪代码级别的核心算法函数。不要在这里写 UI、环境、
QThread、TensorBoard 或训练会话管理代码。
"""

from __future__ import annotations

from importlib import import_module
from pkgutil import iter_modules
from pathlib import Path

from .QAC import QAC
from .a2c import a2c
from .deep_q_learning import deep_q_learning
from .deterministic_actor_critic import deterministic_actor_critic
from .mc_basic import mc_basic
from .mc_epsilon_greedy import mc_epsilon_greedy
from .mc_exploring_starts import mc_exploring_starts
from .off_policy_actor_critic import off_policy_actor_critic
from .policy_iteration import policy_iteration
from .q_learning_fa import q_learning_fa
from .q_learning_off_policy import q_learning_off_policy
from .q_learning_on_policy import q_learning_on_policy
from .reinforce import reinforce
from .sarsa_value_function import sarsa_fa
from .sarsa_control import sarsa_control
from .td_value_fa import td_value_fa
from .truncated_policy_iteration import truncated_policy_iteration
from .value_iteration import value_iteration

__all__: list[str] = [
    "QAC",
    "a2c",
    "deep_q_learning",
    "deterministic_actor_critic",
    "mc_basic",
    "mc_epsilon_greedy",
    "mc_exploring_starts",
    "off_policy_actor_critic",
    "policy_iteration",
    "q_learning_fa",
    "q_learning_off_policy",
    "q_learning_on_policy",
    "reinforce",
    "sarsa_control",
    "sarsa_fa",
    "td_value_fa",
    "truncated_policy_iteration",
    "value_iteration",
]
_PACKAGE_DIR = Path(__file__).resolve().parent


def _export_names(module: object, module_name: str) -> tuple[str, ...]:
    explicit_exports = tuple(getattr(module, "__all__", ()))
    if explicit_exports:
        return explicit_exports

    same_name_callable = getattr(module, module_name, None)
    if callable(same_name_callable):
        return (module_name,)

    return ()


for module_info in iter_modules([str(_PACKAGE_DIR)]):
    module_name = module_info.name
    if module_name == "__init__":
        continue

    module = import_module(f"{__name__}.{module_name}")
    for export_name in _export_names(module, module_name):
        if export_name in __all__:
            continue
        globals()[export_name] = getattr(module, export_name)
        __all__.append(export_name)
