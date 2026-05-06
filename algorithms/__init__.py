"""核心强化学习算法函数包。

这个包只放书中伪代码级别的核心算法函数。不要在这里写 UI、环境、
QThread、TensorBoard 或训练会话管理代码。
"""

from __future__ import annotations

from importlib import import_module
from pkgutil import iter_modules
from pathlib import Path

from .QAC import QAC
from .sarsa_value_function import sarsa_fa

__all__: list[str] = ["QAC", "sarsa_fa"]
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
