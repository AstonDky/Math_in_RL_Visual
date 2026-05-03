"""核心算法函数源码解析器。

算法类只需要把真正的更新逻辑放进一个独立函数，然后调用
``build_algorithm_trace``。UI 就能自动显示该函数的代码行，而不需要
手写伪代码列表。
"""

from __future__ import annotations

import ast
import html
import inspect
import textwrap
from collections.abc import Callable
from typing import Any


def build_algorithm_trace(
    core_function: Callable[..., Any],
    title: str | None = None,
) -> dict[str, Any]:
    """解析核心算法函数，生成 UI 可消费的 trace 字典。"""

    display_function = getattr(core_function, "__wrapped__", core_function)
    lines = extract_core_lines(display_function)
    return {
        "title": title or display_function.__name__,
        "function_name": display_function.__name__,
        "lines": lines,
        "current_line": len(lines),
    }


def extract_core_lines(core_function: Callable[..., Any]) -> list[str]:
    """提取函数体中真正的代码行。

    - 自动去掉 ``def ...`` 这一行。
    - 自动去掉函数 docstring。
    - 保留函数体内的缩进层次，方便 UI 像代码框一样展示。
    """

    source = textwrap.dedent(inspect.getsource(core_function))
    source_lines = source.splitlines()
    tree = ast.parse(source)
    function_node = _first_function_node(tree)
    body_nodes = list(function_node.body)

    if body_nodes and _is_docstring_node(body_nodes[0]):
        body_nodes = body_nodes[1:]
    if body_nodes and isinstance(body_nodes[-1], ast.Return):
        body_nodes = body_nodes[:-1]

    if not body_nodes:
        return ["pass"]

    start_line = min(node.lineno for node in body_nodes)
    end_line = max(getattr(node, "end_lineno", node.lineno) for node in body_nodes)
    body_lines = source_lines[start_line - 1 : end_line]
    return _strip_common_indent(body_lines)


def escape_code_line(line: str) -> str:
    """把源码行转成安全 HTML 文本。"""

    return html.escape(line).replace(" ", "&nbsp;")


def _first_function_node(tree: ast.Module) -> ast.FunctionDef | ast.AsyncFunctionDef:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            return node
    raise ValueError("core_function 必须是普通函数或方法。")


def _is_docstring_node(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    )


def _strip_common_indent(lines: list[str]) -> list[str]:
    non_empty = [line for line in lines if line.strip()]
    if not non_empty:
        return [""]

    indent = min(len(line) - len(line.lstrip()) for line in non_empty)
    stripped = [line[indent:].rstrip() for line in lines]
    return [line for line in stripped if line.strip()]
