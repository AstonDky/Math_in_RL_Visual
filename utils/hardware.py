"""硬件检测与训练后端选择建议。

当前项目是表格型 GridWorld/Q-learning，计算规模很小，CPU 反而最直接。
这里先检测硬件并给出建议；后续接入深度 RL 时，可以基于这些信息启用 CUDA。
"""

from __future__ import annotations

import platform
import subprocess
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class HardwareInfo:
    cpu: str
    gpus: tuple[str, ...]
    backend: str
    note: str


def detect_hardware() -> HardwareInfo:
    """检测 CPU/GPU，并给当前算法选择合适后端。"""

    cpu = _detect_cpu()
    gpus = _detect_gpus()
    has_nvidia = any("nvidia" in gpu.lower() for gpu in gpus)
    has_intel_gpu = any("intel" in gpu.lower() for gpu in gpus)

    if has_nvidia:
        backend = "cpu-table / cuda-ready"
        note = "当前表格 Q-learning 使用 CPU；深度 RL 可切换 NVIDIA CUDA。"
    elif has_intel_gpu:
        backend = "cpu-table / intel-gpu-detected"
        note = "当前表格 Q-learning 使用 CPU；核显适合显示加速，不适合本表格算法加速。"
    else:
        backend = "cpu-table"
        note = "当前表格 Q-learning 使用 CPU。"

    return HardwareInfo(cpu=cpu, gpus=tuple(gpus), backend=backend, note=note)


def _detect_cpu() -> str:
    cpu = platform.processor().strip()
    if cpu:
        return cpu

    output = _run_command(["wmic", "cpu", "get", "name"])
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if len(lines) >= 2:
        return lines[1]

    output = _run_command(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            "(Get-CimInstance Win32_Processor).Name",
        ]
    )
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if lines:
        return lines[0]
    return "unknown CPU"


def _detect_gpus() -> list[str]:
    nvidia = _run_command(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"])
    gpus = [line.strip() for line in nvidia.splitlines() if line.strip()]
    if gpus:
        return gpus

    output = _run_command(["wmic", "path", "win32_VideoController", "get", "name"])
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    gpus = [line for line in lines if line.lower() != "name"]
    if gpus:
        return gpus

    output = _run_command(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            "Get-CimInstance Win32_VideoController | "
            "Select-Object -ExpandProperty Name",
        ]
    )
    return [line.strip() for line in output.splitlines() if line.strip()]


def _run_command(command: list[str]) -> str:
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return completed.stdout
