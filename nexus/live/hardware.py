"""What this computer can run, and which local vision model to use on it.

The model has to fit in the graphics card's memory to be quick, or in main
memory to run at all (slowly) on the processor. Sizes are Ollama's downloads.
"""

from __future__ import annotations

import ctypes
import platform
import shutil
import subprocess
from dataclasses import dataclass

# Ollama tags, smallest first, with the memory they need to run well.
MODELS = (
    ("qwen3-vl:2b", "1.9 GB download; runs on almost any laptop, slowly without a graphics card"),
    ("qwen3-vl:4b", "3.3 GB download; needs a 4 GB graphics card or 16 GB of memory"),
    ("qwen3-vl:8b", "6.1 GB download; needs an 8 GB graphics card or an Apple Silicon Mac with 16 GB"),
)


@dataclass(frozen=True)
class Hardware:
    system: str
    ram_gb: float | None
    gpu_name: str | None
    vram_gb: float | None
    apple_silicon: bool

    def describe(self) -> str:
        parts = [self.system]
        parts.append(f"{self.ram_gb:.0f} GB memory" if self.ram_gb else "memory unknown")
        if self.gpu_name:
            vram = f", {self.vram_gb:.0f} GB" if self.vram_gb else ""
            parts.append(f"{self.gpu_name}{vram}")
        elif self.apple_silicon:
            parts.append("Apple Silicon (memory shared with the graphics)")
        else:
            parts.append("no NVIDIA graphics card found")
        return ", ".join(parts)


def recommend_model(hardware: Hardware) -> tuple[str, str]:
    """The Ollama vision model for this computer, and why."""
    vram = hardware.vram_gb or 0.0
    ram = hardware.ram_gb or 0.0
    if vram >= 8 or (hardware.apple_silicon and ram >= 16):
        return MODELS[2][0], MODELS[2][1]
    if vram >= 4 or ram >= 16:
        return MODELS[1][0], MODELS[1][1]
    return MODELS[0][0], MODELS[0][1]


def _ram_gb() -> float | None:
    try:
        import psutil  # type: ignore

        return psutil.virtual_memory().total / 1024**3
    except Exception:
        pass
    system = platform.system()
    try:
        if system == "Windows":

            class MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = MemoryStatus()
            status.dwLength = ctypes.sizeof(MemoryStatus)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):  # type: ignore[attr-defined]
                return status.ullTotalPhys / 1024**3
        elif system == "Darwin":
            out = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=5)
            return int(out.stdout.strip()) / 1024**3
        else:
            with open("/proc/meminfo", encoding="utf-8") as handle:
                for line in handle:
                    if line.startswith("MemTotal:"):
                        return int(line.split()[1]) / 1024**2
    except Exception:
        return None
    return None


def _nvidia() -> tuple[str | None, float | None]:
    if not shutil.which("nvidia-smi"):
        return None, None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=8,
        )
        first = out.stdout.strip().splitlines()[0]
        name, memory = (part.strip() for part in first.rsplit(",", 1))
        return name, float(memory) / 1024
    except Exception:
        return None, None


def detect() -> Hardware:
    """Look at this computer's memory and graphics card."""
    gpu_name, vram_gb = _nvidia()
    return Hardware(
        system=f"{platform.system()} {platform.release()}".strip(),
        ram_gb=_ram_gb(),
        gpu_name=gpu_name,
        vram_gb=vram_gb,
        apple_silicon=platform.system() == "Darwin" and platform.machine() == "arm64",
    )
