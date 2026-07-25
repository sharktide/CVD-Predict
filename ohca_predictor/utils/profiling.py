"""
Performance profiling utilities for the OHCA Predictor project.

Provides a PerformanceProfiler class with context manager support,
GPU/CPU memory tracking, per-operation timing, a summary() method,
and optional tf.profiler integration.
"""

import gc
import time
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import tensorflow as tf

    _HAS_TF = True
except ImportError:
    _HAS_TF = False


@dataclass
class _ProfileEntry:
    name: str
    start_time: float
    end_time: Optional[float] = None
    cpu_mem_start_bytes: Optional[int] = None
    cpu_mem_end_bytes: Optional[int] = None
    gpu_mem_start_bytes: Optional[int] = None
    gpu_mem_end_bytes: Optional[int] = None

    @property
    def duration_ms(self) -> Optional[float]:
        if self.end_time is not None:
            return (self.end_time - self.start_time) * 1000.0
        return None

    @property
    def cpu_delta_bytes(self) -> Optional[int]:
        if self.cpu_mem_start_bytes is not None and self.cpu_mem_end_bytes is not None:
            return self.cpu_mem_end_bytes - self.cpu_mem_start_bytes
        return None

    @property
    def gpu_delta_bytes(self) -> Optional[int]:
        if self.gpu_mem_start_bytes is not None and self.gpu_mem_end_bytes is not None:
            return self.gpu_mem_end_bytes - self.gpu_mem_start_bytes
        return None


def _get_cpu_memory_bytes() -> Optional[int]:
    """Get current process RSS in bytes (macOS/Linux fallback)."""
    try:
        import resource
        usage = resource.getrusage(resource.RUSAGE_SELF)
        return usage.ru_maxrss * 1024
    except Exception:
        return None


def _get_gpu_memory_bytes() -> Optional[int]:
    """Get current GPU memory usage in bytes if CUDA is available."""
    if not _HAS_TF:
        return None
    try:
        gpus = tf.config.list_physical_devices("GPU")
        if not gpus:
            return None
        from tensorflow.python.client import device_lib
        local_devices = device_lib.list_local_devices()
        total = 0
        for d in local_devices:
            if d.device_type == "GPU":
                mem_limit = d.memory_limit
                if mem_limit is not None:
                    total += int(mem_limit)
        return total if total > 0 else None
    except Exception:
        return None


def _get_gpu_memory_used_bytes() -> Optional[int]:
    """Try to get actual GPU memory used via tf.config.experimental."""
    if not _HAS_TF:
        return None
    try:
        details = tf.config.experimental.get_memory_info("GPU:0")
        if details is not None and "current" in details:
            return details["current"]
    except Exception:
        pass
    return None


class PerformanceProfiler:
    """
    Context-manager-based performance profiler with memory tracking.

    Tracks wall-clock time and GPU/CPU memory for named operations.
    Optionally integrates with tf.profiler for detailed TF graph profiling.
    """

    def __init__(self, enable_tf_profiler: bool = False) -> None:
        self._entries: OrderedDict[str, _ProfileEntry] = OrderedDict()
        self._stack: List[str] = []
        self._enable_tf_profiler = enable_tf_profiler
        self._tf_profiler_running = False
        self._tf_profile_log_dir: Optional[str] = None

    def _current_cpu_mem(self) -> Optional[int]:
        return _get_cpu_memory_bytes()

    def _current_gpu_mem(self) -> Optional[int]:
        return _get_gpu_memory_used_bytes()

    def start(self, name: str) -> None:
        """Start profiling a named operation."""
        entry = _ProfileEntry(
            name=name,
            start_time=time.perf_counter(),
            cpu_mem_start_bytes=self._current_cpu_mem(),
            gpu_mem_start_bytes=self._current_gpu_mem(),
        )
        self._entries[name] = entry
        self._stack.append(name)

    def stop(self, name: Optional[str] = None) -> None:
        """Stop profiling the named operation (or the most recent)."""
        if name is None:
            if self._stack:
                name = self._stack.pop()
            else:
                return
        else:
            if name in self._stack:
                self._stack.remove(name)

        if name in self._entries:
            entry = self._entries[name]
            entry.end_time = time.perf_counter()
            entry.cpu_mem_end_bytes = self._current_cpu_mem()
            entry.gpu_mem_end_bytes = self._current_gpu_mem()

    @contextmanager
    def profile(self, name: str):
        """Context manager to profile a block of code."""
        self.start(name)
        try:
            yield
        finally:
            self.stop(name)

    def start_tf_profiler(self, log_dir: str = "logs/tf_profiler") -> None:
        """Start TensorFlow profiler (requires TF 2.3+)."""
        if not _HAS_TF:
            raise RuntimeError("TensorFlow is required for tf.profiler integration")
        if self._tf_profiler_running:
            return
        tf.profiler.experimental.start(log_dir)
        self._tf_profiler_running = True
        self._tf_profile_log_dir = log_dir

    def stop_tf_profiler(self) -> None:
        """Stop TensorFlow profiler."""
        if not self._tf_profiler_running:
            return
        try:
            tf.profiler.experimental.stop()
        except Exception:
            pass
        self._tf_profiler_running = False

    @contextmanager
    def tf_profile_scope(self, log_dir: str = "logs/tf_profiler"):
        """Context manager for TF profiler."""
        self.start_tf_profiler(log_dir)
        try:
            yield
        finally:
            self.stop_tf_profiler()

    def summary(self) -> Dict[str, Dict[str, Any]]:
        """
        Return a summary of all profiled operations.

        Returns:
            Dict mapping operation names to dicts with keys:
            'duration_ms', 'cpu_delta_mb', 'gpu_delta_mb'.
        """
        results: Dict[str, Dict[str, Any]] = OrderedDict()
        for name, entry in self._entries.items():
            info: Dict[str, Any] = {"name": name}
            dur = entry.duration_ms
            if dur is not None:
                info["duration_ms"] = round(dur, 3)
            cpu_delta = entry.cpu_delta_bytes
            if cpu_delta is not None:
                info["cpu_delta_mb"] = round(cpu_delta / (1024 * 1024), 3)
            gpu_delta = entry.gpu_delta_bytes
            if gpu_delta is not None:
                info["gpu_delta_mb"] = round(gpu_delta / (1024 * 1024), 3)
            results[name] = info
        return results

    def summary_table(self) -> str:
        """Return a formatted table string of the profile summary."""
        data = self.summary()
        if not data:
            return "No profiling data collected."

        lines: List[str] = []
        header = f"{'Operation':<40} {'Duration (ms)':>14} {'CPU Delta (MB)':>15} {'GPU Delta (MB)':>15}"
        lines.append(header)
        lines.append("-" * len(header))

        total_duration = 0.0
        for name, info in data.items():
            dur = info.get("duration_ms", "N/A")
            cpu = info.get("cpu_delta_mb", "N/A")
            gpu = info.get("gpu_delta_mb", "N/A")
            dur_str = f"{dur:>14.3f}" if isinstance(dur, (int, float)) else f"{dur:>14}"
            cpu_str = f"{cpu:>15.3f}" if isinstance(cpu, (int, float)) else f"{cpu:>15}"
            gpu_str = f"{gpu:>15.3f}" if isinstance(gpu, (int, float)) else f"{gpu:>15}"
            lines.append(f"{name:<40} {dur_str} {cpu_str} {gpu_str}")
            if isinstance(dur, (int, float)):
                total_duration += dur

        lines.append("-" * len(header))
        lines.append(f"{'TOTAL':<40} {total_duration:>14.3f}")
        return "\n".join(lines)

    def reset(self) -> None:
        """Clear all collected profiling data."""
        self._entries.clear()
        self._stack.clear()

    def __enter__(self) -> "PerformanceProfiler":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.stop_tf_profiler()
