"""
Structured logging system for the OHCA Predictor project.

Provides a singleton StructuredLogger with console (human-readable) and
file (JSON structured) handlers, TensorBoard integration, and a
log_context decorator for automatic context tracking.
"""

import json
import logging
import os
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import wraps
from typing import Any, Callable, Dict, Optional

try:
    import tensorflow as tf

    _HAS_TENSORBOARD = True
except ImportError:
    _HAS_TENSORBOARD = False


class _JsonFormatter(logging.Formatter):
    """Formatter that outputs log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry: Dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "module": record.module,
            "message": record.getMessage(),
        }
        context = getattr(record, "context", None)
        if context is not None:
            log_entry["context"] = context
        return json.dumps(log_entry, default=str)


class _ConsoleFormatter(logging.Formatter):
    """Formatter for human-readable console output."""

    def __init__(self) -> None:
        super().__init__(
            fmt="%(asctime)s | %(levelname)-8s | %(module)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        context = getattr(record, "context", None)
        if context is not None:
            base += f" | {context}"
        return base


class StructuredLogger:
    """
    Singleton structured logger with console and file handlers.

    Console handler uses a human-readable format. File handler writes
    single-line JSON for machine parsing. Optional TensorBoard summary
    writer integration for scalar logging.
    """

    _instance: Optional["StructuredLogger"] = None
    _initialized: bool = False

    def __new__(cls, *args: Any, **kwargs: Any) -> "StructuredLogger":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        name: str = "ohca_predictor",
        log_dir: str = "logs",
        level: int = logging.DEBUG,
        tensorboard_log_dir: Optional[str] = None,
    ) -> None:
        if self._initialized:
            return
        self._initialized = True

        self._name = name
        self._log_dir = log_dir
        self._level = level

        os.makedirs(log_dir, exist_ok=True)

        self._logger = logging.getLogger(name)
        self._logger.setLevel(level)
        self._logger.handlers.clear()

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(_ConsoleFormatter())
        self._logger.addHandler(console_handler)

        file_path = os.path.join(log_dir, "ohca_predictor.jsonl")
        file_handler = logging.FileHandler(file_path, mode="a", encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(_JsonFormatter())
        self._logger.addHandler(file_handler)

        self._tb_writer = None
        if tensorboard_log_dir is not None and _HAS_TENSORBOARD:
            tb_path = os.path.join(tensorboard_log_dir, "structured_logs")
            self._tb_writer = tf.summary.create_file_writer(tb_path)

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton (useful for testing)."""
        cls._instance = None
        cls._initialized = False

    def _log(
        self,
        level: int,
        message: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        extra: Dict[str, Any] = {}
        if context is not None:
            extra["context"] = context
        record = self._logger.makeRecord(
            name=self._name,
            level=level,
            fn="",
            lno=0,
            msg=message,
            args=(),
            exc_info=None,
            extra=extra,
        )
        self._logger.handle(record)

        if self._tb_writer is not None and _HAS_TENSORBOARD:
            tag = logging.getLevelName(level).lower()
            with self._tb_writer.as_default():
                tf.summary.scalar(
                    f"logs/{tag}",
                    float(level >= logging.WARNING),
                    step=int(time.time() * 1000),
                )

    def debug(
        self, message: str, context: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log a DEBUG message."""
        self._log(logging.DEBUG, message, context)

    def info(
        self, message: str, context: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log an INFO message."""
        self._log(logging.INFO, message, context)

    def warning(
        self, message: str, context: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log a WARNING message."""
        self._log(logging.WARNING, message, context)

    def error(
        self, message: str, context: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log an ERROR message."""
        self._log(logging.ERROR, message, context)

    def critical(
        self, message: str, context: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log a CRITICAL message."""
        self._log(logging.CRITICAL, message, context)

    def log_scalar(
        self, tag: str, value: float, step: int
    ) -> None:
        """Write a scalar value to TensorBoard if a writer is active."""
        if self._tb_writer is not None and _HAS_TENSORBOARD:
            with self._tb_writer.as_default():
                tf.summary.scalar(tag, value, step=step)


def log_context(
    logger: Optional[StructuredLogger] = None,
    **context_kwargs: Any,
) -> Callable:
    """
    Decorator that automatically attaches context to log messages.

    Usage:
        @log_context(logger=my_logger, stage="training", fold=0)
        def train_epoch():
            logger.info("Starting epoch")  # context is auto-attached
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            active_logger = logger or StructuredLogger()
            merged = {**context_kwargs}
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                elapsed = time.perf_counter() - start
                merged["duration_seconds"] = round(elapsed, 4)
                merged["status"] = "success"
                active_logger.info(
                    f"Completed {func.__qualname__}",
                    context=merged,
                )
                return result
            except Exception as exc:
                elapsed = time.perf_counter() - start
                merged["duration_seconds"] = round(elapsed, 4)
                merged["status"] = "error"
                merged["error_type"] = type(exc).__name__
                merged["error_message"] = str(exc)
                active_logger.error(
                    f"Failed {func.__qualname__}",
                    context=merged,
                )
                raise

        return wrapper

    return decorator
