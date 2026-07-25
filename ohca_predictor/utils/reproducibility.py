"""
Reproducibility management for deterministic execution.

Sets seeds for random, NumPy, TensorFlow, and PYTHONHASHSEED. Configures
TensorFlow deterministic settings (inter/intra-op thread counts, deterministic
ops, cuDNN determinism). Provides a context manager for reproducible
execution and a verify_determinism() method that runs the same input twice
and checks for bit-identical output.
"""

import hashlib
import os
import random
import sys
from contextlib import contextmanager
from typing import Any, Callable, Optional

import numpy as np

try:
    import tensorflow as tf

    _HAS_TF = True
except ImportError:
    _HAS_TF = False


class ReproducibilityManager:
    """
    Manages reproducibility across all random sources and TensorFlow.

    Sets seeds, configures deterministic TF settings, and provides a
    context manager for reproducible execution blocks.
    """

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self.numpy_seed = seed + 1000
        self.tensorflow_seed = seed + 2000
        self.python_hash_seed = seed + 3000

        self._original_env: Optional[str] = os.environ.get("PYTHONHASHSEED")
        self._tf_deterministic_original: Optional[bool] = None
        self._tf_threads_original: Optional[int] = None
        self._tf_intra_threads_original: Optional[int] = None
        self._tf_inter_threads_original: Optional[int] = None
        self._tf_cudnn_deterministic_original: Optional[bool] = None
        self._tf_cudnn_benchmark_original: Optional[bool] = None

    def set_all_seeds(self) -> None:
        """Set seeds for all random number generators."""
        os.environ["PYTHONHASHSEED"] = str(self.python_hash_seed)
        random.seed(self.seed)
        np.random.seed(self.numpy_seed)

        if _HAS_TF:
            tf.random.set_seed(self.tensorflow_seed)
            try:
                tf.config.experimental.enable_op_determinism()
            except (AttributeError, RuntimeError):
                pass

    def configure_tf_deterministic(
        self,
        num_threads: Optional[int] = 1,
        deterministic: bool = True,
        cudnn_deterministic: bool = True,
        cudnn_benchmark: bool = False,
    ) -> None:
        """
        Configure TensorFlow for deterministic execution.

        Args:
            num_threads: Number of threads for both intra and inter op
                         parallelism. Set to 1 for full determinism.
            deterministic: Enable tf.config.experimental.enable_op_determinism().
            cudnn_deterministic: Set TF_CUDNN_DETERMINISTIC env var.
            cudnn_benchmark: Set TF_CUDNN_BENCHMARK env var.
        """
        if not _HAS_TF:
            return

        try:
            self._tf_deterministic_original = getattr(
                tf.config, "experimental_enable_op_determinism", None
            )
        except AttributeError:
            self._tf_deterministic_original = None

        self._tf_cudnn_deterministic_original = os.environ.get(
            "TF_CUDNN_DETERMINISTIC"
        )
        self._tf_cudnn_benchmark_original = os.environ.get("TF_CUDNN_BENCHMARK")

        if deterministic:
            os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
        else:
            os.environ.pop("TF_CUDNN_DETERMINISTIC", None)

        if cudnn_benchmark:
            os.environ["TF_CUDNN_BENCHMARK"] = "1"
        else:
            os.environ.pop("TF_CUDNN_BENCHMARK", None)

        os.environ["TF_DETERMINISTIC_OPS"] = "1" if deterministic else "0"

        try:
            tf.config.threading.set_intra_op_parallelism_threads(num_threads)
            tf.config.threading.set_inter_op_parallelism_threads(num_threads)
        except RuntimeError:
            pass

        if deterministic:
            try:
                tf.config.experimental.enable_op_determinism()
            except (AttributeError, RuntimeError):
                pass

        if not cudnn_deterministic:
            try:
                tf.config.experimental.disable_op_determinism()
            except (AttributeError, RuntimeError):
                pass

    def verify_determinism(
        self,
        forward_fn: Callable[[np.ndarray], np.ndarray],
        test_input: np.ndarray,
        num_runs: int = 2,
    ) -> bool:
        """
        Verify that forward_fn produces bit-identical output for the same input.

        Runs forward_fn num_runs times on the same test_input and checks
        that all outputs are element-wise identical.

        Args:
            forward_fn: Function mapping np.ndarray -> np.ndarray.
            test_input: Input array to feed to forward_fn.
            num_runs: Number of runs to compare (must be >= 2).

        Returns:
            True if all outputs are bit-identical, False otherwise.
        """
        if num_runs < 2:
            raise ValueError(f"num_runs must be >= 2, got {num_runs}")

        results = []
        for _ in range(num_runs):
            output = forward_fn(test_input.copy())
            results.append(output.copy())

        for i in range(1, len(results)):
            if not np.array_equal(results[0], results[i]):
                max_diff = float(np.max(np.abs(results[0].astype(float) - results[i].astype(float))))
                return False
        return True

    def verify_determinism_tensorflow(
        self,
        model_fn: Callable[[], Any],
        test_input: Any,
        num_runs: int = 2,
    ) -> bool:
        """
        Verify TensorFlow model determinism with identical inputs.

        Builds the model from model_fn, runs num_runs forward passes on
        test_input, and checks for bit-identical output.

        Args:
            model_fn: Callable returning a compiled tf.keras.Model.
            test_input: Input tensor or numpy array for the model.
            num_runs: Number of forward passes to compare.

        Returns:
            True if all outputs are bit-identical, False otherwise.
        """
        if not _HAS_TF:
            raise RuntimeError("TensorFlow is required for this method")

        if num_runs < 2:
            raise ValueError(f"num_runs must be >= 2, got {num_runs}")

        results = []
        for _ in range(num_runs):
            model = model_fn()
            output = model(test_input, training=False)
            if isinstance(output, (list, tuple)):
                results.append([o.numpy().copy() for o in output])
            else:
                results.append(output.numpy().copy())

        for i in range(1, len(results)):
            if isinstance(results[0], list):
                for j in range(len(results[0])):
                    if not np.array_equal(results[0][j], results[i][j]):
                        return False
            else:
                if not np.array_equal(results[0], results[i]):
                    return False
        return True

    def get_determinism_hash(self) -> str:
        """Return a deterministic hash string representing the current seed state."""
        seed_str = f"{self.seed}:{self.numpy_seed}:{self.tensorflow_seed}:{self.python_hash_seed}"
        return hashlib.sha256(seed_str.encode()).hexdigest()[:16]

    @contextmanager
    def reproducible_scope(self):
        """
        Context manager that temporarily sets all seeds and TF deterministic
        settings, restoring originals on exit.

        Usage:
            rm = ReproducibilityManager(seed=42)
            with rm.reproducible_scope():
                # code here runs deterministically
                ...
        """
        old_hashseed = os.environ.get("PYTHONHASHSEED")
        self.set_all_seeds()
        self.configure_tf_deterministic()
        try:
            yield
        finally:
            if old_hashseed is not None:
                os.environ["PYTHONHASHSEED"] = old_hashseed
            else:
                os.environ.pop("PYTHONHASHSEED", None)

            if _HAS_TF:
                if self._tf_cudnn_deterministic_original is not None:
                    os.environ["TF_CUDNN_DETERMINISTIC"] = (
                        self._tf_cudnn_deterministic_original
                    )
                else:
                    os.environ.pop("TF_CUDNN_DETERMINISTIC", None)

                if self._tf_cudnn_benchmark_original is not None:
                    os.environ["TF_CUDNN_BENCHMARK"] = (
                        self._tf_cudnn_benchmark_original
                    )
                else:
                    os.environ.pop("TF_CUDNN_BENCHMARK", None)

    def reset(self) -> None:
        """Reset all managed environment variables and RNG states."""
        if self._original_env is not None:
            os.environ["PYTHONHASHSEED"] = self._original_env
        else:
            os.environ.pop("PYTHONHASHSEED", None)

        if _HAS_TF:
            if self._tf_cudnn_deterministic_original is not None:
                os.environ["TF_CUDNN_DETERMINISTIC"] = (
                    self._tf_cudnn_deterministic_original
                )
            else:
                os.environ.pop("TF_CUDNN_DETERMINISTIC", None)

            if self._tf_cudnn_benchmark_original is not None:
                os.environ["TF_CUDNN_BENCHMARK"] = (
                    self._tf_cudnn_benchmark_original
                )
            else:
                os.environ.pop("TF_CUDNN_BENCHMARK", None)
