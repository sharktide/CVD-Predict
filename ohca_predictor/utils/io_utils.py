"""
Data I/O utilities for the OHCA Predictor project.

Provides DataIOManager for TFRecord serialization/deserialization of
WindowSample objects, dataset pipeline construction, checkpointing,
SavedModel export, and TFLite conversion.
"""

import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import tensorflow as tf

    _HAS_TF = True
except ImportError:
    _HAS_TF = False


@dataclass
class WindowSample:
    """
    Represents a single multi-modal window sample.

    Contains sensor signals, demographics, clinical metadata, and labels
    for OHCA prediction.
    """

    ecg: np.ndarray
    accelerometer: np.ndarray
    gyroscope: np.ndarray
    ppg: np.ndarray
    spo2: np.ndarray
    temperature: np.ndarray
    respiration: np.ndarray
    demographics: np.ndarray
    medications: np.ndarray
    comorbidities: np.ndarray
    lab_values: np.ndarray
    ohca_label: float
    time_to_event: float
    event_indicator: float
    heart_rate: float
    rhythm: int
    activity_state: int
    spo2_mean: float
    sbp_mean: float
    dbp_mean: float
    patient_id: str
    window_start_hours: float
    window_duration_hours: float
    signal_quality: Dict[str, float] = field(default_factory=dict)


class DataIOManager:
    """
    Manages data serialization, dataset pipelines, and model export.

    Handles TFRecord writing/reading for WindowSample objects, creates
    tf.data.Dataset pipelines, and provides checkpoint/model export methods.
    """

    def __init__(
        self,
        tfrecord_dir: str = "data/tfrecords",
        checkpoint_dir: str = "checkpoints",
        export_dir: str = "exports",
    ) -> None:
        self.tfrecord_dir = tfrecord_dir
        self.checkpoint_dir = checkpoint_dir
        self.export_dir = export_dir
        os.makedirs(tfrecord_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(export_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    #                        TFRecord Serialization                       #
    # ------------------------------------------------------------------ #

    def _bytes_feature(self, value: bytes) -> Any:
        if not _HAS_TF:
            return value
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _float_feature(self, value: float) -> Any:
        if not _HAS_TF:
            return value
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    def _float_list_feature(self, values: np.ndarray) -> Any:
        flat = values.astype(np.float32).flatten().tolist()
        if not _HAS_TF:
            return flat
        return tf.train.Feature(float_list=tf.train.FloatList(value=flat))

    def _int64_feature(self, value: int) -> Any:
        if not _HAS_TF:
            return value
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def _int64_list_feature(self, values: np.ndarray) -> Any:
        flat = values.astype(np.int64).flatten().tolist()
        if not _HAS_TF:
            return flat
        return tf.train.Feature(int64_list=tf.train.Int64List(value=flat))

    def serialize_window_sample(self, sample: WindowSample) -> bytes:
        """
        Serialize a WindowSample to a tf.train.Example proto bytes.

        Args:
            sample: WindowSample to serialize.

        Returns:
            Serialized bytes of the tf.train.Example.
        """
        feature_dict = {
            "ecg": self._bytes_feature(sample.ecg.astype(np.float32).tobytes()),
            "ecg_shape": self._int64_list_feature(np.array(sample.ecg.shape, dtype=np.int64)),
            "accelerometer": self._bytes_feature(sample.accelerometer.astype(np.float32).tobytes()),
            "accelerometer_shape": self._int64_list_feature(np.array(sample.accelerometer.shape, dtype=np.int64)),
            "gyroscope": self._bytes_feature(sample.gyroscope.astype(np.float32).tobytes()),
            "gyroscope_shape": self._int64_list_feature(np.array(sample.gyroscope.shape, dtype=np.int64)),
            "ppg": self._bytes_feature(sample.ppg.astype(np.float32).tobytes()),
            "ppg_shape": self._int64_list_feature(np.array(sample.ppg.shape, dtype=np.int64)),
            "spo2": self._bytes_feature(sample.spo2.astype(np.float32).tobytes()),
            "spo2_shape": self._int64_list_feature(np.array(sample.spo2.shape, dtype=np.int64)),
            "temperature": self._bytes_feature(sample.temperature.astype(np.float32).tobytes()),
            "temperature_shape": self._int64_list_feature(np.array(sample.temperature.shape, dtype=np.int64)),
            "respiration": self._bytes_feature(sample.respiration.astype(np.float32).tobytes()),
            "respiration_shape": self._int64_list_feature(np.array(sample.respiration.shape, dtype=np.int64)),
            "demographics": self._bytes_feature(sample.demographics.astype(np.float32).tobytes()),
            "demographics_shape": self._int64_list_feature(np.array(sample.demographics.shape, dtype=np.int64)),
            "medications": self._bytes_feature(sample.medications.astype(np.float32).tobytes()),
            "medications_shape": self._int64_list_feature(np.array(sample.medications.shape, dtype=np.int64)),
            "comorbidities": self._bytes_feature(sample.comorbidities.astype(np.float32).tobytes()),
            "comorbidities_shape": self._int64_list_feature(np.array(sample.comorbidities.shape, dtype=np.int64)),
            "lab_values": self._bytes_feature(sample.lab_values.astype(np.float32).tobytes()),
            "lab_values_shape": self._int64_list_feature(np.array(sample.lab_values.shape, dtype=np.int64)),
            "ohca_label": self._float_feature(sample.ohca_label),
            "time_to_event": self._float_feature(sample.time_to_event),
            "event_indicator": self._float_feature(sample.event_indicator),
            "heart_rate": self._float_feature(sample.heart_rate),
            "rhythm": self._int64_feature(sample.rhythm),
            "activity_state": self._int64_feature(sample.activity_state),
            "spo2_mean": self._float_feature(sample.spo2_mean),
            "sbp_mean": self._float_feature(sample.sbp_mean),
            "dbp_mean": self._float_feature(sample.dbp_mean),
            "patient_id": self._bytes_feature(sample.patient_id.encode("utf-8")),
            "window_start_hours": self._float_feature(sample.window_start_hours),
            "window_duration_hours": self._float_feature(sample.window_duration_hours),
        }

        for key, value in sample.signal_quality.items():
            feature_dict[f"sq_{key}"] = self._float_feature(value)

        sq_keys = sorted(sample.signal_quality.keys())
        feature_dict["signal_quality_keys"] = self._bytes_feature(
            ",".join(sq_keys).encode("utf-8")
        )
        feature_dict["signal_quality_values"] = self._float_list_feature(
            np.array([sample.signal_quality[k] for k in sq_keys], dtype=np.float32)
        )

        example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
        return example.SerializeToString()

    def deserialize_window_sample(self, raw_bytes: bytes) -> WindowSample:
        """
        Deserialize a serialized tf.train.Example back to a WindowSample.

        Args:
            raw_bytes: Serialized bytes of a tf.train.Example.

        Returns:
            Reconstructed WindowSample.
        """
        example = tf.train.Example()
        example.ParseFromString(raw_bytes)
        features = example.features.feature

        def get_float(name: str) -> float:
            return float(features[name].float_list.value[0])

        def get_int(name: str) -> int:
            return int(features[name].int64_list.value[0])

        def get_float_list(name: str) -> List[float]:
            return list(features[name].float_list.value)

        def get_int_list(name: str) -> List[int]:
            return list(features[name].int64_list.value)

        def get_bytes(name: str) -> bytes:
            return features[name].bytes_list.value[0]

        def reconstruct_array(key: str, shape_key: str, dtype: np.dtype) -> np.ndarray:
            flat = np.array(get_float_list(key), dtype=dtype)
            shape = tuple(get_int_list(shape_key))
            return flat.reshape(shape)

        ecg = reconstruct_array("ecg", "ecg_shape", np.float32)
        accelerometer = reconstruct_array("accelerometer", "accelerometer_shape", np.float32)
        gyroscope = reconstruct_array("gyroscope", "gyroscope_shape", np.float32)
        ppg = reconstruct_array("ppg", "ppg_shape", np.float32)
        spo2 = reconstruct_array("spo2", "spo2_shape", np.float32)
        temperature = reconstruct_array("temperature", "temperature_shape", np.float32)
        respiration = reconstruct_array("respiration", "respiration_shape", np.float32)
        demographics = reconstruct_array("demographics", "demographics_shape", np.float32)
        medications = reconstruct_array("medications", "medications_shape", np.float32)
        comorbidities = reconstruct_array("comorbidities", "comorbidities_shape", np.float32)
        lab_values = reconstruct_array("lab_values", "lab_values_shape", np.float32)

        sq_keys_str = get_bytes("signal_quality_keys").decode("utf-8")
        sq_values = get_float_list("signal_quality_values")
        signal_quality: Dict[str, float] = {}
        if sq_keys_str:
            keys = sq_keys_str.split(",")
            signal_quality = {k: v for k, v in zip(keys, sq_values)}

        return WindowSample(
            ecg=ecg,
            accelerometer=accelerometer,
            gyroscope=gyroscope,
            ppg=ppg,
            spo2=spo2,
            temperature=temperature,
            respiration=respiration,
            demographics=demographics,
            medications=medications,
            comorbidities=comorbidities,
            lab_values=lab_values,
            ohca_label=get_float("ohca_label"),
            time_to_event=get_float("time_to_event"),
            event_indicator=get_float("event_indicator"),
            heart_rate=get_float("heart_rate"),
            rhythm=get_int("rhythm"),
            activity_state=get_int("activity_state"),
            spo2_mean=get_float("spo2_mean"),
            sbp_mean=get_float("sbp_mean"),
            dbp_mean=get_float("dbp_mean"),
            patient_id=get_bytes("patient_id").decode("utf-8"),
            window_start_hours=get_float("window_start_hours"),
            window_duration_hours=get_float("window_duration_hours"),
            signal_quality=signal_quality,
        )

    def write_tfrecord(
        self,
        samples: List[WindowSample],
        filename: str = "samples.tfrecord",
    ) -> str:
        """
        Write a list of WindowSamples to a TFRecord file.

        Args:
            samples: List of WindowSample objects.
            filename: Output filename.

        Returns:
            Path to the written TFRecord file.
        """
        path = os.path.join(self.tfrecord_dir, filename)
        with tf.io.TFRecordWriter(path) as writer:
            for sample in samples:
                serialized = self.serialize_window_sample(sample)
                writer.write(serialized)
        return path

    def read_tfrecord(self, filename: str = "samples.tfrecord") -> List[WindowSample]:
        """
        Read WindowSamples from a TFRecord file.

        Args:
            filename: Input filename relative to tfrecord_dir.

        Returns:
            List of deserialized WindowSample objects.
        """
        path = os.path.join(self.tfrecord_dir, filename)
        samples: List[WindowSample] = []
        for raw_record in tf.data.TFRecordDataset(path):
            sample = self.deserialize_window_sample(raw_record.numpy())
            samples.append(sample)
        return samples

    def write_tfrecord_sharded(
        self,
        samples: List[WindowSample],
        basename: str = "samples",
        num_shards: int = 10,
    ) -> List[str]:
        """
        Write samples across multiple sharded TFRecord files.

        Args:
            samples: List of WindowSample objects.
            basename: Base filename prefix.
            num_shards: Number of output shards.

        Returns:
            List of paths to the written TFRecord shard files.
        """
        paths: List[str] = []
        shard_size = max(1, len(samples) // num_shards)
        for shard_idx in range(num_shards):
            start = shard_idx * shard_size
            if shard_idx == num_shards - 1:
                shard_samples = samples[start:]
            else:
                shard_samples = samples[start:start + shard_size]
            if not shard_samples:
                continue
            filename = f"{basename}_{shard_idx:05d}.tfrecord"
            path = self.write_tfrecord(shard_samples, filename)
            paths.append(path)
        return paths

    # ------------------------------------------------------------------ #
    #                          Dataset Pipeline                           #
    # ------------------------------------------------------------------ #

    def _get_feature_spec(self) -> Dict[str, Any]:
        """Return the tf.io parsing specification for WindowSample records."""
        return {
            "ecg": tf.io.FixedLenFeature([], tf.string),
            "ecg_shape": tf.io.VarLenFeature(tf.int64),
            "accelerometer": tf.io.FixedLenFeature([], tf.string),
            "accelerometer_shape": tf.io.VarLenFeature(tf.int64),
            "gyroscope": tf.io.FixedLenFeature([], tf.string),
            "gyroscope_shape": tf.io.VarLenFeature(tf.int64),
            "ppg": tf.io.FixedLenFeature([], tf.string),
            "ppg_shape": tf.io.VarLenFeature(tf.int64),
            "spo2": tf.io.FixedLenFeature([], tf.string),
            "spo2_shape": tf.io.VarLenFeature(tf.int64),
            "temperature": tf.io.FixedLenFeature([], tf.string),
            "temperature_shape": tf.io.VarLenFeature(tf.int64),
            "respiration": tf.io.FixedLenFeature([], tf.string),
            "respiration_shape": tf.io.VarLenFeature(tf.int64),
            "demographics": tf.io.FixedLenFeature([], tf.string),
            "demographics_shape": tf.io.VarLenFeature(tf.int64),
            "medications": tf.io.FixedLenFeature([], tf.string),
            "medications_shape": tf.io.VarLenFeature(tf.int64),
            "comorbidities": tf.io.FixedLenFeature([], tf.string),
            "comorbidities_shape": tf.io.VarLenFeature(tf.int64),
            "lab_values": tf.io.FixedLenFeature([], tf.string),
            "lab_values_shape": tf.io.VarLenFeature(tf.int64),
            "ohca_label": tf.io.FixedLenFeature([], tf.float32),
            "time_to_event": tf.io.FixedLenFeature([], tf.float32),
            "event_indicator": tf.io.FixedLenFeature([], tf.float32),
            "heart_rate": tf.io.FixedLenFeature([], tf.float32),
            "rhythm": tf.io.FixedLenFeature([], tf.int64),
            "activity_state": tf.io.FixedLenFeature([], tf.int64),
            "spo2_mean": tf.io.FixedLenFeature([], tf.float32),
            "sbp_mean": tf.io.FixedLenFeature([], tf.float32),
            "dbp_mean": tf.io.FixedLenFeature([], tf.float32),
            "patient_id": tf.io.FixedLenFeature([], tf.string),
            "window_start_hours": tf.io.FixedLenFeature([], tf.float32),
            "window_duration_hours": tf.io.FixedLenFeature([], tf.float32),
        }

    def _parse_example(self, raw: tf.Tensor) -> Dict[str, tf.Tensor]:
        """Parse a single serialized Example into a dict of tensors."""
        parsed = tf.io.parse_single_example(raw, self._get_feature_spec())

        def reconstruct(parsed_dict: Dict[str, tf.Tensor], key: str, shape_key: str, dtype: tf.DType) -> tf.Tensor:
            flat = tf.io.decode_raw(parsed_dict[key], dtype)
            shape = tf.cast(tf.sparse.to_dense(parsed_dict[shape_key]), tf.int32)
            return tf.reshape(flat, shape)

        result: Dict[str, tf.Tensor] = {}
        tensor_fields = [
            ("ecg", "ecg_shape", tf.float32),
            ("accelerometer", "accelerometer_shape", tf.float32),
            ("gyroscope", "gyroscope_shape", tf.float32),
            ("ppg", "ppg_shape", tf.float32),
            ("spo2", "spo2_shape", tf.float32),
            ("temperature", "temperature_shape", tf.float32),
            ("respiration", "respiration_shape", tf.float32),
            ("demographics", "demographics_shape", tf.float32),
            ("medications", "medications_shape", tf.float32),
            ("comorbidities", "comorbidities_shape", tf.float32),
            ("lab_values", "lab_values_shape", tf.float32),
        ]
        for field_name, shape_field, dtype in tensor_fields:
            result[field_name] = reconstruct(parsed, field_name, shape_field, dtype)

        result["ohca_label"] = parsed["ohca_label"]
        result["time_to_event"] = parsed["time_to_event"]
        result["event_indicator"] = parsed["event_indicator"]
        result["heart_rate"] = parsed["heart_rate"]
        result["rhythm"] = tf.cast(parsed["rhythm"], tf.int32)
        result["activity_state"] = tf.cast(parsed["activity_state"], tf.int32)
        result["spo2_mean"] = parsed["spo2_mean"]
        result["sbp_mean"] = parsed["sbp_mean"]
        result["dbp_mean"] = parsed["dbp_mean"]
        result["patient_id"] = parsed["patient_id"]
        result["window_start_hours"] = parsed["window_start_hours"]
        result["window_duration_hours"] = parsed["window_duration_hours"]

        return result

    def create_dataset(
        self,
        filenames: Union[str, List[str]],
        batch_size: int = 32,
        shuffle: bool = True,
        shuffle_buffer: int = 10000,
        prefetch: bool = True,
        num_parallel_reads: int = tf.data.AUTOTUNE if _HAS_TF else None,
        num_parallel_calls: int = tf.data.AUTOTUNE if _HAS_TF else None,
        drop_remainder: bool = True,
    ) -> tf.data.Dataset:
        """
        Create a tf.data.Dataset pipeline from TFRecord file(s).

        Args:
            filenames: Single filename or list of filenames (relative to
                       tfrecord_dir).
            batch_size: Batch size.
            shuffle: Whether to shuffle.
            shuffle_buffer: Shuffle buffer size.
            prefetch: Whether to prefetch.
            num_parallel_reads: Parallel reads (AUTOTUNE recommended).
            num_parallel_calls: Parallel parse calls (AUTOTUNE recommended).
            drop_remainder: Drop incomplete final batch.

        Returns:
            tf.data.Dataset yielding dicts of parsed tensors.
        """
        if isinstance(filenames, str):
            filenames = [filenames]
        full_paths = [os.path.join(self.tfrecord_dir, f) for f in filenames]
        dataset = tf.data.TFRecordDataset(
            full_paths, num_parallel_reads=num_parallel_reads
        )
        if shuffle:
            dataset = dataset.shuffle(shuffle_buffer, reshuffle_each_iteration=True)
        dataset = dataset.map(self._parse_example, num_parallel_calls=num_parallel_calls)
        dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
        if prefetch:
            dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset

    # ------------------------------------------------------------------ #
    #                          Checkpointing                              #
    # ------------------------------------------------------------------ #

    def save_checkpoint(
        self,
        model: Any,
        optimizer: Any,
        epoch: int,
        extra_state: Optional[Dict[str, Any]] = None,
        checkpoint_name: str = "ckpt-{epoch:04d}",
    ) -> str:
        """
        Save a training checkpoint with model and optimizer weights.

        Args:
            model: tf.keras.Model instance.
            optimizer: tf.keras.optimizers.Optimizer instance.
            epoch: Current epoch number.
            extra_state: Optional additional state dict (e.g., metrics, RNG state).
            checkpoint_name: Name template for the checkpoint directory.

        Returns:
            Path to the saved checkpoint directory.
        """
        ckpt_name = checkpoint_name.format(epoch=epoch)
        ckpt_dir = os.path.join(self.checkpoint_dir, ckpt_name)
        os.makedirs(ckpt_dir, exist_ok=True)

        model_path = os.path.join(ckpt_dir, "model.weights.h5")
        model.save_weights(model_path)

        optimizer_path = os.path.join(ckpt_dir, "optimizer.npz")
        np.savez(optimizer_path, **{
            f"var_{i}": v.numpy() for i, v in enumerate(optimizer.variables)
        })

        metadata = {
            "epoch": epoch,
            "timestamp": time.time(),
        }
        if extra_state is not None:
            for k, v in extra_state.items():
                if isinstance(v, (int, float, str, bool)):
                    metadata[k] = v

        meta_path = os.path.join(ckpt_dir, "metadata.npy")
        np.save(meta_path, metadata)

        return ckpt_dir

    def load_checkpoint(
        self,
        model: Any,
        optimizer: Any,
        checkpoint_name: str,
    ) -> Dict[str, Any]:
        """
        Load a training checkpoint.

        Args:
            model: tf.keras.Model to load weights into.
            optimizer: tf.keras.optimizers.Optimizer to load state into.
            checkpoint_name: Name of the checkpoint directory.

        Returns:
            Metadata dict from the checkpoint.
        """
        ckpt_dir = os.path.join(self.checkpoint_dir, checkpoint_name)

        model_path = os.path.join(ckpt_dir, "model.weights.h5")
        model.load_weights(model_path)

        optimizer_path = os.path.join(ckpt_dir, "optimizer.npz")
        opt_data = np.load(optimizer_path, allow_pickle=True)
        for i, var in enumerate(optimizer.variables):
            key = f"var_{i}"
            if key in opt_data:
                var.assign(opt_data[key])

        meta_path = os.path.join(ckpt_dir, "metadata.npy")
        metadata = dict(np.load(meta_path, allow_pickle=True).item())
        return metadata

    def find_latest_checkpoint(self) -> Optional[str]:
        """Find the most recent checkpoint directory by name sorting."""
        if not os.path.isdir(self.checkpoint_dir):
            return None
        entries = sorted(
            [
                d for d in os.listdir(self.checkpoint_dir)
                if os.path.isdir(os.path.join(self.checkpoint_dir, d))
            ]
        )
        return entries[-1] if entries else None

    # ------------------------------------------------------------------ #
    #                       Model Export                                  #
    # ------------------------------------------------------------------ #

    def export_saved_model(
        self,
        model: Any,
        version: int = 1,
        serving_input_signature: Optional[Any] = None,
    ) -> str:
        """
        Export a tf.keras.Model as a TensorFlow SavedModel.

        Args:
            model: tf.keras.Model to export.
            version: Version subdirectory number.
            serving_input_signature: Optional serving input signature for
                                     tf.function tracing. If None, a default
                                     signature is inferred from model input.

        Returns:
            Path to the exported SavedModel directory.
        """
        export_path = os.path.join(self.export_dir, "saved_model", str(version))
        os.makedirs(export_path, exist_ok=True)

        @tf.function(input_signature=[serving_input_signature] if serving_input_signature else None)
        def serving_fn(*args: Any, **kwargs: Any) -> Any:
            return model(*args, **kwargs)

        model.export(export_path)
        return export_path

    def export_tflite(
        self,
        model: Any,
        quantize: bool = False,
        optimize: bool = True,
        representative_dataset_fn: Optional[Any] = None,
        filename: str = "model.tflite",
    ) -> str:
        """
        Export a tf.kerasModel to TFLite format.

        Args:
            model: tf.keras.Model to convert.
            quantize: Apply full integer quantization (requires representative_dataset_fn).
            optimize: Apply default optimizations (float16).
            representative_dataset_fn: Callable yielding representative data
                                        for quantization calibration.
            filename: Output filename.

        Returns:
            Path to the written .tflite file.
        """
        converter = tf.lite.TFLiteConverter.from_keras_model(model)

        optimizations = []
        if optimize:
            optimizations.append(tf.lite.Optimize.DEFAULT)
        if quantize:
            optimizations.append(tf.lite.Optimize.DEFAULT)
            converter.optimizations = set(optimizations)
            if representative_dataset_fn is not None:
                converter.representative_dataset = representative_dataset_fn
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS_INT8
            ]
            converter.inference_input_type = tf.float32
            converter.inference_output_type = tf.float32
        else:
            if optimizations:
                converter.optimizations = set(optimizations)

        tflite_model = converter.convert()

        export_path = os.path.join(self.export_dir, filename)
        with open(export_path, "wb") as f:
            f.write(tflite_model)
        return export_path

    def export_onnx(
        self,
        model: Any,
        filename: str = "model.onnx",
        input_shape: Optional[Tuple[int, ...]] = None,
    ) -> str:
        """
        Export a tf.keras.Model to ONNX format using tf2onnx.

        Args:
            model: tf.keras.Model to convert.
            filename: Output filename.
            input_shape: Optional input shape tuple for dynamic axes.

        Returns:
            Path to the written .onnx file.
        """
        try:
            import tf2onnx
            import onnx
        except ImportError:
            raise ImportError("tf2onnx and onnx are required for ONNX export: pip install tf2onnx onnx")

        spec = (tf.TensorSpec(model.input_shape, tf.float32, name="input"),)
        output_path = os.path.join(self.export_dir, filename)
        model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, output_path=output_path)
        return output_path

    def write_samples_to_csv(
        self,
        samples: List[WindowSample],
        filename: str = "window_samples.csv",
    ) -> str:
        """
        Write WindowSample metadata to CSV (signals stored as shape summaries).

        Args:
            samples: List of WindowSample objects.
            filename: Output filename.

        Returns:
            Path to the written CSV file.
        """
        import csv
        path = os.path.join(self.tfrecord_dir, filename)
        fieldnames = [
            "patient_id", "ohca_label", "time_to_event", "event_indicator",
            "heart_rate", "rhythm", "activity_state", "spo2_mean", "sbp_mean",
            "dbp_mean", "window_start_hours", "window_duration_hours",
            "ecg_shape", "accelerometer_shape", "gyroscope_shape", "ppg_shape",
            "spo2_shape", "temperature_shape", "respiration_shape",
            "demographics_shape", "medications_shape", "comorbidities_shape",
            "lab_values_shape", "signal_quality",
        ]
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for s in samples:
                writer.writerow({
                    "patient_id": s.patient_id,
                    "ohca_label": s.ohca_label,
                    "time_to_event": s.time_to_event,
                    "event_indicator": s.event_indicator,
                    "heart_rate": s.heart_rate,
                    "rhythm": s.rhythm,
                    "activity_state": s.activity_state,
                    "spo2_mean": s.spo2_mean,
                    "sbp_mean": s.sbp_mean,
                    "dbp_mean": s.dbp_mean,
                    "window_start_hours": s.window_start_hours,
                    "window_duration_hours": s.window_duration_hours,
                    "ecg_shape": str(s.ecg.shape),
                    "accelerometer_shape": str(s.accelerometer.shape),
                    "gyroscope_shape": str(s.gyroscope.shape),
                    "ppg_shape": str(s.ppg.shape),
                    "spo2_shape": str(s.spo2.shape),
                    "temperature_shape": str(s.temperature.shape),
                    "respiration_shape": str(s.respiration.shape),
                    "demographics_shape": str(s.demographics.shape),
                    "medications_shape": str(s.medications.shape),
                    "comorbidities_shape": str(s.comorbidities.shape),
                    "lab_values_shape": str(s.lab_values.shape),
                    "signal_quality": str(s.signal_quality),
                })


# ------------------------------------------------------------------ #
#           Variable-length padded dataset from WindowSamples          #
# ------------------------------------------------------------------ #

def create_padded_dataset(
    samples: List[WindowSample],
    batch_size: int = 8,
    shuffle: bool = True,
    max_signal_length: int = 50000,
) -> "tf.data.Dataset":
    """Create a tf.data.Dataset from WindowSamples with per-batch padding.

    Since signals have variable lengths, each batch is padded to the
    maximum length within that batch. This avoids wasting memory on
    global padding to the longest signal across the entire dataset.

    Args:
        samples: List of WindowSample objects.
        batch_size: Batch size.
        shuffle: Whether to shuffle.

    Returns:
        A tf.data.Dataset yielding ``(inputs_dict, labels_dict)`` tuples.
    """
    signal_keys = ["ecg", "accelerometer", "gyroscope", "ppg", "spo2", "temperature", "respiration"]
    static_keys = ["demographics", "medications", "comorbidities", "lab_values"]
    label_keys = ["ohca_label", "time_to_event", "event_indicator",
                   "heart_rate", "rhythm", "activity_state", "spo2_mean",
                   "sbp_mean", "dbp_mean"]

    def _sample_to_dict(s: WindowSample) -> Dict[str, tf.Tensor]:
        d = {}
        for k in signal_keys:
            arr = getattr(s, k)
            if arr.ndim == 1:
                arr = arr[:, np.newaxis]
            # Truncate to max length
            if max_signal_length is not None and arr.shape[0] > max_signal_length:
                arr = arr[:max_signal_length]
            d[k] = tf.constant(arr, dtype=tf.float32)
        for k in static_keys:
            d[k] = tf.constant(getattr(s, k), dtype=tf.float32)
        d["ohca_label"] = tf.constant([s.ohca_label], dtype=tf.float32)
        d["time_to_event"] = tf.constant([s.time_to_event], dtype=tf.float32)
        d["event_indicator"] = tf.constant([s.event_indicator], dtype=tf.float32)
        d["heart_rate"] = tf.constant([s.heart_rate], dtype=tf.float32)
        d["rhythm"] = tf.constant([s.rhythm], dtype=tf.int32)
        d["activity_state"] = tf.constant([s.activity_state], dtype=tf.int32)
        d["spo2_mean"] = tf.constant([s.spo2_mean], dtype=tf.float32)
        d["sbp_mean"] = tf.constant([s.sbp_mean], dtype=tf.float32)
        d["dbp_mean"] = tf.constant([s.dbp_mean], dtype=tf.float32)
        return d

    element_spec = None
    if len(samples) > 0:
        element_spec = {k: tf.TensorSpec(shape=None, dtype=v.dtype)
                        for k, v in _sample_to_dict(samples[0]).items()}

    ds = tf.data.Dataset.from_generator(
        lambda: (_sample_to_dict(s) for s in samples),
        output_signature=element_spec,
    )

    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(samples), 1000))

    def _pad_batch(batch_dict):
        padded = {}
        for key in signal_keys:
            if key in batch_dict:
                tensors = batch_dict[key]
                max_len = tf.reduce_max([tf.shape(t)[0] for t in tensors])
                padded[key] = tf.stack([
                    tf.pad(t, [(0, max_len - tf.shape(t)[0]), (0, 0)])
                    for t in tensors
                ])
        for key in static_keys + ["ohca_label", "time_to_event", "event_indicator",
                                   "heart_rate", "spo2_mean", "sbp_mean", "dbp_mean"]:
            if key in batch_dict:
                padded[key] = tf.stack(batch_dict[key])
        for key in ["rhythm", "activity_state"]:
            if key in batch_dict:
                padded[key] = tf.cast(tf.stack(batch_dict[key]), tf.int32)
        batch_size_actual = tf.shape(padded.get("ohca_label", tf.zeros(1)))[0]
        n_survival_bins = 12
        tte = padded.get("time_to_event", tf.zeros((batch_size_actual,)))
        evt = padded.get("event_indicator", tf.zeros((batch_size_actual,)))
        max_dur = 48.0
        bin_edges = tf.linspace(0.0, max_dur, n_survival_bins + 1)
        tte_expanded = tf.expand_dims(tte, 1)
        bin_edges_expanded = tf.expand_dims(bin_edges, 0)
        bin_idx = tf.reduce_sum(tf.cast(tte_expanded > bin_edges_expanded, tf.float32), axis=1)
        bin_idx = tf.cast(tf.clip_by_value(bin_idx, 0, n_survival_bins - 1), tf.int32)
        bin_mask = tf.one_hot(bin_idx, depth=n_survival_bins, dtype=tf.float32)
        event_mask = tf.expand_dims(evt, 1)
        censored_mask = 1.0 - event_mask
        survival_labels = bin_mask * event_mask
        censored_bins = tf.cast(
            tf.sequence_mask(
                tf.cast(tf.clip_by_value(
                    tf.reduce_sum(tf.cast(tte_expanded > bin_edges_expanded, tf.float32), axis=1),
                    0, n_survival_bins
                ), tf.int32),
                maxlen=n_survival_bins,
            ),
            tf.float32,
        )
        survival_labels = survival_labels + censored_bins * censored_mask
        padded["survival_labels"] = survival_labels
        return padded

    ds = ds.padded_batch(
        batch_size,
        padded_shapes={
            "ecg": [None, None],
            "accelerometer": [None, None],
            "gyroscope": [None, None],
            "ppg": [None, None],
            "spo2": [None, None],
            "temperature": [None, None],
            "respiration": [None, None],
            "demographics": [None],
            "medications": [None],
            "comorbidities": [None],
            "lab_values": [None],
            "ohca_label": [None],
            "time_to_event": [None],
            "event_indicator": [None],
            "heart_rate": [None],
            "rhythm": [None],
            "activity_state": [None],
            "spo2_mean": [None],
            "sbp_mean": [None],
            "dbp_mean": [None],
        },
        padding_values={
            "ecg": 0.0,
            "accelerometer": 0.0,
            "gyroscope": 0.0,
            "ppg": 0.0,
            "spo2": 0.0,
            "temperature": 0.0,
            "respiration": 0.0,
            "demographics": 0.0,
            "medications": 0.0,
            "comorbidities": 0.0,
            "lab_values": 0.0,
            "ohca_label": 0.0,
            "time_to_event": 0.0,
            "event_indicator": 0.0,
            "heart_rate": 0.0,
            "rhythm": 0,
            "activity_state": 0,
            "spo2_mean": 0.0,
            "sbp_mean": 0.0,
            "dbp_mean": 0.0,
        },
        drop_remainder=False,
    )

    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


# ------------------------------------------------------------------ #
#    Trainer-compatible dataset: yields (inputs_dict, targets_dict)   #
# ------------------------------------------------------------------ #

def create_trainer_dataset(
    samples: List[WindowSample],
    batch_size: int = 8,
    shuffle: bool = True,
    max_signal_length: int = 50000,
) -> "tf.data.Dataset":
    """Create a tf.data.Dataset that yields ``(inputs_dict, targets_dict)`` tuples.

    This is the format expected by ``OHCATrainer``.  Inputs contain
    sensor signals and static features; targets contain labels and
    auxiliary supervision.

    Args:
        samples: List of ``WindowSample`` objects.
        batch_size: Batch size.
        shuffle: Whether to shuffle.
        max_signal_length: Maximum signal length before truncation.

    Returns:
        A ``tf.data.Dataset`` yielding ``(inputs_dict, targets_dict)``.
    """
    input_keys = [
        "ecg", "accelerometer", "gyroscope", "ppg",
        "spo2", "temperature", "respiration",
        "demographics", "medications", "comorbidities", "lab_values",
    ]
    target_keys = [
        "ohca_label", "time_to_event", "event_indicator",
        "heart_rate", "spo2_mean", "sbp_mean", "dbp_mean",
    ]
    int_target_keys = ["rhythm", "activity_state"]
    signal_keys = [
        "ecg", "accelerometer", "gyroscope", "ppg",
        "spo2", "temperature", "respiration",
    ]

    def _sample_to_pair(s: WindowSample):
        inputs = {}
        targets = {}

        for k in signal_keys:
            arr = getattr(s, k)
            if arr.ndim == 1:
                arr = arr[:, np.newaxis]
            if max_signal_length is not None and arr.shape[0] > max_signal_length:
                arr = arr[:max_signal_length]
            inputs[k] = tf.constant(arr, dtype=tf.float32)

        for k in ["demographics", "medications", "comorbidities", "lab_values"]:
            inputs[k] = tf.constant(getattr(s, k), dtype=tf.float32)

        for k in target_keys:
            val = getattr(s, k)
            targets[k] = tf.constant([val], dtype=tf.float32)

        for k in int_target_keys:
            val = getattr(s, k)
            targets[k] = tf.constant([val], dtype=tf.int32)

        targets["heart_rate"] = tf.constant([s.heart_rate], dtype=tf.float32)

        targets["survival_labels"] = tf.zeros((1, 13), dtype=tf.float32)

        return inputs, targets

    ds = tf.data.Dataset.from_generator(
        lambda: (_sample_to_pair(s) for s in samples),
        output_signature=(
            {
                k: tf.TensorSpec(shape=(None, None), dtype=tf.float32)
                if k in signal_keys
                else tf.TensorSpec(shape=(None,), dtype=tf.float32)
                for k in input_keys
            },
            {
                "ohca_label": tf.TensorSpec(shape=(None,), dtype=tf.float32),
                "time_to_event": tf.TensorSpec(shape=(None,), dtype=tf.float32),
                "event_indicator": tf.TensorSpec(shape=(None,), dtype=tf.float32),
                "heart_rate": tf.TensorSpec(shape=(None,), dtype=tf.float32),
                "spo2_mean": tf.TensorSpec(shape=(None,), dtype=tf.float32),
                "sbp_mean": tf.TensorSpec(shape=(None,), dtype=tf.float32),
                "dbp_mean": tf.TensorSpec(shape=(None,), dtype=tf.float32),
                "rhythm": tf.TensorSpec(shape=(None,), dtype=tf.int32),
                "activity_state": tf.TensorSpec(shape=(None,), dtype=tf.int32),
                "survival_labels": tf.TensorSpec(shape=(None, 13), dtype=tf.float32),
            },
        ),
    )

    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(samples), 1000))

    def _pad_pair(inputs, targets):
        padded_inputs = {}
        for key in signal_keys:
            tensors = inputs[key]
            max_len = tf.reduce_max([tf.shape(t)[0] for t in tensors])
            padded_inputs[key] = tf.stack([
                tf.pad(t, [(0, max_len - tf.shape(t)[0]), (0, 0)])
                for t in tensors
            ])
        for key in ["demographics", "medications", "comorbidities", "lab_values"]:
            padded_inputs[key] = tf.stack(inputs[key])

        padded_targets = {}
        for key in target_keys:
            padded_targets[key] = tf.stack(targets[key])
        for key in int_target_keys:
            padded_targets[key] = tf.cast(tf.stack(targets[key]), tf.int32)

        bs = tf.shape(padded_targets["ohca_label"])[0]
        n_bins = 12
        tte = padded_targets["time_to_event"]
        evt = padded_targets["event_indicator"]
        bin_edges = tf.linspace(0.0, 48.0, n_bins + 1)
        tte_exp = tf.expand_dims(tte, 1)
        be_exp = tf.expand_dims(bin_edges, 0)
        bin_idx = tf.reduce_sum(tf.cast(tte_exp > be_exp, tf.float32), axis=1)
        bin_idx = tf.cast(tf.clip_by_value(bin_idx, 0, n_bins - 1), tf.int32)
        bin_mask = tf.one_hot(bin_idx, depth=n_bins, dtype=tf.float32)
        evt_exp = tf.expand_dims(evt, 1)
        surv = bin_mask * evt_exp
        padded_targets["survival_labels"] = surv

        return padded_inputs, padded_targets

    ds = ds.padded_batch(batch_size, padding_values=0.0, drop_remainder=False)
    ds = ds.map(_pad_pair, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds
