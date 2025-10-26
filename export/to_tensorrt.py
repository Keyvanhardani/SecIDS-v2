"""
Export ONNX models to TensorRT for NVIDIA Jetson

TensorRT provides:
  - INT8/FP16 quantization
  - Kernel fusion and optimization
  - 2-5x speedup on Jetson devices
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import sys

# TensorRT is optional (only on Jetson/CUDA systems)
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False
    print("WARNING: TensorRT not available. Install on Jetson/CUDA system.")


class TensorRTEngine:
    """Wrapper for TensorRT inference engine"""

    def __init__(self, engine_path: str):
        if not TRT_AVAILABLE:
            raise ImportError("TensorRT not available")

        self.logger = trt.Logger(trt.Logger.WARNING)

        # Load engine
        with open(engine_path, 'rb') as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()

        # Allocate buffers
        self.inputs, self.outputs, self.bindings, self.stream = self._allocate_buffers()

    def _allocate_buffers(self):
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()

        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding))
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))

            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            bindings.append(int(device_mem))

            if self.engine.binding_is_input(binding):
                inputs.append({'host': host_mem, 'device': device_mem})
            else:
                outputs.append({'host': host_mem, 'device': device_mem})

        return inputs, outputs, bindings, stream

    def infer(self, input_data: np.ndarray) -> np.ndarray:
        """Run inference"""
        # Copy input to device
        np.copyto(self.inputs[0]['host'], input_data.ravel())
        cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)

        # Execute
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)

        # Copy output to host
        cuda.memcpy_dtoh_async(self.outputs[0]['host'], self.outputs[0]['device'], self.stream)
        self.stream.synchronize()

        return self.outputs[0]['host']


def export_to_tensorrt(
    onnx_path: str,
    output_path: str,
    precision: str = "fp16",
    max_batch_size: int = 1,
    max_workspace_size: int = 1 << 30,  # 1GB
    calibration_data: Optional[np.ndarray] = None
) -> str:
    """
    Convert ONNX model to TensorRT engine.

    Args:
        onnx_path: Input ONNX model
        output_path: Output TensorRT engine (.trt or .engine)
        precision: "fp32", "fp16", or "int8"
        max_batch_size: Maximum batch size
        max_workspace_size: Max memory for TRT optimization
        calibration_data: Calibration dataset for INT8 (required for INT8)

    Returns:
        Path to TensorRT engine
    """

    if not TRT_AVAILABLE:
        raise ImportError(
            "TensorRT not available. "
            "Install on NVIDIA Jetson or CUDA system: pip install tensorrt"
        )

    print(f"Converting {onnx_path} to TensorRT...")
    print(f"  Precision: {precision}")
    print(f"  Max batch size: {max_batch_size}")

    # Create builder and network
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    # Parse ONNX
    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            print("ERROR: Failed to parse ONNX file")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            raise RuntimeError("ONNX parsing failed")

    print("✓ ONNX parsed successfully")

    # Builder config
    config = builder.create_builder_config()
    config.max_workspace_size = max_workspace_size

    # Set precision
    if precision == "fp16":
        if not builder.platform_has_fast_fp16:
            print("WARNING: FP16 not supported on this platform, using FP32")
        else:
            config.set_flag(trt.BuilderFlag.FP16)
            print("✓ FP16 enabled")

    elif precision == "int8":
        if not builder.platform_has_fast_int8:
            print("WARNING: INT8 not supported, using FP16")
            config.set_flag(trt.BuilderFlag.FP16)
        else:
            config.set_flag(trt.BuilderFlag.INT8)

            # INT8 requires calibration
            if calibration_data is None:
                raise ValueError("INT8 requires calibration_data")

            # TODO: Implement INT8 calibrator
            print("WARNING: INT8 calibration not fully implemented yet")
            config.set_flag(trt.BuilderFlag.FP16)  # Fallback to FP16

    # Build engine
    print("\nBuilding TensorRT engine (this may take a few minutes)...")
    engine = builder.build_engine(network, config)

    if engine is None:
        raise RuntimeError("Failed to build TensorRT engine")

    print("✓ Engine built successfully")

    # Serialize engine
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'wb') as f:
        f.write(engine.serialize())

    print(f"✓ Engine saved to {output_path}")

    # Print info
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\nEngine info:")
    print(f"  File size: {size_mb:.2f} MB")
    print(f"  Num bindings: {engine.num_bindings}")

    for i in range(engine.num_bindings):
        name = engine.get_binding_name(i)
        shape = engine.get_binding_shape(i)
        dtype = engine.get_binding_dtype(i)
        print(f"  [{i}] {name}: {shape} ({dtype})")

    return str(output_path)


def calibrate_int8(
    onnx_path: str,
    calibration_dataset: np.ndarray,
    cache_path: str = "calibration.cache"
) -> str:
    """
    Generate INT8 calibration cache.

    Args:
        onnx_path: ONNX model to calibrate
        calibration_dataset: Representative data (N, seq_len, features)
        cache_path: Where to save calibration cache

    Returns:
        Path to calibration cache
    """

    # This is a placeholder - full implementation requires EntropyCalibrator
    print("INT8 calibration is complex and device-specific.")
    print("For production INT8, use NVIDIA's tools or post-training quantization.")

    return cache_path


def benchmark_tensorrt(
    engine_path: str,
    input_shape: Tuple[int, int, int] = (1, 128, 32),
    num_runs: int = 100
):
    """
    Benchmark TensorRT engine latency.

    Args:
        engine_path: Path to TensorRT engine
        input_shape: Input shape
        num_runs: Number of iterations
    """

    if not TRT_AVAILABLE:
        print("TensorRT not available for benchmarking")
        return

    import time

    print(f"Benchmarking {engine_path}...")

    # Load engine
    engine = TensorRTEngine(engine_path)

    # Dummy input
    dummy_input = np.random.randn(*input_shape).astype(np.float32)

    # Warmup
    for _ in range(10):
        engine.infer(dummy_input)

    # Benchmark
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        engine.infer(dummy_input)
        end = time.perf_counter()
        times.append((end - start) * 1000)  # ms

    mean_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)

    print(f"\nLatency (over {num_runs} runs):")
    print(f"  Mean: {mean_time:.2f} ms")
    print(f"  Std:  {std_time:.2f} ms")
    print(f"  Min:  {min_time:.2f} ms")
    print(f"  Throughput: {1000 / mean_time:.1f} samples/sec")


# CLI
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export ONNX to TensorRT")
    parser.add_argument("--onnx", type=str, required=True, help="Input ONNX model")
    parser.add_argument("--output", type=str, required=True, help="Output TensorRT engine")
    parser.add_argument("--precision", type=str, default="fp16", choices=["fp32", "fp16", "int8"])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--benchmark", action="store_true")

    args = parser.parse_args()

    engine_path = export_to_tensorrt(
        onnx_path=args.onnx,
        output_path=args.output,
        precision=args.precision,
        max_batch_size=args.batch_size
    )

    if args.benchmark:
        benchmark_tensorrt(engine_path)
