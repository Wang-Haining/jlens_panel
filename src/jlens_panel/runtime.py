"""Lazy H100 runtime checks and deterministic CUDA settings."""

from __future__ import annotations

import os
import subprocess


class H100RuntimeError(RuntimeError):
    """Raised when a fresh capture lacks the frozen one-H100 runtime."""


def configure_h100_runtime() -> dict[str, object]:
    """Require one visible H100, freeze math settings, and return its identity."""

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise H100RuntimeError("fresh capture requires exactly one visible H100 GPU")
    gpu_name = torch.cuda.get_device_name(0)
    if "H100" not in gpu_name:
        raise H100RuntimeError(f"allocated GPU is not an H100: {gpu_name}")
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    cuda_runtime = torch.version.cuda
    if not isinstance(cuda_runtime, str) or not cuda_runtime:
        raise H100RuntimeError("cannot resolve the CUDA runtime version")
    driver = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        check=False,
        capture_output=True,
        text=True,
    )
    driver_versions = {
        line.strip() for line in driver.stdout.splitlines() if line.strip()
    }
    if driver.returncode != 0 or len(driver_versions) != 1:
        raise H100RuntimeError("cannot resolve the H100 driver version")
    capability = torch.cuda.get_device_capability(0)
    return {
        "torch_version": str(torch.__version__),
        "cuda_runtime": cuda_runtime,
        "cuda_driver_version": next(iter(driver_versions)),
        "gpu_name": gpu_name,
        "gpu_compute_capability": [int(capability[0]), int(capability[1])],
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
    }
