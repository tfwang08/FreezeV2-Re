from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GPUProfile:
    name: str
    python_minor: str
    torch_version: str
    torchvision_version: str
    torchaudio_version: str
    cuda_tag: str
    cuda_version: str
    pointnet_arch: str


def resolve_gpu_profile(capability: tuple[int, int]) -> GPUProfile:
    """Resolve the supported CUDA software profile for one GPU capability."""
    major, minor = map(int, capability)
    if (major, minor) == (8, 0):
        return GPUProfile(
            name="ampere",
            python_minor="3.11",
            torch_version="2.2.2",
            torchvision_version="0.17.2",
            torchaudio_version="2.2.2",
            cuda_tag="cu121",
            cuda_version="12.1",
            pointnet_arch="8.0",
        )
    if (major, minor) == (12, 0):
        return GPUProfile(
            name="blackwell",
            python_minor="3.11",
            torch_version="2.7.1",
            torchvision_version="0.22.1",
            torchaudio_version="2.7.1",
            cuda_tag="cu128",
            cuda_version="12.8",
            pointnet_arch="12.0",
        )
    raise ValueError(
        f"unsupported CUDA compute capability {major}.{minor}; "
        "supported profiles are 8.0 (A800/A100) and 12.0 "
        "(RTX 50-series Blackwell)"
    )
