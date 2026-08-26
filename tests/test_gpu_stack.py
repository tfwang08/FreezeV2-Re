import pytest

from freezev2.gpu_stack import resolve_gpu_profile


def test_ampere_profile_keeps_reproduction_stack():
    profile = resolve_gpu_profile((8, 0))
    assert profile.name == "ampere"
    assert profile.python_minor == "3.11"
    assert profile.torch_version == "2.2.2"
    assert profile.cuda_tag == "cu121"
    assert profile.pointnet_arch == "8.0"


def test_blackwell_profile_uses_cu128_and_sm120():
    profile = resolve_gpu_profile((12, 0))
    assert profile.name == "blackwell"
    assert profile.python_minor == "3.11"
    assert profile.torch_version == "2.7.1"
    assert profile.cuda_tag == "cu128"
    assert profile.pointnet_arch == "12.0"


def test_unknown_gpu_profile_is_rejected():
    with pytest.raises(ValueError, match="unsupported CUDA compute capability"):
        resolve_gpu_profile((9, 0))
