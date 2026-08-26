from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "setup_gedi.sh"


def test_setup_gedi_uses_isolated_cuda_toolkit_prefix():
    text = SCRIPT.read_text()

    assert 'CUDA_TOOLKIT_PREFIX=' in text
    assert 'conda create -y -p "$CUDA_TOOLKIT_PREFIX"' in text
    assert '--override-channels' in text
    assert 'nvidia/label/$CUDA_CONDA_LABEL' in text
    assert 'conda install -y -n freeze -c nvidia' not in text
