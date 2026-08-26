from pathlib import Path


def test_setup_gedi_provisions_complete_isolated_cuda_toolchain():
    script = Path("scripts/setup_gedi.sh").read_text()

    assert '"cuda-libraries-dev=${CUDA_VERSION}.*"' in script
    assert '"gcc_linux-64=12.3.0"' in script
    assert '"gxx_linux-64=12.3.0"' in script
    assert '-c conda-forge' in script
    assert 'export CC="$CUDA_TOOLKIT_PREFIX/bin/x86_64-conda-linux-gnu-cc"' in script
    assert 'export CXX="$CUDA_TOOLKIT_PREFIX/bin/x86_64-conda-linux-gnu-c++"' in script
    assert 'export CUDAHOSTCXX="$CXX"' in script
    assert 'cusparse.h' in script
