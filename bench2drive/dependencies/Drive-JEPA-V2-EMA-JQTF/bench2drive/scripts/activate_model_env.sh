#!/usr/bin/env bash
# Source before running Drive-JEPA in the Jinn task container.

set -u

source /home/myuser/miniconda3/etc/profile.d/conda.sh
conda activate drive-jepa

drive_jepa_python_version="$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
drive_jepa_torch_lib="${CONDA_PREFIX}/lib/python${drive_jepa_python_version}/site-packages/torch/lib"

# Do not inherit system/base-image PyTorch or CUDA user libraries. Keep the
# conda PyTorch libraries first and retain only host driver/compat locations.
export LD_LIBRARY_PATH="${drive_jepa_torch_lib}:${CONDA_PREFIX}/lib:/usr/local/nvidia/lib64:/usr/local/nvidia/lib:/usr/local/cuda/compat/lib"
export PYTHONNOUSERSITE=1
unset PYTHONHOME

unset drive_jepa_python_version drive_jepa_torch_lib

