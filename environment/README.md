# Environment

模型使用 Python 3.9.25 / PyTorch 2.1.0+cu121；CARLA 使用独立 Python 3.8.20 / CARLA 0.9.15。

## 使用本机已有模型环境

```bash
source environment/activate_model.sh
python -c 'import torch; print(torch.__version__, torch.version.cuda)'
```

默认 `MODEL_ENV=/home/myuser/miniconda3/envs/drive-jepa`。无需覆盖原环境。

## 新机器安装模型环境

```bash
conda create -n da-wam-model python=3.9.25 pip -y
conda activate da-wam-model
python -m pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
python -c "from pathlib import Path; s=Path('environment/navsim-requirements-original.txt').read_text(); Path('/tmp/da_wam_requirements.txt').write_text('\n'.join(x for x in s.splitlines() if not x.startswith(('mmcv_full','-f '))))"
python -m pip install -r /tmp/da_wam_requirements.txt
MMCV_WITH_OPS=1 TORCH_CUDA_ARCH_LIST='8.0;8.9;9.0' python -m pip install mmcv-full==1.7.2 --no-binary mmcv-full
```

`navsim.yml` 和 `model-installed-versions.txt` 另提供本机包版本快照。新机器的安装尚未验证；H800/L20 的 MMCV 使用源码编译方式。

## CARLA 客户端

```bash
conda env create -f environment/bench2drive.yml
conda activate da-wam-carla
export B2D_EVAL_ENV="$CONDA_PREFIX"
source bench2drive/Bench2Drive/env.sh
python -c 'import carla; print(carla.__file__)'
```

本机默认评测环境 `/tmp/b2d_road_env_20260921`；可用 `B2D_EVAL_ENV` 覆盖。CARLA 0.9.15 默认 `/mnt/c2-worldmodel/2639639/Bench2Drive/CARLA_0.9.15`，需地图、非 root 用户和 NVIDIA graphics/Vulkan。原生 CARLA 进程不要继承模型的 Conda/Torch 动态库路径；所保留启动器会隔离。

OpenScene 默认 `/mnt/c2-worldmodel/training_data/OpenScene/dataset`，NAVSIM cache 默认 `/mnt/c2-worldmodel/2639639/navsim_exp`；Bench2Drive 数据与缓存默认 `/mnt/c2-worldmodel/2639639/Bench2Drive/` 下原路径。数据、cache、CARLA 二进制不包含在源码中。
