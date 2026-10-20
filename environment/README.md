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

## DA-WAM cache names

NAVSIM 训练缓存默认 `${NAVSIM_EXP_ROOT}/train_da_wam_future_cache`，样本文件为 `da_wam_feature.gz` 和 `da_wam_target.gz`。用 `CACHE_PATH` 指定其他训练缓存目录。旧文件名需要重命名或重新生成。

NAVSIM metric/anchor 缓存默认根目录为 `${NAVSIM_EXP_ROOT}/da_wam_cache`，可用 `DA_WAM_METRIC_CACHE_ROOT` 指向已有根目录；评测也可用 `METRIC_CACHE_PATH` 指定具体 metric 缓存。Bench2Drive 的基础 scorer 缓存目录名为 `da_wam_base`，数值缓存使用 `da_wam_` 前缀。改动源码后需重建 Bench2Drive 缓存包，使源码哈希校验一致。

## NAVSIM v1 training metric cache

`train_metric_cache` 提供场景状态和地图信息，供训练时在线计算 proposal 分数及离线计算 anchor 分数使用。它不同于相机 feature/target 缓存，也不同于完整 anchor score 缓存。

从 [Drive-JEPA 数据集](https://huggingface.co/datasets/LinhanWang/Drive-JEPA/blob/main/train_metric_cache.tar) 下载 `train_metric_cache.tar`，解压到 `${NAVSIM_EXP_ROOT}/da_wam_cache`（或 `DA_WAM_METRIC_CACHE_ROOT`），得到 `train_metric_cache/metadata/train_metric_cache_metadata_node_0.csv`。

下载包中的 metadata CSV 可能保留原机器的绝对路径。训练前用下面的命令备份并更新路径；它会检查目标文件存在，不修改 metric cache 内容。设置路径后，从仓库根目录执行：

```bash
export NAVSIM_EXP_ROOT=/path/to/navsim_exp
export DA_WAM_METRIC_CACHE_ROOT="${NAVSIM_EXP_ROOT}/da_wam_cache"
python - <<'PY'
import csv
import os
import shutil
from pathlib import Path

root = Path(os.environ["DA_WAM_METRIC_CACHE_ROOT"]).expanduser().resolve() / "train_metric_cache"
metadata = root / "metadata" / "train_metric_cache_metadata_node_0.csv"
with metadata.open(newline="") as stream:
    reader = csv.DictReader(stream)
    fields = reader.fieldnames
    rows = list(reader)
for row in rows:
    parts = Path(row["file_name"]).parts
    marker = len(parts) - 1 - parts[::-1].index("train_metric_cache")
    local = root.joinpath(*parts[marker + 1:])
    if not local.is_file():
        raise FileNotFoundError(local)
    row["file_name"] = str(local)
backup = metadata.with_suffix(".csv.bak")
if not backup.exists():
    shutil.copy2(metadata, backup)
temporary = metadata.with_suffix(".csv.tmp")
with temporary.open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
temporary.replace(metadata)
print(f"Updated {len(rows)} metric-cache paths; backup: {backup}")
PY
```

完整分数缓存的下载、计算和校验命令见 [README](../README.md#prepare-the-navsim-v1-anchor-score-cache)。计算入口为 `scripts/cache_navsim_anchor_scores.sh`，默认 8 个 CPU 进程、每次评分 256 条 anchors；可设置 `ANCHOR_MAX_WORKERS`、`ANCHOR_CHUNK_SIZE`。每个进程的原生数学库线程数固定为 1，避免进程和线程过度叠加。此步骤不需要 GPU。
