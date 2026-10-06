"""AIFlow submission for 30-epoch PostTraj Stage A + E2 joint training."""

import os
import shlex

from easydict import EasyDict as edict


cfg = edict()
cfg.FRAME = "general"

_user_path = os.environ.get("userPath", "")
_self_dir = os.path.dirname(os.path.abspath(__file__))
_known_root = _self_dir
_runner_rel = os.path.join(
    "scripts", "training", "run_posttraj_future_joint30.sh"
)
_project_root = ""
for _candidate in (
    _user_path,
    os.path.dirname(_user_path) if _user_path else "",
    _known_root,
    _self_dir,
):
    if _candidate and os.path.isfile(os.path.join(_candidate, _runner_rel)):
        _project_root = _candidate
        break

if not _project_root:
    cfg.CMD = (
        "python3 -c \"raise FileNotFoundError("
        "'PostTraj joint30 runner not found')\""
    )
else:
    _runner = os.path.join(_project_root, _runner_rel)
    cfg.CMD = (
        f"cd {shlex.quote(_project_root)} && "
        f"export NAVSIM_DEVKIT_ROOT={shlex.quote(_project_root)} && "
        "export NAVSIM_EXP_ROOT=/mnt/c2-worldmodel/2639639/navsim_exp && "
        "export OPENSCENE_DATA_ROOT=/mnt/c2-worldmodel/training_data/OpenScene/dataset && "
        "export NUM_GPUS=8 && export BATCH_SIZE=8 && "
        "export MAX_EPOCHS=30 && export LR=1e-4 && "
        "export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 && "
        "export EXPERIMENT_NAME=PB_G_PostTrajFuture_Joint30 && "
        f"bash {shlex.quote(_runner)}"
    )
