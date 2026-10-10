<div align="center">

# DA-WAM: Decision-Aligned Future Latents for Driving World Models

**Ruiguo Zhong · Benshan Ma · Xiaolong Chen · Lang Zhang**  
**Mingyue Feng · Yaonong Wang · Pei Liu · Jun Ma**

[**Paper**](https://arxiv.org/abs/2608.19085) · [**Overview**](#overview) · [**Results**](#results) · [**Visualization**](#visualization) · [**Getting Started**](#getting-started) · [**Citation**](#citation)

</div>

## Overview

**DA-WAM** connects future representation learning to trajectory selection. It first generates trajectory proposals, predicts a distinct future latent for each proposal, and scores each trajectory together with its associated future and the current scene.

- **Candidate-conditioned futures.** Each proposal receives its own compact action-conditioned representation.
- **Expert-matched supervision.** A training-only dense decoder and an EMA target supervise the candidate closest to the expert trajectory.
- **Safety-focused refinement.** Counterfactual trajectories refine the candidate-future path and scorer while keeping proposal generation fixed.

<p align="center">
  <img src="assets/architecture.png" width="100%" alt="DA-WAM architecture: scene encoding, trajectory proposals, candidate-conditioned future prediction, trajectory scoring, and training-only EMA supervision">
</p>
<p align="center"><em>DA-WAM architecture. Shared scene tokens support candidate-specific prediction and scoring; dense future supervision is applied to the expert-matched candidate during training.</em></p>


## Results

### Retained models

| Benchmark | Model / inference setting | Metric | Result |
|---|---|---|---:|
| NAVSIM v1 | SafetyFinal epoch 1; learned candidate scores | PDMS ↑ | **93.6806** | 
| NAVSIM v2.2 | Same v1-trained model; temporal reranking | EPDMS ↑ | **89.2193** |
| Bench2Drive | Four-camera checkpoint, epoch 11 | Driving Score ↑ | **59.04** |

NAVSIM v2 uses training-free `momentum` reranking: `0.875 × learned_score + 0.125 × pair_EC`. NAVSIM v1 PDMS and v2 EPDMS follow different protocols and should not be compared directly.

The tables and figures below are taken from the latest results. Selected baseline rows preserve that manuscript's values and rounding; they are not a continuously updated leaderboard. The linked arXiv v2 is an earlier version of the manuscript.

### NAVSIM v1

Selected rows from the paper's NAVSIM-v1 comparison. All metrics are higher-is-better, on a 0–100 scale.

| Method | NC | DAC | TTC | Comfort | EP | PDMS |
|---|---:|---:|---:|---:|---:|---:|
| PDM-Closed | 94.6 | 99.8 | 89.9 | 86.9 | 99.9 | 89.1 |
| Human driver | 100.0 | 100.0 | 100.0 | 99.9 | 87.5 | 94.8 |
| DriveVLA-W0 | 98.7 | 99.1 | 95.3 | 99.3 | 83.3 | 90.2 |
| ReCogDrive | 97.9 | 97.3 | 94.9 | 100.0 | 87.3 | 90.8 |
| iPad | 98.6 | 98.3 | 94.9 | 100.0 | 88.0 | 91.7 |
| SparseDriveV2 | 98.5 | 98.4 | 95.0 | 99.9 | 88.6 | 92.0 |
| DrivoR | 98.9 | 98.3 | 96.2 | 100.0 | 89.1 | 93.1 |
| DriveSuprim | 98.6 | 98.6 | 95.5 | 100.0 | 91.3 | 93.5 |
| **DA-WAM** | **99.1** | **98.9** | **96.8** | **99.8** | **90.0** | **93.7** |

### NAVSIM v2

Selected rows from the paper's NAVSIM-v2 comparison. All metrics are higher-is-better, on a 0–100 scale.

| Method | Backbone | NC | DAC | DDC | TLC | EP | TTC | LK | HC | EC | EPDMS |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TransFuser | ResNet-34 | 96.9 | 89.9 | 97.8 | 99.7 | 87.1 | 95.4 | 92.7 | 98.3 | 87.2 | 76.7 |
| Hydra-MDP++ | ViT-L | 98.4 | 98.0 | 99.4 | 99.8 | 87.5 | 97.7 | 95.3 | 98.3 | 77.4 | 85.1 |
| DriveSuprim | ViT-L | 97.8 | 97.9 | 99.5 | 99.9 | 90.6 | 97.1 | 96.6 | 98.3 | 77.9 | 86.0 |
| SparseDriveV2 | ResNet-34 | 98.1 | 98.1 | 99.6 | 99.8 | 91.1 | 97.3 | 96.9 | 98.2 | 78.4 | 86.7 |
| DiffusionDriveV2 | ResNet-34 | 97.7 | 96.6 | 99.2 | 99.8 | 88.9 | 97.2 | 96.0 | 97.8 | 91.0 | 87.5 |
| **DA-WAM** | **ViT-L** | **99.2** | **98.8** | **97.8** | **99.5** | **91.1** | **98.7** | **93.3** | **96.7** | **77.1** | **89.2** |

<details>
<summary>Metric abbreviations</summary>

NC: No-at-Fault Collision; DAC: Drivable Area Compliance; TTC: Time to Collision; EP: Ego Progress; DDC: Driving Direction Compliance; TLC: Traffic Light Compliance; LK: Lane Keeping; HC: History Comfort; EC: Extended Comfort. PDMS and EPDMS are the respective aggregate benchmark scores.

</details>

### Bench2Drive

Selected rows from the paper's closed-loop comparison. “—” denotes values not reported in the supplied table.

| Method | Avg. L2 (m) ↓ | Efficiency ↑ | Comfortness ↑ | Success Rate (%) ↑ | Driving Score ↑ |
|---|---:|---:|---:|---:|---:|
| TCP-traj w/o distillation | 1.96 | 78.78 | 22.96 | 20.45 | 49.30 |
| UniAD-Base | 0.73 | 129.21 | 43.58 | 16.36 | 45.81 |
| VAD | 0.91 | 157.94 | 46.01 | 15.00 | 42.35 |
| EgoFSD-S | 0.70 | 178.30 | — | 21.00 | 52.02 |
| BridgeAD | 0.71 | — | — | 22.73 | 50.06 |
| SeerDrive | 0.66 | — | — | 30.17 | 58.32 |
| **DA-WAM** | **0.95** | **140.40** | **14.64** | **24.66** | **59.04** |

### Ablation: future conditioning and safety supervision

| Configuration | Stage 2 safety supervision | PDMS ↑ | NC ↑ | DAC ↑ | EP ↑ | TTC ↑ | Comfort ↑ |
|---|:---:|---:|---:|---:|---:|---:|---:|
| No future prediction | No | 93.31 | 98.45 | 98.27 | 91.36 | 95.48 | 99.99 |
| Shared global future | No | 92.81 | 99.02 | 98.46 | 88.68 | 96.54 | 99.99 |
| Current-latent conditioning | No | 93.25 | 98.44 | 98.19 | 91.38 | 95.49 | 99.94 |
| Action-conditioned future | No | 93.46 | 98.88 | 98.58 | 90.47 | 96.33 | 99.69 |
| **Action-conditioned future** | **Yes** | **93.68** | **99.11** | **98.88** | **89.97** | **96.81** | **99.77** |

<p align="center">
  <img src="assets/posttraining_curve.png" width="100%" alt="Safety and progress metrics across post-training checkpoints, with the retained checkpoint circled">
</p>

Safety-focused refinement improves candidate ranking with a small progress trade-off. In this figure, **epoch 0 is Stage 1**, and epochs 1–3 count completed refinement epochs. The retained model after two completed refinement epochs is saved as **checkpoint epoch 1** under zero-based numbering.

## Visualization

### Safety-focused trajectory selection

<p align="center">
  <img src="assets/safety_refinement.png" width="100%" alt="Traffic-signal and vehicle attention examples alongside Stage 1 and Stage 2 trajectories">
</p>

The paper's examples compare attention responses and selected trajectories before and after refinement. Image-space attention is a diagnostic visualization, not BEV localization or proof of causal influence.

### Future-feature prediction

<p align="center">
  <img src="assets/future_latents.png" width="100%" alt="Truck-scene visualization comparing current observations, observed future frames, predicted future features and EMA target features">
</p>

This figure uses a **separately trained 1.5-second visualization checkpoint**. The retained quantitative NAVSIM model uses a **0.5-second future target**; the visualization checkpoint is not included in the minimal release. Each column is an independent prediction. Feature colors denote a shared PC1 projection, not object identities or semantic classes.

## Getting Started

### Installation

See [environment setup](environment/README.md). Model training/inference uses Python 3.9; the CARLA client uses a separate Python 3.8 environment. OpenScene data, maps, metric caches, Bench2Drive training caches, and CARLA 0.9.15 remain external prerequisites.

```bash
cd /dahuafs/userdata/2639639/Code/leap-auto-wam
source environment/activate_model.sh
```

### Training

#### Prepare the NAVSIM cache (required before Stage 1)

Stage 1 reads `da_wam_feature.gz` and `da_wam_target.gz` from the NAVSIM train/validation cache. The target cache must include `future_camera_features`; the default NAVSIM cache does not enable this target. Generate a dedicated cache first:

```bash
# Uses OPENSCENE_DATA_ROOT and NAVSIM_EXP_ROOT defaults documented in environment/README.md
bash scripts/cache_navsim_v1.sh
```

The script calls `navsim_v1/navsim/planning/script/run_dataset_caching.py` with the DA-WAM agent configuration, `agent.cache_data=true`, and `agent.config.posttraj_future_enabled=true`. It caches offsets `[1]` by default: the next NAVSIM camera frame after the history (nominally 0.5 seconds at 2 Hz). The target builder reads `cam_f0`, crops and resizes the image to 512×256, and stores a float tensor shaped `[1, 3, 256, 512]` plus `future_camera_offset_order`. This is a cached image target, not a precomputed V-JEPA embedding.

The new default cache directory is `${NAVSIM_EXP_ROOT}/train_da_wam_future_cache`, shared by the cache and Stage 1 scripts. OpenScene `navsim_logs/trainval`, `sensor_blobs/trainval`, and the nuPlan maps must exist first (`NUPLAN_MAPS_ROOT` defaults to `${OPENSCENE_DATA_ROOT}/maps`). Override paths or worker count with `OPENSCENE_DATA_ROOT`, `NAVSIM_EXP_ROOT`, `CACHE_PATH`, `NUPLAN_MAPS_ROOT`, and `CACHE_WORKERS`. If deliberately reusing an older cache directory, run `FORCE_CACHE_COMPUTATION=true bash scripts/cache_navsim_v1.sh` so existing target files without `future_camera_features` are rebuilt.

To validate Hydra configuration without computing the cache, run `DRY_RUN=true bash scripts/cache_navsim_v1.sh`.

Cache files are named `da_wam_feature.gz` and `da_wam_target.gz` across NAVSIM and the Bench2Drive cache builder. NAVSIM metric/anchor caches default to `${NAVSIM_EXP_ROOT}/da_wam_cache`; use `DA_WAM_METRIC_CACHE_ROOT` to point to an existing metric/anchor cache root, or `METRIC_CACHE_PATH` for an evaluation cache. Existing feature/target caches with the older filenames must be regenerated or have their files renamed before use. Bench2Drive caches use the `da_wam_` prefix; rebuild its cache bundle after this source change because its validity checks include source hashes. These naming changes do not alter checkpoint tensors.

##### Choosing future offsets and the number of frames

Set `POSTTRAJ_FUTURE_FRAME_OFFSETS` in **both** the cache and training commands. An offset is a frame index relative to the last history frame, not a duration in seconds. With `H` offsets, each cached `future_camera_features` tensor has shape `[H, 3, 256, 512]`; batching adds a leading batch dimension. `future_camera_offset_order` records the index order.

| Offset list | Future frames (H) | Nominal target times at 2 Hz |
| --- | ---: | --- |
| `[1]` (default) | 1 | 0.5 s |
| `[1,2]` | 2 | 0.5, 1.0 s |
| `[1,2,3]` | 3 | 0.5, 1.0, 1.5 s |
| `[1,2,3,4]` | 4 | 0.5, 1.0, 1.5, 2.0 s |

The released training implementation requires **contiguous offsets starting at 1, with at most four frames**. Lists such as `[2]`, `[1,3]`, or `[1,2,3,4,5]` are unsupported; the cache launcher rejects them before computation. Multi-horizon training uses compact mode (`POSTTRAJ_FULL_CANDIDATE_LATENT=false`, the default). Full-candidate latent supervision supports only `[1]`. The EMA teacher uses adjacent frame pairs, and the future loss is averaged across horizons with the total future-loss weight unchanged.

For example, generate a three-frame cache and train with all three frames:

```bash
export CACHE_PATH=/path/to/navsim_exp/train_da_wam_future_h3_cache
export POSTTRAJ_FUTURE_FRAME_OFFSETS='[1,2,3]'
bash scripts/cache_navsim_v1.sh
POSTTRAJ_FULL_CANDIDATE_LATENT=false bash scripts/train_navsim.sh stage1
```

A cache can also contain a **superset** of the training offsets: `select_future_camera_features` selects the requested frames using `future_camera_offset_order`. For example, cache four frames once, then choose one, two, three, or four for a training run:

```bash
export CACHE_PATH=/path/to/navsim_exp/train_da_wam_future_h4_cache
POSTTRAJ_FUTURE_FRAME_OFFSETS='[1,2,3,4]' bash scripts/cache_navsim_v1.sh

# Choose [1], [1,2], [1,2,3], or [1,2,3,4] for this run.
POSTTRAJ_FUTURE_FRAME_OFFSETS='[1,2]' \
POSTTRAJ_FULL_CANDIDATE_LATENT=false \
bash scripts/train_navsim.sh stage1
```

Every requested training offset must exist in the cache; a one-frame cache cannot serve a three-frame run. Use separate cache directories for different cached offset lists. If changing the list in an existing directory, set `FORCE_CACHE_COMPUTATION=true` to rebuild it; changing the environment variable alone does not invalidate existing files. More cached frames increase storage and preprocessing cost. The launcher's final check inspects one target file, not the completeness of the entire cache.

Times in the table are nominal: the builder selects by frame index and does not verify timestamps. Check actual timestamp gaps for strict time-based experiments, particularly when frames are missing.

#### Train models

```bash
# NAVSIM Stage 1: Joint30 (uses the cache created above)
bash scripts/train_navsim.sh stage1

# NAVSIM Stage 2: SafetyFinal; supply the Stage 1 parent checkpoint
CHECKPOINT_PATH=/path/to/stage1_epoch15.ckpt bash scripts/train_navsim.sh stage2

# Bench2Drive: retained four-camera implementation
bash scripts/train_bench2drive.sh
```

The NAVSIM recipe uses 32 proposals and LoRA rank 32. Stage 2 starts from Joint30 epoch 15, freezes proposal generation, and retains SafetyFinal epoch 1. The Stage 1 parent is not bundled with the final-only checkpoints. The NAVSIM v2 result uses the v1-trained model without v2-specific training. Training launchers default to multiple GPUs and require prepared caches.

### Inference and evaluation

```bash
# Configuration checks only
bash scripts/eval_navsim_v1.sh --cfg job
bash scripts/eval_navsim_v2.sh --cfg job
python scripts/prepare_bench2drive_eval.py --check

# Evaluation: each output directory must be new
OUTPUT_DIR=/path/to/new/v1_result bash scripts/eval_navsim_v1.sh
OUTPUT_DIR=/path/to/new/v2_result bash scripts/eval_navsim_v2.sh
python scripts/prepare_bench2drive_eval.py --output /path/to/new/b2d_result
bash /path/to/new/b2d_result/run.sh
```

Override NAVSIM paths with `CHECKPOINT_PATH`, `MODEL_ENV`, `METRIC_CACHE_PATH`, `OPENSCENE_DATA_ROOT`, and `NAVSIM_EXP_ROOT`. Use `B2D_EVAL_ENV` and `CARLA_ROOT` for the CARLA environment. Configuration/import checks do not establish successful full benchmark reproduction.

### Checkpoints

Download the final NAVSIM and Bench2Drive checkpoints and required V-JEPA 2.1 backbone from the [Hugging Face model repository](https://huggingface.co/Ruiguo1/DA-WAM). Verify the files against `checkpoints/final_sha256.json`.
<!-- | File | Purpose |
|---|---|
| `checkpoints/navsim_final.ckpt` | SafetyFinal epoch 1; shared by NAVSIM v1 and v2 |
| `checkpoints/bench2drive_4cam_score11.ckpt` | Final Bench2Drive four-camera epoch 11 |
| `checkpoints/vjepa2_1_vitl.pt` | Required V-JEPA 2.1 pretrained backbone |

These files are retained **locally**; this README does not provide a public checkpoint download. See the [weight manifest](checkpoints/manifest.json) and [SHA256 checksums](checkpoints/final_sha256.json). -->

<details>
<summary><strong>Repository layout</strong></summary>

```text
navsim_v1/       Joint30 / SafetyFinal training and NAVSIM v1 inference
navsim_v2/       Official NAVSIM v2.2 inference and temporal reranking
bench2drive/    Four-camera training, inference and required shared modules
checkpoints/    Final models and required pretrained backbone only
third_party/    V-JEPA runtime dependencies
scripts/        Unified training and inference entrypoints
environment/    Installation and environment configuration
assets/         Paper figures used in this README
```

Only core training/inference code, required dependencies, final weights and README figures are retained here. Historical experiments, raw result archives and editable paper materials remain outside this directory in the sibling backup.

</details>

## Citation

```bibtex
@article{zhong2026dawam,
  title={DA-WAM: Decision-Aligned Future Latents for Driving World Models},
  author={Zhong, Ruiguo and Ma, Benshan and Chen, Xiaolong and Zhang, Lang
          and Feng, Mingyue and Wang, Yaonong and Liu, Pei and Ma, Jun},
  journal={arXiv preprint arXiv:2608.19085},
  year={2026},
  eprint={2608.19085},
  archivePrefix={arXiv},
  primaryClass={cs.RO},
  url={https://arxiv.org/abs/2608.19085}
}
```

## Acknowledgements

We thank the authors and maintainers of the open-source codebases that informed this work: [Drive-JEPA](https://github.com/linhanwang/Drive-JEPA), [SparseDriveV2](https://github.com/swc-17/SparseDriveV2), [DiffusionDrive](https://github.com/hustvl/DiffusionDrive), [WA-JEPA](https://github.com/AFARI-Research/WA-JEPA), and [iPad](https://github.com/Kguo-cs/iPad). We also build on [V-JEPA 2](https://github.com/facebookresearch/vjepa2) and use the [NAVSIM](https://github.com/autonomousvision/navsim), [nuPlan](https://github.com/motional/nuplan-devkit), and [Bench2Drive](https://github.com/Thinklab-SJTU/Bench2Drive) benchmarks and tools. 

## License

See [LICENSE](LICENSE). Bundled third-party components, datasets and pretrained weights remain subject to their respective licenses.
