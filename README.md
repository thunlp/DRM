# DRM

Official code release for **DRM**.

Paper: TODO: paste arXiv or project page link here.

Checkpoints: [Teburile/DRM](https://huggingface.co/Teburile/DRM)

## Overview

DRM uses RewardDiT models to generate reward values from FsfairX-LLaMA3-RM-v0.1 text embeddings.

This repository contains:

- data preparation scripts for ArmoRM and Tulu3 preference data
- RewardDiT model code
- training scripts for the multi-objective and preference RewardDiT models
- a lightweight `ScoreGenerator` for inference with the released checkpoints

The public inference code does not use gate models or reward debiasing transforms.

## Released Checkpoints

The checkpoints are hosted on Hugging Face:

```text
https://huggingface.co/Teburile/DRM
```

| Name | File on Hugging Face | Reward Dim | Description |
| --- | --- | ---: | --- |
| DRM-Multi-8B | `DRM-Multi-8B/model.pth` | 19 | RewardDiT trained on ArmoRM multi-objective labels |
| DRM-Pref-8B | `DRM-Pref-8B/model.pth` | 1 | RewardDiT trained on Tulu3 pair-preference data |

Recommended inference settings for both checkpoints:

```text
mask_split = false
num_steps = 10
guidance_scale = 7.0
num_samples = 32
gate = off
debias = off
```

These are the defaults in `score_generator.py`.

## Environment

The code was checked with:

```text
Python 3.11
torch 2.7.0+cu126
diffusers 0.36.0
transformers 5.8.0
datasets 4.8.5
numpy 2.4.3
tqdm 4.67.3
```

Install the minimal dependencies:

```bash
pip install -r requirements.txt
```

FlashAttention is optional. If installed, the data preparation scripts will use it for faster FsfairX encoder inference.

## Download Checkpoints

Install the Hugging Face Hub CLI or use `huggingface_hub` directly. One simple option is:

```bash
hf download Teburile/DRM DRM-Multi-8B/model.pth --local-dir checkpoints
hf download Teburile/DRM DRM-Pref-8B/model.pth --local-dir checkpoints
```

This creates:

```text
checkpoints/DRM-Multi-8B/model.pth
checkpoints/DRM-Pref-8B/model.pth
```

## Inference

Run DRM-Multi-8B:

```bash
python score_generator.py \
  --ckpt checkpoints/DRM-Multi-8B/model.pth \
  --prompt "What is photosynthesis?" \
  --response "Photosynthesis is the process by which plants convert light into chemical energy."
```

Run DRM-Pref-8B:

```bash
python score_generator.py \
  --ckpt checkpoints/DRM-Pref-8B/model.pth \
  --prompt "What is photosynthesis?" \
  --response "Photosynthesis is the process by which plants convert light into chemical energy."
```

The command prints a list containing one scalar score.

## Data Preparation

Prepare ArmoRM multi-objective data:

```bash
python prepare_armorm_data.py \
  --output_root data/armo_dataset \
  --limit 0 \
  --max_length 4096
```

Prepare Tulu3 pair-preference data:

```bash
python prepare_tulu3_pair_data.py \
  --output_root data/armo_dataset \
  --dataset_split all \
  --limit 0 \
  --max_length 4096
```

Use a small positive `--limit` such as `1000` for debugging.

The scripts default to the official Hugging Face endpoint. If a mirror is needed, pass it explicitly:

```bash
--hf_endpoint https://hf-mirror.com
```

## Training

Train the 19-dimensional DRM-Multi RewardDiT:

```bash
python train_dit_armorm.py \
  --data_root data/armo_dataset \
  --output_dir outputs/dit_armorm \
  --limit None \
  --epochs 20 \
  --batch_size 64
```

Train the one-dimensional DRM-Pref RewardDiT:

```bash
python train_dit_tulu3_pair.py \
  --data_root data/armo_dataset \
  --output_dir outputs/dit_tulu3_pair \
  --limit None \
  --epochs 20 \
  --batch_size 64
```

The Tulu3 pair-preference loss is:

```text
denoising loss + bt_alpha * Bradley-Terry loss + reward_reg_weight * reward_l2
```

The default `reward_reg_weight` is `0.001`, matching the released DRM-Pref-8B checkpoint.

## Repository Structure

```text
.
├── dit.py
├── score_generator.py
├── prepare_armorm_data.py
├── prepare_tulu3_pair_data.py
├── train_dit_armorm.py
├── train_dit_tulu3_pair.py
├── configs/
│   ├── DRM-Multi-8B/config.json
│   └── DRM-Pref-8B/config.json
└── requirements.txt
```

## Citation

TODO: paste BibTeX citation here after the paper is available.

```bibtex
@article{TODO,
  title = {TODO},
  author = {TODO},
  journal = {arXiv preprint},
  year = {TODO}
}
```

## Notes

- Checkpoints are not committed to this GitHub repository. They are hosted at [Teburile/DRM](https://huggingface.co/Teburile/DRM).
- All default paths in the code are relative paths.
- The default encoder is `sfairXC/FsfairX-LLaMA3-RM-v0.1`.
- DRM-Multi-8B is a 19-dimensional RewardDiT.
- DRM-Pref-8B is a one-dimensional RewardDiT.
