# ID System

A plant species identifier built on a fine-tuned MobileNetV3 (small and large variants), trained with early stopping to avoid overfitting.

## Project layout

```
plants1/                 dataset (metadata.csv + train/valid/test image folders)
processed/                saved model checkpoints (model.pt, model_large.pt)
src/id_system/
  id_system.py             training script, MobileNetV3-Small
  id_system_large.py        training script, MobileNetV3-Large
  early_stopping.py         early stopping helper
  predict.py                run a single prediction against a trained model
  predict_test.py           prediction sanity checks
  datacheck.py               dataset validation helpers
```

## Setup

This project uses [uv](https://github.com/astral-sh/uv) for dependency management.

```bash
uv sync
```

## Usage

Train a model:

```bash
uv run id-system
```

Training resumes automatically from `processed/model.pt` if it already exists.

Run a prediction:

```bash
uv run python src/id_system/predict.py
```

## Dataset

`plants1/metadata.csv` maps each class to a `dir_id`, species name, and common name, along with per-split image counts. Images live under `plants1/<split>/<dir_id>/`.
