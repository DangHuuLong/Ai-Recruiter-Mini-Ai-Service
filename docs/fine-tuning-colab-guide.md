# Fine-Tuning Similarity Model on Colab

## 1. Purpose

This guide explains how to run the CV-JD similarity fine-tuning experiment on Google Colab or Kaggle.

The repository stores the training script, dataset snapshot, and experiment instructions. The actual training run should use a GPU runtime outside local VSCode.

## 2. Recommended Environment

Use one of these environments:

- Google Colab with GPU runtime;
- Kaggle Notebook with GPU enabled.

Local VSCode should only be used for quick smoke tests with small sample limits. Do not use local CPU training as the final experiment result.

## 3. Dataset and Baseline

This experiment uses:

```txt
dataset version: v0.2
base model: sentence-transformers/all-MiniLM-L6-v2
```

The baseline result to beat is:

```txt
pairs evaluated: 2275
MAE: 17.6431
RMSE: 22.4497
label accuracy: 0.2822
```

A fine-tuned model should only be considered useful if it improves on the stable `v0.2` validation/test evaluation.

## 4. Colab Setup

Start a new Colab notebook and enable GPU:

```txt
Runtime -> Change runtime type -> Hardware accelerator -> GPU
```

Clone the repository and checkout the experiment branch:

```bash
!git clone https://github.com/DangHuuLong/Ai-Recruiter-Mini-Ai-Service.git
%cd Ai-Recruiter-Mini-Ai-Service
!git checkout experiment/fine-tune-similarity-colab-v0.2
```

Install dependencies:

```bash
!pip install -r requirements.txt
```

If Colab already has a different Torch version, restart the runtime after installation if imports fail.

## 5. Validate Dataset Before Training

Run validation before training:

```bash
!python training/validate_dataset.py --dataset-root datasets --version v0.2
!python training/check_fine_tune_readiness.py --dataset-root datasets --version v0.2
```

Both commands should pass before fine-tuning starts.

## 6. Run a Quick Smoke Test

Before the full GPU run, test that the script and dataset load correctly:

```bash
!python training/fine_tune_similarity.py \
  --dataset-root datasets \
  --version v0.2 \
  --base-model sentence-transformers/all-MiniLM-L6-v2 \
  --output-dir artifacts/models/debug-miniLM-v0.2 \
  --report-path artifacts/reports/debug_fine_tune_similarity_v0.2_report.json \
  --epochs 1 \
  --batch-size 2 \
  --max-train-samples 20 \
  --max-eval-samples 20
```

This run is only a smoke test. Do not compare its metrics with the baseline.

## 7. Run the Fine-Tuning Experiment

Run the full experiment:

```bash
!python training/fine_tune_similarity.py \
  --dataset-root datasets \
  --version v0.2 \
  --base-model sentence-transformers/all-MiniLM-L6-v2 \
  --output-dir artifacts/models/fine-tuned-miniLM-v0.2 \
  --report-path artifacts/reports/fine_tune_similarity_v0.2_report.json \
  --epochs 1 \
  --batch-size 8
```

If GPU memory is limited, reduce the batch size:

```bash
--batch-size 4
```

If the run is stable, try more epochs in a separate experiment:

```bash
--epochs 2
```

Do not choose the final model using the test split repeatedly. Use validation metrics for experiment decisions and keep the test split for final comparison.

## 8. Save Artifacts Outside Git

The repository ignores `artifacts/` by default.

After training, copy artifacts to Google Drive or Kaggle output storage:

```txt
artifacts/models/fine-tuned-miniLM-v0.2/
artifacts/reports/fine_tune_similarity_v0.2_report.json
```

Recommended external storage structure:

```txt
ai-recruiter-artifacts/
  fine-tuned-miniLM-v0.2/
  fine_tune_similarity_v0.2_report.json
```

Do not commit model checkpoints, exported model files, or embedding caches to the repository.

## 9. Result Review Checklist

After the run finishes, review:

- validation MAE and RMSE;
- test MAE and RMSE;
- validation label accuracy;
- test label accuracy;
- whether the fine-tuned model beats the baseline;
- whether the model overfits validation but fails on test.

Baseline to beat:

```txt
MAE < 17.6431
RMSE < 22.4497
label accuracy > 0.2822
```

If the model does not beat the baseline, keep the result as an experiment note but do not integrate it into service inference.

## 10. Next Step After a Successful Run

If the fine-tuned model beats the baseline, create a separate integration branch later.

The integration branch should focus on:

- loading the exported model;
- adding inference configuration;
- comparing deterministic scoring vs model-assisted scoring;
- keeping fallback behavior if the model artifact is missing.
