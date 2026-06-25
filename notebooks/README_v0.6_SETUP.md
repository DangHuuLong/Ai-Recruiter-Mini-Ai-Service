# Cross-Encoder v0.6 Fine-Tuning Setup

## 📋 Status

✅ **Preparation Complete**

- ✅ Dataset v0.5 created: 13,350 pairs (9,350 train / 2,000 val / 2,000 test)
- ✅ Dataset balanced: 20% per class (poor/weak/moderate/strong/excellent)
- ✅ Cross-encoder JSONL ready: `datasets/versions/v0.5/cross_encoder/`
- ✅ Colab notebook ready: `notebooks/fine_tune_cross_encoder_v0_6_colab.ipynb`

## 🚀 How to Run

### Option 1: Google Colab (Recommended)

1. **Open Colab notebook**:
   ```
   notebooks/fine_tune_cross_encoder_v0_6_colab.ipynb
   ```

2. **Upload to Google Colab**:
   - Go to [colab.research.google.com](https://colab.research.google.com)
   - File → Open notebook
   - Upload tab → Select the notebook file
   - Or open from GitHub

3. **Mount Google Drive** (see first cell):
   ```python
   from google.colab import drive
   drive.mount('/content/gdrive')
   ```

4. **Clone/sync repository**:
   - The notebook will clone the repo to your Drive
   - Or manually copy `datasets/versions/v0.5/cross_encoder/` to Drive

5. **Run cells in order**:
   - Setup (dependencies, mount, clone)
   - Helper functions
   - Load dataset
   - Run 1, 2, 3 (sequential or parallel)
   - Summary and report

### Option 2: Local Python (GPU required)

```bash
# Install dependencies
pip install torch sentence-transformers>=3.0.0,<3.3.0 scikit-learn

# Run training programmatically
python -c "
from notebooks.fine_tune_cross_encoder_v0_6_colab import *
# ... (extract training code from notebook cells)
"
```

## 📊 Experiment Runs

### Run 1: MSELoss + Spearman (Baseline)
- **Goal**: Establish baseline on balanced v0.5 data
- **Expected**: Test LabelAcc **> 60.76%** ✓ (ceiling broken)
- **Time**: ~20 min on T4 GPU

### Run 2: BoundaryAwareLoss + Spearman
- **Goal**: Test if boundary loss helps on balanced data
- **Hypothesis**: Marginal improvement (+0.6 pp) if any
- **Time**: ~20 min on T4 GPU

### Run 3: MSELoss + LabelAcc
- **Goal**: Re-test LabelAcc evaluator with 2x validation set (2,000 vs 1,050)
- **Hypothesis**: Larger val set → more stable checkpoint selection
- **Time**: ~20 min on T4 GPU

## 📈 Success Criteria

| Metric | Target | Status |
|--------|--------|--------|
| Run 1 test LabelAcc | > 60.76% | ⏳ TBD |
| Val-test gap | < 2 pp | ⏳ TBD |
| Best run identified | Clear winner | ⏳ TBD |

## 📂 Key Directories

| Path | Contents |
|------|----------|
| `datasets/versions/v0.5/` | Raw JSONL files (JD, resume, pairs) |
| `datasets/versions/v0.5/cross_encoder/` | Training JSONL splits (train/val/test) |
| `artifacts/models/cross-encoder-cv-jd-v0.6-run*/` | Trained model checkpoints (3 runs) |
| `artifacts/reports/` | JSON results for each run + summary |
| `notebooks/fine_tune_cross_encoder_v0_6_colab.ipynb` | Full experiment notebook |

## 🔍 Output Files

After running the notebook, expect:

```
artifacts/
  models/
    cross-encoder-cv-jd-v0.6-run1-mse-spearman/      (best checkpoint)
    cross-encoder-cv-jd-v0.6-run2-boundary-spearman/ (best checkpoint)
    cross-encoder-cv-jd-v0.6-run3-mse-labelacc/      (best checkpoint)
  reports/
    v0.6-run1-mse-spearman_results.json
    v0.6-run2-boundary-spearman_results.json
    v0.6-run3-mse-labelacc_results.json
    v0.6_complete_report.json                        (consolidated)
```

## 💡 Tips

1. **Monitor training**: Watch val_spearman or val_label_acc curves in Colab logs
2. **Early stopping**: Runs auto-save best checkpoint, so stopping after a few epochs is safe
3. **Memory**: If OOM errors, reduce batch_size from 16 → 8
4. **Reproducibility**: Seed=42 is set in split_dataset.py for consistency

## ⚠️ Troubleshooting

### Issue: ModuleNotFoundError during prepare_cross_encoder_data.py
**Solution**: Use `python -m training.prepare_cross_encoder_data` (already done)

### Issue: sentence_transformers version conflict
**Solution**: Pin to `sentence-transformers>=3.0.0,<3.3.0` (FitMixinLoss bug in ≥3.3.0)

### Issue: CUDA out of memory
**Solution**: Reduce batch_size or max_length. Default safe config:
```python
batch_size=16, max_length=512  # Works on T4 GPU (15GB)
```

## 📝 Notes

- All 13,350 pairs use `label_version = rubric_v0.2` (consistent rubric)
- Val set doubled (1,050 → 2,000) for stable checkpoint selection
- Dataset perfectly balanced (20% per class) — no weighted sampling needed
- Expected GPU time: ~60 min total (3 runs × 20 min each)

## 🔗 Related Documentation

- [Plan](../../.claude/plans/tidy-painting-cerf.md) — Full v0.6 strategy
- `docs/similarity-model-cross-encoder-v0.5.md` — Prior v0.5 results
- `training/README.md` — Training pipeline overview
