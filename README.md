# RAID Detection

**Repository:** [github.com/saanikapatil08/raid-detection](https://github.com/saanikapatil08/raid-detection)

A research codebase for evaluating **machine-generated text detectors** on the [RAID benchmark](https://raid-bench.xyz) (ACL 2024). The project loads RAID data, trains or runs baseline detectors, measures performance on clean and adversarial text, and saves metrics and plots for analysis.

---

## What This Project Is About

**Problem:** Large language models can produce text that is hard to distinguish from human writing. Detectors (classifiers that label text as human vs. AI-generated) need rigorous evaluation across models, domains, and evasion strategies.

**RAID** (Robust AI Detection Benchmark) provides a standardized dataset for that evaluation: millions of text samples from many LLMs, human writers, content domains, decoding settings, and **adversarial attacks** designed to fool detectors.

**This repository** implements a small but complete experimental workflow:

1. **Download** RAID `train.csv` / `test.csv`
2. **Explore** the data in Jupyter notebooks
3. **Run detectors** — a fast supervised TF-IDF baseline and a zero-shot RoBERTa classifier
4. **Evaluate** using RAID-aligned metrics (especially **TPR @ 5% FPR**)
5. **Report** JSON results and visualization plots under `results/`

The goal is not to build a production API, but to **reproduce and extend baseline detection experiments** on RAID in a reproducible, script-driven way.

---

## What the Code Does

### End-to-end pipeline (`scripts/run_pipeline.py`)

When you run the pipeline, it performs these steps automatically:

```
train.csv  →  load sample  →  train/eval split (80/20)
                                    ↓
                         ┌──────────┴──────────┐
                         │                     │
                   TF-IDF detector      RoBERTa detector
                   (fit on train)        (zero-shot, no fit)
                         │                     │
                         └──────────┬──────────┘
                                    ↓
              Evaluate on CLEAN subset (attack = "none")
              → accuracy, F1, AUC, TPR @ 5% FPR
              → per-domain breakdown
                                    ↓
              Evaluate on FULL eval set (all attack types)
              → per-attack TPR @ 5% FPR, AUC, F1
              → detection rate by source LLM (GPT-4, Llama, etc.)
                                    ↓
              Save JSON + PNG plots to results/
```

### Detectors

| Detector | Module | How it works |
|----------|--------|--------------|
| **TF-IDF** (`StatisticalDetector`) | `src/models/statistical.py` | Converts text to TF-IDF n-gram features (up to 100k terms), trains logistic regression on RAID labels. Fast; learns patterns specific to the loaded training sample. |
| **RoBERTa** (`ZeroShotDetector`) | `src/models/baseline.py` | Uses HuggingFace `roberta-base-openai-detector` without fine-tuning on RAID. Scores each text with a pretrained classifier. Slower; generalizes from pretraining only. |

### Evaluation (`src/evaluation/metrics.py`)

Each detector outputs a **probability** that text is AI-generated (`y_score`). Predictions use threshold 0.5 unless noted.

| Metric | Meaning |
|--------|---------|
| **Accuracy** | Fraction of correct binary predictions |
| **Precision / Recall / F1** | Standard classification metrics (label 1 = AI) |
| **AUC** | Area under the ROC curve (threshold-independent) |
| **TPR @ 5% FPR** | True positive rate when false positive rate ≤ 5% — **primary RAID metric** |

RAID is **heavily imbalanced** (~90% AI, ~10% human in typical samples). High accuracy can be misleading if the model predicts "human" too often. **Prioritize AUC and TPR @ 5% FPR** when comparing detectors.

### Data loading (`src/data/loader.py`)

Key columns in the CSV files:

| Column | Description |
|--------|-------------|
| `generation` | The text to classify |
| `label` | `0` = human, `1` = AI-generated |
| `model` | Source model (`human`, `gpt-4`, `chatgpt`, `llama-chat`, …) |
| `domain` | Content type (`news`, `reddit`, `arxiv`, `wikipedia`, …) |
| `attack` | Adversarial perturbation (`none`, `synonym`, `whitespace`, …) |
| `decoding` | Decoding strategy used for generation |

Helper functions: `filter_no_attack()`, `filter_by_domain()`, `filter_by_model()`.

---

## Dataset

**RAID** is described in [Dugan et al., ACL 2024](https://aclanthology.org/2024.acl-long.866/).

- **Source:** [raid-bench.xyz](https://raid-bench.xyz) | [Hugging Face](https://huggingface.co/datasets/liamdugan/raid) | [GitHub](https://github.com/liamdugan/raid)
- **Scale:** 6M+ generations in the full benchmark
- **Generators:** ChatGPT, GPT-4, GPT-3, GPT-2, Llama 2 70B, Cohere, MPT-30B, Mistral 7B, and human text
- **Domains:** News, Reddit, arXiv abstracts, books, Wikipedia, poetry, recipes, IMDb reviews
- **Attacks:** 11 adversarial types (paraphrase, misspelling, whitespace, etc.)
- **Local files:** `data/raw/train.csv` (~11 GB), `data/raw/test.csv` (~1.1 GB)

The pipeline uses **`train.csv` only** and holds out an internal eval split. The official `test.csv` is available for future held-out evaluation (labels may be structured differently).

---

## Project Structure

```
raid-detection/
├── data/
│   ├── raw/                  # train.csv, test.csv (after download)
│   └── processed/            # Reserved for preprocessed features
├── notebooks/
│   ├── 01_eda.ipynb          # Dataset exploration and visualizations
│   └── 02_baseline.ipynb     # Interactive detector experiments
├── src/
│   ├── data/loader.py        # load_raid(), filters
│   ├── models/
│   │   ├── statistical.py    # StatisticalDetector (TF-IDF)
│   │   └── baseline.py       # ZeroShotDetector (RoBERTa)
│   ├── evaluation/metrics.py # compute_metrics(), tpr_at_fpr()
│   └── utils/io.py           # save_results(), JSON helpers
├── scripts/
│   ├── download_data.py      # Fetch CSVs from raid-bench.xyz
│   └── run_pipeline.py       # Full train → eval → plots workflow
├── results/                  # Output JSON and PNG figures
├── experiments/              # Experiment configs (optional)
├── requirements.txt
└── setup.py
```

---

## Prerequisites

- **Python 3.9+**
- **~15 GB disk space** for both CSV files
- **8+ GB RAM** recommended for 100k-row pipeline runs
- **Apple Silicon / NVIDIA GPU** optional but speeds up RoBERTa (uses MPS or CUDA via PyTorch)

---

## Installation (Step by Step)

All commands assume you are in the repository root.

### Step 1: Clone the repository

```bash
git clone https://github.com/saanikapatil08/raid-detection.git
cd raid-detection
```

### Step 2: Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

Your shell prompt should show `(.venv)`.

### Step 3: Install Python dependencies

```bash
pip install -r requirements.txt
pip install -e .
```

The editable install (`-e .`) lets you import `src` from anywhere after activation.

**Core packages:** pandas, scikit-learn, transformers, torch, matplotlib, jupyter, requests, tqdm.

### Step 4: Download the dataset

```bash
python scripts/download_data.py
```

Expected output:

```
Output directory: .../raid-detection/data/raw
  [skip] train.csv already exists. Use --force to re-download.
  ...
All files downloaded successfully.
```

First download may take a long time (multi-GB files). Options:

```bash
python scripts/download_data.py --output-dir data/raw   # custom path
python scripts/download_data.py --force                 # re-download
python scripts/download_data.py --hf                    # via Hugging Face datasets
```

### Step 5: Verify the setup

```bash
python -c "from src.data.loader import load_raid; t,_=load_raid('data/raw', train_nrows=5); print('OK:', len(t), 'rows')"
```

You should see `OK: 5 rows` with no import errors.

---

## Running the Pipeline (Detailed)

The main entry point is `scripts/run_pipeline.py`. **Run one command per line** in the terminal (do not paste comment lines starting with `#`).

### Recommended first run (fast)

Uses 5,000 training rows — completes in roughly 10–15 minutes on a typical laptop:

```bash
python scripts/run_pipeline.py --model tfidf --train-sample 5000 --eval-sample 500
```

### Standard TF-IDF run

Matches what was used for strong baseline numbers on RAID samples:

```bash
python scripts/run_pipeline.py --model tfidf --train-sample 100000 --eval-sample 5000
```

Typical results on clean (no-attack) eval: **~99% AUC**, **~99% TPR @ 5% FPR**.

### RoBERTa zero-shot run

Downloads model weights on first run (~500 MB). Much slower than TF-IDF:

```bash
# Start small
python scripts/run_pipeline.py --model roberta --train-sample 10000 --eval-sample 1000

# Larger run (30+ minutes on Apple M-series)
python scripts/run_pipeline.py --model roberta --train-sample 50000 --eval-sample 5000
```

RoBERTa does **not** use the training split for fitting — only for loading data and creating the eval split. The `train-sample` flag still controls how many rows are read from disk.

### Compare both detectors

```bash
python scripts/run_pipeline.py --model all --train-sample 100000 --eval-sample 5000
```

Produces a `model_comparison.png` bar chart in addition to per-model outputs.

### All CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `tfidf` | `tfidf`, `roberta`, or `all` |
| `--train-sample` | `100000` | Max rows read from `train.csv` (controls memory and runtime) |
| `--eval-sample` | all clean rows | Cap on clean-eval rows (`attack=none`); speeds up scoring |
| `--data-dir` | `data/raw` | Directory with `train.csv` |
| `--output-dir` | `results` | Where JSON and PNG files are written |
| `--test-size` | `0.2` | Fraction of loaded data used for evaluation (stratified by label) |

### What you see while it runs

```
Loading up to 100,000 rows from data/raw/train.csv ...
  Loaded 100,000 rows  |  label balance: {1: 91953, 0: 8047}
  Train split: 80,000  |  Eval split: 20,000

============================================================
  Model: tfidf
============================================================
  Fitting on train split ...
  Fitting done.
  Clean eval rows: 4,693
  Overall (clean):
    accuracy                  0.9874
    ...
  Running adversarial eval ...
  ...
Results saved to results/pipeline_tfidf_20260525_084444.json
  Saved results/roc_tfidf.png
  ...
Pipeline complete.
```

### Output files (`results/`)

| File | Contents |
|------|----------|
| `pipeline_<model>_<timestamp>.json` | Overall metrics, per-domain and per-attack tables, sample counts |
| `roc_<model>.png` | ROC curve with TPR @ 5% FPR marked |
| `adversarial_tpr_<model>.png` | Bar chart: detection rate by attack type |
| `detection_by_model_<model>.png` | Bar chart: detection rate by source LLM |
| `model_comparison.png` | TF-IDF vs RoBERTa (only with `--model all`) |

Example JSON structure:

```json
{
  "experiment": "pipeline_tfidf",
  "metrics": { "accuracy": 0.987, "auc": 0.9996, "tpr_at_5pct_fpr": 0.9989 },
  "per_attack": [
    { "attack": "none", "tpr_at_5pct_fpr": 0.999, "n": 4693 },
    { "attack": "synonym", "tpr_at_5pct_fpr": 0.995, "n": 4541 }
  ]
}
```

---

## Running Jupyter Notebooks

Interactive exploration and experiments:

```bash
jupyter notebook notebooks/01_eda.ipynb
```

```bash
jupyter notebook notebooks/02_baseline.ipynb
```

| Notebook | Purpose |
|----------|---------|
| `01_eda.ipynb` | Label balance, domain/model distributions, text length stats |
| `02_baseline.ipynb` | Train and evaluate detectors interactively |

Jupyter prints a URL with a token, e.g. `http://localhost:8888/tree?token=...` — open it in your browser.

If you see **"Notebook is not trusted"**:

```bash
jupyter trust notebooks/01_eda.ipynb notebooks/02_baseline.ipynb
```

To stop the server: press `Ctrl+C` in the terminal, then type `y`.

---

## Using the Python API Directly

For custom experiments outside the pipeline script:

```python
from src.data.loader import load_raid, filter_no_attack
from src.models.statistical import StatisticalDetector
from src.models.baseline import ZeroShotDetector
from src.evaluation.metrics import compute_metrics

# Load data (always use nrows while prototyping — train.csv is 11 GB)
train_df, test_df = load_raid("data/raw", train_nrows=10_000, test_nrows=1_000)

# --- TF-IDF: supervised baseline ---
detector = StatisticalDetector()
detector.fit(
    train_df["generation"].tolist(),
    train_df["label"].tolist(),
)
scores = detector.predict_proba(test_df["generation"].tolist())
preds = (scores >= 0.5).astype(int)

metrics = compute_metrics(
    test_df["label"].tolist(),
    preds,
    scores,
    target_fpr=0.05,
)
print(metrics)

# --- RoBERTa: zero-shot (no fit required) ---
zs = ZeroShotDetector(
    model_name="roberta-base-openai-detector",
    batch_size=32,   # lower if you run out of GPU memory
)
clean = filter_no_attack(test_df)
scores = zs.predict_proba(clean["generation"].tolist()[:50])
```

---

## Typical Workflow Summary

| Goal | Command |
|------|---------|
| First-time setup | `pip install -r requirements.txt && pip install -e .` |
| Get data | `python scripts/download_data.py` |
| Quick sanity check | `python scripts/run_pipeline.py --model tfidf --train-sample 5000 --eval-sample 500` |
| Full TF-IDF benchmark | `python scripts/run_pipeline.py --model tfidf --train-sample 100000 --eval-sample 5000` |
| RoBERTa benchmark | `python scripts/run_pipeline.py --model roberta --train-sample 10000 --eval-sample 1000` |
| Explore data | `jupyter notebook notebooks/01_eda.ipynb` |

---

## Troubleshooting

| Issue | Cause | Fix |
|-------|-------|-----|
| `zsh: parse error near ')'` | Pasted markdown comments with parentheses | Run only the `python scripts/...` line, not `#` comment blocks |
| `zsh: command not found: #` | Shell tried to execute `# or` as a command | Same as above — one command per line |
| Pipeline very slow on load | Reading from 11 GB `train.csv` | Lower `--train-sample` |
| RoBERTa interrupted / hung | Large eval set, transformer inference | Use `--eval-sample 1000`; let first run finish (downloads weights) |
| `NotOpenSSLWarning` (urllib3) | macOS system Python + LibreSSL | Harmless; safe to ignore |
| Out of memory | Too many rows or large batch size | Reduce `--train-sample`, `--eval-sample`, or RoBERTa `batch_size` in code |
| `Call fit() before predict_proba()` | TF-IDF used before training | Call `detector.fit(...)` first |

---

## Citation

If you use RAID or this benchmark setup, cite the RAID paper:

```bibtex
@inproceedings{dugan2024raid,
  title     = {RAID: A Shared Benchmark for Robust Evaluation of Machine-Generated Text Detectors},
  author    = {Dugan, Liam and Hwang, Alyssa and Trischler, Adam and Bansal, Mohit and Callison-Burch, Chris},
  booktitle = {Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (ACL 2024)},
  year      = {2024}
}
```
