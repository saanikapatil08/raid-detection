#!/usr/bin/env python
from __future__ import annotations

"""
End-to-end RAID detection pipeline.

Loads data, trains/runs detectors, evaluates on clean and adversarial subsets,
saves results JSON and plots to results/.

Usage examples
--------------
# Fast TF-IDF baseline on 100k train rows:
    python scripts/run_pipeline.py --model tfidf --train-sample 100000

# Zero-shot RoBERTa on 50k rows:
    python scripts/run_pipeline.py --model roberta --train-sample 50000 --eval-sample 5000

# Both models, compare side-by-side:
    python scripts/run_pipeline.py --model all --train-sample 100000 --eval-sample 5000
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for script mode
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split

# Make src importable when run from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.loader import load_raid, filter_no_attack
from src.evaluation.metrics import compute_metrics, tpr_at_fpr, per_domain_metrics
from src.utils.io import save_results, save_json


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _eval_detector(detector, eval_df: pd.DataFrame) -> pd.DataFrame:
    """Run detector on eval_df and return a copy with y_score / y_pred columns."""
    df = eval_df.copy()
    df["y_score"] = detector.predict_proba(df["generation"].tolist())
    df["y_pred"] = (df["y_score"] >= 0.5).astype(int)
    return df


def _plot_roc(y_true, y_score, label: str, ax: plt.Axes) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    tpr5 = tpr_at_fpr(y_true, y_score, 0.05)
    ax.plot(fpr, tpr, lw=2, label=f"{label}  AUC={roc_auc:.3f}  TPR@5%={tpr5:.3f}")
    ax.scatter([0.05], [tpr5], zorder=5, s=60)


def _attack_breakdown(df: pd.DataFrame, target_fpr: float = 0.05) -> pd.DataFrame:
    """Return per-attack TPR@FPR, AUC, F1 as a DataFrame."""
    rows = []
    for attack, grp in df.groupby("attack"):
        if grp["label"].nunique() < 2:
            continue
        m = compute_metrics(
            grp["label"].tolist(),
            grp["y_pred"].tolist(),
            grp["y_score"].tolist(),
            target_fpr=target_fpr,
        )
        m["attack"] = attack
        m["n"] = len(grp)
        rows.append(m)
    return pd.DataFrame(rows).set_index("attack")


def _save_comparison_plot(results: dict, output_dir: Path) -> None:
    key_metrics = ["accuracy", "f1", "auc", "tpr_at_5pct_fpr"]
    rows = {
        name: {k: v["overall"].get(k, float("nan")) for k in key_metrics}
        for name, v in results.items()
    }
    df = pd.DataFrame(rows).T

    fig, ax = plt.subplots(figsize=(8, 4))
    df.plot(kind="bar", ax=ax, edgecolor="white", width=0.6)
    ax.set_title("Model Comparison — Clean Eval Subset", fontsize=13)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(loc="lower right")
    plt.tight_layout()
    path = output_dir / "model_comparison.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Per-model evaluation
# ---------------------------------------------------------------------------

def run_model(
    name: str,
    detector,
    train_split: pd.DataFrame,
    eval_split: pd.DataFrame,
    eval_sample: int | None,
    output_dir: Path,
    target_fpr: float = 0.05,
) -> dict:
    print(f"\n{'='*60}")
    print(f"  Model: {name}")
    print(f"{'='*60}")

    # --- fit if supervised ---
    if hasattr(detector, "_fitted") and not detector._fitted:
        print("  Fitting on train split ...")
        detector.fit(train_split["generation"].tolist(), train_split["label"].tolist())
        print("  Fitting done.")

    # --- clean eval ---
    eval_clean = filter_no_attack(eval_split)
    if eval_sample and len(eval_clean) > eval_sample:
        eval_clean = eval_clean.sample(eval_sample, random_state=42).reset_index(drop=True)
    print(f"  Clean eval rows: {len(eval_clean):,}")

    eval_clean = _eval_detector(detector, eval_clean)
    y_true = eval_clean["label"].tolist()
    y_score = eval_clean["y_score"].tolist()
    y_pred = eval_clean["y_pred"].tolist()

    overall = compute_metrics(y_true, y_pred, y_score, target_fpr=target_fpr)
    print("  Overall (clean):")
    for k, v in overall.items():
        print(f"    {k:<25} {v:.4f}")

    # --- per-domain ---
    domain_results = per_domain_metrics(
        eval_clean, "y_score", "y_pred", target_fpr=target_fpr
    )

    # --- adversarial eval (all attacks) ---
    print("  Running adversarial eval ...")
    eval_all = _eval_detector(detector, eval_split)
    attack_df = _attack_breakdown(eval_all, target_fpr=target_fpr)
    print(attack_df[["tpr_at_5pct_fpr", "auc", "f1", "n"]].to_string())

    # --- per-model detection rate ---
    ai_rows = eval_all[eval_all["label"] == 1]
    fpr_arr, tpr_arr, thresholds = roc_curve(y_true, y_score)
    valid = fpr_arr <= target_fpr
    threshold_5fpr = float(thresholds[valid][-1]) if valid.any() else 0.5
    model_tpr = (
        ai_rows.groupby("model")["y_score"]
        .apply(lambda s: (s >= threshold_5fpr).mean())
        .sort_values(ascending=False)
    )

    # --- save JSON ---
    out = save_results(
        overall,
        experiment_name=f"pipeline_{name}",
        results_dir=output_dir,
        extra={
            "model": name,
            "subset": "no_attack",
            "n_clean": len(eval_clean),
            "n_total_eval": len(eval_split),
            "per_domain": domain_results,
            "per_attack": attack_df.reset_index().to_dict(orient="records"),
        },
    )

    # --- ROC plot ---
    fig, ax = plt.subplots(figsize=(7, 5))
    _plot_roc(y_true, y_score, name, ax)
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.axvline(0.05, color="tomato", lw=1, linestyle="--", alpha=0.7, label="FPR=5%")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve — {name}")
    ax.legend(loc="lower right")
    plt.tight_layout()
    roc_path = output_dir / f"roc_{name}.png"
    plt.savefig(roc_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved {roc_path}")

    # --- adversarial breakdown plot ---
    if len(attack_df) > 0:
        fig, ax = plt.subplots(figsize=(11, 4))
        attack_df["tpr_at_5pct_fpr"].sort_values(ascending=False).plot(
            kind="bar", ax=ax, color="coral", edgecolor="white"
        )
        ax.axhline(
            overall["tpr_at_5pct_fpr"], color="steelblue", lw=1.5,
            linestyle="--", label=f"Clean TPR@5%FPR={overall['tpr_at_5pct_fpr']:.3f}"
        )
        ax.set_title(f"TPR @ 5% FPR by Attack Type — {name}", fontsize=13)
        ax.set_ylabel("TPR @ 5% FPR")
        ax.set_ylim(0, 1.05)
        ax.tick_params(axis="x", rotation=35)
        ax.legend()
        plt.tight_layout()
        adv_path = output_dir / f"adversarial_tpr_{name}.png"
        plt.savefig(adv_path, bbox_inches="tight")
        plt.close()
        print(f"  Saved {adv_path}")

    # --- per-model detection rate plot ---
    if len(model_tpr) > 0:
        fig, ax = plt.subplots(figsize=(10, 4))
        model_tpr.plot(kind="bar", ax=ax, color="mediumseagreen", edgecolor="white")
        ax.set_title(f"Detection Rate per Source Model @ {target_fpr*100:.0f}% FPR — {name}", fontsize=13)
        ax.set_ylabel("Detection Rate")
        ax.set_ylim(0, 1.05)
        ax.tick_params(axis="x", rotation=35)
        plt.tight_layout()
        mp_path = output_dir / f"detection_by_model_{name}.png"
        plt.savefig(mp_path, bbox_inches="tight")
        plt.close()
        print(f"  Saved {mp_path}")

    return {
        "overall": overall,
        "per_domain": domain_results,
        "attack_df": attack_df,
        "model_tpr": model_tpr,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run the RAID detection pipeline.")
    parser.add_argument(
        "--model",
        choices=["tfidf", "roberta", "all"],
        default="tfidf",
        help="Detector to run (default: tfidf)",
    )
    parser.add_argument(
        "--train-sample",
        type=int,
        default=100_000,
        metavar="N",
        help="Rows to load from train.csv (default: 100000; None = all)",
    )
    parser.add_argument(
        "--eval-sample",
        type=int,
        default=None,
        metavar="N",
        help="Max clean eval rows per model (default: all)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory containing train.csv / test.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Where to save plots and JSON (default: results/)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of loaded data reserved for evaluation (default: 0.2)",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ---- load data ---------------------------------------------------------
    print(f"Loading up to {args.train_sample:,} rows from {args.data_dir}/train.csv ...")
    train_df, _ = load_raid(str(args.data_dir), train_nrows=args.train_sample)
    print(f"  Loaded {len(train_df):,} rows  |  label balance: {train_df['label'].value_counts().to_dict()}")

    train_split, eval_split = train_test_split(
        train_df,
        test_size=args.test_size,
        random_state=42,
        stratify=train_df["label"],
    )
    train_split = train_split.reset_index(drop=True)
    eval_split = eval_split.reset_index(drop=True)
    print(f"  Train split: {len(train_split):,}  |  Eval split: {len(eval_split):,}")

    # ---- build detectors ---------------------------------------------------
    models_to_run: list[tuple[str, object]] = []

    if args.model in ("tfidf", "all"):
        from src.models.statistical import StatisticalDetector
        models_to_run.append(("tfidf", StatisticalDetector()))

    if args.model in ("roberta", "all"):
        from src.models.baseline import ZeroShotDetector
        models_to_run.append(("roberta", ZeroShotDetector(
            model_name="roberta-base-openai-detector", batch_size=64
        )))

    # ---- run each model ----------------------------------------------------
    all_results: dict[str, dict] = {}
    for name, detector in models_to_run:
        all_results[name] = run_model(
            name=name,
            detector=detector,
            train_split=train_split,
            eval_split=eval_split,
            eval_sample=args.eval_sample,
            output_dir=args.output_dir,
        )

    # ---- comparison plot (only if >1 model) --------------------------------
    if len(all_results) > 1:
        _save_comparison_plot(all_results, args.output_dir)

    # ---- final summary table -----------------------------------------------
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    header = f"{'Model':<12}  {'Accuracy':>9}  {'F1':>7}  {'AUC':>7}  {'TPR@5%FPR':>10}"
    print(header)
    print("-" * len(header))
    for name, res in all_results.items():
        m = res["overall"]
        print(
            f"{name:<12}  {m['accuracy']:>9.4f}  {m['f1']:>7.4f}"
            f"  {m.get('auc', float('nan')):>7.4f}  {m.get('tpr_at_5pct_fpr', float('nan')):>10.4f}"
        )

    print(f"\nAll outputs written to {args.output_dir.resolve()}")
    print("Pipeline complete.")


if __name__ == "__main__":
    main()
