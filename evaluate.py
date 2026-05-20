"""
evaluate.py
-----------
Validates threshold and fusion weight choices against labeled validation data.

Inputs:
  validation_data/validation.csv  (from record_validation.py)

Outputs (in validation_data/results/):
  score_distribution.png         — histogram of fatigue scores by ground truth
  roc_curve.png                  — ROC + AUC + current threshold marker
  pr_curve.png                   — precision-recall curve
  f1_vs_threshold.png            — F1 score sweep across thresholds
  confusion_matrix.png           — at default threshold (0.65)
  per_scenario_accuracy.png      — accuracy per recorded scenario
  threshold_sweep.csv            — raw numeric table
  ablation_study.md              — single-signal vs multi-signal F1 table
  weight_grid_search.md          — grid search over weight space
  summary.md                     — final report-ready summary

Usage:
  python evaluate.py
  python evaluate.py --csv validation_data/validation.csv
"""

import argparse
import json
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    confusion_matrix, f1_score, precision_score, recall_score, accuracy_score,
)

# ── Defaults (must match detect.py) ───────────────────────────────────────────
DEFAULT_WEIGHTS = {
    "eye":     0.35,
    "perclos": 0.15,
    "yawn":    0.20,
    "nod":     0.30,
}
DEFAULT_THRESHOLD = 0.65

THRESHOLD_RANGE = np.arange(0.20, 0.91, 0.025)

# Grid search step (will explore 0.0, 0.1, 0.2, ..., 1.0 per signal,
# constrained to sum = 1.0)
GRID_STEP = 0.10

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
DEFAULT_CSV = BASE_DIR / "validation_data" / "validation.csv"


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, comment=None)
    # Drop metadata row if present
    df = df[df["frame_idx"].astype(str).str.startswith("#") == False].copy()
    # Convert types
    df["frame_idx"]  = df["frame_idx"].astype(int)
    df["gt_label"]   = df["gt_label"].astype(int)
    for col in ["ear_value", "eye_signal", "yawn_prob", "perclos",
                "perclos_signal", "nod_intensity", "score_default",
                "pitch_offset"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["eye_signal", "yawn_prob",
                            "perclos_signal", "nod_intensity"])
    return df


def compute_fusion_score(df, weights):
    """Compute fusion score with given weight dict."""
    return (
        weights["eye"]     * df["eye_signal"]
        + weights["perclos"] * df["perclos_signal"]
        + weights["yawn"]    * df["yawn_prob"]
        + weights["nod"]     * df["nod_intensity"]
    )


def metrics_at_threshold(scores, labels, threshold):
    preds = (scores >= threshold).astype(int)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    a  = (tp + tn) / max(tp + fp + tn + fn, 1)
    return {
        "threshold": threshold,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": p, "recall": r, "f1": f1, "accuracy": a,
    }


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_score_distribution(df, out_path, weights):
    scores = compute_fusion_score(df, weights)
    labels = df["gt_label"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(scores[labels == 0], bins=40, alpha=0.65, label="NORMAL (label=0)",
            color="#14B8A6", edgecolor="white")
    ax.hist(scores[labels == 1], bins=40, alpha=0.65, label="FATIGUE (label=1)",
            color="#F59E0B", edgecolor="white")
    ax.axvline(DEFAULT_THRESHOLD, color="red", linestyle="--", linewidth=2,
               label=f"current threshold = {DEFAULT_THRESHOLD}")
    ax.set_xlabel("Fatigue score")
    ax.set_ylabel("Frame count")
    ax.set_title("Fatigue Score Distribution by Ground Truth Label")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_roc_curve(df, out_path, weights):
    scores = compute_fusion_score(df, weights)
    labels = df["gt_label"].values

    fpr, tpr, thresholds = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    # Find current threshold operating point
    cur_idx   = np.argmin(np.abs(thresholds - DEFAULT_THRESHOLD))
    cur_fpr   = fpr[cur_idx]
    cur_tpr   = tpr[cur_idx]

    # Find Youden's J optimum
    youden_j   = tpr - fpr
    opt_idx    = int(np.argmax(youden_j))
    opt_thr    = float(thresholds[opt_idx])
    opt_fpr    = fpr[opt_idx]
    opt_tpr    = tpr[opt_idx]

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(fpr, tpr, color="#0F2A47", linewidth=2.2, label=f"ROC (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="random")
    ax.scatter([cur_fpr], [cur_tpr], color="red", s=120, zorder=5, edgecolor="white",
               label=f"current thr = {DEFAULT_THRESHOLD:.2f}\n(FPR={cur_fpr:.3f}, TPR={cur_tpr:.3f})")
    ax.scatter([opt_fpr], [opt_tpr], color="#F59E0B", s=120, zorder=5,
               edgecolor="white", marker="^",
               label=f"Youden's J opt thr = {opt_thr:.3f}\n(FPR={opt_fpr:.3f}, TPR={opt_tpr:.3f})")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Fatigue Detection")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return roc_auc, opt_thr


def plot_pr_curve(df, out_path, weights):
    scores = compute_fusion_score(df, weights)
    labels = df["gt_label"].values
    p, r, thresholds = precision_recall_curve(labels, scores)
    ap = average_precision_score(labels, scores)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(r, p, color="#0F2A47", linewidth=2.2, label=f"PR (AP = {ap:.3f})")
    baseline = labels.mean()
    ax.axhline(baseline, color="grey", linestyle="--", alpha=0.6,
               label=f"baseline = {baseline:.3f} (class prevalence)")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return ap


def plot_f1_vs_threshold(df, out_path, weights):
    scores = compute_fusion_score(df, weights).values
    labels = df["gt_label"].values

    results = [metrics_at_threshold(scores, labels, t) for t in THRESHOLD_RANGE]
    thrs = [r["threshold"] for r in results]
    f1s  = [r["f1"] for r in results]
    ps   = [r["precision"] for r in results]
    rs   = [r["recall"] for r in results]

    best_idx = int(np.argmax(f1s))
    best_thr = thrs[best_idx]
    best_f1  = f1s[best_idx]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thrs, f1s, color="#0F2A47", linewidth=2.2, label="F1")
    ax.plot(thrs, ps,  color="#14B8A6", linewidth=1.5, linestyle="--", label="Precision")
    ax.plot(thrs, rs,  color="#F59E0B", linewidth=1.5, linestyle="--", label="Recall")
    ax.axvline(DEFAULT_THRESHOLD, color="red", linestyle=":", linewidth=2,
               label=f"current thr = {DEFAULT_THRESHOLD}")
    ax.axvline(best_thr, color="green", linestyle=":", linewidth=2,
               label=f"best F1 thr = {best_thr:.3f} (F1 = {best_f1:.3f})")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_title("Precision / Recall / F1 vs Alert Threshold")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xlim(THRESHOLD_RANGE.min(), THRESHOLD_RANGE.max())
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return best_thr, best_f1, results


def plot_confusion_matrix(df, out_path, weights, threshold):
    scores = compute_fusion_score(df, weights)
    labels = df["gt_label"].values
    preds  = (scores >= threshold).astype(int).values
    cm     = confusion_matrix(labels, preds)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["pred: NORMAL", "pred: FATIGUE"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["true: NORMAL", "true: FATIGUE"])
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=18, fontweight="bold", color=color)
    ax.set_title(f"Confusion Matrix  (threshold = {threshold:.3f})")
    fig.colorbar(im, ax=ax, shrink=0.7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_per_scenario(df, out_path, weights, threshold):
    scores = compute_fusion_score(df, weights)
    df = df.assign(score=scores, pred=(scores >= threshold).astype(int))

    rows = []
    for name, sub in df.groupby("scenario", sort=False):
        gt   = sub["gt_label"].iloc[0]
        acc  = (sub["pred"] == sub["gt_label"]).mean()
        rows.append({
            "scenario": name,
            "gt": gt,
            "n_frames": len(sub),
            "accuracy": acc,
            "mean_score": sub["score"].mean(),
        })
    sdf = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#14B8A6" if g == 0 else "#F59E0B" for g in sdf["gt"]]
    bars = ax.bar(range(len(sdf)), sdf["accuracy"], color=colors, edgecolor="white")
    for i, (acc, score) in enumerate(zip(sdf["accuracy"], sdf["mean_score"])):
        ax.text(i, acc + 0.02, f"acc={acc:.2f}\nμ={score:.2f}",
                ha="center", fontsize=8, color="#333")
    ax.set_xticks(range(len(sdf)))
    ax.set_xticklabels(sdf["scenario"], rotation=30, ha="right")
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Accuracy (within scenario)")
    ax.set_title(f"Per-Scenario Accuracy  (threshold = {threshold:.3f})")
    ax.grid(alpha=0.3, axis="y")
    # Legend
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor="#14B8A6", label="GT = NORMAL (acc should be high)"),
        Patch(facecolor="#F59E0B", label="GT = FATIGUE (acc should be high)"),
    ], loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return sdf


# ── Ablation ──────────────────────────────────────────────────────────────────

def ablation_study(df):
    """Compare single-signal baselines vs multi-signal fusion."""
    labels = df["gt_label"].values

    configs = [
        ("Eye only",       {"eye": 1.0, "perclos": 0.0, "yawn": 0.0, "nod": 0.0}),
        ("PERCLOS only",   {"eye": 0.0, "perclos": 1.0, "yawn": 0.0, "nod": 0.0}),
        ("Yawn only",      {"eye": 0.0, "perclos": 0.0, "yawn": 1.0, "nod": 0.0}),
        ("Nod only",       {"eye": 0.0, "perclos": 0.0, "yawn": 0.0, "nod": 1.0}),
        ("Equal (0.25×4)", {"eye": 0.25,"perclos": 0.25,"yawn": 0.25,"nod": 0.25}),
        ("DEFAULT (ours)", DEFAULT_WEIGHTS),
    ]

    rows = []
    for name, w in configs:
        scores = compute_fusion_score(df, w).values
        # Threshold sweep, pick best F1
        best = max(
            (metrics_at_threshold(scores, labels, t) for t in THRESHOLD_RANGE),
            key=lambda r: r["f1"],
        )
        # Also report metrics at default threshold
        at_default = metrics_at_threshold(scores, labels, DEFAULT_THRESHOLD)
        try:
            roc_auc = auc(*roc_curve(labels, scores)[:2])
        except Exception:
            roc_auc = float("nan")

        rows.append({
            "config":           name,
            "weights":          w,
            "best_threshold":   best["threshold"],
            "best_f1":          best["f1"],
            "best_precision":   best["precision"],
            "best_recall":      best["recall"],
            "f1_at_default":    at_default["f1"],
            "precision_at_def": at_default["precision"],
            "recall_at_def":    at_default["recall"],
            "roc_auc":          roc_auc,
        })
    return rows


# ── Weight grid search ────────────────────────────────────────────────────────

def grid_search_weights(df, step=GRID_STEP):
    """
    Brute-force search over weight space with constraint sum=1.0.
    Each weight ∈ {0.0, step, 2*step, ..., 1.0}.
    Returns top-K configs by best F1 across threshold sweep.
    """
    labels = df["gt_label"].values
    grid   = np.round(np.arange(0.0, 1.0 + 1e-9, step), 4)

    candidates = []
    for we, wp, wy, wn in product(grid, repeat=4):
        if abs((we + wp + wy + wn) - 1.0) > 1e-6:
            continue
        weights = {"eye": float(we), "perclos": float(wp),
                   "yawn": float(wy), "nod": float(wn)}
        scores = compute_fusion_score(df, weights).values
        # Best F1 across thresholds for this config
        best = max(
            (metrics_at_threshold(scores, labels, t) for t in THRESHOLD_RANGE),
            key=lambda r: r["f1"],
        )
        candidates.append({
            "weights": weights,
            "best_threshold": best["threshold"],
            "best_f1": best["f1"],
            "best_precision": best["precision"],
            "best_recall": best["recall"],
        })

    # Sort and pick top 15
    candidates.sort(key=lambda c: -c["best_f1"])
    return candidates


# ── Markdown writers ──────────────────────────────────────────────────────────

def write_threshold_sweep_csv(out_path, results):
    cols = ["threshold", "precision", "recall", "f1", "accuracy", "tp", "fp", "tn", "fn"]
    pd.DataFrame(results)[cols].to_csv(out_path, index=False)


def write_ablation_md(out_path, rows):
    lines = [
        "# Ablation Study — Single-Signal vs Multi-Signal Fusion",
        "",
        "Each row reports the best F1 score achievable for that weight",
        "configuration across the threshold sweep (0.20–0.90, step 0.025).",
        "",
        "| Configuration       | Weights (E / PCL / Y / N) | Best thr | Best F1 | Best P | Best R | ROC AUC | F1 @ thr=0.65 |",
        "|---------------------|---------------------------|----------|---------|--------|--------|---------|---------------|",
    ]
    for r in rows:
        w = r["weights"]
        w_str = f"{w['eye']:.2f}/{w['perclos']:.2f}/{w['yawn']:.2f}/{w['nod']:.2f}"
        lines.append(
            f"| {r['config']:<19} | {w_str:<25} | {r['best_threshold']:>8.3f} | "
            f"{r['best_f1']:>7.3f} | {r['best_precision']:>6.3f} | "
            f"{r['best_recall']:>6.3f} | {r['roc_auc']:>7.3f} | {r['f1_at_default']:>13.3f} |"
        )
    lines += [
        "",
        "**Interpretation:**",
        "- Single-signal rows show the limit of each modality alone.",
        "- Multi-signal fusion should outperform every single-signal baseline.",
        "- DEFAULT (ours) is the production configuration; nearby F1s in the grid search ",
        "  indicate the choice is robust.",
        "",
    ]
    out_path.write_text("\n".join(lines))


def write_grid_md(out_path, top_configs, default_f1, n_top=15):
    # Locate default in the candidates (if present)
    lines = [
        "# Fusion Weight Grid Search",
        "",
        f"Brute-force over all 4-weight combinations on a {GRID_STEP} grid",
        "with constraint Σw = 1.0. For each combination, the best F1 across",
        "the threshold sweep (0.20–0.90, step 0.025) is reported.",
        "",
        f"**Default (ours) weights produce best F1 = {default_f1:.3f}.**",
        "",
        f"## Top {n_top} configurations by best F1",
        "",
        "| Rank | Weights (E / PCL / Y / N) | Best thr | Best F1 | Best P | Best R |",
        "|------|---------------------------|----------|---------|--------|--------|",
    ]
    for rank, c in enumerate(top_configs[:n_top], start=1):
        w = c["weights"]
        marker = ""
        if (abs(w["eye"]     - DEFAULT_WEIGHTS["eye"])     < 1e-6 and
            abs(w["perclos"] - DEFAULT_WEIGHTS["perclos"]) < 1e-6 and
            abs(w["yawn"]    - DEFAULT_WEIGHTS["yawn"])    < 1e-6 and
            abs(w["nod"]     - DEFAULT_WEIGHTS["nod"])     < 1e-6):
            marker = "  ← OURS"
        w_str = f"{w['eye']:.2f}/{w['perclos']:.2f}/{w['yawn']:.2f}/{w['nod']:.2f}"
        lines.append(
            f"| {rank:>4} | {w_str:<25} | {c['best_threshold']:>8.3f} | "
            f"{c['best_f1']:>7.3f} | {c['best_precision']:>6.3f} | "
            f"{c['best_recall']:>6.3f} |{marker}"
        )
    lines += [
        "",
        "**Interpretation:**",
        "- If our weights appear in the top-K, that empirically validates the",
        "  literature-informed choice.",
        "- If a different configuration significantly outperforms ours, that",
        "  is grounds to either adopt the new weights or investigate why",
        "  (e.g. data bias, scenario imbalance).",
        "",
    ]
    out_path.write_text("\n".join(lines))


def write_summary_md(out_path, df, results, weights, roc_auc, ap,
                     best_thr, best_f1, at_default, scenario_df, top_grid):
    n_total  = len(df)
    n_pos    = int(df["gt_label"].sum())
    n_neg    = n_total - n_pos
    n_face   = int(df["face_detected"].sum())

    lines = [
        "# Validation Summary",
        "",
        "## Dataset",
        "",
        f"- Total frames                       : **{n_total}**",
        f"- Frames with face detected          : **{n_face}** ({100*n_face/n_total:.1f}%)",
        f"- FATIGUE frames (GT label = 1)      : **{n_pos}** ({100*n_pos/n_total:.1f}%)",
        f"- NORMAL  frames (GT label = 0)      : **{n_neg}** ({100*n_neg/n_total:.1f}%)",
        f"- Scenarios recorded                 : **{df['scenario'].nunique()}**",
        "",
        "## Headline Metrics  (default weights, threshold = 0.65)",
        "",
        f"| Metric         | Value  |",
        f"|----------------|--------|",
        f"| Accuracy       | {at_default['accuracy']:.3f}  |",
        f"| Precision      | {at_default['precision']:.3f}  |",
        f"| Recall         | {at_default['recall']:.3f}  |",
        f"| F1 score       | {at_default['f1']:.3f}  |",
        f"| ROC AUC        | {roc_auc:.3f}  |",
        f"| Avg. Precision | {ap:.3f}  |",
        "",
        f"## Threshold Validation",
        "",
        f"- **Threshold sweep range**       : 0.20 to 0.90 (step 0.025)",
        f"- **Best F1 threshold (data)**    : **{best_thr:.3f}** (F1 = {best_f1:.3f})",
        f"- **Default threshold (ours)**    : **{DEFAULT_THRESHOLD}** (F1 = {at_default['f1']:.3f})",
        f"- **Delta**                       : {abs(best_thr - DEFAULT_THRESHOLD):.3f} threshold units, "
        f"{(best_f1 - at_default['f1'])*100:.1f}% F1 improvement available",
        "",
        "## Weight Validation",
        "",
        f"Default weights: **eye={DEFAULT_WEIGHTS['eye']}, "
        f"PERCLOS={DEFAULT_WEIGHTS['perclos']}, "
        f"yawn={DEFAULT_WEIGHTS['yawn']}, "
        f"nod={DEFAULT_WEIGHTS['nod']}** (sum = 1.0).",
        "",
        f"Top configuration from grid search:",
    ]
    if top_grid:
        top = top_grid[0]
        w   = top["weights"]
        w_str = f"eye={w['eye']:.2f}, PCL={w['perclos']:.2f}, yawn={w['yawn']:.2f}, nod={w['nod']:.2f}"
        lines.append(f"- **Best weights**: {w_str}  (F1 = {top['best_f1']:.3f}, threshold = {top['best_threshold']:.3f})")
    lines += [
        "",
        "See `ablation_study.md` and `weight_grid_search.md` for details.",
        "",
        "## Per-Scenario Accuracy",
        "",
        "| Scenario             | GT      | Frames | Accuracy | Mean score |",
        "|----------------------|---------|--------|----------|------------|",
    ]
    for _, row in scenario_df.iterrows():
        gt_str = "FATIGUE" if row["gt"] == 1 else "NORMAL"
        lines.append(
            f"| {row['scenario']:<20} | {gt_str:<7} | "
            f"{int(row['n_frames']):>6} | {row['accuracy']:>8.3f} | {row['mean_score']:>10.3f} |"
        )
    lines += [
        "",
        "## Generated Artifacts",
        "",
        "All in `validation_data/results/`:",
        "- `score_distribution.png`       — score histograms by ground truth",
        "- `roc_curve.png`                — ROC + AUC + default threshold marker",
        "- `pr_curve.png`                 — precision-recall curve",
        "- `f1_vs_threshold.png`          — F1/precision/recall vs threshold",
        "- `confusion_matrix.png`         — at default threshold",
        "- `per_scenario_accuracy.png`    — per-scenario performance",
        "- `threshold_sweep.csv`          — raw threshold sweep data",
        "- `ablation_study.md`            — single-signal vs multi-signal",
        "- `weight_grid_search.md`        — grid search results",
        "",
    ]
    out_path.write_text("\n".join(lines))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Validate threshold/weight choices")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV,
                        help="Path to validation CSV from record_validation.py")
    parser.add_argument("--grid-step", type=float, default=GRID_STEP,
                        help=f"Grid search step (default: {GRID_STEP})")
    args = parser.parse_args()

    if not args.csv.exists():
        print(f"\n  [ERROR] CSV not found: {args.csv}")
        print("  Run  python record_validation.py  first.\n")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  Validation Analysis")
    print(f"{'='*60}")
    print(f"  Input CSV  : {args.csv.relative_to(BASE_DIR) if args.csv.is_absolute() else args.csv}")

    # Load
    df = load_csv(args.csv)
    print(f"  Frames     : {len(df)}")
    print(f"  Scenarios  : {df['scenario'].nunique()}  ({list(df['scenario'].unique())})")
    print(f"  Class dist : FATIGUE={int(df['gt_label'].sum())}, "
          f"NORMAL={len(df) - int(df['gt_label'].sum())}")

    # Prepare output dir
    results_dir = args.csv.parent / "results"
    results_dir.mkdir(exist_ok=True)
    print(f"  Output dir : {results_dir.relative_to(BASE_DIR) if results_dir.is_absolute() else results_dir}\n")

    weights = DEFAULT_WEIGHTS
    labels = df["gt_label"].values
    scores = compute_fusion_score(df, weights).values

    # 1. Score distribution
    print("  [1/8] Score distribution plot...")
    plot_score_distribution(df, results_dir / "score_distribution.png", weights)

    # 2. ROC
    print("  [2/8] ROC curve...")
    roc_auc, opt_thr = plot_roc_curve(df, results_dir / "roc_curve.png", weights)

    # 3. PR
    print("  [3/8] PR curve...")
    ap = plot_pr_curve(df, results_dir / "pr_curve.png", weights)

    # 4. F1 vs threshold
    print("  [4/8] F1 vs threshold...")
    best_thr, best_f1, sweep_results = plot_f1_vs_threshold(
        df, results_dir / "f1_vs_threshold.png", weights)
    write_threshold_sweep_csv(results_dir / "threshold_sweep.csv", sweep_results)

    # 5. Confusion matrix
    print("  [5/8] Confusion matrix...")
    plot_confusion_matrix(df, results_dir / "confusion_matrix.png", weights, DEFAULT_THRESHOLD)
    at_default = metrics_at_threshold(scores, labels, DEFAULT_THRESHOLD)

    # 6. Per-scenario
    print("  [6/8] Per-scenario accuracy...")
    scenario_df = plot_per_scenario(df, results_dir / "per_scenario_accuracy.png",
                                     weights, DEFAULT_THRESHOLD)

    # 7. Ablation
    print("  [7/8] Ablation study...")
    ablation_rows = ablation_study(df)
    write_ablation_md(results_dir / "ablation_study.md", ablation_rows)

    # 8. Grid search
    print(f"  [8/8] Grid search (step={args.grid_step})...")
    top_grid = grid_search_weights(df, step=args.grid_step)
    # Find default in grid
    default_in_grid = next(
        (c for c in top_grid
         if all(abs(c["weights"][k] - DEFAULT_WEIGHTS[k]) < 1e-6
                for k in DEFAULT_WEIGHTS)),
        None,
    )
    if default_in_grid:
        default_f1_grid = default_in_grid["best_f1"]
    else:
        # Default weights might not be on grid (e.g. grid step doesn't include them)
        default_f1_grid = best_f1
    write_grid_md(results_dir / "weight_grid_search.md", top_grid, default_f1_grid)

    # Summary
    write_summary_md(
        results_dir / "summary.md",
        df, sweep_results, weights, roc_auc, ap,
        best_thr, best_f1, at_default, scenario_df, top_grid,
    )

    # ── Console summary ───────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Summary")
    print(f"{'='*60}")
    print(f"  Default threshold (0.65):")
    print(f"     Accuracy : {at_default['accuracy']:.3f}")
    print(f"     Precision: {at_default['precision']:.3f}")
    print(f"     Recall   : {at_default['recall']:.3f}")
    print(f"     F1       : {at_default['f1']:.3f}")
    print(f"\n  Optimal threshold (best F1):")
    print(f"     Threshold: {best_thr:.3f}  (vs default {DEFAULT_THRESHOLD})")
    print(f"     F1       : {best_f1:.3f}   (vs default {at_default['f1']:.3f})")
    print(f"\n  ROC AUC     : {roc_auc:.3f}")
    print(f"  Avg Precision: {ap:.3f}")
    if top_grid:
        bw = top_grid[0]["weights"]
        print(f"\n  Best weights (grid search):")
        print(f"     eye={bw['eye']:.2f}  PCL={bw['perclos']:.2f}  "
              f"yawn={bw['yawn']:.2f}  nod={bw['nod']:.2f}  "
              f"→ F1={top_grid[0]['best_f1']:.3f}")
    print(f"\n  All artifacts written to: {results_dir}")
    print(f"  Read  results/summary.md  for a report-ready overview.\n")


if __name__ == "__main__":
    main()