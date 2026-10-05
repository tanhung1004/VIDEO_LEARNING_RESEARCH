from pathlib import Path
import importlib.util

import numpy as np
import pandas as pd

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 30)


# ============================================================
# TRACK B - B4: EVALUATE A-F ON DEV (FROZEN PROTOCOL + LOCKED CONTRACT)
#
# Place this file at:
#   module1/v2_validation/03_evaluate_ablation_af_dev.py
#
# Can only run AFTER:
#   - B3 (02_build_full_ablation_af.py) has produced all 6 fusion_scores.csv
#   - Track A has locked the contract (docs/V2_METRIC_AUDIT.md,
#     docs/V2_STATISTICAL_PROTOCOL.md section 18)
#
# This script does NOT invent any metric formula. Every number here is
# produced by step 13's own audited functions
# (module1/ocr_best_integration/13_compare_old_vs_new_combined.py),
# loaded read-only via importlib - same file V2_METRIC_AUDIT.md lists
# as audited. V1 is never written to.
#
# Locked values reused as-is (not re-derived here):
#   - aggregation = max                      (V2_STATISTICAL_PROTOCOL.md S3)
#   - threshold   = 0.50                     (V2_STATISTICAL_PROTOCOL.md S3)
#   - evaluation universe = 390 same-subject video-concept pairs,
#     153 positive / 237 negative            (V2_METRIC_AUDIT.md S9)
#   - Macro Video F1 = mean of the 40 per-video F1 values
#                                             (V2_METRIC_AUDIT.md S7)
#   - primary endpoint = macro_video_f1      (V2_STATISTICAL_PROTOCOL.md S4)
#
# Selection rule reused as-is (V2_STATISTICAL_PROTOCOL.md section 18):
#   1. Highest Macro Video F1
#   2. Highest Micro F1
#   3. Lowest Macro Video F1 standard deviation
#   4. Deterministic order A -> B -> C -> D -> E -> F
#   No variant-specific threshold/aggregation retuning is applied here -
#   every variant uses the exact same AGGREGATION/THRESHOLD above.
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parents[2]

OCR_BEST_DIR = PROJECT_ROOT / "module1" / "ocr_best_integration"

ABLATION_ROOT = PROJECT_ROOT / "module1" / "results" / "v2_validation" / "ablation"

OUTPUT_DIR = PROJECT_ROOT / "module1" / "results" / "v2_validation"
COMPARISON_FILE = OUTPUT_DIR / "A_F_DEV_COMPARISON.csv"
SUMMARY_FILE = OUTPUT_DIR / "A_F_DEV_SUMMARY.txt"

VARIANT_ORDER = ["A", "B", "C", "D", "E", "F"]

VARIANT_LABEL = {
    "A": "Transcript-only",
    "B": "OCR-only, full-frame cleaned",
    "C": "Transcript + full-frame OCR raw",
    "D": "Transcript + full-frame OCR cleaned",
    "E": "Transcript + ROI OCR cleaned",
    "F": "Combined D + E at score level (mean)",
}

# Locked P1/NEW protocol - not re-derived, taken from
# V2_STATISTICAL_PROTOCOL.md section 3 / section 18.
AGGREGATION = "max"
THRESHOLD = 0.50


def load_python_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Read-only reuse of step 13's audited metric implementation.
STEP13 = load_python_module(
    "step13_eval_af",
    OCR_BEST_DIR / "13_compare_old_vs_new_combined.py",
)


def evaluate_variant(variant, videos, concepts, gt):
    path = ABLATION_ROOT / variant / "fusion_scores.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Variant {variant} score file missing:\n{path}\n"
            "Chay B3 (02_build_full_ablation_af.py) truoc."
        )

    # load_scores() audits row count against an expected value; the chunk
    # grid (172 chunks) and concept-per-subject split are identical for
    # every A-F variant (only the document TEXT differs), so the expected
    # score-row count is the same for all 6 and is read directly off the
    # file rather than hard-coded here.
    raw = pd.read_csv(path)
    scores = STEP13.load_scores(path, f"Variant {variant}", expected_rows=len(raw))

    score_name = f"score_{variant}"

    evidence = STEP13.build_video_evidence(
        scores,
        score_name=score_name,
        aggregation=AGGREGATION,
    )

    pairs = STEP13.build_pair_space(videos, concepts, gt)
    pairs = pairs.merge(
        evidence,
        on=["video_id", "subject_norm", "concept_norm"],
        how="left",
    )
    pairs[score_name] = pairs[score_name].fillna(0.0)
    pred_name = f"pred_{variant}"
    pairs[pred_name] = (pairs[score_name] >= THRESHOLD).astype(int)

    video_rows = []
    for (video_id, subject), group in pairs.groupby(["video_id", "subject"], sort=False):
        metrics = STEP13.compute_metrics(group["y_true"], group[pred_name], group[score_name])
        video_rows.append({"video_id": video_id, "subject": subject, **metrics})
    video_df = pd.DataFrame(video_rows)

    if len(video_df) != STEP13.EXPECTED_VIDEOS:
        raise RuntimeError(f"Variant {variant}: expected {STEP13.EXPECTED_VIDEOS} videos, got {len(video_df)}.")

    pair_metrics = STEP13.compute_metrics(pairs["y_true"], pairs[pred_name], pairs[score_name])

    summary = {
        "variant": variant,
        "definition": VARIANT_LABEL[variant],
        "macro_video_f1": float(video_df["f1"].mean()),
        "macro_video_f1_std": float(video_df["f1"].std(ddof=1)),
        "macro_video_precision": float(video_df["precision"].mean()),
        "macro_video_recall": float(video_df["recall"].mean()),
        "micro_precision": pair_metrics["precision"],
        "micro_recall": pair_metrics["recall"],
        "micro_f1": pair_metrics["f1"],
        "mcc": pair_metrics["mcc"],
        "balanced_accuracy": pair_metrics["balanced_accuracy"],
        "specificity": pair_metrics["specificity"],
        "fpr": pair_metrics["fpr"],
        "pr_auc": pair_metrics["pr_auc"],
        "roc_auc": pair_metrics["roc_auc"],
        "TP": pair_metrics["TP"],
        "FP": pair_metrics["FP"],
        "FN": pair_metrics["FN"],
        "TN": pair_metrics["TN"],
        "n_videos": len(video_df),
        "n_pairs": len(pairs),
        "n_positive": int(pairs["y_true"].sum()),
        "aggregation": AGGREGATION,
        "threshold": THRESHOLD,
        "score_rows": len(scores),
    }

    return summary, pairs, video_df


def apply_selection_rule(comparison):
    """V2_STATISTICAL_PROTOCOL.md section 18:
    1) highest macro_video_f1, 2) highest micro_f1,
    3) lowest macro_video_f1_std, 4) deterministic A->B->C->D->E->F."""

    variant_rank = {v: i for i, v in enumerate(VARIANT_ORDER)}

    ranked = comparison.copy()
    ranked["_rank_key"] = list(
        zip(
            -ranked["macro_video_f1"],
            -ranked["micro_f1"],
            ranked["macro_video_f1_std"],
            ranked["variant"].map(variant_rank),
        )
    )
    ranked = ranked.sort_values("_rank_key").reset_index(drop=True)
    ranked["selection_rank"] = range(1, len(ranked) + 1)
    ranked["selected"] = ranked["selection_rank"] == 1
    ranked = ranked.drop(columns=["_rank_key"])

    # restore canonical A-F display order, keep selection_rank/selected
    ranked = ranked.set_index("variant").loc[VARIANT_ORDER].reset_index()
    return ranked


def main():
    print("=" * 72)
    print("TRACK B - B4: EVALUATE A-F ON DEV (LOCKED CONTRACT)")
    print("=" * 72)
    print()
    print(f"Aggregation = {AGGREGATION}")
    print(f"Threshold   = {THRESHOLD:.2f}")
    print("Primary endpoint = macro_video_f1 (locked before these results were observed)")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    videos = STEP13.load_videos()
    concepts = STEP13.load_concepts()
    gt = STEP13.load_gt()

    # Cross-check the evaluation universe against the numbers already
    # written down in V2_METRIC_AUDIT.md section 9, instead of trusting
    # either source blindly.
    check_pairs = STEP13.build_pair_space(videos, concepts, gt)
    if len(check_pairs) != STEP13.EXPECTED_PAIRS:
        raise RuntimeError(
            f"Evaluation universe mismatch: got {len(check_pairs)} pairs, "
            f"V2_METRIC_AUDIT.md / step13 both expect {STEP13.EXPECTED_PAIRS}."
        )
    if int(check_pairs["y_true"].sum()) != STEP13.EXPECTED_GT_POSITIVES:
        raise RuntimeError(
            f"Positive pair count mismatch: got {int(check_pairs['y_true'].sum())}, "
            f"expected {STEP13.EXPECTED_GT_POSITIVES}."
        )

    rows = []
    video_level = {}

    for variant in VARIANT_ORDER:
        summary, pairs, video_df = evaluate_variant(variant, videos, concepts, gt)
        rows.append(summary)
        video_level[variant] = video_df
        print(
            f"Variant {variant} ({VARIANT_LABEL[variant]}): "
            f"macro_video_f1={summary['macro_video_f1']:.6f}, "
            f"micro_f1={summary['micro_f1']:.6f}, "
            f"std={summary['macro_video_f1_std']:.6f}"
        )

    comparison = pd.DataFrame(rows)
    comparison = apply_selection_rule(comparison)

    comparison.to_csv(COMPARISON_FILE, index=False, encoding="utf-8-sig")

    winner_row = comparison[comparison["selected"]].iloc[0]
    winner = winner_row["variant"]

    tie_on_f1 = int(np.isclose(comparison["macro_video_f1"], winner_row["macro_video_f1"], atol=1e-12).sum()) > 1
    tie_on_micro = int(
        np.isclose(
            comparison.loc[
                np.isclose(comparison["macro_video_f1"], winner_row["macro_video_f1"], atol=1e-12),
                "micro_f1",
            ],
            winner_row["micro_f1"],
            atol=1e-12,
        ).sum()
    ) > 1 if tie_on_f1 else False

    key_columns = [
        "variant",
        "selection_rank",
        "selected",
        "macro_video_f1",
        "macro_video_f1_std",
        "macro_video_precision",
        "macro_video_recall",
        "micro_f1",
        "pr_auc",
        "roc_auc",
    ]
    summary_view = comparison[key_columns].copy()
    float_cols = summary_view.select_dtypes(include="float").columns
    summary_view[float_cols] = summary_view[float_cols].round(6)

    print()
    print("=" * 72)
    print("A-F DEV COMPARISON (summary - full metrics in A_F_DEV_COMPARISON.csv)")
    print("=" * 72)
    print(summary_view.to_string(index=False))

    # --------------------------------------------------------
    # SUMMARY.txt - careful wording per kehoach.docx B4
    # --------------------------------------------------------

    lines = []
    lines.append("A-F DEV ABLATION - SUMMARY")
    lines.append("=" * 72)
    lines.append("")
    lines.append(f"Primary endpoint (locked before these results): macro_video_f1")
    lines.append(f"Aggregation: {AGGREGATION}    Threshold: {THRESHOLD:.2f}")
    lines.append(f"Evaluation universe: {STEP13.EXPECTED_PAIRS} same-subject video-concept pairs "
                  f"({STEP13.EXPECTED_VIDEOS} DEV videos, {STEP13.EXPECTED_GT_POSITIVES} positive pairs)")
    lines.append("")
    lines.append("Selection rule (V2_STATISTICAL_PROTOCOL.md section 18):")
    lines.append("  1) highest macro_video_f1")
    lines.append("  2) highest micro_f1 (tie-break)")
    lines.append("  3) lowest macro_video_f1_std (tie-break)")
    lines.append("  4) deterministic order A -> B -> C -> D -> E -> F (tie-break)")
    lines.append("")
    lines.append("Results (DEV, frozen protocol, all 6 variants same aggregation/threshold):")
    for row in comparison.itertuples():
        lines.append(
            f"  {row.variant} ({VARIANT_LABEL[row.variant]}): "
            f"macro_video_f1={row.macro_video_f1:.6f}, "
            f"std={row.macro_video_f1_std:.6f}, "
            f"micro_f1={row.micro_f1:.6f}, "
            f"rank={row.selection_rank}"
        )
    lines.append("")
    lines.append(
        f"Variant {winner} was selected on DEV under the frozen development protocol."
    )
    if tie_on_f1:
        lines.append(
            "NOTE: macro_video_f1 was tied between multiple variants; the tie-break "
            "rules in the locked selection contract were applied to reach this pick."
        )
    if tie_on_micro:
        lines.append(
            "NOTE: micro_f1 was also tied among the macro_video_f1-tied variants; "
            "the lowest macro_video_f1_std / deterministic order rule was applied."
        )
    lines.append("")
    lines.append(
        "This is a DEV-stage model/variant-selection result. It is NOT an "
        "independent confirmatory estimate, and it does NOT establish that "
        f"variant {winner} is superior in the population. Secondary metrics "
        "(precision, recall, MCC, balanced accuracy, specificity, FPR, "
        "PR-AUC, ROC-AUC) are reported in A_F_DEV_COMPARISON.csv for "
        "interpretation only and were not used to override the primary "
        "selection criterion."
    )
    lines.append("")
    lines.append(f"Saved: {COMPARISON_FILE}")

    SUMMARY_FILE.write_text("\n".join(lines), encoding="utf-8")

    print()
    print(f"Variant {winner} was selected on DEV under the frozen development protocol.")
    print()
    print("Saved:", COMPARISON_FILE)
    print("Saved:", SUMMARY_FILE)
    print()
    print("B4 DONE. No candidate has been locked yet - that is B5")
    print("(04_lock_v2_candidate.py), which also independently re-checks")
    print("this selection before freezing it.")


if __name__ == "__main__":
    main()
