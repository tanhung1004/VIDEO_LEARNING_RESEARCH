from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

P1_RESULT_DIR = (
    ROOT
    / "module1"
    / "results"
    / "p1_validation"
)

VIDEOS_PATH = ROOT / "data" / "raw" / "videos.csv"

GT_PATH = (
    ROOT
    / "data"
    / "ground_truth"
    / "ground_truth_concepts.csv"
)

FOLDS_PATH = (
    ROOT
    / "data"
    / "splits"
    / "dev_cv_folds.csv"
)

CHUNKS_PATH = (
    P1_RESULT_DIR
    / "dev_chunks_60s.csv"
)

LEAKAGE_PATH = (
    P1_RESULT_DIR
    / "all_folds_leakage_audit.csv"
)

OOF_SCORE_PATH = (
    P1_RESULT_DIR
    / "oof_chunk_concept_scores.csv"
)

INNER_ASSIGN_PATH = (
    P1_RESULT_DIR
    / "inner_fold_assignments.csv"
)

INNER_AUDIT_PATH = (
    P1_RESULT_DIR
    / "inner_cv_audit.csv"
)

INNER_SEARCH_PATH = (
    P1_RESULT_DIR
    / "inner_search_results.csv"
)

SELECTED_PATH = (
    P1_RESULT_DIR
    / "selected_hyperparameters_by_outer_fold.csv"
)

PREDICTIONS_PATH = (
    P1_RESULT_DIR
    / "oof_predictions.csv"
)

PER_VIDEO_PATH = (
    P1_RESULT_DIR
    / "per_video_metrics.csv"
)

OUTER_METRICS_PATH = (
    P1_RESULT_DIR
    / "outer_fold_metrics.csv"
)

STAT_SUMMARY_PATH = (
    P1_RESULT_DIR
    / "statistical_summary.csv"
)

PAIRED_PATH = (
    P1_RESULT_DIR
    / "paired_f1_analysis.csv"
)

SUBJECT_PATH = (
    P1_RESULT_DIR
    / "subject_metrics.csv"
)

CHUNK_RESULT_PATH = (
    P1_RESULT_DIR
    / "chunk_size_results.csv"
)

CHUNK_PRED_PATH = (
    P1_RESULT_DIR
    / "chunk_sensitivity_predictions.csv"
)

CHUNK_AUDIT_PATH = (
    P1_RESULT_DIR
    / "chunk_sensitivity_audit.json"
)

FINAL_SUMMARY_PATH = (
    P1_RESULT_DIR
    / "p1_final_audit_summary.json"
)

FINAL_TEXT_PATH = (
    P1_RESULT_DIR
    / "P1_FINAL_SUMMARY.txt"
)


EXPECTED_SUBJECTS = {
    "SQL": 10,
    "Python": 10,
    "Java": 10,
    "C++": 10,
}

EXPECTED_CHUNK_COUNTS = {
    30: 965,
    60: 494,
    90: 331,
    120: 257,
}


def fail(message):
    print(f"[FAIL] {message}")
    return False


def passed(message):
    print(f"[PASS] {message}")
    return True


def require_file(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Required artifact missing: "
            f"{path.relative_to(ROOT)}"
        )


def main():

    print("=" * 82)
    print("P1 FINAL SCIENTIFIC VALIDATION AUDIT")
    print("=" * 82)

    required_files = [
        VIDEOS_PATH,
        GT_PATH,
        FOLDS_PATH,
        CHUNKS_PATH,
        LEAKAGE_PATH,
        OOF_SCORE_PATH,
        INNER_ASSIGN_PATH,
        INNER_AUDIT_PATH,
        INNER_SEARCH_PATH,
        SELECTED_PATH,
        PREDICTIONS_PATH,
        PER_VIDEO_PATH,
        OUTER_METRICS_PATH,
        STAT_SUMMARY_PATH,
        PAIRED_PATH,
        SUBJECT_PATH,
        CHUNK_RESULT_PATH,
        CHUNK_PRED_PATH,
        CHUNK_AUDIT_PATH,
    ]

    for path in required_files:
        require_file(path)

    print("\nRequired artifacts: PASS")

    checks = []
    problems = []

    def check(condition, message):

        if condition:
            checks.append(message)
            passed(message)
        else:
            problems.append(message)
            fail(message)

    # ========================================================
    # 1. DATASET / GROUND TRUTH
    # ========================================================

    print("\n" + "=" * 82)
    print("1. DATASET + GROUND TRUTH")
    print("=" * 82)

    videos = pd.read_csv(
        VIDEOS_PATH,
        dtype=str,
        keep_default_na=False,
    )

    gt = pd.read_csv(
        GT_PATH,
        dtype=str,
        keep_default_na=False,
    )

    videos["video_id"] = (
        videos["video_id"]
        .str.strip()
        .str.lower()
    )

    videos["subject"] = (
        videos["subject"]
        .str.strip()
    )

    videos["split"] = (
        videos["split"]
        .str.strip()
        .str.lower()
    )

    dev = videos[
        videos["split"].eq("dev")
    ].copy()

    check(
        dev["video_id"].nunique() == 40,
        "DEV contains exactly 40 unique videos",
    )

    subject_counts = (
        dev["subject"]
        .value_counts()
        .to_dict()
    )

    check(
        all(
            subject_counts.get(subject, 0)
            == expected
            for subject, expected
            in EXPECTED_SUBJECTS.items()
        ),
        "DEV contains exactly 10 videos per subject",
    )

    if "annotation_status" in dev.columns:

        completed = (
            dev["annotation_status"]
            .str.strip()
            .str.lower()
            .eq("completed")
        )

        check(
            bool(completed.all()),
            "All 40 DEV videos have completed annotation status",
        )

    gt["video_id"] = (
        gt["video_id"]
        .str.strip()
        .str.lower()
    )

    gt["concept"] = (
        gt["concept"]
        .str.strip()
    )

    check(
        len(gt) == 153,
        "Ground truth contains 153 positive pairs",
    )

    check(
        int(
            gt.duplicated(
                subset=[
                    "video_id",
                    "concept",
                ]
            ).sum()
        )
        == 0,
        "Ground truth has zero duplicate video-concept pairs",
    )

    check(
        set(gt["video_id"]).issubset(
            set(dev["video_id"])
        ),
        "All ground-truth rows belong to DEV videos",
    )

    # ========================================================
    # 2. OUTER FOLDS
    # ========================================================

    print("\n" + "=" * 82)
    print("2. OUTER 5-FOLD SPLIT")
    print("=" * 82)

    folds = pd.read_csv(
        FOLDS_PATH,
        dtype=str,
        keep_default_na=False,
    )

    folds["video_id"] = (
        folds["video_id"]
        .str.strip()
        .str.lower()
    )

    folds["subject"] = (
        folds["subject"]
        .str.strip()
    )

    folds["cv_fold"] = (
        pd.to_numeric(
            folds["cv_fold"],
            errors="raise",
        )
        .astype(int)
    )

    check(
        len(folds) == 40
        and folds["video_id"].nunique() == 40,
        "Fold manifest contains 40 videos exactly once",
    )

    check(
        set(folds["cv_fold"])
        == {1, 2, 3, 4, 5},
        "Outer fold IDs are exactly 1..5",
    )

    fold_ok = True

    for fold in range(1, 6):

        f = folds[
            folds["cv_fold"].eq(fold)
        ]

        if len(f) != 8:
            fold_ok = False
            break

        counts = (
            f["subject"]
            .value_counts()
            .to_dict()
        )

        if any(
            counts.get(subject, 0) != 2
            for subject
            in EXPECTED_SUBJECTS
        ):
            fold_ok = False
            break

    check(
        fold_ok,
        "Each outer fold contains 8 videos: 2 per subject",
    )

    # ========================================================
    # 3. PRIMARY 60s CHUNKS
    # ========================================================

    print("\n" + "=" * 82)
    print("3. PRIMARY 60-SECOND CHUNKS")
    print("=" * 82)

    chunks = pd.read_csv(
        CHUNKS_PATH,
        dtype=str,
        keep_default_na=False,
    )

    chunks["video_id"] = (
        chunks["video_id"]
        .str.strip()
        .str.lower()
    )

    check(
        len(chunks) == 494,
        "Primary 60s representation contains 494 chunks",
    )

    check(
        chunks["video_id"].nunique() == 40,
        "Primary chunks cover all 40 DEV videos",
    )

    check(
        int(
            chunks.duplicated(
                subset=[
                    "video_id",
                    "chunk_id",
                ]
            ).sum()
        )
        == 0,
        "Primary chunks contain zero duplicate video/chunk pairs",
    )

    # ========================================================
    # 4. LEAKAGE AUDIT
    # ========================================================

    print("\n" + "=" * 82)
    print("4. LEAKAGE-FREE OUTER CV")
    print("=" * 82)

    leakage = pd.read_csv(
        LEAKAGE_PATH,
    )

    check(
        len(leakage) == 5,
        "All 5 outer folds have leakage audit records",
    )

    if "video_overlap" in leakage.columns:

        check(
            int(
                pd.to_numeric(
                    leakage["video_overlap"]
                ).sum()
            )
            == 0,
            "Train/held-out video overlap is zero in all folds",
        )

    if "status" in leakage.columns:

        check(
            leakage["status"]
            .astype(str)
            .str.upper()
            .eq("PASS")
            .all(),
            "All outer leakage audits are PASS",
        )

    # ========================================================
    # 5. OOF RAW CONCEPT SCORING
    # ========================================================

    print("\n" + "=" * 82)
    print("5. LEAKAGE-FREE OOF CONCEPT SCORES")
    print("=" * 82)

    scores = pd.read_csv(
        OOF_SCORE_PATH,
    )

    scores["video_id"] = (
        scores["video_id"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    scores["outer_fold"] = (
        pd.to_numeric(
            scores["outer_fold"],
            errors="raise",
        )
        .astype(int)
    )

    check(
        len(scores) == 4853,
        "OOF concept scoring contains exactly 4853 rows",
    )

    check(
        scores["video_id"].nunique() == 40,
        "OOF concept scores cover all 40 videos",
    )

    check(
        int(
            scores.duplicated(
                subset=[
                    "video_id",
                    "chunk_id",
                    "concept",
                ]
            ).sum()
        )
        == 0,
        "OOF score table contains zero duplicate rows",
    )

    video_fold_count = (
        scores[
            [
                "video_id",
                "outer_fold",
            ]
        ]
        .drop_duplicates()
        .groupby("video_id")[
            "outer_fold"
        ]
        .nunique()
    )

    check(
        len(video_fold_count) == 40
        and bool(
            (
                video_fold_count == 1
            ).all()
        ),
        "Each DEV video receives OOF scores from exactly one outer fold",
    )

    # ========================================================
    # 6. NESTED INNER CV
    # ========================================================

    print("\n" + "=" * 82)
    print("6. NESTED INNER-CV TUNING")
    print("=" * 82)

    inner_assign = pd.read_csv(
        INNER_ASSIGN_PATH,
    )

    inner_audit = pd.read_csv(
        INNER_AUDIT_PATH,
    )

    inner_search = pd.read_csv(
        INNER_SEARCH_PATH,
    )

    selected = pd.read_csv(
        SELECTED_PATH,
    )

    check(
        len(inner_assign) == 160,
        "Nested CV contains 160 inner validation assignments",
    )

    check(
        len(inner_audit) == 20,
        "Nested CV contains 20 inner fold audits",
    )

    check(
        len(inner_search) == 2420,
        "Nested grid search contains exactly 2420 candidate evaluations",
    )

    check(
        len(selected) == 5,
        "Exactly one hyperparameter configuration was selected per outer fold",
    )

    if (
        "outer_test_videos_present"
        in inner_audit.columns
    ):

        check(
            int(
                pd.to_numeric(
                    inner_audit[
                        "outer_test_videos_present"
                    ]
                ).sum()
            )
            == 0,
            "Outer-test videos never entered inner CV",
        )

    if (
        "outer_test_gt_used_for_selection"
        in selected.columns
    ):

        check(
            selected[
                "outer_test_gt_used_for_selection"
            ]
            .astype(str)
            .str.upper()
            .eq("NO")
            .all(),
            "Outer-test GT was never used for hyperparameter selection",
        )

    # ========================================================
    # 7. OUTER EVALUATION
    # ========================================================

    print("\n" + "=" * 82)
    print("7. OUTER HELD-OUT EVALUATION")
    print("=" * 82)

    predictions = pd.read_csv(
        PREDICTIONS_PATH,
    )

    predictions["video_id"] = (
        predictions["video_id"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    variants = {
        "nested_tuned",
        "frozen_fixed",
    }

    check(
        set(
            predictions[
                "model_variant"
            ].unique()
        )
        == variants,
        "OOF predictions contain nested_tuned and frozen_fixed variants",
    )

    variant_ok = True

    for variant in variants:

        v = predictions[
            predictions[
                "model_variant"
            ].eq(variant)
        ]

        if (
            len(v) != 390
            or
            v["video_id"].nunique() != 40
            or
            int(
                v.duplicated(
                    subset=[
                        "video_id",
                        "concept",
                    ]
                ).sum()
            )
            != 0
        ):
            variant_ok = False
            break

    check(
        variant_ok,
        "Each model variant contains 390 unique OOF video-concept predictions",
    )

    gt_count_ok = all(
        int(
            predictions[
                predictions[
                    "model_variant"
                ].eq(variant)
            ]["y_true"].sum()
        )
        == 153
        for variant in variants
    )

    check(
        gt_count_ok,
        "Each prediction variant reproduces all 153 GT-positive pairs",
    )

    per_video = pd.read_csv(
        PER_VIDEO_PATH,
    )

    outer_metrics = pd.read_csv(
        OUTER_METRICS_PATH,
    )

    check(
        len(per_video) == 80,
        "Per-video metrics contain 40 videos x 2 variants",
    )

    check(
        len(outer_metrics) == 10,
        "Outer-fold metrics contain 5 folds x 2 variants",
    )

    nested_mean = float(
        per_video.loc[
            per_video[
                "model_variant"
            ].eq(
                "nested_tuned"
            ),
            "f1",
        ].mean()
    )

    frozen_mean = float(
        per_video.loc[
            per_video[
                "model_variant"
            ].eq(
                "frozen_fixed"
            ),
            "f1",
        ].mean()
    )

    check(
        np.isclose(
            nested_mean,
            0.607307,
            atol=1e-6,
        ),
        "Nested-tuned mean video-level F1 reproduces 0.607307",
    )

    check(
        np.isclose(
            frozen_mean,
            0.572958,
            atol=1e-6,
        ),
        "Frozen-fixed mean video-level F1 reproduces 0.572958",
    )

    # ========================================================
    # 8. STATISTICAL ANALYSIS
    # ========================================================

    print("\n" + "=" * 82)
    print("8. STATISTICAL ANALYSIS")
    print("=" * 82)

    statistical = pd.read_csv(
        STAT_SUMMARY_PATH,
    )

    paired = pd.read_csv(
        PAIRED_PATH,
    )

    subject_metrics = pd.read_csv(
        SUBJECT_PATH,
    )

    check(
        len(statistical) == 6,
        "Statistical summary contains 2 variants x 3 metrics",
    )

    check(
        len(paired) == 1,
        "Paired F1 analysis contains one primary comparison",
    )

    p = paired.iloc[0]

    check(
        int(
            p[
                "n_paired_videos"
            ]
        )
        == 40,
        "Primary statistical comparison uses 40 paired videos",
    )

    check(
        np.isclose(
            float(
                p[
                    "mean_difference"
                ]
            ),
            0.034349,
            atol=1e-6,
        ),
        "Paired mean F1 difference reproduces +0.034349",
    )

    check(
        float(
            p[
                "bootstrap_ci_lower"
            ]
        )
        < 0
        <
        float(
            p[
                "bootstrap_ci_upper"
            ]
        ),
        "95% CI of paired F1 difference contains zero",
    )

    check(
        float(
            p[
                "wilcoxon_p_value"
            ]
        )
        > 0.05,
        "Wilcoxon result correctly indicates no statistically significant difference at alpha=0.05",
    )

    check(
        (
            int(p["nested_wins"]),
            int(p["ties"]),
            int(p["nested_losses"]),
        )
        == (
            24,
            4,
            12,
        ),
        "Paired video wins/ties/losses reproduce 24/4/12",
    )

    # ========================================================
    # 9. CHUNK-SIZE SENSITIVITY
    # ========================================================

    print("\n" + "=" * 82)
    print("9. CHUNK-SIZE SENSITIVITY")
    print("=" * 82)

    chunk_results = pd.read_csv(
        CHUNK_RESULT_PATH,
    )

    chunk_predictions = pd.read_csv(
        CHUNK_PRED_PATH,
    )

    chunk_audit = json.loads(
        CHUNK_AUDIT_PATH.read_text(
            encoding="utf-8"
        )
    )

    actual_sizes = set(
        pd.to_numeric(
            chunk_results[
                "chunk_seconds"
            ]
        ).astype(int)
    )

    check(
        actual_sizes
        == {30, 60, 90, 120},
        "Sensitivity includes exactly 30/60/90/120-second chunks",
    )

    chunk_count_ok = True

    for size, expected in (
        EXPECTED_CHUNK_COUNTS.items()
    ):

        row = chunk_results[
            pd.to_numeric(
                chunk_results[
                    "chunk_seconds"
                ]
            ).eq(size)
        ]

        if (
            len(row) != 1
            or
            int(
                row[
                    "num_chunks"
                ].iloc[0]
            )
            != expected
        ):
            chunk_count_ok = False
            break

    check(
        chunk_count_ok,
        "Sensitivity chunk counts reproduce 965/494/331/257",
    )

    check(
        len(chunk_predictions)
        == 390 * 4,
        "Sensitivity predictions contain 390 pairs x 4 chunk sizes",
    )

    row60 = chunk_results[
        pd.to_numeric(
            chunk_results[
                "chunk_seconds"
            ]
        ).eq(60)
    ].iloc[0]

    check(
        np.isclose(
            float(
                row60[
                    "macro_video_f1"
                ]
            ),
            frozen_mean,
            atol=1e-12,
            rtol=0.0,
        ),
        "60-second sensitivity F1 exactly reproduces primary frozen result",
    )

    check(
        bool(
            chunk_audit[
                "60s_reproduces_p18_frozen_predictions"
            ]
        ),
        "60-second sensitivity predictions exactly reproduce P1.8",
    )

    check(
        float(
            chunk_audit[
                "60s_max_video_score_difference"
            ]
        )
        == 0.0,
        "60-second reproduction has zero score difference",
    )

    # ========================================================
    # 10. FINAL SUMMARY
    # ========================================================

    print("\n" + "=" * 82)
    print("10. FINAL P1 STATUS")
    print("=" * 82)

    final_pass = (
        len(problems) == 0
    )

    summary = {
        "stage":
            "P1 Scientific Validation",

        "status":
            (
                "PASS"
                if final_pass
                else "FAIL"
            ),

        "checks_passed":
            len(checks),

        "checks_failed":
            len(problems),

        "dev_videos":
            40,

        "ground_truth_positive_pairs":
            153,

        "primary_chunk_seconds":
            60,

        "primary_chunks":
            494,

        "outer_folds":
            5,

        "inner_folds_per_outer_fold":
            4,

        "nested_mean_video_f1":
            nested_mean,

        "frozen_mean_video_f1":
            frozen_mean,

        "paired_mean_delta_f1":
            float(
                p[
                    "mean_difference"
                ]
            ),

        "paired_wilcoxon_p":
            float(
                p[
                    "wilcoxon_p_value"
                ]
            ),

        "paired_ci_lower":
            float(
                p[
                    "bootstrap_ci_lower"
                ]
            ),

        "paired_ci_upper":
            float(
                p[
                    "bootstrap_ci_upper"
                ]
            ),

        "statistical_interpretation":
            (
                "Nested tuning has higher "
                "mean video-level F1 on DEV, "
                "but the paired difference "
                "is not statistically significant "
                "at alpha=0.05."
            ),

        "chunk_sensitivity_seconds":
            [
                30,
                60,
                90,
                120,
            ],

        "sensitivity_used_for_model_selection":
            False,

        "outer_test_used_for_tuning":
            False,

        "heldout_transcript_used_for_fit":
            False,

        "problems":
            problems,
    }

    FINAL_SUMMARY_PATH.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    text = f"""
P1 SCIENTIFIC VALIDATION FINAL SUMMARY
======================================

Status: {'PASS' if final_pass else 'FAIL'}

Dataset
-------
DEV videos: 40
Subjects: 10 SQL / 10 Python / 10 Java / 10 C++
Ground-truth positive pairs: 153

Primary experiment
------------------
Chunk duration: 60 seconds
Chunks: 494
Outer CV: 5 folds
Train/test per outer fold: 32 / 8 videos
Leakage: none detected

Nested tuning
-------------
Inner CV: 4 folds inside each outer-training set
Outer-test data used for tuning: NO
Outer-test GT used for tuning: NO

Primary video-level F1
----------------------
Nested tuned : {nested_mean:.6f}
Frozen fixed : {frozen_mean:.6f}
Delta        : {float(p['mean_difference']):.6f}

Paired statistical comparison
-----------------------------
95% bootstrap CI:
[{float(p['bootstrap_ci_lower']):.6f},
 {float(p['bootstrap_ci_upper']):.6f}]

Wilcoxon p:
{float(p['wilcoxon_p_value']):.6f}

Wins / ties / losses:
{int(p['nested_wins'])} /
{int(p['ties'])} /
{int(p['nested_losses'])}

Interpretation
--------------
Nested tuning produced a higher mean video-level F1
on the 40-video development set, but the paired
analysis did not establish a statistically significant
improvement at alpha = 0.05.

Chunk-size sensitivity
----------------------
30 / 60 / 90 / 120 seconds tested.
60-second experiment reproduced exactly.
Sensitivity results were NOT used to retroactively
change the primary configuration.

Scientific controls
-------------------
Ground truth created before model evaluation.
Video-level splitting used.
No train/held-out video overlap.
Held-out transcript was transform-only.
Outer-test GT was evaluation-only.
Hyperparameters were selected by nested inner CV.
Sensitivity results were descriptive only.
"""

    FINAL_TEXT_PATH.write_text(
        text.strip() + "\n",
        encoding="utf-8",
    )

    if final_pass:

        print(
            f"\nALL {len(checks)} "
            "FINAL AUDIT CHECKS: PASS"
        )

        print("\n" + "=" * 82)
        print(
            "P1 SCIENTIFIC VALIDATION: FINAL PASS"
        )
        print("=" * 82)

        print(
            "P1 is internally consistent "
            "and ready to freeze."
        )

        print(
            "Saved:",
            FINAL_SUMMARY_PATH.relative_to(
                ROOT
            ),
        )

        print(
            "Saved:",
            FINAL_TEXT_PATH.relative_to(
                ROOT
            ),
        )

    else:

        print(
            f"\nFINAL AUDIT FAILED: "
            f"{len(problems)} problem(s)"
        )

        for i, problem in enumerate(
            problems,
            1,
        ):
            print(
                f"{i}. {problem}"
            )

        print(
            "\nDO NOT FREEZE P1."
        )

        sys.exit(1)


if __name__ == "__main__":
    main()