from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy import stats


# =========================================================
# STEP 11 - OLD vs NEW COMPARISON
# SCAFFOLD40 SUPPLEMENTARY DEV COMPARISON
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

COMPARE_ROOT = (
    PROJECT_ROOT
    / "module1"
    / "results"
    / "scaffold40_compare"
)

OLD_ROOT = (
    COMPARE_ROOT
    / "old"
)

NEW_ROOT = (
    COMPARE_ROOT
    / "new"
)


# =========================================================
# INPUT FILES
# =========================================================

OLD_METRICS_FILE = (
    OLD_ROOT
    / "metrics_summary.csv"
)

NEW_METRICS_FILE = (
    NEW_ROOT
    / "metrics_summary.csv"
)

OLD_VIDEO_FILE = (
    OLD_ROOT
    / "per_video_metrics.csv"
)

NEW_VIDEO_FILE = (
    NEW_ROOT
    / "per_video_metrics.csv"
)

OLD_SUBJECT_FILE = (
    OLD_ROOT
    / "per_subject_metrics.csv"
)

NEW_SUBJECT_FILE = (
    NEW_ROOT
    / "per_subject_metrics.csv"
)

OLD_EVAL_FILE = (
    OLD_ROOT
    / "evaluation_pairs.csv"
)

NEW_EVAL_FILE = (
    NEW_ROOT
    / "evaluation_pairs.csv"
)


# =========================================================
# OUTPUT FILES
# =========================================================

SUMMARY_FILE = (
    COMPARE_ROOT
    / "old_vs_new_summary.csv"
)

PER_VIDEO_FILE = (
    COMPARE_ROOT
    / "old_vs_new_per_video.csv"
)

PER_SUBJECT_FILE = (
    COMPARE_ROOT
    / "old_vs_new_per_subject.csv"
)

PAIRED_STATS_FILE = (
    COMPARE_ROOT
    / "old_vs_new_paired_stats.csv"
)

AUDIT_FILE = (
    COMPARE_ROOT
    / "comparison_audit.json"
)


# =========================================================
# HELPERS
# =========================================================

SUBJECT_ORDER = [
    "SQL",
    "Python",
    "Java",
    "C++",
]


def normalize_video_id(value):

    text = (
        str(value)
        .strip()
        .lower()
    )

    if (
        text.startswith("v")
        and text[1:].isdigit()
    ):

        return (
            f"v{int(text[1:])}"
        )

    return text


def video_number(video_id):

    text = normalize_video_id(
        video_id
    )

    if (
        text.startswith("v")
        and text[1:].isdigit()
    ):

        return int(
            text[1:]
        )

    return 999999


def safe_float(value):

    return float(
        value
    )


def paired_stratified_bootstrap_ci(
    paired,
    n_bootstrap=20000,
    random_state=42,
):

    """
    Paired bootstrap on per-video F1 differences.

    Sampling is stratified by subject:
    each bootstrap replicate keeps
    10 SQL + 10 Python + 10 Java + 10 C++ draws.

    Primary quantity:
        mean(new_f1 - old_f1)

    This is descriptive DEV uncertainty,
    not confirmatory TEST inference.
    """

    rng = np.random.default_rng(
        random_state
    )

    bootstrap_deltas = []


    for _ in range(
        n_bootstrap
    ):

        sampled_differences = []


        for subject in SUBJECT_ORDER:

            subject_values = (
                paired[
                    paired[
                        "subject"
                    ]
                    == subject
                ][
                    "delta_f1_new_minus_old"
                ]
                .to_numpy(
                    dtype=float
                )
            )


            if len(
                subject_values
            ) != 10:

                raise RuntimeError(
                    "Bootstrap expected exactly "
                    f"10 videos for {subject}, "
                    f"found {len(subject_values)}."
                )


            sampled = rng.choice(
                subject_values,
                size=len(
                    subject_values
                ),
                replace=True,
            )


            sampled_differences.extend(
                sampled.tolist()
            )


        bootstrap_deltas.append(
            float(
                np.mean(
                    sampled_differences
                )
            )
        )


    bootstrap_deltas = np.asarray(
        bootstrap_deltas,
        dtype=float,
    )


    lower = float(
        np.quantile(
            bootstrap_deltas,
            0.025,
        )
    )


    upper = float(
        np.quantile(
            bootstrap_deltas,
            0.975,
        )
    )


    return (
        lower,
        upper,
    )


def audit_variant_config(
    row,
    variant,
):

    if variant == "old":

        expected = {
            "chunk_seconds": 60,
            "lda_weight": 0.40,
            "lsa_weight": 0.60,
            "aggregation": "top2_mean_score",
            "threshold": 0.40,
        }

    elif variant == "new":

        expected = {
            "chunk_seconds": 180,
            "lda_weight": 0.00,
            "lsa_weight": 1.00,
            "aggregation": "max_final_score",
            "threshold": 0.50,
        }

    else:

        raise ValueError(
            f"Unknown variant: {variant}"
        )


    if int(
        row[
            "chunk_seconds"
        ]
    ) != expected[
        "chunk_seconds"
    ]:

        raise RuntimeError(
            f"{variant}: chunk_seconds mismatch."
        )


    if not np.isclose(
        float(
            row[
                "lda_weight"
            ]
        ),
        expected[
            "lda_weight"
        ],
        atol=1e-12,
        rtol=0.0,
    ):

        raise RuntimeError(
            f"{variant}: LDA weight mismatch."
        )


    if not np.isclose(
        float(
            row[
                "lsa_weight"
            ]
        ),
        expected[
            "lsa_weight"
        ],
        atol=1e-12,
        rtol=0.0,
    ):

        raise RuntimeError(
            f"{variant}: LSA weight mismatch."
        )


    if str(
        row[
            "aggregation"
        ]
    ) != expected[
        "aggregation"
    ]:

        raise RuntimeError(
            f"{variant}: aggregation mismatch."
        )


    if not np.isclose(
        float(
            row[
                "threshold"
            ]
        ),
        expected[
            "threshold"
        ],
        atol=1e-12,
        rtol=0.0,
    ):

        raise RuntimeError(
            f"{variant}: threshold mismatch."
        )


# =========================================================
# MAIN
# =========================================================

def main():

    print(
        "======================================"
    )

    print(
        "SCAFFOLD40 OLD vs NEW COMPARISON"
    )

    print(
        "STEP 11 - FINAL DEV COMPARISON"
    )

    print(
        "======================================"
    )


    # =====================================================
    # 1. CHECK FILES
    # =====================================================

    required_files = [
        OLD_METRICS_FILE,
        NEW_METRICS_FILE,
        OLD_VIDEO_FILE,
        NEW_VIDEO_FILE,
        OLD_SUBJECT_FILE,
        NEW_SUBJECT_FILE,
        OLD_EVAL_FILE,
        NEW_EVAL_FILE,
    ]


    for path in required_files:

        if not path.exists():

            raise FileNotFoundError(
                f"Missing required file: {path}"
            )


    # =====================================================
    # 2. READ DATA
    # =====================================================

    old_metrics = pd.read_csv(
        OLD_METRICS_FILE
    )

    new_metrics = pd.read_csv(
        NEW_METRICS_FILE
    )

    old_video = pd.read_csv(
        OLD_VIDEO_FILE
    )

    new_video = pd.read_csv(
        NEW_VIDEO_FILE
    )

    old_subject = pd.read_csv(
        OLD_SUBJECT_FILE
    )

    new_subject = pd.read_csv(
        NEW_SUBJECT_FILE
    )

    old_eval = pd.read_csv(
        OLD_EVAL_FILE
    )

    new_eval = pd.read_csv(
        NEW_EVAL_FILE
    )


    # =====================================================
    # 3. BASIC AUDIT
    # =====================================================

    if len(
        old_metrics
    ) != 1:

        raise RuntimeError(
            "OLD metrics summary must have 1 row."
        )


    if len(
        new_metrics
    ) != 1:

        raise RuntimeError(
            "NEW metrics summary must have 1 row."
        )


    old_row = (
        old_metrics
        .iloc[0]
    )

    new_row = (
        new_metrics
        .iloc[0]
    )


    audit_variant_config(
        old_row,
        "old",
    )

    audit_variant_config(
        new_row,
        "new",
    )


    # -----------------------------------------------------
    # EXACT 40 VIDEOS
    # -----------------------------------------------------

    expected_ids = {
        f"v{i}"
        for i in range(1, 41)
    }


    for df in [
        old_video,
        new_video,
        old_eval,
        new_eval,
    ]:

        df[
            "video_id"
        ] = (
            df[
                "video_id"
            ]
            .map(
                normalize_video_id
            )
        )


    old_ids = set(
        old_video[
            "video_id"
        ]
    )

    new_ids = set(
        new_video[
            "video_id"
        ]
    )


    if old_ids != expected_ids:

        raise RuntimeError(
            "OLD per-video data is not exactly v1-v40."
        )


    if new_ids != expected_ids:

        raise RuntimeError(
            "NEW per-video data is not exactly v1-v40."
        )


    # -----------------------------------------------------
    # 390 EVALUATION PAIRS EACH
    # -----------------------------------------------------

    if len(
        old_eval
    ) != 390:

        raise RuntimeError(
            "OLD evaluation must contain 390 pairs."
        )


    if len(
        new_eval
    ) != 390:

        raise RuntimeError(
            "NEW evaluation must contain 390 pairs."
        )


    # -----------------------------------------------------
    # SAME VIDEO-CONCEPT EVALUATION SPACE
    # -----------------------------------------------------

    old_eval[
        "_concept_key"
    ] = (
        old_eval[
            "concept"
        ]
        .astype(str)
        .str.strip()
        .str.lower()
    )


    new_eval[
        "_concept_key"
    ] = (
        new_eval[
            "concept"
        ]
        .astype(str)
        .str.strip()
        .str.lower()
    )


    old_pairs = set(
        zip(
            old_eval[
                "video_id"
            ],
            old_eval[
                "_concept_key"
            ],
        )
    )


    new_pairs = set(
        zip(
            new_eval[
                "video_id"
            ],
            new_eval[
                "_concept_key"
            ],
        )
    )


    if (
        old_pairs
        != new_pairs
    ):

        raise RuntimeError(
            "OLD and NEW do not use the same "
            "390 video-concept evaluation space."
        )


    # -----------------------------------------------------
    # SAME Y_TRUE / SAME GT
    # -----------------------------------------------------

    old_truth = (
        old_eval[
            [
                "video_id",
                "_concept_key",
                "y_true",
            ]
        ]
        .rename(
            columns={
                "y_true":
                    "old_y_true"
            }
        )
    )


    new_truth = (
        new_eval[
            [
                "video_id",
                "_concept_key",
                "y_true",
            ]
        ]
        .rename(
            columns={
                "y_true":
                    "new_y_true"
            }
        )
    )


    truth_check = pd.merge(
        old_truth,
        new_truth,
        on=[
            "video_id",
            "_concept_key",
        ],
        how="inner",
        validate="one_to_one",
    )


    truth_mismatch = int(
        (
            truth_check[
                "old_y_true"
            ]
            !=
            truth_check[
                "new_y_true"
            ]
        )
        .sum()
    )


    if truth_mismatch != 0:

        raise RuntimeError(
            "OLD and NEW do not use identical GT."
        )


    if int(
        truth_check[
            "old_y_true"
        ].sum()
    ) != 153:

        raise RuntimeError(
            "Expected exactly 153 positive GT pairs."
        )


    # =====================================================
    # 4. PAIRED PER-VIDEO TABLE
    # =====================================================

    old_video_selected = (
        old_video[
            [
                "video_id",
                "subject",
                "candidate_concepts",
                "gt_positive",
                "predicted_positive",
                "tp",
                "fp",
                "fn",
                "tn",
                "precision",
                "recall",
                "f1",
                "accuracy",
            ]
        ]
        .rename(
            columns={
                "candidate_concepts":
                    "old_candidate_concepts",

                "gt_positive":
                    "old_gt_positive",

                "predicted_positive":
                    "old_predicted_positive",

                "tp":
                    "old_tp",

                "fp":
                    "old_fp",

                "fn":
                    "old_fn",

                "tn":
                    "old_tn",

                "precision":
                    "old_precision",

                "recall":
                    "old_recall",

                "f1":
                    "old_f1",

                "accuracy":
                    "old_accuracy",
            }
        )
    )


    new_video_selected = (
        new_video[
            [
                "video_id",
                "subject",
                "candidate_concepts",
                "gt_positive",
                "predicted_positive",
                "tp",
                "fp",
                "fn",
                "tn",
                "precision",
                "recall",
                "f1",
                "accuracy",
            ]
        ]
        .rename(
            columns={
                "subject":
                    "new_subject",

                "candidate_concepts":
                    "new_candidate_concepts",

                "gt_positive":
                    "new_gt_positive",

                "predicted_positive":
                    "new_predicted_positive",

                "tp":
                    "new_tp",

                "fp":
                    "new_fp",

                "fn":
                    "new_fn",

                "tn":
                    "new_tn",

                "precision":
                    "new_precision",

                "recall":
                    "new_recall",

                "f1":
                    "new_f1",

                "accuracy":
                    "new_accuracy",
            }
        )
    )


    paired = pd.merge(
        old_video_selected,
        new_video_selected,
        on="video_id",
        how="inner",
        validate="one_to_one",
    )


    if len(
        paired
    ) != 40:

        raise RuntimeError(
            "Paired comparison must contain 40 videos."
        )


    if (
        paired[
            "subject"
        ]
        !=
        paired[
            "new_subject"
        ]
    ).any():

        raise RuntimeError(
            "Subject mismatch between OLD and NEW."
        )


    paired = paired.drop(
        columns=[
            "new_subject"
        ]
    )


    if (
        paired[
            "old_gt_positive"
        ]
        !=
        paired[
            "new_gt_positive"
        ]
    ).any():

        raise RuntimeError(
            "Per-video GT count mismatch."
        )


    if (
        paired[
            "old_candidate_concepts"
        ]
        !=
        paired[
            "new_candidate_concepts"
        ]
    ).any():

        raise RuntimeError(
            "Per-video candidate concept space mismatch."
        )


    # -----------------------------------------------------
    # DELTAS = NEW - OLD
    # -----------------------------------------------------

    paired[
        "delta_precision_new_minus_old"
    ] = (
        paired[
            "new_precision"
        ]
        -
        paired[
            "old_precision"
        ]
    )


    paired[
        "delta_recall_new_minus_old"
    ] = (
        paired[
            "new_recall"
        ]
        -
        paired[
            "old_recall"
        ]
    )


    paired[
        "delta_f1_new_minus_old"
    ] = (
        paired[
            "new_f1"
        ]
        -
        paired[
            "old_f1"
        ]
    )


    paired[
        "delta_fp_new_minus_old"
    ] = (
        paired[
            "new_fp"
        ]
        -
        paired[
            "old_fp"
        ]
    )


    paired[
        "delta_fn_new_minus_old"
    ] = (
        paired[
            "new_fn"
        ]
        -
        paired[
            "old_fn"
        ]
    )


    # =====================================================
    # 5. VERIFY MACRO F1
    # =====================================================

    old_macro_recomputed = float(
        paired[
            "old_f1"
        ].mean()
    )


    new_macro_recomputed = float(
        paired[
            "new_f1"
        ].mean()
    )


    old_macro_summary = safe_float(
        old_row[
            "macro_video_f1"
        ]
    )


    new_macro_summary = safe_float(
        new_row[
            "macro_video_f1"
        ]
    )


    if not np.isclose(
        old_macro_recomputed,
        old_macro_summary,
        atol=1e-12,
        rtol=0.0,
    ):

        raise RuntimeError(
            "OLD macro video F1 does not reproduce."
        )


    if not np.isclose(
        new_macro_recomputed,
        new_macro_summary,
        atol=1e-12,
        rtol=0.0,
    ):

        raise RuntimeError(
            "NEW macro video F1 does not reproduce."
        )


    delta_macro = (
        new_macro_summary
        -
        old_macro_summary
    )


    # =====================================================
    # 6. PAIRED BOOTSTRAP CI
    # =====================================================

    (
        ci_lower,
        ci_upper,
    ) = paired_stratified_bootstrap_ci(
        paired,
        n_bootstrap=20000,
        random_state=42,
    )


    # =====================================================
    # 7. WILCOXON
    # =====================================================

    old_f1 = (
        paired[
            "old_f1"
        ]
        .to_numpy(
            dtype=float
        )
    )


    new_f1 = (
        paired[
            "new_f1"
        ]
        .to_numpy(
            dtype=float
        )
    )


    differences = (
        new_f1
        -
        old_f1
    )


    if np.allclose(
        differences,
        0.0,
    ):

        wilcoxon_stat = 0.0
        wilcoxon_p = 1.0

    else:

        wilcoxon_result = (
            stats.wilcoxon(
                new_f1,
                old_f1,
                alternative="two-sided",
                zero_method="wilcox",
            )
        )

        wilcoxon_stat = float(
            wilcoxon_result.statistic
        )

        wilcoxon_p = float(
            wilcoxon_result.pvalue
        )


    # =====================================================
    # 8. PAIRED T-TEST
    # =====================================================

    t_result = (
        stats.ttest_rel(
            new_f1,
            old_f1,
            nan_policy="raise",
        )
    )


    paired_t_stat = float(
        t_result.statistic
    )


    paired_t_p = float(
        t_result.pvalue
    )


    # =====================================================
    # 9. COHEN dz
    # =====================================================

    difference_mean = float(
        np.mean(
            differences
        )
    )


    difference_std = float(
        np.std(
            differences,
            ddof=1,
        )
    )


    if difference_std == 0:

        cohen_dz = float(
            "nan"
        )

    else:

        cohen_dz = (
            difference_mean
            /
            difference_std
        )


    # =====================================================
    # 10. VIDEO COUNTS IMPROVED / WORSE / TIED
    # =====================================================

    tolerance = 1e-12


    improved_videos = int(
        (
            paired[
                "delta_f1_new_minus_old"
            ]
            > tolerance
        )
        .sum()
    )


    worse_videos = int(
        (
            paired[
                "delta_f1_new_minus_old"
            ]
            < -tolerance
        )
        .sum()
    )


    tied_videos = int(
        40
        -
        improved_videos
        -
        worse_videos
    )


    # =====================================================
    # 11. PER-SUBJECT COMPARISON
    # =====================================================

    old_subject_selected = (
        old_subject[
            [
                "subject",
                "videos",
                "evaluation_pairs",
                "gt_positive",
                "predicted_positive",
                "micro_precision",
                "micro_recall",
                "micro_f1",
                "macro_video_precision",
                "macro_video_recall",
                "macro_video_f1",
            ]
        ]
        .rename(
            columns={
                column:
                    f"old_{column}"

                for column
                in old_subject.columns

                if column
                != "subject"
            }
        )
    )


    new_subject_selected = (
        new_subject[
            [
                "subject",
                "videos",
                "evaluation_pairs",
                "gt_positive",
                "predicted_positive",
                "micro_precision",
                "micro_recall",
                "micro_f1",
                "macro_video_precision",
                "macro_video_recall",
                "macro_video_f1",
            ]
        ]
        .rename(
            columns={
                column:
                    f"new_{column}"

                for column
                in new_subject.columns

                if column
                != "subject"
            }
        )
    )


    per_subject = pd.merge(
        old_subject_selected,
        new_subject_selected,
        on="subject",
        how="inner",
        validate="one_to_one",
    )


    if len(
        per_subject
    ) != 4:

        raise RuntimeError(
            "Expected 4 paired subject rows."
        )


    per_subject[
        "delta_macro_video_f1_new_minus_old"
    ] = (
        per_subject[
            "new_macro_video_f1"
        ]
        -
        per_subject[
            "old_macro_video_f1"
        ]
    )


    per_subject[
        "delta_micro_f1_new_minus_old"
    ] = (
        per_subject[
            "new_micro_f1"
        ]
        -
        per_subject[
            "old_micro_f1"
        ]
    )


    # =====================================================
    # 12. WINNER
    # =====================================================

    old_micro_f1 = safe_float(
        old_row[
            "micro_f1"
        ]
    )


    new_micro_f1 = safe_float(
        new_row[
            "micro_f1"
        ]
    )


    if (
        new_macro_summary
        >
        old_macro_summary
        + tolerance
    ):

        winner = "new"

        winner_reason = (
            "NEW has higher primary metric "
            "macro video F1."
        )

    elif (
        old_macro_summary
        >
        new_macro_summary
        + tolerance
    ):

        winner = "old"

        winner_reason = (
            "OLD has higher primary metric "
            "macro video F1."
        )

    else:

        # Secondary criterion only on exact macro tie.
        if (
            new_micro_f1
            >
            old_micro_f1
            + tolerance
        ):

            winner = "new"

            winner_reason = (
                "Macro video F1 tied; "
                "NEW has higher secondary micro F1."
            )

        elif (
            old_micro_f1
            >
            new_micro_f1
            + tolerance
        ):

            winner = "old"

            winner_reason = (
                "Macro video F1 tied; "
                "OLD has higher secondary micro F1."
            )

        else:

            winner = "tie"

            winner_reason = (
                "Macro and micro F1 are tied."
            )


    # =====================================================
    # 13. SUMMARY
    # =====================================================

    summary = pd.DataFrame(
        [
            {
                "old_chunk_seconds":
                    int(
                        old_row[
                            "chunk_seconds"
                        ]
                    ),

                "new_chunk_seconds":
                    int(
                        new_row[
                            "chunk_seconds"
                        ]
                    ),

                "old_macro_video_f1":
                    old_macro_summary,

                "new_macro_video_f1":
                    new_macro_summary,

                "delta_macro_video_f1_new_minus_old":
                    delta_macro,

                "old_micro_precision":
                    safe_float(
                        old_row[
                            "micro_precision"
                        ]
                    ),

                "new_micro_precision":
                    safe_float(
                        new_row[
                            "micro_precision"
                        ]
                    ),

                "delta_micro_precision_new_minus_old":
                    safe_float(
                        new_row[
                            "micro_precision"
                        ]
                    )
                    -
                    safe_float(
                        old_row[
                            "micro_precision"
                        ]
                    ),

                "old_micro_recall":
                    safe_float(
                        old_row[
                            "micro_recall"
                        ]
                    ),

                "new_micro_recall":
                    safe_float(
                        new_row[
                            "micro_recall"
                        ]
                    ),

                "delta_micro_recall_new_minus_old":
                    safe_float(
                        new_row[
                            "micro_recall"
                        ]
                    )
                    -
                    safe_float(
                        old_row[
                            "micro_recall"
                        ]
                    ),

                "old_micro_f1":
                    old_micro_f1,

                "new_micro_f1":
                    new_micro_f1,

                "delta_micro_f1_new_minus_old":
                    new_micro_f1
                    -
                    old_micro_f1,

                "old_tp":
                    int(
                        old_row[
                            "tp"
                        ]
                    ),

                "new_tp":
                    int(
                        new_row[
                            "tp"
                        ]
                    ),

                "old_fp":
                    int(
                        old_row[
                            "fp"
                        ]
                    ),

                "new_fp":
                    int(
                        new_row[
                            "fp"
                        ]
                    ),

                "old_fn":
                    int(
                        old_row[
                            "fn"
                        ]
                    ),

                "new_fn":
                    int(
                        new_row[
                            "fn"
                        ]
                    ),

                "videos_improved_f1":
                    improved_videos,

                "videos_worse_f1":
                    worse_videos,

                "videos_tied_f1":
                    tied_videos,

                "winner":
                    winner,

                "winner_primary_metric":
                    "macro_video_f1",

                "winner_reason":
                    winner_reason,

                "comparison_scope":
                    (
                        "supplementary DEV "
                        "whole-corpus scaffold comparison"
                    ),

                "confirmatory_test":
                    False,
            }
        ]
    )


    # =====================================================
    # 14. PAIRED STATISTICS TABLE
    # =====================================================

    paired_stats = pd.DataFrame(
        [
            {
                "n_paired_videos":
                    40,

                "mean_old_f1":
                    old_macro_summary,

                "mean_new_f1":
                    new_macro_summary,

                "mean_delta_new_minus_old":
                    difference_mean,

                "bootstrap_method":
                    (
                        "paired stratified by subject"
                    ),

                "bootstrap_replicates":
                    20000,

                "bootstrap_seed":
                    42,

                "delta_95_ci_lower":
                    ci_lower,

                "delta_95_ci_upper":
                    ci_upper,

                "wilcoxon_statistic":
                    wilcoxon_stat,

                "wilcoxon_p_two_sided":
                    wilcoxon_p,

                "paired_t_statistic":
                    paired_t_stat,

                "paired_t_p_two_sided":
                    paired_t_p,

                "cohen_dz":
                    cohen_dz,

                "selection_rule":
                    (
                        "winner chosen by macro video F1; "
                        "p-values are descriptive only"
                    ),
            }
        ]
    )


    # =====================================================
    # 15. DETERMINISTIC SORT
    # =====================================================

    paired[
        "_video_number"
    ] = (
        paired[
            "video_id"
        ]
        .map(
            video_number
        )
    )


    paired = (
        paired
        .sort_values(
            "_video_number"
        )
        .drop(
            columns=[
                "_video_number"
            ]
        )
        .reset_index(
            drop=True
        )
    )


    subject_rank = {
        subject:
            index

        for index, subject
        in enumerate(
            SUBJECT_ORDER
        )
    }


    per_subject[
        "_subject_rank"
    ] = (
        per_subject[
            "subject"
        ]
        .map(
            subject_rank
        )
    )


    per_subject = (
        per_subject
        .sort_values(
            "_subject_rank"
        )
        .drop(
            columns=[
                "_subject_rank"
            ]
        )
        .reset_index(
            drop=True
        )
    )


    # =====================================================
    # 16. SAVE CSV OUTPUTS
    # =====================================================

    summary.to_csv(
        SUMMARY_FILE,
        index=False,
        encoding="utf-8-sig",
    )


    paired.to_csv(
        PER_VIDEO_FILE,
        index=False,
        encoding="utf-8-sig",
    )


    per_subject.to_csv(
        PER_SUBJECT_FILE,
        index=False,
        encoding="utf-8-sig",
    )


    paired_stats.to_csv(
        PAIRED_STATS_FILE,
        index=False,
        encoding="utf-8-sig",
    )


    # =====================================================
    # 17. AUDIT JSON
    # =====================================================

    audit = {
        "stage":
            "Scaffold40 OLD vs NEW comparison",

        "status":
            "PASS",

        "scope":
            (
                "supplementary DEV "
                "whole-corpus scaffold comparison"
            ),

        "same_40_dev_videos":
            True,

        "same_390_video_concept_pairs":
            True,

        "same_ground_truth":
            True,

        "positive_gt_pairs":
            153,

        "final_test_used":
            False,

        "new_hyperparameter_search_performed":
            False,

        "winner_selection_metric":
            "macro_video_f1",

        "p_value_used_for_selection":
            False,

        "old_config": {
            "chunk_seconds":
                60,

            "lda_weight":
                0.40,

            "lsa_weight":
                0.60,

            "aggregation":
                "top2_mean_score",

            "threshold":
                0.40,
        },

        "new_config": {
            "chunk_seconds":
                180,

            "lda_weight":
                0.00,

            "lsa_weight":
                1.00,

            "aggregation":
                "max_final_score",

            "threshold":
                0.50,
        },

        "results": {
            "old_macro_video_f1":
                old_macro_summary,

            "new_macro_video_f1":
                new_macro_summary,

            "delta_macro_video_f1":
                delta_macro,

            "old_micro_f1":
                old_micro_f1,

            "new_micro_f1":
                new_micro_f1,

            "delta_micro_f1":
                new_micro_f1
                -
                old_micro_f1,

            "bootstrap_ci":
                [
                    ci_lower,
                    ci_upper,
                ],

            "wilcoxon_p":
                wilcoxon_p,

            "paired_t_p":
                paired_t_p,

            "cohen_dz":
                cohen_dz,
        },

        "winner":
            winner,

        "winner_reason":
            winner_reason,

        "interpretation": (
            "Development/supplementary comparison only. "
            "The scaffold fits representations using the "
            "whole DEV corpus and is not an unbiased "
            "generalization estimate. "
            "P1 leakage-free validation and locked final "
            "configuration remain authoritative."
        ),
    }


    with open(
        AUDIT_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            audit,
            file,
            indent=2,
            ensure_ascii=False,
        )


    # =====================================================
    # 18. PRINT FINAL COMPARISON
    # =====================================================

    print(
        "\n======================================"
    )

    print(
        "OLD vs NEW SUMMARY"
    )

    print(
        "======================================"
    )


    print(
        f"OLD macro video F1 : "
        f"{old_macro_summary:.6f}"
    )

    print(
        f"NEW macro video F1 : "
        f"{new_macro_summary:.6f}"
    )

    print(
        f"Delta NEW - OLD    : "
        f"{delta_macro:+.6f}"
    )


    print(
        "\nOLD micro F1       : "
        f"{old_micro_f1:.6f}"
    )

    print(
        "NEW micro F1       : "
        f"{new_micro_f1:.6f}"
    )

    print(
        "Delta micro F1     : "
        f"{new_micro_f1 - old_micro_f1:+.6f}"
    )


    print(
        "\nPaired bootstrap 95% CI:"
    )

    print(
        f"[{ci_lower:.6f}, "
        f"{ci_upper:.6f}]"
    )


    print(
        "\nWilcoxon p:"
    )

    print(
        f"{wilcoxon_p:.6f}"
    )


    print(
        "Paired t-test p:"
    )

    print(
        f"{paired_t_p:.6f}"
    )


    print(
        "Cohen dz:"
    )

    print(
        f"{cohen_dz:.6f}"
    )


    print(
        "\nVideos improved:",
        improved_videos
    )

    print(
        "Videos worse:",
        worse_videos
    )

    print(
        "Videos tied:",
        tied_videos
    )


    print(
        "\n======================================"
    )

    print(
        "WINNER:",
        winner.upper()
    )

    print(
        winner_reason
    )

    print(
        "======================================"
    )


    print(
        "\nIMPORTANT:"
    )

    print(
        "This is a supplementary DEV "
        "whole-corpus scaffold comparison."
    )

    print(
        "Do NOT describe these scores as "
        "unbiased final generalization performance."
    )

    print(
        "P-values are descriptive and were NOT "
        "used to select the winner."
    )


    print(
        "\n======================================"
    )

    print(
        "STEP 11 PASS"
    )

    print(
        "======================================"
    )


    print(
        "Summary:"
    )

    print(
        SUMMARY_FILE
    )


    print(
        "\nPer-video:"
    )

    print(
        PER_VIDEO_FILE
    )


    print(
        "\nPer-subject:"
    )

    print(
        PER_SUBJECT_FILE
    )


    print(
        "\nPaired statistics:"
    )

    print(
        PAIRED_STATS_FILE
    )


    print(
        "\nAudit:"
    )

    print(
        AUDIT_FILE
    )


    print(
        "======================================"
    )


if __name__ == "__main__":

    main()