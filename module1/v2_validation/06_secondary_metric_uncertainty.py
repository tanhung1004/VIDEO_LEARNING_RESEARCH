from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    average_precision_score,
    roc_auc_score,
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

STEP13_ROOT = (
    PROJECT_ROOT
    / "module1"
    / "results"
    / "ocr_best_integration"
    / "combined_system_comparison"
    / "final_evaluation"
)

PAIR_FILE = (
    STEP13_ROOT
    / "old_vs_new_combined_pair_scores.csv"
)

VIDEO_FILE = (
    STEP13_ROOT
    / "old_vs_new_combined_video_metrics.csv"
)

OVERALL_FILE = (
    STEP13_ROOT
    / "old_vs_new_combined_overall_metrics.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "module1"
    / "results"
    / "v2_validation"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "V2_SECONDARY_METRIC_UNCERTAINTY.csv"
)


# ============================================================
# CONSTANTS
# ============================================================

EXPECTED_VIDEOS = 40
EXPECTED_PAIRS = 390

BOOTSTRAP_ITERATIONS = 10000
BOOTSTRAP_RANDOM_STATE = 42

CI_LOW = 2.5
CI_HIGH = 97.5

TOL = 1e-12


# ============================================================
# HELPERS
# ============================================================

def safe_divide(
    numerator,
    denominator,
):

    if denominator == 0:
        return 0.0

    return float(
        numerator
        / denominator
    )


def confusion_values(
    y_true,
    y_pred,
):

    y_true = np.asarray(
        y_true,
        dtype=int,
    )

    y_pred = np.asarray(
        y_pred,
        dtype=int,
    )

    tp = int(
        np.sum(
            (y_true == 1)
            &
            (y_pred == 1)
        )
    )

    fp = int(
        np.sum(
            (y_true == 0)
            &
            (y_pred == 1)
        )
    )

    fn = int(
        np.sum(
            (y_true == 1)
            &
            (y_pred == 0)
        )
    )

    tn = int(
        np.sum(
            (y_true == 0)
            &
            (y_pred == 0)
        )
    )

    return (
        tn,
        fp,
        fn,
        tp,
    )


def compute_pair_metrics(
    y_true,
    y_pred,
    scores,
):

    y_true = np.asarray(
        y_true,
        dtype=int,
    )

    y_pred = np.asarray(
        y_pred,
        dtype=int,
    )

    scores = np.asarray(
        scores,
        dtype=float,
    )

    (
        tn,
        fp,
        fn,
        tp,
    ) = confusion_values(
        y_true,
        y_pred,
    )

    precision = float(
        precision_score(
            y_true,
            y_pred,
            zero_division=0,
        )
    )

    recall = float(
        recall_score(
            y_true,
            y_pred,
            zero_division=0,
        )
    )

    f1 = float(
        f1_score(
            y_true,
            y_pred,
            zero_division=0,
        )
    )

    specificity = (
        safe_divide(
            tn,
            tn + fp,
        )
    )

    fpr = (
        safe_divide(
            fp,
            fp + tn,
        )
    )

    balanced_accuracy = (
        recall
        + specificity
    ) / 2.0

    mcc = float(
        matthews_corrcoef(
            y_true,
            y_pred,
        )
    )

    unique_true = np.unique(
        y_true
    )

    if len(unique_true) >= 2:

        pr_auc = float(
            average_precision_score(
                y_true,
                scores,
            )
        )

        roc_auc = float(
            roc_auc_score(
                y_true,
                scores,
            )
        )

    else:

        pr_auc = np.nan
        roc_auc = np.nan

    return {
        "micro_precision":
            precision,

        "micro_recall":
            recall,

        "micro_f1":
            f1,

        "mcc":
            mcc,

        "balanced_accuracy":
            balanced_accuracy,

        "specificity":
            specificity,

        "fpr":
            fpr,

        "pr_auc":
            pr_auc,

        "roc_auc":
            roc_auc,
    }


# ============================================================
# LOAD + AUDIT
# ============================================================

def load_inputs():

    pairs = pd.read_csv(
        PAIR_FILE
    )

    video_metrics = pd.read_csv(
        VIDEO_FILE
    )

    overall = pd.read_csv(
        OVERALL_FILE
    )

    return (
        pairs,
        video_metrics,
        overall,
    )


def audit_inputs(
    pairs,
    video_metrics,
    overall,
):

    required_pair_columns = {
        "video_id",
        "subject",
        "y_true",
        "score_OLD",
        "score_NEW",
        "pred_OLD",
        "pred_NEW",
    }

    required_video_columns = {
        "video_id",
        "subject",
        "OLD_precision",
        "OLD_recall",
        "OLD_f1",
        "NEW_precision",
        "NEW_recall",
        "NEW_f1",
    }

    missing_pair = (
        required_pair_columns
        - set(pairs.columns)
    )

    missing_video = (
        required_video_columns
        - set(video_metrics.columns)
    )

    if missing_pair:
        raise RuntimeError(
            "Missing pair columns: "
            f"{sorted(missing_pair)}"
        )

    if missing_video:
        raise RuntimeError(
            "Missing video columns: "
            f"{sorted(missing_video)}"
        )

    if len(pairs) != EXPECTED_PAIRS:
        raise RuntimeError(
            "Unexpected pair count: "
            f"{len(pairs)}"
        )

    if len(video_metrics) != EXPECTED_VIDEOS:
        raise RuntimeError(
            "Unexpected video metric rows: "
            f"{len(video_metrics)}"
        )

    if (
        video_metrics["video_id"]
        .nunique()
        != EXPECTED_VIDEOS
    ):
        raise RuntimeError(
            "Expected exactly 40 unique "
            "video IDs."
        )

    subject_counts = (
        video_metrics[
            "subject"
        ]
        .value_counts()
    )

    if (
        len(subject_counts) != 4
        or not (
            subject_counts == 10
        ).all()
    ):
        raise RuntimeError(
            "Expected 10 videos per subject. "
            f"Observed: "
            f"{subject_counts.to_dict()}"
        )

    if len(overall) != 2:
        raise RuntimeError(
            "Overall metrics must contain "
            "exactly two system rows."
        )

    systems = set(
        overall["system"]
    )

    expected_systems = {
        "OLD_COMBINED",
        "NEW_COMBINED_F",
    }

    if systems != expected_systems:
        raise RuntimeError(
            "Unexpected overall system names: "
            f"{systems}"
        )

    print(
        "Input audit: PASS"
    )

    print(
        "DEV videos:",
        len(video_metrics),
    )

    print(
        "Video-concept pairs:",
        len(pairs),
    )

    print(
        "Subject counts:",
        subject_counts.to_dict(),
    )


# ============================================================
# OBSERVED METRICS
# ============================================================

def get_overall_rows(
    overall,
):

    old_row = (
        overall[
            overall["system"]
            .eq(
                "OLD_COMBINED"
            )
        ]
        .iloc[0]
    )

    new_row = (
        overall[
            overall["system"]
            .eq(
                "NEW_COMBINED_F"
            )
        ]
        .iloc[0]
    )

    return (
        old_row,
        new_row,
    )


# ============================================================
# SUBJECT-STRATIFIED VIDEO RESAMPLING
# ============================================================

def build_video_subject_groups(
    video_metrics,
):

    return {
        subject:
            group.reset_index(
                drop=True
            )

        for subject, group
        in video_metrics.groupby(
            "subject",
            sort=True,
        )
    }


def build_pair_clusters(
    pairs,
):

    clusters = {}

    for (
        subject,
        video_id,
    ), group in pairs.groupby(
        [
            "subject",
            "video_id",
        ],
        sort=True,
    ):

        clusters[
            (
                subject,
                video_id,
            )
        ] = (
            group.reset_index(
                drop=True
            )
        )

    return clusters


def get_subject_video_ids(
    pairs,
):

    result = {}

    for subject, group in (
        pairs.groupby(
            "subject",
            sort=True,
        )
    ):

        result[
            subject
        ] = sorted(
            group[
                "video_id"
            ].unique()
        )

    return result


# ============================================================
# BOOTSTRAP SECONDARY METRICS
# ============================================================

def run_bootstrap(
    pairs,
    video_metrics,
):

    rng = np.random.default_rng(
        BOOTSTRAP_RANDOM_STATE
    )

    video_groups = (
        build_video_subject_groups(
            video_metrics
        )
    )

    pair_clusters = (
        build_pair_clusters(
            pairs
        )
    )

    subject_video_ids = (
        get_subject_video_ids(
            pairs
        )
    )

    metric_names = [
        "macro_video_precision",
        "macro_video_recall",
        "macro_video_f1_std",
        "micro_precision",
        "micro_recall",
        "micro_f1",
        "mcc",
        "balanced_accuracy",
        "specificity",
        "fpr",
        "pr_auc",
        "roc_auc",
    ]

    bootstrap_deltas = {
        metric: []
        for metric
        in metric_names
    }

    subjects = sorted(
        subject_video_ids.keys()
    )

    for _ in range(
        BOOTSTRAP_ITERATIONS
    ):

        sampled_video_rows = []
        sampled_pair_blocks = []

        for subject in subjects:

            video_group = (
                video_groups[
                    subject
                ]
            )

            video_ids = (
                subject_video_ids[
                    subject
                ]
            )

            if (
                len(video_group)
                != len(video_ids)
            ):
                raise RuntimeError(
                    "Video/pair subject counts "
                    "do not match."
                )

            draw_indices = (
                rng.integers(
                    low=0,
                    high=len(video_ids),
                    size=len(video_ids),
                )
            )

            for draw_number, idx in enumerate(
                draw_indices
            ):

                video_id = (
                    video_ids[
                        idx
                    ]
                )

                video_row = (
                    video_group[
                        video_group[
                            "video_id"
                        ].eq(
                            video_id
                        )
                    ]
                )

                if len(video_row) != 1:
                    raise RuntimeError(
                        "Expected exactly one "
                        "video metric row."
                    )

                sampled_video_rows.append(
                    video_row.copy()
                )

                pair_block = (
                    pair_clusters[
                        (
                            subject,
                            video_id,
                        )
                    ]
                    .copy()
                )

                pair_block[
                    "_bootstrap_cluster"
                ] = (
                    f"{subject}_"
                    f"{draw_number}"
                )

                sampled_pair_blocks.append(
                    pair_block
                )

        sampled_videos = pd.concat(
            sampled_video_rows,
            ignore_index=True,
        )

        sampled_pairs = pd.concat(
            sampled_pair_blocks,
            ignore_index=True,
        )

        # ----------------------------------------------------
        # Macro video-level metrics
        # ----------------------------------------------------

        old_macro_precision = float(
            sampled_videos[
                "OLD_precision"
            ].mean()
        )

        new_macro_precision = float(
            sampled_videos[
                "NEW_precision"
            ].mean()
        )

        old_macro_recall = float(
            sampled_videos[
                "OLD_recall"
            ].mean()
        )

        new_macro_recall = float(
            sampled_videos[
                "NEW_recall"
            ].mean()
        )

        old_f1_std = float(
            sampled_videos[
                "OLD_f1"
            ].std(
                ddof=1
            )
        )

        new_f1_std = float(
            sampled_videos[
                "NEW_f1"
            ].std(
                ddof=1
            )
        )

        bootstrap_deltas[
            "macro_video_precision"
        ].append(
            new_macro_precision
            - old_macro_precision
        )

        bootstrap_deltas[
            "macro_video_recall"
        ].append(
            new_macro_recall
            - old_macro_recall
        )

        bootstrap_deltas[
            "macro_video_f1_std"
        ].append(
            new_f1_std
            - old_f1_std
        )

        # ----------------------------------------------------
        # Pooled pair-level metrics,
        # preserving whole video clusters
        # ----------------------------------------------------

        y_true = (
            sampled_pairs[
                "y_true"
            ]
            .to_numpy(
                dtype=int
            )
        )

        old_metrics = (
            compute_pair_metrics(
                y_true,
                sampled_pairs[
                    "pred_OLD"
                ],
                sampled_pairs[
                    "score_OLD"
                ],
            )
        )

        new_metrics = (
            compute_pair_metrics(
                y_true,
                sampled_pairs[
                    "pred_NEW"
                ],
                sampled_pairs[
                    "score_NEW"
                ],
            )
        )

        for metric in [
            "micro_precision",
            "micro_recall",
            "micro_f1",
            "mcc",
            "balanced_accuracy",
            "specificity",
            "fpr",
            "pr_auc",
            "roc_auc",
        ]:

            old_value = (
                old_metrics[
                    metric
                ]
            )

            new_value = (
                new_metrics[
                    metric
                ]
            )

            if (
                np.isnan(
                    old_value
                )
                or np.isnan(
                    new_value
                )
            ):
                continue

            bootstrap_deltas[
                metric
            ].append(
                new_value
                - old_value
            )

    return bootstrap_deltas


# ============================================================
# OUTPUT
# ============================================================

def build_output(
    overall,
    bootstrap_deltas,
):

    (
        old_row,
        new_row,
    ) = get_overall_rows(
        overall
    )

    roles = {
        "macro_video_precision":
            "secondary",

        "macro_video_recall":
            "secondary",

        "macro_video_f1_std":
            "descriptive",

        "micro_precision":
            "secondary",

        "micro_recall":
            "secondary",

        "micro_f1":
            "secondary",

        "mcc":
            "secondary",

        "balanced_accuracy":
            "secondary",

        "specificity":
            "descriptive",

        "fpr":
            "descriptive",

        "pr_auc":
            "secondary",

        "roc_auc":
            "secondary",
    }

    macro_metrics = {
        "macro_video_precision",
        "macro_video_recall",
        "macro_video_f1_std",
    }

    rows = []

    for metric, role in (
        roles.items()
    ):

        old_value = float(
            old_row[
                metric
            ]
        )

        new_value = float(
            new_row[
                metric
            ]
        )

        delta = (
            new_value
            - old_value
        )

        bootstrap_values = np.asarray(
            bootstrap_deltas[
                metric
            ],
            dtype=float,
        )

        if (
            len(bootstrap_values)
            == 0
        ):
            raise RuntimeError(
                "No valid bootstrap values for "
                f"{metric}"
            )

        ci_low = float(
            np.percentile(
                bootstrap_values,
                CI_LOW,
            )
        )

        ci_high = float(
            np.percentile(
                bootstrap_values,
                CI_HIGH,
            )
        )

        if metric in macro_metrics:

            method = (
                "subject-stratified "
                "video bootstrap"
            )

            resampling_unit = "video"

        else:

            method = (
                "subject-stratified "
                "video-cluster bootstrap"
            )

            resampling_unit = (
                "video cluster with all "
                "video-concept pairs retained"
            )

        note = (
            "Secondary/descriptive uncertainty "
            "only; no additional hypothesis "
            "test is introduced."
        )

        if metric == "macro_video_recall":

            note = (
                "Important secondary Recall "
                "analysis requested for V2. "
                "Interval describes the "
                "NEW minus OLD DEV difference."
            )

        if metric == "micro_recall":

            note = (
                "Secondary pooled Recall "
                "analysis using video-cluster "
                "resampling to preserve "
                "within-video dependence."
            )

        rows.append(
            {
                "metric":
                    metric,

                "role":
                    role,

                "old_value":
                    old_value,

                "new_value":
                    new_value,

                "difference_new_minus_old":
                    delta,

                "ci_level":
                    0.95,

                "ci_low":
                    ci_low,

                "ci_high":
                    ci_high,

                "method":
                    method,

                "resampling_unit":
                    resampling_unit,

                "stratified_by":
                    "subject",

                "bootstrap_iterations":
                    BOOTSTRAP_ITERATIONS,

                "bootstrap_random_state":
                    BOOTSTRAP_RANDOM_STATE,

                "n_videos":
                    EXPECTED_VIDEOS,

                "n_pairs":
                    EXPECTED_PAIRS,

                "hypothesis_test":
                    "none",

                "note":
                    note,
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# HISTORICAL VALUE CROSS-CHECK
# ============================================================

def cross_check_observed_values(
    pairs,
    video_metrics,
    overall,
):

    (
        old_row,
        new_row,
    ) = get_overall_rows(
        overall
    )

    checks = []

    checks.append(
        np.isclose(
            video_metrics[
                "OLD_precision"
            ].mean(),
            old_row[
                "macro_video_precision"
            ],
            atol=TOL,
        )
    )

    checks.append(
        np.isclose(
            video_metrics[
                "NEW_precision"
            ].mean(),
            new_row[
                "macro_video_precision"
            ],
            atol=TOL,
        )
    )

    checks.append(
        np.isclose(
            video_metrics[
                "OLD_recall"
            ].mean(),
            old_row[
                "macro_video_recall"
            ],
            atol=TOL,
        )
    )

    checks.append(
        np.isclose(
            video_metrics[
                "NEW_recall"
            ].mean(),
            new_row[
                "macro_video_recall"
            ],
            atol=TOL,
        )
    )

    old_pair = compute_pair_metrics(
        pairs["y_true"],
        pairs["pred_OLD"],
        pairs["score_OLD"],
    )

    new_pair = compute_pair_metrics(
        pairs["y_true"],
        pairs["pred_NEW"],
        pairs["score_NEW"],
    )

    for metric in [
        "micro_precision",
        "micro_recall",
        "micro_f1",
        "mcc",
        "balanced_accuracy",
        "specificity",
        "fpr",
        "pr_auc",
        "roc_auc",
    ]:

        checks.append(
            np.isclose(
                old_pair[
                    metric
                ],
                old_row[
                    metric
                ],
                atol=TOL,
            )
        )

        checks.append(
            np.isclose(
                new_pair[
                    metric
                ],
                new_row[
                    metric
                ],
                atol=TOL,
            )
        )

    if not all(checks):
        raise RuntimeError(
            "Historical metric cross-check "
            "failed."
        )

    print(
        "Historical Step 13 metric "
        "cross-check: PASS"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 72
    )

    print(
        "V2 A6 SECONDARY METRIC UNCERTAINTY"
    )

    print(
        "=" * 72
    )

    (
        pairs,
        video_metrics,
        overall,
    ) = load_inputs()

    audit_inputs(
        pairs,
        video_metrics,
        overall,
    )

    cross_check_observed_values(
        pairs,
        video_metrics,
        overall,
    )

    bootstrap_deltas = (
        run_bootstrap(
            pairs,
            video_metrics,
        )
    )

    output = build_output(
        overall,
        bootstrap_deltas,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()

    print(
        "Secondary metric uncertainty:"
    )

    for row in output.itertuples():

        print(
            f"{row.metric}: "
            f"OLD={row.old_value:.6f}, "
            f"NEW={row.new_value:.6f}, "
            f"delta={row.difference_new_minus_old:+.6f}, "
            f"95% CI=[{row.ci_low:.6f}, "
            f"{row.ci_high:.6f}]"
        )

    print()

    recall_row = (
        output[
            output["metric"]
            .eq(
                "macro_video_recall"
            )
        ]
        .iloc[0]
    )

    print(
        "Macro Video Recall focus:"
    )

    print(
        f"  OLD = "
        f"{recall_row['old_value']:.6f}"
    )

    print(
        f"  NEW = "
        f"{recall_row['new_value']:.6f}"
    )

    print(
        f"  Delta = "
        f"{recall_row['difference_new_minus_old']:+.6f}"
    )

    print(
        "  95% CI = "
        f"[{recall_row['ci_low']:.6f}, "
        f"{recall_row['ci_high']:.6f}]"
    )

    print()

    print(
        "No secondary hypothesis tests "
        "were added."
    )

    print(
        "Saved:",
        OUTPUT_FILE,
    )


if __name__ == "__main__":
    main()