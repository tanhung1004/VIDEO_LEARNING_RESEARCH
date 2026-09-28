from pathlib import Path

import numpy as np
import pandas as pd


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

OLD_STATS_FILE = (
    STEP13_ROOT
    / "old_vs_new_combined_statistics.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "module1"
    / "results"
    / "v2_validation"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "V2_CLUSTERED_STATISTICS.csv"
)


# ============================================================
# FROZEN CURRENT DEV EXPECTATIONS
# ============================================================

EXPECTED_VIDEOS = 40
EXPECTED_PAIRS = 390

BOOTSTRAP_ITERATIONS = 10000
BOOTSTRAP_RANDOM_STATE = 42

CI_LOW_PERCENTILE = 2.5
CI_HIGH_PERCENTILE = 97.5

TOL = 1e-12


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

    old_stats = pd.read_csv(
        OLD_STATS_FILE
    )

    return (
        pairs,
        video_metrics,
        old_stats,
    )


def audit_inputs(
    pairs,
    video_metrics,
    old_stats,
):

    required_pair_columns = {
        "video_id",
        "subject",
        "concept",
        "y_true",
        "pred_OLD",
        "pred_NEW",
    }

    required_video_columns = {
        "video_id",
        "subject",
        "OLD_f1",
        "NEW_f1",
        "delta_f1_NEW_minus_OLD",
    }

    missing_pair_columns = (
        required_pair_columns
        - set(pairs.columns)
    )

    missing_video_columns = (
        required_video_columns
        - set(video_metrics.columns)
    )

    if missing_pair_columns:
        raise RuntimeError(
            "Missing pair columns: "
            f"{sorted(missing_pair_columns)}"
        )

    if missing_video_columns:
        raise RuntimeError(
            "Missing video columns: "
            f"{sorted(missing_video_columns)}"
        )

    if len(pairs) != EXPECTED_PAIRS:
        raise RuntimeError(
            "Unexpected pair count: "
            f"{len(pairs)} "
            f"(expected {EXPECTED_PAIRS})"
        )

    if (
        video_metrics["video_id"]
        .nunique()
        != EXPECTED_VIDEOS
    ):
        raise RuntimeError(
            "Unexpected number of videos in "
            "video_metrics."
        )

    if len(video_metrics) != EXPECTED_VIDEOS:
        raise RuntimeError(
            "video_metrics must contain exactly "
            f"{EXPECTED_VIDEOS} rows."
        )

    if (
        pairs["video_id"]
        .nunique()
        != EXPECTED_VIDEOS
    ):
        raise RuntimeError(
            "Pair table does not contain exactly "
            f"{EXPECTED_VIDEOS} videos."
        )

    pair_video_ids = set(
        pairs["video_id"]
    )

    metric_video_ids = set(
        video_metrics["video_id"]
    )

    if pair_video_ids != metric_video_ids:
        raise RuntimeError(
            "Pair/video metric video-ID sets differ."
        )

    duplicate_pairs = int(
        pairs.duplicated(
            [
                "video_id",
                "concept",
            ]
        ).sum()
    )

    if duplicate_pairs != 0:
        raise RuntimeError(
            "Duplicate video-concept pairs found: "
            f"{duplicate_pairs}"
        )

    for column in [
        "y_true",
        "pred_OLD",
        "pred_NEW",
    ]:

        values = set(
            pairs[column]
            .dropna()
            .astype(int)
            .unique()
        )

        if not values.issubset(
            {0, 1}
        ):
            raise RuntimeError(
                f"{column} is not binary."
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
            "Expected four subjects with "
            "10 DEV videos each. "
            f"Observed: "
            f"{subject_counts.to_dict()}"
        )

    if len(old_stats) != 1:
        raise RuntimeError(
            "Historical statistics CSV must "
            "contain exactly one row."
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
# OBSERVED VALUES
# ============================================================

def observed_macro_f1(
    video_metrics,
):

    old_macro = float(
        video_metrics[
            "OLD_f1"
        ].mean()
    )

    new_macro = float(
        video_metrics[
            "NEW_f1"
        ].mean()
    )

    delta = (
        new_macro
        - old_macro
    )

    return (
        old_macro,
        new_macro,
        delta,
    )


def observed_pair_accuracy(
    pairs,
):

    y_true = (
        pairs["y_true"]
        .to_numpy(
            dtype=int
        )
    )

    old_pred = (
        pairs["pred_OLD"]
        .to_numpy(
            dtype=int
        )
    )

    new_pred = (
        pairs["pred_NEW"]
        .to_numpy(
            dtype=int
        )
    )

    old_correct = (
        old_pred
        == y_true
    )

    new_correct = (
        new_pred
        == y_true
    )

    old_accuracy = float(
        old_correct.mean()
    )

    new_accuracy = float(
        new_correct.mean()
    )

    delta = (
        new_accuracy
        - old_accuracy
    )

    old_correct_new_wrong = int(
        np.sum(
            old_correct
            & (~new_correct)
        )
    )

    old_wrong_new_correct = int(
        np.sum(
            (~old_correct)
            & new_correct
        )
    )

    return {
        "old_accuracy":
            old_accuracy,

        "new_accuracy":
            new_accuracy,

        "delta":
            delta,

        "old_correct_new_wrong":
            old_correct_new_wrong,

        "old_wrong_new_correct":
            old_wrong_new_correct,

        "discordant":
            (
                old_correct_new_wrong
                +
                old_wrong_new_correct
            ),
    }


# ============================================================
# SUBJECT-STRATIFIED VIDEO BOOTSTRAP
# ============================================================

def build_subject_groups(
    frame,
):

    return {
        subject:
            group.reset_index(
                drop=True
            )

        for subject, group
        in frame.groupby(
            "subject",
            sort=True,
        )
    }


def bootstrap_macro_video_f1_delta(
    video_metrics,
):

    rng = np.random.default_rng(
        BOOTSTRAP_RANDOM_STATE
    )

    groups = build_subject_groups(
        video_metrics
    )

    deltas = []

    for _ in range(
        BOOTSTRAP_ITERATIONS
    ):

        sampled_groups = []

        for group in groups.values():

            indices = rng.integers(
                low=0,
                high=len(group),
                size=len(group),
            )

            sampled_groups.append(
                group.iloc[
                    indices
                ]
            )

        sample = pd.concat(
            sampled_groups,
            ignore_index=True,
        )

        old_macro = float(
            sample[
                "OLD_f1"
            ].mean()
        )

        new_macro = float(
            sample[
                "NEW_f1"
            ].mean()
        )

        deltas.append(
            new_macro
            - old_macro
        )

    deltas = np.asarray(
        deltas,
        dtype=float,
    )

    ci_low = float(
        np.percentile(
            deltas,
            CI_LOW_PERCENTILE,
        )
    )

    ci_high = float(
        np.percentile(
            deltas,
            CI_HIGH_PERCENTILE,
        )
    )

    return (
        deltas,
        ci_low,
        ci_high,
    )


# ============================================================
# CLUSTER BOOTSTRAP FOR PAIR-LEVEL CORRECTNESS
# ============================================================

def build_video_clusters(
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


def bootstrap_clustered_pair_accuracy_delta(
    pairs,
):

    rng = np.random.default_rng(
        BOOTSTRAP_RANDOM_STATE
    )

    clusters = build_video_clusters(
        pairs
    )

    subjects = sorted(
        pairs[
            "subject"
        ].unique()
    )

    subject_video_ids = {}

    for subject in subjects:

        subject_video_ids[
            subject
        ] = sorted(
            pairs.loc[
                pairs[
                    "subject"
                ].eq(
                    subject
                ),
                "video_id",
            ].unique()
        )

    deltas = []

    for _ in range(
        BOOTSTRAP_ITERATIONS
    ):

        sampled_blocks = []

        for subject in subjects:

            video_ids = (
                subject_video_ids[
                    subject
                ]
            )

            sampled_indices = (
                rng.integers(
                    low=0,
                    high=len(video_ids),
                    size=len(video_ids),
                )
            )

            for draw_number, idx in enumerate(
                sampled_indices
            ):

                video_id = (
                    video_ids[
                        idx
                    ]
                )

                block = (
                    clusters[
                        (
                            subject,
                            video_id,
                        )
                    ]
                    .copy()
                )

                # Keep repeated bootstrap draws
                # as distinct sampled clusters.
                block[
                    "_bootstrap_cluster"
                ] = (
                    f"{subject}_"
                    f"{draw_number}"
                )

                sampled_blocks.append(
                    block
                )

        sample = pd.concat(
            sampled_blocks,
            ignore_index=True,
        )

        y_true = (
            sample["y_true"]
            .to_numpy(
                dtype=int
            )
        )

        old_pred = (
            sample["pred_OLD"]
            .to_numpy(
                dtype=int
            )
        )

        new_pred = (
            sample["pred_NEW"]
            .to_numpy(
                dtype=int
            )
        )

        old_accuracy = float(
            np.mean(
                old_pred
                == y_true
            )
        )

        new_accuracy = float(
            np.mean(
                new_pred
                == y_true
            )
        )

        deltas.append(
            new_accuracy
            - old_accuracy
        )

    deltas = np.asarray(
        deltas,
        dtype=float,
    )

    ci_low = float(
        np.percentile(
            deltas,
            CI_LOW_PERCENTILE,
        )
    )

    ci_high = float(
        np.percentile(
            deltas,
            CI_HIGH_PERCENTILE,
        )
    )

    return (
        deltas,
        ci_low,
        ci_high,
    )


# ============================================================
# HISTORICAL RESULT CROSS-CHECK
# ============================================================

def audit_against_historical_statistics(
    old_stats,
    old_macro,
    new_macro,
    macro_delta,
    pair_accuracy_info,
):

    row = old_stats.iloc[0]

    checks = []

    checks.append(
        np.isclose(
            old_macro,
            float(
                row[
                    "OLD_macro_video_f1"
                ]
            ),
            atol=TOL,
        )
    )

    checks.append(
        np.isclose(
            new_macro,
            float(
                row[
                    "NEW_macro_video_f1"
                ]
            ),
            atol=TOL,
        )
    )

    checks.append(
        np.isclose(
            macro_delta,
            float(
                row[
                    "delta_macro_video_f1_NEW_minus_OLD"
                ]
            ),
            atol=TOL,
        )
    )

    checks.append(
        (
            pair_accuracy_info[
                "old_correct_new_wrong"
            ]
            ==
            int(
                row[
                    "mcnemar_OLD_correct_NEW_wrong"
                ]
            )
        )
    )

    checks.append(
        (
            pair_accuracy_info[
                "old_wrong_new_correct"
            ]
            ==
            int(
                row[
                    "mcnemar_OLD_wrong_NEW_correct"
                ]
            )
        )
    )

    checks.append(
        (
            pair_accuracy_info[
                "discordant"
            ]
            ==
            int(
                row[
                    "mcnemar_discordant"
                ]
            )
        )
    )

    if not all(checks):
        raise RuntimeError(
            "Frozen Step 13 statistical "
            "cross-check failed."
        )

    print(
        "Historical Step 13 cross-check: PASS"
    )


# ============================================================
# OUTPUT
# ============================================================

def build_summary(
    old_stats,
    old_macro,
    new_macro,
    macro_delta,
    macro_ci_low,
    macro_ci_high,
    pair_accuracy_info,
    pair_ci_low,
    pair_ci_high,
):

    historical = old_stats.iloc[0]

    rows = [
        {
            "analysis":
                "macro_video_f1_delta",

            "role":
                "primary",

            "old_value":
                old_macro,

            "new_value":
                new_macro,

            "delta_new_minus_old":
                macro_delta,

            "ci_level":
                0.95,

            "ci_low":
                macro_ci_low,

            "ci_high":
                macro_ci_high,

            "method":
                (
                    "subject-stratified "
                    "video bootstrap"
                ),

            "resampling_unit":
                "video",

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

            "historical_p_value":
                np.nan,

            "v2_inference_status":
                (
                    "cluster-aware "
                    "development-stage analysis"
                ),

            "note":
                (
                    "Primary Macro Video F1 "
                    "difference. Entire videos "
                    "are resampled within subject."
                ),
        },

        {
            "analysis":
                "pair_accuracy_delta",

            "role":
                "dependence_diagnostic",

            "old_value":
                pair_accuracy_info[
                    "old_accuracy"
                ],

            "new_value":
                pair_accuracy_info[
                    "new_accuracy"
                ],

            "delta_new_minus_old":
                pair_accuracy_info[
                    "delta"
                ],

            "ci_level":
                0.95,

            "ci_low":
                pair_ci_low,

            "ci_high":
                pair_ci_high,

            "method":
                (
                    "subject-stratified "
                    "video-cluster bootstrap"
                ),

            "resampling_unit":
                (
                    "video cluster with all "
                    "video-concept pairs retained"
                ),

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

            "historical_p_value":
                np.nan,

            "v2_inference_status":
                (
                    "cluster-aware "
                    "dependence diagnostic"
                ),

            "note":
                (
                    "Used to assess pair-level "
                    "correctness difference while "
                    "preserving within-video "
                    "dependence."
                ),
        },

        {
            "analysis":
                "historical_exact_mcnemar",

            "role":
                "historical_only",

            "old_value":
                float(
                    pair_accuracy_info[
                        "old_correct_new_wrong"
                    ]
                ),

            "new_value":
                float(
                    pair_accuracy_info[
                        "old_wrong_new_correct"
                    ]
                ),

            "delta_new_minus_old":
                float(
                    pair_accuracy_info[
                        "old_wrong_new_correct"
                    ]
                    -
                    pair_accuracy_info[
                        "old_correct_new_wrong"
                    ]
                ),

            "ci_level":
                np.nan,

            "ci_low":
                np.nan,

            "ci_high":
                np.nan,

            "method":
                "exact McNemar / binomial test",

            "resampling_unit":
                "video-concept pair",

            "stratified_by":
                "none",

            "bootstrap_iterations":
                0,

            "bootstrap_random_state":
                np.nan,

            "n_videos":
                EXPECTED_VIDEOS,

            "n_pairs":
                EXPECTED_PAIRS,

            "historical_p_value":
                float(
                    historical[
                        "mcnemar_exact_p"
                    ]
                ),

            "v2_inference_status":
                (
                    "historical result only; "
                    "not treated as "
                    "cluster-aware V2 inference"
                ),

            "note":
                (
                    "Historical Step 13 McNemar "
                    "uses individual video-concept "
                    "pairs. Multiple pairs belong "
                    "to the same video, so the "
                    "pair-independence assumption "
                    "requires caution."
                ),
        },
    ]

    return pd.DataFrame(
        rows
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 72
    )

    print(
        "V2 A5 CLUSTERED STATISTICAL ANALYSIS"
    )

    print(
        "=" * 72
    )

    (
        pairs,
        video_metrics,
        old_stats,
    ) = load_inputs()

    audit_inputs(
        pairs,
        video_metrics,
        old_stats,
    )

    (
        old_macro,
        new_macro,
        macro_delta,
    ) = observed_macro_f1(
        video_metrics
    )

    pair_accuracy_info = (
        observed_pair_accuracy(
            pairs
        )
    )

    audit_against_historical_statistics(
        old_stats,
        old_macro,
        new_macro,
        macro_delta,
        pair_accuracy_info,
    )

    (
        _,
        macro_ci_low,
        macro_ci_high,
    ) = (
        bootstrap_macro_video_f1_delta(
            video_metrics
        )
    )

    (
        _,
        pair_ci_low,
        pair_ci_high,
    ) = (
        bootstrap_clustered_pair_accuracy_delta(
            pairs
        )
    )

    summary = build_summary(
        old_stats,
        old_macro,
        new_macro,
        macro_delta,
        macro_ci_low,
        macro_ci_high,
        pair_accuracy_info,
        pair_ci_low,
        pair_ci_high,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()

    print(
        "Macro Video F1:"
    )

    print(
        f"  OLD = {old_macro:.6f}"
    )

    print(
        f"  NEW = {new_macro:.6f}"
    )

    print(
        f"  Delta = {macro_delta:+.6f}"
    )

    print(
        "  Clustered 95% CI = "
        f"[{macro_ci_low:.6f}, "
        f"{macro_ci_high:.6f}]"
    )

    print()

    print(
        "Pair correctness:"
    )

    print(
        "  OLD accuracy = "
        f"{pair_accuracy_info['old_accuracy']:.6f}"
    )

    print(
        "  NEW accuracy = "
        f"{pair_accuracy_info['new_accuracy']:.6f}"
    )

    print(
        "  Delta = "
        f"{pair_accuracy_info['delta']:+.6f}"
    )

    print(
        "  Video-cluster 95% CI = "
        f"[{pair_ci_low:.6f}, "
        f"{pair_ci_high:.6f}]"
    )

    print()

    print(
        "Historical McNemar:"
    )

    print(
        "  OLD correct / NEW wrong = "
        f"{pair_accuracy_info['old_correct_new_wrong']}"
    )

    print(
        "  OLD wrong / NEW correct = "
        f"{pair_accuracy_info['old_wrong_new_correct']}"
    )

    print(
        "  Historical exact p = "
        f"{float(old_stats.iloc[0]['mcnemar_exact_p']):.12g}"
    )

    print(
        "  Interpretation: historical "
        "pair-level result only."
    )

    print()

    print(
        "Saved:",
        OUTPUT_FILE,
    )


if __name__ == "__main__":
    main()