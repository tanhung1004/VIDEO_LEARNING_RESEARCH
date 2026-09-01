from pathlib import Path
import json

import numpy as np
import pandas as pd

from scipy import stats


ROOT = Path(__file__).resolve().parents[2]

RESULT_DIR = (
    ROOT
    / "module1"
    / "results"
    / "p1_validation"
)

PER_VIDEO_PATH = (
    RESULT_DIR
    / "per_video_metrics.csv"
)

OUTER_METRICS_PATH = (
    RESULT_DIR
    / "outer_fold_metrics.csv"
)

SUMMARY_PATH = (
    RESULT_DIR
    / "statistical_summary.csv"
)

PAIRED_PATH = (
    RESULT_DIR
    / "paired_f1_analysis.csv"
)

SUBJECT_PATH = (
    RESULT_DIR
    / "subject_metrics.csv"
)

FOLD_SUMMARY_PATH = (
    RESULT_DIR
    / "outer_fold_summary.csv"
)

CONFIG_PATH = (
    RESULT_DIR
    / "statistical_analysis_config.json"
)


# ============================================================
# PRE-SPECIFIED STATISTICAL SETTINGS
# ============================================================

RANDOM_STATE = 42
BOOTSTRAP_ITERATIONS = 10000
CI_LEVEL = 0.95

VARIANTS = [
    "nested_tuned",
    "frozen_fixed",
]

PRIMARY_METRIC = "f1"

SUBJECTS = [
    "SQL",
    "Python",
    "Java",
    "C++",
]


# ============================================================
# BOOTSTRAP
# ============================================================

def stratified_bootstrap_mean(
    data,
    value_col,
    rng,
    iterations=10000,
):
    """
    Bootstrap mean while preserving the original
    number of videos per subject.

    Sampling unit = VIDEO.
    """

    subject_groups = {
        subject: (
            data[
                data["subject"].eq(subject)
            ][value_col]
            .astype(float)
            .to_numpy()
        )
        for subject in SUBJECTS
    }

    for subject, values in subject_groups.items():
        if len(values) != 10:
            raise ValueError(
                f"{subject}: expected 10 videos, "
                f"found {len(values)}"
            )

    samples = np.empty(
        iterations,
        dtype=float,
    )

    for i in range(iterations):

        boot_values = []

        for subject in SUBJECTS:

            values = (
                subject_groups[
                    subject
                ]
            )

            sampled = rng.choice(
                values,
                size=len(values),
                replace=True,
            )

            boot_values.extend(
                sampled.tolist()
            )

        samples[i] = float(
            np.mean(
                boot_values
            )
        )

    return samples


def percentile_ci(
    values,
    level=0.95,
):
    alpha = (
        1.0 - level
    )

    lower = np.quantile(
        values,
        alpha / 2.0,
    )

    upper = np.quantile(
        values,
        1.0 - alpha / 2.0,
    )

    return (
        float(lower),
        float(upper),
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("P1.9 STATISTICAL SUMMARY + PAIRED ANALYSIS")
    print("=" * 80)

    # --------------------------------------------------------
    # 1. LOAD
    # --------------------------------------------------------

    per_video = pd.read_csv(
        PER_VIDEO_PATH,
        dtype={
            "model_variant": str,
            "video_id": str,
            "subject": str,
        },
    )

    outer = pd.read_csv(
        OUTER_METRICS_PATH,
        dtype={
            "model_variant": str,
        },
    )

    per_video[
        "video_id"
    ] = (
        per_video[
            "video_id"
        ]
        .str.strip()
        .str.lower()
    )

    per_video[
        "subject"
    ] = (
        per_video[
            "subject"
        ]
        .str.strip()
    )

    # --------------------------------------------------------
    # 2. INPUT AUDIT
    # --------------------------------------------------------

    if set(
        per_video[
            "model_variant"
        ].unique()
    ) != set(
        VARIANTS
    ):
        raise ValueError(
            "Unexpected model variants "
            "in per_video_metrics.csv"
        )

    for variant in VARIANTS:

        v = per_video[
            per_video[
                "model_variant"
            ].eq(
                variant
            )
        ]

        if (
            v["video_id"]
            .nunique()
            != 40
        ):
            raise ValueError(
                f"{variant}: expected "
                "40 unique videos"
            )

        if len(v) != 40:
            raise ValueError(
                f"{variant}: expected "
                "40 metric rows"
            )

        counts = (
            v[
                "subject"
            ]
            .value_counts()
            .to_dict()
        )

        for subject in SUBJECTS:

            if (
                counts.get(
                    subject,
                    0,
                )
                != 10
            ):
                raise ValueError(
                    f"{variant}: "
                    f"{subject} != 10 videos"
                )

    print("\nInput audit: PASS")
    print("Variants :", VARIANTS)
    print("Videos   : 40 per variant")
    print("Subjects : 10 videos each")

    # --------------------------------------------------------
    # 3. ALIGN PAIRED VIDEOS
    # --------------------------------------------------------

    nested = (
        per_video[
            per_video[
                "model_variant"
            ].eq(
                "nested_tuned"
            )
        ][
            [
                "video_id",
                "subject",
                "precision",
                "recall",
                "f1",
            ]
        ]
        .copy()
    )

    frozen = (
        per_video[
            per_video[
                "model_variant"
            ].eq(
                "frozen_fixed"
            )
        ][
            [
                "video_id",
                "subject",
                "precision",
                "recall",
                "f1",
            ]
        ]
        .copy()
    )

    paired = nested.merge(
        frozen,
        on=[
            "video_id",
            "subject",
        ],
        how="inner",
        suffixes=(
            "_nested",
            "_frozen",
        ),
        validate="one_to_one",
    )

    if len(paired) != 40:
        raise ValueError(
            "Paired alignment != 40 videos"
        )

    paired[
        "delta_f1"
    ] = (
        paired[
            "f1_nested"
        ]
        -
        paired[
            "f1_frozen"
        ]
    )

    paired[
        "delta_precision"
    ] = (
        paired[
            "precision_nested"
        ]
        -
        paired[
            "precision_frozen"
        ]
    )

    paired[
        "delta_recall"
    ] = (
        paired[
            "recall_nested"
        ]
        -
        paired[
            "recall_frozen"
        ]
    )

    # --------------------------------------------------------
    # 4. BOOTSTRAP VARIANT MEANS
    # --------------------------------------------------------

    rng = np.random.default_rng(
        RANDOM_STATE
    )

    summary_rows = []

    for variant in VARIANTS:

        data = per_video[
            per_video[
                "model_variant"
            ].eq(
                variant
            )
        ].copy()

        for metric in [
            "precision",
            "recall",
            "f1",
        ]:

            values = (
                data[
                    metric
                ]
                .astype(float)
                .to_numpy()
            )

            boot = (
                stratified_bootstrap_mean(
                    data=data,
                    value_col=metric,
                    rng=rng,
                    iterations=
                        BOOTSTRAP_ITERATIONS,
                )
            )

            ci_low, ci_high = (
                percentile_ci(
                    boot,
                    CI_LEVEL,
                )
            )

            summary_rows.append(
                {
                    "model_variant":
                        variant,

                    "metric":
                        metric,

                    "n_videos":
                        len(values),

                    "mean":
                        float(
                            np.mean(values)
                        ),

                    "std":
                        float(
                            np.std(
                                values,
                                ddof=1,
                            )
                        ),

                    "median":
                        float(
                            np.median(
                                values
                            )
                        ),

                    "ci_method":
                        (
                            "stratified "
                            "video bootstrap"
                        ),

                    "ci_level":
                        CI_LEVEL,

                    "ci_lower":
                        ci_low,

                    "ci_upper":
                        ci_high,
                }
            )

    summary_df = pd.DataFrame(
        summary_rows
    )

    # --------------------------------------------------------
    # 5. PAIRED PRIMARY F1 ANALYSIS
    # --------------------------------------------------------

    differences = (
        paired[
            "delta_f1"
        ]
        .astype(float)
        .to_numpy()
    )

    # Stratified bootstrap of paired delta.
    delta_boot = (
        stratified_bootstrap_mean(
            data=paired,
            value_col="delta_f1",
            rng=rng,
            iterations=
                BOOTSTRAP_ITERATIONS,
        )
    )

    delta_ci_low, delta_ci_high = (
        percentile_ci(
            delta_boot,
            CI_LEVEL,
        )
    )

    # Primary inferential paired test.
    #
    # zero_method='wilcox' removes exact zero
    # differences from signed-rank calculation.
    try:

        wilcoxon_result = (
            stats.wilcoxon(
                paired[
                    "f1_nested"
                ],
                paired[
                    "f1_frozen"
                ],
                zero_method="wilcox",
                alternative="two-sided",
                method="auto",
            )
        )

        wilcoxon_stat = float(
            wilcoxon_result.statistic
        )

        wilcoxon_p = float(
            wilcoxon_result.pvalue
        )

    except ValueError:

        # Can happen if every paired
        # difference is exactly zero.
        wilcoxon_stat = 0.0
        wilcoxon_p = 1.0

    # Secondary parametric paired test,
    # reported transparently, not used
    # to replace the primary test.
    paired_t = stats.ttest_rel(
        paired[
            "f1_nested"
        ],
        paired[
            "f1_frozen"
        ],
        nan_policy="raise",
    )

    # Paired effect size: Cohen's dz.
    diff_std = float(
        np.std(
            differences,
            ddof=1,
        )
    )

    if diff_std == 0:

        cohens_dz = 0.0

    else:

        cohens_dz = float(
            np.mean(
                differences
            )
            / diff_std
        )

    wins = int(
        (
            paired[
                "delta_f1"
            ]
            > 0
        ).sum()
    )

    ties = int(
        (
            paired[
                "delta_f1"
            ]
            == 0
        ).sum()
    )

    losses = int(
        (
            paired[
                "delta_f1"
            ]
            < 0
        ).sum()
    )

    paired_analysis = pd.DataFrame(
        [
            {
                "comparison":
                    (
                        "nested_tuned "
                        "- frozen_fixed"
                    ),

                "metric":
                    "video_level_f1",

                "n_paired_videos":
                    40,

                "nested_mean_f1":
                    float(
                        paired[
                            "f1_nested"
                        ].mean()
                    ),

                "frozen_mean_f1":
                    float(
                        paired[
                            "f1_frozen"
                        ].mean()
                    ),

                "mean_difference":
                    float(
                        paired[
                            "delta_f1"
                        ].mean()
                    ),

                "median_difference":
                    float(
                        paired[
                            "delta_f1"
                        ].median()
                    ),

                "difference_std":
                    diff_std,

                "bootstrap_ci_lower":
                    delta_ci_low,

                "bootstrap_ci_upper":
                    delta_ci_high,

                "bootstrap_iterations":
                    BOOTSTRAP_ITERATIONS,

                "bootstrap_method":
                    (
                        "stratified paired "
                        "video bootstrap"
                    ),

                "primary_test":
                    "Wilcoxon signed-rank",

                "wilcoxon_statistic":
                    wilcoxon_stat,

                "wilcoxon_p_value":
                    wilcoxon_p,

                "secondary_test":
                    "paired t-test",

                "paired_t_statistic":
                    float(
                        paired_t.statistic
                    ),

                "paired_t_p_value":
                    float(
                        paired_t.pvalue
                    ),

                "effect_size":
                    "Cohen_dz",

                "cohen_dz":
                    cohens_dz,

                "nested_wins":
                    wins,

                "ties":
                    ties,

                "nested_losses":
                    losses,
            }
        ]
    )

    # --------------------------------------------------------
    # 6. SUBJECT-LEVEL DESCRIPTIVE METRICS
    # --------------------------------------------------------

    subject_rows = []

    for variant in VARIANTS:

        data = per_video[
            per_video[
                "model_variant"
            ].eq(
                variant
            )
        ]

        for subject in SUBJECTS:

            group = data[
                data[
                    "subject"
                ].eq(
                    subject
                )
            ]

            subject_rows.append(
                {
                    "model_variant":
                        variant,

                    "subject":
                        subject,

                    "n_videos":
                        len(group),

                    "mean_precision":
                        float(
                            group[
                                "precision"
                            ].mean()
                        ),

                    "mean_recall":
                        float(
                            group[
                                "recall"
                            ].mean()
                        ),

                    "mean_f1":
                        float(
                            group[
                                "f1"
                            ].mean()
                        ),

                    "std_f1":
                        float(
                            group[
                                "f1"
                            ].std(
                                ddof=1
                            )
                        ),
                }
            )

    subject_df = pd.DataFrame(
        subject_rows
    )

    nested_subject = (
        subject_df[
            subject_df[
                "model_variant"
            ].eq(
                "nested_tuned"
            )
        ][
            [
                "subject",
                "mean_f1",
            ]
        ]
        .rename(
            columns={
                "mean_f1":
                    "nested_mean_f1"
            }
        )
    )

    frozen_subject = (
        subject_df[
            subject_df[
                "model_variant"
            ].eq(
                "frozen_fixed"
            )
        ][
            [
                "subject",
                "mean_f1",
            ]
        ]
        .rename(
            columns={
                "mean_f1":
                    "frozen_mean_f1"
            }
        )
    )

    subject_delta = (
        nested_subject
        .merge(
            frozen_subject,
            on="subject",
            validate="one_to_one",
        )
    )

    subject_delta[
        "delta_f1"
    ] = (
        subject_delta[
            "nested_mean_f1"
        ]
        -
        subject_delta[
            "frozen_mean_f1"
        ]
    )

    subject_delta[
        "model_variant"
    ] = (
        "paired_difference"
    )

    subject_delta[
        "n_videos"
    ] = 10

    subject_output = pd.concat(
        [
            subject_df,
            subject_delta[
                [
                    "model_variant",
                    "subject",
                    "n_videos",
                    "nested_mean_f1",
                    "frozen_mean_f1",
                    "delta_f1",
                ]
            ],
        ],
        ignore_index=True,
        sort=False,
    )

    # --------------------------------------------------------
    # 7. OUTER-FOLD DESCRIPTIVE SUMMARY
    # --------------------------------------------------------

    fold_rows = []

    for variant in VARIANTS:

        data = outer[
            outer[
                "model_variant"
            ].eq(
                variant
            )
        ]

        if len(data) != 5:
            raise ValueError(
                f"{variant}: expected "
                "5 outer fold metric rows"
            )

        for metric in [
            "macro_video_f1",
            "micro_f1",
        ]:

            values = (
                data[
                    metric
                ]
                .astype(float)
            )

            fold_rows.append(
                {
                    "model_variant":
                        variant,

                    "metric":
                        metric,

                    "n_outer_folds":
                        5,

                    "mean":
                        float(
                            values.mean()
                        ),

                    "std":
                        float(
                            values.std(
                                ddof=1
                            )
                        ),

                    "min":
                        float(
                            values.min()
                        ),

                    "max":
                        float(
                            values.max()
                        ),
                }
            )

    fold_summary = pd.DataFrame(
        fold_rows
    )

    # --------------------------------------------------------
    # 8. WRITE
    # --------------------------------------------------------

    summary_df.to_csv(
        SUMMARY_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    paired_analysis.to_csv(
        PAIRED_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    subject_output.to_csv(
        SUBJECT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    fold_summary.to_csv(
        FOLD_SUMMARY_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    config = {
        "analysis_stage":
            "P1.9",

        "sampling_unit":
            "video",

        "paired_comparison":
            True,

        "primary_metric":
            (
                "video-level F1"
            ),

        "primary_test":
            (
                "two-sided Wilcoxon "
                "signed-rank"
            ),

        "secondary_test":
            "paired t-test",

        "effect_size":
            "Cohen's dz",

        "confidence_interval":
            (
                "95% percentile "
                "stratified video bootstrap"
            ),

        "bootstrap_iterations":
            BOOTSTRAP_ITERATIONS,

        "bootstrap_random_state":
            RANDOM_STATE,

        "bootstrap_stratification":
            "subject",

        "multiple_testing_note":
            (
                "Primary inferential test "
                "is limited to the "
                "pre-specified video-level "
                "F1 comparison."
            ),
    }

    CONFIG_PATH.write_text(
        json.dumps(
            config,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # 9. TERMINAL REPORT
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("PRIMARY VIDEO-LEVEL F1 SUMMARY")
    print("=" * 80)

    print(
        summary_df[
            summary_df[
                "metric"
            ].eq(
                "f1"
            )
        ][
            [
                "model_variant",
                "n_videos",
                "mean",
                "std",
                "ci_lower",
                "ci_upper",
            ]
        ].to_string(
            index=False
        )
    )

    print("\n" + "=" * 80)
    print("PAIRED F1 COMPARISON")
    print("=" * 80)

    row = (
        paired_analysis
        .iloc[0]
    )

    print(
        "Nested mean F1 :",
        round(
            row[
                "nested_mean_f1"
            ],
            6,
        ),
    )

    print(
        "Frozen mean F1 :",
        round(
            row[
                "frozen_mean_f1"
            ],
            6,
        ),
    )

    print(
        "Mean delta F1  :",
        round(
            row[
                "mean_difference"
            ],
            6,
        ),
    )

    print(
        "95% bootstrap CI:",
        (
            round(
                row[
                    "bootstrap_ci_lower"
                ],
                6,
            ),
            round(
                row[
                    "bootstrap_ci_upper"
                ],
                6,
            ),
        ),
    )

    print(
        "Wilcoxon p     :",
        round(
            row[
                "wilcoxon_p_value"
            ],
            6,
        ),
    )

    print(
        "Paired t-test p:",
        round(
            row[
                "paired_t_p_value"
            ],
            6,
        ),
    )

    print(
        "Cohen's dz     :",
        round(
            row[
                "cohen_dz"
            ],
            6,
        ),
    )

    print(
        "Wins/Ties/Losses:",
        f"{wins}/{ties}/{losses}",
    )

    print("\n" + "=" * 80)
    print("SUBJECT-LEVEL F1 DIFFERENCES")
    print("=" * 80)

    print(
        subject_delta[
            [
                "subject",
                "nested_mean_f1",
                "frozen_mean_f1",
                "delta_f1",
            ]
        ].to_string(
            index=False
        )
    )

    print("\n" + "=" * 80)
    print(
        "P1.9 STATISTICAL ANALYSIS: PASS"
    )
    print("=" * 80)

    print(
        "Sampling unit: VIDEO"
    )

    print(
        "Primary paired test: "
        "Wilcoxon signed-rank"
    )

    print(
        "95% CI: stratified video bootstrap"
    )

    print(
        "No hyperparameters were changed "
        "after outer-test evaluation."
    )

    print(
        "Saved:",
        SUMMARY_PATH.relative_to(
            ROOT
        ),
    )

    print(
        "Saved:",
        PAIRED_PATH.relative_to(
            ROOT
        ),
    )

    print(
        "Saved:",
        SUBJECT_PATH.relative_to(
            ROOT
        ),
    )

    print(
        "Saved:",
        FOLD_SUMMARY_PATH.relative_to(
            ROOT
        ),
    )

    print(
        "Saved:",
        CONFIG_PATH.relative_to(
            ROOT
        ),
    )


if __name__ == "__main__":
    main()