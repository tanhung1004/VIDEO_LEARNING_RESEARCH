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

VIDEO_FILE = (
    STEP13_ROOT
    / "old_vs_new_combined_video_metrics.csv"
)

SUBJECT_FILE = (
    STEP13_ROOT
    / "old_vs_new_combined_subject_metrics.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "module1"
    / "results"
    / "v2_validation"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "V2_SUBJECT_UNCERTAINTY.csv"
)


# ============================================================
# CONSTANTS
# ============================================================

EXPECTED_SUBJECTS = {
    "C++",
    "Java",
    "Python",
    "SQL",
}

EXPECTED_VIDEOS_PER_SUBJECT = 10

BOOTSTRAP_ITERATIONS = 10000
BOOTSTRAP_RANDOM_STATE = 42

CI_LOW = 2.5
CI_HIGH = 97.5

TOL = 1e-12


# ============================================================
# LOAD
# ============================================================

def load_inputs():

    video_metrics = pd.read_csv(
        VIDEO_FILE
    )

    subject_metrics = pd.read_csv(
        SUBJECT_FILE
    )

    return (
        video_metrics,
        subject_metrics,
    )


# ============================================================
# AUDIT
# ============================================================

def audit_inputs(
    video_metrics,
    subject_metrics,
):

    required_video_columns = {
        "video_id",
        "subject",
        "OLD_f1",
        "NEW_f1",
    }

    required_subject_columns = {
        "subject",
        "system",
        "videos",
        "pairs",
        "macro_video_f1",
    }

    missing_video = (
        required_video_columns
        - set(video_metrics.columns)
    )

    missing_subject = (
        required_subject_columns
        - set(subject_metrics.columns)
    )

    if missing_video:
        raise RuntimeError(
            "Missing video columns: "
            f"{sorted(missing_video)}"
        )

    if missing_subject:
        raise RuntimeError(
            "Missing subject columns: "
            f"{sorted(missing_subject)}"
        )

    subjects = set(
        video_metrics["subject"]
    )

    if subjects != EXPECTED_SUBJECTS:
        raise RuntimeError(
            "Unexpected subjects: "
            f"{subjects}"
        )

    counts = (
        video_metrics[
            "subject"
        ]
        .value_counts()
    )

    for subject in EXPECTED_SUBJECTS:

        if (
            counts.get(
                subject,
                0,
            )
            != EXPECTED_VIDEOS_PER_SUBJECT
        ):
            raise RuntimeError(
                f"{subject}: expected "
                f"{EXPECTED_VIDEOS_PER_SUBJECT} "
                "videos."
            )

    if len(subject_metrics) != 8:
        raise RuntimeError(
            "Expected 8 subject metric rows "
            "(4 subjects x 2 systems)."
        )

    print(
        "Input audit: PASS"
    )

    print(
        "Subjects:",
        sorted(subjects),
    )

    print(
        "Videos per subject:",
        counts.to_dict(),
    )


# ============================================================
# HISTORICAL CROSS-CHECK
# ============================================================

def cross_check_subject_metrics(
    video_metrics,
    subject_metrics,
):

    for subject in sorted(
        EXPECTED_SUBJECTS
    ):

        videos = (
            video_metrics[
                video_metrics[
                    "subject"
                ].eq(
                    subject
                )
            ]
        )

        observed_old = float(
            videos[
                "OLD_f1"
            ].mean()
        )

        observed_new = float(
            videos[
                "NEW_f1"
            ].mean()
        )

        old_row = (
            subject_metrics[
                (
                    subject_metrics[
                        "subject"
                    ].eq(
                        subject
                    )
                )
                &
                (
                    subject_metrics[
                        "system"
                    ].eq(
                        "OLD_COMBINED"
                    )
                )
            ]
            .iloc[0]
        )

        new_row = (
            subject_metrics[
                (
                    subject_metrics[
                        "subject"
                    ].eq(
                        subject
                    )
                )
                &
                (
                    subject_metrics[
                        "system"
                    ].eq(
                        "NEW_COMBINED_F"
                    )
                )
            ]
            .iloc[0]
        )

        expected_old = float(
            old_row[
                "macro_video_f1"
            ]
        )

        expected_new = float(
            new_row[
                "macro_video_f1"
            ]
        )

        if not np.isclose(
            observed_old,
            expected_old,
            atol=TOL,
        ):
            raise RuntimeError(
                f"{subject}: OLD Macro Video F1 "
                "cross-check failed."
            )

        if not np.isclose(
            observed_new,
            expected_new,
            atol=TOL,
        ):
            raise RuntimeError(
                f"{subject}: NEW Macro Video F1 "
                "cross-check failed."
            )

    print(
        "Historical subject metric "
        "cross-check: PASS"
    )


# ============================================================
# SUBJECT BOOTSTRAP
# ============================================================

def bootstrap_subject(
    subject_df,
    rng,
):

    n = len(
        subject_df
    )

    old_values = (
        subject_df[
            "OLD_f1"
        ]
        .to_numpy(
            dtype=float
        )
    )

    new_values = (
        subject_df[
            "NEW_f1"
        ]
        .to_numpy(
            dtype=float
        )
    )

    old_boot = []
    new_boot = []
    delta_boot = []

    for _ in range(
        BOOTSTRAP_ITERATIONS
    ):

        indices = rng.integers(
            low=0,
            high=n,
            size=n,
        )

        sampled_old = (
            old_values[
                indices
            ]
        )

        sampled_new = (
            new_values[
                indices
            ]
        )

        old_mean = float(
            sampled_old.mean()
        )

        new_mean = float(
            sampled_new.mean()
        )

        delta = (
            new_mean
            - old_mean
        )

        old_boot.append(
            old_mean
        )

        new_boot.append(
            new_mean
        )

        delta_boot.append(
            delta
        )

    return (
        np.asarray(
            old_boot
        ),
        np.asarray(
            new_boot
        ),
        np.asarray(
            delta_boot
        ),
    )


# ============================================================
# BUILD OUTPUT
# ============================================================

def build_output(
    video_metrics,
    subject_metrics,
):

    rng = np.random.default_rng(
        BOOTSTRAP_RANDOM_STATE
    )

    rows = []

    for subject in sorted(
        EXPECTED_SUBJECTS
    ):

        subject_df = (
            video_metrics[
                video_metrics[
                    "subject"
                ].eq(
                    subject
                )
            ]
            .reset_index(
                drop=True
            )
        )

        old_value = float(
            subject_df[
                "OLD_f1"
            ].mean()
        )

        new_value = float(
            subject_df[
                "NEW_f1"
            ].mean()
        )

        delta = (
            new_value
            - old_value
        )

        (
            old_boot,
            new_boot,
            delta_boot,
        ) = bootstrap_subject(
            subject_df,
            rng,
        )

        old_ci_low = float(
            np.percentile(
                old_boot,
                CI_LOW,
            )
        )

        old_ci_high = float(
            np.percentile(
                old_boot,
                CI_HIGH,
            )
        )

        new_ci_low = float(
            np.percentile(
                new_boot,
                CI_LOW,
            )
        )

        new_ci_high = float(
            np.percentile(
                new_boot,
                CI_HIGH,
            )
        )

        delta_ci_low = float(
            np.percentile(
                delta_boot,
                CI_LOW,
            )
        )

        delta_ci_high = float(
            np.percentile(
                delta_boot,
                CI_HIGH,
            )
        )

        subject_summary = (
            subject_metrics[
                subject_metrics[
                    "subject"
                ].eq(
                    subject
                )
            ]
        )

        pairs = int(
            subject_summary[
                "pairs"
            ].iloc[0]
        )

        rows.append(
            {
                "subject":
                    subject,

                "metric":
                    "macro_video_f1",

                "role":
                    "subject_descriptive",

                "n_videos":
                    len(
                        subject_df
                    ),

                "n_pairs":
                    pairs,

                "old_value":
                    old_value,

                "old_ci_low":
                    old_ci_low,

                "old_ci_high":
                    old_ci_high,

                "new_value":
                    new_value,

                "new_ci_low":
                    new_ci_low,

                "new_ci_high":
                    new_ci_high,

                "difference_new_minus_old":
                    delta,

                "delta_ci_low":
                    delta_ci_low,

                "delta_ci_high":
                    delta_ci_high,

                "ci_level":
                    0.95,

                "method":
                    "paired within-subject video bootstrap",

                "resampling_unit":
                    "video",

                "bootstrap_iterations":
                    BOOTSTRAP_ITERATIONS,

                "bootstrap_random_state":
                    BOOTSTRAP_RANDOM_STATE,

                "hypothesis_test":
                    "none",

                "interpretation_status":
                    "descriptive DEV subject analysis",

                "caveat":
                    (
                        "Only 10 DEV videos are "
                        "available in this subject; "
                        "subject-level estimates "
                        "have limited precision and "
                        "are not independent "
                        "confirmatory evidence."
                    ),
            }
        )

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
        "V2 A7 SUBJECT-LEVEL UNCERTAINTY"
    )

    print(
        "=" * 72
    )

    (
        video_metrics,
        subject_metrics,
    ) = load_inputs()

    audit_inputs(
        video_metrics,
        subject_metrics,
    )

    cross_check_subject_metrics(
        video_metrics,
        subject_metrics,
    )

    output = build_output(
        video_metrics,
        subject_metrics,
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
        "Subject Macro Video F1:"
    )

    for row in output.itertuples():

        print(
            f"{row.subject}: "
            f"OLD={row.old_value:.6f}, "
            f"NEW={row.new_value:.6f}, "
            f"delta={row.difference_new_minus_old:+.6f}, "
            f"95% CI=[{row.delta_ci_low:.6f}, "
            f"{row.delta_ci_high:.6f}]"
        )

    print()

    print(
        "Caveat: each subject contains "
        "only 10 DEV videos."
    )

    print(
        "No subject-level hypothesis "
        "tests were added."
    )

    print()

    print(
        "Saved:",
        OUTPUT_FILE,
    )


if __name__ == "__main__":
    main()