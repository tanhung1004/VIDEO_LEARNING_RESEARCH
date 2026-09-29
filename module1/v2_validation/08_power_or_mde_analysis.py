from pathlib import Path
import math

import numpy as np
import pandas as pd
from scipy import optimize, stats


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

OUTPUT_DIR = (
    PROJECT_ROOT
    / "module1"
    / "results"
    / "v2_validation"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "V2_MDE_SENSITIVITY.csv"
)


# ============================================================
# FIXED ANALYSIS ASSUMPTIONS
# ============================================================

EXPECTED_N = 40

ALPHA = 0.05
SIDEDNESS = "two-sided"

TARGET_POWERS = [
    0.70,
    0.80,
    0.90,
    0.95,
]

PRIMARY_TARGET_POWER = 0.80

TOL = 1e-12


# ============================================================
# LOAD
# ============================================================

def load_video_metrics():

    return pd.read_csv(
        VIDEO_FILE
    )


# ============================================================
# INPUT AUDIT
# ============================================================

def audit_video_metrics(
    df,
):

    required_columns = {
        "video_id",
        "subject",
        "OLD_f1",
        "NEW_f1",
        "delta_f1_NEW_minus_OLD",
    }

    missing = (
        required_columns
        - set(df.columns)
    )

    if missing:
        raise RuntimeError(
            "Missing required columns: "
            f"{sorted(missing)}"
        )

    if len(df) != EXPECTED_N:
        raise RuntimeError(
            f"Expected {EXPECTED_N} videos, "
            f"observed {len(df)}."
        )

    if (
        df["video_id"]
        .nunique()
        != EXPECTED_N
    ):
        raise RuntimeError(
            "Expected 40 unique videos."
        )

    recalculated_delta = (
        df["NEW_f1"]
        - df["OLD_f1"]
    )

    stored_delta = (
        df[
            "delta_f1_NEW_minus_OLD"
        ]
    )

    if not np.allclose(
        recalculated_delta,
        stored_delta,
        atol=TOL,
    ):
        raise RuntimeError(
            "Stored paired F1 deltas do not "
            "match NEW_f1 - OLD_f1."
        )

    subject_counts = (
        df[
            "subject"
        ]
        .value_counts()
        .to_dict()
    )

    expected_subject_counts = {
        "C++": 10,
        "Java": 10,
        "Python": 10,
        "SQL": 10,
    }

    if (
        subject_counts
        != expected_subject_counts
    ):
        raise RuntimeError(
            "Unexpected subject counts: "
            f"{subject_counts}"
        )

    print(
        "Input audit: PASS"
    )

    print(
        "DEV videos:",
        len(df),
    )

    print(
        "Subject counts:",
        subject_counts,
    )


# ============================================================
# PAIRED T-TEST POWER
# ============================================================

def paired_t_power(
    absolute_delta,
    sd_delta,
    n,
    alpha,
):

    if absolute_delta < 0:
        raise ValueError(
            "absolute_delta must be >= 0."
        )

    if sd_delta <= 0:
        raise ValueError(
            "sd_delta must be > 0."
        )

    if n <= 1:
        raise ValueError(
            "n must be greater than 1."
        )

    df = (
        n
        - 1
    )

    standard_error = (
        sd_delta
        / math.sqrt(
            n
        )
    )

    noncentrality = (
        absolute_delta
        / standard_error
    )

    critical_t = (
        stats.t.ppf(
            1.0
            - alpha / 2.0,
            df,
        )
    )

    upper_tail = (
        stats.nct.sf(
            critical_t,
            df,
            noncentrality,
        )
    )

    lower_tail = (
        stats.nct.cdf(
            -critical_t,
            df,
            noncentrality,
        )
    )

    power = float(
        upper_tail
        + lower_tail
    )

    return power


# ============================================================
# MDE SOLVER
# ============================================================

def find_mde(
    target_power,
    sd_delta,
    n,
    alpha,
):

    if not (
        0.0
        < target_power
        < 1.0
    ):
        raise ValueError(
            "target_power must be between "
            "0 and 1."
        )

    def objective(
        absolute_delta,
    ):

        power = paired_t_power(
            absolute_delta=absolute_delta,
            sd_delta=sd_delta,
            n=n,
            alpha=alpha,
        )

        if not np.isfinite(
            power
        ):
            raise RuntimeError(
                "Non-finite power encountered "
                "during MDE solving at "
                f"delta={absolute_delta:.12f}"
            )

        return (
            power
            - target_power
        )

    lower = 0.0

    # One paired-difference SD is already a
    # large enough upper bracket for n=40.
    # Keeping the search interval bounded
    # avoids unstable extreme noncentral-t
    # evaluations in some SciPy builds.
    upper = float(
        sd_delta
    )

    power_at_lower = (
        paired_t_power(
            absolute_delta=lower,
            sd_delta=sd_delta,
            n=n,
            alpha=alpha,
        )
    )

    power_at_upper = (
        paired_t_power(
            absolute_delta=upper,
            sd_delta=sd_delta,
            n=n,
            alpha=alpha,
        )
    )

    if not np.isfinite(
        power_at_lower
    ):
        raise RuntimeError(
            "Power at lower MDE bound "
            "is non-finite."
        )

    if not np.isfinite(
        power_at_upper
    ):
        raise RuntimeError(
            "Power at upper MDE bound "
            "is non-finite."
        )

    if (
        power_at_lower
        > target_power
    ):
        raise RuntimeError(
            "Target power is already exceeded "
            "at delta=0, so the MDE root "
            "cannot be bracketed as expected."
        )

    if (
        power_at_upper
        < target_power
    ):
        raise RuntimeError(
            "Initial MDE bracket is "
            "insufficient. "
            f"power_at_upper="
            f"{power_at_upper:.6f}, "
            f"target_power="
            f"{target_power:.6f}"
        )

    mde = optimize.brentq(
        objective,
        lower,
        upper,
    )

    return float(
        mde
    )


# ============================================================
# BUILD OUTPUT
# ============================================================

def build_output(
    video_metrics,
):

    deltas = (
        video_metrics[
            "delta_f1_NEW_minus_OLD"
        ]
        .to_numpy(
            dtype=float
        )
    )

    n = len(
        deltas
    )

    mean_delta = float(
        np.mean(
            deltas
        )
    )

    sd_delta = float(
        np.std(
            deltas,
            ddof=1,
        )
    )

    rows = []

    for target_power in TARGET_POWERS:

        mde = find_mde(
            target_power=target_power,
            sd_delta=sd_delta,
            n=n,
            alpha=ALPHA,
        )

        standardized_mde = (
            mde
            / sd_delta
        )

        rows.append(
            {
                "analysis":
                    "paired_macro_video_f1_mde",

                "primary_endpoint":
                    "macro_video_f1",

                "comparison":
                    (
                        "NEW combined system "
                        "minus OLD combined system"
                    ),

                "n_paired_videos":
                    n,

                "alpha":
                    ALPHA,

                "sidedness":
                    SIDEDNESS,

                "target_power":
                    target_power,

                "planning_sd_paired_difference":
                    sd_delta,

                "mde_absolute_f1":
                    mde,

                "mde_standardized_dz":
                    standardized_mde,

                "observed_dev_mean_delta":
                    mean_delta,

                "method":
                    (
                        "paired t-test sensitivity "
                        "using noncentral t "
                        "distribution"
                    ),

                "inference_status":
                    (
                        "planning/sensitivity "
                        "description only"
                    ),

                "note":
                    (
                        "The planning SD is estimated "
                        "from the same 40-video DEV "
                        "set. This is not post-hoc "
                        "power and is not independent "
                        "confirmatory evidence."
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# OUTPUT AUDIT
# ============================================================

def audit_output(
    output,
):

    if len(
        output
    ) != len(
        TARGET_POWERS
    ):
        raise RuntimeError(
            "Unexpected number of MDE rows."
        )

    if (
        output[
            "target_power"
        ]
        .tolist()
        != TARGET_POWERS
    ):
        raise RuntimeError(
            "Target power rows do not match "
            "the predefined sensitivity grid."
        )

    numeric_columns = [
        "planning_sd_paired_difference",
        "mde_absolute_f1",
        "mde_standardized_dz",
        "observed_dev_mean_delta",
    ]

    for column in numeric_columns:

        values = (
            output[
                column
            ]
            .to_numpy(
                dtype=float
            )
        )

        if not np.all(
            np.isfinite(
                values
            )
        ):
            raise RuntimeError(
                f"Non-finite values found in "
                f"{column}."
            )

    mdes = (
        output[
            "mde_absolute_f1"
        ]
        .to_numpy(
            dtype=float
        )
    )

    if not np.all(
        np.diff(
            mdes
        )
        > 0
    ):
        raise RuntimeError(
            "MDE should increase as target "
            "power increases."
        )

    print(
        "Output audit: PASS"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 72
    )

    print(
        "V2 A8 MDE / SENSITIVITY ANALYSIS"
    )

    print(
        "=" * 72
    )

    video_metrics = (
        load_video_metrics()
    )

    audit_video_metrics(
        video_metrics
    )

    deltas = (
        video_metrics[
            "delta_f1_NEW_minus_OLD"
        ]
    )

    mean_delta = float(
        deltas.mean()
    )

    sd_delta = float(
        deltas.std(
            ddof=1
        )
    )

    print()

    print(
        "Paired Macro Video F1 deltas:"
    )

    print(
        f"  n = {len(deltas)}"
    )

    print(
        f"  mean = "
        f"{mean_delta:.6f}"
    )

    print(
        f"  SD = "
        f"{sd_delta:.6f}"
    )

    print(
        f"  min = "
        f"{deltas.min():.6f}"
    )

    print(
        f"  max = "
        f"{deltas.max():.6f}"
    )

    output = build_output(
        video_metrics
    )

    audit_output(
        output
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
        "Minimum Detectable Effect:"
    )

    for row in output.itertuples():

        marker = ""

        if np.isclose(
            row.target_power,
            PRIMARY_TARGET_POWER,
        ):
            marker = (
                "  <-- primary sensitivity"
            )

        print(
            f"  power="
            f"{row.target_power:.0%}: "
            f"MDE="
            f"{row.mde_absolute_f1:.6f}, "
            f"dz="
            f"{row.mde_standardized_dz:.6f}"
            f"{marker}"
        )

    print()

    primary_row = (
        output[
            np.isclose(
                output[
                    "target_power"
                ],
                PRIMARY_TARGET_POWER,
            )
        ]
        .iloc[0]
    )

    print(
        "Primary 80% sensitivity:"
    )

    print(
        f"  MDE = "
        f"{primary_row['mde_absolute_f1']:.6f}"
    )

    print(
        f"  standardized dz = "
        f"{primary_row['mde_standardized_dz']:.6f}"
    )

    print()

    print(
        "Interpretation:"
    )

    print(
        "  This is a sensitivity / planning "
        "analysis, not post-hoc power."
    )

    print(
        "  The paired-difference SD is "
        "estimated from the same DEV set."
    )

    print(
        "  No independent confirmatory claim "
        "is made."
    )

    print()

    print(
        "Saved:",
        OUTPUT_FILE,
    )


if __name__ == "__main__":
    main()