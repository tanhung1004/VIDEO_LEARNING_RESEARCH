from pathlib import Path
import sys

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

V2_CODE_ROOT = (
    PROJECT_ROOT
    / "module1"
    / "v2_validation"
)

V2_RESULT_ROOT = (
    PROJECT_ROOT
    / "module1"
    / "results"
    / "v2_validation"
)

DOCS_ROOT = (
    PROJECT_ROOT
    / "docs"
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

SUBJECT_FILE = (
    STEP13_ROOT
    / "old_vs_new_combined_subject_metrics.csv"
)


A5_FILE = (
    V2_RESULT_ROOT
    / "V2_CLUSTERED_STATISTICS.csv"
)

A6_FILE = (
    V2_RESULT_ROOT
    / "V2_SECONDARY_METRIC_UNCERTAINTY.csv"
)

A7_FILE = (
    V2_RESULT_ROOT
    / "V2_SUBJECT_UNCERTAINTY.csv"
)

A8_FILE = (
    V2_RESULT_ROOT
    / "V2_MDE_SENSITIVITY.csv"
)


METRIC_AUDIT_DOC = (
    DOCS_ROOT
    / "V2_METRIC_AUDIT.md"
)

STAT_PROTOCOL_DOC = (
    DOCS_ROOT
    / "V2_STATISTICAL_PROTOCOL.md"
)


OUTPUT_CSV = (
    V2_RESULT_ROOT
    / "V2_FINAL_AUDIT.csv"
)

OUTPUT_SUMMARY = (
    V2_RESULT_ROOT
    / "V2_FINAL_AUDIT_SUMMARY.txt"
)


# ============================================================
# FROZEN EXPECTATIONS
# ============================================================

EXPECTED_VIDEOS = 40
EXPECTED_PAIRS = 390
EXPECTED_POSITIVES = 153
EXPECTED_NEGATIVES = 237

EXPECTED_SUBJECT_COUNTS = {
    "C++": 10,
    "Java": 10,
    "Python": 10,
    "SQL": 10,
}

EXPECTED_SUBJECT_PAIR_COUNTS = {
    "C++": 100,
    "Java": 90,
    "Python": 90,
    "SQL": 110,
}

EXPECTED_OLD_MACRO_F1 = (
    0.5734297034664682
)

EXPECTED_NEW_MACRO_F1 = (
    0.6681238206238206
)

EXPECTED_DELTA_MACRO_F1 = (
    0.09469411715735243
)

EXPECTED_OLD_CONFUSION = {
    "TP": 134,
    "FP": 144,
    "FN": 19,
    "TN": 93,
}

EXPECTED_NEW_CONFUSION = {
    "TP": 116,
    "FP": 79,
    "FN": 37,
    "TN": 158,
}

EXPECTED_A8_POWERS = [
    0.70,
    0.80,
    0.90,
    0.95,
]

TOL = 1e-12


# ============================================================
# AUDIT STORAGE
# ============================================================

AUDIT_ROWS = []


def add_check(
    section,
    check_name,
    status,
    observed="",
    expected="",
    note="",
):

    if status not in {
        "PASS",
        "WARN",
        "FAIL",
    }:
        raise ValueError(
            f"Invalid audit status: {status}"
        )

    AUDIT_ROWS.append(
        {
            "section":
                section,

            "check":
                check_name,

            "status":
                status,

            "observed":
                str(observed),

            "expected":
                str(expected),

            "note":
                note,
        }
    )


def pass_fail(
    condition,
):

    return (
        "PASS"
        if condition
        else "FAIL"
    )


# ============================================================
# FILE EXISTENCE
# ============================================================

def audit_required_files():

    required_code = [
        "05_clustered_statistics.py",
        "06_secondary_metric_uncertainty.py",
        "07_subject_uncertainty.py",
        "08_power_or_mde_analysis.py",
    ]

    required_results = [
        "V2_CLUSTERED_STATISTICS.csv",
        "V2_SECONDARY_METRIC_UNCERTAINTY.csv",
        "V2_SUBJECT_UNCERTAINTY.csv",
        "V2_MDE_SENSITIVITY.csv",
    ]

    for filename in required_code:

        path = (
            V2_CODE_ROOT
            / filename
        )

        exists_and_nonempty = (
            path.exists()
            and path.stat().st_size > 0
        )

        add_check(
            section="files",
            check_name=(
                f"code artifact: {filename}"
            ),
            status=pass_fail(
                exists_and_nonempty
            ),
            observed=(
                path.stat().st_size
                if path.exists()
                else "missing"
            ),
            expected="non-empty file",
        )

    for filename in required_results:

        path = (
            V2_RESULT_ROOT
            / filename
        )

        exists_and_nonempty = (
            path.exists()
            and path.stat().st_size > 0
        )

        add_check(
            section="files",
            check_name=(
                f"result artifact: {filename}"
            ),
            status=pass_fail(
                exists_and_nonempty
            ),
            observed=(
                path.stat().st_size
                if path.exists()
                else "missing"
            ),
            expected="non-empty file",
        )

    for path in [
        METRIC_AUDIT_DOC,
        STAT_PROTOCOL_DOC,
    ]:

        exists_and_nonempty = (
            path.exists()
            and path.stat().st_size > 0
        )

        add_check(
            section="files",
            check_name=(
                f"documentation: {path.name}"
            ),
            status=pass_fail(
                exists_and_nonempty
            ),
            observed=(
                path.stat().st_size
                if path.exists()
                else "missing"
            ),
            expected="non-empty file",
        )


# ============================================================
# LOAD
# ============================================================

def load_data():

    pairs = pd.read_csv(
        PAIR_FILE
    )

    videos = pd.read_csv(
        VIDEO_FILE
    )

    overall = pd.read_csv(
        OVERALL_FILE
    )

    subjects = pd.read_csv(
        SUBJECT_FILE
    )

    a5 = pd.read_csv(
        A5_FILE
    )

    a6 = pd.read_csv(
        A6_FILE
    )

    a7 = pd.read_csv(
        A7_FILE
    )

    a8 = pd.read_csv(
        A8_FILE
    )

    return (
        pairs,
        videos,
        overall,
        subjects,
        a5,
        a6,
        a7,
        a8,
    )


# ============================================================
# STEP 13 FROZEN DATA AUDIT
# ============================================================

def audit_step13(
    pairs,
    videos,
    overall,
    subjects,
):

    # --------------------------------------------------------
    # Pair-space counts
    # --------------------------------------------------------

    add_check(
        "step13",
        "same-subject evaluation pair count",
        pass_fail(
            len(pairs)
            == EXPECTED_PAIRS
        ),
        observed=len(pairs),
        expected=EXPECTED_PAIRS,
    )

    positive_count = int(
        pairs[
            "y_true"
        ].sum()
    )

    negative_count = int(
        len(pairs)
        - positive_count
    )

    add_check(
        "step13",
        "positive ground-truth pairs",
        pass_fail(
            positive_count
            == EXPECTED_POSITIVES
        ),
        observed=positive_count,
        expected=EXPECTED_POSITIVES,
    )

    add_check(
        "step13",
        "negative ground-truth pairs",
        pass_fail(
            negative_count
            == EXPECTED_NEGATIVES
        ),
        observed=negative_count,
        expected=EXPECTED_NEGATIVES,
    )

    if {
        "video_id",
        "concept",
    }.issubset(
        pairs.columns
    ):

        duplicate_pairs = int(
            pairs.duplicated(
                subset=[
                    "video_id",
                    "concept",
                ]
            ).sum()
        )

        add_check(
            "step13",
            "duplicate video-concept rows",
            pass_fail(
                duplicate_pairs == 0
            ),
            observed=duplicate_pairs,
            expected=0,
        )

    # --------------------------------------------------------
    # Video counts
    # --------------------------------------------------------

    unique_videos = int(
        videos[
            "video_id"
        ].nunique()
    )

    add_check(
        "step13",
        "DEV video count",
        pass_fail(
            unique_videos
            == EXPECTED_VIDEOS
        ),
        observed=unique_videos,
        expected=EXPECTED_VIDEOS,
    )

    subject_counts = (
        videos[
            "subject"
        ]
        .value_counts()
        .to_dict()
    )

    add_check(
        "step13",
        "balanced subject video counts",
        pass_fail(
            subject_counts
            == EXPECTED_SUBJECT_COUNTS
        ),
        observed=subject_counts,
        expected=EXPECTED_SUBJECT_COUNTS,
    )

    # --------------------------------------------------------
    # Pair count per subject
    # --------------------------------------------------------

    pair_subject_counts = (
        pairs[
            "subject"
        ]
        .value_counts()
        .to_dict()
    )

    add_check(
        "step13",
        "subject pair-space counts",
        pass_fail(
            pair_subject_counts
            == EXPECTED_SUBJECT_PAIR_COUNTS
        ),
        observed=pair_subject_counts,
        expected=EXPECTED_SUBJECT_PAIR_COUNTS,
    )

    # --------------------------------------------------------
    # Overall system rows
    # --------------------------------------------------------

    expected_systems = {
        "OLD_COMBINED",
        "NEW_COMBINED_F",
    }

    systems = set(
        overall[
            "system"
        ]
    )

    add_check(
        "step13",
        "system identifiers",
        pass_fail(
            systems
            == expected_systems
        ),
        observed=systems,
        expected=expected_systems,
        note=(
            "Historical artifact names are "
            "preserved."
        ),
    )

    old_row = (
        overall[
            overall[
                "system"
            ].eq(
                "OLD_COMBINED"
            )
        ]
        .iloc[0]
    )

    new_row = (
        overall[
            overall[
                "system"
            ].eq(
                "NEW_COMBINED_F"
            )
        ]
        .iloc[0]
    )

    old_macro_f1 = float(
        old_row[
            "macro_video_f1"
        ]
    )

    new_macro_f1 = float(
        new_row[
            "macro_video_f1"
        ]
    )

    delta_macro_f1 = (
        new_macro_f1
        - old_macro_f1
    )

    add_check(
        "step13",
        "OLD Macro Video F1",
        pass_fail(
            np.isclose(
                old_macro_f1,
                EXPECTED_OLD_MACRO_F1,
                atol=TOL,
            )
        ),
        observed=old_macro_f1,
        expected=EXPECTED_OLD_MACRO_F1,
    )

    add_check(
        "step13",
        "NEW Macro Video F1",
        pass_fail(
            np.isclose(
                new_macro_f1,
                EXPECTED_NEW_MACRO_F1,
                atol=TOL,
            )
        ),
        observed=new_macro_f1,
        expected=EXPECTED_NEW_MACRO_F1,
    )

    add_check(
        "step13",
        "Macro Video F1 delta",
        pass_fail(
            np.isclose(
                delta_macro_f1,
                EXPECTED_DELTA_MACRO_F1,
                atol=TOL,
            )
        ),
        observed=delta_macro_f1,
        expected=EXPECTED_DELTA_MACRO_F1,
    )

    # --------------------------------------------------------
    # Confusion counts
    # --------------------------------------------------------

    old_confusion = {
        key:
            int(
                old_row[
                    key
                ]
            )
        for key in [
            "TP",
            "FP",
            "FN",
            "TN",
        ]
    }

    new_confusion = {
        key:
            int(
                new_row[
                    key
                ]
            )
        for key in [
            "TP",
            "FP",
            "FN",
            "TN",
        ]
    }

    add_check(
        "step13",
        "OLD confusion counts",
        pass_fail(
            old_confusion
            == EXPECTED_OLD_CONFUSION
        ),
        observed=old_confusion,
        expected=EXPECTED_OLD_CONFUSION,
    )

    add_check(
        "step13",
        "NEW confusion counts",
        pass_fail(
            new_confusion
            == EXPECTED_NEW_CONFUSION
        ),
        observed=new_confusion,
        expected=EXPECTED_NEW_CONFUSION,
    )

    # --------------------------------------------------------
    # Subject file
    # --------------------------------------------------------

    subject_names = set(
        subjects[
            "subject"
        ]
    )

    add_check(
        "step13",
        "subject metric subjects",
        pass_fail(
            subject_names
            == set(
                EXPECTED_SUBJECT_COUNTS
            )
        ),
        observed=subject_names,
        expected=set(
            EXPECTED_SUBJECT_COUNTS
        ),
    )

    add_check(
        "step13",
        "subject metric row count",
        pass_fail(
            len(subjects) == 8
        ),
        observed=len(subjects),
        expected=8,
        note=(
            "4 subjects x 2 systems."
        ),
    )


# ============================================================
# A5 DEPENDENCE AUDIT
# ============================================================

def audit_a5(
    a5,
    overall,
):

    analyses = set(
        a5[
            "analysis"
        ]
    )

    expected = {
        "macro_video_f1_delta",
        "pair_accuracy_delta",
        "historical_exact_mcnemar",
    }

    add_check(
        "A5",
        "required A5 analyses",
        pass_fail(
            analyses == expected
        ),
        observed=analyses,
        expected=expected,
    )

    primary = (
        a5[
            a5[
                "analysis"
            ].eq(
                "macro_video_f1_delta"
            )
        ]
        .iloc[0]
    )

    add_check(
        "A5",
        "primary endpoint role",
        pass_fail(
            str(
                primary[
                    "role"
                ]
            ).lower()
            == "primary"
        ),
        observed=primary[
            "role"
        ],
        expected="primary",
    )

    add_check(
        "A5",
        "primary resampling unit",
        pass_fail(
            str(
                primary[
                    "resampling_unit"
                ]
            ).lower()
            == "video"
        ),
        observed=primary[
            "resampling_unit"
        ],
        expected="video",
    )

    historical = (
        a5[
            a5[
                "analysis"
            ].eq(
                "historical_exact_mcnemar"
            )
        ]
        .iloc[0]
    )

    historical_status = str(
        historical[
            "v2_inference_status"
        ]
    ).lower()

    add_check(
        "A5",
        "historical McNemar not cluster-aware inference",
        pass_fail(
            (
                "historical"
                in historical_status
            )
            and (
                "not treated"
                in historical_status
            )
            and (
                "cluster-aware"
                in historical_status
            )
        ),
        observed=historical[
            "v2_inference_status"
        ],
        expected=(
            "historical only; not treated "
            "as cluster-aware V2 inference"
        ),
    )

    old_macro = float(
        overall.loc[
            overall[
                "system"
            ].eq(
                "OLD_COMBINED"
            ),
            "macro_video_f1",
        ].iloc[0]
    )

    new_macro = float(
        overall.loc[
            overall[
                "system"
            ].eq(
                "NEW_COMBINED_F"
            ),
            "macro_video_f1",
        ].iloc[0]
    )

    add_check(
        "A5",
        "A5 primary values match Step13",
        pass_fail(
            np.isclose(
                float(
                    primary[
                        "old_value"
                    ]
                ),
                old_macro,
                atol=TOL,
            )
            and
            np.isclose(
                float(
                    primary[
                        "new_value"
                    ]
                ),
                new_macro,
                atol=TOL,
            )
        ),
        observed=(
            float(
                primary[
                    "old_value"
                ]
            ),
            float(
                primary[
                    "new_value"
                ]
            ),
        ),
        expected=(
            old_macro,
            new_macro,
        ),
    )


# ============================================================
# A6 SECONDARY UNCERTAINTY AUDIT
# ============================================================

def audit_a6(
    a6,
):

    expected_metrics = {
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
    }

    observed_metrics = set(
        a6[
            "metric"
        ]
    )

    add_check(
        "A6",
        "secondary metric set",
        pass_fail(
            observed_metrics
            == expected_metrics
        ),
        observed=observed_metrics,
        expected=expected_metrics,
    )

    hypothesis_values = set(
        a6[
            "hypothesis_test"
        ]
        .astype(str)
        .str.lower()
    )

    add_check(
        "A6",
        "no additional secondary hypothesis tests",
        pass_fail(
            hypothesis_values
            == {"none"}
        ),
        observed=hypothesis_values,
        expected={"none"},
        note=(
            "Avoids mass secondary "
            "p-value testing."
        ),
    )

    recall = (
        a6[
            a6[
                "metric"
            ].eq(
                "macro_video_recall"
            )
        ]
        .iloc[0]
    )

    recall_values_finite = all(
        np.isfinite(
            [
                recall[
                    "old_value"
                ],
                recall[
                    "new_value"
                ],
                recall[
                    "difference_new_minus_old"
                ],
                recall[
                    "ci_low"
                ],
                recall[
                    "ci_high"
                ],
            ]
        )
    )

    add_check(
        "A6",
        "Macro Video Recall uncertainty present",
        pass_fail(
            recall_values_finite
        ),
        observed=(
            recall[
                "difference_new_minus_old"
            ],
            recall[
                "ci_low"
            ],
            recall[
                "ci_high"
            ],
        ),
        expected=(
            "finite observed delta "
            "and 95% interval"
        ),
    )

    add_check(
        "A6",
        "Macro Video Recall role",
        pass_fail(
            str(
                recall[
                    "role"
                ]
            ).lower()
            == "secondary"
        ),
        observed=recall[
            "role"
        ],
        expected="secondary",
    )


# ============================================================
# A7 SUBJECT UNCERTAINTY AUDIT
# ============================================================

def audit_a7(
    a7,
    subject_metrics,
):

    subjects = set(
        a7[
            "subject"
        ]
    )

    expected_subjects = set(
        EXPECTED_SUBJECT_COUNTS
    )

    add_check(
        "A7",
        "all four subjects present",
        pass_fail(
            subjects
            == expected_subjects
        ),
        observed=subjects,
        expected=expected_subjects,
    )

    n_values = set(
        a7[
            "n_videos"
        ]
        .astype(int)
    )

    add_check(
        "A7",
        "10 DEV videos per subject",
        pass_fail(
            n_values == {10}
        ),
        observed=n_values,
        expected={10},
    )

    hypothesis_values = set(
        a7[
            "hypothesis_test"
        ]
        .astype(str)
        .str.lower()
    )

    add_check(
        "A7",
        "no subject-level hypothesis tests",
        pass_fail(
            hypothesis_values
            == {"none"}
        ),
        observed=hypothesis_values,
        expected={"none"},
    )

    metric_values = set(
        a7[
            "metric"
        ]
    )

    add_check(
        "A7",
        "subject analysis metric",
        pass_fail(
            metric_values
            == {"macro_video_f1"}
        ),
        observed=metric_values,
        expected={"macro_video_f1"},
    )

    # Cross-check OLD/NEW point estimates
    point_estimates_match = True

    for subject in sorted(
        expected_subjects
    ):

        a7_row = (
            a7[
                a7[
                    "subject"
                ].eq(
                    subject
                )
            ]
            .iloc[0]
        )

        old_expected = float(
            subject_metrics.loc[
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
                ),
                "macro_video_f1",
            ].iloc[0]
        )

        new_expected = float(
            subject_metrics.loc[
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
                ),
                "macro_video_f1",
            ].iloc[0]
        )

        if not (
            np.isclose(
                float(
                    a7_row[
                        "old_value"
                    ]
                ),
                old_expected,
                atol=TOL,
            )
            and
            np.isclose(
                float(
                    a7_row[
                        "new_value"
                    ]
                ),
                new_expected,
                atol=TOL,
            )
        ):
            point_estimates_match = False

    add_check(
        "A7",
        "subject estimates match Step13",
        pass_fail(
            point_estimates_match
        ),
        observed=point_estimates_match,
        expected=True,
    )


# ============================================================
# A8 MDE / SENSITIVITY AUDIT
# ============================================================

def audit_a8(
    a8,
    videos,
):

    observed_powers = (
        a8[
            "target_power"
        ]
        .astype(float)
        .tolist()
    )

    add_check(
        "A8",
        "MDE target power grid",
        pass_fail(
            np.allclose(
                observed_powers,
                EXPECTED_A8_POWERS,
                atol=TOL,
            )
        ),
        observed=observed_powers,
        expected=EXPECTED_A8_POWERS,
    )

    add_check(
        "A8",
        "paired video sample size",
        pass_fail(
            set(
                a8[
                    "n_paired_videos"
                ]
                .astype(int)
            )
            == {40}
        ),
        observed=set(
            a8[
                "n_paired_videos"
            ]
            .astype(int)
        ),
        expected={40},
    )

    add_check(
        "A8",
        "primary endpoint",
        pass_fail(
            set(
                a8[
                    "primary_endpoint"
                ]
            )
            == {
                "macro_video_f1"
            }
        ),
        observed=set(
            a8[
                "primary_endpoint"
            ]
        ),
        expected={
            "macro_video_f1"
        },
    )

    add_check(
        "A8",
        "two-sided alpha 0.05",
        pass_fail(
            set(
                a8[
                    "sidedness"
                ]
            )
            == {
                "two-sided"
            }
            and
            np.allclose(
                a8[
                    "alpha"
                ],
                0.05,
                atol=TOL,
            )
        ),
        observed=(
            set(
                a8[
                    "sidedness"
                ]
            ),
            set(
                a8[
                    "alpha"
                ]
            ),
        ),
        expected=(
            {"two-sided"},
            {0.05},
        ),
    )

    deltas = (
        videos[
            "delta_f1_NEW_minus_OLD"
        ]
        .to_numpy(
            dtype=float
        )
    )

    expected_sd = float(
        np.std(
            deltas,
            ddof=1,
        )
    )

    expected_mean = float(
        np.mean(
            deltas
        )
    )

    add_check(
        "A8",
        "planning SD matches paired DEV deltas",
        pass_fail(
            np.allclose(
                a8[
                    "planning_sd_paired_difference"
                ],
                expected_sd,
                atol=TOL,
            )
        ),
        observed=set(
            a8[
                "planning_sd_paired_difference"
            ]
        ),
        expected=expected_sd,
    )

    add_check(
        "A8",
        "observed DEV mean delta cross-check",
        pass_fail(
            np.allclose(
                a8[
                    "observed_dev_mean_delta"
                ],
                expected_mean,
                atol=TOL,
            )
        ),
        observed=set(
            a8[
                "observed_dev_mean_delta"
            ]
        ),
        expected=expected_mean,
    )

    mdes = (
        a8[
            "mde_absolute_f1"
        ]
        .to_numpy(
            dtype=float
        )
    )

    add_check(
        "A8",
        "MDE increases with target power",
        pass_fail(
            np.all(
                np.diff(
                    mdes
                )
                > 0
            )
        ),
        observed=mdes.tolist(),
        expected=(
            "strictly increasing"
        ),
    )

    inference_text = " ".join(
        a8[
            "inference_status"
        ]
        .astype(str)
        .str.lower()
        .tolist()
    )

    notes_text = " ".join(
        a8[
            "note"
        ]
        .astype(str)
        .str.lower()
        .tolist()
    )

    add_check(
        "A8",
        "MDE is planning/sensitivity only",
        pass_fail(
            "planning"
            in inference_text
            and
            "sensitivity"
            in inference_text
            and
            "not post-hoc power"
            in notes_text
            and
            "not independent confirmatory"
            in notes_text
        ),
        observed=(
            a8[
                "inference_status"
            ]
            .iloc[0]
        ),
        expected=(
            "planning/sensitivity; "
            "not post-hoc power; "
            "not independent confirmation"
        ),
    )


# ============================================================
# DOCUMENTATION AUDIT
# ============================================================

def read_doc(
    path,
):

    if not path.exists():
        return ""

    return (
        path
        .read_text(
            encoding="utf-8",
            errors="replace",
        )
        .lower()
    )


def doc_contains_any(
    text,
    phrases,
):

    return any(
        phrase.lower()
        in text
        for phrase in phrases
    )


def audit_docs():

    metric_text = read_doc(
        METRIC_AUDIT_DOC
    )

    protocol_text = read_doc(
        STAT_PROTOCOL_DOC
    )

    metric_checks = [
        (
            "metric audit mentions Macro Video metrics",
            [
                "macro video",
                "macro_video",
            ],
        ),
        (
            "metric audit mentions Micro metrics",
            [
                "micro",
            ],
        ),
        (
            "metric audit documents aggregation",
            [
                "top2_mean",
                "aggregation",
            ],
        ),
        (
            "metric audit documents thresholding",
            [
                "threshold",
            ],
        ),
        (
            "metric audit documents 390-pair universe",
            [
                "390",
                "same-subject",
                "same subject",
            ],
        ),
    ]

    for (
        name,
        phrases,
    ) in metric_checks:

        found = doc_contains_any(
            metric_text,
            phrases,
        )

        add_check(
            "documentation",
            name,
            (
                "PASS"
                if found
                else "WARN"
            ),
            observed=found,
            expected=True,
            note=(
                "WARN means the audit artifact "
                "may be correct but the written "
                "documentation should be expanded."
            ),
        )

    protocol_checks = [
        (
            "protocol identifies DEV role",
            [
                "dev",
                "development",
            ],
        ),
        (
            "protocol identifies primary endpoint",
            [
                "macro video f1",
                "macro_video_f1",
            ],
        ),
        (
            "protocol distinguishes secondary metrics",
            [
                "secondary",
            ],
        ),
        (
            "protocol limits independent confirmatory claims",
            [
                "independent",
                "confirmatory",
            ],
        ),
        (
            "protocol addresses multiple comparisons",
            [
                "multiple comparison",
                "multiple comparisons",
            ],
        ),
        (
            "protocol documents no further DEV tuning",
            [
                "no further",
                "further dev tuning",
                "no retune",
                "no re-tune",
                "retuning",
            ],
        ),
    ]

    for (
        name,
        phrases,
    ) in protocol_checks:

        found = doc_contains_any(
            protocol_text,
            phrases,
        )

        add_check(
            "documentation",
            name,
            (
                "PASS"
                if found
                else "WARN"
            ),
            observed=found,
            expected=True,
            note=(
                "Missing wording does not change "
                "numeric artifacts, but should be "
                "resolved before final reporting."
            ),
        )

    naming_groups = {
        "P1 frozen transcript comparator": [
            "p1 frozen transcript comparator",
        ],

        "OLD combined system": [
            "old combined system",
        ],

        "NEW combined system": [
            "new combined system",
        ],
    }

    for label, phrases in (
        naming_groups.items()
    ):

        found = doc_contains_any(
            protocol_text,
            phrases,
        )

        add_check(
            "documentation",
            f"V2 naming documented: {label}",
            (
                "PASS"
                if found
                else "WARN"
            ),
            observed=found,
            expected=True,
            note=(
                "Historical V1 filenames should "
                "remain unchanged."
            ),
        )


# ============================================================
# CROSS-CUTTING SCIENTIFIC RULES
# ============================================================

def audit_cross_cutting(
    a5,
    a6,
    a7,
    a8,
):

    # Primary endpoint remains Macro Video F1.
    primary_rows = (
        a5[
            a5[
                "role"
            ].astype(str)
            .str.lower()
            .eq(
                "primary"
            )
        ]
    )

    primary_is_macro_f1 = (
        len(primary_rows) == 1
        and
        primary_rows[
            "analysis"
        ]
        .iloc[0]
        == "macro_video_f1_delta"
    )

    add_check(
        "scientific_rules",
        "single primary endpoint retained",
        pass_fail(
            primary_is_macro_f1
        ),
        observed=(
            primary_rows[
                "analysis"
            ]
            .tolist()
        ),
        expected=[
            "macro_video_f1_delta"
        ],
    )

    # Secondary analyses contain no newly-added hypothesis tests.
    no_a6_tests = (
        set(
            a6[
                "hypothesis_test"
            ]
            .astype(str)
            .str.lower()
        )
        == {"none"}
    )

    no_a7_tests = (
        set(
            a7[
                "hypothesis_test"
            ]
            .astype(str)
            .str.lower()
        )
        == {"none"}
    )

    add_check(
        "scientific_rules",
        "no mass secondary/subject hypothesis testing",
        pass_fail(
            no_a6_tests
            and no_a7_tests
        ),
        observed=(
            no_a6_tests,
            no_a7_tests,
        ),
        expected=(
            True,
            True,
        ),
    )

    # Historical McNemar is retained but explicitly limited.
    historical = (
        a5[
            a5[
                "analysis"
            ].eq(
                "historical_exact_mcnemar"
            )
        ]
        .iloc[0]
    )

    status_text = str(
        historical[
            "v2_inference_status"
        ]
    ).lower()

    add_check(
        "scientific_rules",
        "pair-level McNemar limited to historical role",
        pass_fail(
            "historical"
            in status_text
            and
            "not treated"
            in status_text
            and
            "cluster-aware"
            in status_text
        ),
        observed=historical[
            "v2_inference_status"
        ],
        expected=(
            "historical result only; "
            "not cluster-aware V2 inference"
        ),
    )

    # A8 explicitly says no independent confirmation.
    a8_notes = " ".join(
        a8[
            "note"
        ]
        .astype(str)
        .str.lower()
    )

    add_check(
        "scientific_rules",
        "MDE does not claim independent confirmation",
        pass_fail(
            "not independent confirmatory"
            in a8_notes
        ),
        observed=(
            "not independent confirmatory"
            in a8_notes
        ),
        expected=True,
    )


# ============================================================
# SAVE REPORT
# ============================================================

def save_report():

    V2_RESULT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    audit_df = pd.DataFrame(
        AUDIT_ROWS
    )

    audit_df.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    pass_count = int(
        (
            audit_df[
                "status"
            ]
            == "PASS"
        ).sum()
    )

    warn_count = int(
        (
            audit_df[
                "status"
            ]
            == "WARN"
        ).sum()
    )

    fail_count = int(
        (
            audit_df[
                "status"
            ]
            == "FAIL"
        ).sum()
    )

    if fail_count > 0:
        overall_status = "FAIL"

    elif warn_count > 0:
        overall_status = (
            "PASS_WITH_WARNINGS"
        )

    else:
        overall_status = "PASS"

    lines = []

    lines.append(
        "=" * 72
    )

    lines.append(
        "V2 FINAL AUDIT SUMMARY"
    )

    lines.append(
        "=" * 72
    )

    lines.append(
        f"Overall status: {overall_status}"
    )

    lines.append(
        f"PASS: {pass_count}"
    )

    lines.append(
        f"WARN: {warn_count}"
    )

    lines.append(
        f"FAIL: {fail_count}"
    )

    lines.append("")

    lines.append(
        "Frozen evaluation universe:"
    )

    lines.append(
        "  DEV videos: 40"
    )

    lines.append(
        "  Same-subject video-concept pairs: 390"
    )

    lines.append(
        "  Positive pairs: 153"
    )

    lines.append(
        "  Negative pairs: 237"
    )

    lines.append(
        "  Videos per subject: 10"
    )

    lines.append("")

    lines.append(
        "Primary endpoint:"
    )

    lines.append(
        "  Macro Video F1"
    )

    lines.append(
        "  OLD combined system: 0.573430"
    )

    lines.append(
        "  NEW combined system: 0.668124"
    )

    lines.append(
        "  DEV delta: +0.094694"
    )

    lines.append("")

    lines.append(
        "Interpretation constraints:"
    )

    lines.append(
        "  - 40 videos are DEV, not an "
        "independent final test set."
    )

    lines.append(
        "  - Historical pair-level McNemar "
        "is not treated as cluster-aware "
        "V2 inference."
    )

    lines.append(
        "  - Secondary uncertainty is "
        "reported without adding mass "
        "hypothesis tests."
    )

    lines.append(
        "  - Subject analyses are descriptive "
        "with only 10 DEV videos per subject."
    )

    lines.append(
        "  - MDE is sensitivity/planning "
        "analysis, not post-hoc power."
    )

    if warn_count > 0:

        lines.append("")

        lines.append(
            "Warnings requiring review:"
        )

        warning_rows = (
            audit_df[
                audit_df[
                    "status"
                ].eq(
                    "WARN"
                )
            ]
        )

        for row in (
            warning_rows.itertuples()
        ):

            lines.append(
                f"  - [{row.section}] "
                f"{row.check}"
            )

    if fail_count > 0:

        lines.append("")

        lines.append(
            "Failed checks:"
        )

        failure_rows = (
            audit_df[
                audit_df[
                    "status"
                ].eq(
                    "FAIL"
                )
            ]
        )

        for row in (
            failure_rows.itertuples()
        ):

            lines.append(
                f"  - [{row.section}] "
                f"{row.check}"
            )

    summary_text = (
        "\n".join(
            lines
        )
        + "\n"
    )

    OUTPUT_SUMMARY.write_text(
        summary_text,
        encoding="utf-8",
    )

    return (
        audit_df,
        overall_status,
        pass_count,
        warn_count,
        fail_count,
        summary_text,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 72
    )

    print(
        "V2 A9 FINAL AUDIT"
    )

    print(
        "=" * 72
    )

    audit_required_files()

    (
        pairs,
        videos,
        overall,
        subjects,
        a5,
        a6,
        a7,
        a8,
    ) = load_data()

    audit_step13(
        pairs,
        videos,
        overall,
        subjects,
    )

    audit_a5(
        a5,
        overall,
    )

    audit_a6(
        a6,
    )

    audit_a7(
        a7,
        subjects,
    )

    audit_a8(
        a8,
        videos,
    )

    audit_docs()

    audit_cross_cutting(
        a5,
        a6,
        a7,
        a8,
    )

    (
        audit_df,
        overall_status,
        pass_count,
        warn_count,
        fail_count,
        summary_text,
    ) = save_report()

    print()

    print(
        summary_text
    )

    print(
        "Saved:"
    )

    print(
        f"  {OUTPUT_CSV}"
    )

    print(
        f"  {OUTPUT_SUMMARY}"
    )

    if fail_count > 0:

        print()

        print(
            "FINAL AUDIT FAILED."
        )

        print(
            "Resolve FAIL items before "
            "closing Track A."
        )

        sys.exit(1)

    if warn_count > 0:

        print()

        print(
            "FINAL AUDIT PASSED WITH WARNINGS."
        )

        print(
            "Numeric/statistical artifacts "
            "passed, but documentation "
            "warnings should be reviewed."
        )

    else:

        print()

        print(
            "FINAL AUDIT PASS."
        )

        print(
            "Track A artifacts and protocol "
            "checks are internally consistent."
        )


if __name__ == "__main__":
    main()