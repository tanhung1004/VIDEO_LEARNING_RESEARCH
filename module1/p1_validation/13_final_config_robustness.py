from pathlib import Path
import importlib.util
import json

import numpy as np
import pandas as pd

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
)


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

P1_DIR = (
    ROOT
    / "module1"
    / "p1_validation"
)

RESULT_DIR = (
    ROOT
    / "module1"
    / "results"
    / "p1_validation"
)

CONFIG_PATH = (
    RESULT_DIR
    / "final_dev_hyperparameters.json"
)

SEARCH_PATH = (
    RESULT_DIR
    / "final_dev_search_results.csv"
)

RAW_PATH = (
    RESULT_DIR
    / "final_dev_raw_oof_scores.csv"
)

GT_PATH = (
    ROOT
    / "data"
    / "ground_truth"
    / "ground_truth_concepts.csv"
)

PREDICTIONS_PATH = (
    RESULT_DIR
    / "final_config_predictions.csv"
)

PER_VIDEO_PATH = (
    RESULT_DIR
    / "final_config_per_video_metrics.csv"
)

SUBJECT_PATH = (
    RESULT_DIR
    / "final_config_subject_metrics.csv"
)

FOLD_PATH = (
    RESULT_DIR
    / "final_config_fold_metrics.csv"
)

LOCAL_SENSITIVITY_PATH = (
    RESULT_DIR
    / "final_config_local_sensitivity.csv"
)

AUDIT_PATH = (
    RESULT_DIR
    / "final_config_robustness_audit.json"
)


BOOTSTRAP_ITERATIONS = 10000
RANDOM_STATE = 42


# ============================================================
# LOAD VALIDATED P1.7 FUNCTIONS
# ============================================================

def load_python_module(name, path):

    spec = (
        importlib.util
        .spec_from_file_location(
            name,
            path,
        )
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"Cannot import {path}"
        )

    module = (
        importlib.util
        .module_from_spec(
            spec
        )
    )

    spec.loader.exec_module(
        module
    )

    return module


TUNING = load_python_module(
    "p1_nested_tuning",
    P1_DIR
    / "07_nested_inner_tuning.py",
)


# ============================================================
# HELPERS
# ============================================================

def require_file(path):

    if not path.exists():
        raise FileNotFoundError(
            f"Missing required file: "
            f"{path.relative_to(ROOT)}"
        )


def normalize_video_id(series):

    return (
        series
        .astype(str)
        .str.strip()
        .str.lower()
    )


def json_default(obj):

    if isinstance(obj, np.generic):
        return obj.item()

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    raise TypeError(
        f"Object of type "
        f"{obj.__class__.__name__} "
        "is not JSON serializable"
    )


def build_evaluation(
    aggregated,
    positive_gt,
    threshold,
):

    evaluation = (
        aggregated.copy()
    )

    evaluation["y_true"] = [
        1
        if (
            row.video_id,
            row.concept,
        ) in positive_gt
        else 0
        for row in (
            evaluation.itertuples()
        )
    ]

    evaluation["y_pred"] = (
        evaluation[
            "video_score"
        ]
        >= threshold
    ).astype(int)

    return evaluation


def calculate_metrics(
    evaluation,
):

    per_video_rows = []

    for (
        video_id,
        subject
    ), group in (
        evaluation.groupby(
            [
                "video_id",
                "subject",
            ]
        )
    ):

        y_true = (
            group["y_true"]
            .astype(int)
        )

        y_pred = (
            group["y_pred"]
            .astype(int)
        )

        per_video_rows.append(
            {
                "video_id":
                    video_id,

                "subject":
                    subject,

                "precision":
                    precision_score(
                        y_true,
                        y_pred,
                        zero_division=0,
                    ),

                "recall":
                    recall_score(
                        y_true,
                        y_pred,
                        zero_division=0,
                    ),

                "f1":
                    f1_score(
                        y_true,
                        y_pred,
                        zero_division=0,
                    ),
            }
        )

    per_video = pd.DataFrame(
        per_video_rows
    )

    y_true_all = (
        evaluation[
            "y_true"
        ].astype(int)
    )

    y_pred_all = (
        evaluation[
            "y_pred"
        ].astype(int)
    )

    metrics = {
        "macro_video_precision":
            float(
                per_video[
                    "precision"
                ].mean()
            ),

        "macro_video_recall":
            float(
                per_video[
                    "recall"
                ].mean()
            ),

        "macro_video_f1":
            float(
                per_video[
                    "f1"
                ].mean()
            ),

        "micro_precision":
            float(
                precision_score(
                    y_true_all,
                    y_pred_all,
                    zero_division=0,
                )
            ),

        "micro_recall":
            float(
                recall_score(
                    y_true_all,
                    y_pred_all,
                    zero_division=0,
                )
            ),

        "micro_f1":
            float(
                f1_score(
                    y_true_all,
                    y_pred_all,
                    zero_division=0,
                )
            ),

        "true_positive_labels":
            int(
                evaluation[
                    "y_true"
                ].sum()
            ),

        "predicted_positive_labels":
            int(
                evaluation[
                    "y_pred"
                ].sum()
            ),
    }

    return metrics, per_video


def stratified_bootstrap_ci(
    per_video,
):

    rng = np.random.default_rng(
        RANDOM_STATE
    )

    subjects = sorted(
        per_video[
            "subject"
        ].unique()
    )

    bootstrap_means = []

    for _ in range(
        BOOTSTRAP_ITERATIONS
    ):

        sampled_parts = []

        for subject in subjects:

            subject_rows = (
                per_video[
                    per_video[
                        "subject"
                    ].eq(subject)
                ]
            )

            indices = (
                rng.integers(
                    low=0,
                    high=len(
                        subject_rows
                    ),
                    size=len(
                        subject_rows
                    ),
                )
            )

            sampled_parts.append(
                subject_rows
                .iloc[indices]
            )

        sampled = pd.concat(
            sampled_parts,
            ignore_index=True,
        )

        bootstrap_means.append(
            float(
                sampled[
                    "f1"
                ].mean()
            )
        )

    lower = float(
        np.percentile(
            bootstrap_means,
            2.5,
        )
    )

    upper = float(
        np.percentile(
            bootstrap_means,
            97.5,
        )
    )

    return lower, upper


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 82)
    print(
        "P1.13 FINAL CONFIG "
        "ROBUSTNESS AUDIT"
    )
    print("=" * 82)

    for path in [
        CONFIG_PATH,
        SEARCH_PATH,
        RAW_PATH,
        GT_PATH,
    ]:
        require_file(path)

    # --------------------------------------------------------
    # Load selected P1.12 configuration
    # --------------------------------------------------------

    config = json.loads(
        CONFIG_PATH.read_text(
            encoding="utf-8"
        )
    )

    chunk_seconds = int(
        config[
            "chunk_seconds"
        ]
    )

    alpha = float(
        config[
            "alpha"
        ]
    )

    aggregation = str(
        config[
            "aggregation"
        ]
    )

    threshold = float(
        config[
            "threshold"
        ]
    )

    print(
        "\nSelected P1.12 config:"
    )

    print(
        f"chunk       = "
        f"{chunk_seconds}s"
    )

    print(
        f"alpha       = "
        f"{alpha}"
    )

    print(
        f"LSA weight  = "
        f"{1.0-alpha}"
    )

    print(
        f"aggregation = "
        f"{aggregation}"
    )

    print(
        f"threshold   = "
        f"{threshold}"
    )

    # --------------------------------------------------------
    # Load search results
    # --------------------------------------------------------

    search = pd.read_csv(
        SEARCH_PATH
    )

    selected_flag = (
        search[
            "selected_final_dev_config"
        ]
    )

    if selected_flag.dtype == bool:
        selected_mask = selected_flag
    else:
        selected_mask = (
            selected_flag
            .astype(str)
            .str.strip()
            .str.lower()
            .eq("true")
        )

    selected_rows = (
        search[
            selected_mask
        ]
    )

    if len(selected_rows) != 1:
        raise ValueError(
            "Expected exactly one "
            "selected P1.12 config"
        )

    selected_search = (
        selected_rows.iloc[0]
    )

    # --------------------------------------------------------
    # Load raw OOF scores
    # --------------------------------------------------------

    raw = pd.read_csv(
        RAW_PATH
    )

    raw[
        "video_id"
    ] = normalize_video_id(
        raw[
            "video_id"
        ]
    )

    raw[
        "chunk_seconds"
    ] = (
        pd.to_numeric(
            raw[
                "chunk_seconds"
            ],
            errors="raise",
        )
        .astype(int)
    )

    selected_raw = (
        raw[
            raw[
                "chunk_seconds"
            ].eq(
                chunk_seconds
            )
        ]
        .copy()
    )

    if (
        selected_raw[
            "video_id"
        ].nunique()
        != 40
    ):
        raise ValueError(
            "Selected chunk scores "
            "do not cover 40 videos"
        )

    # --------------------------------------------------------
    # GT
    # --------------------------------------------------------

    gt = pd.read_csv(
        GT_PATH
    )

    gt[
        "video_id"
    ] = normalize_video_id(
        gt[
            "video_id"
        ]
    )

    gt[
        "concept"
    ] = (
        gt[
            "concept"
        ]
        .astype(str)
        .str.strip()
    )

    positive_gt = set(
        zip(
            gt[
                "video_id"
            ],
            gt[
                "concept"
            ],
        )
    )

    if len(positive_gt) != 153:
        raise ValueError(
            "Expected 153 "
            "positive GT pairs"
        )

    # ========================================================
    # 1. REPRODUCE SELECTED CONFIG
    # ========================================================

    print(
        "\n1. SELECTED CONFIG "
        "REPRODUCTION"
    )

    aggregated = (
        TUNING
        .aggregate_scores(
            selected_raw,
            alpha,
            aggregation,
        )
    )

    evaluation = (
        build_evaluation(
            aggregated,
            positive_gt,
            threshold,
        )
    )

    metrics, per_video = (
        calculate_metrics(
            evaluation
        )
    )

    if len(evaluation) != 390:
        raise ValueError(
            "Expected 390 "
            "video-concept pairs"
        )

    expected_f1 = float(
        selected_search[
            "macro_video_f1"
        ]
    )

    if not np.isclose(
        metrics[
            "macro_video_f1"
        ],
        expected_f1,
        atol=1e-12,
        rtol=0.0,
    ):
        raise ValueError(
            "P1.13 does not reproduce "
            "P1.12 selected macro F1"
        )

    print(
        "P1.12 macro F1 : "
        f"{expected_f1:.6f}"
    )

    print(
        "P1.13 macro F1 : "
        f"{metrics['macro_video_f1']:.6f}"
    )

    print(
        "Reproduction   : PASS"
    )

    # ========================================================
    # 2. OVERALL PERFORMANCE PROFILE
    # ========================================================

    print(
        "\n2. OVERALL PERFORMANCE"
    )

    print(
        "Macro precision : "
        f"{metrics['macro_video_precision']:.6f}"
    )

    print(
        "Macro recall    : "
        f"{metrics['macro_video_recall']:.6f}"
    )

    print(
        "Macro F1        : "
        f"{metrics['macro_video_f1']:.6f}"
    )

    print(
        "Micro precision : "
        f"{metrics['micro_precision']:.6f}"
    )

    print(
        "Micro recall    : "
        f"{metrics['micro_recall']:.6f}"
    )

    print(
        "Micro F1        : "
        f"{metrics['micro_f1']:.6f}"
    )

    # ========================================================
    # 3. BOOTSTRAP 95% CI
    # ========================================================

    print(
        "\n3. DESCRIPTIVE "
        "STRATIFIED BOOTSTRAP CI"
    )

    ci_lower, ci_upper = (
        stratified_bootstrap_ci(
            per_video
        )
    )

    print(
        "Macro video F1 95% CI: "
        f"[{ci_lower:.6f}, "
        f"{ci_upper:.6f}]"
    )

    print(
        "NOTE: descriptive DEV "
        "post-selection CI only."
    )

    # ========================================================
    # 4. FOLD STABILITY
    # ========================================================

    print(
        "\n4. FIVE-FOLD STABILITY"
    )

    fold_rows = []

    for fold in [
        1,
        2,
        3,
        4,
        5,
    ]:

        fold_raw = (
            selected_raw[
                selected_raw[
                    "outer_fold"
                ].eq(fold)
            ]
        )

        fold_aggregated = (
            TUNING
            .aggregate_scores(
                fold_raw,
                alpha,
                aggregation,
            )
        )

        fold_evaluation = (
            build_evaluation(
                fold_aggregated,
                positive_gt,
                threshold,
            )
        )

        fold_metrics, _ = (
            calculate_metrics(
                fold_evaluation
            )
        )

        fold_rows.append(
            {
                "outer_fold":
                    fold,

                **fold_metrics,
            }
        )

        print(
            f"Fold {fold}: "
            f"macro F1 = "
            f"{fold_metrics['macro_video_f1']:.6f}"
        )

    fold_df = pd.DataFrame(
        fold_rows
    )

    fold_mean = float(
        fold_df[
            "macro_video_f1"
        ].mean()
    )

    fold_std = float(
        fold_df[
            "macro_video_f1"
        ].std(ddof=1)
    )

    fold_min = float(
        fold_df[
            "macro_video_f1"
        ].min()
    )

    fold_max = float(
        fold_df[
            "macro_video_f1"
        ].max()
    )

    print(
        f"Fold mean : "
        f"{fold_mean:.6f}"
    )

    print(
        f"Fold std  : "
        f"{fold_std:.6f}"
    )

    print(
        f"Fold range: "
        f"{fold_min:.6f} "
        f"to {fold_max:.6f}"
    )

    # ========================================================
    # 5. SUBJECT ROBUSTNESS
    # ========================================================

    print(
        "\n5. SUBJECT PERFORMANCE"
    )

    subject_rows = []

    for subject, group in (
        evaluation.groupby(
            "subject"
        )
    ):

        subject_metrics, (
            subject_per_video
        ) = calculate_metrics(
            group
        )

        subject_rows.append(
            {
                "subject":
                    subject,

                "num_videos":
                    int(
                        subject_per_video[
                            "video_id"
                        ].nunique()
                    ),

                **subject_metrics,
            }
        )

        print(
            f"{subject}: "
            f"macro F1 = "
            f"{subject_metrics['macro_video_f1']:.6f}"
        )

    subject_df = pd.DataFrame(
        subject_rows
    )

    # ========================================================
    # 6. LOCAL ONE-FACTOR SENSITIVITY
    # ========================================================

    print(
        "\n6. LOCAL ONE-FACTOR "
        "SENSITIVITY"
    )

    local_rows = []

    selected_f1 = float(
        selected_search[
            "macro_video_f1"
        ]
    )

    # --------------------------------------------------------
    # Chunk alternatives
    # --------------------------------------------------------

    for candidate_chunk in sorted(
        search[
            "chunk_seconds"
        ].unique()
    ):

        if int(
            candidate_chunk
        ) == chunk_seconds:
            continue

        match = search[
            search[
                "chunk_seconds"
            ].eq(
                candidate_chunk
            )
            &
            np.isclose(
                search["alpha"],
                alpha,
            )
            &
            search[
                "aggregation"
            ].eq(
                aggregation
            )
            &
            np.isclose(
                search[
                    "threshold"
                ],
                threshold,
            )
        ]

        if len(match) == 1:

            row = match.iloc[0]

            local_rows.append(
                {
                    "parameter_changed":
                        "chunk_seconds",

                    "from_value":
                        str(
                            chunk_seconds
                        ),

                    "to_value":
                        str(
                            int(
                                candidate_chunk
                            )
                        ),

                    "macro_video_f1":
                        float(
                            row[
                                "macro_video_f1"
                            ]
                        ),

                    "delta_f1_vs_selected":
                        float(
                            row[
                                "macro_video_f1"
                            ]
                        )
                        - selected_f1,
                }
            )

    # --------------------------------------------------------
    # Alpha one-step neighbors
    # --------------------------------------------------------

    alpha_values = sorted(
        float(x)
        for x in (
            search[
                "alpha"
            ].unique()
        )
    )

    if alpha in alpha_values:

        alpha_index = (
            alpha_values.index(
                alpha
            )
        )

        neighbor_indices = [
            alpha_index - 1,
            alpha_index + 1,
        ]

        for idx in neighbor_indices:

            if (
                idx < 0
                or idx >= len(
                    alpha_values
                )
            ):
                continue

            candidate_alpha = (
                alpha_values[idx]
            )

            match = search[
                search[
                    "chunk_seconds"
                ].eq(
                    chunk_seconds
                )
                &
                np.isclose(
                    search[
                        "alpha"
                    ],
                    candidate_alpha,
                )
                &
                search[
                    "aggregation"
                ].eq(
                    aggregation
                )
                &
                np.isclose(
                    search[
                        "threshold"
                    ],
                    threshold,
                )
            ]

            if len(match) == 1:

                row = match.iloc[0]

                local_rows.append(
                    {
                        "parameter_changed":
                            "alpha",

                        "from_value":
                            str(alpha),

                        "to_value":
                            str(
                                candidate_alpha
                            ),

                        "macro_video_f1":
                            float(
                                row[
                                    "macro_video_f1"
                                ]
                            ),

                        "delta_f1_vs_selected":
                            float(
                                row[
                                    "macro_video_f1"
                                ]
                            )
                            - selected_f1,
                    }
                )

    # --------------------------------------------------------
    # Threshold one-step neighbors
    # --------------------------------------------------------

    threshold_values = sorted(
        float(x)
        for x in (
            search[
                "threshold"
            ].unique()
        )
    )

    if threshold in threshold_values:

        threshold_index = (
            threshold_values.index(
                threshold
            )
        )

        neighbor_indices = [
            threshold_index - 1,
            threshold_index + 1,
        ]

        for idx in neighbor_indices:

            if (
                idx < 0
                or idx >= len(
                    threshold_values
                )
            ):
                continue

            candidate_threshold = (
                threshold_values[idx]
            )

            match = search[
                search[
                    "chunk_seconds"
                ].eq(
                    chunk_seconds
                )
                &
                np.isclose(
                    search[
                        "alpha"
                    ],
                    alpha,
                )
                &
                search[
                    "aggregation"
                ].eq(
                    aggregation
                )
                &
                np.isclose(
                    search[
                        "threshold"
                    ],
                    candidate_threshold,
                )
            ]

            if len(match) == 1:

                row = match.iloc[0]

                local_rows.append(
                    {
                        "parameter_changed":
                            "threshold",

                        "from_value":
                            str(
                                threshold
                            ),

                        "to_value":
                            str(
                                candidate_threshold
                            ),

                        "macro_video_f1":
                            float(
                                row[
                                    "macro_video_f1"
                                ]
                            ),

                        "delta_f1_vs_selected":
                            float(
                                row[
                                    "macro_video_f1"
                                ]
                            )
                            - selected_f1,
                    }
                )

    # --------------------------------------------------------
    # Aggregation alternatives
    # --------------------------------------------------------

    for candidate_aggregation in (
        search[
            "aggregation"
        ].unique()
    ):

        if (
            candidate_aggregation
            == aggregation
        ):
            continue

        match = search[
            search[
                "chunk_seconds"
            ].eq(
                chunk_seconds
            )
            &
            np.isclose(
                search[
                    "alpha"
                ],
                alpha,
            )
            &
            search[
                "aggregation"
            ].eq(
                candidate_aggregation
            )
            &
            np.isclose(
                search[
                    "threshold"
                ],
                threshold,
            )
        ]

        if len(match) == 1:

            row = match.iloc[0]

            local_rows.append(
                {
                    "parameter_changed":
                        "aggregation",

                    "from_value":
                        aggregation,

                    "to_value":
                        str(
                            candidate_aggregation
                        ),

                    "macro_video_f1":
                        float(
                            row[
                                "macro_video_f1"
                            ]
                        ),

                    "delta_f1_vs_selected":
                        float(
                            row[
                                "macro_video_f1"
                            ]
                        )
                        - selected_f1,
                }
            )

    local_df = pd.DataFrame(
        local_rows
    )

    if not local_df.empty:

        local_df = (
            local_df
            .sort_values(
                [
                    "parameter_changed",
                    "macro_video_f1",
                ],
                ascending=[
                    True,
                    False,
                ],
            )
            .reset_index(
                drop=True
            )
        )

        print(
            local_df[
                [
                    "parameter_changed",
                    "from_value",
                    "to_value",
                    "macro_video_f1",
                    "delta_f1_vs_selected",
                ]
            ].to_string(
                index=False
            )
        )

    # ========================================================
    # 7. SEARCH-PLATEAU + BOUNDARY AUDIT
    # ========================================================

    print(
        "\n7. SEARCH-PLATEAU / "
        "BOUNDARY AUDIT"
    )

    within_005 = int(
        (
            search[
                "macro_video_f1"
            ]
            >= selected_f1 - 0.005
        ).sum()
    )

    within_010 = int(
        (
            search[
                "macro_video_f1"
            ]
            >= selected_f1 - 0.010
        ).sum()
    )

    within_020 = int(
        (
            search[
                "macro_video_f1"
            ]
            >= selected_f1 - 0.020
        ).sum()
    )

    chunk_values = sorted(
        int(x)
        for x in (
            search[
                "chunk_seconds"
            ].unique()
        )
    )

    boundary_flags = {
        "chunk_at_lower_boundary":
            bool(
                chunk_seconds
                == min(chunk_values)
            ),

        "chunk_at_upper_boundary":
            bool(
                chunk_seconds
                == max(chunk_values)
            ),

        "alpha_at_lower_boundary":
            bool(
                np.isclose(
                    alpha,
                    min(alpha_values),
                )
            ),

        "alpha_at_upper_boundary":
            bool(
                np.isclose(
                    alpha,
                    max(alpha_values),
                )
            ),

        "threshold_at_lower_boundary":
            bool(
                np.isclose(
                    threshold,
                    min(threshold_values),
                )
            ),

        "threshold_at_upper_boundary":
            bool(
                np.isclose(
                    threshold,
                    max(threshold_values),
                )
            ),
    }

    print(
        "Configs within 0.005 F1: "
        f"{within_005}"
    )

    print(
        "Configs within 0.010 F1: "
        f"{within_010}"
    )

    print(
        "Configs within 0.020 F1: "
        f"{within_020}"
    )

    for key, value in (
        boundary_flags.items()
    ):

        print(
            f"{key}: {value}"
        )

    # ========================================================
    # SAVE OUTPUTS
    # ========================================================

    evaluation[
        "chunk_seconds"
    ] = chunk_seconds

    evaluation[
        "alpha"
    ] = alpha

    evaluation[
        "aggregation"
    ] = aggregation

    evaluation[
        "threshold"
    ] = threshold

    evaluation.to_csv(
        PREDICTIONS_PATH,
        index=False,
    )

    per_video.to_csv(
        PER_VIDEO_PATH,
        index=False,
    )

    subject_df.to_csv(
        SUBJECT_PATH,
        index=False,
    )

    fold_df.to_csv(
        FOLD_PATH,
        index=False,
    )

    local_df.to_csv(
        LOCAL_SENSITIVITY_PATH,
        index=False,
    )

    audit = {
        "stage":
            "P1.13 Final Config "
            "Robustness Audit",

        "status":
            "PASS",

        "selected_configuration": {
            "chunk_seconds":
                chunk_seconds,

            "alpha":
                alpha,

            "lsa_weight":
                1.0 - alpha,

            "aggregation":
                aggregation,

            "threshold":
                threshold,
        },

        "reproduced_p1_12":
            True,

        "overall_metrics":
            metrics,

        "macro_video_f1_ci_95": {
            "lower":
                ci_lower,

            "upper":
                ci_upper,

            "method":
                (
                    "subject-stratified "
                    "video bootstrap"
                ),

            "iterations":
                BOOTSTRAP_ITERATIONS,

            "random_state":
                RANDOM_STATE,

            "interpretation":
                (
                    "descriptive DEV "
                    "post-selection CI only"
                ),
        },

        "fold_stability": {
            "mean":
                fold_mean,

            "std":
                fold_std,

            "min":
                fold_min,

            "max":
                fold_max,

            "range":
                fold_max
                - fold_min,
        },

        "search_plateau": {
            "within_0.005_f1":
                within_005,

            "within_0.010_f1":
                within_010,

            "within_0.020_f1":
                within_020,
        },

        "boundary_flags":
            boundary_flags,

        "config_changed_after_selection":
            False,

        "new_hyperparameters_searched":
            False,

        "p_value_used_for_selection":
            False,

        "final_test_used":
            False,

        "note":
            (
                "P1.13 audits the "
                "P1.12-selected configuration. "
                "It does not retune or replace "
                "the selected configuration."
            ),
    }

    AUDIT_PATH.write_text(
        json.dumps(
            audit,
            indent=2,
            ensure_ascii=False,
            default=json_default,
        ),
        encoding="utf-8",
    )

    # ========================================================
    # FINAL CONSOLE SUMMARY
    # ========================================================

    print(
        "\n" + "=" * 82
    )

    print(
        "P1.13 FINAL CONFIG "
        "ROBUSTNESS SUMMARY"
    )

    print(
        "=" * 82
    )

    print(
        f"Config       : "
        f"{chunk_seconds}s | "
        f"alpha={alpha} | "
        f"{aggregation} | "
        f"threshold={threshold}"
    )

    print(
        f"Macro F1     : "
        f"{metrics['macro_video_f1']:.6f}"
    )

    print(
        f"95% CI       : "
        f"[{ci_lower:.6f}, "
        f"{ci_upper:.6f}]"
    )

    print(
        f"Fold F1 std  : "
        f"{fold_std:.6f}"
    )

    print(
        f"Fold range   : "
        f"{fold_min:.6f} - "
        f"{fold_max:.6f}"
    )

    print(
        f"Near-best <= .01: "
        f"{within_010} configs"
    )

    print()

    print(
        "Boundary flags:"
    )

    for key, value in (
        boundary_flags.items()
    ):
        if value:
            print(
                f"  {key} = TRUE"
            )

    print(
        "\nNo config was changed."
    )

    print(
        "No new hyperparameter "
        "search was performed."
    )

    print(
        "Final TEST was not used."
    )

    print(
        "\nP1.13 FINAL CONFIG "
        "ROBUSTNESS AUDIT: PASS"
    )


if __name__ == "__main__":
    main()