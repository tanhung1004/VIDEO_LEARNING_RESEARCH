from pathlib import Path

import pandas as pd


# =========================================================
# STEP 10 - PREDICT + EVALUATE
# SCAFFOLD40 OLD vs NEW COMPARISON
# =========================================================

from step00_compare_config import (
    VARIANT,
    RUN_ROOT,
    CHUNK_SECONDS,
    LDA_WEIGHT,
    LSA_WEIGHT,
    AGGREGATION_COLUMN,
    PREDICTION_THRESHOLD,
)


# =========================================================
# 1. PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]


EVIDENCE_FILE = (
    RUN_ROOT
    / "video_concept_evidence.csv"
)


GT_FILE = (
    PROJECT_ROOT
    / "data"
    / "ground_truth"
    / "ground_truth_concepts.csv"
)


PREDICTIONS_FILE = (
    RUN_ROOT
    / "predictions.csv"
)


EVALUATION_FILE = (
    RUN_ROOT
    / "evaluation_pairs.csv"
)


METRICS_FILE = (
    RUN_ROOT
    / "metrics_summary.csv"
)


PER_VIDEO_FILE = (
    RUN_ROOT
    / "per_video_metrics.csv"
)


PER_SUBJECT_FILE = (
    RUN_ROOT
    / "per_subject_metrics.csv"
)


# =========================================================
# 2. HELPERS
# =========================================================

def normalize_video_id(value):

    text = str(value).strip().lower()

    if text.startswith("v"):

        number = text[1:]

        if number.isdigit():

            return f"v{int(number)}"

    return text


def concept_key(value):

    return (
        str(value)
        .strip()
        .lower()
    )


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


def safe_divide(
    numerator,
    denominator,
):

    if denominator == 0:

        return 0.0

    return (
        numerator
        / denominator
    )


def calculate_binary_metrics(
    y_true,
    y_pred,
):

    y_true = pd.Series(
        y_true
    ).astype(int)

    y_pred = pd.Series(
        y_pred
    ).astype(int)


    tp = int(
        (
            (y_true == 1)
            &
            (y_pred == 1)
        ).sum()
    )


    fp = int(
        (
            (y_true == 0)
            &
            (y_pred == 1)
        ).sum()
    )


    fn = int(
        (
            (y_true == 1)
            &
            (y_pred == 0)
        ).sum()
    )


    tn = int(
        (
            (y_true == 0)
            &
            (y_pred == 0)
        ).sum()
    )


    precision = safe_divide(
        tp,
        tp + fp,
    )


    recall = safe_divide(
        tp,
        tp + fn,
    )


    f1 = safe_divide(
        2
        * precision
        * recall,

        precision
        + recall,
    )


    accuracy = safe_divide(
        tp + tn,
        tp + fp + fn + tn,
    )


    return {
        "tp":
            tp,

        "fp":
            fp,

        "fn":
            fn,

        "tn":
            tn,

        "precision":
            precision,

        "recall":
            recall,

        "f1":
            f1,

        "accuracy":
            accuracy,
    }


# =========================================================
# 3. MAIN
# =========================================================

def main():

    print(
        "======================================"
    )

    print(
        "SCAFFOLD40 OLD vs NEW COMPARISON"
    )

    print(
        "STEP 10 - PREDICT + EVALUATE"
    )

    print(
        "======================================"
    )


    # =====================================================
    # 4. CHECK FILES
    # =====================================================

    if not EVIDENCE_FILE.exists():

        raise FileNotFoundError(
            "STEP 10 cannot find evidence file: "
            f"{EVIDENCE_FILE}"
        )


    if not GT_FILE.exists():

        raise FileNotFoundError(
            "STEP 10 cannot find GT file: "
            f"{GT_FILE}"
        )


    # =====================================================
    # 5. READ
    # =====================================================

    evidence = pd.read_csv(
        EVIDENCE_FILE
    )


    gt = pd.read_csv(
        GT_FILE
    )


    # =====================================================
    # 6. CONFIG AUDIT
    # =====================================================

    print(
        "\n======================================"
    )

    print(
        "STEP 10 CONFIG"
    )

    print(
        "======================================"
    )

    print(
        "Variant:",
        VARIANT
    )

    print(
        "Chunk seconds:",
        CHUNK_SECONDS
    )

    print(
        "LDA weight:",
        LDA_WEIGHT
    )

    print(
        "LSA weight:",
        LSA_WEIGHT
    )

    print(
        "Aggregation:",
        AGGREGATION_COLUMN
    )

    print(
        "Threshold:",
        PREDICTION_THRESHOLD
    )

    print(
        "======================================"
    )


    # =====================================================
    # 7. EVIDENCE AUDIT
    # =====================================================

    required_evidence_columns = {
        "video_id",
        "subject",
        "concept",
        "num_chunks",
        "max_final_score",
        "mean_final_score",
        "top2_mean_score",
    }


    missing_evidence_columns = (
        required_evidence_columns
        - set(evidence.columns)
    )


    if missing_evidence_columns:

        raise RuntimeError(
            "Evidence missing columns: "
            f"{sorted(missing_evidence_columns)}"
        )


    if (
        AGGREGATION_COLUMN
        not in evidence.columns
    ):

        raise RuntimeError(
            "Configured aggregation column "
            "does not exist: "
            f"{AGGREGATION_COLUMN}"
        )


    # -----------------------------------------------------
    # EXACTLY 390 VIDEO-CONCEPT PAIRS
    # -----------------------------------------------------

    if len(evidence) != 390:

        raise RuntimeError(
            "Expected 390 evidence rows. "
            f"Found={len(evidence)}"
        )


    evidence[
        "video_id"
    ] = (
        evidence[
            "video_id"
        ]
        .map(
            normalize_video_id
        )
    )


    evidence[
        "_concept_key"
    ] = (
        evidence[
            "concept"
        ]
        .map(
            concept_key
        )
    )


    expected_ids = {
        f"v{i}"
        for i in range(1, 41)
    }


    evidence_ids = set(
        evidence[
            "video_id"
        ]
    )


    if evidence_ids != expected_ids:

        raise RuntimeError(
            "Evidence is not exactly v1-v40. "
            f"Missing={sorted(expected_ids - evidence_ids)}, "
            f"Unexpected={sorted(evidence_ids - expected_ids)}"
        )


    duplicate_evidence = (
        evidence
        .duplicated(
            [
                "video_id",
                "_concept_key",
            ]
        )
        .sum()
    )


    if duplicate_evidence != 0:

        raise RuntimeError(
            "Duplicate evaluation pairs found: "
            f"{duplicate_evidence}"
        )


    # -----------------------------------------------------
    # 10 VIDEOS / SUBJECT
    # -----------------------------------------------------

    subject_counts = (
        evidence[
            [
                "video_id",
                "subject",
            ]
        ]
        .drop_duplicates()[
            "subject"
        ]
        .value_counts()
        .to_dict()
    )


    expected_subject_counts = {
        "SQL": 10,
        "Python": 10,
        "Java": 10,
        "C++": 10,
    }


    if (
        subject_counts
        != expected_subject_counts
    ):

        raise RuntimeError(
            "Unexpected video subject counts: "
            f"{subject_counts}"
        )


    # =====================================================
    # 8. GT AUDIT
    # =====================================================

    required_gt_columns = {
        "video_id",
        "concept",
    }


    missing_gt_columns = (
        required_gt_columns
        - set(gt.columns)
    )


    if missing_gt_columns:

        raise RuntimeError(
            "Ground truth missing columns: "
            f"{sorted(missing_gt_columns)}"
        )


    gt[
        "video_id"
    ] = (
        gt[
            "video_id"
        ]
        .map(
            normalize_video_id
        )
    )


    gt[
        "_concept_key"
    ] = (
        gt[
            "concept"
        ]
        .map(
            concept_key
        )
    )


    # -----------------------------------------------------
    # ONLY DEV v1-v40
    # -----------------------------------------------------

    gt_dev = (
        gt[
            gt[
                "video_id"
            ]
            .isin(
                expected_ids
            )
        ]
        .copy()
    )


    duplicate_gt = (
        gt_dev
        .duplicated(
            [
                "video_id",
                "_concept_key",
            ]
        )
        .sum()
    )


    if duplicate_gt != 0:

        raise RuntimeError(
            "Duplicate DEV GT positive pairs: "
            f"{duplicate_gt}"
        )


    positive_gt = set(
        zip(
            gt_dev[
                "video_id"
            ],

            gt_dev[
                "_concept_key"
            ],
        )
    )


    # P0 locked GT for v1-v40
    if len(
        positive_gt
    ) != 153:

        raise RuntimeError(
            "Expected 153 positive DEV GT pairs. "
            f"Found={len(positive_gt)}"
        )


    # -----------------------------------------------------
    # EVERY GT POSITIVE MUST EXIST IN EVALUATION SPACE
    # -----------------------------------------------------

    evidence_pairs = set(
        zip(
            evidence[
                "video_id"
            ],

            evidence[
                "_concept_key"
            ],
        )
    )


    gt_not_in_evidence = (
        positive_gt
        - evidence_pairs
    )


    if gt_not_in_evidence:

        raise RuntimeError(
            "Some GT positives are absent from "
            "390-pair evaluation space: "
            f"{sorted(gt_not_in_evidence)}"
        )


    print(
        "\n======================================"
    )

    print(
        "STEP 10 INPUT AUDIT"
    )

    print(
        "======================================"
    )

    print(
        "DEV videos:",
        evidence[
            "video_id"
        ].nunique()
    )

    print(
        "Evaluation pairs:",
        len(evidence)
    )

    print(
        "Positive GT pairs:",
        len(positive_gt)
    )

    print(
        "Duplicate evidence pairs:",
        duplicate_evidence
    )

    print(
        "Duplicate GT pairs:",
        duplicate_gt
    )

    print(
        "Subject counts:",
        subject_counts
    )

    print(
        "======================================"
    )


    # =====================================================
    # 9. BUILD Y_TRUE
    # =====================================================

    evaluation = (
        evidence
        .copy()
    )


    evaluation[
        "y_true"
    ] = [

        1
        if (
            video_id,
            ckey,
        )
        in positive_gt

        else 0

        for (
            video_id,
            ckey,
        )
        in zip(
            evaluation[
                "video_id"
            ],

            evaluation[
                "_concept_key"
            ],
        )
    ]


    # Audit y_true positives
    if int(
        evaluation[
            "y_true"
        ].sum()
    ) != 153:

        raise RuntimeError(
            "y_true positive count mismatch."
        )


    # =====================================================
    # 10. PREDICT
    # =====================================================

    evaluation[
        "decision_score"
    ] = (
        pd.to_numeric(
            evaluation[
                AGGREGATION_COLUMN
            ],
            errors="raise",
        )
    )


    evaluation[
        "y_pred"
    ] = (
        evaluation[
            "decision_score"
        ]
        >= PREDICTION_THRESHOLD
    ).astype(int)


    evaluation[
        "variant"
    ] = VARIANT


    evaluation[
        "chunk_seconds"
    ] = CHUNK_SECONDS


    evaluation[
        "aggregation_column"
    ] = AGGREGATION_COLUMN


    evaluation[
        "threshold"
    ] = PREDICTION_THRESHOLD


    # =====================================================
    # 11. MICRO / OVERALL METRICS
    # =====================================================

    micro = (
        calculate_binary_metrics(
            evaluation[
                "y_true"
            ],

            evaluation[
                "y_pred"
            ],
        )
    )


    # =====================================================
    # 12. PER-VIDEO METRICS
    # =====================================================

    per_video_rows = []


    for (
        video_id,
        group,
    ) in evaluation.groupby(
        "video_id",
        sort=False,
    ):

        metrics = (
            calculate_binary_metrics(
                group[
                    "y_true"
                ],

                group[
                    "y_pred"
                ],
            )
        )


        subjects = (
            group[
                "subject"
            ]
            .dropna()
            .astype(str)
            .unique()
        )


        if len(subjects) != 1:

            raise RuntimeError(
                f"Video {video_id} has "
                "multiple subjects."
            )


        per_video_rows.append({

            "variant":
                VARIANT,

            "video_id":
                video_id,

            "subject":
                subjects[0],

            "candidate_concepts":
                len(group),

            "gt_positive":
                int(
                    group[
                        "y_true"
                    ].sum()
                ),

            "predicted_positive":
                int(
                    group[
                        "y_pred"
                    ].sum()
                ),

            **metrics,
        })


    per_video = pd.DataFrame(
        per_video_rows
    )


    if len(
        per_video
    ) != 40:

        raise RuntimeError(
            "Expected exactly 40 per-video rows."
        )


    # =====================================================
    # 13. MACRO VIDEO METRICS
    # =====================================================

    macro_precision = float(
        per_video[
            "precision"
        ].mean()
    )


    macro_recall = float(
        per_video[
            "recall"
        ].mean()
    )


    macro_f1 = float(
        per_video[
            "f1"
        ].mean()
    )


    # =====================================================
    # 14. PER-SUBJECT METRICS
    # =====================================================

    per_subject_rows = []


    for (
        subject,
        group,
    ) in evaluation.groupby(
        "subject",
        sort=False,
    ):

        subject_metrics = (
            calculate_binary_metrics(
                group[
                    "y_true"
                ],

                group[
                    "y_pred"
                ],
            )
        )


        video_subset = (
            per_video[
                per_video[
                    "subject"
                ]
                == subject
            ]
        )


        per_subject_rows.append({

            "variant":
                VARIANT,

            "subject":
                subject,

            "videos":
                group[
                    "video_id"
                ].nunique(),

            "evaluation_pairs":
                len(group),

            "gt_positive":
                int(
                    group[
                        "y_true"
                    ].sum()
                ),

            "predicted_positive":
                int(
                    group[
                        "y_pred"
                    ].sum()
                ),

            "micro_precision":
                subject_metrics[
                    "precision"
                ],

            "micro_recall":
                subject_metrics[
                    "recall"
                ],

            "micro_f1":
                subject_metrics[
                    "f1"
                ],

            "macro_video_precision":
                float(
                    video_subset[
                        "precision"
                    ].mean()
                ),

            "macro_video_recall":
                float(
                    video_subset[
                        "recall"
                    ].mean()
                ),

            "macro_video_f1":
                float(
                    video_subset[
                        "f1"
                    ].mean()
                ),
        })


    per_subject = pd.DataFrame(
        per_subject_rows
    )


    if len(
        per_subject
    ) != 4:

        raise RuntimeError(
            "Expected exactly 4 subject rows."
        )


    # =====================================================
    # 15. BUILD SUMMARY
    # =====================================================

    summary = pd.DataFrame(
        [
            {
                "variant":
                    VARIANT,

                "dev_videos":
                    40,

                "evaluation_pairs":
                    len(
                        evaluation
                    ),

                "positive_gt_pairs":
                    int(
                        evaluation[
                            "y_true"
                        ].sum()
                    ),

                "predicted_positive_pairs":
                    int(
                        evaluation[
                            "y_pred"
                        ].sum()
                    ),

                "chunk_seconds":
                    CHUNK_SECONDS,

                "lda_weight":
                    LDA_WEIGHT,

                "lsa_weight":
                    LSA_WEIGHT,

                "aggregation":
                    AGGREGATION_COLUMN,

                "threshold":
                    PREDICTION_THRESHOLD,

                "tp":
                    micro[
                        "tp"
                    ],

                "fp":
                    micro[
                        "fp"
                    ],

                "fn":
                    micro[
                        "fn"
                    ],

                "tn":
                    micro[
                        "tn"
                    ],

                "micro_precision":
                    micro[
                        "precision"
                    ],

                "micro_recall":
                    micro[
                        "recall"
                    ],

                "micro_f1":
                    micro[
                        "f1"
                    ],

                "micro_accuracy":
                    micro[
                        "accuracy"
                    ],

                "macro_video_precision":
                    macro_precision,

                "macro_video_recall":
                    macro_recall,

                "macro_video_f1":
                    macro_f1,
            }
        ]
    )


    # =====================================================
    # 16. PREDICTION FILE
    # =====================================================

    predictions = (
        evaluation[
            evaluation[
                "y_pred"
            ]
            == 1
        ]
        .copy()
    )


    # =====================================================
    # 17. DETERMINISTIC SORT
    # =====================================================

    evaluation[
        "_video_number"
    ] = (
        evaluation[
            "video_id"
        ]
        .map(
            video_number
        )
    )


    evaluation = (
        evaluation
        .sort_values(
            [
                "_video_number",
                "concept",
            ]
        )
        .drop(
            columns=[
                "_video_number",
                "_concept_key",
            ],
            errors="ignore",
        )
        .reset_index(
            drop=True
        )
    )


    predictions[
        "_video_number"
    ] = (
        predictions[
            "video_id"
        ]
        .map(
            video_number
        )
    )


    predictions = (
        predictions
        .sort_values(
            [
                "_video_number",
                "decision_score",
                "concept",
            ],
            ascending=[
                True,
                False,
                True,
            ],
        )
        .drop(
            columns=[
                "_video_number",
                "_concept_key",
            ],
            errors="ignore",
        )
        .reset_index(
            drop=True
        )
    )


    per_video[
        "_video_number"
    ] = (
        per_video[
            "video_id"
        ]
        .map(
            video_number
        )
    )


    per_video = (
        per_video
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


    subject_order = {
        "SQL": 1,
        "Python": 2,
        "Java": 3,
        "C++": 4,
    }


    per_subject[
        "_subject_order"
    ] = (
        per_subject[
            "subject"
        ]
        .map(
            subject_order
        )
    )


    per_subject = (
        per_subject
        .sort_values(
            "_subject_order"
        )
        .drop(
            columns=[
                "_subject_order"
            ]
        )
        .reset_index(
            drop=True
        )
    )


    # =====================================================
    # 18. SAVE OUTPUTS
    # =====================================================

    predictions.to_csv(
        PREDICTIONS_FILE,
        index=False,
        encoding="utf-8-sig",
    )


    evaluation.to_csv(
        EVALUATION_FILE,
        index=False,
        encoding="utf-8-sig",
    )


    summary.to_csv(
        METRICS_FILE,
        index=False,
        encoding="utf-8-sig",
    )


    per_video.to_csv(
        PER_VIDEO_FILE,
        index=False,
        encoding="utf-8-sig",
    )


    per_subject.to_csv(
        PER_SUBJECT_FILE,
        index=False,
        encoding="utf-8-sig",
    )


    # =====================================================
    # 19. FINAL AUDIT
    # =====================================================

    if len(
        evaluation
    ) != 390:

        raise RuntimeError(
            "Final evaluation row count != 390."
        )


    if len(
        per_video
    ) != 40:

        raise RuntimeError(
            "Final per-video count != 40."
        )


    if len(
        per_subject
    ) != 4:

        raise RuntimeError(
            "Final per-subject count != 4."
        )


    # =====================================================
    # 20. PRINT RESULTS
    # =====================================================

    print(
        "\n======================================"
    )

    print(
        "STEP 10 RESULTS"
    )

    print(
        "======================================"
    )

    print(
        "Variant:",
        VARIANT
    )

    print(
        "Aggregation:",
        AGGREGATION_COLUMN
    )

    print(
        "Threshold:",
        PREDICTION_THRESHOLD
    )


    print(
        "\nTP:",
        micro[
            "tp"
        ]
    )

    print(
        "FP:",
        micro[
            "fp"
        ]
    )

    print(
        "FN:",
        micro[
            "fn"
        ]
    )

    print(
        "TN:",
        micro[
            "tn"
        ]
    )


    print(
        "\nMicro precision:",
        f"{micro['precision']:.6f}"
    )

    print(
        "Micro recall:",
        f"{micro['recall']:.6f}"
    )

    print(
        "Micro F1:",
        f"{micro['f1']:.6f}"
    )


    print(
        "\nMacro video precision:",
        f"{macro_precision:.6f}"
    )

    print(
        "Macro video recall:",
        f"{macro_recall:.6f}"
    )

    print(
        "Macro video F1:",
        f"{macro_f1:.6f}"
    )


    print(
        "\nPredicted positive pairs:",
        len(
            predictions
        )
    )


    print(
        "\n======================================"
    )

    print(
        "STEP 10 PASS"
    )

    print(
        "======================================"
    )

    print(
        "DEV videos: 40"
    )

    print(
        "Evaluation pairs:",
        len(
            evaluation
        )
    )

    print(
        "Positive GT pairs:",
        int(
            evaluation[
                "y_true"
            ].sum()
        )
    )

    print(
        "Per-video rows:",
        len(
            per_video
        )
    )

    print(
        "Per-subject rows:",
        len(
            per_subject
        )
    )


    print(
        "\nMetrics:"
    )

    print(
        METRICS_FILE
    )


    print(
        "\nPredictions:"
    )

    print(
        PREDICTIONS_FILE
    )


    print(
        "\nEvaluation pairs:"
    )

    print(
        EVALUATION_FILE
    )


    print(
        "\nPer-video metrics:"
    )

    print(
        PER_VIDEO_FILE
    )


    print(
        "\nPer-subject metrics:"
    )

    print(
        PER_SUBJECT_FILE
    )


    print(
        "======================================"
    )


if __name__ == "__main__":

    main()