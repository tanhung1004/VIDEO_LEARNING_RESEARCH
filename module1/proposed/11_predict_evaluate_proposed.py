from pathlib import Path
import pandas as pd


# =========================================================
# 1. PROJECT PATH
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# =========================================================
# 2. INPUT
# =========================================================

RESULT_DIR = (
    PROJECT_ROOT
    / "module1"
    / "results"
    / "proposed"
)

EVIDENCE_FILE = (
    RESULT_DIR
    / "video_concept_evidence.csv"
)

GROUND_TRUTH_FILE = (
    PROJECT_ROOT
    / "data"
    / "ground_truth"
    / "ground_truth_concepts.csv"
)


# =========================================================
# 3. OUTPUT
# =========================================================

PREDICTED_FILE = (
    RESULT_DIR
    / "predicted_concepts.csv"
)

DETAIL_FILE = (
    RESULT_DIR
    / "evaluation_details.csv"
)

METRICS_FILE = (
    RESULT_DIR
    / "evaluation_metrics.csv"
)


# =========================================================
# 4. SAME THRESHOLD AS BASELINE
# =========================================================
#
# Không tune lại threshold cho OCR.
#
# Baseline:
# top2_mean_score >= 0.40
#
# Proposed:
# top2_mean_score >= 0.40
#
# =========================================================

THRESHOLD = 0.40

METHOD = "transcript_ocr"


# =========================================================
# 5. NORMALIZE TEXT
# =========================================================

def normalize_text(value):

    if pd.isna(value):
        return ""

    return (
        str(value)
        .strip()
        .lower()
    )


# =========================================================
# 6. SAFE DIVISION
# =========================================================

def safe_divide(
    numerator,
    denominator
):

    if denominator == 0:
        return 0.0

    return (
        numerator
        / denominator
    )


# =========================================================
# 7. CALCULATE METRICS
# =========================================================

def calculate_metrics(
    details,
    video_id
):

    tp = int(
        (
            details["result"]
            == "TP"
        ).sum()
    )

    fp = int(
        (
            details["result"]
            == "FP"
        ).sum()
    )

    fn = int(
        (
            details["result"]
            == "FN"
        ).sum()
    )

    tn = int(
        (
            details["result"]
            == "TN"
        ).sum()
    )


    precision = safe_divide(
        tp,
        tp + fp
    )

    recall = safe_divide(
        tp,
        tp + fn
    )

    f1 = safe_divide(
        2 * precision * recall,
        precision + recall
    )


    ground_truth_count = int(
        details[
            "ground_truth"
        ].sum()
    )

    predicted_count = int(
        details[
            "predicted"
        ].sum()
    )


    return {

        "video_id":
            video_id,

        "method":
            METHOD,

        "threshold":
            THRESHOLD,

        "ground_truth_count":
            ground_truth_count,

        "predicted_count":
            predicted_count,

        "TP":
            tp,

        "FP":
            fp,

        "FN":
            fn,

        "TN":
            tn,

        "precision":
            round(
                precision,
                4
            ),

        "recall":
            round(
                recall,
                4
            ),

        "f1":
            round(
                f1,
                4
            )
    }


# =========================================================
# 8. MAIN
# =========================================================

def main():

    print(
        "======================================"
    )

    print(
        "MODULE 1 - PROPOSED"
    )

    print(
        "STEP 11 - PREDICTION + EVALUATION"
    )

    print(
        "METHOD: TRANSCRIPT + OCR"
    )

    print(
        "THRESHOLD:",
        THRESHOLD
    )

    print(
        "======================================"
    )


    # =====================================================
    # 9. CHECK FILES
    # =====================================================

    if not EVIDENCE_FILE.exists():

        raise FileNotFoundError(
            f"Không tìm thấy evidence file:\n"
            f"{EVIDENCE_FILE}"
        )


    if not GROUND_TRUTH_FILE.exists():

        raise FileNotFoundError(
            f"Không tìm thấy ground truth:\n"
            f"{GROUND_TRUTH_FILE}"
        )


    # =====================================================
    # 10. READ DATA
    # =====================================================

    evidence = pd.read_csv(
        EVIDENCE_FILE
    )

    ground_truth = pd.read_csv(
        GROUND_TRUTH_FILE
    )


    print(
        "Evidence rows:",
        len(evidence)
    )

    print(
        "Ground truth rows:",
        len(ground_truth)
    )


    # =====================================================
    # 11. CHECK REQUIRED COLUMNS
    # =====================================================

    required_evidence_columns = [

        "video_id",
        "subject",
        "concept",
        "top2_mean_score"
    ]


    for column in required_evidence_columns:

        if column not in evidence.columns:

            raise ValueError(
                f"Evidence thiếu cột: {column}"
            )


    required_gt_columns = [

        "video_id",
        "concept"
    ]


    for column in required_gt_columns:

        if column not in ground_truth.columns:

            raise ValueError(
                f"Ground truth thiếu cột: {column}"
            )


    # =====================================================
    # 12. NORMALIZE KEYS
    # =====================================================

    evidence["video_id_norm"] = (
        evidence["video_id"]
        .apply(
            normalize_text
        )
    )

    evidence["concept_norm"] = (
        evidence["concept"]
        .apply(
            normalize_text
        )
    )


    ground_truth["video_id_norm"] = (
        ground_truth["video_id"]
        .apply(
            normalize_text
        )
    )

    ground_truth["concept_norm"] = (
        ground_truth["concept"]
        .apply(
            normalize_text
        )
    )


    # =====================================================
    # 13. NUMERIC SCORE
    # =====================================================

    evidence["top2_mean_score"] = (
        pd.to_numeric(
            evidence[
                "top2_mean_score"
            ],
            errors="coerce"
        )
        .fillna(0.0)
    )


    # =====================================================
    # 14. PREDICTION
    #
    # SAME RULE AS BASELINE:
    #
    # top2_mean_score >= 0.40
    #        ↓
    # predicted concept
    #
    # =====================================================

    evidence["predicted"] = (

        evidence[
            "top2_mean_score"
        ]

        >= THRESHOLD
    ).astype(int)


    predicted = (

        evidence[
            evidence["predicted"]
            == 1
        ]

        .copy()
    )


    # =====================================================
    # 15. SAVE PREDICTED CONCEPTS
    # =====================================================

    predicted_output = (

        predicted

        .sort_values(
            [
                "video_id",
                "top2_mean_score"
            ],
            ascending=[
                True,
                False
            ]
        )

        .reset_index(
            drop=True
        )
    )


    predicted_output[
        "threshold"
    ] = THRESHOLD


    predicted_output.to_csv(

        PREDICTED_FILE,

        index=False,

        encoding="utf-8-sig"
    )


    # =====================================================
    # 16. BUILD CANDIDATE UNIVERSE
    #
    # Dùng UNION:
    #
    # evidence concepts
    #       +
    # ground truth concepts
    #
    # Như vậy nếu concept có trong GT nhưng vì lý do nào đó
    # không xuất hiện trong evidence thì vẫn được tính FN.
    #
    # =====================================================

    evidence_pairs = (

        evidence[
            [
                "video_id_norm",
                "concept_norm"
            ]
        ]

        .drop_duplicates()
    )


    gt_pairs = (

        ground_truth[
            [
                "video_id_norm",
                "concept_norm"
            ]
        ]

        .drop_duplicates()
    )


    candidate_pairs = (

        pd.concat(
            [
                evidence_pairs,
                gt_pairs
            ],
            ignore_index=True
        )

        .drop_duplicates()

        .reset_index(
            drop=True
        )
    )


    # =====================================================
    # 17. LOOKUP GROUND TRUTH
    # =====================================================

    gt_set = set(

        zip(
            ground_truth[
                "video_id_norm"
            ],

            ground_truth[
                "concept_norm"
            ]
        )
    )


    # =====================================================
    # 18. LOOKUP PREDICTION
    # =====================================================

    prediction_lookup = {}

    for _, row in evidence.iterrows():

        key = (

            row[
                "video_id_norm"
            ],

            row[
                "concept_norm"
            ]
        )

        prediction_lookup[
            key
        ] = {

            "video_id":
                row["video_id"],

            "subject":
                row["subject"],

            "concept":
                row["concept"],

            "top2_mean_score":
                float(
                    row[
                        "top2_mean_score"
                    ]
                ),

            "max_final_score":
                row.get(
                    "max_final_score",
                    0.0
                ),

            "mean_final_score":
                row.get(
                    "mean_final_score",
                    0.0
                ),

            "best_chunk_id":
                row.get(
                    "best_chunk_id",
                    ""
                ),

            "best_start_sec":
                row.get(
                    "best_start_sec",
                    ""
                ),

            "best_end_sec":
                row.get(
                    "best_end_sec",
                    ""
                ),

            "predicted":
                int(
                    row["predicted"]
                )
        }


    # =====================================================
    # 19. GT DISPLAY LOOKUP
    # =====================================================

    gt_lookup = {}

    for _, row in ground_truth.iterrows():

        key = (

            row[
                "video_id_norm"
            ],

            row[
                "concept_norm"
            ]
        )

        gt_lookup[key] = {

            "video_id":
                row["video_id"],

            "subject":
                row.get(
                    "subject",
                    ""
                ),

            "concept":
                row["concept"]
        }


    # =====================================================
    # 20. EVALUATION DETAILS
    # =====================================================

    detail_rows = []


    for _, candidate in (
        candidate_pairs.iterrows()
    ):

        key = (

            candidate[
                "video_id_norm"
            ],

            candidate[
                "concept_norm"
            ]
        )


        # ---------------------------------------------
        # DISPLAY INFO
        # ---------------------------------------------

        if key in prediction_lookup:

            info = prediction_lookup[
                key
            ]

        else:

            gt_info = gt_lookup[
                key
            ]

            info = {

                "video_id":
                    gt_info[
                        "video_id"
                    ],

                "subject":
                    gt_info[
                        "subject"
                    ],

                "concept":
                    gt_info[
                        "concept"
                    ],

                "top2_mean_score":
                    0.0,

                "max_final_score":
                    0.0,

                "mean_final_score":
                    0.0,

                "best_chunk_id":
                    "",

                "best_start_sec":
                    "",

                "best_end_sec":
                    "",

                "predicted":
                    0
            }


        # ---------------------------------------------
        # GT
        # ---------------------------------------------

        actual = int(
            key in gt_set
        )


        predicted_value = int(
            info[
                "predicted"
            ]
        )


        # ---------------------------------------------
        # TP / FP / FN / TN
        # ---------------------------------------------

        if (
            actual == 1
            and predicted_value == 1
        ):

            result = "TP"


        elif (
            actual == 0
            and predicted_value == 1
        ):

            result = "FP"


        elif (
            actual == 1
            and predicted_value == 0
        ):

            result = "FN"


        else:

            result = "TN"


        # ---------------------------------------------
        # SAVE
        # ---------------------------------------------

        detail_rows.append({

            "video_id":
                info["video_id"],

            "subject":
                info["subject"],

            "concept":
                info["concept"],

            "method":
                METHOD,

            "top2_mean_score":
                round(
                    float(
                        info[
                            "top2_mean_score"
                        ]
                    ),
                    6
                ),

            "threshold":
                THRESHOLD,

            "ground_truth":
                actual,

            "predicted":
                predicted_value,

            "result":
                result,

            "max_final_score":
                info[
                    "max_final_score"
                ],

            "mean_final_score":
                info[
                    "mean_final_score"
                ],

            "best_chunk_id":
                info[
                    "best_chunk_id"
                ],

            "best_start_sec":
                info[
                    "best_start_sec"
                ],

            "best_end_sec":
                info[
                    "best_end_sec"
                ]
        })


    details = pd.DataFrame(
        detail_rows
    )


    # =====================================================
    # 21. SORT DETAILS
    # =====================================================

    result_order = {

        "FP": 0,
        "FN": 1,
        "TP": 2,
        "TN": 3
    }


    details["result_order"] = (

        details[
            "result"
        ]

        .map(
            result_order
        )
    )


    details = (

        details

        .sort_values(
            [
                "video_id",
                "result_order",
                "top2_mean_score"
            ],
            ascending=[
                True,
                True,
                False
            ]
        )

        .drop(
            columns=[
                "result_order"
            ]
        )

        .reset_index(
            drop=True
        )
    )


    # =====================================================
    # 22. SAVE DETAILS
    # =====================================================

    details.to_csv(

        DETAIL_FILE,

        index=False,

        encoding="utf-8-sig"
    )


    # =====================================================
    # 23. METRICS PER VIDEO
    # =====================================================

    metrics_rows = []


    video_ids = sorted(
        details[
            "video_id"
        ]
        .astype(str)
        .unique()
    )


    for video_id in video_ids:

        video_details = (

            details[
                details[
                    "video_id"
                ].astype(str)
                == str(video_id)
            ]

            .copy()
        )


        metrics_rows.append(

            calculate_metrics(
                video_details,
                video_id
            )
        )


    # =====================================================
    # 24. OVERALL METRICS
    # =====================================================

    overall = calculate_metrics(

        details,

        "OVERALL"
    )


    metrics_rows.append(
        overall
    )


    metrics = pd.DataFrame(
        metrics_rows
    )


    # =====================================================
    # 25. SAVE METRICS
    # =====================================================

    metrics.to_csv(

        METRICS_FILE,

        index=False,

        encoding="utf-8-sig"
    )


    # =====================================================
    # 26. PRINT PREDICTIONS
    # =====================================================

    print(
        "\n========== PREDICTED CONCEPTS =========="
    )


    if predicted_output.empty:

        print(
            "Không có concept nào vượt threshold."
        )

    else:

        print(

            predicted_output[
                [
                    "video_id",
                    "concept",
                    "top2_mean_score"
                ]
            ]

            .to_string(
                index=False
            )
        )


    # =====================================================
    # 27. PRINT ERRORS
    # =====================================================

    errors = (

        details[
            details[
                "result"
            ].isin(
                [
                    "FP",
                    "FN"
                ]
            )
        ]
    )


    print(
        "\n========== FP / FN =========="
    )


    if errors.empty:

        print(
            "Không có FP/FN."
        )

    else:

        print(

            errors[
                [
                    "video_id",
                    "concept",
                    "top2_mean_score",
                    "ground_truth",
                    "predicted",
                    "result"
                ]
            ]

            .to_string(
                index=False
            )
        )


    # =====================================================
    # 28. PRINT METRICS
    # =====================================================

    print(
        "\n========== EVALUATION =========="
    )


    print(

        metrics[
            [
                "video_id",
                "ground_truth_count",
                "predicted_count",
                "TP",
                "FP",
                "FN",
                "precision",
                "recall",
                "f1"
            ]
        ]

        .to_string(
            index=False
        )
    )


    # =====================================================
    # 29. OVERALL
    # =====================================================

    print(
        "\n========== OVERALL =========="
    )

    print(
        "Ground Truth:",
        overall[
            "ground_truth_count"
        ]
    )

    print(
        "Predicted:",
        overall[
            "predicted_count"
        ]
    )

    print(
        "TP:",
        overall["TP"]
    )

    print(
        "FP:",
        overall["FP"]
    )

    print(
        "FN:",
        overall["FN"]
    )

    print(
        "Precision:",
        f'{overall["precision"]:.4f}'
    )

    print(
        "Recall:",
        f'{overall["recall"]:.4f}'
    )

    print(
        "F1:",
        f'{overall["f1"]:.4f}'
    )


    # =====================================================
    # 30. DONE
    # =====================================================

    print(
        "\n======================================"
    )

    print(
        "HOÀN THÀNH STEP 11"
    )

    print(
        "\nPredicted concepts:"
    )

    print(
        PREDICTED_FILE
    )

    print(
        "\nEvaluation details:"
    )

    print(
        DETAIL_FILE
    )

    print(
        "\nEvaluation metrics:"
    )

    print(
        METRICS_FILE
    )

    print(
        "======================================"
    )


if __name__ == "__main__":
    main()