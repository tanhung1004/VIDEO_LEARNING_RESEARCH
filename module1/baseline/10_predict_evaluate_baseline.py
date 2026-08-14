from pathlib import Path
import pandas as pd


# =========================================================
# 1. CONFIG
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

EVIDENCE_FILE = (
    PROJECT_ROOT
    / "module1"
    / "results"
    / "baseline"
    / "video_concept_evidence.csv"
)

GROUND_TRUTH_FILE = (
    PROJECT_ROOT
    / "data"
    / "ground_truth"
    / "ground_truth_concepts.csv"
)

RESULT_DIR = (
    PROJECT_ROOT
    / "module1"
    / "results"
    / "baseline"
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# PREDICTION THRESHOLD
# =========================================================

PREDICTION_THRESHOLD = 0.40


# =========================================================
# 2. NORMALIZE TEXT
# =========================================================

def normalize_text(value):

    return (
        str(value)
        .strip()
        .lower()
    )


# =========================================================
# 3. METRICS
# =========================================================

def calculate_metrics(tp, fp, fn):

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    return (
        precision,
        recall,
        f1
    )


# =========================================================
# 4. MAIN
# =========================================================

def main():

    print("======================================")
    print("MODULE 1 - TRANSCRIPT BASELINE")
    print("STEP 10 - PREDICT + EVALUATE")
    print("======================================")

    # -----------------------------------------------------
    # CHECK FILE
    # -----------------------------------------------------

    if not EVIDENCE_FILE.exists():

        print("Không tìm thấy:")
        print(EVIDENCE_FILE)

        return

    if not GROUND_TRUTH_FILE.exists():

        print("Không tìm thấy:")
        print(GROUND_TRUTH_FILE)

        return


    # -----------------------------------------------------
    # READ DATA
    # -----------------------------------------------------

    evidence = pd.read_csv(
        EVIDENCE_FILE
    )

    ground_truth = pd.read_csv(
        GROUND_TRUTH_FILE
    )


    # -----------------------------------------------------
    # NORMALIZE
    # -----------------------------------------------------

    evidence["video_id"] = (
        evidence["video_id"]
        .apply(normalize_text)
    )

    evidence["subject"] = (
        evidence["subject"]
        .apply(normalize_text)
    )

    evidence["concept_norm"] = (
        evidence["concept"]
        .apply(normalize_text)
    )


    ground_truth["video_id"] = (
        ground_truth["video_id"]
        .apply(normalize_text)
    )

    ground_truth["subject"] = (
        ground_truth["subject"]
        .apply(normalize_text)
    )

    ground_truth["concept_norm"] = (
        ground_truth["concept"]
        .apply(normalize_text)
    )


    # =====================================================
    # 5. PREDICTION
    # =====================================================

    predictions = evidence[
        evidence["top2_mean_score"]
        >= PREDICTION_THRESHOLD
    ].copy()


    predictions["prediction"] = True


    prediction_file = (
        RESULT_DIR
        / "predicted_concepts.csv"
    )


    predictions.to_csv(
        prediction_file,
        index=False,
        encoding="utf-8-sig"
    )


    print(
        "\nPrediction threshold:",
        PREDICTION_THRESHOLD
    )

    print(
        "Số predicted concepts:",
        len(predictions)
    )


    # =====================================================
    # 6. EVALUATE TỪNG VIDEO
    # =====================================================

    metrics_rows = []
    detail_rows = []

    total_tp = 0
    total_fp = 0
    total_fn = 0


    video_ids = sorted(
        ground_truth[
            "video_id"
        ].unique()
    )


    for video_id in video_ids:

        gt_video = ground_truth[
            ground_truth["video_id"]
            == video_id
        ]

        pred_video = predictions[
            predictions["video_id"]
            == video_id
        ]


        # ---------------------------------------------
        # Ground truth set
        # ---------------------------------------------

        gt_set = set(
            gt_video[
                "concept_norm"
            ].tolist()
        )


        # ---------------------------------------------
        # Prediction set
        # ---------------------------------------------

        pred_set = set(
            pred_video[
                "concept_norm"
            ].tolist()
        )


        # ---------------------------------------------
        # TP / FP / FN
        # ---------------------------------------------

        tp_set = (
            gt_set
            & pred_set
        )

        fp_set = (
            pred_set
            - gt_set
        )

        fn_set = (
            gt_set
            - pred_set
        )


        tp = len(tp_set)
        fp = len(fp_set)
        fn = len(fn_set)


        total_tp += tp
        total_fp += fp
        total_fn += fn


        precision, recall, f1 = (
            calculate_metrics(
                tp,
                fp,
                fn
            )
        )


        subject = (
            gt_video["subject"]
            .iloc[0]
        )


        metrics_rows.append({

            "video_id":
                video_id,

            "subject":
                subject,

            "method":
                "transcript_only",

            "threshold":
                PREDICTION_THRESHOLD,

            "ground_truth_count":
                len(gt_set),

            "predicted_count":
                len(pred_set),

            "TP":
                tp,

            "FP":
                fp,

            "FN":
                fn,

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
                ),

            "correct_concepts":
                "; ".join(
                    sorted(tp_set)
                ),

            "missing_concepts":
                "; ".join(
                    sorted(fn_set)
                ),

            "extra_concepts":
                "; ".join(
                    sorted(fp_set)
                )
        })


        # ---------------------------------------------
        # DETAIL FILE
        # ---------------------------------------------

        all_concepts = (
            gt_set
            | pred_set
        )


        for concept in sorted(
            all_concepts
        ):

            if concept in tp_set:

                result = "TP"

            elif concept in fp_set:

                result = "FP"

            else:

                result = "FN"


            detail_rows.append({

                "video_id":
                    video_id,

                "subject":
                    subject,

                "method":
                    "transcript_only",

                "concept":
                    concept,

                "result":
                    result
            })


    # =====================================================
    # 7. OVERALL METRICS
    # =====================================================

    (
        overall_precision,
        overall_recall,
        overall_f1
    ) = calculate_metrics(
        total_tp,
        total_fp,
        total_fn
    )


    metrics_rows.append({

        "video_id":
            "OVERALL",

        "subject":
            "ALL",

        "method":
            "transcript_only",

        "threshold":
            PREDICTION_THRESHOLD,

        "ground_truth_count":
            total_tp + total_fn,

        "predicted_count":
            total_tp + total_fp,

        "TP":
            total_tp,

        "FP":
            total_fp,

        "FN":
            total_fn,

        "precision":
            round(
                overall_precision,
                4
            ),

        "recall":
            round(
                overall_recall,
                4
            ),

        "f1":
            round(
                overall_f1,
                4
            ),

        "correct_concepts":
            "",

        "missing_concepts":
            "",

        "extra_concepts":
            ""
    })


    # =====================================================
    # 8. SAVE
    # =====================================================

    metrics_df = pd.DataFrame(
        metrics_rows
    )

    details_df = pd.DataFrame(
        detail_rows
    )


    metrics_file = (
        RESULT_DIR
        / "evaluation_metrics.csv"
    )

    details_file = (
        RESULT_DIR
        / "evaluation_details.csv"
    )


    metrics_df.to_csv(
        metrics_file,
        index=False,
        encoding="utf-8-sig"
    )

    details_df.to_csv(
        details_file,
        index=False,
        encoding="utf-8-sig"
    )


    # =====================================================
    # 9. PRINT RESULT
    # =====================================================

    print(
        "\n========== BASELINE RESULT =========="
    )


    print(
        metrics_df[
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


    print(
        "\n======================================"
    )

    print(
        "HOÀN THÀNH BASELINE TRANSCRIPT-ONLY"
    )


    print(
        "\nPredictions:"
    )

    print(
        prediction_file
    )


    print(
        "\nMetrics:"
    )

    print(
        metrics_file
    )


    print(
        "\nDetails:"
    )

    print(
        details_file
    )


    print(
        "======================================"
    )


if __name__ == "__main__":
    main()