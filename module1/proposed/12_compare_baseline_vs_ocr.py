from pathlib import Path
import pandas as pd


# =========================================================
# 1. PROJECT PATH
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# =========================================================
# 2. INPUT
# =========================================================

BASELINE_FILE = (
    PROJECT_ROOT
    / "module1"
    / "results"
    / "baseline"
    / "evaluation_metrics.csv"
)

PROPOSED_FILE = (
    PROJECT_ROOT
    / "module1"
    / "results"
    / "proposed"
    / "evaluation_metrics.csv"
)


# =========================================================
# 3. OUTPUT
# =========================================================

OUTPUT_DIR = (
    PROJECT_ROOT
    / "module1"
    / "results"
    / "comparison"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "baseline_vs_ocr.csv"
)


# =========================================================
# 4. MAIN
# =========================================================

def main():

    print("======================================")
    print("MODULE 1 - FINAL COMPARISON")
    print("TRANSCRIPT ONLY vs TRANSCRIPT + OCR")
    print("======================================")


    # -----------------------------------------------------
    # READ RESULTS
    # -----------------------------------------------------

    baseline = pd.read_csv(
        BASELINE_FILE
    )

    proposed = pd.read_csv(
        PROPOSED_FILE
    )


    # -----------------------------------------------------
    # GET OVERALL ROW
    # -----------------------------------------------------

    baseline_overall = baseline[
        baseline["video_id"].astype(str)
        == "OVERALL"
    ].iloc[0]

    proposed_overall = proposed[
        proposed["video_id"].astype(str)
        == "OVERALL"
    ].iloc[0]


    # -----------------------------------------------------
    # BUILD TABLE
    # -----------------------------------------------------

    comparison = pd.DataFrame([

        {
            "Method":
                "Transcript-only",

            "Ground Truth":
                int(
                    baseline_overall[
                        "ground_truth_count"
                    ]
                ),

            "Predicted":
                int(
                    baseline_overall[
                        "predicted_count"
                    ]
                ),

            "TP":
                int(
                    baseline_overall[
                        "TP"
                    ]
                ),

            "FP":
                int(
                    baseline_overall[
                        "FP"
                    ]
                ),

            "FN":
                int(
                    baseline_overall[
                        "FN"
                    ]
                ),

            "Precision":
                float(
                    baseline_overall[
                        "precision"
                    ]
                ),

            "Recall":
                float(
                    baseline_overall[
                        "recall"
                    ]
                ),

            "F1":
                float(
                    baseline_overall[
                        "f1"
                    ]
                )
        },

        {
            "Method":
                "Transcript+OCR",

            "Ground Truth":
                int(
                    proposed_overall[
                        "ground_truth_count"
                    ]
                ),

            "Predicted":
                int(
                    proposed_overall[
                        "predicted_count"
                    ]
                ),

            "TP":
                int(
                    proposed_overall[
                        "TP"
                    ]
                ),

            "FP":
                int(
                    proposed_overall[
                        "FP"
                    ]
                ),

            "FN":
                int(
                    proposed_overall[
                        "FN"
                    ]
                ),

            "Precision":
                float(
                    proposed_overall[
                        "precision"
                    ]
                ),

            "Recall":
                float(
                    proposed_overall[
                        "recall"
                    ]
                ),

            "F1":
                float(
                    proposed_overall[
                        "f1"
                    ]
                )
        }

    ])


    # -----------------------------------------------------
    # DIFFERENCE ROW
    # -----------------------------------------------------

    difference = {

        "Method":
            "Difference (OCR - Baseline)",

        "Ground Truth":
            int(
                proposed_overall[
                    "ground_truth_count"
                ]
                -
                baseline_overall[
                    "ground_truth_count"
                ]
            ),

        "Predicted":
            int(
                proposed_overall[
                    "predicted_count"
                ]
                -
                baseline_overall[
                    "predicted_count"
                ]
            ),

        "TP":
            int(
                proposed_overall["TP"]
                -
                baseline_overall["TP"]
            ),

        "FP":
            int(
                proposed_overall["FP"]
                -
                baseline_overall["FP"]
            ),

        "FN":
            int(
                proposed_overall["FN"]
                -
                baseline_overall["FN"]
            ),

        "Precision":
            round(
                float(
                    proposed_overall[
                        "precision"
                    ]
                    -
                    baseline_overall[
                        "precision"
                    ]
                ),
                4
            ),

        "Recall":
            round(
                float(
                    proposed_overall[
                        "recall"
                    ]
                    -
                    baseline_overall[
                        "recall"
                    ]
                ),
                4
            ),

        "F1":
            round(
                float(
                    proposed_overall[
                        "f1"
                    ]
                    -
                    baseline_overall[
                        "f1"
                    ]
                ),
                4
            )
    }


    comparison = pd.concat(
        [
            comparison,
            pd.DataFrame(
                [difference]
            )
        ],
        ignore_index=True
    )


    # -----------------------------------------------------
    # ROUND
    # -----------------------------------------------------

    for column in [
        "Precision",
        "Recall",
        "F1"
    ]:

        comparison[column] = (
            comparison[column]
            .astype(float)
            .round(4)
        )


    # -----------------------------------------------------
    # SAVE
    # -----------------------------------------------------

    comparison.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )


    # -----------------------------------------------------
    # PRINT
    # -----------------------------------------------------

    print(
        "\n========== FINAL RESULT =========="
    )

    print(
        comparison.to_string(
            index=False
        )
    )


    print(
        "\n======================================"
    )

    print(
        "HOÀN THÀNH MODULE 1 COMPARISON"
    )

    print(
        "\nOutput:"
    )

    print(
        OUTPUT_FILE
    )

    print(
        "======================================"
    )


if __name__ == "__main__":
    main()