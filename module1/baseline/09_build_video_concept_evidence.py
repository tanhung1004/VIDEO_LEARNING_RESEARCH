from pathlib import Path
import pandas as pd


# =========================================================
# 1. PATH
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULT_DIR = (
    PROJECT_ROOT
    / "module1"
    / "results"
    / "baseline"
)

INPUT_FILE = (
    RESULT_DIR
    / "fusion_scores.csv"
)

OUTPUT_FILE = (
    RESULT_DIR
    / "video_concept_evidence.csv"
)


# =========================================================
# 2. MAIN
# =========================================================

def main():

    print("======================================")
    print("MODULE 1 - BASELINE")
    print("STEP 09A - VIDEO CONCEPT EVIDENCE")
    print("======================================")

    # -----------------------------------------------------
    # CHECK INPUT
    # -----------------------------------------------------

    if not INPUT_FILE.exists():

        print("Không tìm thấy:")
        print(INPUT_FILE)

        return

    # -----------------------------------------------------
    # READ
    # -----------------------------------------------------

    df = pd.read_csv(INPUT_FILE)

    print(
        "Số dòng fusion:",
        len(df)
    )

    # -----------------------------------------------------
    # ĐẾM SỐ CHUNK CỦA MỖI VIDEO
    # -----------------------------------------------------

    chunk_counts = (
        df
        .groupby("video_id")["chunk_id"]
        .nunique()
        .to_dict()
    )

    result_rows = []

    # -----------------------------------------------------
    # GROUP THEO VIDEO + CONCEPT
    # -----------------------------------------------------

    grouped = df.groupby(
        [
            "video_id",
            "subject",
            "concept"
        ]
    )

    for (
        video_id,
        subject,
        concept
    ), group in grouped:

        num_chunks = (
            chunk_counts[
                video_id
            ]
        )

        # =============================================
        # SCORE STATISTICS
        # =============================================

        max_final = (
            group["final_score"]
            .max()
        )

        mean_final = (
            group["final_score"]
            .mean()
        )

        max_lsa = (
            group["lsa_score"]
            .max()
        )

        max_lda = (
            group["lda_score"]
            .max()
        )

        # =============================================
        # RANK SUPPORT
        # =============================================

        top3_count = int(
            (
                group["rank"] <= 3
            ).sum()
        )

        top1_count = int(
            (
                group["rank"] == 1
            ).sum()
        )

        top3_ratio = (
            top3_count
            / num_chunks
        )

        top1_ratio = (
            top1_count
            / num_chunks
        )

        # =============================================
        # TOP-2 STRONGEST EVIDENCE
        # =============================================

        strongest = (
            group["final_score"]
            .nlargest(2)
        )

        top2_mean = (
            strongest.mean()
        )

        # =============================================
        # BEST CHUNK
        # =============================================

        best_index = (
            group["final_score"]
            .idxmax()
        )

        best_row = df.loc[
            best_index
        ]

        # =============================================
        # SAVE ROW
        # =============================================

        result_rows.append({

            "video_id":
                video_id,

            "subject":
                subject,

            "concept":
                concept,

            "num_chunks":
                num_chunks,

            "max_final_score":
                round(
                    float(max_final),
                    4
                ),

            "mean_final_score":
                round(
                    float(mean_final),
                    4
                ),

            "top2_mean_score":
                round(
                    float(top2_mean),
                    4
                ),

            "top3_count":
                top3_count,

            "top3_ratio":
                round(
                    float(top3_ratio),
                    4
                ),

            "top1_count":
                top1_count,

            "top1_ratio":
                round(
                    float(top1_ratio),
                    4
                ),

            "max_lsa_score":
                round(
                    float(max_lsa),
                    4
                ),

            "max_lda_score":
                round(
                    float(max_lda),
                    4
                ),

            "best_chunk_id":
                int(
                    best_row["chunk_id"]
                ),

            "best_start_sec":
                best_row["start_sec"],

            "best_end_sec":
                best_row["end_sec"]
        })


    # =====================================================
    # CREATE DATAFRAME
    # =====================================================

    result = pd.DataFrame(
        result_rows
    )


    # -----------------------------------------------------
    # SORT
    # -----------------------------------------------------

    result = (
        result
        .sort_values(
            [
                "video_id",
                "top2_mean_score",
                "top3_ratio"
            ],
            ascending=[
                True,
                False,
                False
            ]
        )
        .reset_index(drop=True)
    )


    # -----------------------------------------------------
    # SAVE
    # -----------------------------------------------------

    result.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )


    # =====================================================
    # PREVIEW
    # =====================================================

    print(
        "\n========== VIDEO CONCEPT EVIDENCE =========="
    )

    preview_columns = [
        "video_id",
        "concept",
        "max_final_score",
        "top2_mean_score",
        "top3_count",
        "top3_ratio",
        "max_lsa_score",
        "max_lda_score"
    ]

    print(
        result[
            preview_columns
        ]
        .head(40)
        .to_string(index=False)
    )


    print(
        "\n======================================"
    )

    print(
        "HOÀN THÀNH STEP 09A"
    )

    print(
        "Output:"
    )

    print(
        OUTPUT_FILE
    )

    print(
        "======================================"
    )


if __name__ == "__main__":
    main()