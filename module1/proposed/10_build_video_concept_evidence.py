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
    / "proposed"
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
# 2. TOP-K MEAN
# =========================================================

def top_k_mean(series, k=2):

    values = (
        series
        .sort_values(
            ascending=False
        )
        .head(k)
    )

    if len(values) == 0:
        return 0.0

    return float(
        values.mean()
    )


# =========================================================
# 3. MAIN
# =========================================================

def main():

    print("======================================")
    print("MODULE 1 - PROPOSED")
    print("STEP 10 - VIDEO CONCEPT EVIDENCE")
    print("METHOD: TRANSCRIPT + OCR")
    print("======================================")


    # -----------------------------------------------------
    # READ FUSION SCORES
    # -----------------------------------------------------

    df = pd.read_csv(
        INPUT_FILE
    )

    print(
        "Fusion rows:",
        len(df)
    )


    # -----------------------------------------------------
    # SORT
    # -----------------------------------------------------

    df = (
        df
        .sort_values(
            [
                "video_id",
                "concept",
                "chunk_id"
            ]
        )
        .reset_index(
            drop=True
        )
    )


    # -----------------------------------------------------
    # BUILD VIDEO-CONCEPT EVIDENCE
    # -----------------------------------------------------

    result_rows = []


    for (
        video_id,
        subject,
        method,
        concept
    ), group in df.groupby(
        [
            "video_id",
            "subject",
            "method",
            "concept"
        ]
    ):

        group = group.copy()


        # =================================================
        # NUMBER OF CHUNKS
        # =================================================

        num_chunks = len(
            group
        )


        # =================================================
        # SCORE STATISTICS
        # =================================================

        max_final_score = float(
            group[
                "final_score"
            ].max()
        )

        mean_final_score = float(
            group[
                "final_score"
            ].mean()
        )

        top2_mean_score = top_k_mean(
            group[
                "final_score"
            ],
            k=2
        )


        # =================================================
        # TOP-3 EVIDENCE
        # =================================================

        top3_count = int(
            (
                group["rank"] <= 3
            ).sum()
        )

        top3_ratio = (
            top3_count
            / num_chunks
            if num_chunks > 0
            else 0.0
        )


        # =================================================
        # TOP-1 EVIDENCE
        # =================================================

        top1_count = int(
            (
                group["rank"] == 1
            ).sum()
        )

        top1_ratio = (
            top1_count
            / num_chunks
            if num_chunks > 0
            else 0.0
        )


        # =================================================
        # RAW LSA / LDA SUPPORT
        # =================================================

        max_lsa_score = float(
            group[
                "lsa_score"
            ].max()
        )

        max_lda_score = float(
            group[
                "lda_score"
            ].max()
        )


        # =================================================
        # BEST CHUNK
        # =================================================

        best_index = (
            group[
                "final_score"
            ]
            .idxmax()
        )

        best_row = group.loc[
            best_index
        ]


        best_chunk_id = int(
            best_row[
                "chunk_id"
            ]
        )

        best_start_sec = float(
            best_row[
                "start_sec"
            ]
        )

        best_end_sec = float(
            best_row[
                "end_sec"
            ]
        )


        # =================================================
        # SAVE
        # =================================================

        result_rows.append({

            "video_id":
                video_id,

            "subject":
                subject,

            "method":
                method,

            "concept":
                concept,

            "num_chunks":
                num_chunks,

            "max_final_score":
                round(
                    max_final_score,
                    6
                ),

            "mean_final_score":
                round(
                    mean_final_score,
                    6
                ),

            "top2_mean_score":
                round(
                    top2_mean_score,
                    6
                ),

            "top3_count":
                top3_count,

            "top3_ratio":
                round(
                    top3_ratio,
                    6
                ),

            "top1_count":
                top1_count,

            "top1_ratio":
                round(
                    top1_ratio,
                    6
                ),

            "max_lsa_score":
                round(
                    max_lsa_score,
                    6
                ),

            "max_lda_score":
                round(
                    max_lda_score,
                    6
                ),

            "best_chunk_id":
                best_chunk_id,

            "best_start_sec":
                best_start_sec,

            "best_end_sec":
                best_end_sec
        })


    # -----------------------------------------------------
    # DATAFRAME
    # -----------------------------------------------------

    result = pd.DataFrame(
        result_rows
    )


    result = (
        result
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


    # -----------------------------------------------------
    # SAVE
    # -----------------------------------------------------

    result.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )


    # -----------------------------------------------------
    # PREVIEW
    # -----------------------------------------------------

    print(
        "\n========== VIDEO CONCEPT EVIDENCE =========="
    )

    print(
        result[
            [
                "video_id",
                "concept",
                "max_final_score",
                "top2_mean_score",
                "top3_count",
                "top1_count",
                "best_chunk_id"
            ]
        ]
        .head(30)
        .to_string(
            index=False
        )
    )


    # -----------------------------------------------------
    # SUMMARY
    # -----------------------------------------------------

    print(
        "\nVideo-concept rows:",
        len(result)
    )

    print(
        "\n======================================"
    )

    print(
        "HOÀN THÀNH STEP 10"
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