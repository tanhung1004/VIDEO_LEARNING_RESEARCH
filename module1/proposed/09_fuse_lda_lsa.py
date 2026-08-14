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

LDA_FILE = (
    RESULT_DIR
    / "lda_concept_scores.csv"
)

LSA_FILE = (
    RESULT_DIR
    / "lsa_concept_scores.csv"
)


# =========================================================
# 3. OUTPUT
# =========================================================

FUSION_FILE = (
    RESULT_DIR
    / "fusion_scores.csv"
)

TOP_FILE = (
    RESULT_DIR
    / "fusion_top_concepts.csv"
)


# =========================================================
# 4. FUSION CONFIG
# =========================================================
#
# GIỮ Y CHANG BASELINE
#
# final =
#     0.4 * normalized LDA
#   + 0.6 * normalized LSA
#
# =========================================================

LDA_WEIGHT = 0.40
LSA_WEIGHT = 0.60


# =========================================================
# 5. MIN-MAX NORMALIZATION
# =========================================================

def minmax_normalize(series):

    minimum = series.min()
    maximum = series.max()

    if maximum == minimum:

        return pd.Series(
            [0.0] * len(series),
            index=series.index
        )

    return (
        (series - minimum)
        /
        (maximum - minimum)
    )


# =========================================================
# 6. MAIN
# =========================================================

def main():

    print("======================================")
    print("MODULE 1 - PROPOSED")
    print("STEP 09 - FUSE LDA + LSA")
    print("INPUT: TRANSCRIPT + OCR")
    print("======================================")


    # -----------------------------------------------------
    # 7. READ DATA
    # -----------------------------------------------------

    lda = pd.read_csv(
        LDA_FILE
    )

    lsa = pd.read_csv(
        LSA_FILE
    )


    print(
        "LDA rows:",
        len(lda)
    )

    print(
        "LSA rows:",
        len(lsa)
    )


    # -----------------------------------------------------
    # 8. MERGE KEYS
    # -----------------------------------------------------

    merge_columns = [
        "video_id",
        "subject",
        "method",
        "chunk_id",
        "start_sec",
        "end_sec",
        "concept"
    ]


    # -----------------------------------------------------
    # 9. MERGE LDA + LSA
    # -----------------------------------------------------

    merged = pd.merge(

        lda,

        lsa,

        on=merge_columns,

        how="inner"
    )


    print(
        "Merged rows:",
        len(merged)
    )


    if len(merged) == 0:

        raise ValueError(
            "Không merge được LDA và LSA. "
            "Kiểm tra cột method / chunk / concept."
        )


    # =====================================================
    # 10. NORMALIZE PER VIDEO + CHUNK
    #
    # Rất quan trọng:
    #
    # normalize các concept cạnh tranh
    # TRONG CÙNG một chunk.
    #
    # Không normalize toàn dataset.
    # =====================================================

    group_columns = [
        "video_id",
        "chunk_id"
    ]


    merged["lda_norm"] = (

        merged

        .groupby(
            group_columns
        )["lda_score"]

        .transform(
            minmax_normalize
        )
    )


    merged["lsa_norm"] = (

        merged

        .groupby(
            group_columns
        )["lsa_score"]

        .transform(
            minmax_normalize
        )
    )


    # =====================================================
    # 11. FUSION
    # =====================================================

    merged["final_score"] = (

        LDA_WEIGHT
        * merged["lda_norm"]

        +

        LSA_WEIGHT
        * merged["lsa_norm"]
    )


    # =====================================================
    # 12. RANK CONCEPT PER CHUNK
    # =====================================================

    merged["rank"] = (

        merged

        .groupby(
            [
                "video_id",
                "chunk_id"
            ]
        )["final_score"]

        .rank(
            method="first",
            ascending=False
        )

        .astype(int)
    )


    # =====================================================
    # 13. SORT
    # =====================================================

    merged = (

        merged

        .sort_values(
            [
                "video_id",
                "chunk_id",
                "rank"
            ]
        )

        .reset_index(
            drop=True
        )
    )


    # =====================================================
    # 14. ROUND SCORES
    # =====================================================

    for column in [
        "lda_score",
        "lsa_score",
        "lda_norm",
        "lsa_norm",
        "final_score"
    ]:

        merged[column] = (

            merged[column]

            .astype(float)

            .round(6)
        )


    # =====================================================
    # 15. SAVE ALL FUSION SCORES
    # =====================================================

    merged.to_csv(

        FUSION_FILE,

        index=False,

        encoding="utf-8-sig"
    )


    # =====================================================
    # 16. TOP 3 CONCEPT PER CHUNK
    # =====================================================

    top_concepts = (

        merged[
            merged["rank"] <= 3
        ]

        .copy()

        .reset_index(
            drop=True
        )
    )


    top_concepts.to_csv(

        TOP_FILE,

        index=False,

        encoding="utf-8-sig"
    )


    # =====================================================
    # 17. PREVIEW
    # =====================================================

    print(
        "\n========== FUSION TOP CONCEPTS =========="
    )

    print(

        top_concepts[
            [
                "video_id",
                "chunk_id",
                "concept",
                "lda_norm",
                "lsa_norm",
                "final_score",
                "rank"
            ]
        ]

        .head(30)

        .to_string(
            index=False
        )
    )


    # =====================================================
    # 18. SCORE SUMMARY
    # =====================================================

    print(
        "\n========== SCORE SUMMARY =========="
    )

    print(
        "Final min:",
        round(
            merged["final_score"].min(),
            6
        )
    )

    print(
        "Final max:",
        round(
            merged["final_score"].max(),
            6
        )
    )

    print(
        "Final mean:",
        round(
            merged["final_score"].mean(),
            6
        )
    )


    # =====================================================
    # 19. DONE
    # =====================================================

    print(
        "\n======================================"
    )

    print(
        "HOÀN THÀNH STEP 09"
    )

    print(
        "\nFusion scores:"
    )

    print(
        FUSION_FILE
    )

    print(
        "\nTop concepts:"
    )

    print(
        TOP_FILE
    )

    print(
        "======================================"
    )


if __name__ == "__main__":
    main()