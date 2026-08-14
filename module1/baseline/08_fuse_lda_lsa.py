from pathlib import Path
import pandas as pd


# =========================================================
# 1. CONFIG
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULT_DIR = (
    PROJECT_ROOT
    / "module1"
    / "results"
    / "baseline"
)

LDA_FILE = (
    RESULT_DIR
    / "lda_concept_scores.csv"
)

LSA_FILE = (
    RESULT_DIR
    / "lsa_concept_scores.csv"
)

OUTPUT_ALL = (
    RESULT_DIR
    / "fusion_scores.csv"
)

OUTPUT_TOP = (
    RESULT_DIR
    / "fusion_top_concepts.csv"
)


# =========================================================
# 2. FUSION WEIGHTS
# =========================================================

# LSA giữ vai trò semantic matching chính
LSA_WEIGHT = 0.60

# LDA cung cấp topic evidence
LDA_WEIGHT = 0.40


# =========================================================
# 3. MIN-MAX NORMALIZATION
# =========================================================

def normalize_series(series):
    """
    Chuẩn hóa một nhóm score về khoảng 0 -> 1.

    Ví dụ:

    raw:
        0.10
        0.20
        0.40

    normalized:
        0.00
        0.33
        1.00
    """

    min_value = series.min()
    max_value = series.max()

    # Nếu tất cả score giống nhau
    if max_value == min_value:
        return pd.Series(
            0.0,
            index=series.index
        )

    return (
        series - min_value
    ) / (
        max_value - min_value
    )


# =========================================================
# 4. MAIN
# =========================================================

def main():

    print("======================================")
    print("MODULE 1 - BASELINE")
    print("STEP 08 - FUSE LDA + LSA")
    print("======================================")

    # =====================================================
    # 5. CHECK INPUT
    # =====================================================

    if not LDA_FILE.exists():

        print("Không tìm thấy LDA file:")
        print(LDA_FILE)

        return

    if not LSA_FILE.exists():

        print("Không tìm thấy LSA file:")
        print(LSA_FILE)

        return


    # =====================================================
    # 6. READ DATA
    # =====================================================

    lda = pd.read_csv(
        LDA_FILE
    )

    lsa = pd.read_csv(
        LSA_FILE
    )


    print(
        "Số dòng LDA:",
        len(lda)
    )

    print(
        "Số dòng LSA:",
        len(lsa)
    )


    # =====================================================
    # 7. MERGE LDA + LSA
    # =====================================================

    key_columns = [
        "video_id",
        "subject",
        "method",
        "chunk_id",
        "start_sec",
        "end_sec",
        "concept"
    ]


    merged = lda.merge(
        lsa,
        on=key_columns,
        how="inner"
    )


    print(
        "Số dòng sau merge:",
        len(merged)
    )


    if merged.empty:

        print(
            "Không merge được LDA và LSA."
        )

        return


    # =====================================================
    # 8. NORMALIZE SCORE TRONG TỪNG CHUNK
    # =====================================================

    # Quan trọng:
    #
    # Không normalize toàn dataset.
    #
    # Mỗi chunk sẽ tự so sánh các concept
    # thuộc chunk đó với nhau.


    merged["lda_norm"] = (
        merged
        .groupby(
            [
                "video_id",
                "chunk_id"
            ]
        )["lda_score"]
        .transform(
            normalize_series
        )
    )


    merged["lsa_norm"] = (
        merged
        .groupby(
            [
                "video_id",
                "chunk_id"
            ]
        )["lsa_score"]
        .transform(
            normalize_series
        )
    )


    # =====================================================
    # 9. FUSION
    # =====================================================

    merged["final_score"] = (

        LDA_WEIGHT
        * merged["lda_norm"]

        +

        LSA_WEIGHT
        * merged["lsa_norm"]
    )


    # Làm tròn để xem cho đẹp
    merged["lda_norm"] = (
        merged["lda_norm"]
        .round(4)
    )

    merged["lsa_norm"] = (
        merged["lsa_norm"]
        .round(4)
    )

    merged["final_score"] = (
        merged["final_score"]
        .round(4)
    )


    # =====================================================
    # 10. RANK CONCEPT TRONG TỪNG CHUNK
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
    # 11. SORT
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
        .reset_index(drop=True)
    )


    # =====================================================
    # 12. SAVE ALL FUSION SCORES
    # =====================================================

    merged.to_csv(
        OUTPUT_ALL,
        index=False,
        encoding="utf-8-sig"
    )


    # =====================================================
    # 13. TOP 3 CONCEPT / CHUNK
    # =====================================================

    top = (
        merged[
            merged["rank"] <= 3
        ]
        .copy()
        .reset_index(drop=True)
    )


    top.to_csv(
        OUTPUT_TOP,
        index=False,
        encoding="utf-8-sig"
    )


    # =====================================================
    # 14. TERMINAL PREVIEW
    # =====================================================

    print(
        "\n========== TOP FUSION RESULTS =========="
    )


    preview_columns = [
        "video_id",
        "chunk_id",
        "concept",
        "lda_score",
        "lsa_score",
        "lda_norm",
        "lsa_norm",
        "final_score",
        "rank"
    ]


    print(
        top[
            preview_columns
        ]
        .head(40)
        .to_string(
            index=False
        )
    )


    # =====================================================
    # 15. DONE
    # =====================================================

    print(
        "\n======================================"
    )

    print(
        "HOÀN THÀNH STEP 08"
    )


    print(
        "\nAll fusion scores:"
    )

    print(
        OUTPUT_ALL
    )


    print(
        "\nTop concepts:"
    )

    print(
        OUTPUT_TOP
    )


    print(
        "======================================"
    )


if __name__ == "__main__":
    main()