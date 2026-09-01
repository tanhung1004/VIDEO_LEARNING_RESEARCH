import pandas as pd


# =========================================================
# STEP 08 - FUSE LDA + LSA
# SCAFFOLD40 OLD vs NEW COMPARISON
# =========================================================

from step00_compare_config import (
    VARIANT,
    RUN_ROOT,
    LDA_WEIGHT,
    LSA_WEIGHT,
)


# =========================================================
# 1. INPUT / OUTPUT
# =========================================================

RESULT_DIR = RUN_ROOT


LDA_FILE = (
    RESULT_DIR
    / "lda_concept_scores.csv"
)


LSA_FILE = (
    RESULT_DIR
    / "lsa_concept_scores.csv"
)


OUTPUT_FILE = (
    RESULT_DIR
    / "fusion_scores.csv"
)


TOP_FILE = (
    RESULT_DIR
    / "fusion_top_concepts.csv"
)


# =========================================================
# 2. MIN-MAX NORMALIZATION
# =========================================================

def normalize_series(series):

    min_value = series.min()
    max_value = series.max()

    # Nếu tất cả concept score giống nhau
    # thì scaffold cũ trả về 0.
    if max_value == min_value:

        return pd.Series(
            0.0,
            index=series.index,
        )

    return (
        series - min_value
    ) / (
        max_value - min_value
    )


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
        "STEP 08 - FUSE LDA + LSA"
    )

    print(
        "======================================"
    )


    # =====================================================
    # 4. CHECK INPUT
    # =====================================================

    if not LDA_FILE.exists():

        raise FileNotFoundError(
            "Không tìm thấy LDA file: "
            f"{LDA_FILE}"
        )


    if not LSA_FILE.exists():

        raise FileNotFoundError(
            "Không tìm thấy LSA file: "
            f"{LSA_FILE}"
        )


    # =====================================================
    # 5. READ DATA
    # =====================================================

    lda = pd.read_csv(
        LDA_FILE
    )

    lsa = pd.read_csv(
        LSA_FILE
    )


    print(
        "\nVariant:",
        VARIANT
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
        "LDA rows:",
        len(lda)
    )

    print(
        "LSA rows:",
        len(lsa)
    )


    # =====================================================
    # 6. INPUT AUDIT
    # =====================================================

    required_lda_columns = {
        "video_id",
        "subject",
        "method",
        "chunk_id",
        "start_sec",
        "end_sec",
        "concept",
        "lda_score",
    }


    required_lsa_columns = {
        "video_id",
        "subject",
        "method",
        "chunk_id",
        "start_sec",
        "end_sec",
        "concept",
        "lsa_score",
    }


    missing_lda = (
        required_lda_columns
        - set(lda.columns)
    )

    missing_lsa = (
        required_lsa_columns
        - set(lsa.columns)
    )


    if missing_lda:

        raise RuntimeError(
            "LDA input missing columns: "
            f"{sorted(missing_lda)}"
        )


    if missing_lsa:

        raise RuntimeError(
            "LSA input missing columns: "
            f"{sorted(missing_lsa)}"
        )


    # -----------------------------------------------------
    # BOTH FILES MUST HAVE SAME ROW COUNT
    # -----------------------------------------------------

    if len(lda) != len(lsa):

        raise RuntimeError(
            "LDA and LSA row count mismatch. "
            f"LDA={len(lda)}, "
            f"LSA={len(lsa)}"
        )


    # -----------------------------------------------------
    # EXACT v1-v40
    # -----------------------------------------------------

    expected_ids = {
        f"v{i}"
        for i in range(1, 41)
    }


    lda_ids = set(
        lda["video_id"]
        .astype(str)
        .str.strip()
    )


    lsa_ids = set(
        lsa["video_id"]
        .astype(str)
        .str.strip()
    )


    if lda_ids != expected_ids:

        raise RuntimeError(
            "LDA input is not exactly v1-v40."
        )


    if lsa_ids != expected_ids:

        raise RuntimeError(
            "LSA input is not exactly v1-v40."
        )


    # -----------------------------------------------------
    # DUPLICATE SCORES
    # -----------------------------------------------------

    duplicate_lda = (
        lda
        .duplicated(
            [
                "video_id",
                "chunk_id",
                "concept",
            ]
        )
        .sum()
    )


    duplicate_lsa = (
        lsa
        .duplicated(
            [
                "video_id",
                "chunk_id",
                "concept",
            ]
        )
        .sum()
    )


    if duplicate_lda != 0:

        raise RuntimeError(
            "Duplicate LDA score rows: "
            f"{duplicate_lda}"
        )


    if duplicate_lsa != 0:

        raise RuntimeError(
            "Duplicate LSA score rows: "
            f"{duplicate_lsa}"
        )


    print(
        "Input videos:",
        len(lda_ids)
    )

    print(
        "Duplicate LDA rows:",
        duplicate_lda
    )

    print(
        "Duplicate LSA rows:",
        duplicate_lsa
    )


    # =====================================================
    # 7. MERGE LDA + LSA
    # =====================================================

    merge_columns = [
        "video_id",
        "subject",
        "method",
        "chunk_id",
        "start_sec",
        "end_sec",
        "concept",
    ]


    merged = pd.merge(
        lda,
        lsa,
        on=merge_columns,
        how="inner",
        validate="one_to_one",
    )


    if merged.empty:

        raise RuntimeError(
            "Không merge được LDA và LSA."
        )


    if len(merged) != len(lda):

        raise RuntimeError(
            "Fusion merge lost rows. "
            f"Expected={len(lda)}, "
            f"Merged={len(merged)}"
        )


    print(
        "Merged rows:",
        len(merged)
    )


    # =====================================================
    # 8. NORMALIZE SCORE TRONG TỪNG CHUNK
    # =====================================================
    #
    # QUAN TRỌNG:
    #
    # Không normalize toàn bộ dataset.
    #
    # Trong mỗi (video_id, chunk_id),
    # concept cùng subject được so với nhau.
    #
    # Đây là logic của scaffold cũ.
    # =====================================================

    merged["lda_norm"] = (
        merged
        .groupby(
            [
                "video_id",
                "chunk_id",
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
                "chunk_id",
            ]
        )["lsa_score"]
        .transform(
            normalize_series
        )
    )


    # =====================================================
    # 9. FUSION
    # =====================================================
    #
    # OLD
    # final = 0.4 * LDA + 0.6 * LSA
    #
    # NEW
    # final = 0.0 * LDA + 1.0 * LSA
    #
    # =====================================================

    merged["final_score"] = (

        LDA_WEIGHT
        * merged["lda_norm"]

        +

        LSA_WEIGHT
        * merged["lsa_norm"]
    )


    # -----------------------------------------------------
    # Scaffold cũ round score trước video aggregation
    # -----------------------------------------------------

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
    # 10. RANK CONCEPTS WITHIN EACH CHUNK
    # =====================================================

    merged["rank"] = (
        merged
        .groupby(
            [
                "video_id",
                "chunk_id",
            ]
        )["final_score"]
        .rank(
            method="first",
            ascending=False,
        )
        .astype(int)
    )


    # =====================================================
    # 11. FINAL AUDIT
    # =====================================================

    output_ids = set(
        merged["video_id"]
        .astype(str)
        .str.strip()
    )


    if output_ids != expected_ids:

        raise RuntimeError(
            "STEP 08 output is not exactly v1-v40."
        )


    duplicate_output = (
        merged
        .duplicated(
            [
                "video_id",
                "chunk_id",
                "concept",
            ]
        )
        .sum()
    )


    if duplicate_output != 0:

        raise RuntimeError(
            "Duplicate fusion rows found: "
            f"{duplicate_output}"
        )


    # -----------------------------------------------------
    # NORMALIZATION RANGE AUDIT
    # -----------------------------------------------------

    if not (
        merged["lda_norm"]
        .between(
            0.0,
            1.0,
        )
        .all()
    ):

        raise RuntimeError(
            "lda_norm outside [0,1]."
        )


    if not (
        merged["lsa_norm"]
        .between(
            0.0,
            1.0,
        )
        .all()
    ):

        raise RuntimeError(
            "lsa_norm outside [0,1]."
        )


    if not (
        merged["final_score"]
        .between(
            0.0,
            1.0,
        )
        .all()
    ):

        raise RuntimeError(
            "final_score outside [0,1]."
        )


    # -----------------------------------------------------
    # NEW MUST BE LSA-ONLY
    # -----------------------------------------------------

    if VARIANT == "new":

        mismatch = (
            merged["final_score"]
            != merged["lsa_norm"]
        ).sum()

        if mismatch != 0:

            raise RuntimeError(
                "NEW fusion must equal LSA normalized "
                f"score. Mismatched rows={mismatch}"
            )


    # =====================================================
    # 12. SAVE FULL FUSION SCORES
    # =====================================================

    merged = (
        merged
        .sort_values(
            [
                "video_id",
                "chunk_id",
                "rank",
            ]
        )
        .reset_index(
            drop=True
        )
    )


    merged.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )


    # =====================================================
    # 13. SAVE TOP 3 CONCEPTS PER CHUNK
    # =====================================================

    top_df = (
        merged[
            merged["rank"] <= 3
        ]
        .copy()
    )


    top_df.to_csv(
        TOP_FILE,
        index=False,
        encoding="utf-8-sig",
    )


    # =====================================================
    # 14. COMPLETE
    # =====================================================

    print(
        "\n======================================"
    )

    print(
        "STEP 08 PASS"
    )

    print(
        "======================================"
    )

    print(
        "Variant:",
        VARIANT
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
        "DEV videos:",
        merged[
            "video_id"
        ].nunique()
    )

    print(
        "Fusion rows:",
        len(merged)
    )

    print(
        "Duplicate fusion rows:",
        duplicate_output
    )

    print(
        "Final score min:",
        merged[
            "final_score"
        ].min()
    )

    print(
        "Final score max:",
        merged[
            "final_score"
        ].max()
    )


    print(
        "\nFusion file:"
    )

    print(
        OUTPUT_FILE
    )


    print(
        "\nTop concept file:"
    )

    print(
        TOP_FILE
    )


    print(
        "======================================"
    )


if __name__ == "__main__":

    main()