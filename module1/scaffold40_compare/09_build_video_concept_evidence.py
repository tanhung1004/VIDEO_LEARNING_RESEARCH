import pandas as pd


# =========================================================
# STEP 09 - BUILD VIDEO-LEVEL CONCEPT EVIDENCE
# SCAFFOLD40 OLD vs NEW COMPARISON
# =========================================================

from step00_compare_config import (
    VARIANT,
    RUN_ROOT,
)


# =========================================================
# 1. INPUT / OUTPUT
# =========================================================

RESULT_DIR = RUN_ROOT


INPUT_FILE = (
    RESULT_DIR
    / "fusion_scores.csv"
)


OUTPUT_FILE = (
    RESULT_DIR
    / "video_concept_evidence.csv"
)


# =========================================================
# 2. VIDEO SORT HELPER
# =========================================================

def video_number(video_id):

    text = str(video_id).strip()

    if (
        text.startswith("v")
        and text[1:].isdigit()
    ):
        return int(text[1:])

    return 999999


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
        "STEP 09 - BUILD VIDEO CONCEPT EVIDENCE"
    )

    print(
        "======================================"
    )


    # =====================================================
    # 4. CHECK INPUT
    # =====================================================

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            "STEP 09 cannot find fusion file: "
            f"{INPUT_FILE}"
        )


    # =====================================================
    # 5. READ FUSION SCORES
    # =====================================================

    df = pd.read_csv(
        INPUT_FILE
    )


    # =====================================================
    # 6. INPUT AUDIT
    # =====================================================

    required_columns = {
        "video_id",
        "subject",
        "chunk_id",
        "start_sec",
        "end_sec",
        "concept",
        "lda_score",
        "lsa_score",
        "lda_norm",
        "lsa_norm",
        "final_score",
        "rank",
    }


    missing_columns = (
        required_columns
        - set(df.columns)
    )


    if missing_columns:

        raise RuntimeError(
            "STEP 09 fusion input missing columns: "
            f"{sorted(missing_columns)}"
        )


    # -----------------------------------------------------
    # EXACT v1 -> v40
    # -----------------------------------------------------

    expected_ids = {
        f"v{i}"
        for i in range(1, 41)
    }


    actual_ids = set(
        df["video_id"]
        .astype(str)
        .str.strip()
    )


    if actual_ids != expected_ids:

        raise RuntimeError(
            "STEP 09 input is not exactly v1-v40. "
            f"Missing={sorted(expected_ids - actual_ids)}, "
            f"Unexpected={sorted(actual_ids - expected_ids)}"
        )


    # -----------------------------------------------------
    # DUPLICATE CHUNK-CONCEPT SCORE
    # -----------------------------------------------------

    duplicate_rows = (
        df
        .duplicated(
            [
                "video_id",
                "chunk_id",
                "concept",
            ]
        )
        .sum()
    )


    if duplicate_rows != 0:

        raise RuntimeError(
            "STEP 09 found duplicate fusion rows: "
            f"{duplicate_rows}"
        )


    # -----------------------------------------------------
    # SUBJECT DISTRIBUTION
    # -----------------------------------------------------

    video_subject = (
        df[
            [
                "video_id",
                "subject",
            ]
        ]
        .drop_duplicates()
    )


    subject_counts = (
        video_subject["subject"]
        .astype(str)
        .str.strip()
        .value_counts()
        .to_dict()
    )


    expected_subject_counts = {
        "SQL": 10,
        "Python": 10,
        "Java": 10,
        "C++": 10,
    }


    if subject_counts != expected_subject_counts:

        raise RuntimeError(
            "Unexpected subject distribution: "
            f"{subject_counts}"
        )


    # -----------------------------------------------------
    # COUNT UNIQUE CHUNKS
    # -----------------------------------------------------

    unique_chunks = (
        df[
            [
                "video_id",
                "chunk_id",
            ]
        ]
        .drop_duplicates()
    )


    total_chunks = len(
        unique_chunks
    )


    chunk_counts = (
        unique_chunks
        .groupby(
            "video_id"
        )
        .size()
        .to_dict()
    )


    print(
        "\n======================================"
    )

    print(
        "STEP 09 INPUT AUDIT"
    )

    print(
        "======================================"
    )

    print(
        "Variant:",
        VARIANT
    )

    print(
        "DEV videos:",
        df["video_id"].nunique()
    )

    print(
        "Unique chunks:",
        total_chunks
    )

    print(
        "Fusion rows:",
        len(df)
    )

    print(
        "Duplicate fusion rows:",
        duplicate_rows
    )

    print(
        "Subject counts:",
        subject_counts
    )

    print(
        "Input:"
    )

    print(
        INPUT_FILE
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


    # =====================================================
    # 7. GROUP BY VIDEO + CONCEPT
    # =====================================================
    #
    # Một concept có nhiều chunk evidence.
    #
    # Ta gom chúng lại thành video-level evidence:
    #
    # - max
    # - mean
    # - top2 mean
    # - rank support
    # - best chunk / timestamp
    #
    # CẢ OLD VÀ NEW đều tạo tất cả các cột.
    #
    # STEP 10 mới chọn:
    #
    # OLD -> top2_mean_score
    # NEW -> max_final_score
    # =====================================================

    grouped = (
        df
        .groupby(
            [
                "video_id",
                "subject",
                "concept",
            ],
            sort=False,
        )
    )


    result_rows = []


    for (
        video_id,
        subject,
        concept,
    ), group in grouped:


        # -------------------------------------------------
        # NUMBER OF CHUNKS IN THIS VIDEO
        # -------------------------------------------------

        num_chunks = int(
            chunk_counts[
                video_id
            ]
        )


        # Mỗi concept phải có đúng 1 score / chunk
        if len(group) != num_chunks:

            raise RuntimeError(
                "Incomplete concept evidence. "
                f"video={video_id}, "
                f"concept={concept}, "
                f"rows={len(group)}, "
                f"chunks={num_chunks}"
            )


        # =================================================
        # 8. SCORE STATISTICS
        # =================================================

        max_final = (
            group[
                "final_score"
            ]
            .max()
        )


        mean_final = (
            group[
                "final_score"
            ]
            .mean()
        )


        max_lsa = (
            group[
                "lsa_score"
            ]
            .max()
        )


        max_lda = (
            group[
                "lda_score"
            ]
            .max()
        )


        # =================================================
        # 9. RANK SUPPORT
        # =================================================

        top3_count = int(
            (
                group[
                    "rank"
                ]
                <= 3
            )
            .sum()
        )


        top1_count = int(
            (
                group[
                    "rank"
                ]
                == 1
            )
            .sum()
        )


        top3_ratio = (
            top3_count
            / num_chunks
        )


        top1_ratio = (
            top1_count
            / num_chunks
        )


        # =================================================
        # 10. TOP-2 STRONGEST EVIDENCE
        # =================================================
        #
        # OLD configuration sẽ dùng cột này.
        # =================================================

        strongest = (
            group[
                "final_score"
            ]
            .nlargest(2)
        )


        top2_mean = (
            strongest.mean()
        )


        # =================================================
        # 11. BEST CHUNK
        # =================================================
        #
        # NEW configuration sẽ dùng max_final_score,
        # còn best chunk giữ timestamp evidence.
        # =================================================

        best_index = (
            group[
                "final_score"
            ]
            .idxmax()
        )


        best_row = (
            df.loc[
                best_index
            ]
        )


        # =================================================
        # 12. SAVE VIDEO-CONCEPT ROW
        # =================================================

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
                    float(
                        max_final
                    ),
                    4,
                ),

            "mean_final_score":
                round(
                    float(
                        mean_final
                    ),
                    4,
                ),

            "top2_mean_score":
                round(
                    float(
                        top2_mean
                    ),
                    4,
                ),

            "top3_count":
                top3_count,

            "top3_ratio":
                round(
                    float(
                        top3_ratio
                    ),
                    4,
                ),

            "top1_count":
                top1_count,

            "top1_ratio":
                round(
                    float(
                        top1_ratio
                    ),
                    4,
                ),

            "max_lsa_score":
                round(
                    float(
                        max_lsa
                    ),
                    4,
                ),

            "max_lda_score":
                round(
                    float(
                        max_lda
                    ),
                    4,
                ),

            "best_chunk_id":
                int(
                    best_row[
                        "chunk_id"
                    ]
                ),

            "best_start_sec":
                float(
                    best_row[
                        "start_sec"
                    ]
                ),

            "best_end_sec":
                float(
                    best_row[
                        "end_sec"
                    ]
                ),

            "best_final_score":
                round(
                    float(
                        best_row[
                            "final_score"
                        ]
                    ),
                    4,
                ),

            "best_chunk_rank":
                int(
                    best_row[
                        "rank"
                    ]
                ),
        })


    # =====================================================
    # 13. BUILD DATAFRAME
    # =====================================================

    evidence = pd.DataFrame(
        result_rows
    )


    # =====================================================
    # 14. OUTPUT AUDIT
    # =====================================================
    #
    # 10 SQL videos    × 11 SQL concepts    = 110
    # 10 Python videos ×  9 Python concepts =  90
    # 10 Java videos   ×  9 Java concepts   =  90
    # 10 C++ videos    × 10 C++ concepts    = 100
    #
    # TOTAL = 390 video-concept rows
    # =====================================================

    expected_evidence_rows = 390


    if len(evidence) != expected_evidence_rows:

        raise RuntimeError(
            "STEP 09 evidence row count mismatch. "
            f"Expected={expected_evidence_rows}, "
            f"Found={len(evidence)}"
        )


    evidence_ids = set(
        evidence[
            "video_id"
        ]
        .astype(str)
        .str.strip()
    )


    if evidence_ids != expected_ids:

        raise RuntimeError(
            "STEP 09 evidence does not contain "
            "exactly v1-v40."
        )


    duplicate_evidence = (
        evidence
        .duplicated(
            [
                "video_id",
                "concept",
            ]
        )
        .sum()
    )


    if duplicate_evidence != 0:

        raise RuntimeError(
            "Duplicate video-concept evidence rows: "
            f"{duplicate_evidence}"
        )


    # -----------------------------------------------------
    # MAX MUST MATCH BEST FINAL SCORE
    # -----------------------------------------------------

    max_best_mismatch = (
        evidence[
            "max_final_score"
        ]
        != evidence[
            "best_final_score"
        ]
    ).sum()


    if max_best_mismatch != 0:

        raise RuntimeError(
            "max_final_score != best_final_score "
            f"for {max_best_mismatch} rows."
        )


    # -----------------------------------------------------
    # SCORE RANGE
    # -----------------------------------------------------

    for column in [
        "max_final_score",
        "mean_final_score",
        "top2_mean_score",
        "best_final_score",
    ]:

        if not (
            evidence[
                column
            ]
            .between(
                0.0,
                1.0,
            )
            .all()
        ):

            raise RuntimeError(
                f"{column} outside [0,1]."
            )


    # =====================================================
    # 15. DETERMINISTIC SORT
    # =====================================================

    evidence[
        "_video_number"
    ] = (
        evidence[
            "video_id"
        ]
        .map(
            video_number
        )
    )


    evidence = (
        evidence
        .sort_values(
            [
                "_video_number",
                "concept",
            ]
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


    # =====================================================
    # 16. SAVE
    # =====================================================

    evidence.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )


    # =====================================================
    # 17. COMPLETE
    # =====================================================

    print(
        "\n======================================"
    )

    print(
        "STEP 09 PASS"
    )

    print(
        "======================================"
    )

    print(
        "Variant:",
        VARIANT
    )

    print(
        "DEV videos:",
        evidence[
            "video_id"
        ].nunique()
    )

    print(
        "Input chunks:",
        total_chunks
    )

    print(
        "Fusion rows:",
        len(df)
    )

    print(
        "Video-concept evidence rows:",
        len(evidence)
    )

    print(
        "Expected evidence rows:",
        expected_evidence_rows
    )

    print(
        "Duplicate evidence rows:",
        duplicate_evidence
    )

    print(
        "Max/best mismatches:",
        max_best_mismatch
    )


    print(
        "\nEvidence file:"
    )

    print(
        OUTPUT_FILE
    )


    print(
        "======================================"
    )


if __name__ == "__main__":

    main()