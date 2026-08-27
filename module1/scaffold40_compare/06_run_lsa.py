from pathlib import Path
import sys

import pandas as pd


# =========================================================
# STEP 06 - LSA SEMANTIC MATCHING
# SCAFFOLD40 OLD vs NEW COMPARISON
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

sys.path.append(
    str(PROJECT_ROOT / "module1" / "src")
)


# =========================================================
# IMPORT MODEL
# =========================================================

from preprocessing import preprocess_text

from lsa_model import (
    train_lsa,
    calculate_similarity,
)


# =========================================================
# IMPORT OLD / NEW CONFIG
# =========================================================

from step00_compare_config import (
    VARIANT,
    CHUNK_FILE,
    RUN_ROOT,
)


# =========================================================
# 1. INPUT / OUTPUT
# =========================================================

# Đọc đúng chunks của variant hiện tại
#
# OLD:
# module1/results/scaffold40_compare/old/transcript_chunks.csv
#
# NEW:
# module1/results/scaffold40_compare/new/transcript_chunks.csv

INPUT_FILE = CHUNK_FILE


CONCEPT_FILE = (
    PROJECT_ROOT
    / "data"
    / "concepts"
    / "concept_catalog.csv"
)


# Output OLD / NEW tách riêng.
# KHÔNG ghi vào module1/results/baseline.
RESULT_DIR = RUN_ROOT

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# 2. INPUT AUDIT
# =========================================================

def audit_input(chunks, concepts):

    # -----------------------------------------------------
    # REQUIRED CHUNK COLUMNS
    # -----------------------------------------------------

    required_chunk_columns = {
        "video_id",
        "subject",
        "chunk_id",
        "start_sec",
        "end_sec",
        "processed_text",
    }

    missing_chunk_columns = (
        required_chunk_columns
        - set(chunks.columns)
    )

    if missing_chunk_columns:

        raise RuntimeError(
            "STEP 06 chunk input missing columns: "
            f"{sorted(missing_chunk_columns)}"
        )


    # -----------------------------------------------------
    # REQUIRED CONCEPT COLUMNS
    # -----------------------------------------------------

    required_concept_columns = {
        "subject",
        "concept",
        "description",
    }

    missing_concept_columns = (
        required_concept_columns
        - set(concepts.columns)
    )

    if missing_concept_columns:

        raise RuntimeError(
            "Concept catalog missing columns: "
            f"{sorted(missing_concept_columns)}"
        )


    # -----------------------------------------------------
    # EXACTLY v1 -> v40
    # -----------------------------------------------------

    expected_ids = {
        f"v{i}"
        for i in range(1, 41)
    }

    actual_ids = set(
        chunks["video_id"]
        .astype(str)
        .str.strip()
    )

    missing_ids = sorted(
        expected_ids - actual_ids
    )

    unexpected_ids = sorted(
        actual_ids - expected_ids
    )

    if missing_ids or unexpected_ids:

        raise RuntimeError(
            "STEP 06 input is not exactly v1-v40. "
            f"Missing={missing_ids}, "
            f"Unexpected={unexpected_ids}"
        )


    # -----------------------------------------------------
    # DUPLICATE CHUNKS
    # -----------------------------------------------------

    duplicate_chunks = (
        chunks
        .duplicated(
            [
                "video_id",
                "chunk_id",
            ]
        )
        .sum()
    )

    if duplicate_chunks != 0:

        raise RuntimeError(
            "Duplicate chunks found: "
            f"{duplicate_chunks}"
        )


    # -----------------------------------------------------
    # EXACT 39 CONCEPTS
    # -----------------------------------------------------

    if len(concepts) != 39:

        raise RuntimeError(
            "Expected exactly 39 concepts. "
            f"Found: {len(concepts)}"
        )


    duplicate_concepts = (
        concepts
        .duplicated(
            [
                "subject",
                "concept",
            ]
        )
        .sum()
    )

    if duplicate_concepts != 0:

        raise RuntimeError(
            "Duplicate concept rows found: "
            f"{duplicate_concepts}"
        )


    # -----------------------------------------------------
    # 10 VIDEOS / SUBJECT
    # -----------------------------------------------------

    video_subject = (
        chunks[
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
            "Unexpected video subject distribution. "
            f"Found: {subject_counts}"
        )


    # -----------------------------------------------------
    # CONCEPT DISTRIBUTION
    # -----------------------------------------------------

    concept_counts = (
        concepts["subject"]
        .astype(str)
        .str.strip()
        .value_counts()
        .to_dict()
    )

    expected_concept_counts = {
        "SQL": 11,
        "Python": 9,
        "Java": 9,
        "C++": 10,
    }

    if concept_counts != expected_concept_counts:

        raise RuntimeError(
            "Unexpected concept distribution. "
            f"Found: {concept_counts}"
        )


    print(
        "\n======================================"
    )

    print(
        "STEP 06 INPUT AUDIT"
    )

    print(
        "======================================"
    )

    print(
        "Variant:",
        VARIANT
    )

    print(
        "Input videos:",
        chunks["video_id"].nunique()
    )

    print(
        "Input chunks:",
        len(chunks)
    )

    print(
        "Concepts:",
        len(concepts)
    )

    print(
        "Video subject counts:",
        subject_counts
    )

    print(
        "Concept subject counts:",
        concept_counts
    )

    print(
        "Duplicate chunks:",
        duplicate_chunks
    )

    print(
        "Duplicate concepts:",
        duplicate_concepts
    )

    print(
        "Input:"
    )

    print(
        INPUT_FILE
    )

    print(
        "Output directory:"
    )

    print(
        RESULT_DIR
    )

    print(
        "======================================"
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
        "STEP 06 - LSA SEMANTIC MATCHING"
    )

    print(
        "======================================"
    )


    # =====================================================
    # 4. CHECK INPUT FILES
    # =====================================================

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            "STEP 06 cannot find chunk file: "
            f"{INPUT_FILE}"
        )


    if not CONCEPT_FILE.exists():

        raise FileNotFoundError(
            "STEP 06 cannot find concept catalog: "
            f"{CONCEPT_FILE}"
        )


    # =====================================================
    # 5. READ DATA
    # =====================================================

    chunks = pd.read_csv(
        INPUT_FILE
    )

    concepts = pd.read_csv(
        CONCEPT_FILE
    )


    audit_input(
        chunks,
        concepts,
    )


    # -----------------------------------------------------
    # PREPARE CHUNK TEXT
    # -----------------------------------------------------

    chunks["processed_text"] = (
        chunks["processed_text"]
        .fillna("")
        .astype(str)
    )


    empty_chunks = (
        chunks["processed_text"]
        .str.strip()
        .eq("")
        .sum()
    )

    if empty_chunks != 0:

        raise RuntimeError(
            "STEP 06 found empty processed_text chunks: "
            f"{empty_chunks}"
        )


    document_texts = (
        chunks["processed_text"]
        .tolist()
    )


    # -----------------------------------------------------
    # PREPARE CONCEPT TEXT
    #
    # Giữ logic semantic representation:
    # concept name + semantic description
    # rồi preprocess bằng cùng preprocessing.
    # -----------------------------------------------------

    concepts["concept"] = (
        concepts["concept"]
        .fillna("")
        .astype(str)
    )

    concepts["description"] = (
        concepts["description"]
        .fillna("")
        .astype(str)
    )


    concept_texts = []

    for _, concept in concepts.iterrows():

        concept_text = (
            str(concept["concept"])
            + " "
            + str(concept["description"])
        )

        concept_text = (
            preprocess_text(
                concept_text
            )
        )

        concept_texts.append(
            concept_text
        )


    if any(
        not text.strip()
        for text in concept_texts
    ):

        raise RuntimeError(
            "STEP 06 produced an empty concept text."
        )


    print(
        "\nDocuments/chunks:",
        len(document_texts)
    )

    print(
        "Concept texts:",
        len(concept_texts)
    )


    # =====================================================
    # 6. TRAIN LSA
    # =====================================================
    #
    # GIỮ NGUYÊN train_lsa() của scaffold cũ.
    #
    # Không đổi:
    # - TF-IDF configuration
    # - max features
    # - SVD logic
    # - number of latent dimensions
    #
    # =====================================================

    (
        vectorizer,
        svd,
        document_vectors,
        concept_vectors,
    ) = train_lsa(
        document_texts,
        concept_texts,
    )


    print(
        "Vocabulary size:",
        len(
            vectorizer
            .get_feature_names_out()
        )
    )

    print(
        "LSA dimensions:",
        svd.n_components
    )


    # =====================================================
    # 7. CALCULATE SEMANTIC SIMILARITY
    # =====================================================

    similarity_matrix = (
        calculate_similarity(
            document_vectors,
            concept_vectors,
        )
    )


    expected_shape = (
        len(chunks),
        len(concepts),
    )

    if similarity_matrix.shape != expected_shape:

        raise RuntimeError(
            "Unexpected LSA similarity matrix shape. "
            f"Expected={expected_shape}, "
            f"Found={similarity_matrix.shape}"
        )


    # =====================================================
    # 8. ONLY COMPARE CONCEPTS OF SAME SUBJECT
    # =====================================================

    result_rows = []


    for chunk_index, chunk in (
        chunks.iterrows()
    ):

        chunk_subject = (
            str(
                chunk["subject"]
            )
            .strip()
        )


        for concept_index, concept in (
            concepts.iterrows()
        ):

            concept_subject = (
                str(
                    concept["subject"]
                )
                .strip()
            )


            # SQL chunk chỉ so SQL concepts,
            # Python chỉ so Python,...
            if (
                concept_subject
                != chunk_subject
            ):

                continue


            score = (
                similarity_matrix[
                    chunk_index,
                    concept_index,
                ]
            )


            result_rows.append({
                "video_id":
                    chunk["video_id"],

                "subject":
                    chunk_subject,

                "method":
                    "transcript_only",

                "chunk_id":
                    chunk["chunk_id"],

                "start_sec":
                    chunk["start_sec"],

                "end_sec":
                    chunk["end_sec"],

                "concept":
                    concept["concept"],

                "lsa_score":
                    round(
                        float(score),
                        4,
                    ),
            })


    scores_df = pd.DataFrame(
        result_rows
    )


    # =====================================================
    # 9. OUTPUT AUDIT
    # =====================================================

    # Expected number of rows =
    #
    # mỗi chunk × số concept thuộc subject của chunk

    chunk_counts = (
        chunks
        .groupby("subject")
        .size()
        .to_dict()
    )

    concept_counts = (
        concepts
        .groupby("subject")
        .size()
        .to_dict()
    )


    expected_score_rows = sum(
        chunk_counts.get(
            subject,
            0,
        )
        *
        concept_counts.get(
            subject,
            0,
        )
        for subject in concept_counts
    )


    if len(scores_df) != expected_score_rows:

        raise RuntimeError(
            "STEP 06 LSA score row count mismatch. "
            f"Expected={expected_score_rows}, "
            f"Found={len(scores_df)}"
        )


    output_ids = set(
        scores_df["video_id"]
        .astype(str)
        .str.strip()
    )

    expected_ids = {
        f"v{i}"
        for i in range(1, 41)
    }

    if output_ids != expected_ids:

        raise RuntimeError(
            "STEP 06 output does not contain "
            "exactly v1-v40."
        )


    duplicate_scores = (
        scores_df
        .duplicated(
            [
                "video_id",
                "chunk_id",
                "concept",
            ]
        )
        .sum()
    )

    if duplicate_scores != 0:

        raise RuntimeError(
            "Duplicate LSA score rows found: "
            f"{duplicate_scores}"
        )


    # =====================================================
    # 10. SAVE ALL LSA SCORES
    # =====================================================

    scores_file = (
        RESULT_DIR
        / "lsa_concept_scores.csv"
    )


    scores_df.to_csv(
        scores_file,
        index=False,
        encoding="utf-8-sig",
    )


    # =====================================================
    # 11. SAVE TOP 3 CONCEPTS PER CHUNK
    # =====================================================

    top3_df = (
        scores_df
        .sort_values(
            [
                "video_id",
                "chunk_id",
                "lsa_score",
            ],
            ascending=[
                True,
                True,
                False,
            ],
        )
        .groupby(
            [
                "video_id",
                "chunk_id",
            ],
            as_index=False,
            group_keys=False,
        )
        .head(3)
        .copy()
    )


    top3_file = (
        RESULT_DIR
        / "lsa_top_concepts.csv"
    )


    top3_df.to_csv(
        top3_file,
        index=False,
        encoding="utf-8-sig",
    )


    # =====================================================
    # 12. COMPLETE
    # =====================================================

    print(
        "\n======================================"
    )

    print(
        "STEP 06 PASS"
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
        scores_df[
            "video_id"
        ].nunique()
    )

    print(
        "Input chunks:",
        len(chunks)
    )

    print(
        "Concepts:",
        len(concepts)
    )

    print(
        "LSA score rows:",
        len(scores_df)
    )

    print(
        "Expected score rows:",
        expected_score_rows
    )

    print(
        "Duplicate score rows:",
        duplicate_scores
    )

    print(
        "Vocabulary size:",
        len(
            vectorizer
            .get_feature_names_out()
        )
    )

    print(
        "LSA dimensions:",
        svd.n_components
    )


    print(
        "\nLSA score file:"
    )

    print(
        scores_file
    )


    print(
        "\nTop concepts file:"
    )

    print(
        top3_file
    )


    print(
        "======================================"
    )


if __name__ == "__main__":

    main()