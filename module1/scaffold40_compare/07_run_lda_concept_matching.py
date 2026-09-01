from pathlib import Path
import sys

import pandas as pd


# =========================================================
# STEP 07 - LDA CONCEPT MATCHING
# SCAFFOLD40 OLD vs NEW COMPARISON
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

sys.path.append(
    str(PROJECT_ROOT / "module1" / "src")
)


# =========================================================
# IMPORT MODEL HELPERS
# =========================================================

from preprocessing import preprocess_text

from lda_model import (
    train_lda,
    get_topic_words,
    calculate_topic_concept_affinity,
    calculate_document_concept_scores,
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

# OLD:
# .../scaffold40_compare/old/transcript_chunks.csv
#
# NEW:
# .../scaffold40_compare/new/transcript_chunks.csv

INPUT_FILE = CHUNK_FILE


CONCEPT_FILE = (
    PROJECT_ROOT
    / "data"
    / "concepts"
    / "concept_catalog.csv"
)


# Không ghi vào frozen baseline.
RESULT_DIR = RUN_ROOT

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# 2. INPUT AUDIT
# =========================================================

def audit_input(chunks, concepts):

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
            "STEP 07 chunk input missing columns: "
            f"{sorted(missing_chunk_columns)}"
        )


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
            "STEP 07 concept catalog missing columns: "
            f"{sorted(missing_concept_columns)}"
        )


    # -----------------------------------------------------
    # EXACT v1 -> v40
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
            "STEP 07 input is not exactly v1-v40. "
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
    # VIDEO SUBJECT COUNTS
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
    # CONCEPT SUBJECT COUNTS
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
        "STEP 07 INPUT AUDIT"
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
        "STEP 07 - LDA CONCEPT MATCHING"
    )

    print(
        "======================================"
    )


    # =====================================================
    # 4. CHECK INPUT FILES
    # =====================================================

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            "STEP 07 cannot find chunk file: "
            f"{INPUT_FILE}"
        )


    if not CONCEPT_FILE.exists():

        raise FileNotFoundError(
            "STEP 07 cannot find concept catalog: "
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


    # =====================================================
    # 6. PREPARE TRANSCRIPT DOCUMENTS
    # =====================================================

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
            "STEP 07 found empty processed_text chunks: "
            f"{empty_chunks}"
        )


    documents = (
        chunks["processed_text"]
        .tolist()
    )


    # =====================================================
    # 7. PREPARE CONCEPT TEXTS
    # =====================================================

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

        text = (
            str(concept["concept"])
            + " "
            + str(concept["description"])
        )

        text = preprocess_text(
            text
        )

        concept_texts.append(
            text
        )


    if any(
        not text.strip()
        for text in concept_texts
    ):

        raise RuntimeError(
            "STEP 07 produced empty concept text."
        )


    print(
        "\nDocuments/chunks:",
        len(documents)
    )

    print(
        "Concept texts:",
        len(concept_texts)
    )


    # =====================================================
    # 8. TRAIN LDA
    # =====================================================
    #
    # GIỮ NGUYÊN train_lda() của scaffold cũ.
    #
    # Step 07 của scaffold tự fit LDA để tạo:
    #
    # document-topic matrix
    #       ↓
    # topic-concept affinity
    #       ↓
    # document-concept scores
    #
    # Không thay hyperparameters.
    # =====================================================

    (
        vectorizer,
        lda_model,
        document_topic_matrix,
    ) = train_lda(
        documents
    )


    print(
        "Vocabulary size:",
        len(
            vectorizer
            .get_feature_names_out()
        )
    )

    print(
        "Latent topics:",
        lda_model.n_components
    )


    if (
        document_topic_matrix.shape[0]
        != len(chunks)
    ):

        raise RuntimeError(
            "LDA document-topic matrix row count "
            "does not match chunk count."
        )


    # =====================================================
    # 9. SAVE LDA TOPICS
    # =====================================================

    topics = get_topic_words(
        lda_model,
        vectorizer,
    )

    topics_df = pd.DataFrame(
        topics
    )


    topics_file = (
        RESULT_DIR
        / "lda_topics_step7.csv"
    )


    topics_df.to_csv(
        topics_file,
        index=False,
        encoding="utf-8-sig",
    )


    # =====================================================
    # 10. TOPIC <-> CONCEPT AFFINITY
    # =====================================================

    topic_concept_affinity = (
        calculate_topic_concept_affinity(
            lda_model=lda_model,
            vectorizer=vectorizer,
            concept_texts=concept_texts,
        )
    )


    expected_affinity_shape = (
        lda_model.n_components,
        len(concepts),
    )

    if (
        topic_concept_affinity.shape
        != expected_affinity_shape
    ):

        raise RuntimeError(
            "Unexpected topic-concept affinity shape. "
            f"Expected={expected_affinity_shape}, "
            f"Found={topic_concept_affinity.shape}"
        )


    # -----------------------------------------------------
    # SAVE AFFINITY
    # -----------------------------------------------------

    affinity_rows = []


    for topic_id in range(
        lda_model.n_components
    ):

        for concept_index, concept in (
            concepts.iterrows()
        ):

            affinity_rows.append({

                "topic_id":
                    topic_id,

                "subject":
                    concept["subject"],

                "concept":
                    concept["concept"],

                "topic_concept_affinity":
                    round(
                        float(
                            topic_concept_affinity[
                                topic_id,
                                concept_index,
                            ]
                        ),
                        4,
                    ),
            })


    affinity_df = pd.DataFrame(
        affinity_rows
    )


    affinity_file = (
        RESULT_DIR
        / "lda_topic_concept_affinity.csv"
    )


    affinity_df.to_csv(
        affinity_file,
        index=False,
        encoding="utf-8-sig",
    )


    # =====================================================
    # 11. CHUNK -> CONCEPT SCORE
    # =====================================================

    lda_concept_matrix = (
        calculate_document_concept_scores(
            document_topic_matrix,
            topic_concept_affinity,
        )
    )


    expected_score_shape = (
        len(chunks),
        len(concepts),
    )

    if (
        lda_concept_matrix.shape
        != expected_score_shape
    ):

        raise RuntimeError(
            "Unexpected LDA concept matrix shape. "
            f"Expected={expected_score_shape}, "
            f"Found={lda_concept_matrix.shape}"
        )


    # =====================================================
    # 12. BUILD SAME-SUBJECT RESULTS
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


            # Chỉ compare concept cùng subject.
            if (
                chunk_subject
                != concept_subject
            ):

                continue


            score = (
                lda_concept_matrix[
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
                    int(
                        chunk["chunk_id"]
                    ),

                "start_sec":
                    chunk["start_sec"],

                "end_sec":
                    chunk["end_sec"],

                "concept":
                    concept["concept"],

                "lda_score":
                    round(
                        float(score),
                        4,
                    ),
            })


    scores_df = pd.DataFrame(
        result_rows
    )


    # =====================================================
    # 13. OUTPUT AUDIT
    # =====================================================

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
            "STEP 07 LDA score row count mismatch. "
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
            "STEP 07 output is not exactly v1-v40."
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
            "Duplicate LDA concept score rows found: "
            f"{duplicate_scores}"
        )


    # =====================================================
    # 14. SAVE ALL LDA CONCEPT SCORES
    # =====================================================

    scores_file = (
        RESULT_DIR
        / "lda_concept_scores.csv"
    )


    scores_df.to_csv(
        scores_file,
        index=False,
        encoding="utf-8-sig",
    )


    # =====================================================
    # 15. TOP 3 LDA CONCEPTS PER CHUNK
    # =====================================================

    top3_df = (
        scores_df
        .sort_values(
            [
                "video_id",
                "chunk_id",
                "lda_score",
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
        / "lda_top_concepts.csv"
    )


    top3_df.to_csv(
        top3_file,
        index=False,
        encoding="utf-8-sig",
    )


    # =====================================================
    # 16. COMPLETE
    # =====================================================

    print(
        "\n======================================"
    )

    print(
        "STEP 07 PASS"
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
        "LDA score rows:",
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
        "Latent topics:",
        lda_model.n_components
    )


    print(
        "\nTopics file:"
    )

    print(
        topics_file
    )


    print(
        "\nAffinity file:"
    )

    print(
        affinity_file
    )


    print(
        "\nLDA concept score file:"
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