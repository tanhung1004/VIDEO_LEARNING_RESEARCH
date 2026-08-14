from pathlib import Path
import sys

import pandas as pd


# =========================================================
# 1. PROJECT PATH
# =========================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

# Cho phép baseline import code trong module1/src
sys.path.append(
    str(
        PROJECT_ROOT
        / "module1"
        / "src"
    )
)


from preprocessing import preprocess_text

from lda_model import (
    train_lda,
    get_topic_words,
    calculate_topic_concept_affinity,
    calculate_document_concept_scores
)


# =========================================================
# 2. INPUT / OUTPUT PATH
# =========================================================

CHUNK_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "transcript"
    / "transcript_chunks.csv"
)

CONCEPT_FILE = (
    PROJECT_ROOT
    / "data"
    / "concepts"
    / "concept_catalog.csv"
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
# 3. MAIN
# =========================================================

def main():

    print(
        "======================================"
    )

    print(
        "MODULE 1 - BASELINE"
    )

    print(
        "STEP 07 - LDA CONCEPT MATCHING"
    )

    print(
        "======================================"
    )


    # =====================================================
    # 4. CHECK INPUT
    # =====================================================

    if not CHUNK_FILE.exists():

        print(
            "Không tìm thấy chunk file:"
        )

        print(
            CHUNK_FILE
        )

        return


    if not CONCEPT_FILE.exists():

        print(
            "Không tìm thấy concept catalog:"
        )

        print(
            CONCEPT_FILE
        )

        return


    # =====================================================
    # 5. READ DATA
    # =====================================================

    chunks = pd.read_csv(
        CHUNK_FILE
    )

    concepts = pd.read_csv(
        CONCEPT_FILE
    )


    # -----------------------------------------------------
    # Clean transcript column
    # -----------------------------------------------------

    chunks["processed_text"] = (
        chunks["processed_text"]
        .fillna("")
        .astype(str)
    )


    # -----------------------------------------------------
    # Clean concept columns
    # -----------------------------------------------------

    concepts["subject"] = (
        concepts["subject"]
        .fillna("")
        .astype(str)
    )

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


    # =====================================================
    # 6. PREPARE CONCEPT TEXT
    # =====================================================

    # Ví dụ:
    #
    # concept:
    # group by
    #
    # description:
    # group database rows by one or more columns
    #
    # concept_text:
    # group by group database rows...
    #

    concepts["concept_text"] = (
        concepts["concept"]
        + " "
        + concepts["description"]
    )


    # Dùng cùng preprocessing với transcript
    concepts[
        "processed_concept_text"
    ] = (
        concepts["concept_text"]
        .apply(
            preprocess_text
        )
    )


    document_texts = (
        chunks[
            "processed_text"
        ]
        .tolist()
    )


    concept_texts = (
        concepts[
            "processed_concept_text"
        ]
        .tolist()
    )


    print(
        "Transcript chunks:",
        len(document_texts)
    )

    print(
        "Concepts:",
        len(concept_texts)
    )


    # =====================================================
    # 7. TRAIN LDA
    # =====================================================

    (
        vectorizer,
        lda_model,
        document_topic_matrix
    ) = train_lda(
        document_texts
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


    # =====================================================
    # 8. SAVE LDA TOPICS
    # =====================================================

    topics = get_topic_words(
        lda_model,
        vectorizer
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
        encoding="utf-8-sig"
    )


    # =====================================================
    # 9. TOPIC <-> CONCEPT AFFINITY
    # =====================================================

    topic_concept_affinity = (
        calculate_topic_concept_affinity(
            lda_model=lda_model,
            vectorizer=vectorizer,
            concept_texts=concept_texts
        )
    )


    # -----------------------------------------------------
    # Save affinity để kiểm tra
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
                                concept_index
                            ]
                        ),
                        4
                    )
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
        encoding="utf-8-sig"
    )


    # =====================================================
    # 10. CHUNK -> CONCEPT SCORE
    # =====================================================

    lda_concept_matrix = (
        calculate_document_concept_scores(
            document_topic_matrix,
            topic_concept_affinity
        )
    )


    # =====================================================
    # 11. BUILD RESULT
    # =====================================================

    result_rows = []


    for chunk_index, chunk in (
        chunks.iterrows()
    ):

        chunk_subject = str(
            chunk["subject"]
        )


        for concept_index, concept in (
            concepts.iterrows()
        ):

            concept_subject = str(
                concept["subject"]
            )


            # =============================================
            # Chỉ compare concept cùng subject
            # =============================================

            if (
                chunk_subject
                != concept_subject
            ):
                continue


            score = (
                lda_concept_matrix[
                    chunk_index,
                    concept_index
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
                        4
                    )

            })


    scores_df = pd.DataFrame(
        result_rows
    )


    # =====================================================
    # 12. SAVE ALL LDA CONCEPT SCORES
    # =====================================================

    score_file = (
        RESULT_DIR
        / "lda_concept_scores.csv"
    )


    scores_df.to_csv(
        score_file,
        index=False,
        encoding="utf-8-sig"
    )


    # =====================================================
    # 13. TOP 3 CONCEPT CỦA MỖI CHUNK
    # =====================================================

    top_matches = (
        scores_df
        .sort_values(
            [
                "video_id",
                "chunk_id",
                "lda_score"
            ],

            ascending=[
                True,
                True,
                False
            ]
        )

        .groupby(
            [
                "video_id",
                "chunk_id"
            ]
        )

        .head(3)

        .reset_index(
            drop=True
        )
    )


    top_file = (
        RESULT_DIR
        / "lda_top_concepts.csv"
    )


    top_matches.to_csv(
        top_file,
        index=False,
        encoding="utf-8-sig"
    )


    # =====================================================
    # 14. TERMINAL PREVIEW
    # =====================================================

    print(
        "\n========== LDA TOPICS =========="
    )


    for _, row in (
        topics_df.iterrows()
    ):

        print(
            f"\nTopic {row['topic_id']}:"
        )

        print(
            row["top_words"]
        )


    print(
        "\n========== TOP LDA CONCEPTS =========="
    )


    print(
        top_matches[
            [
                "video_id",
                "chunk_id",
                "concept",
                "lda_score"
            ]
        ]
        .head(40)
        .to_string(
            index=False
        )
    )


    # =====================================================
    # 15. SUMMARY
    # =====================================================

    print(
        "\n======================================"
    )

    print(
        "HOÀN THÀNH STEP 07"
    )


    print(
        "\nLDA topics:"
    )

    print(
        topics_file
    )


    print(
        "\nTopic-concept affinity:"
    )

    print(
        affinity_file
    )


    print(
        "\nAll concept scores:"
    )

    print(
        score_file
    )


    print(
        "\nTop concepts:"
    )

    print(
        top_file
    )


    print(
        "======================================"
    )


if __name__ == "__main__":
    main()