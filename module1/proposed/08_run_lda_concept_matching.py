from pathlib import Path
import sys

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# 1. PROJECT PATH
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SRC_DIR = (
    PROJECT_ROOT
    / "module1"
    / "src"
)

sys.path.append(
    str(SRC_DIR)
)

from preprocessing import preprocess_text


# =========================================================
# 2. INPUT
# =========================================================

CHUNK_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "transcript_ocr"
    / "transcript_ocr_chunks.csv"
)

CONCEPT_FILE = (
    PROJECT_ROOT
    / "data"
    / "concepts"
    / "concept_catalog.csv"
)


# =========================================================
# 3. OUTPUT
# =========================================================

RESULT_DIR = (
    PROJECT_ROOT
    / "module1"
    / "results"
    / "proposed"
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

TOPIC_FILE = (
    RESULT_DIR
    / "lda_topics_step8.csv"
)

AFFINITY_FILE = (
    RESULT_DIR
    / "lda_topic_concept_affinity.csv"
)

SCORE_FILE = (
    RESULT_DIR
    / "lda_concept_scores.csv"
)

TOP_CONCEPT_FILE = (
    RESULT_DIR
    / "lda_top_concepts.csv"
)


# =========================================================
# 4. SAME LDA CONFIG AS BASELINE
# =========================================================

NUM_TOPICS = 6

TOP_WORDS = 12

RANDOM_STATE = 42


# =========================================================
# 5. TRAIN LDA
# =========================================================

def train_lda(document_texts):

    vectorizer = CountVectorizer(

        ngram_range=(1, 2),

        max_df=0.95,

        max_features=3000
    )

    document_term_matrix = (
        vectorizer.fit_transform(
            document_texts
        )
    )


    lda_model = LatentDirichletAllocation(

        n_components=NUM_TOPICS,

        max_iter=50,

        learning_method="batch",

        random_state=RANDOM_STATE
    )


    document_topic_matrix = (
        lda_model.fit_transform(
            document_term_matrix
        )
    )


    return (
        vectorizer,
        lda_model,
        document_topic_matrix
    )


# =========================================================
# 6. GET TOP WORDS PER TOPIC
# =========================================================

def build_topic_table(
    lda_model,
    vectorizer
):

    feature_names = (
        vectorizer
        .get_feature_names_out()
    )

    rows = []


    for topic_id, topic_weights in enumerate(
        lda_model.components_
    ):

        top_indices = (
            topic_weights
            .argsort()[::-1][
                :TOP_WORDS
            ]
        )


        top_words = [

            feature_names[index]

            for index in top_indices
        ]


        rows.append({

            "topic_id":
                topic_id,

            "top_words":
                ", ".join(
                    top_words
                )
        })


    return pd.DataFrame(
        rows
    )


# =========================================================
# 7. TOPIC-WORD DISTRIBUTION
# =========================================================
#
# lda.components_ chứa trọng số word trong từng topic.
#
# Normalize mỗi topic thành probability distribution.
# =========================================================

def get_topic_word_distribution(
    lda_model
):

    topic_word = (
        lda_model
        .components_
        .astype(float)
    )


    row_sum = (
        topic_word
        .sum(
            axis=1,
            keepdims=True
        )
    )


    row_sum[
        row_sum == 0
    ] = 1.0


    topic_word_distribution = (
        topic_word
        / row_sum
    )


    return (
        topic_word_distribution
    )


# =========================================================
# 8. BUILD CONCEPT VECTORS
# =========================================================
#
# Rất quan trọng:
#
# Concepts phải nằm trong CÙNG vocabulary
# với LDA.
#
# Không fit vectorizer mới.
# =========================================================

def build_concept_vectors(
    concepts,
    vectorizer
):

    concepts = concepts.copy()


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


    concepts["concept_text"] = (

        concepts["concept"]

        + " "

        + concepts["description"]
    )


    concepts["processed_concept_text"] = (

        concepts[
            "concept_text"
        ]

        .apply(
            preprocess_text
        )
    )


    concept_vectors = (
        vectorizer
        .transform(
            concepts[
                "processed_concept_text"
            ]
        )
    )


    return (
        concepts,
        concept_vectors
    )


# =========================================================
# 9. TOPIC → CONCEPT AFFINITY
# =========================================================
#
# topic-word vector
#       vs
# concept vector
#
# cosine similarity
#
# Đây là bước sửa so với cách LDA matching cũ.
# =========================================================

def calculate_topic_concept_affinity(
    topic_word_distribution,
    concept_vectors
):

    affinity = cosine_similarity(

        topic_word_distribution,

        concept_vectors
    )


    return affinity


# =========================================================
# 10. DOCUMENT → CONCEPT SCORE
# =========================================================
#
# Document-topic:
#
#    D x T
#
# Topic-concept:
#
#    T x C
#
# =>
#
# Document-concept:
#
#    D x C
#
# =========================================================

def calculate_document_concept_scores(
    document_topic_matrix,
    topic_concept_affinity
):

    return np.matmul(

        document_topic_matrix,

        topic_concept_affinity
    )


# =========================================================
# 11. MAIN
# =========================================================

def main():

    print(
        "======================================"
    )

    print(
        "MODULE 1 - PROPOSED"
    )

    print(
        "STEP 08 - LDA CONCEPT MATCHING"
    )

    print(
        "INPUT: TRANSCRIPT + OCR"
    )

    print(
        "======================================"
    )


    # =====================================================
    # 12. READ DATA
    # =====================================================

    chunks = pd.read_csv(
        CHUNK_FILE
    )

    concepts = pd.read_csv(
        CONCEPT_FILE
    )


    chunks["processed_text"] = (

        chunks[
            "processed_text"
        ]

        .fillna("")

        .astype(str)
    )


    concepts["subject"] = (

        concepts[
            "subject"
        ]

        .fillna("")

        .astype(str)

        .str.strip()
    )


    print(
        "Số Transcript+OCR chunks:",
        len(chunks)
    )

    print(
        "Số concepts:",
        len(concepts)
    )


    # =====================================================
    # 13. TRAIN LDA
    # =====================================================

    document_texts = (

        chunks[
            "processed_text"
        ]

        .tolist()
    )


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
        "Số topics:",
        NUM_TOPICS
    )


    # =====================================================
    # 14. SAVE TOPICS
    # =====================================================

    topic_df = build_topic_table(

        lda_model,

        vectorizer
    )


    topic_df.to_csv(

        TOPIC_FILE,

        index=False,

        encoding="utf-8-sig"
    )


    # =====================================================
    # 15. TOPIC WORD DISTRIBUTION
    # =====================================================

    topic_word_distribution = (
        get_topic_word_distribution(
            lda_model
        )
    )


    # =====================================================
    # 16. CONCEPT VECTORS
    # =====================================================

    (
        concepts,
        concept_vectors
    ) = build_concept_vectors(

        concepts,

        vectorizer
    )


    # =====================================================
    # 17. TOPIC-CONCEPT AFFINITY
    # =====================================================

    affinity_matrix = (
        calculate_topic_concept_affinity(

            topic_word_distribution,

            concept_vectors
        )
    )


    # =====================================================
    # 18. SAVE TOPIC-CONCEPT AFFINITY
    # =====================================================

    affinity_rows = []


    for topic_id in range(
        NUM_TOPICS
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

                "affinity_score":
                    round(
                        float(
                            affinity_matrix[
                                topic_id,
                                concept_index
                            ]
                        ),
                        6
                    )
            })


    affinity_df = pd.DataFrame(
        affinity_rows
    )


    affinity_df.to_csv(

        AFFINITY_FILE,

        index=False,

        encoding="utf-8-sig"
    )


    # =====================================================
    # 19. DOCUMENT-CONCEPT SCORES
    # =====================================================

    document_concept_scores = (
        calculate_document_concept_scores(

            document_topic_matrix,

            affinity_matrix
        )
    )


    # =====================================================
    # 20. BUILD CHUNK-CONCEPT RESULTS
    #
    # Chỉ giữ concept cùng subject.
    # =====================================================

    score_rows = []


    for chunk_index, chunk in (
        chunks.iterrows()
    ):

        chunk_subject = str(
            chunk["subject"]
        ).strip()


        for concept_index, concept in (
            concepts.iterrows()
        ):

            concept_subject = str(
                concept["subject"]
            ).strip()


            # ---------------------------------------------
            # SQL chỉ so SQL
            # Python chỉ so Python
            # Java chỉ so Java
            # ---------------------------------------------

            if (
                concept_subject
                != chunk_subject
            ):
                continue


            score = (
                document_concept_scores[
                    chunk_index,
                    concept_index
                ]
            )


            score_rows.append({

                "video_id":
                    chunk["video_id"],

                "subject":
                    chunk_subject,

                "method":
                    "transcript_ocr",

                "chunk_id":
                    chunk["chunk_id"],

                "start_sec":
                    chunk["start_sec"],

                "end_sec":
                    chunk["end_sec"],

                "concept":
                    concept["concept"],

                "lda_score":
                    round(
                        float(score),
                        6
                    )
            })


    scores_df = pd.DataFrame(
        score_rows
    )


    # =====================================================
    # 21. SAVE ALL LDA CONCEPT SCORES
    # =====================================================

    scores_df.to_csv(

        SCORE_FILE,

        index=False,

        encoding="utf-8-sig"
    )


    # =====================================================
    # 22. TOP 3 CONCEPTS PER CHUNK
    # =====================================================

    top_concepts = (

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


    top_concepts.to_csv(

        TOP_CONCEPT_FILE,

        index=False,

        encoding="utf-8-sig"
    )


    # =====================================================
    # 23. PREVIEW
    # =====================================================

    print(
        "\n========== LDA TOPICS =========="
    )

    print(
        topic_df.to_string(
            index=False
        )
    )


    print(
        "\n========== TOP LDA CONCEPTS =========="
    )

    print(

        top_concepts[
            [
                "video_id",
                "chunk_id",
                "concept",
                "lda_score"
            ]
        ]

        .head(20)

        .to_string(
            index=False
        )
    )


    # =====================================================
    # 24. SCORE RANGE CHECK
    # =====================================================

    if not scores_df.empty:

        print(
            "\nLDA score min:",
            round(
                scores_df[
                    "lda_score"
                ].min(),
                6
            )
        )

        print(
            "LDA score max:",
            round(
                scores_df[
                    "lda_score"
                ].max(),
                6
            )
        )

        print(
            "LDA score mean:",
            round(
                scores_df[
                    "lda_score"
                ].mean(),
                6
            )
        )


    # =====================================================
    # 25. DONE
    # =====================================================

    print(
        "\n======================================"
    )

    print(
        "HOÀN THÀNH STEP 08"
    )

    print(
        "\nTopics:"
    )

    print(
        TOPIC_FILE
    )

    print(
        "\nTopic-concept affinity:"
    )

    print(
        AFFINITY_FILE
    )

    print(
        "\nLDA concept scores:"
    )

    print(
        SCORE_FILE
    )

    print(
        "\nTop LDA concepts:"
    )

    print(
        TOP_CONCEPT_FILE
    )

    print(
        "======================================"
    )


if __name__ == "__main__":
    main()