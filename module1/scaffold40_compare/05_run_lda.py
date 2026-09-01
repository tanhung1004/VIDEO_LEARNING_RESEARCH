from pathlib import Path
import sys

import numpy as np
import pandas as pd


# =========================================================
# STEP 05 - LDA TOPIC MODELLING
# SCAFFOLD40 OLD vs NEW COMPARISON
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

sys.path.append(
    str(PROJECT_ROOT / "module1" / "src")
)


# =========================================================
# IMPORT MODEL
# =========================================================

from lda_model import (
    train_lda,
    get_topic_words
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

# Đọc đúng chunk do STEP 04 của variant hiện tại tạo:
#
# OLD:
# module1/results/scaffold40_compare/old/transcript_chunks.csv
#
# NEW:
# module1/results/scaffold40_compare/new/transcript_chunks.csv

INPUT_FILE = CHUNK_FILE


# Mọi output của OLD / NEW đều nằm riêng.
# Không được ghi vào module1/results/baseline.
RESULT_DIR = RUN_ROOT

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# 2. AUDIT INPUT
# =========================================================

def audit_input(df):

    required_columns = {
        "video_id",
        "subject",
        "chunk_id",
        "processed_text",
    }

    missing_columns = (
        required_columns
        - set(df.columns)
    )

    if missing_columns:

        raise RuntimeError(
            "STEP 05 input missing columns: "
            f"{sorted(missing_columns)}"
        )


    # -----------------------------------------------------
    # EXACTLY v1 -> v40
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

    missing_ids = sorted(
        expected_ids
        - actual_ids
    )

    unexpected_ids = sorted(
        actual_ids
        - expected_ids
    )

    if missing_ids or unexpected_ids:

        raise RuntimeError(
            "STEP 05 input is not exactly v1-v40. "
            f"Missing={missing_ids}, "
            f"Unexpected={unexpected_ids}"
        )


    # -----------------------------------------------------
    # NO DUPLICATE CHUNKS
    # -----------------------------------------------------

    duplicate_chunks = (
        df
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
            "Duplicate video_id/chunk_id rows found: "
            f"{duplicate_chunks}"
        )


    # -----------------------------------------------------
    # EACH SUBJECT MUST HAVE 10 VIDEOS
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
            "Unexpected subject distribution. "
            f"Found: {subject_counts}"
        )


    print(
        "\n======================================"
    )

    print(
        "STEP 05 INPUT AUDIT"
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
        df["video_id"].nunique()
    )

    print(
        "Input chunks:",
        len(df)
    )

    print(
        "Subject counts:",
        subject_counts
    )

    print(
        "Duplicate chunks:",
        duplicate_chunks
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
        "STEP 05 - LDA TOPIC MODELLING"
    )

    print(
        "======================================"
    )


    # -----------------------------------------------------
    # CHECK INPUT FILE
    # -----------------------------------------------------

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            "STEP 05 cannot find chunk file: "
            f"{INPUT_FILE}"
        )


    # -----------------------------------------------------
    # READ CHUNKS
    # -----------------------------------------------------

    df = pd.read_csv(
        INPUT_FILE
    )


    # -----------------------------------------------------
    # AUDIT EXACT 40 DEV VIDEOS
    # -----------------------------------------------------

    audit_input(
        df
    )


    # -----------------------------------------------------
    # PREPARE DOCUMENTS
    # -----------------------------------------------------

    df["processed_text"] = (
        df["processed_text"]
        .fillna("")
        .astype(str)
    )


    # Không cho empty documents đi vào model
    empty_documents = (
        df["processed_text"]
        .str.strip()
        .eq("")
        .sum()
    )

    if empty_documents != 0:

        raise RuntimeError(
            "STEP 05 found empty processed_text chunks: "
            f"{empty_documents}"
        )


    documents = (
        df["processed_text"]
        .tolist()
    )


    print(
        "\nDocuments/chunks:",
        len(documents)
    )


    # =====================================================
    # 4. TRAIN LDA
    # =====================================================
    #
    # GIỮ NGUYÊN train_lda() CỦA SƯỜN CŨ.
    #
    # Không đổi:
    # - number of topics
    # - random_state
    # - vectorizer settings
    # - max_iter
    #
    # Experiment này chỉ đổi configuration được khai báo
    # trong step00_compare_config.py.
    # =====================================================

    (
        vectorizer,
        lda_model,
        topic_matrix
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
        "Number of topics:",
        lda_model.n_components
    )


    # =====================================================
    # 5. GET TOP WORDS OF EACH TOPIC
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
        / "lda_topics.csv"
    )


    topics_df.to_csv(
        topics_file,
        index=False,
        encoding="utf-8-sig"
    )


    # -----------------------------------------------------
    # DISPLAY TOPICS
    # -----------------------------------------------------

    print(
        "\n========== LDA TOPICS =========="
    )

    for _, row in topics_df.iterrows():

        print(
            f"\nTopic {row['topic_id']}:"
        )

        print(
            row["top_words"]
        )


    # =====================================================
    # 6. DOMINANT TOPIC FOR EACH CHUNK
    # =====================================================

    if topic_matrix.shape[0] != len(df):

        raise RuntimeError(
            "LDA topic matrix row count does not "
            "match input chunk count."
        )


    df["dominant_topic"] = (
        np.argmax(
            topic_matrix,
            axis=1
        )
    )


    df["dominant_topic_score"] = (
        np.max(
            topic_matrix,
            axis=1
        )
    )


    # -----------------------------------------------------
    # Save probability of every topic
    # -----------------------------------------------------

    for topic_id in range(
        lda_model.n_components
    ):

        df[
            f"topic_{topic_id}_score"
        ] = (
            topic_matrix[
                :,
                topic_id
            ]
        )


    chunk_file = (
        RESULT_DIR
        / "lda_chunk_topics.csv"
    )


    df.to_csv(
        chunk_file,
        index=False,
        encoding="utf-8-sig"
    )


    # =====================================================
    # 7. FINAL OUTPUT AUDIT
    # =====================================================

    output_ids = set(
        df["video_id"]
        .astype(str)
        .str.strip()
    )

    expected_ids = {
        f"v{i}"
        for i in range(1, 41)
    }

    if output_ids != expected_ids:

        raise RuntimeError(
            "STEP 05 output lost DEV videos."
        )


    if len(df) != len(documents):

        raise RuntimeError(
            "STEP 05 output chunk count mismatch."
        )


    # =====================================================
    # 8. COMPLETE
    # =====================================================

    print(
        "\n======================================"
    )

    print(
        "STEP 05 PASS"
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
        "Chunks:",
        len(df)
    )

    print(
        "Vocabulary size:",
        len(
            vectorizer
            .get_feature_names_out()
        )
    )

    print(
        "Topics:",
        lda_model.n_components
    )

    print(
        "\nTopic file:"
    )

    print(
        topics_file
    )

    print(
        "\nChunk topic file:"
    )

    print(
        chunk_file
    )

    print(
        "======================================"
    )


if __name__ == "__main__":

    main()