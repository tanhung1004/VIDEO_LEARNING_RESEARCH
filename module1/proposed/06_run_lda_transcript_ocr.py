from pathlib import Path
import sys

import numpy as np
import pandas as pd


# =========================================================
# 1. ĐƯỜNG DẪN PROJECT
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

sys.path.append(
    str(PROJECT_ROOT / "module1" / "src")
)

from lda_model import (
    train_lda,
    get_topic_words
)


INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "transcript_ocr"
    / "transcript_ocr_chunks.csv"
)

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


# =========================================================
# 2. MAIN
# =========================================================

def main():

    print("======================================")
    print("MODULE 1 - BASELINE")
    print("STEP 05 - LDA TOPIC MODELLING")
    print("======================================")

    if not INPUT_FILE.exists():

        print("Không tìm thấy:")
        print(INPUT_FILE)

        return

    # -----------------------------------------------------
    # 3. ĐỌC 51 CHUNK
    # -----------------------------------------------------

    df = pd.read_csv(INPUT_FILE)

    df["processed_text"] = (
        df["processed_text"]
        .fillna("")
        .astype(str)
    )

    documents = (
        df["processed_text"].tolist()
    )

    print("Số document/chunk:", len(documents))

    # -----------------------------------------------------
    # 4. TRAIN LDA
    # -----------------------------------------------------

    (
        vectorizer,
        lda_model,
        topic_matrix
    ) = train_lda(documents)

    print(
        "Vocabulary size:",
        len(vectorizer.get_feature_names_out())
    )

    print(
        "Số topic:",
        lda_model.n_components
    )

    # -----------------------------------------------------
    # 5. LẤY TOP WORDS CỦA TỪNG TOPIC
    # -----------------------------------------------------

    topics = get_topic_words(
        lda_model,
        vectorizer
    )

    topics_df = pd.DataFrame(topics)

    topics_file = (
        RESULT_DIR
        / "lda_topics.csv"
    )

    topics_df.to_csv(
        topics_file,
        index=False,
        encoding="utf-8-sig"
    )

    # In topic ra Terminal
    print("\n========== LDA TOPICS ==========")

    for _, row in topics_df.iterrows():

        print(
            f"\nTopic {row['topic_id']}:"
        )

        print(row["top_words"])

    # -----------------------------------------------------
    # 6. TOPIC CỦA TỪNG CHUNK
    # -----------------------------------------------------

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

    # Lưu probability từng topic
    for topic_id in range(
        lda_model.n_components
    ):

        df[f"topic_{topic_id}_score"] = (
            topic_matrix[:, topic_id]
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

    # -----------------------------------------------------
    # 7. HOÀN THÀNH
    # -----------------------------------------------------

    print("\n======================================")
    print("HOÀN THÀNH STEP 06")

    print("\nTopic file:")
    print(topics_file)

    print("\nChunk topic file:")
    print(chunk_file)

    print("======================================")


if __name__ == "__main__":
    main()