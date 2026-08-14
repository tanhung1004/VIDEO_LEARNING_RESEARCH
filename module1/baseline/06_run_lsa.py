from pathlib import Path
import sys
import pandas as pd


# =========================================================
# 1. ĐƯỜNG DẪN PROJECT
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

sys.path.append(
    str(PROJECT_ROOT / "module1" / "src")
)

from preprocessing import preprocess_text

from lsa_model import (
    train_lsa,
    calculate_similarity
)


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
# 2. MAIN
# =========================================================

def main():

    print("======================================")
    print("MODULE 1 - BASELINE")
    print("STEP 06 - LSA SEMANTIC MATCHING")
    print("======================================")

    # -----------------------------------------------------
    # 3. ĐỌC DATA
    # -----------------------------------------------------

    chunks = pd.read_csv(CHUNK_FILE)

    concepts = pd.read_csv(CONCEPT_FILE)

    chunks["processed_text"] = (
        chunks["processed_text"]
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

    # -----------------------------------------------------
    # 4. PREPROCESS CONCEPT TEXT
    # -----------------------------------------------------

    concepts["concept_text"] = (
        concepts["concept"]
        + " "
        + concepts["description"]
    )

    concepts["processed_concept_text"] = (
        concepts["concept_text"]
        .apply(preprocess_text)
    )

    document_texts = (
        chunks["processed_text"].tolist()
    )

    concept_texts = (
        concepts[
            "processed_concept_text"
        ].tolist()
    )

    print(
        "Số transcript chunks:",
        len(document_texts)
    )

    print(
        "Số concepts:",
        len(concept_texts)
    )

    # -----------------------------------------------------
    # 5. TRAIN LSA
    # -----------------------------------------------------

    (
        vectorizer,
        svd,
        document_vectors,
        concept_vectors
    ) = train_lsa(
        document_texts,
        concept_texts
    )

    print(
        "Vocabulary size:",
        len(
            vectorizer
            .get_feature_names_out()
        )
    )

    print(
        "Số latent dimensions:",
        svd.n_components
    )

    # -----------------------------------------------------
    # 6. CALCULATE SIMILARITY
    # -----------------------------------------------------

    similarity_matrix = (
        calculate_similarity(
            document_vectors,
            concept_vectors
        )
    )

    result_rows = []

    # -----------------------------------------------------
    # 7. CHỈ SO CONCEPT CÙNG SUBJECT
    # -----------------------------------------------------

    for chunk_index, chunk in (
        chunks.iterrows()
    ):

        subject = chunk["subject"]

        for concept_index, concept in (
            concepts.iterrows()
        ):

            # SQL chỉ so SQL,
            # Java chỉ so Java,...
            if concept["subject"] != subject:
                continue

            score = similarity_matrix[
                chunk_index,
                concept_index
            ]

            result_rows.append({
                "video_id": chunk["video_id"],
                "subject": subject,
                "method": "transcript_only",

                "chunk_id": chunk["chunk_id"],
                "start_sec": chunk["start_sec"],
                "end_sec": chunk["end_sec"],

                "concept": concept["concept"],

                "lsa_score": round(
                    float(score),
                    4
                )
            })

    scores_df = pd.DataFrame(
        result_rows
    )

    # -----------------------------------------------------
    # 8. LƯU TẤT CẢ SCORE
    # -----------------------------------------------------

    scores_file = (
        RESULT_DIR
        / "lsa_concept_scores.csv"
    )

    scores_df.to_csv(
        scores_file,
        index=False,
        encoding="utf-8-sig"
    )

    # -----------------------------------------------------
    # 9. LẤY TOP 3 CONCEPT MỖI CHUNK
    # -----------------------------------------------------

    top_matches = (
        scores_df
        .sort_values(
            [
                "video_id",
                "chunk_id",
                "lsa_score"
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
        .reset_index(drop=True)
    )

    top_file = (
        RESULT_DIR
        / "lsa_top_matches.csv"
    )

    top_matches.to_csv(
        top_file,
        index=False,
        encoding="utf-8-sig"
    )

    # -----------------------------------------------------
    # 10. PRINT PREVIEW
    # -----------------------------------------------------

    print(
        "\n========== TOP LSA MATCHES =========="
    )

    print(
        top_matches[
            [
                "video_id",
                "chunk_id",
                "concept",
                "lsa_score"
            ]
        ]
        .head(20)
        .to_string(index=False)
    )

    print("\n======================================")
    print("HOÀN THÀNH STEP 06")

    print("\nAll scores:")
    print(scores_file)

    print("\nTop matches:")
    print(top_file)

    print("======================================")


if __name__ == "__main__":
    main()