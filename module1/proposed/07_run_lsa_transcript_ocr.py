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


# =========================================================
# 2. INPUT
# =========================================================
#
# Khác baseline:
#
# baseline:
# transcript_chunks.csv
#
# proposed:
# transcript_ocr_chunks.csv
#
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


# =========================================================
# 4. MAIN
# =========================================================

def main():

    print("======================================")
    print("MODULE 1 - PROPOSED")
    print("STEP 07 - LSA SEMANTIC MATCHING")
    print("INPUT: TRANSCRIPT + OCR")
    print("======================================")


    # -----------------------------------------------------
    # 5. ĐỌC DATA
    # -----------------------------------------------------

    chunks = pd.read_csv(
        CHUNK_FILE
    )

    concepts = pd.read_csv(
        CONCEPT_FILE
    )


    # -----------------------------------------------------
    # 6. CHUẨN HÓA TEXT
    # -----------------------------------------------------

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
    # 7. PREPROCESS CONCEPT TEXT
    #
    # Giữ y chang baseline:
    #
    # concept + description
    #       ↓
    # preprocess_text()
    #
    # -----------------------------------------------------

    concepts["concept_text"] = (
        concepts["concept"]
        + " "
        + concepts["description"]
    )

    concepts["processed_concept_text"] = (
        concepts["concept_text"]
        .apply(
            preprocess_text
        )
    )


    # -----------------------------------------------------
    # 8. DOCUMENT TEXT
    #
    # processed_text ở đây đã là:
    #
    # Transcript + OCR
    #       ↓
    # baseline preprocessing
    #
    # -----------------------------------------------------

    document_texts = (
        chunks["processed_text"]
        .tolist()
    )

    concept_texts = (
        concepts[
            "processed_concept_text"
        ]
        .tolist()
    )


    print(
        "Số Transcript+OCR chunks:",
        len(document_texts)
    )

    print(
        "Số concepts:",
        len(concept_texts)
    )


    # -----------------------------------------------------
    # 9. TRAIN LSA
    #
    # Sử dụng đúng train_lsa()
    # từ module1/src/lsa_model.py
    #
    # Không thay thuật toán baseline.
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
    # 10. CALCULATE SEMANTIC SIMILARITY
    #
    # chunk vector
    #      vs
    # concept vector
    #
    # -----------------------------------------------------

    similarity_matrix = (
        calculate_similarity(
            document_vectors,
            concept_vectors
        )
    )


    # -----------------------------------------------------
    # 11. BUILD RESULT
    # -----------------------------------------------------

    result_rows = []


    # -----------------------------------------------------
    # 12. CHỈ SO CONCEPT CÙNG SUBJECT
    #
    # SQL    -> SQL concepts
    # Python -> Python concepts
    # Java   -> Java concepts
    #
    # Giữ đúng baseline.
    # -----------------------------------------------------

    for chunk_index, chunk in (
        chunks.iterrows()
    ):

        subject = str(
            chunk["subject"]
        ).strip()


        for concept_index, concept in (
            concepts.iterrows()
        ):

            concept_subject = str(
                concept["subject"]
            ).strip()


            if (
                concept_subject
                != subject
            ):
                continue


            score = similarity_matrix[
                chunk_index,
                concept_index
            ]


            result_rows.append({

                "video_id":
                    chunk["video_id"],

                "subject":
                    subject,

                # -----------------------------------------
                # Đây là proposed
                # -----------------------------------------

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

                "lsa_score":
                    round(
                        float(score),
                        4
                    )
            })


    # -----------------------------------------------------
    # 13. DATAFRAME
    # -----------------------------------------------------

    scores_df = pd.DataFrame(
        result_rows
    )


    # -----------------------------------------------------
    # 14. SAVE ALL LSA SCORES
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
    # 15. TOP 3 CONCEPT MỖI CHUNK
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

        .reset_index(
            drop=True
        )
    )


    # -----------------------------------------------------
    # 16. SAVE TOP MATCHES
    # -----------------------------------------------------

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
    # 17. PRINT PREVIEW
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

        .to_string(
            index=False
        )
    )


    # -----------------------------------------------------
    # 18. SUMMARY
    # -----------------------------------------------------

    print(
        "\n======================================"
    )

    print(
        "HOÀN THÀNH STEP 07"
    )

    print(
        "\nAll LSA scores:"
    )

    print(
        scores_file
    )

    print(
        "\nTop LSA matches:"
    )

    print(
        top_file
    )

    print(
        "======================================"
    )


if __name__ == "__main__":
    main()