from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

SRC_DIR = ROOT / "module1" / "src"

sys.path.insert(
    0,
    str(SRC_DIR),
)

from preprocessing import preprocess_text

from lda_model import (
    train_lda,
    calculate_topic_concept_affinity,
    calculate_document_concept_scores,
)

from lsa_model import (
    train_lsa,
    calculate_similarity,
)


# ============================================================
# PATHS
# ============================================================

CHUNKS_PATH = (
    ROOT
    / "module1"
    / "results"
    / "p1_validation"
    / "dev_chunks_60s.csv"
)

FOLDS_PATH = (
    ROOT
    / "data"
    / "splits"
    / "dev_cv_folds.csv"
)

CONCEPT_PATH = (
    ROOT
    / "data"
    / "concepts"
    / "concept_catalog.csv"
)

OUTPUT_DIR = (
    ROOT
    / "module1"
    / "results"
    / "p1_validation"
)

OOF_PATH = (
    OUTPUT_DIR
    / "oof_chunk_concept_scores.csv"
)

AUDIT_PATH = (
    OUTPUT_DIR
    / "oof_score_audit.csv"
)


EXPECTED_FOLDS = {1, 2, 3, 4, 5}


# ============================================================
# BASELINE-EXACT MIN-MAX NORMALIZATION
# ============================================================

def normalize_series(series):

    min_value = series.min()
    max_value = series.max()

    # Exact behavior of frozen baseline
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


def numeric_video_sort(video_id):

    return int(
        str(video_id)
        .strip()
        .lower()
        .replace("v", "")
    )


def main():

    print("=" * 78)
    print("P1.6 BUILD LEAKAGE-FREE OOF CONCEPT SCORES")
    print("=" * 78)

    # --------------------------------------------------------
    # 1. LOAD INPUTS
    # --------------------------------------------------------

    chunks = pd.read_csv(
        CHUNKS_PATH,
        dtype=str,
        keep_default_na=False,
    )

    folds = pd.read_csv(
        FOLDS_PATH,
        dtype=str,
        keep_default_na=False,
    )

    concepts = pd.read_csv(
        CONCEPT_PATH,
        dtype=str,
        keep_default_na=False,
    )

    chunks["video_id"] = (
        chunks["video_id"]
        .str.strip()
        .str.lower()
    )

    chunks["subject"] = (
        chunks["subject"]
        .str.strip()
    )

    folds["video_id"] = (
        folds["video_id"]
        .str.strip()
        .str.lower()
    )

    folds["subject"] = (
        folds["subject"]
        .str.strip()
    )

    folds["cv_fold"] = pd.to_numeric(
        folds["cv_fold"],
        errors="raise",
    ).astype(int)

    concepts["subject"] = (
        concepts["subject"]
        .str.strip()
    )

    concepts["concept"] = (
        concepts["concept"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    concepts["description"] = (
        concepts["description"]
        .fillna("")
        .astype(str)
    )

    # --------------------------------------------------------
    # 2. INPUT AUDIT
    # --------------------------------------------------------

    errors = []

    if chunks["video_id"].nunique() != 40:
        errors.append(
            "Chunk file must contain 40 unique videos."
        )

    if folds["video_id"].nunique() != 40:
        errors.append(
            "Fold manifest must contain 40 unique videos."
        )

    if set(folds["cv_fold"]) != EXPECTED_FOLDS:
        errors.append(
            "Fold IDs must be exactly 1..5."
        )

    if len(concepts) != 39:
        errors.append(
            f"Expected 39 concepts, found {len(concepts)}."
        )

    duplicate_chunks = int(
        chunks.duplicated(
            subset=[
                "video_id",
                "chunk_id",
            ]
        ).sum()
    )

    if duplicate_chunks != 0:
        errors.append(
            f"Duplicate chunks: {duplicate_chunks}"
        )

    if (
        set(chunks["video_id"])
        != set(folds["video_id"])
    ):
        errors.append(
            "Chunk video IDs do not match fold manifest."
        )

    if errors:

        print("\nP1.6 INPUT AUDIT: FAIL")

        for i, error in enumerate(
            errors,
            1,
        ):
            print(f"{i}. {error}")

        sys.exit(1)

    print("\nInput audit: PASS")
    print(
        "Videos  :",
        chunks["video_id"].nunique(),
    )
    print(
        "Chunks  :",
        len(chunks),
    )
    print(
        "Concepts:",
        len(concepts),
    )

    # --------------------------------------------------------
    # 3. PREPARE FIXED CONCEPT TEXT
    # --------------------------------------------------------

    concepts["concept_text"] = (
        concepts["concept"]
        + " "
        + concepts["description"]
    )

    concepts[
        "processed_concept_text"
    ] = (
        concepts[
            "concept_text"
        ]
        .apply(preprocess_text)
    )

    concept_texts = (
        concepts[
            "processed_concept_text"
        ]
        .tolist()
    )

    concept_counts = (
        concepts
        .groupby("subject")
        .size()
        .to_dict()
    )

    print(
        "\nConcepts per subject:",
        concept_counts,
    )

    # --------------------------------------------------------
    # 4. OUTER-FOLD OOF SCORING
    # --------------------------------------------------------

    all_score_rows = []
    audit_rows = []

    all_video_ids = set(
        folds["video_id"]
    )

    for fold in range(1, 6):

        print("\n" + "=" * 78)
        print(f"OUTER FOLD {fold}")
        print("=" * 78)

        heldout_video_ids = set(
            folds.loc[
                folds["cv_fold"].eq(fold),
                "video_id",
            ]
        )

        train_video_ids = (
            all_video_ids
            - heldout_video_ids
        )

        overlap = (
            train_video_ids
            & heldout_video_ids
        )

        if len(train_video_ids) != 32:
            raise ValueError(
                f"Fold {fold}: train videos != 32"
            )

        if len(heldout_video_ids) != 8:
            raise ValueError(
                f"Fold {fold}: held-out videos != 8"
            )

        if overlap:
            raise ValueError(
                f"Fold {fold}: train/test overlap"
            )

        train_chunks = (
            chunks[
                chunks["video_id"].isin(
                    train_video_ids
                )
            ]
            .copy()
            .reset_index(drop=True)
        )

        heldout_chunks = (
            chunks[
                chunks["video_id"].isin(
                    heldout_video_ids
                )
            ]
            .copy()
            .reset_index(drop=True)
        )

        train_texts = (
            train_chunks[
                "processed_text"
            ]
            .fillna("")
            .astype(str)
            .tolist()
        )

        heldout_texts = (
            heldout_chunks[
                "processed_text"
            ]
            .fillna("")
            .astype(str)
            .tolist()
        )

        # ====================================================
        # 5. LDA
        # ====================================================
        #
        # FIT:
        #   TRAIN CHUNKS ONLY
        #
        # HELD-OUT:
        #   TRANSFORM ONLY
        # ====================================================

        (
            lda_vectorizer,
            lda_model,
            _
        ) = train_lda(
            train_texts
        )

        heldout_term_matrix = (
            lda_vectorizer
            .transform(
                heldout_texts
            )
        )

        heldout_topic_matrix = (
            lda_model
            .transform(
                heldout_term_matrix
            )
        )

        topic_concept_affinity = (
            calculate_topic_concept_affinity(
                lda_model=lda_model,
                vectorizer=lda_vectorizer,
                concept_texts=concept_texts,
            )
        )

        lda_score_matrix = (
            calculate_document_concept_scores(
                heldout_topic_matrix,
                topic_concept_affinity,
            )
        )

        # ====================================================
        # 6. LSA
        # ====================================================
        #
        # FIT:
        #   TRAIN CHUNKS + FIXED CONCEPT CATALOG
        #
        # HELD-OUT:
        #   TRANSFORM ONLY
        # ====================================================

        (
            lsa_vectorizer,
            svd,
            _,
            concept_vectors,
        ) = train_lsa(
            train_texts,
            concept_texts,
        )

        heldout_tfidf = (
            lsa_vectorizer
            .transform(
                heldout_texts
            )
        )

        heldout_lsa_vectors = (
            svd
            .transform(
                heldout_tfidf
            )
        )

        lsa_score_matrix = (
            calculate_similarity(
                heldout_lsa_vectors,
                concept_vectors,
            )
        )

        # ====================================================
        # 7. BUILD SAME-SUBJECT HELD-OUT SCORES
        # ====================================================

        fold_rows = []

        for chunk_index, chunk in (
            heldout_chunks.iterrows()
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

                # Exact baseline rule:
                # compare only concepts from same subject
                if (
                    chunk_subject
                    != concept_subject
                ):
                    continue

                # Important:
                # baseline saves raw LDA/LSA scores
                # rounded to 4 decimals BEFORE fusion.
                lda_score = round(
                    float(
                        lda_score_matrix[
                            chunk_index,
                            concept_index,
                        ]
                    ),
                    4,
                )

                lsa_score = round(
                    float(
                        lsa_score_matrix[
                            chunk_index,
                            concept_index,
                        ]
                    ),
                    4,
                )

                fold_rows.append(
                    {
                        "outer_fold":
                            fold,

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
                            float(
                                chunk["start_sec"]
                            ),

                        "end_sec":
                            float(
                                chunk["end_sec"]
                            ),

                        "concept":
                            concept["concept"],

                        "lda_score":
                            lda_score,

                        "lsa_score":
                            lsa_score,
                    }
                )

        fold_scores = pd.DataFrame(
            fold_rows
        )

        if fold_scores.empty:
            raise ValueError(
                f"Fold {fold}: no concept scores generated"
            )

        # ====================================================
        # 8. BASELINE-EXACT PER-CHUNK NORMALIZATION
        # ====================================================
        #
        # This does NOT use GT.
        # This does NOT choose alpha.
        #
        # Each chunk compares only its same-subject concepts.
        # ====================================================

        fold_scores["lda_norm"] = (
            fold_scores
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

        fold_scores["lsa_norm"] = (
            fold_scores
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

        # ----------------------------------------------------
        # 9. FOLD AUDIT
        # ----------------------------------------------------

        expected_fold_rows = int(
            sum(
                concept_counts[
                    subject
                ]
                for subject in (
                    heldout_chunks[
                        "subject"
                    ]
                )
            )
        )

        actual_fold_rows = len(
            fold_scores
        )

        if (
            actual_fold_rows
            != expected_fold_rows
        ):
            raise ValueError(
                f"Fold {fold}: expected "
                f"{expected_fold_rows} score rows, "
                f"found {actual_fold_rows}"
            )

        duplicate_scores = int(
            fold_scores.duplicated(
                subset=[
                    "video_id",
                    "chunk_id",
                    "concept",
                ]
            ).sum()
        )

        if duplicate_scores != 0:
            raise ValueError(
                f"Fold {fold}: duplicate concept scores"
            )

        # Every chunk must contain exactly the
        # number of concepts in its own subject.
        chunk_score_counts = (
            fold_scores
            .groupby(
                [
                    "video_id",
                    "chunk_id",
                    "subject",
                ]
            )
            .size()
            .reset_index(
                name="score_count"
            )
        )

        chunk_score_counts[
            "expected_count"
        ] = (
            chunk_score_counts[
                "subject"
            ]
            .map(
                concept_counts
            )
        )

        bad_chunk_counts = (
            chunk_score_counts[
                chunk_score_counts[
                    "score_count"
                ]
                !=
                chunk_score_counts[
                    "expected_count"
                ]
            ]
        )

        if not bad_chunk_counts.empty:
            raise ValueError(
                f"Fold {fold}: invalid "
                "concept count for one or more chunks"
            )

        all_score_rows.extend(
            fold_scores.to_dict(
                "records"
            )
        )

        audit_rows.append(
            {
                "outer_fold":
                    fold,

                "train_videos":
                    len(train_video_ids),

                "heldout_videos":
                    len(heldout_video_ids),

                "train_chunks":
                    len(train_chunks),

                "heldout_chunks":
                    len(heldout_chunks),

                "score_rows":
                    actual_fold_rows,

                "expected_score_rows":
                    expected_fold_rows,

                "lda_vocab_size":
                    len(
                        lda_vectorizer
                        .get_feature_names_out()
                    ),

                "lda_topics":
                    int(
                        lda_model
                        .n_components
                    ),

                "lsa_vocab_size":
                    len(
                        lsa_vectorizer
                        .get_feature_names_out()
                    ),

                "lsa_components":
                    int(
                        svd.n_components
                    ),

                "video_overlap":
                    len(overlap),

                "heldout_text_used_in_fit":
                    "NO",

                "ground_truth_used":
                    "NO",

                "alpha_selected":
                    "NO",

                "threshold_selected":
                    "NO",

                "status":
                    "PASS",
            }
        )

        print(
            "Train videos    :",
            len(train_video_ids),
        )

        print(
            "Held-out videos :",
            len(heldout_video_ids),
        )

        print(
            "Train chunks    :",
            len(train_chunks),
        )

        print(
            "Held-out chunks :",
            len(heldout_chunks),
        )

        print(
            "Score rows      :",
            actual_fold_rows,
        )

        print(
            "LDA vocabulary :",
            len(
                lda_vectorizer
                .get_feature_names_out()
            ),
        )

        print(
            "LSA vocabulary :",
            len(
                lsa_vectorizer
                .get_feature_names_out()
            ),
        )

        print(
            "LSA dimensions :",
            int(
                svd.n_components
            ),
        )

        print(
            "Held-out text in fit: NO"
        )

        print(
            "Ground truth used   : NO"
        )

        print(
            f"Fold {fold}: PASS"
        )

    # --------------------------------------------------------
    # 10. COMBINE ALL OOF SCORES
    # --------------------------------------------------------

    oof = pd.DataFrame(
        all_score_rows
    )

    audit = pd.DataFrame(
        audit_rows
    )

    # --------------------------------------------------------
    # 11. GLOBAL OOF AUDIT
    # --------------------------------------------------------

    if (
        oof["video_id"].nunique()
        != 40
    ):
        raise ValueError(
            "OOF scores do not cover all 40 videos"
        )

    # Each video must belong to exactly one outer fold.
    video_fold_counts = (
        oof[
            [
                "video_id",
                "outer_fold",
            ]
        ]
        .drop_duplicates()
        .groupby(
            "video_id"
        )[
            "outer_fold"
        ]
        .nunique()
    )

    if not (
        video_fold_counts == 1
    ).all():
        raise ValueError(
            "A video appears in more than one OOF fold"
        )

    if len(
        video_fold_counts
    ) != 40:
        raise ValueError(
            "OOF video coverage != 40"
        )

    global_duplicates = int(
        oof.duplicated(
            subset=[
                "video_id",
                "chunk_id",
                "concept",
            ]
        ).sum()
    )

    if global_duplicates != 0:
        raise ValueError(
            f"Global duplicate score rows: "
            f"{global_duplicates}"
        )

    # Expected global row count:
    # each chunk x all concepts of its subject.
    expected_global_rows = int(
        sum(
            concept_counts[
                subject
            ]
            for subject in (
                chunks[
                    "subject"
                ]
            )
        )
    )

    if (
        len(oof)
        != expected_global_rows
    ):
        raise ValueError(
            f"Expected {expected_global_rows} "
            f"global score rows, "
            f"found {len(oof)}"
        )

    # --------------------------------------------------------
    # 12. SORT + WRITE
    # --------------------------------------------------------

    oof["_video_num"] = (
        oof["video_id"]
        .str.extract(
            r"(\d+)",
            expand=False,
        )
        .astype(int)
    )

    oof = (
        oof
        .sort_values(
            [
                "_video_num",
                "chunk_id",
                "concept",
            ]
        )
        .drop(
            columns=[
                "_video_num"
            ]
        )
        .reset_index(
            drop=True
        )
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    oof.to_csv(
        OOF_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    audit.to_csv(
        AUDIT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # 13. FINAL REPORT
    # --------------------------------------------------------

    print("\n" + "=" * 78)
    print("OOF SCORE AUDIT")
    print("=" * 78)

    print(
        audit[
            [
                "outer_fold",
                "train_videos",
                "heldout_videos",
                "heldout_chunks",
                "score_rows",
                "video_overlap",
                "status",
            ]
        ].to_string(
            index=False
        )
    )

    print("\nGlobal OOF videos :", oof["video_id"].nunique())
    print("Global OOF chunks :", chunks["video_id"].count())
    print("Global score rows :", len(oof))
    print("Expected rows     :", expected_global_rows)
    print("Duplicate rows    :", global_duplicates)

    print(
        "Each video in exactly one outer fold:",
        bool(
            (
                video_fold_counts
                == 1
            ).all()
        ),
    )

    print("\nGround truth read during scoring: NO")
    print("Alpha selected during scoring    : NO")
    print("Threshold selected during scoring: NO")
    print("Outer held-out text used in fit  : NO")

    print("\n" + "=" * 78)
    print(
        "P1.6 LEAKAGE-FREE OOF "
        "CONCEPT SCORING: PASS"
    )
    print("=" * 78)

    print(
        "40/40 DEV videos received "
        "out-of-fold concept scores."
    )

    print(
        "LDA/LSA scores were produced "
        "without held-out transcript fitting."
    )

    print(
        "No ground truth was used "
        "to select scoring hyperparameters."
    )

    print(
        "Saved:",
        OOF_PATH.relative_to(ROOT),
    )

    print(
        "Saved:",
        AUDIT_PATH.relative_to(ROOT),
    )


if __name__ == "__main__":
    main()