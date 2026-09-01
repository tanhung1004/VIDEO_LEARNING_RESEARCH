from pathlib import Path
import json
import sys

import pandas as pd

from sklearn.feature_extraction.text import (
    CountVectorizer,
    TfidfVectorizer,
)
from sklearn.decomposition import (
    LatentDirichletAllocation,
    TruncatedSVD,
)


ROOT = Path(__file__).resolve().parents[2]

sys.path.append(
    str(ROOT / "module1" / "src")
)

from preprocessing import preprocess_text


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

OUTPUT_PATH = (
    OUTPUT_DIR
    / "fold1_leakage_free_sanity.json"
)


# ============================================================
# FROZEN BASELINE CONFIG
# ============================================================

FOLD = 1

NUM_TOPICS = 6
RANDOM_STATE = 42

LDA_MAX_FEATURES = 3000
LDA_MAX_DF = 0.95
LDA_MAX_ITER = 50

LSA_MAX_FEATURES = 5000
LSA_MAX_COMPONENTS = 100


def main():
    print("=" * 76)
    print("P1.4 FOLD 1 LEAKAGE-FREE FIT / TRANSFORM SANITY")
    print("=" * 76)

    # --------------------------------------------------------
    # 1. LOAD DATA
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

    folds["video_id"] = (
        folds["video_id"]
        .str.strip()
        .str.lower()
    )

    folds["cv_fold"] = (
        folds["cv_fold"]
        .str.strip()
    )

    # --------------------------------------------------------
    # 2. VIDEO-LEVEL SPLIT
    # --------------------------------------------------------

    test_video_ids = set(
        folds.loc[
            folds["cv_fold"].eq(str(FOLD)),
            "video_id",
        ]
    )

    all_video_ids = set(
        folds["video_id"]
    )

    train_video_ids = (
        all_video_ids
        - test_video_ids
    )

    if len(all_video_ids) != 40:
        raise ValueError(
            f"Expected 40 DEV videos, "
            f"found {len(all_video_ids)}"
        )

    if len(train_video_ids) != 32:
        raise ValueError(
            f"Expected 32 train videos, "
            f"found {len(train_video_ids)}"
        )

    if len(test_video_ids) != 8:
        raise ValueError(
            f"Expected 8 held-out videos, "
            f"found {len(test_video_ids)}"
        )

    overlap = (
        train_video_ids
        & test_video_ids
    )

    if overlap:
        raise ValueError(
            f"Train/test video overlap: "
            f"{sorted(overlap)}"
        )

    train_chunks = chunks[
        chunks["video_id"].isin(
            train_video_ids
        )
    ].copy()

    test_chunks = chunks[
        chunks["video_id"].isin(
            test_video_ids
        )
    ].copy()

    if train_chunks.empty:
        raise ValueError(
            "Train chunks are empty"
        )

    if test_chunks.empty:
        raise ValueError(
            "Held-out chunks are empty"
        )

    # --------------------------------------------------------
    # 3. CRITICAL LEAKAGE ASSERTIONS
    # --------------------------------------------------------

    train_chunk_video_ids = set(
        train_chunks["video_id"]
    )

    test_chunk_video_ids = set(
        test_chunks["video_id"]
    )

    if (
        train_chunk_video_ids
        & test_chunk_video_ids
    ):
        raise ValueError(
            "LEAKAGE: video appears in both "
            "train and held-out chunks"
        )

    if not test_chunk_video_ids.issubset(
        test_video_ids
    ):
        raise ValueError(
            "Held-out chunk/video mismatch"
        )

    train_texts = (
        train_chunks["processed_text"]
        .fillna("")
        .astype(str)
        .tolist()
    )

    test_texts = (
        test_chunks["processed_text"]
        .fillna("")
        .astype(str)
        .tolist()
    )

    # --------------------------------------------------------
    # 4. LDA — FIT TRAIN ONLY
    # --------------------------------------------------------

    print("\n[LDA]")
    print(
        "FIT source:",
        "TRAIN CHUNKS ONLY",
    )

    lda_vectorizer = CountVectorizer(
        ngram_range=(1, 2),
        min_df=1,
        max_df=LDA_MAX_DF,
        max_features=LDA_MAX_FEATURES,
    )

    # FIT ONLY ON TRAIN
    X_train_lda = (
        lda_vectorizer
        .fit_transform(
            train_texts
        )
    )

    lda_model = (
        LatentDirichletAllocation(
            n_components=NUM_TOPICS,
            max_iter=LDA_MAX_ITER,
            learning_method="batch",
            random_state=RANDOM_STATE,
        )
    )

    # FIT ONLY ON TRAIN
    train_topic_vectors = (
        lda_model
        .fit_transform(
            X_train_lda
        )
    )

    print(
        "Train fit complete:",
        X_train_lda.shape,
    )

    # --------------------------------------------------------
    # 5. LDA — HELD-OUT TRANSFORM ONLY
    # --------------------------------------------------------

    print(
        "Held-out operation:",
        "TRANSFORM ONLY",
    )

    X_test_lda = (
        lda_vectorizer
        .transform(
            test_texts
        )
    )

    test_topic_vectors = (
        lda_model
        .transform(
            X_test_lda
        )
    )

    print(
        "Held-out transform complete:",
        X_test_lda.shape,
    )

    # --------------------------------------------------------
    # 6. PREPARE FIXED CONCEPT TEXT
    # --------------------------------------------------------

    concepts["concept_text"] = (
        concepts["concept"]
        .fillna("")
        .astype(str)
        + " "
        + concepts["description"]
        .fillna("")
        .astype(str)
    )

    concept_texts = (
        concepts["concept_text"]
        .apply(preprocess_text)
        .tolist()
    )

    # Concept catalog is fixed before CV and is not
    # held-out video content.
    #
    # To preserve the baseline LSA design, the semantic
    # space is fitted using:
    #
    # TRAIN transcript chunks + FIXED concept texts
    #
    # Held-out transcript chunks are excluded from fit.

    lsa_fit_texts = (
        list(train_texts)
        + list(concept_texts)
    )

    # --------------------------------------------------------
    # 7. LSA — FIT TRAIN + FIXED CONCEPT TEXT ONLY
    # --------------------------------------------------------

    print("\n[LSA]")
    print(
        "FIT source:",
        "TRAIN CHUNKS + FIXED CONCEPT CATALOG",
    )

    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=1,
        max_features=LSA_MAX_FEATURES,
    )

    # FIT WITHOUT HELD-OUT VIDEO TEXT
    X_lsa_fit = (
        tfidf
        .fit_transform(
            lsa_fit_texts
        )
    )

    max_allowed = min(
        X_lsa_fit.shape[0] - 1,
        X_lsa_fit.shape[1] - 1,
        LSA_MAX_COMPONENTS,
    )

    if max_allowed < 2:
        raise ValueError(
            "Not enough data for LSA"
        )

    svd = TruncatedSVD(
        n_components=max_allowed,
        random_state=RANDOM_STATE,
    )

    # FIT WITHOUT HELD-OUT VIDEO TEXT
    latent_fit = (
        svd
        .fit_transform(
            X_lsa_fit
        )
    )

    num_train_chunks = len(
        train_texts
    )

    train_lsa_vectors = (
        latent_fit[
            :num_train_chunks
        ]
    )

    concept_vectors = (
        latent_fit[
            num_train_chunks:
        ]
    )

    print(
        "Train/concept fit complete:",
        X_lsa_fit.shape,
    )

    # --------------------------------------------------------
    # 8. LSA — HELD-OUT TRANSFORM ONLY
    # --------------------------------------------------------

    print(
        "Held-out operation:",
        "TRANSFORM ONLY",
    )

    X_test_lsa = (
        tfidf
        .transform(
            test_texts
        )
    )

    test_lsa_vectors = (
        svd
        .transform(
            X_test_lsa
        )
    )

    print(
        "Held-out transform complete:",
        X_test_lsa.shape,
    )

    # --------------------------------------------------------
    # 9. FINAL SHAPE ASSERTIONS
    # --------------------------------------------------------

    if (
        train_topic_vectors.shape[0]
        != len(train_chunks)
    ):
        raise ValueError(
            "LDA train vector count mismatch"
        )

    if (
        test_topic_vectors.shape[0]
        != len(test_chunks)
    ):
        raise ValueError(
            "LDA test vector count mismatch"
        )

    if (
        train_lsa_vectors.shape[0]
        != len(train_chunks)
    ):
        raise ValueError(
            "LSA train vector count mismatch"
        )

    if (
        test_lsa_vectors.shape[0]
        != len(test_chunks)
    ):
        raise ValueError(
            "LSA test vector count mismatch"
        )

    if (
        concept_vectors.shape[0]
        != len(concepts)
    ):
        raise ValueError(
            "Concept vector count mismatch"
        )

    # --------------------------------------------------------
    # 10. AUDIT SUMMARY
    # --------------------------------------------------------

    train_subject_counts = (
        folds[
            folds["video_id"].isin(
                train_video_ids
            )
        ]
        .groupby("subject")
        ["video_id"]
        .nunique()
        .to_dict()
    )

    test_subject_counts = (
        folds[
            folds["video_id"].isin(
                test_video_ids
            )
        ]
        .groupby("subject")
        ["video_id"]
        .nunique()
        .to_dict()
    )

    summary = {
        "fold": FOLD,

        "train_video_count":
            len(train_video_ids),

        "heldout_video_count":
            len(test_video_ids),

        "train_chunk_count":
            len(train_chunks),

        "heldout_chunk_count":
            len(test_chunks),

        "train_video_ids":
            sorted(train_video_ids),

        "heldout_video_ids":
            sorted(test_video_ids),

        "train_subject_counts":
            train_subject_counts,

        "heldout_subject_counts":
            test_subject_counts,

        "video_overlap_count":
            len(overlap),

        "lda_fit_scope":
            "train_chunks_only",

        "lda_heldout_operation":
            "transform_only",

        "lda_vocabulary_size":
            len(
                lda_vectorizer
                .get_feature_names_out()
            ),

        "lda_topics":
            NUM_TOPICS,

        "lsa_fit_scope":
            (
                "train_chunks_plus_"
                "fixed_concept_catalog"
            ),

        "lsa_heldout_operation":
            "transform_only",

        "lsa_vocabulary_size":
            len(
                tfidf
                .get_feature_names_out()
            ),

        "lsa_components":
            int(
                svd.n_components
            ),

        "concept_count":
            len(concepts),

        "heldout_text_used_in_fit":
            False,

        "leakage_free_sanity":
            True,
    }

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PATH.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # 11. HUMAN-READABLE RESULT
    # --------------------------------------------------------

    print("\n" + "=" * 76)
    print("VIDEO-LEVEL SPLIT")
    print("=" * 76)

    print(
        "Train videos:",
        len(train_video_ids),
    )

    print(
        "Held-out videos:",
        len(test_video_ids),
    )

    print(
        "Train chunks:",
        len(train_chunks),
    )

    print(
        "Held-out chunks:",
        len(test_chunks),
    )

    print(
        "Video overlap:",
        len(overlap),
    )

    print(
        "\nHeld-out Fold 1 IDs:"
    )

    print(
        ", ".join(
            sorted(
                test_video_ids,
                key=lambda x: int(
                    x[1:]
                ),
            )
        )
    )

    print(
        "\nTrain subjects:",
        train_subject_counts,
    )

    print(
        "Held-out subjects:",
        test_subject_counts,
    )

    print("\n" + "=" * 76)
    print("LEAKAGE AUDIT")
    print("=" * 76)

    print(
        "LDA vectorizer fit:"
        " TRAIN ONLY"
    )

    print(
        "LDA model fit:"
        " TRAIN ONLY"
    )

    print(
        "LDA held-out:"
        " TRANSFORM ONLY"
    )

    print(
        "LSA TF-IDF/SVD fit:"
        " TRAIN + FIXED CONCEPTS ONLY"
    )

    print(
        "LSA held-out:"
        " TRANSFORM ONLY"
    )

    print(
        "Held-out text used in fit:"
        " NO"
    )

    print("\n" + "=" * 76)
    print(
        "P1.4 FOLD 1 "
        "LEAKAGE-FREE SANITY: PASS"
    )
    print("=" * 76)

    print(
        "No held-out Fold 1 transcript "
        "text participated in model fitting."
    )

    print(
        "Saved:",
        OUTPUT_PATH.relative_to(ROOT),
    )


if __name__ == "__main__":
    main()