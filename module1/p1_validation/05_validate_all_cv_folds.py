from pathlib import Path
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

AUDIT_PATH = (
    OUTPUT_DIR
    / "all_folds_leakage_audit.csv"
)

HELDOUT_PATH = (
    OUTPUT_DIR
    / "heldout_video_assignment.csv"
)


# ============================================================
# FROZEN BASELINE CONFIG
# ============================================================

NUM_TOPICS = 6
RANDOM_STATE = 42

LDA_MAX_FEATURES = 3000
LDA_MAX_DF = 0.95
LDA_MAX_ITER = 50

LSA_MAX_FEATURES = 5000
LSA_MAX_COMPONENTS = 100

EXPECTED_FOLDS = {1, 2, 3, 4, 5}


def numeric_video_sort(video_id):
    return int(
        str(video_id)
        .strip()
        .lower()
        .replace("v", "")
    )


def main():
    print("=" * 78)
    print("P1.5 ALL-FOLD LEAKAGE-FREE CV VALIDATION")
    print("=" * 78)

    # --------------------------------------------------------
    # 1. LOAD
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

    folds["cv_fold"] = pd.to_numeric(
        folds["cv_fold"],
        errors="raise",
    ).astype(int)

    # --------------------------------------------------------
    # 2. GLOBAL INPUT AUDIT
    # --------------------------------------------------------

    errors = []

    if folds["video_id"].nunique() != 40:
        errors.append(
            "Fold manifest must contain 40 unique videos."
        )

    if set(folds["cv_fold"]) != EXPECTED_FOLDS:
        errors.append(
            "Fold IDs must be exactly {1,2,3,4,5}."
        )

    if chunks["video_id"].nunique() != 40:
        errors.append(
            "Chunk file must contain 40 unique DEV videos."
        )

    duplicate_chunks = int(
        chunks.duplicated(
            subset=["video_id", "chunk_id"]
        ).sum()
    )

    if duplicate_chunks != 0:
        errors.append(
            f"Duplicate video/chunk pairs: {duplicate_chunks}"
        )

    fold_video_ids = set(
        folds["video_id"]
    )

    chunk_video_ids = set(
        chunks["video_id"]
    )

    if fold_video_ids != chunk_video_ids:
        errors.append(
            "Video IDs differ between folds and chunk file."
        )

    if errors:
        print("\nP1.5 INPUT AUDIT: FAIL")

        for i, error in enumerate(errors, 1):
            print(f"{i}. {error}")

        sys.exit(1)

    print("\nInput audit: PASS")
    print("DEV videos :", len(fold_video_ids))
    print("DEV chunks :", len(chunks))
    print("Concepts   :", len(concepts))

    # --------------------------------------------------------
    # 3. FIXED CONCEPT TEXT
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

    # --------------------------------------------------------
    # 4. LOOP ALL FIVE OUTER FOLDS
    # --------------------------------------------------------

    audit_rows = []
    heldout_rows = []

    for fold in range(1, 6):

        print("\n" + "=" * 78)
        print(f"FOLD {fold}")
        print("=" * 78)

        heldout_video_ids = set(
            folds.loc[
                folds["cv_fold"].eq(fold),
                "video_id",
            ]
        )

        train_video_ids = (
            fold_video_ids
            - heldout_video_ids
        )

        # ----------------------------------------------------
        # Video-level split assertions
        # ----------------------------------------------------

        overlap = (
            train_video_ids
            & heldout_video_ids
        )

        if len(train_video_ids) != 32:
            raise ValueError(
                f"Fold {fold}: "
                f"train video count != 32"
            )

        if len(heldout_video_ids) != 8:
            raise ValueError(
                f"Fold {fold}: "
                f"held-out video count != 8"
            )

        if overlap:
            raise ValueError(
                f"Fold {fold}: "
                f"train/test overlap {sorted(overlap)}"
            )

        train_chunks = chunks[
            chunks["video_id"].isin(
                train_video_ids
            )
        ].copy()

        heldout_chunks = chunks[
            chunks["video_id"].isin(
                heldout_video_ids
            )
        ].copy()

        if (
            set(train_chunks["video_id"])
            & set(heldout_chunks["video_id"])
        ):
            raise ValueError(
                f"Fold {fold}: chunk-level video leakage"
            )

        train_texts = (
            train_chunks["processed_text"]
            .fillna("")
            .astype(str)
            .tolist()
        )

        heldout_texts = (
            heldout_chunks["processed_text"]
            .fillna("")
            .astype(str)
            .tolist()
        )

        # ----------------------------------------------------
        # 5. LDA — TRAIN FIT ONLY
        # ----------------------------------------------------

        lda_vectorizer = CountVectorizer(
            ngram_range=(1, 2),
            min_df=1,
            max_df=LDA_MAX_DF,
            max_features=LDA_MAX_FEATURES,
        )

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

        train_topic_vectors = (
            lda_model
            .fit_transform(
                X_train_lda
            )
        )

        # HELD-OUT = TRANSFORM ONLY

        X_heldout_lda = (
            lda_vectorizer
            .transform(
                heldout_texts
            )
        )

        heldout_topic_vectors = (
            lda_model
            .transform(
                X_heldout_lda
            )
        )

        # ----------------------------------------------------
        # 6. LSA — TRAIN + FIXED CONCEPTS FIT
        # ----------------------------------------------------

        lsa_fit_texts = (
            list(train_texts)
            + list(concept_texts)
        )

        tfidf = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1,
            max_features=LSA_MAX_FEATURES,
        )

        X_lsa_fit = (
            tfidf
            .fit_transform(
                lsa_fit_texts
            )
        )

        n_components = min(
            X_lsa_fit.shape[0] - 1,
            X_lsa_fit.shape[1] - 1,
            LSA_MAX_COMPONENTS,
        )

        if n_components < 2:
            raise ValueError(
                f"Fold {fold}: insufficient LSA dimensions"
            )

        svd = TruncatedSVD(
            n_components=n_components,
            random_state=RANDOM_STATE,
        )

        latent_fit = (
            svd
            .fit_transform(
                X_lsa_fit
            )
        )

        train_lsa_vectors = (
            latent_fit[
                :len(train_texts)
            ]
        )

        concept_vectors = (
            latent_fit[
                len(train_texts):
            ]
        )

        # HELD-OUT = TRANSFORM ONLY

        X_heldout_lsa = (
            tfidf
            .transform(
                heldout_texts
            )
        )

        heldout_lsa_vectors = (
            svd
            .transform(
                X_heldout_lsa
            )
        )

        # ----------------------------------------------------
        # 7. SHAPE AUDIT
        # ----------------------------------------------------

        if (
            train_topic_vectors.shape[0]
            != len(train_chunks)
        ):
            raise ValueError(
                f"Fold {fold}: "
                "LDA train shape mismatch"
            )

        if (
            heldout_topic_vectors.shape[0]
            != len(heldout_chunks)
        ):
            raise ValueError(
                f"Fold {fold}: "
                "LDA held-out shape mismatch"
            )

        if (
            train_lsa_vectors.shape[0]
            != len(train_chunks)
        ):
            raise ValueError(
                f"Fold {fold}: "
                "LSA train shape mismatch"
            )

        if (
            heldout_lsa_vectors.shape[0]
            != len(heldout_chunks)
        ):
            raise ValueError(
                f"Fold {fold}: "
                "LSA held-out shape mismatch"
            )

        if (
            concept_vectors.shape[0]
            != len(concepts)
        ):
            raise ValueError(
                f"Fold {fold}: "
                "concept vector shape mismatch"
            )

        # ----------------------------------------------------
        # 8. SUBJECT BALANCE
        # ----------------------------------------------------

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

        heldout_subject_counts = (
            folds[
                folds["video_id"].isin(
                    heldout_video_ids
                )
            ]
            .groupby("subject")
            ["video_id"]
            .nunique()
            .to_dict()
        )

        for subject in [
            "SQL",
            "Python",
            "Java",
            "C++",
        ]:
            if (
                train_subject_counts.get(
                    subject,
                    0,
                )
                != 8
            ):
                raise ValueError(
                    f"Fold {fold}: "
                    f"{subject} train != 8"
                )

            if (
                heldout_subject_counts.get(
                    subject,
                    0,
                )
                != 2
            ):
                raise ValueError(
                    f"Fold {fold}: "
                    f"{subject} held-out != 2"
                )

        # ----------------------------------------------------
        # 9. SAVE AUDIT ROW
        # ----------------------------------------------------

        audit_rows.append(
            {
                "fold": fold,

                "train_videos":
                    len(train_video_ids),

                "heldout_videos":
                    len(heldout_video_ids),

                "train_chunks":
                    len(train_chunks),

                "heldout_chunks":
                    len(heldout_chunks),

                "video_overlap":
                    len(overlap),

                "lda_vocab_size":
                    len(
                        lda_vectorizer
                        .get_feature_names_out()
                    ),

                "lda_topics":
                    NUM_TOPICS,

                "lsa_vocab_size":
                    len(
                        tfidf
                        .get_feature_names_out()
                    ),

                "lsa_components":
                    int(n_components),

                "concept_count":
                    len(concepts),

                "lda_fit_scope":
                    "train_only",

                "lda_heldout_operation":
                    "transform_only",

                "lsa_fit_scope":
                    "train_plus_fixed_concepts",

                "lsa_heldout_operation":
                    "transform_only",

                "heldout_text_used_in_fit":
                    "NO",

                "status":
                    "PASS",
            }
        )

        for video_id in sorted(
            heldout_video_ids,
            key=numeric_video_sort,
        ):
            subject = folds.loc[
                folds["video_id"].eq(
                    video_id
                ),
                "subject",
            ].iloc[0]

            heldout_rows.append(
                {
                    "video_id": video_id,
                    "subject": subject,
                    "heldout_fold": fold,
                }
            )

        print(
            "Train videos   :",
            len(train_video_ids),
        )

        print(
            "Held-out videos:",
            len(heldout_video_ids),
        )

        print(
            "Train chunks   :",
            len(train_chunks),
        )

        print(
            "Held-out chunks:",
            len(heldout_chunks),
        )

        print(
            "Video overlap  :",
            len(overlap),
        )

        print(
            "Held-out IDs   :",
            ", ".join(
                sorted(
                    heldout_video_ids,
                    key=numeric_video_sort,
                )
            ),
        )

        print(
            "LDA held-out   : TRANSFORM ONLY"
        )

        print(
            "LSA held-out   : TRANSFORM ONLY"
        )

        print(
            f"Fold {fold}: PASS"
        )

    # --------------------------------------------------------
    # 10. GLOBAL HELD-OUT COVERAGE AUDIT
    # --------------------------------------------------------

    audit_df = pd.DataFrame(
        audit_rows
    )

    heldout_df = pd.DataFrame(
        heldout_rows
    )

    heldout_counts = (
        heldout_df[
            "video_id"
        ]
        .value_counts()
    )

    if len(heldout_df) != 40:
        raise ValueError(
            f"Expected 40 held-out assignments, "
            f"found {len(heldout_df)}"
        )

    if heldout_df[
        "video_id"
    ].nunique() != 40:
        raise ValueError(
            "Not all 40 videos were held out."
        )

    bad_counts = heldout_counts[
        heldout_counts != 1
    ]

    if not bad_counts.empty:
        raise ValueError(
            "Some videos were held out "
            "more or less than once: "
            f"{bad_counts.to_dict()}"
        )

    expected_ids = set(
        folds["video_id"]
    )

    actual_heldout_ids = set(
        heldout_df["video_id"]
    )

    if actual_heldout_ids != expected_ids:
        raise ValueError(
            "Held-out coverage does not "
            "match DEV video registry."
        )

    # --------------------------------------------------------
    # 11. WRITE AUDIT ARTIFACTS
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    audit_df.to_csv(
        AUDIT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    heldout_df = (
        heldout_df
        .sort_values(
            "video_id",
            key=lambda s:
                s.str.extract(
                    r"(\d+)",
                    expand=False,
                ).astype(int),
        )
        .reset_index(drop=True)
    )

    heldout_df.to_csv(
        HELDOUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # 12. FINAL REPORT
    # --------------------------------------------------------

    print("\n" + "=" * 78)
    print("ALL-FOLD AUDIT TABLE")
    print("=" * 78)

    print(
        audit_df[
            [
                "fold",
                "train_videos",
                "heldout_videos",
                "train_chunks",
                "heldout_chunks",
                "video_overlap",
                "status",
            ]
        ].to_string(
            index=False
        )
    )

    print("\n" + "=" * 78)
    print("GLOBAL HELD-OUT COVERAGE")
    print("=" * 78)

    print(
        "Held-out assignments:",
        len(heldout_df),
    )

    print(
        "Unique held-out videos:",
        heldout_df[
            "video_id"
        ].nunique(),
    )

    print(
        "Held out exactly once:",
        (
            heldout_counts.eq(1)
            .all()
        ),
    )

    print(
        "Held-out text used in fit:"
        " NO (all folds)"
    )

    print("\n" + "=" * 78)
    print(
        "P1.5 ALL-FOLD "
        "LEAKAGE-FREE CV VALIDATION: PASS"
    )
    print("=" * 78)

    print(
        "5/5 folds passed."
    )

    print(
        "40/40 DEV videos were held out "
        "exactly once."
    )

    print(
        "Every fold used 32 train videos "
        "and 8 held-out videos."
    )

    print(
        "No train/held-out video overlap "
        "was detected."
    )

    print(
        "Held-out transcript text never "
        "participated in LDA/LSA fitting."
    )

    print(
        "Saved:",
        AUDIT_PATH.relative_to(ROOT),
    )

    print(
        "Saved:",
        HELDOUT_PATH.relative_to(ROOT),
    )


if __name__ == "__main__":
    main()