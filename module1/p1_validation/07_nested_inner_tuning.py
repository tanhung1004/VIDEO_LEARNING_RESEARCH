from pathlib import Path
import json
import sys

import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
)


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

GT_PATH = (
    ROOT
    / "data"
    / "ground_truth"
    / "ground_truth_concepts.csv"
)

OUTPUT_DIR = (
    ROOT
    / "module1"
    / "results"
    / "p1_validation"
)

INNER_ASSIGNMENT_PATH = (
    OUTPUT_DIR
    / "inner_fold_assignments.csv"
)

INNER_AUDIT_PATH = (
    OUTPUT_DIR
    / "inner_cv_audit.csv"
)

SEARCH_PATH = (
    OUTPUT_DIR
    / "inner_search_results.csv"
)

SELECTED_PATH = (
    OUTPUT_DIR
    / "selected_hyperparameters_by_outer_fold.csv"
)

CONFIG_PATH = (
    OUTPUT_DIR
    / "nested_tuning_config.json"
)


# ============================================================
# PRE-SPECIFIED EXPERIMENT CONFIGURATION
# ============================================================

OUTER_FOLDS = [1, 2, 3, 4, 5]

INNER_SPLITS = 4
RANDOM_STATE = 42

ALPHAS = [
    round(i / 10, 1)
    for i in range(11)
]

AGGREGATIONS = [
    "max",
    "mean",
    "top2_mean",
    "top3_mean",
]

THRESHOLDS = [
    round(
        0.20 + 0.05 * i,
        2,
    )
    for i in range(11)
]

# Frozen pilot configuration.
BASELINE_ALPHA = 0.40
BASELINE_AGGREGATION = "top2_mean"
BASELINE_THRESHOLD = 0.40

# Only used for deterministic tie-breaking.
AGG_TIE_ORDER = {
    "top2_mean": 0,
    "max": 1,
    "mean": 2,
    "top3_mean": 3,
}


# ============================================================
# BASELINE-EXACT NORMALIZATION
# ============================================================

def normalize_series(series):

    min_value = series.min()
    max_value = series.max()

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


def video_number(video_id):

    return int(
        str(video_id)
        .strip()
        .lower()
        .replace("v", "")
    )


# ============================================================
# BUILD LEAKAGE-FREE SCORES FOR ONE INNER VALIDATION FOLD
# ============================================================

def score_inner_validation(
    train_chunks,
    validation_chunks,
    concepts,
    concept_texts,
    outer_fold,
    inner_fold,
):

    train_chunks = (
        train_chunks
        .copy()
        .reset_index(drop=True)
    )

    validation_chunks = (
        validation_chunks
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

    validation_texts = (
        validation_chunks[
            "processed_text"
        ]
        .fillna("")
        .astype(str)
        .tolist()
    )

    # --------------------------------------------------------
    # LDA
    #
    # FIT       = inner-train transcript only
    # TRANSFORM = inner-validation transcript only
    # --------------------------------------------------------

    (
        lda_vectorizer,
        lda_model,
        _
    ) = train_lda(
        train_texts
    )

    validation_term_matrix = (
        lda_vectorizer
        .transform(
            validation_texts
        )
    )

    validation_topic_matrix = (
        lda_model
        .transform(
            validation_term_matrix
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
            validation_topic_matrix,
            topic_concept_affinity,
        )
    )

    # --------------------------------------------------------
    # LSA
    #
    # FIT = inner-train transcript
    #       + fixed concept catalog
    #
    # inner-validation transcript = TRANSFORM ONLY
    # --------------------------------------------------------

    (
        lsa_vectorizer,
        svd,
        _,
        concept_vectors,
    ) = train_lsa(
        train_texts,
        concept_texts,
    )

    validation_tfidf = (
        lsa_vectorizer
        .transform(
            validation_texts
        )
    )

    validation_lsa_vectors = (
        svd
        .transform(
            validation_tfidf
        )
    )

    lsa_score_matrix = (
        calculate_similarity(
            validation_lsa_vectors,
            concept_vectors,
        )
    )

    # --------------------------------------------------------
    # Same-subject concept scores
    # --------------------------------------------------------

    rows = []

    for chunk_index, chunk in (
        validation_chunks.iterrows()
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

            if (
                chunk_subject
                != concept_subject
            ):
                continue

            # Frozen baseline writes these
            # raw scores at 4 decimals before fusion.
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

            rows.append(
                {
                    "outer_fold":
                        outer_fold,

                    "inner_fold":
                        inner_fold,

                    "video_id":
                        chunk["video_id"],

                    "subject":
                        chunk_subject,

                    "chunk_id":
                        int(
                            chunk["chunk_id"]
                        ),

                    "concept":
                        concept["concept"],

                    "lda_score":
                        lda_score,

                    "lsa_score":
                        lsa_score,
                }
            )

    scores = pd.DataFrame(
        rows
    )

    if scores.empty:
        raise ValueError(
            f"Outer {outer_fold}, inner {inner_fold}: "
            "no validation scores"
        )

    # --------------------------------------------------------
    # Baseline-exact per-chunk normalization
    # --------------------------------------------------------

    scores["lda_norm"] = (
        scores
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

    scores["lsa_norm"] = (
        scores
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

    model_info = {
        "lda_vocab_size":
            len(
                lda_vectorizer
                .get_feature_names_out()
            ),

        "lda_topics":
            int(
                lda_model.n_components
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
    }

    return scores, model_info


# ============================================================
# FUSION + VIDEO AGGREGATION
# ============================================================

def aggregate_scores(
    scores,
    alpha,
    aggregation,
):

    work = scores[
        [
            "video_id",
            "subject",
            "chunk_id",
            "concept",
            "lda_norm",
            "lsa_norm",
        ]
    ].copy()

    # alpha = LDA weight
    # 1-alpha = LSA weight
    #
    # Frozen baseline:
    # alpha = 0.40
    work["final_score"] = (
        alpha
        * work["lda_norm"]
        +
        (1.0 - alpha)
        * work["lsa_norm"]
    )

    # Frozen baseline rounds final chunk score
    # before video-level evidence aggregation.
    work["final_score"] = (
        work["final_score"]
        .round(4)
    )

    grouped = work.groupby(
        [
            "video_id",
            "subject",
            "concept",
        ],
        sort=False,
    )["final_score"]

    if aggregation == "max":

        result = (
            grouped
            .max()
            .reset_index(
                name="video_score"
            )
        )

    elif aggregation == "mean":

        result = (
            grouped
            .mean()
            .reset_index(
                name="video_score"
            )
        )

    elif aggregation == "top2_mean":

        result = (
            grouped
            .agg(
                lambda s:
                    s.nlargest(2)
                    .mean()
            )
            .reset_index(
                name="video_score"
            )
        )

    elif aggregation == "top3_mean":

        result = (
            grouped
            .agg(
                lambda s:
                    s.nlargest(3)
                    .mean()
            )
            .reset_index(
                name="video_score"
            )
        )

    else:

        raise ValueError(
            f"Unknown aggregation: "
            f"{aggregation}"
        )

    result["video_score"] = (
        result["video_score"]
        .round(4)
    )

    return result


# ============================================================
# EVALUATION ON INNER-OOF VIDEOS ONLY
# ============================================================

def evaluate_candidate(
    aggregated,
    positive_gt,
    threshold,
):

    evaluation = aggregated.copy()

    evaluation["y_true"] = [
        1
        if (
            row.video_id,
            row.concept,
        )
        in positive_gt
        else 0
        for row in evaluation.itertuples()
    ]

    evaluation["y_pred"] = (
        evaluation["video_score"]
        >= threshold
    ).astype(int)

    # --------------------------------------------------------
    # Primary metric:
    # mean F1 across videos
    # --------------------------------------------------------

    per_video_rows = []

    for video_id, group in (
        evaluation
        .groupby(
            "video_id"
        )
    ):

        y_true = (
            group["y_true"]
            .astype(int)
        )

        y_pred = (
            group["y_pred"]
            .astype(int)
        )

        per_video_rows.append(
            {
                "video_id":
                    video_id,

                "precision":
                    precision_score(
                        y_true,
                        y_pred,
                        zero_division=0,
                    ),

                "recall":
                    recall_score(
                        y_true,
                        y_pred,
                        zero_division=0,
                    ),

                "f1":
                    f1_score(
                        y_true,
                        y_pred,
                        zero_division=0,
                    ),
            }
        )

    per_video = pd.DataFrame(
        per_video_rows
    )

    macro_video_precision = float(
        per_video[
            "precision"
        ].mean()
    )

    macro_video_recall = float(
        per_video[
            "recall"
        ].mean()
    )

    macro_video_f1 = float(
        per_video[
            "f1"
        ].mean()
    )

    # --------------------------------------------------------
    # Secondary micro metrics
    # --------------------------------------------------------

    y_true_all = (
        evaluation[
            "y_true"
        ].astype(int)
    )

    y_pred_all = (
        evaluation[
            "y_pred"
        ].astype(int)
    )

    micro_precision = float(
        precision_score(
            y_true_all,
            y_pred_all,
            zero_division=0,
        )
    )

    micro_recall = float(
        recall_score(
            y_true_all,
            y_pred_all,
            zero_division=0,
        )
    )

    micro_f1 = float(
        f1_score(
            y_true_all,
            y_pred_all,
            zero_division=0,
        )
    )

    return {
        "macro_video_precision":
            macro_video_precision,

        "macro_video_recall":
            macro_video_recall,

        "macro_video_f1":
            macro_video_f1,

        "micro_precision":
            micro_precision,

        "micro_recall":
            micro_recall,

        "micro_f1":
            micro_f1,

        "num_video_concept_pairs":
            len(evaluation),

        "num_true_positive_labels":
            int(
                evaluation[
                    "y_true"
                ].sum()
            ),

        "num_predicted_positive_labels":
            int(
                evaluation[
                    "y_pred"
                ].sum()
            ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("P1.7 NESTED INNER-CV HYPERPARAMETER TUNING")
    print("=" * 80)

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

    gt = pd.read_csv(
        GT_PATH,
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

    chunks["chunk_id"] = pd.to_numeric(
        chunks["chunk_id"],
        errors="raise",
    ).astype(int)

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

    concepts = (
        concepts
        .reset_index(drop=True)
    )

    gt["video_id"] = (
        gt["video_id"]
        .str.strip()
        .str.lower()
    )

    gt["concept"] = (
        gt["concept"]
        .str.strip()
    )

    # --------------------------------------------------------
    # 2. GLOBAL INPUT AUDIT
    # --------------------------------------------------------

    if (
        folds["video_id"]
        .nunique()
        != 40
    ):
        raise ValueError(
            "Expected 40 unique DEV videos"
        )

    if (
        chunks["video_id"]
        .nunique()
        != 40
    ):
        raise ValueError(
            "Expected 40 videos in chunk file"
        )

    if len(concepts) != 39:
        raise ValueError(
            f"Expected 39 concepts, "
            f"found {len(concepts)}"
        )

    if set(
        folds["cv_fold"]
    ) != {
        1, 2, 3, 4, 5
    }:
        raise ValueError(
            "Outer folds must be 1..5"
        )

    print("\nInput audit: PASS")
    print("Videos  : 40")
    print(
        "Chunks  :",
        len(chunks),
    )
    print(
        "Concepts:",
        len(concepts),
    )

    # --------------------------------------------------------
    # 3. FIXED CONCEPT TEXT
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
        .apply(
            preprocess_text
        )
    )

    concept_texts = (
        concepts[
            "processed_concept_text"
        ]
        .tolist()
    )

    # --------------------------------------------------------
    # 4. OUTPUT COLLECTORS
    # --------------------------------------------------------

    assignment_rows = []
    audit_rows = []
    search_rows = []
    selected_rows = []

    all_video_ids = set(
        folds[
            "video_id"
        ]
    )

    # ========================================================
    # 5. OUTER LOOP
    # ========================================================

    for outer_fold in OUTER_FOLDS:

        print("\n" + "=" * 80)
        print(
            f"OUTER FOLD {outer_fold}"
        )
        print("=" * 80)

        outer_test_ids = set(
            folds.loc[
                folds[
                    "cv_fold"
                ].eq(
                    outer_fold
                ),
                "video_id",
            ]
        )

        outer_train_ids = (
            all_video_ids
            - outer_test_ids
        )

        if (
            len(
                outer_train_ids
            )
            != 32
        ):
            raise ValueError(
                f"Outer {outer_fold}: "
                "train count != 32"
            )

        if (
            len(
                outer_test_ids
            )
            != 8
        ):
            raise ValueError(
                f"Outer {outer_fold}: "
                "test count != 8"
            )

        if (
            outer_train_ids
            & outer_test_ids
        ):
            raise ValueError(
                f"Outer {outer_fold}: "
                "train/test overlap"
            )

        # ----------------------------------------------------
        # Metadata for ONLY outer-training videos.
        #
        # Outer-test videos are completely absent
        # from inner fold generation.
        # ----------------------------------------------------

        outer_train_meta = (
            folds[
                folds[
                    "video_id"
                ].isin(
                    outer_train_ids
                )
            ][
                [
                    "video_id",
                    "subject",
                ]
            ]
            .copy()
        )

        outer_train_meta[
            "_video_num"
        ] = (
            outer_train_meta[
                "video_id"
            ]
            .str.extract(
                r"(\d+)",
                expand=False,
            )
            .astype(int)
        )

        outer_train_meta = (
            outer_train_meta
            .sort_values(
                "_video_num"
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

        # ----------------------------------------------------
        # Four-fold stratified INNER CV
        # ----------------------------------------------------

        inner_cv = StratifiedKFold(
            n_splits=INNER_SPLITS,
            shuffle=True,
            random_state=RANDOM_STATE,
        )

        inner_score_parts = []

        # ====================================================
        # INNER LOOP
        # ====================================================

        for inner_fold, (
            inner_train_idx,
            inner_val_idx,
        ) in enumerate(
            inner_cv.split(
                outer_train_meta[
                    [
                        "video_id"
                    ]
                ],
                outer_train_meta[
                    "subject"
                ],
            ),
            start=1,
        ):

            inner_train_meta = (
                outer_train_meta
                .iloc[
                    inner_train_idx
                ]
                .copy()
            )

            inner_val_meta = (
                outer_train_meta
                .iloc[
                    inner_val_idx
                ]
                .copy()
            )

            inner_train_ids = set(
                inner_train_meta[
                    "video_id"
                ]
            )

            inner_val_ids = set(
                inner_val_meta[
                    "video_id"
                ]
            )

            if (
                len(
                    inner_train_ids
                )
                != 24
            ):
                raise ValueError(
                    f"Outer {outer_fold}, "
                    f"inner {inner_fold}: "
                    "train != 24"
                )

            if (
                len(
                    inner_val_ids
                )
                != 8
            ):
                raise ValueError(
                    f"Outer {outer_fold}, "
                    f"inner {inner_fold}: "
                    "validation != 8"
                )

            if (
                inner_train_ids
                & inner_val_ids
            ):
                raise ValueError(
                    f"Outer {outer_fold}, "
                    f"inner {inner_fold}: "
                    "train/validation overlap"
                )

            # Critical nested-CV assertion:
            # outer test cannot enter inner train OR val.
            if (
                (
                    inner_train_ids
                    | inner_val_ids
                )
                & outer_test_ids
            ):
                raise ValueError(
                    f"Outer {outer_fold}, "
                    f"inner {inner_fold}: "
                    "OUTER TEST LEAKAGE"
                )

            inner_train_subjects = (
                inner_train_meta[
                    "subject"
                ]
                .value_counts()
                .to_dict()
            )

            inner_val_subjects = (
                inner_val_meta[
                    "subject"
                ]
                .value_counts()
                .to_dict()
            )

            for subject in [
                "SQL",
                "Python",
                "Java",
                "C++",
            ]:

                if (
                    inner_train_subjects
                    .get(
                        subject,
                        0,
                    )
                    != 6
                ):
                    raise ValueError(
                        f"Outer {outer_fold}, "
                        f"inner {inner_fold}: "
                        f"{subject} train != 6"
                    )

                if (
                    inner_val_subjects
                    .get(
                        subject,
                        0,
                    )
                    != 2
                ):
                    raise ValueError(
                        f"Outer {outer_fold}, "
                        f"inner {inner_fold}: "
                        f"{subject} val != 2"
                    )

            inner_train_chunks = (
                chunks[
                    chunks[
                        "video_id"
                    ].isin(
                        inner_train_ids
                    )
                ]
                .copy()
            )

            inner_val_chunks = (
                chunks[
                    chunks[
                        "video_id"
                    ].isin(
                        inner_val_ids
                    )
                ]
                .copy()
            )

            scores, model_info = (
                score_inner_validation(
                    train_chunks=
                        inner_train_chunks,

                    validation_chunks=
                        inner_val_chunks,

                    concepts=
                        concepts,

                    concept_texts=
                        concept_texts,

                    outer_fold=
                        outer_fold,

                    inner_fold=
                        inner_fold,
                )
            )

            inner_score_parts.append(
                scores
            )

            for _, row in (
                inner_val_meta
                .iterrows()
            ):

                assignment_rows.append(
                    {
                        "outer_fold":
                            outer_fold,

                        "video_id":
                            row[
                                "video_id"
                            ],

                        "subject":
                            row[
                                "subject"
                            ],

                        "inner_validation_fold":
                            inner_fold,
                    }
                )

            audit_rows.append(
                {
                    "outer_fold":
                        outer_fold,

                    "inner_fold":
                        inner_fold,

                    "inner_train_videos":
                        len(
                            inner_train_ids
                        ),

                    "inner_validation_videos":
                        len(
                            inner_val_ids
                        ),

                    "inner_train_chunks":
                        len(
                            inner_train_chunks
                        ),

                    "inner_validation_chunks":
                        len(
                            inner_val_chunks
                        ),

                    "outer_test_videos_present":
                        len(
                            (
                                inner_train_ids
                                | inner_val_ids
                            )
                            & outer_test_ids
                        ),

                    "lda_vocab_size":
                        model_info[
                            "lda_vocab_size"
                        ],

                    "lda_topics":
                        model_info[
                            "lda_topics"
                        ],

                    "lsa_vocab_size":
                        model_info[
                            "lsa_vocab_size"
                        ],

                    "lsa_components":
                        model_info[
                            "lsa_components"
                        ],

                    "outer_test_text_used":
                        "NO",

                    "outer_test_gt_used":
                        "NO",

                    "status":
                        "PASS",
                }
            )

            print(
                f"Inner {inner_fold}: "
                f"24 train / "
                f"8 validation — PASS"
            )

        # ----------------------------------------------------
        # 6. COMBINE INNER OOF SCORES
        # ----------------------------------------------------

        inner_scores = pd.concat(
            inner_score_parts,
            ignore_index=True,
        )

        inner_video_folds = (
            inner_scores[
                [
                    "video_id",
                    "inner_fold",
                ]
            ]
            .drop_duplicates()
        )

        if (
            inner_video_folds[
                "video_id"
            ].nunique()
            != 32
        ):
            raise ValueError(
                f"Outer {outer_fold}: "
                "inner OOF coverage != 32"
            )

        per_video_inner_count = (
            inner_video_folds
            .groupby(
                "video_id"
            )[
                "inner_fold"
            ]
            .nunique()
        )

        if not (
            per_video_inner_count
            == 1
        ).all():
            raise ValueError(
                f"Outer {outer_fold}: "
                "video appears in multiple "
                "inner validation folds"
            )

        if (
            set(
                inner_scores[
                    "video_id"
                ]
            )
            & outer_test_ids
        ):
            raise ValueError(
                f"Outer {outer_fold}: "
                "outer-test score leakage "
                "inside inner tuning"
            )

        # ----------------------------------------------------
        # 7. GT FOR OUTER TRAIN ONLY
        # ----------------------------------------------------

        positive_gt = set(
            zip(
                gt.loc[
                    gt[
                        "video_id"
                    ].isin(
                        outer_train_ids
                    ),
                    "video_id",
                ],

                gt.loc[
                    gt[
                        "video_id"
                    ].isin(
                        outer_train_ids
                    ),
                    "concept",
                ],
            )
        )

        # ====================================================
        # 8. PRE-SPECIFIED GRID SEARCH
        # ====================================================

        outer_search_rows = []

        for alpha in ALPHAS:

            for aggregation in AGGREGATIONS:

                aggregated = aggregate_scores(
                    scores=
                        inner_scores,

                    alpha=
                        alpha,

                    aggregation=
                        aggregation,
                )

                if (
                    aggregated[
                        "video_id"
                    ].nunique()
                    != 32
                ):
                    raise ValueError(
                        f"Outer {outer_fold}: "
                        "aggregated coverage != 32"
                    )

                for threshold in THRESHOLDS:

                    metrics = (
                        evaluate_candidate(
                            aggregated=
                                aggregated,

                            positive_gt=
                                positive_gt,

                            threshold=
                                threshold,
                        )
                    )

                    row = {
                        "outer_fold":
                            outer_fold,

                        "alpha":
                            alpha,

                        "aggregation":
                            aggregation,

                        "threshold":
                            threshold,

                        **metrics,

                        # Tie-break fields.
                        "baseline_aggregation_penalty":
                            (
                                0
                                if aggregation
                                ==
                                BASELINE_AGGREGATION
                                else 1
                            ),

                        "alpha_distance_from_baseline":
                            round(
                                abs(
                                    alpha
                                    -
                                    BASELINE_ALPHA
                                ),
                                4,
                            ),

                        "threshold_distance_from_baseline":
                            round(
                                abs(
                                    threshold
                                    -
                                    BASELINE_THRESHOLD
                                ),
                                4,
                            ),

                        "aggregation_tie_order":
                            AGG_TIE_ORDER[
                                aggregation
                            ],
                    }

                    outer_search_rows.append(
                        row
                    )

        outer_search = pd.DataFrame(
            outer_search_rows
        )

        expected_grid_size = (
            len(ALPHAS)
            * len(AGGREGATIONS)
            * len(THRESHOLDS)
        )

        if (
            len(
                outer_search
            )
            != expected_grid_size
        ):
            raise ValueError(
                f"Outer {outer_fold}: "
                "grid size mismatch"
            )

        # ----------------------------------------------------
        # 9. DETERMINISTIC SELECTION
        #
        # Priority:
        # 1. highest macro video F1
        # 2. highest micro F1
        #
        # Exact ties only:
        # 3. prefer original aggregation
        # 4. alpha closest to 0.40
        # 5. threshold closest to 0.40
        # 6. deterministic fixed ordering
        #
        # NO OUTER TEST PERFORMANCE HERE.
        # ----------------------------------------------------

        ranked = (
            outer_search
            .sort_values(
                by=[
                    "macro_video_f1",
                    "micro_f1",
                    "baseline_aggregation_penalty",
                    "alpha_distance_from_baseline",
                    "threshold_distance_from_baseline",
                    "aggregation_tie_order",
                    "alpha",
                    "threshold",
                ],
                ascending=[
                    False,
                    False,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                ],
                kind="stable",
            )
            .reset_index(drop=True)
        )

        selected = ranked.iloc[0]

        # ----------------------------------------------------
        # Frozen baseline candidate for comparison INSIDE
        # outer-train only.
        # ----------------------------------------------------

        baseline_candidate = (
            outer_search[
                outer_search[
                    "alpha"
                ].eq(
                    BASELINE_ALPHA
                )
                &
                outer_search[
                    "aggregation"
                ].eq(
                    BASELINE_AGGREGATION
                )
                &
                outer_search[
                    "threshold"
                ].eq(
                    BASELINE_THRESHOLD
                )
            ]
        )

        if len(
            baseline_candidate
        ) != 1:
            raise ValueError(
                "Frozen baseline candidate "
                "not uniquely found"
            )

        baseline_candidate = (
            baseline_candidate
            .iloc[0]
        )

        selected_rows.append(
            {
                "outer_fold":
                    outer_fold,

                "selected_alpha":
                    float(
                        selected[
                            "alpha"
                        ]
                    ),

                "selected_aggregation":
                    selected[
                        "aggregation"
                    ],

                "selected_threshold":
                    float(
                        selected[
                            "threshold"
                        ]
                    ),

                "inner_macro_video_f1":
                    float(
                        selected[
                            "macro_video_f1"
                        ]
                    ),

                "inner_micro_f1":
                    float(
                        selected[
                            "micro_f1"
                        ]
                    ),

                "frozen_baseline_inner_macro_f1":
                    float(
                        baseline_candidate[
                            "macro_video_f1"
                        ]
                    ),

                "frozen_baseline_inner_micro_f1":
                    float(
                        baseline_candidate[
                            "micro_f1"
                        ]
                    ),

                "inner_cv_folds":
                    INNER_SPLITS,

                "outer_train_videos":
                    32,

                "outer_test_videos":
                    8,

                "outer_test_gt_used_for_selection":
                    "NO",

                "outer_test_text_used_for_inner_models":
                    "NO",

                "selection_metric":
                    "macro_video_f1",

                "secondary_metric":
                    "micro_f1",

                "status":
                    "PASS",
            }
        )

        search_rows.extend(
            outer_search.to_dict(
                "records"
            )
        )

        print(
            "\nInner OOF coverage : 32/32"
        )

        print(
            "Grid candidates    :",
            expected_grid_size,
        )

        print(
            "Selected alpha     :",
            selected["alpha"],
        )

        print(
            "Selected aggregation:",
            selected[
                "aggregation"
            ],
        )

        print(
            "Selected threshold :",
            selected[
                "threshold"
            ],
        )

        print(
            "Inner macro F1     :",
            round(
                float(
                    selected[
                        "macro_video_f1"
                    ]
                ),
                4,
            ),
        )

        print(
            "Outer-test GT used : NO"
        )

        print(
            "Outer-test text in inner models: NO"
        )

        print(
            f"Outer Fold {outer_fold} tuning: PASS"
        )

    # ========================================================
    # 10. GLOBAL AUDITS
    # ========================================================

    assignments = pd.DataFrame(
        assignment_rows
    )

    audits = pd.DataFrame(
        audit_rows
    )

    search_results = pd.DataFrame(
        search_rows
    )

    selected_df = pd.DataFrame(
        selected_rows
    )

    # 32 inner validation assignments
    # for each of 5 outer folds.
    if (
        len(
            assignments
        )
        != 160
    ):
        raise ValueError(
            f"Expected 160 inner assignments, "
            f"found {len(assignments)}"
        )

    if (
        len(
            audits
        )
        != 20
    ):
        raise ValueError(
            f"Expected 20 inner fold audits, "
            f"found {len(audits)}"
        )

    expected_total_search = (
        5
        * len(ALPHAS)
        * len(AGGREGATIONS)
        * len(THRESHOLDS)
    )

    if (
        len(
            search_results
        )
        != expected_total_search
    ):
        raise ValueError(
            f"Expected "
            f"{expected_total_search} "
            f"search rows, found "
            f"{len(search_results)}"
        )

    if (
        len(
            selected_df
        )
        != 5
    ):
        raise ValueError(
            "Expected one selected "
            "configuration per outer fold"
        )

    if (
        audits[
            "outer_test_videos_present"
        ].astype(int).sum()
        != 0
    ):
        raise ValueError(
            "Outer test leakage detected "
            "inside inner CV"
        )

    # ========================================================
    # 11. WRITE ONLY AFTER ALL AUDITS PASS
    # ========================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    assignments = (
        assignments
        .sort_values(
            by=[
                "outer_fold",
                "video_id",
            ],
            key=lambda s:
                (
                    s.str.extract(
                        r"(\d+)",
                        expand=False,
                    ).astype(int)
                    if s.name
                    == "video_id"
                    else s
                ),
        )
        .reset_index(drop=True)
    )

    assignments.to_csv(
        INNER_ASSIGNMENT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    audits.to_csv(
        INNER_AUDIT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    search_results.to_csv(
        SEARCH_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    selected_df.to_csv(
        SELECTED_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    config = {
        "outer_folds":
            OUTER_FOLDS,

        "inner_splits":
            INNER_SPLITS,

        "inner_split_method":
            "StratifiedKFold",

        "inner_random_state":
            RANDOM_STATE,

        "alpha_grid":
            ALPHAS,

        "aggregation_grid":
            AGGREGATIONS,

        "threshold_grid":
            THRESHOLDS,

        "primary_selection_metric":
            "mean video-level F1",

        "secondary_selection_metric":
            "micro F1",

        "tie_break_rule": [
            "prefer frozen top2_mean aggregation",
            "alpha closest to 0.40",
            "threshold closest to 0.40",
            "fixed deterministic ordering",
        ],

        "outer_test_used_for_tuning":
            False,

        "fixed_concept_catalog_allowed_in_lsa_fit":
            True,
    }

    CONFIG_PATH.write_text(
        json.dumps(
            config,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # ========================================================
    # 12. FINAL REPORT
    # ========================================================

    print("\n" + "=" * 80)
    print("SELECTED HYPERPARAMETERS")
    print("=" * 80)

    print(
        selected_df[
            [
                "outer_fold",
                "selected_alpha",
                "selected_aggregation",
                "selected_threshold",
                "inner_macro_video_f1",
                "inner_micro_f1",
            ]
        ].to_string(
            index=False
        )
    )

    print("\nInner fold audits      :", len(audits))
    print("Inner assignments      :", len(assignments))
    print("Grid search rows       :", len(search_results))
    print("Selected configurations:", len(selected_df))

    print(
        "Outer test present in inner CV:",
        int(
            audits[
                "outer_test_videos_present"
            ]
            .astype(int)
            .sum()
        ),
    )

    print(
        "Outer-test GT used for selection: NO"
    )

    print(
        "Outer-test text used in inner models: NO"
    )

    print("\n" + "=" * 80)
    print(
        "P1.7 NESTED INNER-CV "
        "HYPERPARAMETER TUNING: PASS"
    )
    print("=" * 80)

    print(
        "Hyperparameters were selected "
        "using INNER out-of-fold predictions "
        "from outer-training videos only."
    )

    print(
        "Outer held-out videos were not "
        "used for model fitting or "
        "hyperparameter selection."
    )

    print(
        "Saved:",
        INNER_ASSIGNMENT_PATH.relative_to(ROOT),
    )

    print(
        "Saved:",
        INNER_AUDIT_PATH.relative_to(ROOT),
    )

    print(
        "Saved:",
        SEARCH_PATH.relative_to(ROOT),
    )

    print(
        "Saved:",
        SELECTED_PATH.relative_to(ROOT),
    )

    print(
        "Saved:",
        CONFIG_PATH.relative_to(ROOT),
    )


if __name__ == "__main__":
    main()