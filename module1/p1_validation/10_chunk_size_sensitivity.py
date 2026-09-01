from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

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

VIDEOS_PATH = (
    ROOT
    / "data"
    / "raw"
    / "videos.csv"
)

TRANSCRIPT_DIR = (
    ROOT
    / "data"
    / "raw"
    / "transcripts"
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

P18_PREDICTIONS_PATH = (
    ROOT
    / "module1"
    / "results"
    / "p1_validation"
    / "oof_predictions.csv"
)

OUTPUT_DIR = (
    ROOT
    / "module1"
    / "results"
    / "p1_validation"
)

SENSITIVITY_DIR = (
    OUTPUT_DIR
    / "chunk_sensitivity"
)

RESULT_PATH = (
    OUTPUT_DIR
    / "chunk_size_results.csv"
)

FOLD_PATH = (
    OUTPUT_DIR
    / "chunk_size_fold_metrics.csv"
)

SUBJECT_PATH = (
    OUTPUT_DIR
    / "chunk_size_subject_metrics.csv"
)

PREDICTION_PATH = (
    OUTPUT_DIR
    / "chunk_sensitivity_predictions.csv"
)

AUDIT_PATH = (
    OUTPUT_DIR
    / "chunk_sensitivity_audit.json"
)


# ============================================================
# PRE-SPECIFIED SENSITIVITY CONFIG
# ============================================================

CHUNK_SIZES = [
    30,
    60,
    90,
    120,
]

PRIMARY_CHUNK_SIZE = 60

# Frozen baseline configuration.
#
# Sensitivity analysis changes ONLY chunk size.
FROZEN_ALPHA = 0.40
FROZEN_AGGREGATION = "top2_mean"
FROZEN_THRESHOLD = 0.40


# ============================================================
# HELPERS
# ============================================================

def video_number(video_id):

    return int(
        str(video_id)
        .strip()
        .lower()
        .replace("v", "")
    )


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


def calculate_metrics(
    y_true,
    y_pred,
):

    y_true = (
        pd.Series(
            y_true
        )
        .astype(int)
    )

    y_pred = (
        pd.Series(
            y_pred
        )
        .astype(int)
    )

    return {
        "precision":
            float(
                precision_score(
                    y_true,
                    y_pred,
                    zero_division=0,
                )
            ),

        "recall":
            float(
                recall_score(
                    y_true,
                    y_pred,
                    zero_division=0,
                )
            ),

        "f1":
            float(
                f1_score(
                    y_true,
                    y_pred,
                    zero_division=0,
                )
            ),
    }


# ============================================================
# CHUNK BUILDER
# ============================================================

def build_video_chunks(
    video_id,
    subject,
    chunk_seconds,
):

    transcript_path = (
        TRANSCRIPT_DIR
        / f"{video_id}_transcript.csv"
    )

    if not transcript_path.exists():

        raise FileNotFoundError(
            f"Missing transcript: "
            f"{transcript_path}"
        )

    df = pd.read_csv(
        transcript_path,
    )

    required = {
        "start_sec",
        "end_sec",
        "text",
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:

        raise ValueError(
            f"{video_id}: "
            f"missing columns "
            f"{sorted(missing)}"
        )

    df["start_sec"] = (
        pd.to_numeric(
            df["start_sec"],
            errors="coerce",
        )
    )

    df["end_sec"] = (
        pd.to_numeric(
            df["end_sec"],
            errors="coerce",
        )
    )

    df["text"] = (
        df["text"]
        .fillna("")
        .astype(str)
    )

    df = (
        df
        .dropna(
            subset=[
                "start_sec",
                "end_sec",
            ]
        )
    )

    if df.empty:

        raise ValueError(
            f"{video_id}: "
            "no valid transcript rows"
        )

    max_time = float(
        df[
            "end_sec"
        ].max()
    )

    rows = []

    chunk_id = 0
    start_time = 0.0

    while (
        start_time
        <= max_time
    ):

        end_time = (
            start_time
            + chunk_seconds
        )

        chunk_df = df[
            (
                df[
                    "start_sec"
                ]
                < end_time
            )
            &
            (
                df[
                    "end_sec"
                ]
                >= start_time
            )
        ]

        raw_text = " ".join(
            chunk_df[
                "text"
            ].tolist()
        )

        processed_text = (
            preprocess_text(
                raw_text
            )
        )

        if (
            processed_text
            .strip()
        ):

            rows.append(
                {
                    "video_id":
                        video_id,

                    "subject":
                        subject,

                    "chunk_seconds":
                        chunk_seconds,

                    "chunk_id":
                        chunk_id,

                    "start_sec":
                        start_time,

                    "end_sec":
                        end_time,

                    "processed_text":
                        processed_text,
                }
            )

        chunk_id += 1

        start_time += (
            chunk_seconds
        )

    return rows


def build_dev_chunks(
    videos,
    chunk_seconds,
):

    rows = []

    for _, video in (
        videos.iterrows()
    ):

        rows.extend(
            build_video_chunks(
                video_id=
                    video[
                        "video_id"
                    ],

                subject=
                    video[
                        "subject"
                    ],

                chunk_seconds=
                    chunk_seconds,
            )
        )

    chunks = pd.DataFrame(
        rows
    )

    if (
        chunks[
            "video_id"
        ].nunique()
        != 40
    ):

        raise ValueError(
            f"{chunk_seconds}s: "
            "not all 40 videos represented"
        )

    duplicates = (
        chunks.duplicated(
            subset=[
                "video_id",
                "chunk_id",
            ]
        ).sum()
    )

    if duplicates:

        raise ValueError(
            f"{chunk_seconds}s: "
            f"duplicate chunks "
            f"{duplicates}"
        )

    return chunks


# ============================================================
# SCORE ONE HELD-OUT OUTER FOLD
# ============================================================

def score_outer_fold(
    train_chunks,
    heldout_chunks,
    concepts,
    concept_texts,
    outer_fold,
):

    train_chunks = (
        train_chunks
        .copy()
        .reset_index(
            drop=True
        )
    )

    heldout_chunks = (
        heldout_chunks
        .copy()
        .reset_index(
            drop=True
        )
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

    # --------------------------------------------------------
    # LDA
    # TRAIN ONLY fit
    # HELD-OUT transform only
    # --------------------------------------------------------

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

    affinity = (
        calculate_topic_concept_affinity(
            lda_model=
                lda_model,

            vectorizer=
                lda_vectorizer,

            concept_texts=
                concept_texts,
        )
    )

    lda_matrix = (
        calculate_document_concept_scores(
            heldout_topic_matrix,
            affinity,
        )
    )

    # --------------------------------------------------------
    # LSA
    # TRAIN transcript + FIXED concepts fit
    # HELD-OUT transcript transform only
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

    heldout_tfidf = (
        lsa_vectorizer
        .transform(
            heldout_texts
        )
    )

    heldout_lsa = (
        svd.transform(
            heldout_tfidf
        )
    )

    lsa_matrix = (
        calculate_similarity(
            heldout_lsa,
            concept_vectors,
        )
    )

    # --------------------------------------------------------
    # SAME-SUBJECT SCORE TABLE
    # --------------------------------------------------------

    rows = []

    for chunk_index, chunk in (
        heldout_chunks
        .iterrows()
    ):

        subject = str(
            chunk[
                "subject"
            ]
        ).strip()

        for concept_index, concept in (
            concepts.iterrows()
        ):

            if (
                subject
                !=
                str(
                    concept[
                        "subject"
                    ]
                ).strip()
            ):

                continue

            # Exact frozen behavior:
            # raw component scores rounded
            # before normalization/fusion.
            lda_score = round(
                float(
                    lda_matrix[
                        chunk_index,
                        concept_index,
                    ]
                ),
                4,
            )

            lsa_score = round(
                float(
                    lsa_matrix[
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

                    "video_id":
                        chunk[
                            "video_id"
                        ],

                    "subject":
                        subject,

                    "chunk_id":
                        int(
                            chunk[
                                "chunk_id"
                            ]
                        ),

                    "concept":
                        concept[
                            "concept"
                        ],

                    "lda_score":
                        lda_score,

                    "lsa_score":
                        lsa_score,
                }
            )

    scores = pd.DataFrame(
        rows
    )

    scores[
        "lda_norm"
    ] = (
        scores
        .groupby(
            [
                "video_id",
                "chunk_id",
            ]
        )[
            "lda_score"
        ]
        .transform(
            normalize_series
        )
    )

    scores[
        "lsa_norm"
    ] = (
        scores
        .groupby(
            [
                "video_id",
                "chunk_id",
            ]
        )[
            "lsa_score"
        ]
        .transform(
            normalize_series
        )
    )

    return scores


# ============================================================
# FROZEN FUSION / VIDEO AGGREGATION
# ============================================================

def build_video_predictions(
    scores,
    positive_gt,
):

    work = scores.copy()

    work[
        "final_score"
    ] = (
        FROZEN_ALPHA
        * work[
            "lda_norm"
        ]
        +
        (
            1.0
            - FROZEN_ALPHA
        )
        * work[
            "lsa_norm"
        ]
    )

    work[
        "final_score"
    ] = (
        work[
            "final_score"
        ]
        .round(4)
    )

    # Frozen top2_mean.
    video_scores = (
        work
        .groupby(
            [
                "outer_fold",
                "video_id",
                "subject",
                "concept",
            ],
            sort=False,
        )[
            "final_score"
        ]
        .agg(
            lambda s:
                s.nlargest(2)
                .mean()
        )
        .reset_index(
            name=
                "video_score"
        )
    )

    video_scores[
        "video_score"
    ] = (
        video_scores[
            "video_score"
        ]
        .round(4)
    )

    video_scores[
        "y_true"
    ] = [
        1
        if (
            row.video_id,
            row.concept,
        )
        in positive_gt
        else 0
        for row in (
            video_scores
            .itertuples()
        )
    ]

    video_scores[
        "y_pred"
    ] = (
        video_scores[
            "video_score"
        ]
        >=
        FROZEN_THRESHOLD
    ).astype(int)

    return video_scores


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("P1.10 CHUNK-SIZE SENSITIVITY ANALYSIS")
    print("=" * 80)

    # --------------------------------------------------------
    # LOAD METADATA
    # --------------------------------------------------------

    videos = pd.read_csv(
        VIDEOS_PATH,
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

    reference = pd.read_csv(
        P18_PREDICTIONS_PATH,
    )

    videos[
        "video_id"
    ] = (
        videos[
            "video_id"
        ]
        .str.strip()
        .str.lower()
    )

    videos[
        "subject"
    ] = (
        videos[
            "subject"
        ]
        .str.strip()
    )

    videos[
        "split"
    ] = (
        videos[
            "split"
        ]
        .str.strip()
        .str.lower()
    )

    folds[
        "video_id"
    ] = (
        folds[
            "video_id"
        ]
        .str.strip()
        .str.lower()
    )

    folds[
        "subject"
    ] = (
        folds[
            "subject"
        ]
        .str.strip()
    )

    folds[
        "cv_fold"
    ] = (
        pd.to_numeric(
            folds[
                "cv_fold"
            ],
            errors="raise",
        )
        .astype(int)
    )

    concepts[
        "subject"
    ] = (
        concepts[
            "subject"
        ]
        .str.strip()
    )

    concepts[
        "concept"
    ] = (
        concepts[
            "concept"
        ]
        .str.strip()
    )

    concepts[
        "description"
    ] = (
        concepts[
            "description"
        ]
        .fillna("")
        .astype(str)
    )

    concepts = (
        concepts
        .reset_index(
            drop=True
        )
    )

    gt[
        "video_id"
    ] = (
        gt[
            "video_id"
        ]
        .str.strip()
        .str.lower()
    )

    gt[
        "concept"
    ] = (
        gt[
            "concept"
        ]
        .str.strip()
    )

    reference[
        "video_id"
    ] = (
        reference[
            "video_id"
        ]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    # --------------------------------------------------------
    # EXACT DEV SET
    # --------------------------------------------------------

    expected_ids = {
        f"v{i}"
        for i in range(
            1,
            41,
        )
    }

    dev = videos[
        videos[
            "video_id"
        ].isin(
            expected_ids
        )
        &
        videos[
            "split"
        ].eq(
            "dev"
        )
    ].copy()

    if (
        set(
            dev[
                "video_id"
            ]
        )
        != expected_ids
    ):

        raise ValueError(
            "DEV registry does not "
            "contain exactly v1-v40"
        )

    dev[
        "_video_num"
    ] = (
        dev[
            "video_id"
        ]
        .str.extract(
            r"(\d+)",
            expand=False,
        )
        .astype(int)
    )

    dev = (
        dev
        .sort_values(
            "_video_num"
        )
        .drop(
            columns=
                "_video_num"
        )
        .reset_index(
            drop=True
        )
    )

    if (
        folds[
            "video_id"
        ].nunique()
        != 40
    ):

        raise ValueError(
            "Fold manifest != 40 videos"
        )

    if len(
        concepts
    ) != 39:

        raise ValueError(
            "Concept catalog != 39"
        )

    positive_gt = set(
        zip(
            gt[
                "video_id"
            ],
            gt[
                "concept"
            ],
        )
    )

    concepts[
        "concept_text"
    ] = (
        concepts[
            "concept"
        ]
        + " "
        + concepts[
            "description"
        ]
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

    print("\nInput audit: PASS")
    print("DEV videos:", len(dev))
    print("Concepts  :", len(concepts))
    print("GT rows   :", len(gt))

    # --------------------------------------------------------
    # EXPERIMENT
    # --------------------------------------------------------

    all_predictions = []

    result_rows = []

    fold_metric_rows = []

    subject_metric_rows = []

    chunk_counts = {}

    SENSITIVITY_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for chunk_seconds in (
        CHUNK_SIZES
    ):

        print("\n" + "=" * 80)
        print(
            f"CHUNK SIZE = "
            f"{chunk_seconds} SECONDS"
        )
        print("=" * 80)

        chunks = (
            build_dev_chunks(
                videos=dev,
                chunk_seconds=
                    chunk_seconds,
            )
        )

        chunk_counts[
            str(
                chunk_seconds
            )
        ] = int(
            len(
                chunks
            )
        )

        chunks.to_csv(
            SENSITIVITY_DIR
            / (
                f"dev_chunks_"
                f"{chunk_seconds}s.csv"
            ),
            index=False,
            encoding="utf-8-sig",
        )

        print(
            "Total chunks:",
            len(chunks),
        )

        # Critical reproduction check.
        if (
            chunk_seconds
            == PRIMARY_CHUNK_SIZE
            and len(chunks)
            != 494
        ):

            raise ValueError(
                "60s chunk builder failed "
                "to reproduce 494 chunks"
            )

        score_parts = []

        # ----------------------------------------------------
        # SAME 5 OUTER FOLDS
        # ----------------------------------------------------

        for outer_fold in range(
            1,
            6,
        ):

            heldout_ids = set(
                folds.loc[
                    folds[
                        "cv_fold"
                    ].eq(
                        outer_fold
                    ),
                    "video_id",
                ]
            )

            all_ids = set(
                folds[
                    "video_id"
                ]
            )

            train_ids = (
                all_ids
                - heldout_ids
            )

            if (
                len(
                    train_ids
                )
                != 32
                or
                len(
                    heldout_ids
                )
                != 8
            ):

                raise ValueError(
                    f"{chunk_seconds}s "
                    f"fold {outer_fold}: "
                    "invalid video split"
                )

            if (
                train_ids
                & heldout_ids
            ):

                raise ValueError(
                    f"{chunk_seconds}s "
                    f"fold {outer_fold}: "
                    "video leakage"
                )

            train_chunks = (
                chunks[
                    chunks[
                        "video_id"
                    ].isin(
                        train_ids
                    )
                ]
                .copy()
            )

            heldout_chunks = (
                chunks[
                    chunks[
                        "video_id"
                    ].isin(
                        heldout_ids
                    )
                ]
                .copy()
            )

            fold_scores = (
                score_outer_fold(
                    train_chunks=
                        train_chunks,

                    heldout_chunks=
                        heldout_chunks,

                    concepts=
                        concepts,

                    concept_texts=
                        concept_texts,

                    outer_fold=
                        outer_fold,
                )
            )

            score_parts.append(
                fold_scores
            )

            print(
                f"Fold {outer_fold}: "
                f"{len(train_chunks)} "
                "train chunks / "
                f"{len(heldout_chunks)} "
                "held-out chunks — PASS"
            )

        scores = pd.concat(
            score_parts,
            ignore_index=True,
        )

        predictions = (
            build_video_predictions(
                scores=
                    scores,

                positive_gt=
                    positive_gt,
            )
        )

        predictions[
            "chunk_seconds"
        ] = (
            chunk_seconds
        )

        # ----------------------------------------------------
        # GLOBAL AUDIT
        # ----------------------------------------------------

        if (
            predictions[
                "video_id"
            ].nunique()
            != 40
        ):

            raise ValueError(
                f"{chunk_seconds}s: "
                "prediction coverage != 40"
            )

        if len(
            predictions
        ) != 390:

            raise ValueError(
                f"{chunk_seconds}s: "
                f"expected 390 "
                "video-concept pairs, "
                f"found {len(predictions)}"
            )

        duplicates = int(
            predictions
            .duplicated(
                subset=[
                    "video_id",
                    "concept",
                ]
            )
            .sum()
        )

        if duplicates:

            raise ValueError(
                f"{chunk_seconds}s: "
                "duplicate predictions"
            )

        if int(
            predictions[
                "y_true"
            ].sum()
        ) != len(
            positive_gt
        ):

            raise ValueError(
                f"{chunk_seconds}s: "
                "GT count mismatch"
            )

        # ----------------------------------------------------
        # PER-VIDEO METRICS
        # ----------------------------------------------------

        video_rows = []

        for (
            video_id,
            subject,
        ), group in (
            predictions
            .groupby(
                [
                    "video_id",
                    "subject",
                ]
            )
        ):

            metric = (
                calculate_metrics(
                    group[
                        "y_true"
                    ],
                    group[
                        "y_pred"
                    ],
                )
            )

            video_rows.append(
                {
                    "video_id":
                        video_id,

                    "subject":
                        subject,

                    **metric,
                }
            )

        video_metrics = (
            pd.DataFrame(
                video_rows
            )
        )

        micro = (
            calculate_metrics(
                predictions[
                    "y_true"
                ],
                predictions[
                    "y_pred"
                ],
            )
        )

        result_rows.append(
            {
                "chunk_seconds":
                    chunk_seconds,

                "num_chunks":
                    len(chunks),

                "macro_video_precision":
                    float(
                        video_metrics[
                            "precision"
                        ].mean()
                    ),

                "macro_video_recall":
                    float(
                        video_metrics[
                            "recall"
                        ].mean()
                    ),

                "macro_video_f1":
                    float(
                        video_metrics[
                            "f1"
                        ].mean()
                    ),

                "micro_precision":
                    micro[
                        "precision"
                    ],

                "micro_recall":
                    micro[
                        "recall"
                    ],

                "micro_f1":
                    micro[
                        "f1"
                    ],

                "alpha":
                    FROZEN_ALPHA,

                "aggregation":
                    FROZEN_AGGREGATION,

                "threshold":
                    FROZEN_THRESHOLD,

                "is_primary_chunk_size":
                    (
                        chunk_seconds
                        ==
                        PRIMARY_CHUNK_SIZE
                    ),
            }
        )

        # ----------------------------------------------------
        # FOLD METRICS
        # ----------------------------------------------------

        for outer_fold, group in (
            predictions.groupby(
                "outer_fold"
            )
        ):

            fold_video_ids = (
                group[
                    "video_id"
                ].unique()
            )

            pv = (
                video_metrics[
                    video_metrics[
                        "video_id"
                    ].isin(
                        fold_video_ids
                    )
                ]
            )

            fold_micro = (
                calculate_metrics(
                    group[
                        "y_true"
                    ],
                    group[
                        "y_pred"
                    ],
                )
            )

            fold_metric_rows.append(
                {
                    "chunk_seconds":
                        chunk_seconds,

                    "outer_fold":
                        int(
                            outer_fold
                        ),

                    "heldout_videos":
                        len(
                            fold_video_ids
                        ),

                    "macro_video_f1":
                        float(
                            pv[
                                "f1"
                            ].mean()
                        ),

                    "micro_f1":
                        fold_micro[
                            "f1"
                        ],
                }
            )

        # ----------------------------------------------------
        # SUBJECT METRICS
        # ----------------------------------------------------

        for subject in [
            "SQL",
            "Python",
            "Java",
            "C++",
        ]:

            pv = (
                video_metrics[
                    video_metrics[
                        "subject"
                    ].eq(
                        subject
                    )
                ]
            )

            subject_metric_rows.append(
                {
                    "chunk_seconds":
                        chunk_seconds,

                    "subject":
                        subject,

                    "n_videos":
                        len(pv),

                    "mean_precision":
                        float(
                            pv[
                                "precision"
                            ].mean()
                        ),

                    "mean_recall":
                        float(
                            pv[
                                "recall"
                            ].mean()
                        ),

                    "mean_f1":
                        float(
                            pv[
                                "f1"
                            ].mean()
                        ),
                }
            )

        all_predictions.append(
            predictions
        )

        print(
            "Macro video F1:",
            round(
                result_rows[
                    -1
                ][
                    "macro_video_f1"
                ],
                6,
            ),
        )

    # --------------------------------------------------------
    # COMBINE
    # --------------------------------------------------------

    results = pd.DataFrame(
        result_rows
    )

    fold_results = pd.DataFrame(
        fold_metric_rows
    )

    subject_results = pd.DataFrame(
        subject_metric_rows
    )

    sensitivity_predictions = (
        pd.concat(
            all_predictions,
            ignore_index=True,
        )
    )

    # --------------------------------------------------------
    # CRITICAL 60s REPRODUCTION CHECK
    # --------------------------------------------------------

    reference_60 = (
        reference[
            reference[
                "model_variant"
            ].eq(
                "frozen_fixed"
            )
        ][
            [
                "video_id",
                "concept",
                "video_score",
                "y_true",
                "y_pred",
            ]
        ]
        .copy()
    )

    sensitivity_60 = (
        sensitivity_predictions[
            sensitivity_predictions[
                "chunk_seconds"
            ].eq(
                60
            )
        ][
            [
                "video_id",
                "concept",
                "video_score",
                "y_true",
                "y_pred",
            ]
        ]
        .copy()
    )

    comparison = (
        sensitivity_60
        .merge(
            reference_60,
            on=[
                "video_id",
                "concept",
            ],
            suffixes=(
                "_new",
                "_reference",
            ),
            validate=
                "one_to_one",
        )
    )

    if len(
        comparison
    ) != 390:

        raise ValueError(
            "60s reproduction "
            "comparison != 390 pairs"
        )

    if not (
        comparison[
            "y_true_new"
        ].astype(int)
        ==
        comparison[
            "y_true_reference"
        ].astype(int)
    ).all():

        raise ValueError(
            "60s y_true does not "
            "reproduce P1.8"
        )

    if not (
        comparison[
            "y_pred_new"
        ].astype(int)
        ==
        comparison[
            "y_pred_reference"
        ].astype(int)
    ).all():

        raise ValueError(
            "60s predictions do not "
            "reproduce P1.8 frozen baseline"
        )

    score_difference = (
        comparison[
            "video_score_new"
        ].astype(float)
        -
        comparison[
            "video_score_reference"
        ].astype(float)
    ).abs()

    max_score_difference = float(
        score_difference.max()
    )

    if (
        max_score_difference
        > 1e-8
    ):

        raise ValueError(
            "60s video scores differ "
            "from P1.8 frozen baseline: "
            f"max diff={max_score_difference}"
        )

    p18_frozen_f1 = (
        reference[
            reference[
                "model_variant"
            ].eq(
                "frozen_fixed"
            )
        ]
    )

    p18_video_f1 = []

    for _, group in (
        p18_frozen_f1
        .groupby(
            "video_id"
        )
    ):

        p18_video_f1.append(
            f1_score(
                group[
                    "y_true"
                ],
                group[
                    "y_pred"
                ],
                zero_division=0,
            )
        )

    p18_macro_f1 = float(
        np.mean(
            p18_video_f1
        )
    )

    sensitivity_60_f1 = float(
        results.loc[
            results[
                "chunk_seconds"
            ].eq(
                60
            ),
            "macro_video_f1",
        ].iloc[0]
    )

    if not np.isclose(
        p18_macro_f1,
        sensitivity_60_f1,
        atol=1e-12,
        rtol=0.0,
    ):

        raise ValueError(
            "60s macro F1 failed "
            "to reproduce P1.8"
        )

    # --------------------------------------------------------
    # DIFFERENCE FROM PRIMARY 60s
    # --------------------------------------------------------

    results[
        "delta_macro_f1_vs_60s"
    ] = (
        results[
            "macro_video_f1"
        ]
        -
        sensitivity_60_f1
    )

    # Important:
    # this is descriptive sensitivity only.
    #
    # We DO NOT promote the best chunk size.
    results[
        "used_for_model_selection"
    ] = False

    # --------------------------------------------------------
    # WRITE
    # --------------------------------------------------------

    results.to_csv(
        RESULT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    fold_results.to_csv(
        FOLD_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    subject_results.to_csv(
        SUBJECT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    sensitivity_predictions.to_csv(
        PREDICTION_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    audit = {
        "analysis_stage":
            "P1.10",

        "purpose":
            (
                "chunk-size sensitivity "
                "analysis"
            ),

        "chunk_sizes_seconds":
            CHUNK_SIZES,

        "primary_pre_specified_chunk_size":
            PRIMARY_CHUNK_SIZE,

        "only_factor_changed":
            "chunk duration",

        "fixed_alpha":
            FROZEN_ALPHA,

        "fixed_aggregation":
            FROZEN_AGGREGATION,

        "fixed_threshold":
            FROZEN_THRESHOLD,

        "same_outer_folds":
            True,

        "fit_policy":
            (
                "outer-train fit; "
                "outer-held-out transform only"
            ),

        "heldout_text_used_in_fit":
            False,

        "used_for_hyperparameter_selection":
            False,

        "best_chunk_size_promoted":
            False,

        "chunk_counts":
            chunk_counts,

        "60s_expected_chunks":
            494,

        "60s_reproduces_p18_frozen_predictions":
            True,

        "60s_max_video_score_difference":
            max_score_difference,

        "60s_p18_macro_video_f1":
            p18_macro_f1,

        "60s_sensitivity_macro_video_f1":
            sensitivity_60_f1,
    }

    AUDIT_PATH.write_text(
        json.dumps(
            audit,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # REPORT
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("CHUNK-SIZE RESULTS")
    print("=" * 80)

    print(
        results[
            [
                "chunk_seconds",
                "num_chunks",
                "macro_video_precision",
                "macro_video_recall",
                "macro_video_f1",
                "micro_f1",
                "delta_macro_f1_vs_60s",
            ]
        ].to_string(
            index=False
        )
    )

    print("\n" + "=" * 80)
    print("60-SECOND REPRODUCTION AUDIT")
    print("=" * 80)

    print(
        "Expected 60s chunks       : 494"
    )

    print(
        "Actual 60s chunks         :",
        chunk_counts["60"],
    )

    print(
        "390 predictions reproduced:",
        True,
    )

    print(
        "Max video-score difference:",
        max_score_difference,
    )

    print(
        "P1.8 frozen macro F1      :",
        round(
            p18_macro_f1,
            6,
        ),
    )

    print(
        "Sensitivity 60s macro F1  :",
        round(
            sensitivity_60_f1,
            6,
        ),
    )

    print("\n" + "=" * 80)
    print(
        "P1.10 CHUNK-SIZE "
        "SENSITIVITY: PASS"
    )
    print("=" * 80)

    print(
        "Only chunk duration was varied."
    )

    print(
        "No chunk size was selected "
        "using these sensitivity results."
    )

    print(
        "60-second primary experiment "
        "was reproduced exactly."
    )

    print(
        "Held-out transcript text "
        "was never used for fitting."
    )

    print(
        "Saved:",
        RESULT_PATH.relative_to(
            ROOT
        ),
    )

    print(
        "Saved:",
        FOLD_PATH.relative_to(
            ROOT
        ),
    )

    print(
        "Saved:",
        SUBJECT_PATH.relative_to(
            ROOT
        ),
    )

    print(
        "Saved:",
        PREDICTION_PATH.relative_to(
            ROOT
        ),
    )

    print(
        "Saved:",
        AUDIT_PATH.relative_to(
            ROOT
        ),
    )

    print(
        "Derived chunks saved under:",
        SENSITIVITY_DIR.relative_to(
            ROOT
        ),
    )


if __name__ == "__main__":
    main()