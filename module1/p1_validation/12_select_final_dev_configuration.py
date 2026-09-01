from pathlib import Path
import importlib.util
import json

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

P1_DIR = (
    ROOT
    / "module1"
    / "p1_validation"
)

RESULT_DIR = (
    ROOT
    / "module1"
    / "results"
    / "p1_validation"
)

CHUNK_DIR = (
    RESULT_DIR
    / "chunk_sensitivity"
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

RAW_60_PATH = (
    RESULT_DIR
    / "oof_chunk_concept_scores.csv"
)

RAW_OUTPUT_PATH = (
    RESULT_DIR
    / "final_dev_raw_oof_scores.csv"
)

SEARCH_OUTPUT_PATH = (
    RESULT_DIR
    / "final_dev_search_results.csv"
)

SUMMARY_OUTPUT_PATH = (
    RESULT_DIR
    / "final_dev_selection_summary.csv"
)

CONFIG_OUTPUT_PATH = (
    RESULT_DIR
    / "final_dev_hyperparameters.json"
)

AUDIT_OUTPUT_PATH = (
    RESULT_DIR
    / "final_dev_selection_audit.json"
)


# ============================================================
# FINAL DEV SEARCH SPACE
# ============================================================

CHUNK_SECONDS = [
    30,
    60,
    90,
    120,
]

FOLDS = [
    1,
    2,
    3,
    4,
    5,
]


# ============================================================
# LOAD EXISTING VALIDATED MODULES
# ============================================================

def load_python_module(
    name,
    path,
):

    spec = (
        importlib.util
        .spec_from_file_location(
            name,
            path,
        )
    )

    if (
        spec is None
        or spec.loader is None
    ):
        raise ImportError(
            f"Cannot import {path}"
        )

    module = (
        importlib.util
        .module_from_spec(
            spec
        )
    )

    spec.loader.exec_module(
        module
    )

    return module


SENSITIVITY = load_python_module(
    "p1_chunk_sensitivity",
    P1_DIR
    / "10_chunk_size_sensitivity.py",
)

TUNING = load_python_module(
    "p1_nested_tuning",
    P1_DIR
    / "07_nested_inner_tuning.py",
)


# Reuse EXACT P1.7 search grid.
ALPHAS = list(
    TUNING.ALPHAS
)

AGGREGATIONS = list(
    TUNING.AGGREGATIONS
)

THRESHOLDS = list(
    TUNING.THRESHOLDS
)


# ============================================================
# HELPERS
# ============================================================

def normalize_video_id(series):

    return (
        series
        .astype(str)
        .str.strip()
        .str.lower()
    )


def require_file(path):

    if not path.exists():

        raise FileNotFoundError(
            f"Missing required file: "
            f"{path.relative_to(ROOT)}"
        )


def find_chunk_file(
    chunk_seconds,
):

    expected_names = [
        f"dev_chunks_{chunk_seconds}s.csv",
        f"dev_chunks_{chunk_seconds}.csv",
        f"chunks_{chunk_seconds}s.csv",
        f"chunks_{chunk_seconds}.csv",
    ]

    for name in expected_names:

        candidate = (
            CHUNK_DIR
            / name
        )

        if candidate.exists():
            return candidate

    required_columns = {
        "video_id",
        "subject",
        "chunk_id",
        "processed_text",
    }

    matches = []

    if CHUNK_DIR.exists():

        for path in (
            CHUNK_DIR
            .rglob("*.csv")
        ):

            if (
                str(chunk_seconds)
                not in path.stem
            ):
                continue

            try:

                columns = set(
                    pd.read_csv(
                        path,
                        nrows=2,
                    ).columns
                )

            except Exception:
                continue

            if (
                required_columns
                .issubset(columns)
            ):
                matches.append(
                    path
                )

    if len(matches) == 1:
        return matches[0]

    if len(matches) == 0:

        raise FileNotFoundError(
            f"Could not locate "
            f"{chunk_seconds}s chunk CSV "
            f"inside "
            f"{CHUNK_DIR.relative_to(ROOT)}"
        )

    raise ValueError(
        f"Ambiguous {chunk_seconds}s "
        f"chunk files:\n"
        +
        "\n".join(
            str(
                p.relative_to(ROOT)
            )
            for p in matches
        )
    )


def load_chunks(
    path,
):

    chunks = pd.read_csv(
        path,
    )

    required = {
        "video_id",
        "subject",
        "chunk_id",
        "processed_text",
    }

    missing = (
        required
        - set(chunks.columns)
    )

    if missing:

        raise ValueError(
            f"{path.name}: "
            f"missing columns "
            f"{sorted(missing)}"
        )

    chunks["video_id"] = (
        normalize_video_id(
            chunks["video_id"]
        )
    )

    chunks["subject"] = (
        chunks["subject"]
        .astype(str)
        .str.strip()
    )

    chunks[
        "processed_text"
    ] = (
        chunks[
            "processed_text"
        ]
        .fillna("")
        .astype(str)
    )

    duplicates = int(
        chunks.duplicated(
            subset=[
                "video_id",
                "chunk_id",
            ]
        ).sum()
    )

    if duplicates != 0:

        raise ValueError(
            f"{path.name}: "
            f"{duplicates} duplicate "
            f"video/chunk pairs"
        )

    return chunks


# ============================================================
# BUILD RAW LEAKAGE-FREE OOF SCORES
# ============================================================

def build_raw_scores(
    chunk_seconds,
    chunks,
    folds,
    concepts,
    concept_texts,
):

    score_parts = []

    for outer_fold in FOLDS:

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

        train_ids = set(
            folds.loc[
                ~folds[
                    "cv_fold"
                ].eq(
                    outer_fold
                ),
                "video_id",
            ]
        )

        if (
            heldout_ids
            & train_ids
        ):

            raise ValueError(
                f"{chunk_seconds}s "
                f"fold {outer_fold}: "
                "train/test overlap"
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

        if (
            train_chunks.empty
            or heldout_chunks.empty
        ):

            raise ValueError(
                f"{chunk_seconds}s "
                f"fold {outer_fold}: "
                "empty train or heldout set"
            )

        print(
            f"{chunk_seconds}s | "
            f"fold {outer_fold} | "
            f"train chunks "
            f"{len(train_chunks)} | "
            f"heldout chunks "
            f"{len(heldout_chunks)}"
        )

        scores = (
            SENSITIVITY
            .score_outer_fold(
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

        scores[
            "chunk_seconds"
        ] = chunk_seconds

        score_parts.append(
            scores
        )

    combined = pd.concat(
        score_parts,
        ignore_index=True,
    )

    combined[
        "video_id"
    ] = normalize_video_id(
        combined[
            "video_id"
        ]
    )

    if (
        combined[
            "video_id"
        ].nunique()
        != 40
    ):

        raise ValueError(
            f"{chunk_seconds}s: "
            "raw OOF scores do not "
            "cover 40 videos"
        )

    assignment_counts = (
        combined[
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
        assignment_counts.eq(1)
        .all()
    ):

        raise ValueError(
            f"{chunk_seconds}s: "
            "a video appears in "
            "multiple held-out folds"
        )

    return combined


# ============================================================
# EVALUATE ONE CONFIGURATION
# ============================================================

def evaluate_configuration(
    scores,
    positive_gt,
    alpha,
    aggregation,
    threshold,
):

    aggregated = (
        TUNING
        .aggregate_scores(
            scores,
            alpha,
            aggregation,
        )
    )

    overall = (
        TUNING
        .evaluate_candidate(
            aggregated,
            positive_gt,
            threshold,
        )
    )

    fold_f1s = []

    for fold in FOLDS:

        fold_scores = (
            scores[
                scores[
                    "outer_fold"
                ].eq(
                    fold
                )
            ]
            .copy()
        )

        fold_aggregated = (
            TUNING
            .aggregate_scores(
                fold_scores,
                alpha,
                aggregation,
            )
        )

        fold_metrics = (
            TUNING
            .evaluate_candidate(
                fold_aggregated,
                positive_gt,
                threshold,
            )
        )

        fold_f1s.append(
            float(
                fold_metrics[
                    "macro_video_f1"
                ]
            )
        )

    fold_std = float(
        np.std(
            fold_f1s,
            ddof=1,
        )
    )

    return (
        overall,
        fold_f1s,
        fold_std,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 82)
    print(
        "P1.12 FINAL DEV "
        "CONFIGURATION SELECTION"
    )
    print("=" * 82)

    required_files = [
        FOLDS_PATH,
        CONCEPT_PATH,
        GT_PATH,
        RAW_60_PATH,
    ]

    for path in required_files:
        require_file(path)

    # --------------------------------------------------------
    # Dataset metadata
    # --------------------------------------------------------

    folds = pd.read_csv(
        FOLDS_PATH,
    )

    folds[
        "video_id"
    ] = normalize_video_id(
        folds[
            "video_id"
        ]
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

    if (
        len(folds) != 40
        or folds[
            "video_id"
        ].nunique() != 40
    ):

        raise ValueError(
            "Fold manifest must contain "
            "40 unique DEV videos"
        )

    if (
        set(
            folds[
                "cv_fold"
            ].unique()
        )
        != set(FOLDS)
    ):

        raise ValueError(
            "Expected CV folds 1..5"
        )

    # --------------------------------------------------------
    # Concepts
    # --------------------------------------------------------

    concepts = (
        pd.read_csv(
            CONCEPT_PATH,
        )
        .reset_index(
            drop=True
        )
    )

    concepts[
        "subject"
    ] = (
        concepts[
            "subject"
        ]
        .astype(str)
        .str.strip()
    )

    concepts[
        "concept"
    ] = (
        concepts[
            "concept"
        ]
        .astype(str)
        .str.strip()
    )

    if len(concepts) != 39:

        raise ValueError(
            f"Expected 39 concepts, "
            f"found {len(concepts)}"
        )

    # Same concept-text preprocessing
    # already used by the validated P1 code.
    concept_texts = [
        SENSITIVITY
        .preprocess_text(
            concept
        )
        for concept in (
            concepts[
                "concept"
            ].tolist()
        )
    ]

    # --------------------------------------------------------
    # Positive-only Ground Truth
    # --------------------------------------------------------

    gt = pd.read_csv(
        GT_PATH,
    )

    gt[
        "video_id"
    ] = normalize_video_id(
        gt[
            "video_id"
        ]
    )

    gt[
        "concept"
    ] = (
        gt[
            "concept"
        ]
        .astype(str)
        .str.strip()
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

    if len(positive_gt) != 153:

        raise ValueError(
            f"Expected 153 positive "
            f"GT pairs, "
            f"found {len(positive_gt)}"
        )

    # ========================================================
    # 1. RAW OOF SCORE GENERATION
    # ========================================================

    print(
        "\n1. BUILD / REUSE "
        "LEAKAGE-FREE RAW SCORES"
    )

    raw_by_size = {}

    # --------------------------------------------------------
    # 60s:
    # Reuse P1.6 validated raw OOF scores.
    # --------------------------------------------------------

    raw_60 = pd.read_csv(
        RAW_60_PATH,
    )

    raw_60[
        "video_id"
    ] = normalize_video_id(
        raw_60[
            "video_id"
        ]
    )

    raw_60[
        "outer_fold"
    ] = (
        pd.to_numeric(
            raw_60[
                "outer_fold"
            ],
            errors="raise",
        )
        .astype(int)
    )

    required_score_columns = {
        "outer_fold",
        "video_id",
        "subject",
        "chunk_id",
        "concept",
        "lda_score",
        "lsa_score",
        "lda_norm",
        "lsa_norm",
    }

    missing = (
        required_score_columns
        - set(
            raw_60.columns
        )
    )

    if missing:

        raise ValueError(
            "P1.6 60s raw score file "
            f"is missing "
            f"{sorted(missing)}"
        )

    if (
        raw_60[
            "video_id"
        ].nunique()
        != 40
    ):

        raise ValueError(
            "P1.6 60s raw scores "
            "do not cover 40 videos"
        )

    raw_60[
        "chunk_seconds"
    ] = 60

    raw_by_size[
        60
    ] = raw_60.copy()

    print(
        "60s | reused P1.6 raw OOF "
        f"scores | rows={len(raw_60)}"
    )

    # --------------------------------------------------------
    # 30/90/120:
    # Generate raw scores using the validated
    # P1.10 leakage-free score_outer_fold().
    # --------------------------------------------------------

    for chunk_seconds in [
        30,
        90,
        120,
    ]:

        chunk_path = (
            find_chunk_file(
                chunk_seconds
            )
        )

        print(
            f"\n{chunk_seconds}s chunks: "
            f"{chunk_path.relative_to(ROOT)}"
        )

        chunks = load_chunks(
            chunk_path
        )

        if (
            chunks[
                "video_id"
            ].nunique()
            != 40
        ):

            raise ValueError(
                f"{chunk_seconds}s chunks "
                "do not cover all "
                "40 DEV videos"
            )

        raw_scores = (
            build_raw_scores(
                chunk_seconds=
                    chunk_seconds,

                chunks=
                    chunks,

                folds=
                    folds,

                concepts=
                    concepts,

                concept_texts=
                    concept_texts,
            )
        )

        raw_by_size[
            chunk_seconds
        ] = raw_scores

        print(
            f"{chunk_seconds}s raw OOF rows: "
            f"{len(raw_scores)}"
        )

    # Save all raw scores together.
    raw_all = pd.concat(
        [
            raw_by_size[
                size
            ].copy()
            for size in CHUNK_SECONDS
        ],
        ignore_index=True,
        sort=False,
    )

    raw_all.to_csv(
        RAW_OUTPUT_PATH,
        index=False,
    )

    # ========================================================
    # 2. JOINT CONFIGURATION SEARCH
    # ========================================================

    print(
        "\n2. JOINT CONFIGURATION SEARCH"
    )

    total_expected = (
        len(CHUNK_SECONDS)
        * len(ALPHAS)
        * len(AGGREGATIONS)
        * len(THRESHOLDS)
    )

    print(
        f"Chunk sizes     : "
        f"{CHUNK_SECONDS}"
    )

    print(
        f"Alphas          : "
        f"{len(ALPHAS)}"
    )

    print(
        f"Aggregations    : "
        f"{AGGREGATIONS}"
    )

    print(
        f"Thresholds      : "
        f"{len(THRESHOLDS)}"
    )

    print(
        f"Expected configs: "
        f"{total_expected}"
    )

    search_rows = []

    deterministic_order = 0

    for chunk_seconds in (
        CHUNK_SECONDS
    ):

        scores = (
            raw_by_size[
                chunk_seconds
            ]
        )

        print(
            f"\nSearching "
            f"{chunk_seconds}s..."
        )

        for alpha in ALPHAS:

            for aggregation in (
                AGGREGATIONS
            ):

                # Aggregate once for this
                # chunk/alpha/aggregation.
                aggregated_all = (
                    TUNING
                    .aggregate_scores(
                        scores,
                        alpha,
                        aggregation,
                    )
                )

                aggregated_by_fold = {}

                for fold in FOLDS:

                    fold_scores = (
                        scores[
                            scores[
                                "outer_fold"
                            ].eq(
                                fold
                            )
                        ]
                    )

                    aggregated_by_fold[
                        fold
                    ] = (
                        TUNING
                        .aggregate_scores(
                            fold_scores,
                            alpha,
                            aggregation,
                        )
                    )

                for threshold in (
                    THRESHOLDS
                ):

                    overall = (
                        TUNING
                        .evaluate_candidate(
                            aggregated_all,
                            positive_gt,
                            threshold,
                        )
                    )

                    fold_f1s = []

                    for fold in FOLDS:

                        fold_metrics = (
                            TUNING
                            .evaluate_candidate(
                                aggregated_by_fold[
                                    fold
                                ],
                                positive_gt,
                                threshold,
                            )
                        )

                        fold_f1s.append(
                            float(
                                fold_metrics[
                                    "macro_video_f1"
                                ]
                            )
                        )

                    fold_std = float(
                        np.std(
                            fold_f1s,
                            ddof=1,
                        )
                    )

                    search_rows.append(
                        {
                            "chunk_seconds":
                                chunk_seconds,

                            "alpha":
                                float(alpha),

                            "aggregation":
                                aggregation,

                            "threshold":
                                float(
                                    threshold
                                ),

                            "macro_video_precision":
                                overall[
                                    "macro_video_precision"
                                ],

                            "macro_video_recall":
                                overall[
                                    "macro_video_recall"
                                ],

                            "macro_video_f1":
                                overall[
                                    "macro_video_f1"
                                ],

                            "micro_precision":
                                overall[
                                    "micro_precision"
                                ],

                            "micro_recall":
                                overall[
                                    "micro_recall"
                                ],

                            "micro_f1":
                                overall[
                                    "micro_f1"
                                ],

                            "fold1_macro_f1":
                                fold_f1s[0],

                            "fold2_macro_f1":
                                fold_f1s[1],

                            "fold3_macro_f1":
                                fold_f1s[2],

                            "fold4_macro_f1":
                                fold_f1s[3],

                            "fold5_macro_f1":
                                fold_f1s[4],

                            "fold_f1_std":
                                fold_std,

                            "num_video_concept_pairs":
                                overall[
                                    "num_video_concept_pairs"
                                ],

                            "num_true_positive_labels":
                                overall[
                                    "num_true_positive_labels"
                                ],

                            "num_predicted_positive_labels":
                                overall[
                                    "num_predicted_positive_labels"
                                ],

                            "deterministic_order":
                                deterministic_order,
                        }
                    )

                    deterministic_order += 1

    search = pd.DataFrame(
        search_rows
    )

    if len(search) != total_expected:

        raise ValueError(
            f"Expected "
            f"{total_expected} configs, "
            f"found {len(search)}"
        )

    # ========================================================
    # 3. SELECT EXACTLY ONE FINAL DEV CONFIG
    # ========================================================

    # Selection rule:
    #
    # 1. Highest macro video-level F1
    # 2. Highest micro F1
    # 3. Lowest fold-to-fold F1 std
    # 4. Deterministic grid order ONLY
    #
    # No preference for:
    # - 60s
    # - alpha 0.4
    # - top2_mean
    # - threshold 0.4

    ranked = (
        search
        .sort_values(
            by=[
                "macro_video_f1",
                "micro_f1",
                "fold_f1_std",
                "deterministic_order",
            ],
            ascending=[
                False,
                False,
                True,
                True,
            ],
            kind="mergesort",
        )
        .reset_index(
            drop=True
        )
    )

    selected = (
        ranked.iloc[0]
    )

    search[
        "selected_final_dev_config"
    ] = False

    selected_mask = (
        search[
            "deterministic_order"
        ].eq(
            int(
                selected[
                    "deterministic_order"
                ]
            )
        )
    )

    search.loc[
        selected_mask,
        "selected_final_dev_config",
    ] = True

    search.to_csv(
        SEARCH_OUTPUT_PATH,
        index=False,
    )

    # ========================================================
    # 4. FROZEN BASELINE REFERENCE
    # ========================================================

    baseline_rows = search[
        search[
            "chunk_seconds"
        ].eq(60)
        &
        np.isclose(
            search["alpha"],
            0.4,
        )
        &
        search[
            "aggregation"
        ].eq(
            "top2_mean"
        )
        &
        np.isclose(
            search[
                "threshold"
            ],
            0.4,
        )
    ]

    if len(baseline_rows) != 1:

        raise ValueError(
            "Could not uniquely identify "
            "60s frozen baseline config"
        )

    baseline = (
        baseline_rows.iloc[0]
    )

    # ========================================================
    # 5. SAVE FINAL CONFIG
    # ========================================================

    config = {
        "stage":
            "P1.12 Final DEV "
            "Configuration Selection",

        "status":
            "SELECTED_ON_DEV",

        "chunk_seconds":
            int(
                selected[
                    "chunk_seconds"
                ]
            ),

        "alpha":
            float(
                selected[
                    "alpha"
                ]
            ),

        "lsa_weight":
            float(
                1.0
                -
                float(
                    selected[
                        "alpha"
                    ]
                )
            ),

        "aggregation":
            str(
                selected[
                    "aggregation"
                ]
            ),

        "threshold":
            float(
                selected[
                    "threshold"
                ]
            ),

        "selection_metric":
            "mean_video_level_f1",

        "selected_macro_video_f1":
            float(
                selected[
                    "macro_video_f1"
                ]
            ),

        "selected_micro_f1":
            float(
                selected[
                    "micro_f1"
                ]
            ),

        "selected_fold_f1_std":
            float(
                selected[
                    "fold_f1_std"
                ]
            ),

        "search_space": {
            "chunk_seconds":
                CHUNK_SECONDS,

            "alphas":
                ALPHAS,

            "aggregations":
                AGGREGATIONS,

            "thresholds":
                THRESHOLDS,

            "num_configurations":
                int(
                    total_expected
                ),
        },

        "selection_rule": [
            "highest macro video F1",
            "highest micro F1",
            "lowest fold F1 std",
            "deterministic grid order",
        ],

        "baseline_preference_used":
            False,

        "selected_using":
            "40-video DEV only",

        "final_test_used":
            False,

        "important_note":
            (
                "The selected DEV-CV "
                "performance is a "
                "model-selection score, "
                "not an unbiased final "
                "test estimate."
            ),
    }

    CONFIG_OUTPUT_PATH.write_text(
        json.dumps(
            config,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # ========================================================
    # 6. HUMAN-READABLE SUMMARY TABLE
    # ========================================================

    summary = pd.DataFrame(
        [
            {
                "variant":
                    "final_dev_selected",

                "chunk_seconds":
                    int(
                        selected[
                            "chunk_seconds"
                        ]
                    ),

                "alpha":
                    float(
                        selected[
                            "alpha"
                        ]
                    ),

                "aggregation":
                    selected[
                        "aggregation"
                    ],

                "threshold":
                    float(
                        selected[
                            "threshold"
                        ]
                    ),

                "macro_video_f1":
                    float(
                        selected[
                            "macro_video_f1"
                        ]
                    ),

                "micro_f1":
                    float(
                        selected[
                            "micro_f1"
                        ]
                    ),

                "fold_f1_std":
                    float(
                        selected[
                            "fold_f1_std"
                        ]
                    ),
            },

            {
                "variant":
                    "frozen_baseline_60s",

                "chunk_seconds":
                    60,

                "alpha":
                    0.4,

                "aggregation":
                    "top2_mean",

                "threshold":
                    0.4,

                "macro_video_f1":
                    float(
                        baseline[
                            "macro_video_f1"
                        ]
                    ),

                "micro_f1":
                    float(
                        baseline[
                            "micro_f1"
                        ]
                    ),

                "fold_f1_std":
                    float(
                        baseline[
                            "fold_f1_std"
                        ]
                    ),
            },
        ]
    )

    summary[
        "delta_macro_f1_vs_frozen"
    ] = (
        summary[
            "macro_video_f1"
        ]
        -
        float(
            baseline[
                "macro_video_f1"
            ]
        )
    )

    summary.to_csv(
        SUMMARY_OUTPUT_PATH,
        index=False,
    )

    # ========================================================
    # 7. AUDIT
    # ========================================================

    raw_audit = {}

    for size in CHUNK_SECONDS:

        raw = (
            raw_by_size[
                size
            ]
        )

        raw_audit[
            str(size)
        ] = {
            "rows":
                int(len(raw)),

            "videos":
                int(
                    raw[
                        "video_id"
                    ].nunique()
                ),

            "folds":
                sorted(
                    int(x)
                    for x in (
                        raw[
                            "outer_fold"
                        ]
                        .unique()
                    )
                ),
        }

    audit = {
        "status":
            "PASS",

        "dev_videos":
            40,

        "cv_folds":
            FOLDS,

        "raw_score_audit":
            raw_audit,

        "num_search_configurations":
            int(
                len(search)
            ),

        "expected_search_configurations":
            int(
                total_expected
            ),

        "selected_config_count":
            int(
                search[
                    "selected_final_dev_config"
                ].sum()
            ),

        "primary_metric":
            "mean_video_level_f1",

        "final_test_used":
            False,

        "baseline_preference_used":
            False,

        "p_value_used_for_selection":
            False,

        "heldout_text_used_for_fit":
            False,

        "selected_configuration": {
            "chunk_seconds":
                int(
                    selected[
                        "chunk_seconds"
                    ]
                ),

            "alpha":
                float(
                    selected[
                        "alpha"
                    ]
                ),

            "aggregation":
                str(
                    selected[
                        "aggregation"
                    ]
                ),

            "threshold":
                float(
                    selected[
                        "threshold"
                    ]
                ),
        },
    }

    AUDIT_OUTPUT_PATH.write_text(
        json.dumps(
            audit,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print(
        "\n" + "=" * 82
    )

    print(
        "P1.12 FINAL DEV "
        "CONFIGURATION SELECTED"
    )

    print(
        "=" * 82
    )

    print(
        f"chunk       : "
        f"{int(selected['chunk_seconds'])}s"
    )

    print(
        f"alpha       : "
        f"{float(selected['alpha']):.1f}"
    )

    print(
        f"LSA weight  : "
        f"{1.0 - float(selected['alpha']):.1f}"
    )

    print(
        f"aggregation : "
        f"{selected['aggregation']}"
    )

    print(
        f"threshold   : "
        f"{float(selected['threshold']):.2f}"
    )

    print()

    print(
        f"DEV-CV macro video F1 : "
        f"{float(selected['macro_video_f1']):.6f}"
    )

    print(
        f"DEV-CV micro F1       : "
        f"{float(selected['micro_f1']):.6f}"
    )

    print(
        f"Fold F1 std           : "
        f"{float(selected['fold_f1_std']):.6f}"
    )

    print()

    print(
        "Frozen 60s baseline "
        f"macro F1 : "
        f"{float(baseline['macro_video_f1']):.6f}"
    )

    print(
        "DEV selection delta  : "
        f"{float(selected['macro_video_f1']) - float(baseline['macro_video_f1']):+.6f}"
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "This is the single "
        "configuration selected "
        "using DEV only."
    )

    print(
        "Its DEV-CV score is NOT "
        "an unbiased final-test result."
    )

    print(
        "Final TEST was not used."
    )

    print(
        "\nSaved:"
    )

    print(
        RAW_OUTPUT_PATH.relative_to(
            ROOT
        )
    )

    print(
        SEARCH_OUTPUT_PATH.relative_to(
            ROOT
        )
    )

    print(
        SUMMARY_OUTPUT_PATH.relative_to(
            ROOT
        )
    )

    print(
        CONFIG_OUTPUT_PATH.relative_to(
            ROOT
        )
    )

    print(
        AUDIT_OUTPUT_PATH.relative_to(
            ROOT
        )
    )

    print(
        "\nP1.12 FINAL DEV "
        "CONFIG SELECTION: PASS"
    )


if __name__ == "__main__":
    main()