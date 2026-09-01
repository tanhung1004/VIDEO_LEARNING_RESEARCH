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

EXTENSION_DIR = (
    RESULT_DIR
    / "boundary_extension"
)

EXTENSION_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

VIDEOS_PATH = (
    ROOT
    / "data"
    / "raw"
    / "videos.csv"
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

P112_RAW_PATH = (
    RESULT_DIR
    / "final_dev_raw_oof_scores.csv"
)

P112_CONFIG_PATH = (
    RESULT_DIR
    / "final_dev_hyperparameters.json"
)

P112_SEARCH_PATH = (
    RESULT_DIR
    / "final_dev_search_results.csv"
)

EXTENDED_RAW_PATH = (
    RESULT_DIR
    / "boundary_extended_raw_oof_scores.csv"
)

SEARCH_PATH = (
    RESULT_DIR
    / "boundary_extended_search_results.csv"
)

CONFIG_PATH = (
    RESULT_DIR
    / "boundary_extended_hyperparameters.json"
)

SUMMARY_PATH = (
    RESULT_DIR
    / "boundary_extension_summary.csv"
)

AUDIT_PATH = (
    RESULT_DIR
    / "boundary_extension_audit.json"
)


# ============================================================
# ONE-TIME EXTENSION POLICY
# ============================================================

OLD_CHUNKS = [
    30,
    60,
    90,
    120,
]

NEW_CHUNKS = [
    150,
    180,
]

ALL_CHUNKS = (
    OLD_CHUNKS
    + NEW_CHUNKS
)

FOLDS = [
    1,
    2,
    3,
    4,
    5,
]


# ============================================================
# LOAD VALIDATED MODULES
# ============================================================

def load_module(
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


SENSITIVITY = load_module(
    "p1_chunk_sensitivity",
    P1_DIR
    / "10_chunk_size_sensitivity.py",
)

TUNING = load_module(
    "p1_nested_tuning",
    P1_DIR
    / "07_nested_inner_tuning.py",
)


ALPHAS = [
    float(x)
    for x in TUNING.ALPHAS
]

AGGREGATIONS = list(
    TUNING.AGGREGATIONS
)

THRESHOLDS = sorted(
    set(
        [
            0.10,
            0.15,
        ]
        +
        [
            float(x)
            for x
            in TUNING.THRESHOLDS
        ]
    )
)


# ============================================================
# HELPERS
# ============================================================

def require_file(path):

    if not path.exists():

        raise FileNotFoundError(
            f"Missing file: "
            f"{path.relative_to(ROOT)}"
        )


def normalize_video_id(series):

    return (
        series
        .astype(str)
        .str.strip()
        .str.lower()
    )


def json_default(value):

    if isinstance(
        value,
        np.bool_,
    ):
        return bool(value)

    if isinstance(
        value,
        np.integer,
    ):
        return int(value)

    if isinstance(
        value,
        np.floating,
    ):
        return float(value)

    if isinstance(
        value,
        np.ndarray,
    ):
        return value.tolist()

    raise TypeError(
        f"{type(value).__name__} "
        "is not JSON serializable"
    )


def extract_chunks(
    result,
):

    if isinstance(
        result,
        pd.DataFrame,
    ):
        return result.copy()

    if isinstance(
        result,
        tuple,
    ):

        dataframes = [
            item
            for item in result
            if isinstance(
                item,
                pd.DataFrame,
            )
        ]

        if len(dataframes) == 1:
            return dataframes[0].copy()

    raise TypeError(
        "build_dev_chunks() "
        "did not return a unique "
        "DataFrame"
    )


def validate_chunks(
    chunks,
    chunk_seconds,
):

    required = {
        "video_id",
        "subject",
        "chunk_id",
        "processed_text",
    }

    missing = (
        required
        - set(
            chunks.columns
        )
    )

    if missing:

        raise ValueError(
            f"{chunk_seconds}s chunks "
            f"missing columns: "
            f"{sorted(missing)}"
        )

    chunks[
        "video_id"
    ] = normalize_video_id(
        chunks[
            "video_id"
        ]
    )

    chunks[
        "subject"
    ] = (
        chunks[
            "subject"
        ]
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

    if (
        chunks[
            "video_id"
        ].nunique()
        != 40
    ):

        raise ValueError(
            f"{chunk_seconds}s does not "
            "cover 40 DEV videos"
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
            f"{chunk_seconds}s has "
            f"{duplicates} duplicate "
            "video/chunk pairs"
        )

    return chunks


def validate_raw_scores(
    raw,
    chunk_seconds,
):

    required = {
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
        required
        - set(
            raw.columns
        )
    )

    if missing:

        raise ValueError(
            f"{chunk_seconds}s raw scores "
            f"missing {sorted(missing)}"
        )

    raw[
        "video_id"
    ] = normalize_video_id(
        raw[
            "video_id"
        ]
    )

    raw[
        "outer_fold"
    ] = (
        pd.to_numeric(
            raw[
                "outer_fold"
            ],
            errors="raise",
        )
        .astype(int)
    )

    if (
        raw[
            "video_id"
        ].nunique()
        != 40
    ):

        raise ValueError(
            f"{chunk_seconds}s raw scores "
            "do not cover 40 videos"
        )

    assignments = (
        raw[
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
        assignments.eq(1)
        .all()
    ):

        raise ValueError(
            f"{chunk_seconds}s video "
            "assigned to multiple "
            "held-out folds"
        )

    duplicate_scores = int(
        raw.duplicated(
            subset=[
                "outer_fold",
                "video_id",
                "chunk_id",
                "concept",
            ]
        ).sum()
    )

    if duplicate_scores != 0:

        raise ValueError(
            f"{chunk_seconds}s has "
            f"{duplicate_scores} "
            "duplicate score rows"
        )

    return raw


# ============================================================
# RAW OOF SCORING FOR NEW CHUNK SIZES
# ============================================================

def build_raw_oof_scores(
    chunks,
    chunk_seconds,
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

        overlap = (
            train_ids
            & heldout_ids
        )

        if overlap:

            raise ValueError(
                f"{chunk_seconds}s "
                f"fold {outer_fold}: "
                "video leakage detected"
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
                "empty train/heldout"
            )

        print(
            f"{chunk_seconds}s | "
            f"fold {outer_fold} | "
            f"train chunks "
            f"{len(train_chunks)} | "
            f"heldout chunks "
            f"{len(heldout_chunks)}"
        )

        fold_scores = (
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

        fold_scores[
            "chunk_seconds"
        ] = chunk_seconds

        score_parts.append(
            fold_scores
        )

    raw = pd.concat(
        score_parts,
        ignore_index=True,
        sort=False,
    )

    raw = validate_raw_scores(
        raw,
        chunk_seconds,
    )

    return raw


# ============================================================
# SEARCH HELPERS
# ============================================================

def evaluate_configuration(
    raw,
    positive_gt,
    alpha,
    aggregation,
    threshold,
):

    aggregated = (
        TUNING
        .aggregate_scores(
            raw,
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

        fold_raw = (
            raw[
                raw[
                    "outer_fold"
                ].eq(fold)
            ]
        )

        fold_aggregated = (
            TUNING
            .aggregate_scores(
                fold_raw,
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
        "P1.13b ONE-TIME "
        "BOUNDARY EXTENSION CHECK"
    )
    print("=" * 82)

    for path in [
        VIDEOS_PATH,
        FOLDS_PATH,
        CONCEPT_PATH,
        GT_PATH,
        P112_RAW_PATH,
        P112_CONFIG_PATH,
        P112_SEARCH_PATH,
    ]:
        require_file(path)

    # ========================================================
    # INPUT DATA
    # ========================================================

    videos = pd.read_csv(
        VIDEOS_PATH,
        dtype=str,
        keep_default_na=False,
    )

    videos[
        "video_id"
    ] = normalize_video_id(
        videos[
            "video_id"
        ]
    )

    videos[
        "subject"
    ] = (
        videos[
            "subject"
        ]
        .astype(str)
        .str.strip()
    )

    videos[
        "split"
    ] = (
        videos[
            "split"
        ]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    dev = (
        videos[
            videos[
                "split"
            ].eq("dev")
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    if (
        len(dev) != 40
        or dev[
            "video_id"
        ].nunique() != 40
    ):

        raise ValueError(
            "Expected exactly "
            "40 DEV videos"
        )

    folds = pd.read_csv(
        FOLDS_PATH
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
            "Expected 40 fold assignments"
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
            "Expected folds 1..5"
        )

    concepts = (
        pd.read_csv(
            CONCEPT_PATH
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
            "Expected 39 concepts"
        )

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

    gt = pd.read_csv(
        GT_PATH
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
            "Expected 153 "
            "positive GT pairs"
        )

    # ========================================================
    # ORIGINAL P1.12 CONFIG
    # ========================================================

    original_config = json.loads(
        P112_CONFIG_PATH.read_text(
            encoding="utf-8"
        )
    )

    original_chunk = int(
        original_config[
            "chunk_seconds"
        ]
    )

    original_alpha = float(
        original_config[
            "alpha"
        ]
    )

    original_aggregation = str(
        original_config[
            "aggregation"
        ]
    )

    original_threshold = float(
        original_config[
            "threshold"
        ]
    )

    original_search = pd.read_csv(
        P112_SEARCH_PATH
    )

    original_selected = (
        original_search[
            original_search[
                "selected_final_dev_config"
            ].astype(str)
            .str.strip()
            .str.lower()
            .eq("true")
        ]
    )

    if len(original_selected) != 1:

        raise ValueError(
            "Expected exactly one "
            "P1.12 selected config"
        )

    original_f1 = float(
        original_selected.iloc[0][
            "macro_video_f1"
        ]
    )

    print(
        "\nOriginal P1.12 candidate:"
    )

    print(
        f"{original_chunk}s | "
        f"alpha={original_alpha} | "
        f"{original_aggregation} | "
        f"threshold={original_threshold}"
    )

    print(
        f"Macro F1 = "
        f"{original_f1:.6f}"
    )

    # ========================================================
    # 1. REUSE OLD RAW SCORES
    # ========================================================

    print(
        "\n1. REUSE P1.12 RAW "
        "OOF SCORES"
    )

    p112_raw = pd.read_csv(
        P112_RAW_PATH
    )

    p112_raw[
        "video_id"
    ] = normalize_video_id(
        p112_raw[
            "video_id"
        ]
    )

    p112_raw[
        "chunk_seconds"
    ] = (
        pd.to_numeric(
            p112_raw[
                "chunk_seconds"
            ],
            errors="raise",
        )
        .astype(int)
    )

    raw_by_size = {}

    for size in OLD_CHUNKS:

        part = (
            p112_raw[
                p112_raw[
                    "chunk_seconds"
                ].eq(size)
            ]
            .copy()
        )

        part = validate_raw_scores(
            part,
            size,
        )

        raw_by_size[
            size
        ] = part

        print(
            f"{size}s reused | "
            f"rows={len(part)}"
        )

    # ========================================================
    # 2. BUILD ONLY 150s / 180s
    # ========================================================

    print(
        "\n2. BUILD NEW BOUNDARY "
        "CHUNK SIZES"
    )

    chunk_counts = {}

    for size in NEW_CHUNKS:

        print(
            f"\nBuilding {size}s chunks..."
        )

        result = (
            SENSITIVITY
            .build_dev_chunks(
                dev.copy(),
                size,
            )
        )

        chunks = extract_chunks(
            result
        )

        chunks = validate_chunks(
            chunks,
            size,
        )

        chunk_counts[
            str(size)
        ] = int(
            len(chunks)
        )

        chunk_path = (
            EXTENSION_DIR
            / f"dev_chunks_{size}s.csv"
        )

        chunks.to_csv(
            chunk_path,
            index=False,
        )

        print(
            f"{size}s chunks: "
            f"{len(chunks)}"
        )

        raw = (
            build_raw_oof_scores(
                chunks=
                    chunks,

                chunk_seconds=
                    size,

                folds=
                    folds,

                concepts=
                    concepts,

                concept_texts=
                    concept_texts,
            )
        )

        raw_by_size[
            size
        ] = raw

        print(
            f"{size}s raw OOF rows: "
            f"{len(raw)}"
        )

    # ========================================================
    # SAVE EXTENDED RAW
    # ========================================================

    extended_raw = pd.concat(
        [
            raw_by_size[
                size
            ]
            for size
            in ALL_CHUNKS
        ],
        ignore_index=True,
        sort=False,
    )

    extended_raw.to_csv(
        EXTENDED_RAW_PATH,
        index=False,
    )

    # ========================================================
    # 3. ONE-TIME EXTENDED GRID SEARCH
    # ========================================================

    print(
        "\n3. ONE-TIME EXTENDED "
        "GRID SEARCH"
    )

    expected_configs = (
        len(ALL_CHUNKS)
        * len(ALPHAS)
        * len(AGGREGATIONS)
        * len(THRESHOLDS)
    )

    print(
        f"Chunks       : "
        f"{ALL_CHUNKS}"
    )

    print(
        f"Alphas       : "
        f"{len(ALPHAS)}"
    )

    print(
        f"Aggregations : "
        f"{AGGREGATIONS}"
    )

    print(
        f"Thresholds   : "
        f"{THRESHOLDS}"
    )

    print(
        f"Expected     : "
        f"{expected_configs} configs"
    )

    search_rows = []

    deterministic_order = 0

    for size in ALL_CHUNKS:

        raw = raw_by_size[
            size
        ]

        print(
            f"\nSearching {size}s..."
        )

        for alpha in ALPHAS:

            for aggregation in (
                AGGREGATIONS
            ):

                aggregated_all = (
                    TUNING
                    .aggregate_scores(
                        raw,
                        alpha,
                        aggregation,
                    )
                )

                aggregated_folds = {}

                for fold in FOLDS:

                    fold_raw = (
                        raw[
                            raw[
                                "outer_fold"
                            ].eq(fold)
                        ]
                    )

                    aggregated_folds[
                        fold
                    ] = (
                        TUNING
                        .aggregate_scores(
                            fold_raw,
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
                                aggregated_folds[
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
                                int(size),

                            "alpha":
                                float(alpha),

                            "aggregation":
                                aggregation,

                            "threshold":
                                float(
                                    threshold
                                ),

                            "macro_video_precision":
                                float(
                                    overall[
                                        "macro_video_precision"
                                    ]
                                ),

                            "macro_video_recall":
                                float(
                                    overall[
                                        "macro_video_recall"
                                    ]
                                ),

                            "macro_video_f1":
                                float(
                                    overall[
                                        "macro_video_f1"
                                    ]
                                ),

                            "micro_precision":
                                float(
                                    overall[
                                        "micro_precision"
                                    ]
                                ),

                            "micro_recall":
                                float(
                                    overall[
                                        "micro_recall"
                                    ]
                                ),

                            "micro_f1":
                                float(
                                    overall[
                                        "micro_f1"
                                    ]
                                ),

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

                            "deterministic_order":
                                deterministic_order,
                        }
                    )

                    deterministic_order += 1

    search = pd.DataFrame(
        search_rows
    )

    if (
        len(search)
        != expected_configs
    ):

        raise ValueError(
            f"Expected "
            f"{expected_configs}, "
            f"found {len(search)}"
        )

    # ========================================================
    # 4. SELECT EXTENDED WINNER
    # ========================================================

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

    selected = ranked.iloc[0]

    winner_order = int(
        selected[
            "deterministic_order"
        ]
    )

    search[
        "selected_boundary_extended_config"
    ] = (
        search[
            "deterministic_order"
        ].eq(
            winner_order
        )
    )

    search.to_csv(
        SEARCH_PATH,
        index=False,
    )

    selected_chunk = int(
        selected[
            "chunk_seconds"
        ]
    )

    selected_alpha = float(
        selected[
            "alpha"
        ]
    )

    selected_aggregation = str(
        selected[
            "aggregation"
        ]
    )

    selected_threshold = float(
        selected[
            "threshold"
        ]
    )

    selected_f1 = float(
        selected[
            "macro_video_f1"
        ]
    )

    selected_micro = float(
        selected[
            "micro_f1"
        ]
    )

    selected_std = float(
        selected[
            "fold_f1_std"
        ]
    )

    # ========================================================
    # 5. BOUNDARY STATUS
    # ========================================================

    chunk_upper_boundary = bool(
        selected_chunk
        == max(
            ALL_CHUNKS
        )
    )

    threshold_lower_boundary = bool(
        np.isclose(
            selected_threshold,
            min(
                THRESHOLDS
            ),
        )
    )

    alpha_lower_boundary = bool(
        np.isclose(
            selected_alpha,
            min(
                ALPHAS
            ),
        )
    )

    changed_from_p112 = not (
        selected_chunk
        == original_chunk
        and np.isclose(
            selected_alpha,
            original_alpha,
        )
        and selected_aggregation
        == original_aggregation
        and np.isclose(
            selected_threshold,
            original_threshold,
        )
    )

    delta_vs_p112 = (
        selected_f1
        - original_f1
    )

    # ========================================================
    # 6. SAVE CONFIG
    # ========================================================

    output_config = {
        "stage":
            (
                "P1.13b One-Time "
                "Boundary Extension"
            ),

        "status":
            "BOUNDARY_EXTENSION_COMPLETE",

        "chunk_seconds":
            selected_chunk,

        "alpha":
            selected_alpha,

        "lsa_weight":
            1.0
            - selected_alpha,

        "aggregation":
            selected_aggregation,

        "threshold":
            selected_threshold,

        "macro_video_f1":
            selected_f1,

        "micro_f1":
            selected_micro,

        "fold_f1_std":
            selected_std,

        "original_p1_12": {
            "chunk_seconds":
                original_chunk,

            "alpha":
                original_alpha,

            "aggregation":
                original_aggregation,

            "threshold":
                original_threshold,

            "macro_video_f1":
                original_f1,
        },

        "changed_from_p1_12":
            bool(
                changed_from_p112
            ),

        "delta_macro_f1_vs_p1_12":
            float(
                delta_vs_p112
            ),

        "boundary_status": {
            "chunk_at_extended_upper_boundary":
                chunk_upper_boundary,

            "alpha_at_natural_lower_boundary":
                alpha_lower_boundary,

            "threshold_at_extended_lower_boundary":
                threshold_lower_boundary,
        },

        "selection_metric":
            "mean_video_level_f1",

        "selection_rule": [
            "highest macro video F1",
            "highest micro F1",
            "lowest fold F1 std",
            "deterministic grid order",
        ],

        "development_only":
            True,

        "final_test_used":
            False,

        "extension_policy":
            (
                "One pre-declared "
                "boundary extension only. "
                "No further grid expansion "
                "will be performed even if "
                "the selected configuration "
                "remains on an extended "
                "boundary."
            ),
    }

    CONFIG_PATH.write_text(
        json.dumps(
            output_config,
            indent=2,
            ensure_ascii=False,
            default=json_default,
        ),
        encoding="utf-8",
    )

    # ========================================================
    # 7. SUMMARY
    # ========================================================

    summary = pd.DataFrame(
        [
            {
                "stage":
                    "P1.12_original",

                "chunk_seconds":
                    original_chunk,

                "alpha":
                    original_alpha,

                "aggregation":
                    original_aggregation,

                "threshold":
                    original_threshold,

                "macro_video_f1":
                    original_f1,
            },

            {
                "stage":
                    "P1.13b_extended",

                "chunk_seconds":
                    selected_chunk,

                "alpha":
                    selected_alpha,

                "aggregation":
                    selected_aggregation,

                "threshold":
                    selected_threshold,

                "macro_video_f1":
                    selected_f1,
            },
        ]
    )

    summary[
        "delta_vs_original_p1_12"
    ] = (
        summary[
            "macro_video_f1"
        ]
        - original_f1
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    # ========================================================
    # 8. AUDIT
    # ========================================================

    audit = {
        "status":
            "PASS",

        "old_raw_scores_reused":
            OLD_CHUNKS,

        "new_raw_scores_generated":
            NEW_CHUNKS,

        "new_chunk_counts":
            chunk_counts,

        "all_chunk_sizes":
            ALL_CHUNKS,

        "thresholds":
            THRESHOLDS,

        "alphas":
            ALPHAS,

        "aggregations":
            AGGREGATIONS,

        "num_configurations":
            int(
                len(search)
            ),

        "expected_configurations":
            int(
                expected_configs
            ),

        "same_five_dev_folds":
            True,

        "video_level_split":
            True,

        "heldout_text_used_for_fit":
            False,

        "p_value_used_for_selection":
            False,

        "final_test_used":
            False,

        "selected_count":
            int(
                search[
                    "selected_boundary_extended_config"
                ].sum()
            ),

        "changed_from_p1_12":
            bool(
                changed_from_p112
            ),

        "further_grid_extension_allowed":
            False,

        "reason_for_extension":
            (
                "P1.13 found the "
                "P1.12 winner at the "
                "upper chunk boundary "
                "and lower threshold "
                "boundary."
            ),
    }

    AUDIT_PATH.write_text(
        json.dumps(
            audit,
            indent=2,
            ensure_ascii=False,
            default=json_default,
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
        "P1.13b BOUNDARY "
        "EXTENSION RESULT"
    )

    print(
        "=" * 82
    )

    print(
        "Original P1.12:"
    )

    print(
        f"  {original_chunk}s | "
        f"alpha={original_alpha} | "
        f"{original_aggregation} | "
        f"threshold={original_threshold}"
    )

    print(
        f"  macro F1 = "
        f"{original_f1:.6f}"
    )

    print()

    print(
        "Extended winner:"
    )

    print(
        f"  {selected_chunk}s | "
        f"alpha={selected_alpha} | "
        f"{selected_aggregation} | "
        f"threshold={selected_threshold}"
    )

    print(
        f"  macro F1 = "
        f"{selected_f1:.6f}"
    )

    print(
        f"  micro F1 = "
        f"{selected_micro:.6f}"
    )

    print(
        f"  fold std = "
        f"{selected_std:.6f}"
    )

    print(
        f"  delta vs P1.12 = "
        f"{delta_vs_p112:+.6f}"
    )

    print()

    print(
        f"Config changed: "
        f"{changed_from_p112}"
    )

    print(
        "Extended boundary flags:"
    )

    print(
        "  chunk upper boundary: "
        f"{chunk_upper_boundary}"
    )

    print(
        "  alpha natural lower "
        "boundary: "
        f"{alpha_lower_boundary}"
    )

    print(
        "  threshold lower boundary: "
        f"{threshold_lower_boundary}"
    )

    print(
        "\nSTOP RULE:"
    )

    print(
        "No further search-space "
        "extension will be performed."
    )

    print(
        "Any remaining boundary "
        "behavior will be reported "
        "as a limitation."
    )

    print(
        "Final TEST was not used."
    )

    print(
        "\nP1.13b ONE-TIME "
        "BOUNDARY EXTENSION: PASS"
    )


if __name__ == "__main__":
    main()