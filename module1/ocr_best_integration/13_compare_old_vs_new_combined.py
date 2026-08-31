from pathlib import Path

import numpy as np
import pandas as pd

from scipy import stats
from scipy.stats import binomtest

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    matthews_corrcoef,
    average_precision_score,
    roc_auc_score,
)


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

COMBINED_ROOT = (
    PROJECT_ROOT
    / "module1"
    / "results"
    / "ocr_best_integration"
    / "combined_system_comparison"
)

# Step 12 went through a few filename revisions.
# Resolve the valid frozen score file by expected row count.
OLD_SCORE_CANDIDATES = [
    COMBINED_ROOT
    / "OLD_combined"
    / "fusion_scores.csv",

    COMBINED_ROOT
    / "OLD_transcript_plus_D"
    / "fusion_scores.csv",
]

NEW_SCORE_CANDIDATES = [
    COMBINED_ROOT
    / "NEW_combined_F"
    / "fusion_scores.csv",

    COMBINED_ROOT
    / "NEW_transcript_plus_F"
    / "fusion_scores.csv",

    COMBINED_ROOT
    / "NEW_F_mean"
    / "fusion_scores.csv",
]


GT_FILE = (
    PROJECT_ROOT
    / "data"
    / "ground_truth"
    / "ground_truth_concepts.csv"
)

CONCEPT_FILE = (
    PROJECT_ROOT
    / "data"
    / "concepts"
    / "concept_catalog.csv"
)

VIDEOS_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "videos.csv"
)


RESULT_DIR = (
    COMBINED_ROOT
    / "final_evaluation"
)

PAIR_FILE = (
    RESULT_DIR
    / "old_vs_new_combined_pair_scores.csv"
)

VIDEO_FILE = (
    RESULT_DIR
    / "old_vs_new_combined_video_metrics.csv"
)

SUBJECT_FILE = (
    RESULT_DIR
    / "old_vs_new_combined_subject_metrics.csv"
)

METRICS_FILE = (
    RESULT_DIR
    / "old_vs_new_combined_overall_metrics.csv"
)

STATS_FILE = (
    RESULT_DIR
    / "old_vs_new_combined_statistics.csv"
)

BOOTSTRAP_FILE = (
    RESULT_DIR
    / "old_vs_new_combined_bootstrap_deltas.csv"
)

AUDIT_FILE = (
    RESULT_DIR
    / "old_vs_new_combined_audit.csv"
)

SUMMARY_FILE = (
    RESULT_DIR
    / "OLD_VS_NEW_COMBINED_SUMMARY.txt"
)


# =========================================================
# LOCKED EXPECTATIONS
# =========================================================

DEV_IDS = [
    f"v{i}"
    for i in range(1, 41)
]

EXPECTED_VIDEOS = 40
EXPECTED_CONCEPTS = 39
EXPECTED_PAIRS = 390
EXPECTED_GT_POSITIVES = 153

EXPECTED_OLD_SCORE_ROWS = 4853
EXPECTED_NEW_SCORE_ROWS = 1689


# =========================================================
# LOCKED MODEL CONFIG
# =========================================================

OLD_AGGREGATION = "top2_mean"
OLD_TOP_K = 2
OLD_THRESHOLD = 0.40

NEW_AGGREGATION = "max"
NEW_THRESHOLD = 0.50

BOOTSTRAP_ITERATIONS = 10000
BOOTSTRAP_RANDOM_STATE = 42


# =========================================================
# HELPERS
# =========================================================

def video_sort_key(video_id):
    try:
        return int(
            str(video_id)
            .strip()
            .lower()
            .replace("v", "")
        )

    except Exception:
        return 999999


def norm_subject(value):
    value = (
        str(value)
        .strip()
        .lower()
    )

    aliases = {
        "cpp": "c++",
        "cxx": "c++",
        "c plus plus": "c++",
    }

    return aliases.get(
        value,
        value,
    )


def norm_concept(value):
    return (
        str(value)
        .strip()
        .lower()
    )


def safe_divide(a, b):
    if b == 0:
        return 0.0

    return float(a / b)


# =========================================================
# RESOLVE STEP 12 OUTPUT FILES
# =========================================================

def resolve_score_file(
    candidates,
    expected_rows,
    label,
):
    valid = []

    for path in candidates:

        if not path.exists():
            continue

        try:
            df = pd.read_csv(
                path
            )

            rows = len(df)

        except Exception:
            continue

        print(
            f"{label} candidate: "
            f"{path} -> {rows} rows"
        )

        if rows == expected_rows:
            valid.append(
                path
            )

    if not valid:

        raise FileNotFoundError(
            f"\nCould not find valid {label} score file "
            f"with {expected_rows} rows.\n"
            f"Checked:\n"
            + "\n".join(
                str(path)
                for path in candidates
            )
        )

    selected = valid[0]

    if len(valid) > 1:
        print(
            f"WARNING: multiple valid {label} files found."
        )

        print(
            "Using first locked candidate:"
        )

    print(
        f"{label} selected:"
    )

    print(
        selected
    )

    return selected


# =========================================================
# VIDEO / CONCEPT / GT
# =========================================================

def load_videos():
    videos = pd.read_csv(
        VIDEOS_FILE
    )

    required = {
        "video_id",
        "subject",
    }

    missing = (
        required
        - set(videos.columns)
    )

    if missing:
        raise ValueError(
            f"videos.csv missing columns: "
            f"{sorted(missing)}"
        )

    videos["video_id"] = (
        videos["video_id"]
        .astype(str)
        .str.strip()
    )

    videos = videos[
        videos["video_id"]
        .isin(
            DEV_IDS
        )
    ].copy()

    videos["subject"] = (
        videos["subject"]
        .astype(str)
        .str.strip()
    )

    videos["subject_norm"] = (
        videos["subject"]
        .apply(
            norm_subject
        )
    )

    duplicate_ids = int(
        videos.duplicated(
            "video_id"
        ).sum()
    )

    missing_ids = sorted(
        set(DEV_IDS)
        - set(
            videos["video_id"]
        ),
        key=video_sort_key,
    )

    if (
        len(videos)
        != EXPECTED_VIDEOS
        or duplicate_ids != 0
        or missing_ids
    ):
        raise RuntimeError(
            "DEV video audit failed."
        )

    return videos[
        [
            "video_id",
            "subject",
            "subject_norm",
        ]
    ].copy()


def load_concepts():
    concepts = pd.read_csv(
        CONCEPT_FILE
    )

    required = {
        "subject",
        "concept",
    }

    missing = (
        required
        - set(concepts.columns)
    )

    if missing:
        raise ValueError(
            f"concept catalog missing: "
            f"{sorted(missing)}"
        )

    concepts["subject"] = (
        concepts["subject"]
        .astype(str)
        .str.strip()
    )

    concepts["concept"] = (
        concepts["concept"]
        .astype(str)
        .str.strip()
    )

    concepts["subject_norm"] = (
        concepts["subject"]
        .apply(
            norm_subject
        )
    )

    concepts["concept_norm"] = (
        concepts["concept"]
        .apply(
            norm_concept
        )
    )

    duplicates = int(
        concepts.duplicated(
            [
                "subject_norm",
                "concept_norm",
            ]
        ).sum()
    )

    if (
        len(concepts)
        != EXPECTED_CONCEPTS
        or duplicates != 0
    ):
        raise RuntimeError(
            "Concept catalog audit failed."
        )

    return concepts[
        [
            "subject",
            "subject_norm",
            "concept",
            "concept_norm",
        ]
    ].copy()


def load_gt():
    gt = pd.read_csv(
        GT_FILE
    )

    required = {
        "video_id",
        "subject",
        "concept",
    }

    missing = (
        required
        - set(gt.columns)
    )

    if missing:
        raise ValueError(
            f"GT missing columns: "
            f"{sorted(missing)}"
        )

    gt["video_id"] = (
        gt["video_id"]
        .astype(str)
        .str.strip()
    )

    gt = gt[
        gt["video_id"]
        .isin(
            DEV_IDS
        )
    ].copy()

    gt["subject_norm"] = (
        gt["subject"]
        .apply(
            norm_subject
        )
    )

    gt["concept_norm"] = (
        gt["concept"]
        .apply(
            norm_concept
        )
    )

    duplicates = int(
        gt.duplicated(
            [
                "video_id",
                "subject_norm",
                "concept_norm",
            ]
        ).sum()
    )

    if duplicates != 0:
        raise RuntimeError(
            f"Duplicate GT rows: {duplicates}"
        )

    if len(gt) != EXPECTED_GT_POSITIVES:
        raise RuntimeError(
            f"Expected {EXPECTED_GT_POSITIVES} "
            f"DEV GT positives, found {len(gt)}."
        )

    return gt[
        [
            "video_id",
            "subject_norm",
            "concept_norm",
        ]
    ].copy()


# =========================================================
# COMPLETE 390-PAIR SPACE
# =========================================================

def build_pair_space(
    videos,
    concepts,
    gt,
):
    rows = []

    for _, video in videos.iterrows():

        subject_concepts = (
            concepts[
                concepts[
                    "subject_norm"
                ]
                == video[
                    "subject_norm"
                ]
            ]
        )

        if subject_concepts.empty:
            raise RuntimeError(
                f"No concepts found for "
                f"{video['video_id']}."
            )

        for _, concept in (
            subject_concepts.iterrows()
        ):

            rows.append(
                {
                    "video_id":
                        video["video_id"],

                    "subject":
                        video["subject"],

                    "subject_norm":
                        video["subject_norm"],

                    "concept":
                        concept["concept"],

                    "concept_norm":
                        concept["concept_norm"],
                }
            )

    pairs = pd.DataFrame(
        rows
    )

    gt_keys = set(
        map(
            tuple,
            gt[
                [
                    "video_id",
                    "subject_norm",
                    "concept_norm",
                ]
            ].itertuples(
                index=False,
                name=None,
            ),
        )
    )

    pairs["y_true"] = (
        pairs.apply(
            lambda row:
                int(
                    (
                        row["video_id"],
                        row["subject_norm"],
                        row["concept_norm"],
                    )
                    in gt_keys
                ),
            axis=1,
        )
    )

    duplicates = int(
        pairs.duplicated(
            [
                "video_id",
                "concept_norm",
            ]
        ).sum()
    )

    if (
        len(pairs) != EXPECTED_PAIRS
        or duplicates != 0
        or int(
            pairs["y_true"].sum()
        ) != EXPECTED_GT_POSITIVES
    ):
        raise RuntimeError(
            "Evaluation pair-space audit failed."
        )

    return pairs


# =========================================================
# LOAD CHUNK-CONCEPT SCORES
# =========================================================

def load_scores(
    path,
    label,
    expected_rows,
):
    df = pd.read_csv(
        path
    )

    required = {
        "video_id",
        "subject",
        "chunk_id",
        "concept",
        "final_score",
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            f"{label} missing columns: "
            f"{sorted(missing)}"
        )

    df["video_id"] = (
        df["video_id"]
        .astype(str)
        .str.strip()
    )

    df["subject_norm"] = (
        df["subject"]
        .apply(
            norm_subject
        )
    )

    df["concept_norm"] = (
        df["concept"]
        .apply(
            norm_concept
        )
    )

    df["final_score"] = (
        pd.to_numeric(
            df["final_score"],
            errors="coerce",
        )
    )

    nan_count = int(
        df["final_score"]
        .isna()
        .sum()
    )

    duplicates = int(
        df.duplicated(
            [
                "video_id",
                "chunk_id",
                "concept_norm",
            ]
        ).sum()
    )

    out_of_range = int(
        (
            (
                df["final_score"]
                < -1e-9
            )
            |
            (
                df["final_score"]
                > 1.0 + 1e-9
            )
        ).sum()
    )

    print()
    print(
        f"========== {label} SCORE AUDIT =========="
    )

    print(
        "Rows:",
        len(df),
    )

    print(
        "Videos:",
        df["video_id"].nunique(),
    )

    print(
        "NaN:",
        nan_count,
    )

    print(
        "Duplicates:",
        duplicates,
    )

    print(
        "Out of range:",
        out_of_range,
    )

    if (
        len(df) != expected_rows
        or df["video_id"].nunique()
        != EXPECTED_VIDEOS
        or nan_count != 0
        or duplicates != 0
        or out_of_range != 0
    ):
        raise RuntimeError(
            f"{label} score audit FAILED."
        )

    print(
        f"{label} SCORE AUDIT PASS"
    )

    return df


# =========================================================
# VIDEO-CONCEPT AGGREGATION
# =========================================================

def build_video_evidence(
    scores,
    score_name,
    aggregation,
):
    rows = []

    grouped = scores.groupby(
        [
            "video_id",
            "subject_norm",
            "concept_norm",
        ],
        sort=False,
    )

    for (
        video_id,
        subject_norm,
        concept_norm,
    ), group in grouped:

        values = (
            group["final_score"]
            .astype(float)
            .sort_values(
                ascending=False
            )
        )

        if values.empty:
            evidence = 0.0
            chunks_used = 0

        elif aggregation == "top2_mean":

            selected = values.head(
                OLD_TOP_K
            )

            evidence = float(
                selected.mean()
            )

            chunks_used = len(
                selected
            )

        elif aggregation == "max":

            evidence = float(
                values.iloc[0]
            )

            chunks_used = 1

        else:
            raise ValueError(
                f"Unknown aggregation: "
                f"{aggregation}"
            )

        rows.append(
            {
                "video_id":
                    video_id,

                "subject_norm":
                    subject_norm,

                "concept_norm":
                    concept_norm,

                score_name:
                    evidence,

                f"{score_name}_chunks_used":
                    int(chunks_used),
            }
        )

    result = pd.DataFrame(
        rows
    )

    duplicates = int(
        result.duplicated(
            [
                "video_id",
                "subject_norm",
                "concept_norm",
            ]
        ).sum()
    )

    if duplicates != 0:
        raise RuntimeError(
            f"{score_name}: duplicate evidence."
        )

    return result


# =========================================================
# METRICS
# =========================================================

def confusion_values(
    y_true,
    y_pred,
):
    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=[
            0,
            1,
        ],
    )

    tn, fp, fn, tp = (
        matrix.ravel()
    )

    return (
        int(tn),
        int(fp),
        int(fn),
        int(tp),
    )


def compute_metrics(
    y_true,
    y_pred,
    scores,
):
    y_true = np.asarray(
        y_true,
        dtype=int,
    )

    y_pred = np.asarray(
        y_pred,
        dtype=int,
    )

    scores = np.asarray(
        scores,
        dtype=float,
    )

    tn, fp, fn, tp = (
        confusion_values(
            y_true,
            y_pred,
        )
    )

    precision = float(
        precision_score(
            y_true,
            y_pred,
            zero_division=0,
        )
    )

    recall = float(
        recall_score(
            y_true,
            y_pred,
            zero_division=0,
        )
    )

    f1 = float(
        f1_score(
            y_true,
            y_pred,
            zero_division=0,
        )
    )

    specificity = (
        safe_divide(
            tn,
            tn + fp,
        )
    )

    fpr = (
        safe_divide(
            fp,
            fp + tn,
        )
    )

    # Avoid sklearn one-class warning.
    balanced_accuracy = (
        (
            recall
            + specificity
        )
        / 2.0
    )

    mcc = float(
        matthews_corrcoef(
            y_true,
            y_pred,
        )
    )

    unique_true = (
        np.unique(
            y_true
        )
    )

    if len(
        unique_true
    ) >= 2:

        pr_auc = float(
            average_precision_score(
                y_true,
                scores,
            )
        )

        roc_auc = float(
            roc_auc_score(
                y_true,
                scores,
            )
        )

    else:

        # PR-AP is still interpretable for a positive-only
        # video, but AUC is not.
        if int(
            y_true.sum()
        ) > 0:

            pr_auc = float(
                average_precision_score(
                    y_true,
                    scores,
                )
            )

        else:
            pr_auc = np.nan

        roc_auc = np.nan

    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,

        "precision":
            precision,

        "recall":
            recall,

        "f1":
            f1,

        "specificity":
            specificity,

        "fpr":
            fpr,

        "mcc":
            mcc,

        "balanced_accuracy":
            balanced_accuracy,

        "pr_auc":
            pr_auc,

        "roc_auc":
            roc_auc,

        "predicted_positive":
            int(
                y_pred.sum()
            ),

        "actual_positive":
            int(
                y_true.sum()
            ),
    }


# =========================================================
# PER-VIDEO
# =========================================================

def build_video_metrics(
    pairs,
):
    rows = []

    for (
        video_id,
        subject,
    ), group in pairs.groupby(
        [
            "video_id",
            "subject",
        ],
        sort=False,
    ):

        row = {
            "video_id":
                video_id,

            "subject":
                subject,

            "concepts":
                len(group),

            "gt_positive":
                int(
                    group["y_true"]
                    .sum()
                ),
        }

        for system in [
            "OLD",
            "NEW",
        ]:

            metrics = compute_metrics(
                group["y_true"],
                group[
                    f"pred_{system}"
                ],
                group[
                    f"score_{system}"
                ],
            )

            for key, value in (
                metrics.items()
            ):
                row[
                    f"{system}_{key}"
                ] = value

        row[
            "delta_f1_NEW_minus_OLD"
        ] = (
            row["NEW_f1"]
            - row["OLD_f1"]
        )

        rows.append(
            row
        )

    result = pd.DataFrame(
        rows
    )

    result["_order"] = (
        result["video_id"]
        .map(
            video_sort_key
        )
    )

    result = (
        result
        .sort_values(
            "_order"
        )
        .drop(
            columns=[
                "_order"
            ]
        )
        .reset_index(
            drop=True
        )
    )

    if len(result) != 40:
        raise RuntimeError(
            "Per-video output is not 40 rows."
        )

    return result


# =========================================================
# mAP
# =========================================================

def calculate_map(
    pairs,
    system,
):
    values = []

    for _, group in (
        pairs.groupby(
            "video_id"
        )
    ):

        y_true = (
            group["y_true"]
            .to_numpy(
                dtype=int
            )
        )

        scores = (
            group[
                f"score_{system}"
            ]
            .to_numpy(
                dtype=float
            )
        )

        if int(
            y_true.sum()
        ) == 0:
            continue

        ap = (
            average_precision_score(
                y_true,
                scores,
            )
        )

        values.append(
            float(ap)
        )

    if not values:
        return np.nan

    return float(
        np.mean(
            values
        )
    )


# =========================================================
# OVERALL
# =========================================================

def build_overall_metrics(
    pairs,
    video_metrics,
):
    rows = []

    configs = {
        "OLD": {
            "name":
                "OLD_COMBINED",

            "aggregation":
                OLD_AGGREGATION,

            "threshold":
                OLD_THRESHOLD,
        },

        "NEW": {
            "name":
                "NEW_COMBINED_F",

            "aggregation":
                NEW_AGGREGATION,

            "threshold":
                NEW_THRESHOLD,
        },
    }

    for system in [
        "OLD",
        "NEW",
    ]:

        pair_metrics = (
            compute_metrics(
                pairs["y_true"],
                pairs[
                    f"pred_{system}"
                ],
                pairs[
                    f"score_{system}"
                ],
            )
        )

        rows.append(
            {
                "system":
                    configs[
                        system
                    ][
                        "name"
                    ],

                "aggregation":
                    configs[
                        system
                    ][
                        "aggregation"
                    ],

                "threshold":
                    configs[
                        system
                    ][
                        "threshold"
                    ],

                "macro_video_precision":
                    float(
                        video_metrics[
                            f"{system}_precision"
                        ].mean()
                    ),

                "macro_video_recall":
                    float(
                        video_metrics[
                            f"{system}_recall"
                        ].mean()
                    ),

                "macro_video_f1":
                    float(
                        video_metrics[
                            f"{system}_f1"
                        ].mean()
                    ),

                "macro_video_f1_std":
                    float(
                        video_metrics[
                            f"{system}_f1"
                        ].std(
                            ddof=1
                        )
                    ),

                "micro_precision":
                    pair_metrics[
                        "precision"
                    ],

                "micro_recall":
                    pair_metrics[
                        "recall"
                    ],

                "micro_f1":
                    pair_metrics[
                        "f1"
                    ],

                "mcc":
                    pair_metrics[
                        "mcc"
                    ],

                "balanced_accuracy":
                    pair_metrics[
                        "balanced_accuracy"
                    ],

                "specificity":
                    pair_metrics[
                        "specificity"
                    ],

                "fpr":
                    pair_metrics[
                        "fpr"
                    ],

                "pr_auc":
                    pair_metrics[
                        "pr_auc"
                    ],

                "roc_auc":
                    pair_metrics[
                        "roc_auc"
                    ],

                "mAP_video":
                    calculate_map(
                        pairs,
                        system,
                    ),

                "TP":
                    pair_metrics["TP"],

                "FP":
                    pair_metrics["FP"],

                "FN":
                    pair_metrics["FN"],

                "TN":
                    pair_metrics["TN"],

                "predicted_positive":
                    pair_metrics[
                        "predicted_positive"
                    ],

                "actual_positive":
                    pair_metrics[
                        "actual_positive"
                    ],
            }
        )

    return pd.DataFrame(
        rows
    )


# =========================================================
# PER-SUBJECT
# =========================================================

def build_subject_metrics(
    pairs,
    video_metrics,
):
    rows = []

    subjects = sorted(
        pairs["subject"]
        .unique()
        .tolist()
    )

    for subject in subjects:

        pair_subset = pairs[
            pairs["subject"]
            == subject
        ]

        video_subset = (
            video_metrics[
                video_metrics[
                    "subject"
                ]
                == subject
            ]
        )

        for system in [
            "OLD",
            "NEW",
        ]:

            metrics = compute_metrics(
                pair_subset[
                    "y_true"
                ],
                pair_subset[
                    f"pred_{system}"
                ],
                pair_subset[
                    f"score_{system}"
                ],
            )

            rows.append(
                {
                    "subject":
                        subject,

                    "system":
                        (
                            "OLD_COMBINED"
                            if system
                            == "OLD"
                            else
                            "NEW_COMBINED_F"
                        ),

                    "videos":
                        video_subset[
                            "video_id"
                        ].nunique(),

                    "pairs":
                        len(
                            pair_subset
                        ),

                    "macro_video_precision":
                        float(
                            video_subset[
                                f"{system}_precision"
                            ].mean()
                        ),

                    "macro_video_recall":
                        float(
                            video_subset[
                                f"{system}_recall"
                            ].mean()
                        ),

                    "macro_video_f1":
                        float(
                            video_subset[
                                f"{system}_f1"
                            ].mean()
                        ),

                    "micro_precision":
                        metrics[
                            "precision"
                        ],

                    "micro_recall":
                        metrics[
                            "recall"
                        ],

                    "micro_f1":
                        metrics[
                            "f1"
                        ],

                    "mcc":
                        metrics[
                            "mcc"
                        ],

                    "balanced_accuracy":
                        metrics[
                            "balanced_accuracy"
                        ],

                    "pr_auc":
                        metrics[
                            "pr_auc"
                        ],

                    "roc_auc":
                        metrics[
                            "roc_auc"
                        ],
                }
            )

    return pd.DataFrame(
        rows
    )


# =========================================================
# SUBJECT-STRATIFIED PAIRED BOOTSTRAP
# =========================================================

def bootstrap_macro_delta(
    video_metrics,
):
    rng = np.random.default_rng(
        BOOTSTRAP_RANDOM_STATE
    )

    groups = {
        subject:
            group.reset_index(
                drop=True
            )

        for subject, group
        in video_metrics.groupby(
            "subject"
        )
    }

    deltas = []

    for _ in range(
        BOOTSTRAP_ITERATIONS
    ):

        sampled = []

        for _, group in (
            groups.items()
        ):

            indices = (
                rng.integers(
                    0,
                    len(group),
                    size=len(group),
                )
            )

            sampled.append(
                group.iloc[
                    indices
                ]
            )

        sample = pd.concat(
            sampled,
            ignore_index=True,
        )

        old_macro = float(
            sample[
                "OLD_f1"
            ].mean()
        )

        new_macro = float(
            sample[
                "NEW_f1"
            ].mean()
        )

        deltas.append(
            new_macro
            - old_macro
        )

    deltas = np.asarray(
        deltas,
        dtype=float,
    )

    ci_low = float(
        np.percentile(
            deltas,
            2.5,
        )
    )

    ci_high = float(
        np.percentile(
            deltas,
            97.5,
        )
    )

    return (
        deltas,
        ci_low,
        ci_high,
    )


# =========================================================
# PAIRED STATS
# =========================================================

def paired_statistics(
    pairs,
    video_metrics,
):
    old_f1 = (
        video_metrics[
            "OLD_f1"
        ]
        .to_numpy(
            dtype=float
        )
    )

    new_f1 = (
        video_metrics[
            "NEW_f1"
        ]
        .to_numpy(
            dtype=float
        )
    )

    deltas = (
        new_f1
        - old_f1
    )

    # -----------------------------------------------------
    # Wilcoxon
    # -----------------------------------------------------

    if np.allclose(
        deltas,
        0.0,
    ):

        wilcoxon_stat = 0.0
        wilcoxon_p = 1.0

    else:

        result = stats.wilcoxon(
            new_f1,
            old_f1,
            zero_method="wilcox",
            alternative="two-sided",
        )

        wilcoxon_stat = float(
            result.statistic
        )

        wilcoxon_p = float(
            result.pvalue
        )

    # -----------------------------------------------------
    # Paired t-test
    # -----------------------------------------------------

    if np.allclose(
        deltas,
        0.0,
    ):

        t_stat = 0.0
        t_p = 1.0

    else:

        t_result = (
            stats.ttest_rel(
                new_f1,
                old_f1,
            )
        )

        t_stat = float(
            t_result.statistic
        )

        t_p = float(
            t_result.pvalue
        )

    # -----------------------------------------------------
    # Cohen dz
    # -----------------------------------------------------

    delta_std = float(
        np.std(
            deltas,
            ddof=1,
        )
    )

    if delta_std == 0:
        cohen_dz = 0.0

    else:
        cohen_dz = float(
            np.mean(
                deltas
            )
            / delta_std
        )

    # -----------------------------------------------------
    # Exact McNemar
    # -----------------------------------------------------

    y_true = (
        pairs["y_true"]
        .to_numpy(
            dtype=int
        )
    )

    old_pred = (
        pairs[
            "pred_OLD"
        ]
        .to_numpy(
            dtype=int
        )
    )

    new_pred = (
        pairs[
            "pred_NEW"
        ]
        .to_numpy(
            dtype=int
        )
    )

    old_correct = (
        old_pred
        == y_true
    )

    new_correct = (
        new_pred
        == y_true
    )

    old_correct_new_wrong = int(
        np.sum(
            old_correct
            & (~new_correct)
        )
    )

    old_wrong_new_correct = int(
        np.sum(
            (~old_correct)
            & new_correct
        )
    )

    discordant = (
        old_correct_new_wrong
        + old_wrong_new_correct
    )

    if discordant == 0:
        mcnemar_p = 1.0

    else:
        mcnemar_p = float(
            binomtest(
                old_wrong_new_correct,
                n=discordant,
                p=0.5,
                alternative="two-sided",
            ).pvalue
        )

    improved = int(
        np.sum(
            deltas > 1e-12
        )
    )

    worse = int(
        np.sum(
            deltas < -1e-12
        )
    )

    tied = int(
        len(deltas)
        - improved
        - worse
    )

    return {
        "wilcoxon_statistic":
            wilcoxon_stat,

        "wilcoxon_p":
            wilcoxon_p,

        "paired_t_statistic":
            t_stat,

        "paired_t_p":
            t_p,

        "cohen_dz":
            cohen_dz,

        "mcnemar_OLD_correct_NEW_wrong":
            old_correct_new_wrong,

        "mcnemar_OLD_wrong_NEW_correct":
            old_wrong_new_correct,

        "mcnemar_discordant":
            discordant,

        "mcnemar_exact_p":
            mcnemar_p,

        "videos_improved":
            improved,

        "videos_worse":
            worse,

        "videos_tied":
            tied,
    }


# =========================================================
# MAIN
# =========================================================

def main():
    print("=" * 72)

    print(
        "STEP 13 - FINAL OLD VS NEW COMBINED SYSTEM COMPARISON"
    )

    print("=" * 72)

    print()
    print(
        "OLD COMBINED:"
    )

    print(
        "Transcript OLD + OCR D"
    )

    print(
        "60s | LDA=.40 | LSA=.60"
    )

    print(
        f"{OLD_AGGREGATION} | "
        f"threshold={OLD_THRESHOLD:.2f}"
    )

    print()

    print(
        "NEW COMBINED:"
    )

    print(
        "Transcript NEW + OCR F"
    )

    print(
        "180s | LDA=0 | LSA=1"
    )

    print(
        "OCR F = mean score of "
        "Transcript+D and Transcript+E"
    )

    print(
        f"{NEW_AGGREGATION} | "
        f"threshold={NEW_THRESHOLD:.2f}"
    )

    print()

    print(
        "LOCKED evaluation."
    )

    print(
        "NO threshold / aggregation / OCR retuning."
    )

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # =====================================================
    # RESOLVE SCORE FILES
    # =====================================================

    old_score_file = (
        resolve_score_file(
            OLD_SCORE_CANDIDATES,
            EXPECTED_OLD_SCORE_ROWS,
            "OLD",
        )
    )

    new_score_file = (
        resolve_score_file(
            NEW_SCORE_CANDIDATES,
            EXPECTED_NEW_SCORE_ROWS,
            "NEW",
        )
    )

    # =====================================================
    # LOAD DATA
    # =====================================================

    videos = load_videos()

    concepts = load_concepts()

    gt = load_gt()

    pairs = build_pair_space(
        videos,
        concepts,
        gt,
    )

    old_scores = load_scores(
        old_score_file,
        "OLD COMBINED",
        EXPECTED_OLD_SCORE_ROWS,
    )

    new_scores = load_scores(
        new_score_file,
        "NEW COMBINED F",
        EXPECTED_NEW_SCORE_ROWS,
    )

    print()
    print(
        "========== EVALUATION SPACE =========="
    )

    print(
        "DEV videos:",
        len(videos),
    )

    print(
        "Concepts:",
        len(concepts),
    )

    print(
        "Pairs:",
        len(pairs),
    )

    print(
        "GT positives:",
        int(
            pairs[
                "y_true"
            ].sum()
        ),
    )

    # =====================================================
    # AGGREGATE TO VIDEO-CONCEPT
    # =====================================================

    old_evidence = (
        build_video_evidence(
            old_scores,
            "score_OLD",
            OLD_AGGREGATION,
        )
    )

    new_evidence = (
        build_video_evidence(
            new_scores,
            "score_NEW",
            NEW_AGGREGATION,
        )
    )

    print()
    print(
        "OLD evidence rows:",
        len(
            old_evidence
        ),
    )

    print(
        "NEW evidence rows:",
        len(
            new_evidence
        ),
    )

    pairs = pairs.merge(
        old_evidence,
        on=[
            "video_id",
            "subject_norm",
            "concept_norm",
        ],
        how="left",
        validate="one_to_one",
    )

    pairs = pairs.merge(
        new_evidence,
        on=[
            "video_id",
            "subject_norm",
            "concept_norm",
        ],
        how="left",
        validate="one_to_one",
    )

    missing_old = int(
        pairs[
            "score_OLD"
        ]
        .isna()
        .sum()
    )

    missing_new = int(
        pairs[
            "score_NEW"
        ]
        .isna()
        .sum()
    )

    if (
        missing_old != 0
        or missing_new != 0
    ):
        raise RuntimeError(
            "Missing video-concept evidence: "
            f"OLD={missing_old}, "
            f"NEW={missing_new}"
        )

    # =====================================================
    # LOCKED PREDICTIONS
    # =====================================================

    pairs[
        "pred_OLD"
    ] = (
        pairs[
            "score_OLD"
        ]
        >= OLD_THRESHOLD
    ).astype(int)

    pairs[
        "pred_NEW"
    ] = (
        pairs[
            "score_NEW"
        ]
        >= NEW_THRESHOLD
    ).astype(int)

    # =====================================================
    # METRICS
    # =====================================================

    video_metrics = (
        build_video_metrics(
            pairs
        )
    )

    overall = (
        build_overall_metrics(
            pairs,
            video_metrics,
        )
    )

    subject_metrics = (
        build_subject_metrics(
            pairs,
            video_metrics,
        )
    )

    # =====================================================
    # BOOTSTRAP
    # =====================================================

    (
        bootstrap_deltas,
        ci_low,
        ci_high,
    ) = (
        bootstrap_macro_delta(
            video_metrics
        )
    )

    bootstrap_df = pd.DataFrame(
        {
            "iteration":
                np.arange(
                    1,
                    len(
                        bootstrap_deltas
                    )
                    + 1,
                ),

            "delta_macro_video_f1_NEW_minus_OLD":
                bootstrap_deltas,
        }
    )

    # =====================================================
    # PAIRED STATS
    # =====================================================

    paired = paired_statistics(
        pairs,
        video_metrics,
    )

    old_row = overall[
        overall["system"]
        == "OLD_COMBINED"
    ].iloc[0]

    new_row = overall[
        overall["system"]
        == "NEW_COMBINED_F"
    ].iloc[0]

    old_macro = float(
        old_row[
            "macro_video_f1"
        ]
    )

    new_macro = float(
        new_row[
            "macro_video_f1"
        ]
    )

    delta_macro = (
        new_macro
        - old_macro
    )

    old_micro = float(
        old_row[
            "micro_f1"
        ]
    )

    new_micro = float(
        new_row[
            "micro_f1"
        ]
    )

    delta_micro = (
        new_micro
        - old_micro
    )

    statistics_df = pd.DataFrame(
        [
            {
                "primary_metric":
                    "macro_video_f1",

                "OLD_macro_video_f1":
                    old_macro,

                "NEW_macro_video_f1":
                    new_macro,

                "delta_macro_video_f1_NEW_minus_OLD":
                    delta_macro,

                "OLD_micro_f1":
                    old_micro,

                "NEW_micro_f1":
                    new_micro,

                "delta_micro_f1_NEW_minus_OLD":
                    delta_micro,

                "bootstrap_iterations":
                    BOOTSTRAP_ITERATIONS,

                "bootstrap_seed":
                    BOOTSTRAP_RANDOM_STATE,

                "bootstrap_95ci_low":
                    ci_low,

                "bootstrap_95ci_high":
                    ci_high,

                **paired,
            }
        ]
    )

    # =====================================================
    # FINAL AUDIT
    # =====================================================

    duplicates = int(
        pairs.duplicated(
            [
                "video_id",
                "concept_norm",
            ]
        ).sum()
    )

    audit = pd.DataFrame(
        [
            {
                "check":
                    "DEV_videos",

                "value":
                    len(videos),

                "expected":
                    EXPECTED_VIDEOS,

                "pass":
                    len(videos)
                    == EXPECTED_VIDEOS,
            },

            {
                "check":
                    "concepts",

                "value":
                    len(concepts),

                "expected":
                    EXPECTED_CONCEPTS,

                "pass":
                    len(concepts)
                    == EXPECTED_CONCEPTS,
            },

            {
                "check":
                    "pairs",

                "value":
                    len(pairs),

                "expected":
                    EXPECTED_PAIRS,

                "pass":
                    len(pairs)
                    == EXPECTED_PAIRS,
            },

            {
                "check":
                    "GT_positives",

                "value":
                    int(
                        pairs[
                            "y_true"
                        ].sum()
                    ),

                "expected":
                    EXPECTED_GT_POSITIVES,

                "pass":
                    int(
                        pairs[
                            "y_true"
                        ].sum()
                    )
                    == EXPECTED_GT_POSITIVES,
            },

            {
                "check":
                    "pair_duplicates",

                "value":
                    duplicates,

                "expected":
                    0,

                "pass":
                    duplicates
                    == 0,
            },

            {
                "check":
                    "missing_OLD_evidence",

                "value":
                    missing_old,

                "expected":
                    0,

                "pass":
                    missing_old
                    == 0,
            },

            {
                "check":
                    "missing_NEW_evidence",

                "value":
                    missing_new,

                "expected":
                    0,

                "pass":
                    missing_new
                    == 0,
            },

            {
                "check":
                    "OLD_score_rows",

                "value":
                    len(
                        old_scores
                    ),

                "expected":
                    EXPECTED_OLD_SCORE_ROWS,

                "pass":
                    len(
                        old_scores
                    )
                    == EXPECTED_OLD_SCORE_ROWS,
            },

            {
                "check":
                    "NEW_score_rows",

                "value":
                    len(
                        new_scores
                    ),

                "expected":
                    EXPECTED_NEW_SCORE_ROWS,

                "pass":
                    len(
                        new_scores
                    )
                    == EXPECTED_NEW_SCORE_ROWS,
            },
        ]
    )

    all_pass = bool(
        audit[
            "pass"
        ].all()
    )

    if not all_pass:

        print()
        print(
            audit.to_string(
                index=False
            )
        )

        raise RuntimeError(
            "STEP 13 FINAL AUDIT FAILED."
        )

    # =====================================================
    # SAVE
    # =====================================================

    pairs["_order"] = (
        pairs["video_id"]
        .map(
            video_sort_key
        )
    )

    pairs = (
        pairs
        .sort_values(
            [
                "_order",
                "concept",
            ]
        )
        .drop(
            columns=[
                "_order"
            ]
        )
        .reset_index(
            drop=True
        )
    )

    pairs.to_csv(
        PAIR_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    video_metrics.to_csv(
        VIDEO_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    subject_metrics.to_csv(
        SUBJECT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    overall.to_csv(
        METRICS_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    statistics_df.to_csv(
        STATS_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    bootstrap_df.to_csv(
        BOOTSTRAP_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    audit.to_csv(
        AUDIT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    # =====================================================
    # WINNER
    # =====================================================

    if delta_macro > 1e-12:
        winner = (
            "NEW COMBINED F"
        )

    elif delta_macro < -1e-12:
        winner = (
            "OLD COMBINED"
        )

    else:
        winner = (
            "TIE"
        )

    # =====================================================
    # PRINT RESULTS
    # =====================================================

    print()
    print("=" * 72)

    print(
        "FINAL OLD VS NEW COMBINED RESULTS"
    )

    print("=" * 72)

    print()
    print(
        "PRIMARY - MACRO VIDEO F1"
    )

    print(
        f"OLD COMBINED: {old_macro:.6f}"
    )

    print(
        f"NEW COMBINED: {new_macro:.6f}"
    )

    print(
        f"Delta NEW-OLD: {delta_macro:+.6f}"
    )

    print()
    print(
        "MICRO F1"
    )

    print(
        f"OLD COMBINED: {old_micro:.6f}"
    )

    print(
        f"NEW COMBINED: {new_micro:.6f}"
    )

    print(
        f"Delta NEW-OLD: {delta_micro:+.6f}"
    )

    print()
    print(
        "OVERALL METRICS"
    )

    display_columns = [
        "system",
        "macro_video_precision",
        "macro_video_recall",
        "macro_video_f1",
        "micro_precision",
        "micro_recall",
        "micro_f1",
        "mcc",
        "balanced_accuracy",
        "specificity",
        "fpr",
        "pr_auc",
        "roc_auc",
        "mAP_video",
        "TP",
        "FP",
        "FN",
        "TN",
    ]

    print(
        overall[
            display_columns
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "PER-SUBJECT MACRO VIDEO F1"
    )

    pivot = (
        subject_metrics.pivot(
            index="subject",
            columns="system",
            values="macro_video_f1",
        )
    )

    if (
        "OLD_COMBINED"
        in pivot.columns
        and "NEW_COMBINED_F"
        in pivot.columns
    ):

        pivot[
            "delta_NEW_minus_OLD"
        ] = (
            pivot[
                "NEW_COMBINED_F"
            ]
            - pivot[
                "OLD_COMBINED"
            ]
        )

    print(
        pivot.to_string()
    )

    print()
    print(
        "PAIRED ROBUSTNESS / STATISTICS"
    )

    print(
        "Subject-stratified paired "
        "bootstrap 95% CI:"
    )

    print(
        f"[{ci_low:.6f}, "
        f"{ci_high:.6f}]"
    )

    print(
        "Wilcoxon p:",
        f"{paired['wilcoxon_p']:.6f}",
    )

    print(
        "Paired t-test p:",
        f"{paired['paired_t_p']:.6f}",
    )

    print(
        "Cohen dz:",
        f"{paired['cohen_dz']:.6f}",
    )

    print(
        "McNemar exact p:",
        f"{paired['mcnemar_exact_p']:.6f}",
    )

    print()
    print(
        "McNemar discordant pairs:"
    )

    print(
        "OLD correct / NEW wrong:",
        paired[
            "mcnemar_OLD_correct_NEW_wrong"
        ],
    )

    print(
        "OLD wrong / NEW correct:",
        paired[
            "mcnemar_OLD_wrong_NEW_correct"
        ],
    )

    print()
    print(
        "Per-video:"
    )

    print(
        "Improved:",
        paired[
            "videos_improved"
        ],
    )

    print(
        "Worse:",
        paired[
            "videos_worse"
        ],
    )

    print(
        "Tied:",
        paired[
            "videos_tied"
        ],
    )

    print()
    print(
        "WINNER BY PRIMARY METRIC:",
        winner,
    )

    # =====================================================
    # SUMMARY TXT
    # =====================================================

    summary = f"""
FINAL OLD VS NEW COMBINED SYSTEM COMPARISON
===========================================

Dataset:
40 DEV videos
39 concepts
390 video-concept pairs
153 positive GT pairs

OLD COMBINED:
Transcript OLD + OCR D
60s | LDA=.40 | LSA=.60
Aggregation = {OLD_AGGREGATION}
Threshold = {OLD_THRESHOLD:.2f}

NEW COMBINED:
Transcript NEW + OCR F
180s | LDA=0 | LSA=1
OCR F = mean score fusion of D and E branches
Aggregation = {NEW_AGGREGATION}
Threshold = {NEW_THRESHOLD:.2f}

PRIMARY:
OLD Macro Video F1 = {old_macro:.6f}
NEW Macro Video F1 = {new_macro:.6f}
Delta NEW-OLD      = {delta_macro:+.6f}

SECONDARY:
OLD Micro F1 = {old_micro:.6f}
NEW Micro F1 = {new_micro:.6f}
Delta        = {delta_micro:+.6f}

Subject-stratified paired bootstrap 95% CI:
[{ci_low:.6f}, {ci_high:.6f}]

Wilcoxon p      = {paired["wilcoxon_p"]:.6f}
Paired t-test p = {paired["paired_t_p"]:.6f}
Cohen dz        = {paired["cohen_dz"]:.6f}
McNemar exact p = {paired["mcnemar_exact_p"]:.6f}

Per-video:
Improved = {paired["videos_improved"]}
Worse    = {paired["videos_worse"]}
Tied     = {paired["videos_tied"]}

Winner by primary Macro Video F1:
{winner}

Scientific status:
This is a locked DEV system comparison.
No threshold, aggregation, OCR strategy,
or model configuration was retuned in Step 13.

The final unseen TEST v41-v60 remains untouched.
""".strip()

    SUMMARY_FILE.write_text(
        summary,
        encoding="utf-8",
    )

    print()
    print(
        "Outputs:"
    )

    print(
        METRICS_FILE
    )

    print(
        VIDEO_FILE
    )

    print(
        SUBJECT_FILE
    )

    print(
        STATS_FILE
    )

    print(
        SUMMARY_FILE
    )

    print()
    print(
        "STEP 13 PASS - "
        "FINAL OLD VS NEW COMBINED "
        "COMPARISON COMPLETE"
    )


if __name__ == "__main__":
    main()