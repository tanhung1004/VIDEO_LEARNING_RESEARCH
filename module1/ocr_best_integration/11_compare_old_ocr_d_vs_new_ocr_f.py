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
    balanced_accuracy_score,
    average_precision_score,
    roc_auc_score,
)


# =========================================================
# CONFIG
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

D_SCORE_FILE = (
    PROJECT_ROOT
    / "module1"
    / "results"
    / "ocr_best_integration"
    / "ocr_only_comparison"
    / "D_fullframe"
    / "fusion_scores.csv"
)

F_SCORE_FILE = (
    PROJECT_ROOT
    / "module1"
    / "results"
    / "ocr_best_integration"
    / "ocr_only_comparison"
    / "F_mean"
    / "fusion_scores.csv"
)

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
    PROJECT_ROOT
    / "module1"
    / "results"
    / "ocr_best_integration"
    / "ocr_d_vs_f_comparison"
)

PAIR_FILE = (
    RESULT_DIR
    / "ocr_d_vs_f_pair_scores.csv"
)

VIDEO_FILE = (
    RESULT_DIR
    / "ocr_d_vs_f_video_metrics.csv"
)

SUBJECT_FILE = (
    RESULT_DIR
    / "ocr_d_vs_f_subject_metrics.csv"
)

METRICS_FILE = (
    RESULT_DIR
    / "ocr_d_vs_f_overall_metrics.csv"
)

STATS_FILE = (
    RESULT_DIR
    / "ocr_d_vs_f_statistics.csv"
)

BOOTSTRAP_FILE = (
    RESULT_DIR
    / "ocr_d_vs_f_bootstrap_deltas.csv"
)

AUDIT_FILE = (
    RESULT_DIR
    / "ocr_d_vs_f_audit.csv"
)

SUMMARY_FILE = (
    RESULT_DIR
    / "OCR_D_VS_F_SUMMARY.txt"
)


DEV_IDS = [
    f"v{i}"
    for i in range(1, 41)
]

EXPECTED_VIDEOS = 40
EXPECTED_CONCEPTS = 39
EXPECTED_PAIRS = 390
EXPECTED_GT_POSITIVES = 153


# =========================================================
# LOCKED OCR-ONLY EVALUATION CONFIG
#
# SAME for D and F.
# NO tuning in Step 11.
# =========================================================

AGGREGATION = "top2_mean"
TOP_K = 2

THRESHOLD = 0.40

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


def safe_divide(
    numerator,
    denominator,
):
    if denominator == 0:
        return 0.0

    return (
        numerator
        / denominator
    )


# =========================================================
# LOAD METADATA
# =========================================================

def load_videos():
    if not VIDEOS_FILE.exists():
        raise FileNotFoundError(
            VIDEOS_FILE
        )

    videos = pd.read_csv(
        VIDEOS_FILE
    )

    required = {
        "video_id",
        "subject",
    }

    missing = (
        required
        - set(
            videos.columns
        )
    )

    if missing:
        raise ValueError(
            f"videos.csv missing: "
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
            "DEV video metadata audit failed."
        )

    return videos[
        [
            "video_id",
            "subject",
            "subject_norm",
        ]
    ].copy()


def load_concepts():
    if not CONCEPT_FILE.exists():
        raise FileNotFoundError(
            CONCEPT_FILE
        )

    concepts = pd.read_csv(
        CONCEPT_FILE
    )

    required = {
        "subject",
        "concept",
    }

    missing = (
        required
        - set(
            concepts.columns
        )
    )

    if missing:
        raise ValueError(
            f"Concept file missing: "
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
    if not GT_FILE.exists():
        raise FileNotFoundError(
            GT_FILE
        )

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
        - set(
            gt.columns
        )
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
            f"Duplicate GT pairs: {duplicates}"
        )

    if len(gt) != EXPECTED_GT_POSITIVES:
        raise RuntimeError(
            f"Expected "
            f"{EXPECTED_GT_POSITIVES} "
            f"DEV GT positives, "
            f"found {len(gt)}."
        )

    return gt[
        [
            "video_id",
            "subject_norm",
            "concept_norm",
        ]
    ].copy()


# =========================================================
# BUILD COMPLETE 390-PAIR EVALUATION SPACE
# =========================================================

def build_pair_space(
    videos,
    concepts,
    gt,
):
    rows = []

    for _, video in videos.iterrows():

        video_id = (
            video[
                "video_id"
            ]
        )

        subject = (
            video[
                "subject"
            ]
        )

        subject_norm = (
            video[
                "subject_norm"
            ]
        )

        subject_concepts = (
            concepts[
                concepts[
                    "subject_norm"
                ]
                == subject_norm
            ]
        )

        if subject_concepts.empty:
            raise RuntimeError(
                f"No concepts for "
                f"{video_id} / {subject}"
            )

        for _, concept in (
            subject_concepts.iterrows()
        ):

            rows.append(
                {
                    "video_id":
                        video_id,

                    "subject":
                        subject,

                    "subject_norm":
                        subject_norm,

                    "concept":
                        concept[
                            "concept"
                        ],

                    "concept_norm":
                        concept[
                            "concept_norm"
                        ],
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
            ]
            .itertuples(
                index=False,
                name=None,
            ),
        )
    )

    pairs["y_true"] = (
        pairs.apply(
            lambda row:
                1
                if (
                    row[
                        "video_id"
                    ],
                    row[
                        "subject_norm"
                    ],
                    row[
                        "concept_norm"
                    ],
                )
                in gt_keys
                else 0,
            axis=1,
        )
        .astype(int)
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
        len(pairs)
        != EXPECTED_PAIRS
        or duplicates != 0
        or int(
            pairs[
                "y_true"
            ].sum()
        )
        != EXPECTED_GT_POSITIVES
    ):
        raise RuntimeError(
            "Evaluation pair-space audit failed."
        )

    return pairs


# =========================================================
# LOAD SCORE FILE
# =========================================================

def load_score_file(
    path,
    label,
):
    if not path.exists():
        raise FileNotFoundError(
            f"{label} score file missing:\n"
            f"{path}"
        )

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
        - set(
            df.columns
        )
    )

    if missing:
        raise ValueError(
            f"{label} score file missing: "
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

    if (
        df["final_score"]
        .isna()
        .any()
    ):
        raise RuntimeError(
            f"{label}: NaN scores detected."
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

    if duplicates != 0:
        raise RuntimeError(
            f"{label}: duplicate "
            f"chunk-concept rows = "
            f"{duplicates}"
        )

    return df


# =========================================================
# VIDEO-CONCEPT EVIDENCE
#
# Fixed OLD aggregation:
# top2_mean
# =========================================================

def build_video_evidence(
    scores,
    score_name,
):
    grouped = (
        scores
        .groupby(
            [
                "video_id",
                "subject_norm",
                "concept_norm",
            ],
            sort=False,
        )
    )

    rows = []

    for (
        video_id,
        subject_norm,
        concept_norm,
    ), group in grouped:

        values = (
            pd.to_numeric(
                group[
                    "final_score"
                ],
                errors="coerce",
            )
            .dropna()
            .astype(float)
            .sort_values(
                ascending=False
            )
        )

        if values.empty:
            evidence = 0.0
            top_count = 0

        else:
            top_values = (
                values.head(
                    TOP_K
                )
            )

            evidence = float(
                top_values.mean()
            )

            top_count = int(
                len(
                    top_values
                )
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
                    top_count,
            }
        )

    evidence = pd.DataFrame(
        rows
    )

    duplicates = int(
        evidence.duplicated(
            [
                "video_id",
                "subject_norm",
                "concept_norm",
            ]
        ).sum()
    )

    if duplicates != 0:
        raise RuntimeError(
            f"{score_name}: duplicate "
            f"video-concept evidence rows."
        )

    return evidence


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


def compute_pair_metrics(
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

    precision = (
        precision_score(
            y_true,
            y_pred,
            zero_division=0,
        )
    )

    recall = (
        recall_score(
            y_true,
            y_pred,
            zero_division=0,
        )
    )

    f1 = (
        f1_score(
            y_true,
            y_pred,
            zero_division=0,
        )
    )

    mcc = (
        matthews_corrcoef(
            y_true,
            y_pred,
        )
    )

    balanced_accuracy = (
        balanced_accuracy_score(
            y_true,
            y_pred,
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

    if len(
        np.unique(
            y_true
        )
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
        pr_auc = np.nan
        roc_auc = np.nan

    return {
        "TP":
            tp,

        "FP":
            fp,

        "FN":
            fn,

        "TN":
            tn,

        "precision":
            float(
                precision
            ),

        "recall":
            float(
                recall
            ),

        "f1":
            float(
                f1
            ),

        "specificity":
            float(
                specificity
            ),

        "fpr":
            float(
                fpr
            ),

        "mcc":
            float(
                mcc
            ),

        "balanced_accuracy":
            float(
                balanced_accuracy
            ),

        "pr_auc":
            float(
                pr_auc
            ),

        "roc_auc":
            float(
                roc_auc
            ),

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
# PER-VIDEO METRICS
# =========================================================

def build_video_metrics(
    pairs,
):
    rows = []

    grouped = pairs.groupby(
        [
            "video_id",
            "subject",
        ],
        sort=False,
    )

    for (
        video_id,
        subject,
    ), group in grouped:

        row = {
            "video_id":
                video_id,

            "subject":
                subject,

            "concepts":
                len(group),

            "gt_positive":
                int(
                    group[
                        "y_true"
                    ].sum()
                ),
        }

        for system in [
            "D",
            "F",
        ]:

            metrics = (
                compute_pair_metrics(
                    group[
                        "y_true"
                    ],
                    group[
                        f"pred_{system}"
                    ],
                    group[
                        f"score_{system}"
                    ],
                )
            )

            for key, value in (
                metrics.items()
            ):

                row[
                    f"{system}_{key}"
                ] = value

        row[
            "delta_f1_F_minus_D"
        ] = (
            row[
                "F_f1"
            ]
            - row[
                "D_f1"
            ]
        )

        rows.append(
            row
        )

    result = pd.DataFrame(
        rows
    )

    result[
        "_video_order"
    ] = (
        result[
            "video_id"
        ]
        .map(
            video_sort_key
        )
    )

    result = (
        result
        .sort_values(
            "_video_order"
        )
        .drop(
            columns=[
                "_video_order"
            ]
        )
        .reset_index(
            drop=True
        )
    )

    if len(result) != EXPECTED_VIDEOS:
        raise RuntimeError(
            "Per-video metrics row count "
            "is not 40."
        )

    return result


# =========================================================
# mAP ACROSS VIDEOS
# =========================================================

def calculate_map(
    pairs,
    system,
):
    ap_values = []

    for _, group in (
        pairs.groupby(
            "video_id"
        )
    ):

        y_true = (
            group[
                "y_true"
            ]
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

        ap_values.append(
            float(
                ap
            )
        )

    if not ap_values:
        return np.nan

    return float(
        np.mean(
            ap_values
        )
    )


# =========================================================
# OVERALL METRICS
# =========================================================

def build_overall_metrics(
    pairs,
    video_metrics,
):
    rows = []

    for system in [
        "D",
        "F",
    ]:

        pair_metrics = (
            compute_pair_metrics(
                pairs[
                    "y_true"
                ],
                pairs[
                    f"pred_{system}"
                ],
                pairs[
                    f"score_{system}"
                ],
            )
        )

        macro_video_precision = float(
            video_metrics[
                f"{system}_precision"
            ].mean()
        )

        macro_video_recall = float(
            video_metrics[
                f"{system}_recall"
            ].mean()
        )

        macro_video_f1 = float(
            video_metrics[
                f"{system}_f1"
            ].mean()
        )

        macro_video_f1_std = float(
            video_metrics[
                f"{system}_f1"
            ].std(
                ddof=1
            )
        )

        map_value = (
            calculate_map(
                pairs,
                system,
            )
        )

        rows.append(
            {
                "system":
                    (
                        "OLD_OCR_D"
                        if system
                        == "D"
                        else "NEW_OCR_F"
                    ),

                "aggregation":
                    AGGREGATION,

                "threshold":
                    THRESHOLD,

                "macro_video_precision":
                    macro_video_precision,

                "macro_video_recall":
                    macro_video_recall,

                "macro_video_f1":
                    macro_video_f1,

                "macro_video_f1_std":
                    macro_video_f1_std,

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
                    map_value,

                "TP":
                    pair_metrics[
                        "TP"
                    ],

                "FP":
                    pair_metrics[
                        "FP"
                    ],

                "FN":
                    pair_metrics[
                        "FN"
                    ],

                "TN":
                    pair_metrics[
                        "TN"
                    ],

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
# SUBJECT METRICS
# =========================================================

def build_subject_metrics(
    pairs,
    video_metrics,
):
    rows = []

    subjects = (
        sorted(
            pairs[
                "subject"
            ]
            .unique()
            .tolist()
        )
    )

    for subject in subjects:

        pair_subset = (
            pairs[
                pairs[
                    "subject"
                ]
                == subject
            ]
        )

        video_subset = (
            video_metrics[
                video_metrics[
                    "subject"
                ]
                == subject
            ]
        )

        for system in [
            "D",
            "F",
        ]:

            metrics = (
                compute_pair_metrics(
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
            )

            rows.append(
                {
                    "subject":
                        subject,

                    "system":
                        (
                            "OLD_OCR_D"
                            if system
                            == "D"
                            else "NEW_OCR_F"
                        ),

                    "videos":
                        video_subset[
                            "video_id"
                        ].nunique(),

                    "pairs":
                        len(
                            pair_subset
                        ),

                    "macro_video_f1":
                        float(
                            video_subset[
                                f"{system}_f1"
                            ].mean()
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

                    "micro_f1":
                        metrics[
                            "f1"
                        ],

                    "micro_precision":
                        metrics[
                            "precision"
                        ],

                    "micro_recall":
                        metrics[
                            "recall"
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

def stratified_paired_bootstrap(
    video_metrics,
):
    rng = np.random.default_rng(
        BOOTSTRAP_RANDOM_STATE
    )

    subject_groups = {
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

        sampled_rows = []

        for subject, group in (
            subject_groups.items()
        ):

            sample_indices = (
                rng.integers(
                    0,
                    len(group),
                    size=len(group),
                )
            )

            sampled = (
                group.iloc[
                    sample_indices
                ]
            )

            sampled_rows.append(
                sampled
            )

        sample = pd.concat(
            sampled_rows,
            ignore_index=True,
        )

        d_macro = float(
            sample[
                "D_f1"
            ].mean()
        )

        f_macro = float(
            sample[
                "F_f1"
            ].mean()
        )

        deltas.append(
            f_macro
            - d_macro
        )

    deltas = np.asarray(
        deltas,
        dtype=float,
    )

    lower = float(
        np.percentile(
            deltas,
            2.5,
        )
    )

    upper = float(
        np.percentile(
            deltas,
            97.5,
        )
    )

    return (
        deltas,
        lower,
        upper,
    )


# =========================================================
# PAIRED STATISTICS
# =========================================================

def paired_statistics(
    pairs,
    video_metrics,
):
    d_f1 = (
        video_metrics[
            "D_f1"
        ]
        .to_numpy(
            dtype=float
        )
    )

    f_f1 = (
        video_metrics[
            "F_f1"
        ]
        .to_numpy(
            dtype=float
        )
    )

    delta = (
        f_f1
        - d_f1
    )

    # -----------------------------------------------------
    # Wilcoxon
    # -----------------------------------------------------

    if np.allclose(
        delta,
        0.0,
    ):

        wilcoxon_stat = 0.0
        wilcoxon_p = 1.0

    else:

        wilcoxon_result = (
            stats.wilcoxon(
                f_f1,
                d_f1,
                zero_method="wilcox",
                alternative="two-sided",
            )
        )

        wilcoxon_stat = float(
            wilcoxon_result.statistic
        )

        wilcoxon_p = float(
            wilcoxon_result.pvalue
        )

    # -----------------------------------------------------
    # Paired t-test
    # -----------------------------------------------------

    t_result = (
        stats.ttest_rel(
            f_f1,
            d_f1,
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
            delta,
            ddof=1,
        )
    )

    if delta_std == 0:
        cohen_dz = 0.0

    else:
        cohen_dz = float(
            np.mean(
                delta
            )
            / delta_std
        )

    # -----------------------------------------------------
    # McNemar exact on the SAME 390 pair predictions
    # -----------------------------------------------------

    y_true = (
        pairs[
            "y_true"
        ]
        .to_numpy(
            dtype=int
        )
    )

    d_pred = (
        pairs[
            "pred_D"
        ]
        .to_numpy(
            dtype=int
        )
    )

    f_pred = (
        pairs[
            "pred_F"
        ]
        .to_numpy(
            dtype=int
        )
    )

    d_correct = (
        d_pred
        == y_true
    )

    f_correct = (
        f_pred
        == y_true
    )

    d_correct_f_wrong = int(
        np.sum(
            d_correct
            & (
                ~f_correct
            )
        )
    )

    d_wrong_f_correct = int(
        np.sum(
            (
                ~d_correct
            )
            & f_correct
        )
    )

    discordant = (
        d_correct_f_wrong
        + d_wrong_f_correct
    )

    if discordant == 0:
        mcnemar_p = 1.0

    else:
        mcnemar_p = float(
            binomtest(
                d_wrong_f_correct,
                n=discordant,
                p=0.5,
                alternative="two-sided",
            ).pvalue
        )

    improved = int(
        np.sum(
            delta > 1e-12
        )
    )

    worse = int(
        np.sum(
            delta < -1e-12
        )
    )

    tied = int(
        len(delta)
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

        "mcnemar_D_correct_F_wrong":
            d_correct_f_wrong,

        "mcnemar_D_wrong_F_correct":
            d_wrong_f_correct,

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
        "STEP 11 - COMPARE OLD OCR D VS NEW OCR F"
    )

    print("=" * 72)

    print(
        "Comparison:"
    )

    print(
        "OLD OCR D = full-frame cleaned OCR"
    )

    print(
        "NEW OCR F = mean score fusion of D + ROI E"
    )

    print()

    print(
        "Evaluation configuration LOCKED:"
    )

    print(
        f"aggregation = {AGGREGATION}"
    )

    print(
        f"threshold   = {THRESHOLD:.2f}"
    )

    print(
        "No threshold tuning in Step 11."
    )

    print(
        "Same 40 DEV videos / same GT / same concepts."
    )

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # =====================================================
    # LOAD
    # =====================================================

    videos = load_videos()

    concepts = load_concepts()

    gt = load_gt()

    pair_space = (
        build_pair_space(
            videos,
            concepts,
            gt,
        )
    )

    print()
    print(
        "========== EVALUATION SPACE =========="
    )

    print(
        "DEV videos:",
        len(
            videos
        ),
    )

    print(
        "Concepts:",
        len(
            concepts
        ),
    )

    print(
        "Video-concept pairs:",
        len(
            pair_space
        ),
    )

    print(
        "GT positive pairs:",
        int(
            pair_space[
                "y_true"
            ].sum()
        ),
    )

    # =====================================================
    # D / F SCORES
    # =====================================================

    d_scores = (
        load_score_file(
            D_SCORE_FILE,
            "OLD OCR D",
        )
    )

    f_scores = (
        load_score_file(
            F_SCORE_FILE,
            "NEW OCR F",
        )
    )

    print()
    print(
        "D chunk-concept score rows:",
        len(
            d_scores
        ),
    )

    print(
        "F chunk-concept score rows:",
        len(
            f_scores
        ),
    )

    # =====================================================
    # VIDEO EVIDENCE
    # =====================================================

    d_evidence = (
        build_video_evidence(
            d_scores,
            "score_D",
        )
    )

    f_evidence = (
        build_video_evidence(
            f_scores,
            "score_F",
        )
    )

    print()
    print(
        "D video-concept evidence rows:",
        len(
            d_evidence
        ),
    )

    print(
        "F video-concept evidence rows:",
        len(
            f_evidence
        ),
    )

    # =====================================================
    # MERGE INTO COMPLETE PAIR SPACE
    # =====================================================

    pairs = pair_space.merge(
        d_evidence,
        on=[
            "video_id",
            "subject_norm",
            "concept_norm",
        ],
        how="left",
        validate="one_to_one",
    )

    pairs = pairs.merge(
        f_evidence,
        on=[
            "video_id",
            "subject_norm",
            "concept_norm",
        ],
        how="left",
        validate="one_to_one",
    )

    missing_d = int(
        pairs[
            "score_D"
        ].isna().sum()
    )

    missing_f = int(
        pairs[
            "score_F"
        ].isna().sum()
    )

    if (
        missing_d != 0
        or missing_f != 0
    ):

        raise RuntimeError(
            "Missing D/F evidence "
            f"(D={missing_d}, F={missing_f})."
        )

    pairs["pred_D"] = (
        pairs[
            "score_D"
        ]
        >= THRESHOLD
    ).astype(int)

    pairs["pred_F"] = (
        pairs[
            "score_F"
        ]
        >= THRESHOLD
    ).astype(int)

    # =====================================================
    # VIDEO METRICS
    # =====================================================

    video_metrics = (
        build_video_metrics(
            pairs
        )
    )

    # =====================================================
    # OVERALL
    # =====================================================

    overall = (
        build_overall_metrics(
            pairs,
            video_metrics,
        )
    )

    # =====================================================
    # SUBJECT
    # =====================================================

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
        bootstrap_ci_low,
        bootstrap_ci_high,
    ) = (
        stratified_paired_bootstrap(
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

            "delta_macro_video_f1_F_minus_D":
                bootstrap_deltas,
        }
    )

    # =====================================================
    # PAIRED STATISTICS
    # =====================================================

    paired_stats = (
        paired_statistics(
            pairs,
            video_metrics,
        )
    )

    d_macro = float(
        overall.loc[
            overall[
                "system"
            ]
            == "OLD_OCR_D",
            "macro_video_f1",
        ].iloc[0]
    )

    f_macro = float(
        overall.loc[
            overall[
                "system"
            ]
            == "NEW_OCR_F",
            "macro_video_f1",
        ].iloc[0]
    )

    delta_macro = (
        f_macro
        - d_macro
    )

    d_micro = float(
        overall.loc[
            overall[
                "system"
            ]
            == "OLD_OCR_D",
            "micro_f1",
        ].iloc[0]
    )

    f_micro = float(
        overall.loc[
            overall[
                "system"
            ]
            == "NEW_OCR_F",
            "micro_f1",
        ].iloc[0]
    )

    delta_micro = (
        f_micro
        - d_micro
    )

    statistics_row = {
        "primary_metric":
            "macro_video_f1",

        "D_macro_video_f1":
            d_macro,

        "F_macro_video_f1":
            f_macro,

        "delta_macro_video_f1_F_minus_D":
            delta_macro,

        "D_micro_f1":
            d_micro,

        "F_micro_f1":
            f_micro,

        "delta_micro_f1_F_minus_D":
            delta_micro,

        "bootstrap_iterations":
            BOOTSTRAP_ITERATIONS,

        "bootstrap_seed":
            BOOTSTRAP_RANDOM_STATE,

        "bootstrap_95ci_low":
            bootstrap_ci_low,

        "bootstrap_95ci_high":
            bootstrap_ci_high,

        **paired_stats,
    }

    statistics_df = pd.DataFrame(
        [
            statistics_row
        ]
    )

    # =====================================================
    # AUDIT
    # =====================================================

    pair_key_duplicates = int(
        pairs.duplicated(
            [
                "video_id",
                "concept_norm",
            ]
        ).sum()
    )

    score_d_range_bad = int(
        (
            (
                pairs[
                    "score_D"
                ]
                < -1e-9
            )
            |
            (
                pairs[
                    "score_D"
                ]
                > 1.0
                + 1e-9
            )
        ).sum()
    )

    score_f_range_bad = int(
        (
            (
                pairs[
                    "score_F"
                ]
                < -1e-9
            )
            |
            (
                pairs[
                    "score_F"
                ]
                > 1.0
                + 1e-9
            )
        ).sum()
    )

    audit = pd.DataFrame(
        [
            {
                "check":
                    "DEV_videos",

                "value":
                    videos[
                        "video_id"
                    ].nunique(),

                "expected":
                    EXPECTED_VIDEOS,

                "pass":
                    videos[
                        "video_id"
                    ].nunique()
                    == EXPECTED_VIDEOS,
            },

            {
                "check":
                    "concepts",

                "value":
                    len(
                        concepts
                    ),

                "expected":
                    EXPECTED_CONCEPTS,

                "pass":
                    len(
                        concepts
                    )
                    == EXPECTED_CONCEPTS,
            },

            {
                "check":
                    "evaluation_pairs",

                "value":
                    len(
                        pairs
                    ),

                "expected":
                    EXPECTED_PAIRS,

                "pass":
                    len(
                        pairs
                    )
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
                    pair_key_duplicates,

                "expected":
                    0,

                "pass":
                    pair_key_duplicates
                    == 0,
            },

            {
                "check":
                    "missing_D_evidence",

                "value":
                    missing_d,

                "expected":
                    0,

                "pass":
                    missing_d
                    == 0,
            },

            {
                "check":
                    "missing_F_evidence",

                "value":
                    missing_f,

                "expected":
                    0,

                "pass":
                    missing_f
                    == 0,
            },

            {
                "check":
                    "D_score_out_of_range",

                "value":
                    score_d_range_bad,

                "expected":
                    0,

                "pass":
                    score_d_range_bad
                    == 0,
            },

            {
                "check":
                    "F_score_out_of_range",

                "value":
                    score_f_range_bad,

                "expected":
                    0,

                "pass":
                    score_f_range_bad
                    == 0,
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
            "Step 11 audit failed."
        )

    # =====================================================
    # SAVE OUTPUTS
    # =====================================================

    pairs[
        "_video_order"
    ] = (
        pairs[
            "video_id"
        ]
        .map(
            video_sort_key
        )
    )

    pairs = (
        pairs
        .sort_values(
            [
                "_video_order",
                "concept",
            ]
        )
        .drop(
            columns=[
                "_video_order"
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
            "NEW OCR F"
        )

    elif delta_macro < -1e-12:
        winner = (
            "OLD OCR D"
        )

    else:
        winner = (
            "TIE"
        )

    # =====================================================
    # PRINT
    # =====================================================

    print()
    print("=" * 72)

    print(
        "OCR D VS F FINAL RESULTS"
    )

    print("=" * 72)

    print()
    print(
        "PRIMARY - MACRO VIDEO F1"
    )

    print(
        f"OLD OCR D: {d_macro:.6f}"
    )

    print(
        f"NEW OCR F: {f_macro:.6f}"
    )

    print(
        f"Delta F-D: {delta_macro:+.6f}"
    )

    print()

    print(
        "MICRO F1"
    )

    print(
        f"OLD OCR D: {d_micro:.6f}"
    )

    print(
        f"NEW OCR F: {f_micro:.6f}"
    )

    print(
        f"Delta F-D: {delta_micro:+.6f}"
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
        ]
        .to_string(
            index=False
        )
    )

    print()
    print(
        "PER-SUBJECT MACRO VIDEO F1"
    )

    subject_pivot = (
        subject_metrics.pivot(
            index="subject",
            columns="system",
            values="macro_video_f1",
        )
    )

    if (
        "OLD_OCR_D"
        in subject_pivot.columns
        and "NEW_OCR_F"
        in subject_pivot.columns
    ):

        subject_pivot[
            "delta_F_minus_D"
        ] = (
            subject_pivot[
                "NEW_OCR_F"
            ]
            - subject_pivot[
                "OLD_OCR_D"
            ]
        )

    print(
        subject_pivot.to_string()
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
        f"[{bootstrap_ci_low:.6f}, "
        f"{bootstrap_ci_high:.6f}]"
    )

    print(
        "Wilcoxon p:",
        f"{paired_stats['wilcoxon_p']:.6f}",
    )

    print(
        "Paired t-test p:",
        f"{paired_stats['paired_t_p']:.6f}",
    )

    print(
        "Cohen dz:",
        f"{paired_stats['cohen_dz']:.6f}",
    )

    print(
        "McNemar exact p:",
        f"{paired_stats['mcnemar_exact_p']:.6f}",
    )

    print()

    print(
        "McNemar discordant pairs:"
    )

    print(
        "D correct / F wrong:",
        paired_stats[
            "mcnemar_D_correct_F_wrong"
        ],
    )

    print(
        "D wrong / F correct:",
        paired_stats[
            "mcnemar_D_wrong_F_correct"
        ],
    )

    print()

    print(
        "Per-video:"
    )

    print(
        "Improved:",
        paired_stats[
            "videos_improved"
        ],
    )

    print(
        "Worse:",
        paired_stats[
            "videos_worse"
        ],
    )

    print(
        "Tied:",
        paired_stats[
            "videos_tied"
        ],
    )

    print()

    print(
        "WINNER BY PRIMARY METRIC:",
        winner,
    )

    print()

    print(
        "NOTE:"
    )

    print(
        "This is a DEV OCR-only comparison."
    )

    print(
        "F was predefined before this evaluation; "
        "Step 11 performs no retuning."
    )

    # =====================================================
    # TEXT SUMMARY
    # =====================================================

    summary_text = f"""
OCR D VS F COMPARISON
=====================

Dataset:
40 DEV videos
39 concepts
390 video-concept pairs
153 positive GT pairs

Evaluation:
Aggregation = {AGGREGATION}
Threshold = {THRESHOLD:.2f}
No tuning performed in Step 11.

OLD OCR D:
Macro Video F1 = {d_macro:.6f}
Micro F1       = {d_micro:.6f}

NEW OCR F:
Macro Video F1 = {f_macro:.6f}
Micro F1       = {f_micro:.6f}

Delta Macro F1 (F-D) = {delta_macro:+.6f}
Delta Micro F1 (F-D) = {delta_micro:+.6f}

Subject-stratified paired bootstrap 95% CI:
[{bootstrap_ci_low:.6f}, {bootstrap_ci_high:.6f}]

Wilcoxon p       = {paired_stats["wilcoxon_p"]:.6f}
Paired t-test p  = {paired_stats["paired_t_p"]:.6f}
Cohen dz         = {paired_stats["cohen_dz"]:.6f}
McNemar exact p  = {paired_stats["mcnemar_exact_p"]:.6f}

Per-video:
Improved = {paired_stats["videos_improved"]}
Worse    = {paired_stats["videos_worse"]}
Tied     = {paired_stats["videos_tied"]}

Winner by primary Macro Video F1:
{winner}

Scientific status:
Development/supplementary OCR-only comparison.
NEW OCR F was predefined before this evaluation.
No threshold, aggregation, OCR method, or fusion retuning
was performed using these comparison results.
""".strip()

    SUMMARY_FILE.write_text(
        summary_text,
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
        "STEP 11 PASS - "
        "OLD OCR D VS NEW OCR F COMPARISON COMPLETE"
    )


if __name__ == "__main__":
    main()