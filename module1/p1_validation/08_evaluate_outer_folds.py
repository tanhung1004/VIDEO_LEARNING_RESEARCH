from pathlib import Path
import json
import sys

import pandas as pd

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


ROOT = Path(__file__).resolve().parents[2]

# ============================================================
# PATHS
# ============================================================

OOF_SCORE_PATH = (
    ROOT
    / "module1"
    / "results"
    / "p1_validation"
    / "oof_chunk_concept_scores.csv"
)

SELECTED_PATH = (
    ROOT
    / "module1"
    / "results"
    / "p1_validation"
    / "selected_hyperparameters_by_outer_fold.csv"
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

PREDICTIONS_PATH = (
    OUTPUT_DIR
    / "oof_predictions.csv"
)

PER_VIDEO_PATH = (
    OUTPUT_DIR
    / "per_video_metrics.csv"
)

OUTER_METRICS_PATH = (
    OUTPUT_DIR
    / "outer_fold_metrics.csv"
)

AUDIT_PATH = (
    OUTPUT_DIR
    / "outer_evaluation_audit.json"
)


# ============================================================
# FROZEN PILOT CONFIG
#
# Evaluated as a PRE-SPECIFIED reference only.
# It is NOT selected using outer-test performance.
# ============================================================

FROZEN_ALPHA = 0.40
FROZEN_AGGREGATION = "top2_mean"
FROZEN_THRESHOLD = 0.40


def video_number(video_id):
    return int(
        str(video_id)
        .strip()
        .lower()
        .replace("v", "")
    )


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
            "outer_fold",
            "video_id",
            "subject",
            "chunk_id",
            "concept",
            "lda_norm",
            "lsa_norm",
        ]
    ].copy()

    # alpha = LDA weight
    # 1 - alpha = LSA weight
    work["final_score"] = (
        alpha
        * work["lda_norm"]
        +
        (1.0 - alpha)
        * work["lsa_norm"]
    )

    # Frozen pipeline behavior:
    # round chunk-level fused score before aggregation.
    work["final_score"] = (
        work["final_score"]
        .round(4)
    )

    grouped = work.groupby(
        [
            "outer_fold",
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
                    s.nlargest(2).mean()
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
                    s.nlargest(3).mean()
            )
            .reset_index(
                name="video_score"
            )
        )

    else:

        raise ValueError(
            f"Unknown aggregation: {aggregation}"
        )

    result["video_score"] = (
        result["video_score"]
        .round(4)
    )

    return result


# ============================================================
# BUILD PREDICTIONS FOR ONE OUTER FOLD
# ============================================================

def predict_fold(
    fold_scores,
    alpha,
    aggregation,
    threshold,
    positive_gt,
    variant,
):

    aggregated = aggregate_scores(
        scores=fold_scores,
        alpha=alpha,
        aggregation=aggregation,
    )

    aggregated["y_true"] = [
        1
        if (
            row.video_id,
            row.concept,
        )
        in positive_gt
        else 0
        for row in aggregated.itertuples()
    ]

    aggregated["y_pred"] = (
        aggregated["video_score"]
        >= threshold
    ).astype(int)

    aggregated["model_variant"] = (
        variant
    )

    aggregated["alpha"] = float(
        alpha
    )

    aggregated["aggregation"] = (
        aggregation
    )

    aggregated["threshold"] = float(
        threshold
    )

    return aggregated


# ============================================================
# METRICS
# ============================================================

def calculate_binary_metrics(
    y_true,
    y_pred,
):

    y_true = pd.Series(
        y_true
    ).astype(int)

    y_pred = pd.Series(
        y_pred
    ).astype(int)

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    tn, fp, fn, tp = (
        confusion_matrix(
            y_true,
            y_pred,
            labels=[0, 1],
        )
        .ravel()
    )

    return {
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "TN": int(tn),

        "precision":
            float(precision),

        "recall":
            float(recall),

        "f1":
            float(f1),
    }


def build_per_video_metrics(
    predictions,
):

    rows = []

    for (
        variant,
        fold,
        video_id,
        subject,
    ), group in predictions.groupby(
        [
            "model_variant",
            "outer_fold",
            "video_id",
            "subject",
        ],
        sort=False,
    ):

        metrics = calculate_binary_metrics(
            group["y_true"],
            group["y_pred"],
        )

        rows.append(
            {
                "model_variant":
                    variant,

                "outer_fold":
                    int(fold),

                "video_id":
                    video_id,

                "subject":
                    subject,

                "num_concepts":
                    len(group),

                "true_positive_labels":
                    int(
                        group["y_true"].sum()
                    ),

                "predicted_positive_labels":
                    int(
                        group["y_pred"].sum()
                    ),

                **metrics,
            }
        )

    result = pd.DataFrame(
        rows
    )

    result["_video_num"] = (
        result["video_id"]
        .str.extract(
            r"(\d+)",
            expand=False,
        )
        .astype(int)
    )

    result = (
        result
        .sort_values(
            [
                "model_variant",
                "outer_fold",
                "_video_num",
            ]
        )
        .drop(
            columns="_video_num"
        )
        .reset_index(drop=True)
    )

    return result


def build_outer_fold_metrics(
    predictions,
    per_video,
):

    rows = []

    for (
        variant,
        fold,
    ), group in predictions.groupby(
        [
            "model_variant",
            "outer_fold",
        ],
        sort=False,
    ):

        pv = per_video[
            per_video[
                "model_variant"
            ].eq(
                variant
            )
            &
            per_video[
                "outer_fold"
            ].eq(
                fold
            )
        ]

        micro = calculate_binary_metrics(
            group["y_true"],
            group["y_pred"],
        )

        rows.append(
            {
                "model_variant":
                    variant,

                "outer_fold":
                    int(fold),

                "heldout_videos":
                    group[
                        "video_id"
                    ].nunique(),

                "video_concept_pairs":
                    len(group),

                "positive_gt_pairs":
                    int(
                        group[
                            "y_true"
                        ].sum()
                    ),

                "predicted_positive_pairs":
                    int(
                        group[
                            "y_pred"
                        ].sum()
                    ),

                # Primary metrics:
                # mean across held-out videos
                "macro_video_precision":
                    float(
                        pv[
                            "precision"
                        ].mean()
                    ),

                "macro_video_recall":
                    float(
                        pv[
                            "recall"
                        ].mean()
                    ),

                "macro_video_f1":
                    float(
                        pv[
                            "f1"
                        ].mean()
                    ),

                # Secondary micro metrics
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

                "TP":
                    micro["TP"],

                "FP":
                    micro["FP"],

                "FN":
                    micro["FN"],

                "TN":
                    micro["TN"],
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("P1.8 OUTER HELD-OUT EVALUATION")
    print("=" * 80)

    # --------------------------------------------------------
    # 1. LOAD
    # --------------------------------------------------------

    scores = pd.read_csv(
        OOF_SCORE_PATH,
        dtype={
            "video_id": str,
            "subject": str,
            "concept": str,
        },
    )

    selected = pd.read_csv(
        SELECTED_PATH,
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

    # --------------------------------------------------------
    # 2. NORMALIZE TYPES
    # --------------------------------------------------------

    scores["video_id"] = (
        scores["video_id"]
        .str.strip()
        .str.lower()
    )

    scores["subject"] = (
        scores["subject"]
        .str.strip()
    )

    scores["concept"] = (
        scores["concept"]
        .str.strip()
    )

    scores["outer_fold"] = (
        pd.to_numeric(
            scores["outer_fold"],
            errors="raise",
        )
        .astype(int)
    )

    scores["chunk_id"] = (
        pd.to_numeric(
            scores["chunk_id"],
            errors="raise",
        )
        .astype(int)
    )

    scores["lda_norm"] = (
        pd.to_numeric(
            scores["lda_norm"],
            errors="raise",
        )
    )

    scores["lsa_norm"] = (
        pd.to_numeric(
            scores["lsa_norm"],
            errors="raise",
        )
    )

    selected["outer_fold"] = (
        pd.to_numeric(
            selected[
                "outer_fold"
            ],
            errors="raise",
        )
        .astype(int)
    )

    selected[
        "selected_alpha"
    ] = pd.to_numeric(
        selected[
            "selected_alpha"
        ],
        errors="raise",
    )

    selected[
        "selected_threshold"
    ] = pd.to_numeric(
        selected[
            "selected_threshold"
        ],
        errors="raise",
    )

    folds["video_id"] = (
        folds["video_id"]
        .str.strip()
        .str.lower()
    )

    folds["cv_fold"] = (
        pd.to_numeric(
            folds["cv_fold"],
            errors="raise",
        )
        .astype(int)
    )

    concepts["subject"] = (
        concepts["subject"]
        .str.strip()
    )

    concepts["concept"] = (
        concepts["concept"]
        .str.strip()
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
    # 3. INPUT AUDIT
    # --------------------------------------------------------

    if (
        scores[
            "video_id"
        ].nunique()
        != 40
    ):
        raise ValueError(
            "OOF scores must cover 40 videos"
        )

    if len(
        selected
    ) != 5:
        raise ValueError(
            "Expected exactly 5 selected "
            "outer-fold configurations"
        )

    if set(
        selected[
            "outer_fold"
        ]
    ) != {
        1, 2, 3, 4, 5
    }:
        raise ValueError(
            "Selected hyperparameters "
            "must cover folds 1..5"
        )

    if (
        folds[
            "video_id"
        ].nunique()
        != 40
    ):
        raise ValueError(
            "Fold manifest must cover 40 videos"
        )

    if len(
        concepts
    ) != 39:
        raise ValueError(
            "Concept catalog must contain 39 concepts"
        )

    # Positive-only GT is expected.
    if gt.duplicated(
        subset=[
            "video_id",
            "concept",
        ]
    ).any():
        raise ValueError(
            "Duplicate GT positive pairs found"
        )

    positive_gt = set(
        zip(
            gt["video_id"],
            gt["concept"],
        )
    )

    print("\nInput audit: PASS")
    print(
        "OOF videos      :",
        scores[
            "video_id"
        ].nunique(),
    )
    print(
        "OOF score rows  :",
        len(scores),
    )
    print(
        "GT positive rows:",
        len(gt),
    )
    print(
        "Selected configs:",
        len(selected),
    )

    # --------------------------------------------------------
    # 4. EVALUATE EACH OUTER FOLD
    # --------------------------------------------------------

    prediction_parts = []

    for fold in range(
        1,
        6,
    ):

        print("\n" + "=" * 80)
        print(
            f"OUTER FOLD {fold}"
        )
        print("=" * 80)

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

        expected_ids = set(
            folds.loc[
                folds[
                    "cv_fold"
                ].eq(
                    fold
                ),
                "video_id",
            ]
        )

        actual_ids = set(
            fold_scores[
                "video_id"
            ]
        )

        if (
            actual_ids
            != expected_ids
        ):
            raise ValueError(
                f"Fold {fold}: OOF video IDs "
                "do not match held-out manifest"
            )

        if len(
            actual_ids
        ) != 8:
            raise ValueError(
                f"Fold {fold}: expected "
                "8 held-out videos"
            )

        config = (
            selected[
                selected[
                    "outer_fold"
                ].eq(
                    fold
                )
            ]
        )

        if len(
            config
        ) != 1:
            raise ValueError(
                f"Fold {fold}: selected "
                "configuration not unique"
            )

        config = (
            config
            .iloc[0]
        )

        alpha = float(
            config[
                "selected_alpha"
            ]
        )

        aggregation = str(
            config[
                "selected_aggregation"
            ]
        )

        threshold = float(
            config[
                "selected_threshold"
            ]
        )

        # ----------------------------------------------------
        # A. NESTED-TUNED CONFIG
        #
        # Selected using outer-training videos only.
        # ----------------------------------------------------

        tuned_predictions = (
            predict_fold(
                fold_scores=
                    fold_scores,

                alpha=
                    alpha,

                aggregation=
                    aggregation,

                threshold=
                    threshold,

                positive_gt=
                    positive_gt,

                variant=
                    "nested_tuned",
            )
        )

        # ----------------------------------------------------
        # B. FROZEN ORIGINAL CONFIG
        #
        # Pre-specified reference.
        # No tuning from outer-test GT.
        # ----------------------------------------------------

        frozen_predictions = (
            predict_fold(
                fold_scores=
                    fold_scores,

                alpha=
                    FROZEN_ALPHA,

                aggregation=
                    FROZEN_AGGREGATION,

                threshold=
                    FROZEN_THRESHOLD,

                positive_gt=
                    positive_gt,

                variant=
                    "frozen_fixed",
            )
        )

        prediction_parts.extend(
            [
                tuned_predictions,
                frozen_predictions,
            ]
        )

        print(
            "Held-out videos:",
            len(actual_ids),
        )

        print(
            "Nested config  :",
            f"alpha={alpha}, "
            f"agg={aggregation}, "
            f"threshold={threshold}",
        )

        print(
            "Frozen config  :",
            "alpha=0.4, "
            "agg=top2_mean, "
            "threshold=0.4",
        )

        print(
            "Outer-test used for tuning: NO"
        )

        print(
            f"Fold {fold}: PREDICTIONS COMPLETE"
        )

    # --------------------------------------------------------
    # 5. COMBINE
    # --------------------------------------------------------

    predictions = pd.concat(
        prediction_parts,
        ignore_index=True,
    )

    # --------------------------------------------------------
    # 6. GLOBAL PREDICTION AUDIT
    # --------------------------------------------------------

    # 40 videos x subject-specific concept catalog:
    #
    # SQL    10 x 11 = 110
    # Python 10 x  9 =  90
    # Java   10 x  9 =  90
    # C++    10 x 10 = 100
    #
    # Total = 390 pairs per model variant.

    concept_counts = (
        concepts
        .groupby(
            "subject"
        )
        .size()
        .to_dict()
    )

    expected_pairs = int(
        sum(
            concept_counts[
                row.subject
            ]
            for row in (
                folds[
                    [
                        "video_id",
                        "subject",
                    ]
                ]
                .itertuples()
            )
        )
    )

    if expected_pairs != 390:
        raise ValueError(
            f"Expected 390 video-concept "
            f"pairs, calculated {expected_pairs}"
        )

    for variant in [
        "nested_tuned",
        "frozen_fixed",
    ]:

        v = predictions[
            predictions[
                "model_variant"
            ].eq(
                variant
            )
        ]

        if (
            v[
                "video_id"
            ].nunique()
            != 40
        ):
            raise ValueError(
                f"{variant}: video coverage != 40"
            )

        if len(
            v
        ) != expected_pairs:
            raise ValueError(
                f"{variant}: expected "
                f"{expected_pairs} pairs, "
                f"found {len(v)}"
            )

        duplicate_pairs = int(
            v.duplicated(
                subset=[
                    "video_id",
                    "concept",
                ]
            ).sum()
        )

        if duplicate_pairs != 0:
            raise ValueError(
                f"{variant}: duplicate "
                "video/concept predictions"
            )

        if int(
            v[
                "y_true"
            ].sum()
        ) != len(
            positive_gt
        ):
            raise ValueError(
                f"{variant}: GT positive count "
                "does not match source GT"
            )

    # --------------------------------------------------------
    # 7. METRICS
    # --------------------------------------------------------

    per_video = (
        build_per_video_metrics(
            predictions
        )
    )

    outer_metrics = (
        build_outer_fold_metrics(
            predictions,
            per_video,
        )
    )

    # --------------------------------------------------------
    # 8. GLOBAL SUMMARY FOR TERMINAL
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("GLOBAL OUT-OF-FOLD RESULTS")
    print("=" * 80)

    summary_rows = []

    for variant in [
        "nested_tuned",
        "frozen_fixed",
    ]:

        pred = predictions[
            predictions[
                "model_variant"
            ].eq(
                variant
            )
        ]

        pv = per_video[
            per_video[
                "model_variant"
            ].eq(
                variant
            )
        ]

        micro = calculate_binary_metrics(
            pred["y_true"],
            pred["y_pred"],
        )

        summary_rows.append(
            {
                "model_variant":
                    variant,

                "macro_video_precision":
                    pv[
                        "precision"
                    ].mean(),

                "macro_video_recall":
                    pv[
                        "recall"
                    ].mean(),

                "macro_video_f1":
                    pv[
                        "f1"
                    ].mean(),

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

                "TP":
                    micro["TP"],

                "FP":
                    micro["FP"],

                "FN":
                    micro["FN"],

                "TN":
                    micro["TN"],
            }
        )

    summary = pd.DataFrame(
        summary_rows
    )

    print(
        summary.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # 9. WRITE RESULTS
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions["_video_num"] = (
        predictions[
            "video_id"
        ]
        .str.extract(
            r"(\d+)",
            expand=False,
        )
        .astype(int)
    )

    predictions = (
        predictions
        .sort_values(
            [
                "model_variant",
                "outer_fold",
                "_video_num",
                "concept",
            ]
        )
        .drop(
            columns="_video_num"
        )
        .reset_index(
            drop=True
        )
    )

    predictions.to_csv(
        PREDICTIONS_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    per_video.to_csv(
        PER_VIDEO_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    outer_metrics.to_csv(
        OUTER_METRICS_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    audit = {
        "evaluation_stage":
            "P1.8",

        "dev_videos":
            40,

        "outer_folds":
            5,

        "heldout_videos_per_fold":
            8,

        "video_concept_pairs_per_variant":
            expected_pairs,

        "positive_ground_truth_pairs":
            len(
                positive_gt
            ),

        "model_variants": [
            "nested_tuned",
            "frozen_fixed",
        ],

        "nested_tuned_hyperparameters_source":
            (
                "P1.7 inner CV on "
                "outer-training videos only"
            ),

        "frozen_fixed_config": {
            "alpha":
                FROZEN_ALPHA,

            "aggregation":
                FROZEN_AGGREGATION,

            "threshold":
                FROZEN_THRESHOLD,
        },

        "outer_test_used_for_hyperparameter_selection":
            False,

        "outer_test_used_for_evaluation_only":
            True,

        "primary_metric":
            "mean video-level F1",

        "secondary_metrics":
            [
                "micro precision",
                "micro recall",
                "micro F1",
            ],
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
    # 10. FINAL REPORT
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("P1.8 OUTER HELD-OUT EVALUATION: PASS")
    print("=" * 80)

    print(
        "40/40 DEV videos were evaluated "
        "out-of-fold."
    )

    print(
        "Each video's prediction came from "
        "a model that did not fit that video."
    )

    print(
        "Nested hyperparameters were selected "
        "without outer-test GT."
    )

    print(
        "Outer-test GT was used for "
        "evaluation only."
    )

    print(
        "Saved:",
        PREDICTIONS_PATH.relative_to(
            ROOT
        ),
    )

    print(
        "Saved:",
        PER_VIDEO_PATH.relative_to(
            ROOT
        ),
    )

    print(
        "Saved:",
        OUTER_METRICS_PATH.relative_to(
            ROOT
        ),
    )

    print(
        "Saved:",
        AUDIT_PATH.relative_to(
            ROOT
        ),
    )


if __name__ == "__main__":
    main()