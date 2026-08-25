from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

RESULT_DIR = (
    ROOT
    / "module1"
    / "results"
    / "p1_validation"
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

GT_PATH = (
    ROOT
    / "data"
    / "ground_truth"
    / "ground_truth_concepts.csv"
)

BOUNDARY_CONFIG_PATH = (
    RESULT_DIR
    / "boundary_extended_hyperparameters.json"
)

BOUNDARY_SEARCH_PATH = (
    RESULT_DIR
    / "boundary_extended_search_results.csv"
)

BOUNDARY_AUDIT_PATH = (
    RESULT_DIR
    / "boundary_extension_audit.json"
)

BOUNDARY_RAW_PATH = (
    RESULT_DIR
    / "boundary_extended_raw_oof_scores.csv"
)

FINAL_CONFIG_PATH = (
    RESULT_DIR
    / "FINAL_MODEL_CONFIG.json"
)

FINAL_AUDIT_PATH = (
    RESULT_DIR
    / "p1_final_lock_audit.json"
)

FINAL_SUMMARY_PATH = (
    RESULT_DIR
    / "P1_FINAL_LOCK_SUMMARY.txt"
)


# ============================================================
# LOCKED P1.13b SEARCH SPACE
# ============================================================

EXPECTED_CHUNKS = [
    30,
    60,
    90,
    120,
    150,
    180,
]

EXPECTED_ALPHAS = [
    round(x / 10, 1)
    for x in range(11)
]

EXPECTED_AGGREGATIONS = [
    "max",
    "mean",
    "top2_mean",
    "top3_mean",
]

EXPECTED_THRESHOLDS = [
    round(0.10 + 0.05 * i, 2)
    for i in range(13)
]

EXPECTED_SEARCH_ROWS = (
    len(EXPECTED_CHUNKS)
    * len(EXPECTED_ALPHAS)
    * len(EXPECTED_AGGREGATIONS)
    * len(EXPECTED_THRESHOLDS)
)

EXPECTED_DEV_VIDEOS = 40
EXPECTED_DEV_GT_POSITIVES = 153
EXPECTED_FOLDS = {1, 2, 3, 4, 5}


# ============================================================
# HELPERS
# ============================================================

def require_file(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing required file: "
            f"{path.relative_to(ROOT)}"
        )


def normalize_video_id(series):
    return (
        series
        .astype(str)
        .str.strip()
        .str.lower()
    )


def normalize_subject(series):
    return (
        series
        .astype(str)
        .str.strip()
        .str.lower()
    )


def json_default(value):
    if isinstance(value, np.bool_):
        return bool(value)

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    if isinstance(value, np.ndarray):
        return value.tolist()

    raise TypeError(
        f"{type(value).__name__} "
        "is not JSON serializable"
    )


def exact_float_set(values, decimals=8):
    return {
        round(float(x), decimals)
        for x in values
    }


def config_core(
    chunk_seconds,
    alpha,
    aggregation,
    threshold,
):
    return {
        "chunk_seconds":
            int(chunk_seconds),

        "alpha":
            float(alpha),

        "lsa_weight":
            float(1.0 - alpha),

        "aggregation":
            str(aggregation),

        "threshold":
            float(threshold),
    }


def fingerprint(data):
    canonical = json.dumps(
        data,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )

    return hashlib.sha256(
        canonical.encode("utf-8")
    ).hexdigest()


def selected_mask(series):
    return (
        series
        .astype(str)
        .str.strip()
        .str.lower()
        .eq("true")
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 82)
    print(
        "P1.15 FINAL AUDIT + "
        "CONFIGURATION LOCK"
    )
    print("=" * 82)

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for path in [
        VIDEOS_PATH,
        FOLDS_PATH,
        GT_PATH,
        BOUNDARY_CONFIG_PATH,
        BOUNDARY_SEARCH_PATH,
        BOUNDARY_AUDIT_PATH,
        BOUNDARY_RAW_PATH,
    ]:
        require_file(path)

    checks = []

    def check(name, condition, detail):
        condition = bool(condition)

        checks.append(
            {
                "check":
                    name,

                "status":
                    (
                        "PASS"
                        if condition
                        else "FAIL"
                    ),

                "detail":
                    str(detail),
            }
        )

        if not condition:
            raise ValueError(
                f"{name}: {detail}"
            )

    # ========================================================
    # 1. DATASET / DEV SPLIT AUDIT
    # ========================================================

    print(
        "\n1. DEV DATASET / "
        "FOLD AUDIT"
    )

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
    ] = normalize_subject(
        videos[
            "subject"
        ]
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
    )

    check(
        "exactly_40_dev_videos",
        (
            len(dev)
            == EXPECTED_DEV_VIDEOS
            and dev[
                "video_id"
            ].nunique()
            == EXPECTED_DEV_VIDEOS
        ),
        (
            f"rows={len(dev)}, "
            f"unique="
            f"{dev['video_id'].nunique()}"
        ),
    )

    check(
        "dev_video_ids_unique",
        not dev[
            "video_id"
        ].duplicated().any(),
        (
            "no duplicate DEV "
            "video_id allowed"
        ),
    )

    subject_counts = (
        dev[
            "subject"
        ]
        .value_counts()
        .to_dict()
    )

    check(
        "balanced_dev_subjects",
        subject_counts
        == {
            "sql": 10,
            "python": 10,
            "java": 10,
            "c++": 10,
        },
        subject_counts,
    )

    dev_ids = set(
        dev[
            "video_id"
        ]
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

    check(
        "folds_cover_exact_dev_ids",
        (
            len(folds)
            == EXPECTED_DEV_VIDEOS
            and set(
                folds[
                    "video_id"
                ]
            )
            == dev_ids
        ),
        (
            f"fold rows={len(folds)}, "
            "fold IDs must equal DEV IDs"
        ),
    )

    check(
        "five_expected_folds",
        set(
            folds[
                "cv_fold"
            ].unique()
        )
        == EXPECTED_FOLDS,
        sorted(
            folds[
                "cv_fold"
            ].unique()
        ),
    )

    fold_counts = (
        folds[
            "cv_fold"
        ]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    check(
        "eight_videos_per_fold",
        all(
            int(count) == 8
            for count
            in fold_counts.values()
        )
        and len(
            fold_counts
        ) == 5,
        fold_counts,
    )

    print(
        "DEV videos : 40"
    )

    print(
        "Subjects   : "
        f"{subject_counts}"
    )

    print(
        "Folds      : "
        f"{fold_counts}"
    )

    # ========================================================
    # 2. DEV GROUND TRUTH AUDIT
    # ========================================================

    print(
        "\n2. DEV GROUND TRUTH "
        "AUDIT"
    )

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

    dev_gt = (
        gt[
            gt[
                "video_id"
            ].isin(
                dev_ids
            )
        ]
        .copy()
    )

    dev_gt_unique = (
        dev_gt[
            [
                "video_id",
                "concept",
            ]
        ]
        .drop_duplicates()
    )

    check(
        "dev_gt_153_positive_pairs",
        len(
            dev_gt_unique
        )
        == EXPECTED_DEV_GT_POSITIVES,
        (
            f"positive DEV pairs="
            f"{len(dev_gt_unique)}"
        ),
    )

    check(
        "dev_gt_no_duplicate_pairs",
        len(
            dev_gt
        )
        == len(
            dev_gt_unique
        ),
        (
            "duplicate "
            "(video_id, concept) "
            "not allowed"
        ),
    )

    print(
        "DEV GT positive pairs : "
        f"{len(dev_gt_unique)}"
    )

    # ========================================================
    # 3. P1.13b SEARCH AUDIT
    # ========================================================

    print(
        "\n3. P1.13b SEARCH / "
        "SELECTION AUDIT"
    )

    boundary_config = json.loads(
        BOUNDARY_CONFIG_PATH
        .read_text(
            encoding="utf-8"
        )
    )

    boundary_audit = json.loads(
        BOUNDARY_AUDIT_PATH
        .read_text(
            encoding="utf-8"
        )
    )

    search = pd.read_csv(
        BOUNDARY_SEARCH_PATH
    )

    check(
        "p1_13b_audit_pass",
        str(
            boundary_audit.get(
                "status",
                ""
            )
        ).strip().upper()
        == "PASS",
        boundary_audit.get(
            "status"
        ),
    )

    check(
        "p1_13b_final_test_not_used",
        (
            boundary_audit.get(
                "final_test_used"
            )
            is False
            and boundary_config.get(
                "final_test_used"
            )
            is False
        ),
        (
            "both P1.13b audit "
            "and config must say "
            "final_test_used=False"
        ),
    )

    check(
        "one_time_extension_stop_rule",
        (
            boundary_audit.get(
                "further_grid_extension_allowed"
            )
            is False
        ),
        (
            "P1.13b stop rule must "
            "forbid further grid "
            "extension"
        ),
    )

    check(
        "search_has_3432_configs",
        len(
            search
        )
        == EXPECTED_SEARCH_ROWS,
        (
            f"rows={len(search)}, "
            f"expected="
            f"{EXPECTED_SEARCH_ROWS}"
        ),
    )

    actual_chunks = sorted(
        int(x)
        for x in (
            search[
                "chunk_seconds"
            ].unique()
        )
    )

    actual_alphas = sorted(
        exact_float_set(
            search[
                "alpha"
            ].unique(),
            decimals=2,
        )
    )

    actual_aggs = sorted(
        str(x)
        for x in (
            search[
                "aggregation"
            ].unique()
        )
    )

    actual_thresholds = sorted(
        exact_float_set(
            search[
                "threshold"
            ].unique(),
            decimals=2,
        )
    )

    check(
        "search_chunk_grid_exact",
        actual_chunks
        == EXPECTED_CHUNKS,
        actual_chunks,
    )

    check(
        "search_alpha_grid_exact",
        actual_alphas
        == EXPECTED_ALPHAS,
        actual_alphas,
    )

    check(
        "search_aggregation_grid_exact",
        set(
            actual_aggs
        )
        == set(
            EXPECTED_AGGREGATIONS
        ),
        actual_aggs,
    )

    check(
        "search_threshold_grid_exact",
        actual_thresholds
        == EXPECTED_THRESHOLDS,
        actual_thresholds,
    )

    selected_rows = (
        search[
            selected_mask(
                search[
                    "selected_boundary_extended_config"
                ]
            )
        ]
    )

    check(
        "exactly_one_selected_config",
        len(
            selected_rows
        )
        == 1,
        (
            f"selected rows="
            f"{len(selected_rows)}"
        ),
    )

    selected = (
        selected_rows
        .iloc[0]
    )

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

    best = ranked.iloc[0]

    selected_order = int(
        selected[
            "deterministic_order"
        ]
    )

    best_order = int(
        best[
            "deterministic_order"
        ]
    )

    check(
        "selected_row_is_rank1_by_declared_rule",
        selected_order
        == best_order,
        (
            f"selected_order="
            f"{selected_order}, "
            f"best_order="
            f"{best_order}"
        ),
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

    selected_macro_f1 = float(
        selected[
            "macro_video_f1"
        ]
    )

    selected_micro_f1 = float(
        selected[
            "micro_f1"
        ]
    )

    selected_fold_std = float(
        selected[
            "fold_f1_std"
        ]
    )

    check(
        "json_config_matches_selected_row",
        (
            int(
                boundary_config[
                    "chunk_seconds"
                ]
            )
            == selected_chunk
            and np.isclose(
                float(
                    boundary_config[
                        "alpha"
                    ]
                ),
                selected_alpha,
            )
            and str(
                boundary_config[
                    "aggregation"
                ]
            )
            == selected_aggregation
            and np.isclose(
                float(
                    boundary_config[
                        "threshold"
                    ]
                ),
                selected_threshold,
            )
        ),
        (
            "boundary config JSON "
            "must match selected "
            "search row"
        ),
    )

    print(
        "Selected config:"
    )

    print(
        f"  chunk       = "
        f"{selected_chunk}s"
    )

    print(
        f"  alpha       = "
        f"{selected_alpha}"
    )

    print(
        f"  LSA weight  = "
        f"{1.0-selected_alpha}"
    )

    print(
        f"  aggregation = "
        f"{selected_aggregation}"
    )

    print(
        f"  threshold   = "
        f"{selected_threshold}"
    )

    print(
        f"  DEV macro F1= "
        f"{selected_macro_f1:.6f}"
    )

    print(
        f"  DEV micro F1= "
        f"{selected_micro_f1:.6f}"
    )

    # ========================================================
    # 4. RAW OOF / FINAL-TEST ISOLATION AUDIT
    # ========================================================

    print(
        "\n4. RAW OOF / FINAL TEST "
        "ISOLATION AUDIT"
    )

    raw = pd.read_csv(
        BOUNDARY_RAW_PATH
    )

    raw[
        "video_id"
    ] = normalize_video_id(
        raw[
            "video_id"
        ]
    )

    raw[
        "chunk_seconds"
    ] = (
        pd.to_numeric(
            raw[
                "chunk_seconds"
            ],
            errors="raise",
        )
        .astype(int)
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

    raw_ids = set(
        raw[
            "video_id"
        ].unique()
    )

    check(
        "raw_oof_contains_dev_only",
        raw_ids
        == dev_ids,
        (
            f"raw unique IDs="
            f"{len(raw_ids)}; must "
            "equal exact DEV ID set"
        ),
    )

    check(
        "raw_oof_has_all_six_chunk_sizes",
        sorted(
            raw[
                "chunk_seconds"
            ].unique()
        )
        == EXPECTED_CHUNKS,
        sorted(
            raw[
                "chunk_seconds"
            ].unique()
        ),
    )

    video_fold_counts = (
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

    check(
        "each_video_has_one_heldout_outer_fold",
        (
            len(
                video_fold_counts
            )
            == EXPECTED_DEV_VIDEOS
            and video_fold_counts
            .eq(1)
            .all()
        ),
        (
            "every DEV video must "
            "belong to exactly one "
            "outer held-out fold"
        ),
    )

    check(
        "raw_oof_folds_are_1_to_5",
        set(
            raw[
                "outer_fold"
            ].unique()
        )
        == EXPECTED_FOLDS,
        sorted(
            raw[
                "outer_fold"
            ].unique()
        ),
    )

    non_dev_ids = (
        raw_ids
        - dev_ids
    )

    check(
        "no_non_dev_or_test_ids_in_selection_scores",
        len(
            non_dev_ids
        )
        == 0,
        sorted(
            non_dev_ids
        ),
    )

    print(
        "Raw OOF videos : "
        f"{len(raw_ids)} DEV only"
    )

    print(
        "Final TEST IDs : "
        "0 used in selection scores"
    )

    # ========================================================
    # 5. FINAL CONFIGURATION LOCK
    # ========================================================

    print(
        "\n5. FINAL CONFIGURATION "
        "LOCK"
    )

    core = config_core(
        selected_chunk,
        selected_alpha,
        selected_aggregation,
        selected_threshold,
    )

    config_hash = fingerprint(
        core
    )

    remaining_boundary = bool(
        selected_chunk
        == max(
            EXPECTED_CHUNKS
        )
    )

    final_config = {
        "stage":
            "P1.15 Final Configuration Lock",

        "status":
            "LOCKED",

        **core,

        "selection_stage":
            (
                "P1.13b One-Time "
                "Boundary Extension"
            ),

        "selected_on":
            "DEV",

        "dev_video_count":
            EXPECTED_DEV_VIDEOS,

        "dev_macro_video_f1":
            selected_macro_f1,

        "dev_micro_f1":
            selected_micro_f1,

        "dev_fold_f1_std":
            selected_fold_std,

        "selection_metric":
            "macro_video_f1",

        "selection_rule": [
            "highest macro video F1",
            "highest micro F1",
            "lowest fold F1 std",
            "deterministic grid order",
        ],

        "evaluated_search_space": {
            "chunk_seconds":
                EXPECTED_CHUNKS,

            "alpha":
                EXPECTED_ALPHAS,

            "aggregation":
                EXPECTED_AGGREGATIONS,

            "threshold":
                EXPECTED_THRESHOLDS,

            "num_configurations":
                EXPECTED_SEARCH_ROWS,
        },

        "final_test_used":
            False,

        "further_dev_tuning_allowed":
            False,

        "p1_14_status":
            (
                "SKIPPED_NOT_REQUIRED_"
                "FOR_SELECTION_OR_LOCK"
            ),

        "remaining_limitation":
            (
                "The selected chunk "
                "duration is 180s, the "
                "upper boundary of the "
                "one-time extended DEV "
                "search. Per the "
                "pre-declared stop rule, "
                "the search space is not "
                "extended further. The "
                "configuration is best "
                "among evaluated DEV "
                "configurations, not a "
                "claim of global optimum."
            ),

        "configuration_sha256":
            config_hash,
    }

    if FINAL_CONFIG_PATH.exists():

        existing = json.loads(
            FINAL_CONFIG_PATH
            .read_text(
                encoding="utf-8"
            )
        )

        existing_core = {
            key:
                existing.get(key)
            for key in [
                "chunk_seconds",
                "alpha",
                "lsa_weight",
                "aggregation",
                "threshold",
            ]
        }

        if (
            existing_core
            != core
        ):

            raise RuntimeError(
                "FINAL_MODEL_CONFIG.json "
                "already exists with a "
                "different configuration. "
                "LOCKED config must not be "
                "silently overwritten."
            )

        print(
            "Existing lock matches "
            "selected config."
        )

    FINAL_CONFIG_PATH.write_text(
        json.dumps(
            final_config,
            indent=2,
            ensure_ascii=False,
            default=json_default,
        ),
        encoding="utf-8",
    )

    check(
        "final_config_status_locked",
        final_config[
            "status"
        ]
        == "LOCKED",
        final_config[
            "status"
        ],
    )

    check(
        "further_dev_tuning_disabled",
        final_config[
            "further_dev_tuning_allowed"
        ]
        is False,
        (
            "no more DEV-driven "
            "hyperparameter changes"
        ),
    )

    check(
        "remaining_chunk_boundary_documented",
        (
            (
                not remaining_boundary
            )
            or (
                "upper boundary"
                in final_config[
                    "remaining_limitation"
                ]
            )
        ),
        final_config[
            "remaining_limitation"
        ],
    )

    print(
        "LOCKED:"
    )

    print(
        f"  {selected_chunk}s | "
        f"alpha={selected_alpha} | "
        f"{selected_aggregation} | "
        f"threshold="
        f"{selected_threshold}"
    )

    print(
        "SHA256:"
    )

    print(
        f"  {config_hash}"
    )

    # ========================================================
    # 6. SAVE AUDIT + SUMMARY
    # ========================================================

    audit_df = pd.DataFrame(
        checks
    )

    failed = int(
        audit_df[
            "status"
        ].eq("FAIL")
        .sum()
    )

    passed = int(
        audit_df[
            "status"
        ].eq("PASS")
        .sum()
    )

    final_audit = {
        "stage":
            "P1.15 Final Audit + Lock",

        "status":
            (
                "PASS"
                if failed == 0
                else "FAIL"
            ),

        "checks_passed":
            passed,

        "checks_failed":
            failed,

        "checks":
            checks,

        "final_configuration":
            final_config,

        "p1_14_required":
            False,

        "p1_14_reason":
            (
                "P1.14 was a proposed "
                "supplementary robustness "
                "comparison. It is not "
                "required to select or "
                "lock the final DEV "
                "configuration because "
                "P1.13b already completed "
                "the declared search and "
                "selection."
            ),

        "final_test_used":
            False,

        "config_changed_after_lock":
            False,
    }

    FINAL_AUDIT_PATH.write_text(
        json.dumps(
            final_audit,
            indent=2,
            ensure_ascii=False,
            default=json_default,
        ),
        encoding="utf-8",
    )

    summary_lines = [
        "=" * 82,
        "P1.15 FINAL AUDIT + CONFIGURATION LOCK",
        "=" * 82,
        "",
        "FINAL LOCKED CONFIGURATION",
        (
            f"chunk_seconds = "
            f"{selected_chunk}"
        ),
        (
            f"alpha = "
            f"{selected_alpha}"
        ),
        (
            f"lsa_weight = "
            f"{1.0-selected_alpha}"
        ),
        (
            f"aggregation = "
            f"{selected_aggregation}"
        ),
        (
            f"threshold = "
            f"{selected_threshold}"
        ),
        "",
        (
            f"DEV macro video F1 = "
            f"{selected_macro_f1:.6f}"
        ),
        (
            f"DEV micro F1 = "
            f"{selected_micro_f1:.6f}"
        ),
        (
            f"DEV fold F1 std = "
            f"{selected_fold_std:.6f}"
        ),
        "",
        (
            f"Search configurations "
            f"evaluated = "
            f"{EXPECTED_SEARCH_ROWS}"
        ),
        (
            f"Checks passed = "
            f"{passed}"
        ),
        (
            f"Checks failed = "
            f"{failed}"
        ),
        "",
        "Selection used DEV only.",
        "Final TEST was not used.",
        (
            "No further DEV-driven "
            "hyperparameter tuning is "
            "allowed after this lock."
        ),
        (
            "180s remains the upper "
            "boundary of the one-time "
            "extended search and is "
            "documented as a limitation."
        ),
        (
            "P1.14 supplementary "
            "comparison was skipped; "
            "it was not required for "
            "selection or locking."
        ),
        "",
        (
            f"Configuration SHA256 = "
            f"{config_hash}"
        ),
        "",
        (
            "P1.15 FINAL AUDIT + "
            "CONFIGURATION LOCK: PASS"
        ),
        "",
        "P1 STATUS: COMPLETE / LOCKED",
    ]

    FINAL_SUMMARY_PATH.write_text(
        "\n".join(
            summary_lines
        ),
        encoding="utf-8",
    )

    # ========================================================
    # FINAL CONSOLE
    # ========================================================

    print(
        "\n" + "=" * 82
    )

    print(
        "P1.15 FINAL LOCK SUMMARY"
    )

    print(
        "=" * 82
    )

    print(
        f"Checks passed : {passed}"
    )

    print(
        f"Checks failed : {failed}"
    )

    print()

    print(
        "Final config  : "
        f"{selected_chunk}s | "
        f"alpha="
        f"{selected_alpha} | "
        f"{selected_aggregation} | "
        f"threshold="
        f"{selected_threshold}"
    )

    print(
        f"DEV macro F1  : "
        f"{selected_macro_f1:.6f}"
    )

    print(
        f"DEV micro F1  : "
        f"{selected_micro_f1:.6f}"
    )

    print(
        f"Fold F1 std   : "
        f"{selected_fold_std:.6f}"
    )

    print()

    print(
        "Selected on DEV only."
    )

    print(
        "Final TEST was not used."
    )

    print(
        "No further DEV tuning "
        "is allowed."
    )

    if remaining_boundary:

        print(
            "Limitation: 180s is "
            "still the upper chunk "
            "boundary."
        )

    print(
        "P1.14 supplementary "
        "comparison: SKIPPED."
    )

    print(
        "\nP1.15 FINAL AUDIT + "
        "CONFIGURATION LOCK: PASS"
    )

    print(
        "P1 STATUS: COMPLETE / LOCKED"
    )


if __name__ == "__main__":
    main()