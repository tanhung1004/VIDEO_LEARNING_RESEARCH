from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

GT_PATH = ROOT / "data" / "ground_truth" / "ground_truth_concepts.csv"
VIDEOS_PATH = ROOT / "data" / "raw" / "videos.csv"
CATALOG_PATH = ROOT / "data" / "concepts" / "concept_catalog.csv"

REVIEW_DIR = (
    ROOT
    / "data"
    / "ground_truth"
    / "annotation_working"
    / "review_assisted"
)

CONFIG = {
    "SQL": {
        "file": "sql_annotation_matrix.csv",
        "ids": [f"v{i}" for i in range(6, 14)],
        "concepts": [
            "select",
            "where",
            "group by",
            "having",
            "count",
            "aggregate function",
            "join",
            "inner join",
            "left join",
            "subquery",
            "order by",
        ],
    },
    "Python": {
        "file": "python_annotation_matrix.csv",
        "ids": [f"v{i}" for i in range(14, 23)],
        "concepts": [
            "for loop",
            "for",
            "iteration",
            "range",
            "list",
            "nested loop",
            "index",
            "break",
            "continue",
        ],
    },
    "Java": {
        "file": "java_annotation_matrix.csv",
        "ids": [f"v{i}" for i in range(23, 31)],
        "concepts": [
            "class",
            "object",
            "method",
            "interface",
            "implements",
            "extends",
            "inheritance",
            "abstract class",
            "abstract method",
        ],
    },
    "C++": {
        "file": "cpp_annotation_matrix.csv",
        "ids": [f"v{i}" for i in range(31, 41)],
        "concepts": [
            "variable",
            "data type",
            "pointer",
            "reference",
            "class",
            "object",
            "constructor",
            "inheritance",
            "polymorphism",
            "template",
        ],
    },
}

DEV_IDS = {
    video_id
    for cfg in CONFIG.values()
    for video_id in cfg["ids"]
}

errors = []

print("=" * 72)
print("P0.6 GROUND TRUTH MERGE")
print("=" * 72)

# ---------------------------------------------------------------------
# Load files
# ---------------------------------------------------------------------

gt_old = pd.read_csv(
    GT_PATH,
    dtype=str,
    keep_default_na=False,
)

videos = pd.read_csv(
    VIDEOS_PATH,
    dtype=str,
    keep_default_na=False,
)

catalog = pd.read_csv(
    CATALOG_PATH,
    dtype=str,
    keep_default_na=False,
)

required_gt_columns = ["video_id", "subject", "concept"]

if gt_old.columns.tolist() != required_gt_columns:
    errors.append(
        f"Unexpected GT schema: {gt_old.columns.tolist()}"
    )

for col in ["video_id", "subject", "concept"]:
    if col not in catalog.columns:
        errors.append(
            f"concept_catalog.csv missing required column: {col}"
            if col == "video_id"
            else f"concept_catalog.csv missing required column: {col}"
        )

# Catalog normally only needs subject + concept.
# Remove the accidental video_id requirement from the check above.
errors = [
    e for e in errors
    if e != "concept_catalog.csv missing required column: video_id"
]

for col in ["subject", "concept"]:
    if col not in catalog.columns:
        errors.append(
            f"concept_catalog.csv missing required column: {col}"
        )

for col in ["video_id", "subject", "annotation_status"]:
    if col not in videos.columns:
        errors.append(
            f"videos.csv missing required column: {col}"
        )

if errors:
    print("\nPRE-MERGE VALIDATION: FAIL")
    for i, error in enumerate(errors, 1):
        print(f"{i}. {error}")
    sys.exit(1)

gt_old["video_id"] = gt_old["video_id"].str.strip().str.lower()
gt_old["subject"] = gt_old["subject"].str.strip()
gt_old["concept"] = gt_old["concept"].str.strip()

videos["video_id"] = videos["video_id"].str.strip().str.lower()
videos["subject"] = videos["subject"].str.strip()

catalog["subject"] = catalog["subject"].str.strip()
catalog["concept"] = catalog["concept"].str.strip()

# ---------------------------------------------------------------------
# Preserve all existing GT rows outside v6-v40.
# This preserves pilot v1-v5 exactly and also makes reruns idempotent.
# ---------------------------------------------------------------------

preserved_gt = gt_old[
    ~gt_old["video_id"].isin(DEV_IDS)
].copy()

pilot_before = gt_old[
    gt_old["video_id"].isin({f"v{i}" for i in range(1, 6)})
].copy()

print(f"\nExisting GT rows       : {len(gt_old)}")
print(f"Preserved GT rows      : {len(preserved_gt)}")
print(f"Pilot v1-v5 rows       : {len(pilot_before)}")

# ---------------------------------------------------------------------
# Convert positive matrix labels to positive-only GT rows.
# ---------------------------------------------------------------------

positive_rows = []

for subject, cfg in CONFIG.items():
    path = REVIEW_DIR / cfg["file"]

    df = pd.read_csv(
        path,
        dtype=str,
        keep_default_na=False,
    )

    df["video_id"] = df["video_id"].str.strip().str.lower()

    subject_positive_count = 0

    for _, row in df.iterrows():
        video_id = row["video_id"]

        if video_id not in cfg["ids"]:
            errors.append(
                f"{subject}: unexpected video_id {video_id}"
            )
            continue

        if row["review_status"].strip().lower() != "completed":
            errors.append(
                f"{subject}: {video_id} is not completed"
            )

        if not row["annotator"].strip():
            errors.append(
                f"{subject}: {video_id} has blank annotator"
            )

        for concept in cfg["concepts"]:
            label = row[concept].strip()

            if label not in {"0", "1"}:
                errors.append(
                    f"{subject}: {video_id}/{concept} invalid label "
                    f"{repr(label)}"
                )
                continue

            if label == "1":
                positive_rows.append(
                    {
                        "video_id": video_id,
                        "subject": subject,
                        "concept": concept,
                    }
                )
                subject_positive_count += 1

    print(
        f"{subject:<8} new positive pairs: "
        f"{subject_positive_count}"
    )

new_gt = pd.DataFrame(
    positive_rows,
    columns=required_gt_columns,
)

print(f"\nNew v6-v40 positives   : {len(new_gt)}")

# ---------------------------------------------------------------------
# Validate video -> subject mapping.
# ---------------------------------------------------------------------

video_subject_map = (
    videos.set_index("video_id")["subject"].to_dict()
)

for _, row in new_gt.iterrows():
    video_id = row["video_id"]
    subject = row["subject"]

    if video_id not in video_subject_map:
        errors.append(
            f"{video_id}: missing from videos.csv"
        )
    elif video_subject_map[video_id] != subject:
        errors.append(
            f"{video_id}: subject mismatch "
            f"matrix={subject}, videos.csv={video_subject_map[video_id]}"
        )

# ---------------------------------------------------------------------
# Validate subject/concept against concept catalog.
# ---------------------------------------------------------------------

allowed_pairs = set(
    zip(
        catalog["subject"],
        catalog["concept"],
    )
)

invalid_pairs = []

for _, row in new_gt.iterrows():
    pair = (row["subject"], row["concept"])

    if pair not in allowed_pairs:
        invalid_pairs.append(
            (
                row["video_id"],
                row["subject"],
                row["concept"],
            )
        )

if invalid_pairs:
    errors.append(
        f"Invalid subject/concept pairs: {invalid_pairs}"
    )

# ---------------------------------------------------------------------
# Merge.
# ---------------------------------------------------------------------

merged_gt = pd.concat(
    [preserved_gt, new_gt],
    ignore_index=True,
)

# Numeric video ordering: v1, v2, ..., v40.
merged_gt["_video_num"] = (
    merged_gt["video_id"]
    .str.extract(r"^v(\d+)$", expand=False)
)

if merged_gt["_video_num"].isna().any():
    bad_ids = merged_gt.loc[
        merged_gt["_video_num"].isna(),
        "video_id",
    ].tolist()

    errors.append(
        f"Invalid video_id format in GT: {bad_ids}"
    )
else:
    merged_gt["_video_num"] = (
        merged_gt["_video_num"].astype(int)
    )

# ---------------------------------------------------------------------
# Duplicate validation.
# ---------------------------------------------------------------------

dups = merged_gt[
    merged_gt.duplicated(
        subset=["video_id", "concept"],
        keep=False,
    )
]

if not dups.empty:
    errors.append(
        "Duplicate (video_id, concept) pairs: "
        + str(
            dups[
                ["video_id", "subject", "concept"]
            ].to_dict("records")
        )
    )

# ---------------------------------------------------------------------
# Verify pilot GT v1-v5 is unchanged.
# ---------------------------------------------------------------------

pilot_after = merged_gt[
    merged_gt["video_id"].isin(
        {f"v{i}" for i in range(1, 6)}
    )
][required_gt_columns].copy()

pilot_before_cmp = (
    pilot_before[required_gt_columns]
    .sort_values(required_gt_columns)
    .reset_index(drop=True)
)

pilot_after_cmp = (
    pilot_after
    .sort_values(required_gt_columns)
    .reset_index(drop=True)
)

if not pilot_before_cmp.equals(pilot_after_cmp):
    errors.append(
        "Pilot v1-v5 ground truth changed unexpectedly"
    )

# ---------------------------------------------------------------------
# Validate exact development video coverage.
# ---------------------------------------------------------------------

actual_dev_ids = set(new_gt["video_id"])

missing_dev_ids = sorted(DEV_IDS - actual_dev_ids)

# A video may legitimately have zero positives, so check matrices,
# not positive-only new_gt, for video coverage.
matrix_ids = set()

for subject, cfg in CONFIG.items():
    path = REVIEW_DIR / cfg["file"]

    df = pd.read_csv(
        path,
        dtype=str,
        keep_default_na=False,
    )

    matrix_ids.update(
        df["video_id"].str.strip().str.lower().tolist()
    )

missing_matrix_ids = sorted(DEV_IDS - matrix_ids)
extra_matrix_ids = sorted(matrix_ids - DEV_IDS)

if missing_matrix_ids:
    errors.append(
        f"Missing development matrix IDs: {missing_matrix_ids}"
    )

if extra_matrix_ids:
    errors.append(
        f"Unexpected matrix IDs: {extra_matrix_ids}"
    )

# ---------------------------------------------------------------------
# Prepare videos.csv update, but DO NOT write until all validation passes.
# ---------------------------------------------------------------------

videos_new = videos.copy()

dev_mask = videos_new["video_id"].isin(DEV_IDS)

found_dev_ids = set(
    videos_new.loc[dev_mask, "video_id"]
)

if found_dev_ids != DEV_IDS:
    errors.append(
        "videos.csv does not contain exactly v6-v40 "
        f"(missing={sorted(DEV_IDS - found_dev_ids)})"
    )

videos_new.loc[
    dev_mask,
    "annotation_status",
] = "completed"

# ---------------------------------------------------------------------
# Final validation result.
# ---------------------------------------------------------------------

if errors:
    print("\n" + "=" * 72)
    print("P0.6 GROUND TRUTH MERGE: FAIL")
    print("NO FILES WERE WRITTEN.")
    print("\nProblems found:")

    for i, error in enumerate(errors, 1):
        print(f"{i}. {error}")

    sys.exit(1)

# Sort only after all format checks passed.
merged_gt = (
    merged_gt
    .sort_values(
        ["_video_num", "subject", "concept"],
        kind="stable",
    )
    .drop(columns=["_video_num"])
    .reset_index(drop=True)
)

# ---------------------------------------------------------------------
# Atomic-ish write using temporary files.
# ---------------------------------------------------------------------

gt_tmp = GT_PATH.with_suffix(".tmp.csv")
videos_tmp = VIDEOS_PATH.with_suffix(".tmp.csv")

merged_gt.to_csv(
    gt_tmp,
    index=False,
    encoding="utf-8-sig",
)

videos_new.to_csv(
    videos_tmp,
    index=False,
    encoding="utf-8-sig",
)

gt_tmp.replace(GT_PATH)
videos_tmp.replace(VIDEOS_PATH)

# ---------------------------------------------------------------------
# Post-write audit.
# ---------------------------------------------------------------------

gt_check = pd.read_csv(
    GT_PATH,
    dtype=str,
    keep_default_na=False,
)

videos_check = pd.read_csv(
    VIDEOS_PATH,
    dtype=str,
    keep_default_na=False,
)

gt_check["video_id"] = (
    gt_check["video_id"].str.strip().str.lower()
)

videos_check["video_id"] = (
    videos_check["video_id"].str.strip().str.lower()
)

duplicate_count = int(
    gt_check.duplicated(
        subset=["video_id", "concept"]
    ).sum()
)

incomplete_dev = videos_check[
    videos_check["video_id"].isin(DEV_IDS)
    & videos_check["annotation_status"]
        .str.strip()
        .str.lower()
        .ne("completed")
]["video_id"].tolist()

print("\n" + "=" * 72)
print("POST-MERGE AUDIT")
print("=" * 72)

print(f"Ground truth rows      : {len(gt_check)}")
print(f"Duplicate pairs        : {duplicate_count}")
print(f"Pilot v1-v5 preserved  : PASS")
print(f"v6-v40 status completed: "
      f"{'PASS' if not incomplete_dev else 'FAIL'}")

if duplicate_count != 0 or incomplete_dev:
    print("\nPOST-MERGE AUDIT: FAIL")
    sys.exit(1)

print("\nP0.6 GROUND TRUTH MERGE: PASS")
print("Positive-only GT for v6-v40 merged successfully.")
print("Pilot v1-v5 ground truth preserved.")
print("videos.csv annotation_status updated for v6-v40.")