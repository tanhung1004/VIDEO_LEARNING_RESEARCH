from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

VIDEOS_CSV = ROOT / "data" / "raw" / "videos.csv"
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

errors = []

videos = pd.read_csv(
    VIDEOS_CSV,
    dtype=str,
    keep_default_na=False,
)

videos["video_id"] = videos["video_id"].str.strip().str.lower()

total_rows = 0
total_expected = sum(len(cfg["ids"]) for cfg in CONFIG.values())

print("=" * 72)
print("P0.6 ANNOTATION MATRIX AUDIT")
print("=" * 72)

for subject, cfg in CONFIG.items():
    path = REVIEW_DIR / cfg["file"]

    if not path.exists():
        errors.append(f"{subject}: missing file {path}")
        print(f"\n[{subject}] FAIL - file not found")
        continue

    df = pd.read_csv(
        path,
        dtype=str,
        keep_default_na=False,
    )

    df["video_id"] = df["video_id"].str.strip().str.lower()

    expected_ids = set(cfg["ids"])
    actual_ids = set(df["video_id"])
    total_rows += len(df)

    print(f"\n[{subject}]")
    print(f"Rows       : {len(df)}/{len(expected_ids)}")

    # ----------------------------------------------------------
    # 1. Row count / IDs
    # ----------------------------------------------------------
    missing_ids = sorted(expected_ids - actual_ids)
    extra_ids = sorted(actual_ids - expected_ids)
    duplicate_ids = sorted(
        df.loc[df["video_id"].duplicated(keep=False), "video_id"]
        .unique()
        .tolist()
    )

    ids_ok = (
        len(df) == len(expected_ids)
        and not missing_ids
        and not extra_ids
        and not duplicate_ids
    )

    print(f"IDs        : {'PASS' if ids_ok else 'FAIL'}")

    if missing_ids:
        errors.append(f"{subject}: missing IDs {missing_ids}")

    if extra_ids:
        errors.append(f"{subject}: extra IDs {extra_ids}")

    if duplicate_ids:
        errors.append(f"{subject}: duplicate IDs {duplicate_ids}")

    # ----------------------------------------------------------
    # 2. Required columns
    # ----------------------------------------------------------
    required_columns = (
        ["video_id", "title", "url"]
        + cfg["concepts"]
        + ["annotator", "review_status"]
    )

    missing_columns = [
        col for col in required_columns if col not in df.columns
    ]

    if missing_columns:
        errors.append(
            f"{subject}: missing columns {missing_columns}"
        )
        print("Columns    : FAIL")
        continue

    print("Columns    : PASS")

    # ----------------------------------------------------------
    # 3. Labels must be exactly 0 or 1
    # ----------------------------------------------------------
    bad_labels = []

    for concept in cfg["concepts"]:
        bad = df.loc[
            ~df[concept].isin(["0", "1"]),
            ["video_id", concept],
        ]

        for _, row in bad.iterrows():
            bad_labels.append(
                f"{row['video_id']}:{concept}={repr(row[concept])}"
            )

    labels_ok = len(bad_labels) == 0
    print(f"Labels     : {'PASS' if labels_ok else 'FAIL'}")

    if bad_labels:
        errors.append(
            f"{subject}: invalid/blank labels -> {bad_labels}"
        )

    # ----------------------------------------------------------
    # 4. Annotator must exist
    # ----------------------------------------------------------
    blank_annotator = df[
        df["annotator"].str.strip().eq("")
    ]["video_id"].tolist()

    annotator_ok = len(blank_annotator) == 0
    print(
        f"Annotator  : {'PASS' if annotator_ok else 'FAIL'}"
    )

    if blank_annotator:
        errors.append(
            f"{subject}: blank annotator -> {blank_annotator}"
        )

    # ----------------------------------------------------------
    # 5. All rows completed
    # ----------------------------------------------------------
    incomplete = df[
        df["review_status"].str.strip().str.lower().ne("completed")
    ]["video_id"].tolist()

    completed_ok = len(incomplete) == 0
    print(
        f"Completed  : {'PASS' if completed_ok else 'FAIL'}"
    )

    if incomplete:
        errors.append(
            f"{subject}: incomplete review -> {incomplete}"
        )

    # ----------------------------------------------------------
    # 6. Cross-check against videos.csv
    # ----------------------------------------------------------
    expected_videos = videos[
        videos["video_id"].isin(expected_ids)
    ].copy()

    video_table_ids = set(expected_videos["video_id"])

    videos_csv_ok = video_table_ids == expected_ids

    if not videos_csv_ok:
        errors.append(
            f"{subject}: videos.csv IDs mismatch"
        )

    # subject check
    wrong_subject = expected_videos[
        expected_videos["subject"].str.strip() != subject
    ]["video_id"].tolist()

    if wrong_subject:
        videos_csv_ok = False
        errors.append(
            f"{subject}: subject mismatch in videos.csv -> "
            f"{wrong_subject}"
        )

    # URL synchronization check
    url_check = df[["video_id", "url"]].merge(
        expected_videos[["video_id", "url"]],
        on="video_id",
        how="left",
        suffixes=("_matrix", "_videos"),
    )

    bad_urls = url_check[
        url_check["url_matrix"].str.strip()
        != url_check["url_videos"].str.strip()
    ]["video_id"].tolist()

    if bad_urls:
        videos_csv_ok = False
        errors.append(
            f"{subject}: URL mismatch -> {bad_urls}"
        )

    print(
        f"videos.csv : {'PASS' if videos_csv_ok else 'FAIL'}"
    )

print("\n" + "=" * 72)
print(f"TOTAL MATRIX ROWS: {total_rows}/{total_expected}")

if total_rows != total_expected:
    errors.append(
        f"Total matrix rows expected {total_expected}, got {total_rows}"
    )

if errors:
    print("\nP0.6 ANNOTATION MATRIX AUDIT: FAIL")
    print("\nProblems found:")

    for i, error in enumerate(errors, 1):
        print(f"{i}. {error}")

    sys.exit(1)

print("\nP0.6 ANNOTATION MATRIX AUDIT: PASS")
print("All 35 development annotations are complete and structurally valid.")
print("Safe to proceed to Ground Truth merge.")