from pathlib import Path
import sys

import pandas as pd
from sklearn.model_selection import StratifiedKFold


ROOT = Path(__file__).resolve().parents[2]

VIDEOS_PATH = ROOT / "data" / "raw" / "videos.csv"
SPLITS_DIR = ROOT / "data" / "splits"
FOLDS_PATH = SPLITS_DIR / "dev_cv_folds.csv"

N_SPLITS = 5
RANDOM_STATE = 42

EXPECTED_IDS = {f"v{i}" for i in range(1, 41)}
EXPECTED_SUBJECTS = {"SQL", "Python", "Java", "C++"}

errors = []

print("=" * 72)
print("P0.7 DEV 5-FOLD CV ASSIGNMENT")
print("=" * 72)

# ---------------------------------------------------------------------
# Load videos.csv
# ---------------------------------------------------------------------

videos = pd.read_csv(
    VIDEOS_PATH,
    dtype=str,
    keep_default_na=False,
)

required_columns = [
    "video_id",
    "subject",
    "split",
    "annotation_status",
    "cv_fold",
]

missing_columns = [
    col for col in required_columns
    if col not in videos.columns
]

if missing_columns:
    print(f"Missing columns: {missing_columns}")
    sys.exit(1)

videos["video_id"] = (
    videos["video_id"]
    .str.strip()
    .str.lower()
)

videos["subject"] = videos["subject"].str.strip()
videos["split"] = videos["split"].str.strip().str.lower()
videos["annotation_status"] = (
    videos["annotation_status"]
    .str.strip()
    .str.lower()
)

# ---------------------------------------------------------------------
# Select exact DEV v1-v40
# ---------------------------------------------------------------------

dev = videos[
    videos["video_id"].isin(EXPECTED_IDS)
].copy()

actual_ids = set(dev["video_id"])

missing_ids = sorted(EXPECTED_IDS - actual_ids)
extra_ids = sorted(actual_ids - EXPECTED_IDS)

if missing_ids:
    errors.append(
        f"Missing DEV IDs: {missing_ids}"
    )

if extra_ids:
    errors.append(
        f"Unexpected DEV IDs: {extra_ids}"
    )

if len(dev) != 40:
    errors.append(
        f"Expected 40 DEV rows, found {len(dev)}"
    )

duplicate_ids = sorted(
    dev.loc[
        dev["video_id"].duplicated(keep=False),
        "video_id",
    ]
    .unique()
    .tolist()
)

if duplicate_ids:
    errors.append(
        f"Duplicate video IDs: {duplicate_ids}"
    )

# ---------------------------------------------------------------------
# Validate split and annotation status
# ---------------------------------------------------------------------

wrong_split = dev[
    dev["split"] != "dev"
]["video_id"].tolist()

if wrong_split:
    errors.append(
        f"These v1-v40 videos are not split=dev: {wrong_split}"
    )

incomplete = dev[
    dev["annotation_status"] != "completed"
]["video_id"].tolist()

if incomplete:
    errors.append(
        f"These DEV videos are not annotation completed: {incomplete}"
    )

# ---------------------------------------------------------------------
# Validate subjects
# ---------------------------------------------------------------------

actual_subjects = set(dev["subject"])

if actual_subjects != EXPECTED_SUBJECTS:
    errors.append(
        f"Unexpected subjects: {sorted(actual_subjects)}"
    )

subject_counts = (
    dev["subject"]
    .value_counts()
    .sort_index()
)

print("\nDEV subject counts:")
print(subject_counts.to_string())

for subject in sorted(EXPECTED_SUBJECTS):
    count = int(subject_counts.get(subject, 0))

    if count != 10:
        errors.append(
            f"{subject}: expected 10 videos, found {count}"
        )

# ---------------------------------------------------------------------
# Stable order before StratifiedKFold
# ---------------------------------------------------------------------

dev["_video_num"] = (
    dev["video_id"]
    .str.extract(r"^v(\d+)$", expand=False)
)

if dev["_video_num"].isna().any():
    bad_ids = dev.loc[
        dev["_video_num"].isna(),
        "video_id",
    ].tolist()

    errors.append(
        f"Invalid video_id format: {bad_ids}"
    )

if errors:
    print("\nP0.7 PRE-SPLIT VALIDATION: FAIL")

    for i, error in enumerate(errors, 1):
        print(f"{i}. {error}")

    sys.exit(1)

dev["_video_num"] = dev["_video_num"].astype(int)

dev = (
    dev
    .sort_values("_video_num")
    .reset_index(drop=True)
)

# ---------------------------------------------------------------------
# Stratified 5-fold assignment
# ---------------------------------------------------------------------

skf = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE,
)

dev["cv_fold_new"] = ""

X = dev[["video_id"]]
y = dev["subject"]

for fold, (_, test_idx) in enumerate(
    skf.split(X, y),
    start=1,
):
    dev.loc[test_idx, "cv_fold_new"] = str(fold)

# ---------------------------------------------------------------------
# Validate every video got exactly one fold
# ---------------------------------------------------------------------

invalid_fold = dev[
    ~dev["cv_fold_new"].isin(
        [str(i) for i in range(1, N_SPLITS + 1)]
    )
]["video_id"].tolist()

if invalid_fold:
    errors.append(
        f"Invalid/unassigned folds: {invalid_fold}"
    )

# ---------------------------------------------------------------------
# Audit fold size and subject balance
# ---------------------------------------------------------------------

print("\nFold distribution:")
fold_table = pd.crosstab(
    dev["cv_fold_new"],
    dev["subject"],
)

fold_table = fold_table.reindex(
    index=[str(i) for i in range(1, N_SPLITS + 1)],
    columns=["SQL", "Python", "Java", "C++"],
    fill_value=0,
)

fold_table["TOTAL"] = fold_table.sum(axis=1)

print(fold_table.to_string())

for fold in range(1, N_SPLITS + 1):
    fold_str = str(fold)

    fold_rows = dev[
        dev["cv_fold_new"] == fold_str
    ]

    if len(fold_rows) != 8:
        errors.append(
            f"Fold {fold}: expected 8 videos, "
            f"found {len(fold_rows)}"
        )

    for subject in EXPECTED_SUBJECTS:
        count = int(
            (
                fold_rows["subject"] == subject
            ).sum()
        )

        if count != 2:
            errors.append(
                f"Fold {fold}: {subject} expected 2, "
                f"found {count}"
            )

# ---------------------------------------------------------------------
# Ensure all 40 unique videos occur exactly once
# ---------------------------------------------------------------------

if dev["video_id"].nunique() != 40:
    errors.append(
        "DEV fold table does not contain 40 unique videos"
    )

# ---------------------------------------------------------------------
# Abort before writing if anything failed
# ---------------------------------------------------------------------

if errors:
    print("\n" + "=" * 72)
    print("P0.7 CV FOLD ASSIGNMENT: FAIL")
    print("NO FILES WERE WRITTEN.")

    for i, error in enumerate(errors, 1):
        print(f"{i}. {error}")

    sys.exit(1)

# ---------------------------------------------------------------------
# Build persistent fold manifest
# ---------------------------------------------------------------------

fold_manifest = dev[
    [
        "video_id",
        "subject",
        "title",
        "url",
        "cv_fold_new",
    ]
].copy()

fold_manifest = fold_manifest.rename(
    columns={
        "cv_fold_new": "cv_fold",
    }
)

fold_manifest = fold_manifest.sort_values(
    by="video_id",
    key=lambda s: (
        s.str.extract(r"(\d+)", expand=False).astype(int)
    ),
)

# ---------------------------------------------------------------------
# Update only cv_fold in videos.csv
# ---------------------------------------------------------------------

videos_new = videos.copy()

fold_map = dict(
    zip(
        dev["video_id"],
        dev["cv_fold_new"],
    )
)

mask = videos_new["video_id"].isin(EXPECTED_IDS)

videos_new.loc[
    mask,
    "cv_fold",
] = videos_new.loc[
    mask,
    "video_id",
].map(fold_map)

# ---------------------------------------------------------------------
# Write through temporary files
# ---------------------------------------------------------------------

SPLITS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

folds_tmp = FOLDS_PATH.with_suffix(".tmp.csv")
videos_tmp = VIDEOS_PATH.with_suffix(".tmp.csv")

fold_manifest.to_csv(
    folds_tmp,
    index=False,
    encoding="utf-8-sig",
)

videos_new.to_csv(
    videos_tmp,
    index=False,
    encoding="utf-8-sig",
)

folds_tmp.replace(FOLDS_PATH)
videos_tmp.replace(VIDEOS_PATH)

# ---------------------------------------------------------------------
# Post-write audit
# ---------------------------------------------------------------------

check_folds = pd.read_csv(
    FOLDS_PATH,
    dtype=str,
    keep_default_na=False,
)

check_videos = pd.read_csv(
    VIDEOS_PATH,
    dtype=str,
    keep_default_na=False,
)

check_folds["video_id"] = (
    check_folds["video_id"].str.strip().str.lower()
)

check_videos["video_id"] = (
    check_videos["video_id"].str.strip().str.lower()
)

if len(check_folds) != 40:
    errors.append(
        f"Post-write fold manifest rows = {len(check_folds)}"
    )

if check_folds["video_id"].nunique() != 40:
    errors.append(
        "Post-write manifest does not have 40 unique IDs"
    )

video_fold_check = (
    check_videos[
        check_videos["video_id"].isin(EXPECTED_IDS)
    ][["video_id", "cv_fold"]]
    .merge(
        check_folds[["video_id", "cv_fold"]],
        on="video_id",
        suffixes=("_videos", "_manifest"),
    )
)

mismatch = video_fold_check[
    video_fold_check["cv_fold_videos"]
    != video_fold_check["cv_fold_manifest"]
]

if not mismatch.empty:
    errors.append(
        "videos.csv cv_fold does not match dev_cv_folds.csv"
    )

print("\n" + "=" * 72)

if errors:
    print("P0.7 POST-WRITE AUDIT: FAIL")

    for i, error in enumerate(errors, 1):
        print(f"{i}. {error}")

    sys.exit(1)

print("P0.7 DEV 5-FOLD CV ASSIGNMENT: PASS")
print("40/40 DEV videos assigned exactly once.")
print("Each fold contains 8 videos.")
print("Each fold contains exactly 2 videos per subject.")
print(f"StratifiedKFold random_state = {RANDOM_STATE}")
print(f"Saved: {FOLDS_PATH.relative_to(ROOT)}")
print("videos.csv cv_fold synchronized with manifest.")