from pathlib import Path
import pandas as pd


# =========================================================
# PROJECT PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

VIDEOS_FILE = PROJECT_ROOT / "data" / "raw" / "videos.csv"
CONCEPT_FILE = PROJECT_ROOT / "data" / "concepts" / "concept_catalog.csv"
GT_FILE = PROJECT_ROOT / "data" / "ground_truth" / "ground_truth_concepts.csv"

WORK_DIR = PROJECT_ROOT / "module1" / "ocr_best_integration"

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ocr_best_integration"
)

RESULT_DIR = (
    PROJECT_ROOT
    / "module1"
    / "results"
    / "ocr_best_integration"
)

KEYFRAME_DIR = DATA_DIR / "keyframes_full"
ROI_KEYFRAME_DIR = DATA_DIR / "keyframes_roi"

FULL_OCR_RAW_FILE = RESULT_DIR / "ocr_full_raw.csv"
FULL_OCR_CLEAN_FILE = RESULT_DIR / "ocr_full_clean.csv"

ROI_OCR_RAW_FILE = RESULT_DIR / "ocr_roi_raw.csv"
ROI_OCR_CLEAN_FILE = RESULT_DIR / "ocr_roi_clean.csv"

KEYFRAME_SUMMARY_FILE = RESULT_DIR / "keyframe_summary.csv"
ROI_SUMMARY_FILE = RESULT_DIR / "keyframe_roi_summary.csv"
ROI_PREVIEW_DIR = RESULT_DIR / "roi_preview"


# =========================================================
# FROZEN EXPERIMENT CONFIG
# =========================================================

DEV_IDS = [f"v{i}" for i in range(1, 41)]

EXPECTED_SUBJECT_COUNTS = {
    "SQL": 10,
    "Python": 10,
    "Java": 10,
    "C++": 10,
}

# ---------------------------------------------------------
# NEW TRANSCRIPT FOUNDATION — FROZEN
# ---------------------------------------------------------

NEW_CHUNK_SECONDS = 180
NEW_LDA_WEIGHT = 0.0
NEW_LSA_WEIGHT = 1.0
NEW_AGGREGATION = "max_final_score"
NEW_THRESHOLD = 0.50


# ---------------------------------------------------------
# BEST OCR METHOD FROM COLLABORATOR PACKAGE
# ---------------------------------------------------------

BEST_OCR_METHOD = "F"

BEST_OCR_DESCRIPTION = (
    "cleaned full-frame OCR + cleaned ROI OCR; "
    "late score-level fusion"
)

BEST_OCR_SCORE_FUSION = "mean"

# Inherited OCR sampling interval
FRAME_INTERVAL_SECONDS = 10


# =========================================================
# HELPERS
# =========================================================

def sort_video_ids(values):
    """
    Sort IDs like v1, v2, ..., v40 numerically.
    Falls back to string ordering if needed.
    """

    def key_func(value):
        value = str(value).strip()

        if value.lower().startswith("v"):
            try:
                return int(value[1:])
            except ValueError:
                pass

        return 999999

    return sorted(values, key=key_func)


def detect_gt_concept_column(df):
    """
    Detect concept identifier/name column in ground-truth file.

    We intentionally do not hard-code concept_id because
    the project's GT schema may use another concept column name.
    """

    candidates = [
        "concept_id",
        "concept",
        "concept_name",
        "learning_concept",
    ]

    for column in candidates:
        if column in df.columns:
            return column

    raise RuntimeError(
        "Cannot identify concept column in ground truth.\n"
        f"Available GT columns: {df.columns.tolist()}"
    )


# =========================================================
# MAIN AUDIT
# =========================================================

def main():

    print("=" * 72)
    print("OCR BEST INTEGRATION — STEP 00 CONFIG / INPUT AUDIT")
    print("=" * 72)

    # -----------------------------------------------------
    # Check authoritative input files
    # -----------------------------------------------------

    required_files = {
        "videos.csv": VIDEOS_FILE,
        "concept_catalog.csv": CONCEPT_FILE,
        "ground_truth_concepts.csv": GT_FILE,
    }

    print()
    print("--- REQUIRED FILES ---")

    missing_files = []

    for name, path in required_files.items():

        exists = path.exists()

        print(
            f"{'PASS' if exists else 'FAIL'} : "
            f"{name} -> {path}"
        )

        if not exists:
            missing_files.append(str(path))

    if missing_files:
        raise RuntimeError(
            "Missing required files:\n"
            + "\n".join(missing_files)
        )

    # -----------------------------------------------------
    # Load authoritative inputs
    # -----------------------------------------------------

    videos = pd.read_csv(VIDEOS_FILE)
    concepts = pd.read_csv(CONCEPT_FILE)
    gt = pd.read_csv(GT_FILE)

    print()
    print("--- INPUT SCHEMAS ---")
    print("videos.csv columns:")
    print(videos.columns.tolist())

    print()
    print("concept_catalog.csv columns:")
    print(concepts.columns.tolist())

    print()
    print("ground_truth_concepts.csv columns:")
    print(gt.columns.tolist())

    # -----------------------------------------------------
    # Required schema checks
    # -----------------------------------------------------

    video_required_columns = {
        "video_id",
        "subject",
        "url",
        "status",
    }

    missing_video_columns = (
        video_required_columns
        - set(videos.columns)
    )

    if missing_video_columns:
        raise RuntimeError(
            "videos.csv missing columns: "
            + str(sorted(missing_video_columns))
        )

    if "video_id" not in gt.columns:
        raise RuntimeError(
            "ground_truth_concepts.csv does not contain video_id. "
            f"Columns: {gt.columns.tolist()}"
        )

    # Detect GT concept column from actual schema.
    gt_concept_col = detect_gt_concept_column(gt)

    # -----------------------------------------------------
    # Normalize video registry
    # -----------------------------------------------------

    videos = videos.copy()

    videos["video_id"] = (
        videos["video_id"]
        .astype(str)
        .str.strip()
    )

    videos["subject"] = (
        videos["subject"]
        .astype(str)
        .str.strip()
    )

    videos["status"] = (
        videos["status"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    videos["url"] = (
        videos["url"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    # -----------------------------------------------------
    # Select exactly DEV v1-v40
    # -----------------------------------------------------

    selected = videos[
        videos["status"].eq("selected")
    ].copy()

    selected = selected[
        selected["video_id"].isin(DEV_IDS)
    ].copy()

    ids = set(selected["video_id"])
    expected_ids = set(DEV_IDS)

    missing_ids = sort_video_ids(
        expected_ids - ids
    )

    extra_ids = sort_video_ids(
        ids - expected_ids
    )

    duplicate_video_ids = int(
        selected.duplicated(
            subset=["video_id"]
        ).sum()
    )

    subject_counts = (
        selected["subject"]
        .value_counts()
        .to_dict()
    )

    empty_urls = int(
        selected["url"]
        .eq("")
        .sum()
    )

    # -----------------------------------------------------
    # Ground truth DEV v1-v40
    # -----------------------------------------------------

    gt = gt.copy()

    gt["video_id"] = (
        gt["video_id"]
        .astype(str)
        .str.strip()
    )

    gt[gt_concept_col] = (
        gt[gt_concept_col]
        .astype(str)
        .str.strip()
    )

    dev_gt = gt[
        gt["video_id"].isin(DEV_IDS)
    ].copy()

    duplicate_gt = int(
        dev_gt.duplicated(
            subset=[
                "video_id",
                gt_concept_col,
            ]
        ).sum()
    )

    empty_gt_concepts = int(
        dev_gt[gt_concept_col]
        .eq("")
        .sum()
    )

    # -----------------------------------------------------
    # Concept catalog basic audit
    # -----------------------------------------------------

    concept_count = len(concepts)

    # -----------------------------------------------------
    # Print dataset audit
    # -----------------------------------------------------

    print()
    print("--- DEV DATASET AUDIT ---")

    print(
        "DEV videos:",
        len(selected)
    )

    print(
        "Subject counts:",
        subject_counts
    )

    print(
        "Missing IDs:",
        missing_ids
    )

    print(
        "Extra IDs:",
        extra_ids
    )

    print(
        "Duplicate video IDs:",
        duplicate_video_ids
    )

    print(
        "Empty URLs:",
        empty_urls
    )

    print()
    print("--- CONCEPT / GT AUDIT ---")

    print(
        "Concept catalog rows:",
        concept_count
    )

    print(
        "DEV positive GT pairs:",
        len(dev_gt)
    )

    print(
        "GT concept column:",
        gt_concept_col
    )

    print(
        "Duplicate DEV GT pairs:",
        duplicate_gt
    )

    print(
        "Empty GT concepts:",
        empty_gt_concepts
    )

    # -----------------------------------------------------
    # Frozen NEW transcript config
    # -----------------------------------------------------

    print()
    print("--- FROZEN NEW TRANSCRIPT FOUNDATION ---")

    print(
        "Chunk seconds:",
        NEW_CHUNK_SECONDS
    )

    print(
        "LDA weight:",
        NEW_LDA_WEIGHT
    )

    print(
        "LSA weight:",
        NEW_LSA_WEIGHT
    )

    print(
        "Aggregation:",
        NEW_AGGREGATION
    )

    print(
        "Threshold:",
        NEW_THRESHOLD
    )

    # -----------------------------------------------------
    # Best OCR method
    # -----------------------------------------------------

    print()
    print("--- SELECTED NEW OCR METHOD ---")

    print(
        "Method:",
        BEST_OCR_METHOD
    )

    print(
        "Description:",
        BEST_OCR_DESCRIPTION
    )

    print(
        "Score fusion:",
        BEST_OCR_SCORE_FUSION
    )

    print(
        "Frame interval:",
        FRAME_INTERVAL_SECONDS,
        "seconds"
    )

    # -----------------------------------------------------
    # Scientific integrity checks
    # -----------------------------------------------------

    checks = {
        "40 DEV videos":
            len(selected) == 40,

        "10 videos per subject":
            subject_counts
            == EXPECTED_SUBJECT_COUNTS,

        "no missing DEV IDs":
            len(missing_ids) == 0,

        "no extra DEV IDs":
            len(extra_ids) == 0,

        "no duplicate video IDs":
            duplicate_video_ids == 0,

        "no empty URLs":
            empty_urls == 0,

        "39 concepts":
            concept_count == 39,

        "153 DEV GT positive pairs":
            len(dev_gt) == 153,

        "no duplicate GT pairs":
            duplicate_gt == 0,

        "no empty GT concept values":
            empty_gt_concepts == 0,
    }

    print()
    print("--- CHECKS ---")

    failed = []

    for name, passed in checks.items():

        label = (
            "PASS"
            if passed
            else "FAIL"
        )

        print(
            f"{label} : {name}"
        )

        if not passed:
            failed.append(name)

    # -----------------------------------------------------
    # Stop immediately if audit fails
    # -----------------------------------------------------

    if failed:

        print()
        print("OCR STEP 00 FAILED")

        raise RuntimeError(
            "Failed checks: "
            + ", ".join(failed)
        )

    # -----------------------------------------------------
    # Create isolated NEW OCR workspace
    # only after scientific audit passes
    # -----------------------------------------------------

    folders_to_create = [
        DATA_DIR,
        RESULT_DIR,
        KEYFRAME_DIR,
        ROI_KEYFRAME_DIR,
        ROI_PREVIEW_DIR,
    ]

    for folder in folders_to_create:
        folder.mkdir(
            parents=True,
            exist_ok=True,
        )

    # -----------------------------------------------------
    # Final output
    # -----------------------------------------------------

    print()
    print("=" * 72)
    print("OCR STEP 00 PASS")
    print("=" * 72)

    print()
    print(
        "Workspace:",
        WORK_DIR
    )

    print(
        "Data output:",
        DATA_DIR
    )

    print(
        "Result output:",
        RESULT_DIR
    )

    print()
    print(
        "NEW TRANSCRIPT CONFIG REMAINS FROZEN."
    )

    print(
        "BEST OCR METHOD LOCKED FOR THIS INTEGRATION: "
        "F / score-level mean."
    )

    print()
    print(
        "NO OCR HAS BEEN RUN YET."
    )


if __name__ == "__main__":
    main()