from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CANDIDATE_FILE = (
    PROJECT_ROOT / "data" / "raw" / "video_candidates.csv"
)

VIDEOS_FILE = (
    PROJECT_ROOT / "data" / "raw" / "videos.csv"
)


# =========================================================
# FROZEN C++ SELECTION
#
# Selection basis:
# - English transcript available
# - coverage of the 10 C++ concepts
# - channel diversity
# - content-style diversity
# - reasonable duration
#
# NOT used:
# - LDA
# - LSA
# - OCR
# - predictions
# - F1
# =========================================================

PROMOTION_MAP = {
    "c040": "v31",
    "c041": "v32",
    "c042": "v33",
    "c043": "v34",
    "c044": "v35",
    "c045": "v36",
    "c047": "v37",
    "c048": "v38",
    "c049": "v39",
    "c051": "v40",
}


EXPECTED_VIDEO_COLUMNS = [
    "video_id",
    "subject",
    "title",
    "url",
    "status",
    "split",
    "cv_fold",
    "annotation_status",
    "content_style",
    "duration_min",
    "channel",
]


def clean_text(value):
    if pd.isna(value):
        return ""
    return str(value).strip()


def main():
    candidates = pd.read_csv(CANDIDATE_FILE)
    videos = pd.read_csv(VIDEOS_FILE)

    # =====================================================
    # Validate videos.csv schema
    # =====================================================

    missing_columns = [
        col
        for col in EXPECTED_VIDEO_COLUMNS
        if col not in videos.columns
    ]

    if missing_columns:
        raise ValueError(
            f"videos.csv missing columns: {missing_columns}"
        )

    # =====================================================
    # Get frozen C++ subset
    # =====================================================

    selected_ids = list(PROMOTION_MAP.keys())

    selected = candidates[
        candidates["candidate_id"].isin(selected_ids)
    ].copy()

    if len(selected) != len(selected_ids):
        found = set(selected["candidate_id"])
        missing = set(selected_ids) - found

        raise ValueError(
            f"Missing selected C++ candidates: "
            f"{sorted(missing)}"
        )

    if selected["candidate_id"].duplicated().any():
        raise ValueError(
            "Duplicate selected C++ candidate_id."
        )

    # =====================================================
    # Validate subject
    # =====================================================

    bad_subject = selected[
        selected["subject"]
        .astype(str)
        .str.strip()
        .str.lower()
        != "c++"
    ]

    if not bad_subject.empty:
        raise ValueError(
            "Non-C++ candidate in frozen C++ selection."
        )

    # =====================================================
    # Validate transcript
    # =====================================================

    bad_transcript = selected[
        selected["transcript_available"]
        .astype(str)
        .str.strip()
        .str.lower()
        != "yes"
    ]

    if not bad_transcript.empty:
        raise ValueError(
            "Selected C++ candidate without "
            "confirmed English transcript."
        )

    # =====================================================
    # Validate eligibility
    # =====================================================

    bad_decision = selected[
        selected["decision"]
        .astype(str)
        .str.strip()
        .str.lower()
        != "include"
    ]

    if not bad_decision.empty:
        raise ValueError(
            "Selected C++ candidate is not "
            "decision=include."
        )

    # =====================================================
    # URL uniqueness inside selection
    # =====================================================

    if selected["url"].duplicated().any():
        raise ValueError(
            "Duplicate URL inside selected C++ subset."
        )

    # =====================================================
    # Build official v31-v40 rows
    # =====================================================

    promotion_rows = []

    for candidate_id, video_id in PROMOTION_MAP.items():
        row = selected[
            selected["candidate_id"] == candidate_id
        ].iloc[0]

        promotion_rows.append({
            "video_id": video_id,
            "subject": "C++",
            "title": clean_text(row["title"]),
            "url": clean_text(row["url"]),
            "status": "selected",
            "split": "dev",
            "cv_fold": "",
            "annotation_status": "pending",
            "content_style":
                clean_text(row["content_style"]),
            "duration_min":
                clean_text(row["duration_min"]),
            "channel":
                clean_text(row["channel"]),
        })

    promotion_df = pd.DataFrame(
        promotion_rows,
        columns=EXPECTED_VIDEO_COLUMNS,
    )

    # =====================================================
    # Protect against accidental re-promotion
    # =====================================================

    target_video_ids = set(PROMOTION_MAP.values())

    existing_targets = videos[
        videos["video_id"].isin(target_video_ids)
    ].copy()

    if not existing_targets.empty:

        if len(existing_targets) != len(target_video_ids):
            raise ValueError(
                "Only some of v31-v40 already exist. "
                "Inspect videos.csv."
            )

        expected_map = dict(
            zip(
                promotion_df["video_id"],
                promotion_df["url"],
            )
        )

        actual_map = dict(
            zip(
                existing_targets["video_id"],
                existing_targets["url"],
            )
        )

        if expected_map != actual_map:
            raise ValueError(
                "Existing v31-v40 URLs do not match "
                "the frozen C++ selection."
            )

        print(
            "C++ v31-v40 already promoted "
            "with correct frozen URLs."
        )

    else:

        # =================================================
        # Prevent overlap with v1-v30
        # =================================================

        overlap = (
            set(promotion_df["url"])
            & set(videos["url"].astype(str))
        )

        if overlap:
            raise ValueError(
                "C++ URL already exists in videos.csv: "
                f"{sorted(overlap)}"
            )

        videos = pd.concat(
            [videos, promotion_df],
            ignore_index=True,
        )

        videos.to_csv(
            VIDEOS_FILE,
            index=False,
            encoding="utf-8-sig",
        )

        print(
            "Promoted C++ candidates successfully."
        )

    # =====================================================
    # Final mapping audit
    # =====================================================

    final_videos = pd.read_csv(VIDEOS_FILE)

    cpp_new = final_videos[
        final_videos["video_id"].isin(target_video_ids)
    ].copy()

    cpp_new["_num"] = (
        cpp_new["video_id"]
        .str.replace("v", "", regex=False)
        .astype(int)
    )

    cpp_new = cpp_new.sort_values("_num")

    print()
    print("=" * 72)
    print("C++ PROMOTION MAP")
    print("=" * 72)

    print(
        cpp_new[
            [
                "video_id",
                "title",
                "channel",
                "content_style",
                "url",
                "annotation_status",
            ]
        ].to_string(index=False)
    )

    # =====================================================
    # Whole DEV audit
    # =====================================================

    print()
    print("=" * 72)
    print("FINAL P0.4 DEV DATASET AUDIT")
    print("=" * 72)

    dev = final_videos[
        (final_videos["status"] == "selected")
        & (final_videos["split"] == "dev")
    ].copy()

    print("TOTAL DEV:", len(dev))

    print()
    print(dev["subject"].value_counts())

    print()
    print(
        "DUP VIDEO IDs:",
        int(dev["video_id"].duplicated().sum()),
    )

    print(
        "DUP URLs:",
        int(dev["url"].duplicated().sum()),
    )

    # =====================================================
    # Check complete v1-v40 sequence
    # =====================================================

    expected_ids = {
        f"v{i}"
        for i in range(1, 41)
    }

    actual_ids = set(
        dev["video_id"].astype(str)
    )

    missing_ids = sorted(
        expected_ids - actual_ids,
        key=lambda x: int(x[1:]),
    )

    extra_ids = sorted(
        actual_ids - expected_ids
    )

    print()
    print("MISSING IDs:", missing_ids)
    print("EXTRA IDs:", extra_ids)

    # =====================================================
    # Hard final checks
    # =====================================================

    expected_subject_counts = {
        "SQL": 10,
        "Python": 10,
        "Java": 10,
        "C++": 10,
    }

    actual_subject_counts = (
        dev["subject"]
        .value_counts()
        .to_dict()
    )

    if len(dev) != 40:
        raise ValueError(
            f"Expected 40 DEV videos, got {len(dev)}."
        )

    if actual_subject_counts != expected_subject_counts:
        raise ValueError(
            "Subject distribution is not 10/10/10/10: "
            f"{actual_subject_counts}"
        )

    if dev["video_id"].duplicated().any():
        raise ValueError(
            "Duplicate video_id in DEV dataset."
        )

    if dev["url"].duplicated().any():
        raise ValueError(
            "Duplicate URL in DEV dataset."
        )

    if missing_ids:
        raise ValueError(
            f"Missing DEV video IDs: {missing_ids}"
        )

    if extra_ids:
        raise ValueError(
            f"Unexpected DEV video IDs: {extra_ids}"
        )

    print()
    print("=" * 72)
    print("P0.4 VIDEO SELECTION: PASS")
    print("=" * 72)


if __name__ == "__main__":
    main()