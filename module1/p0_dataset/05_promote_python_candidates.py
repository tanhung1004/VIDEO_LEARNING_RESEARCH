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
# FROZEN PYTHON SELECTION
#
# IMPORTANT:
# - selected before model evaluation
# - do not change based on LDA / LSA / OCR / F1
# - URL must be copied exactly from video_candidates.csv
# =========================================================

PROMOTION_MAP = {
    "c013": "v14",
    "c014": "v15",
    "c015": "v16",
    "c016": "v17",
    "c022": "v18",
    "c023": "v19",
    "c024": "v20",
    "c025": "v21",
    "c026": "v22",
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

    # -----------------------------------------------------
    # Validate videos.csv schema
    # -----------------------------------------------------

    missing_columns = [
        col
        for col in EXPECTED_VIDEO_COLUMNS
        if col not in videos.columns
    ]

    if missing_columns:
        raise ValueError(
            f"videos.csv missing columns: {missing_columns}"
        )

    # -----------------------------------------------------
    # Validate candidate IDs
    # -----------------------------------------------------

    selected_ids = list(PROMOTION_MAP.keys())

    selected = candidates[
        candidates["candidate_id"].isin(selected_ids)
    ].copy()

    if len(selected) != len(selected_ids):
        found = set(selected["candidate_id"])
        missing = set(selected_ids) - found

        raise ValueError(
            f"Missing selected candidates: {sorted(missing)}"
        )

    if selected["candidate_id"].duplicated().any():
        raise ValueError(
            "Duplicate candidate_id in selected Python rows."
        )

    # -----------------------------------------------------
    # Validate subject
    # -----------------------------------------------------

    bad_subject = selected[
        selected["subject"]
        .astype(str)
        .str.strip()
        .str.lower()
        != "python"
    ]

    if not bad_subject.empty:
        raise ValueError(
            "Selected candidate with non-Python subject."
        )

    # -----------------------------------------------------
    # Validate transcript screening
    # -----------------------------------------------------

    bad_transcript = selected[
        selected["transcript_available"]
        .astype(str)
        .str.strip()
        .str.lower()
        != "yes"
    ]

    if not bad_transcript.empty:
        raise ValueError(
            "Selected Python candidate without "
            "confirmed English transcript."
        )

    # -----------------------------------------------------
    # Validate eligibility
    # -----------------------------------------------------

    bad_decision = selected[
        selected["decision"]
        .astype(str)
        .str.strip()
        .str.lower()
        != "include"
    ]

    if not bad_decision.empty:
        raise ValueError(
            "Selected Python candidate is not include."
        )

    # -----------------------------------------------------
    # Candidate URL duplicate check
    # -----------------------------------------------------

    if selected["url"].duplicated().any():
        raise ValueError(
            "Duplicate URLs inside selected Python candidates."
        )

    # -----------------------------------------------------
    # Prepare official rows
    # -----------------------------------------------------

    promotion_rows = []

    for candidate_id, video_id in PROMOTION_MAP.items():
        row = selected[
            selected["candidate_id"] == candidate_id
        ].iloc[0]

        promotion_rows.append({
            "video_id": video_id,
            "subject": "Python",
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

    # -----------------------------------------------------
    # Check whether v14-v22 already exist
    # -----------------------------------------------------

    target_video_ids = set(PROMOTION_MAP.values())

    existing_targets = videos[
        videos["video_id"].isin(target_video_ids)
    ].copy()

    if not existing_targets.empty:
        if len(existing_targets) != len(target_video_ids):
            raise ValueError(
                "Only some of v14-v22 already exist. "
                "Stop and inspect videos.csv manually."
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
                "v14-v22 already exist but URLs "
                "do not match frozen Python selection."
            )

        print(
            "Python videos v14-v22 are already promoted "
            "with the correct frozen URLs."
        )

    else:
        # -------------------------------------------------
        # Prevent URL overlap with existing selected videos
        # -------------------------------------------------

        overlap = set(promotion_df["url"]) & set(
            videos["url"].astype(str)
        )

        if overlap:
            raise ValueError(
                f"Python URLs already exist in videos.csv: "
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

        print("Promoted Python candidates successfully.")

    # -----------------------------------------------------
    # Final audit
    # -----------------------------------------------------

    final_videos = pd.read_csv(VIDEOS_FILE)

    print()
    print("=" * 72)
    print("PYTHON PROMOTION MAP")
    print("=" * 72)

    python_new = final_videos[
        final_videos["video_id"].isin(target_video_ids)
    ].copy()

    python_new["_num"] = (
        python_new["video_id"]
        .str.replace("v", "", regex=False)
        .astype(int)
    )

    python_new = python_new.sort_values("_num")

    print(
        python_new[
            [
                "video_id",
                "title",
                "channel",
                "url",
                "annotation_status",
            ]
        ].to_string(index=False)
    )

    print()
    print("=" * 72)
    print("DEV DATASET AUDIT")
    print("=" * 72)

    dev = final_videos[
        (final_videos["status"] == "selected")
        & (final_videos["split"] == "dev")
    ]

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


if __name__ == "__main__":
    main()