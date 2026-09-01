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
# FROZEN JAVA SELECTION
#
# Selected using:
# - transcript availability
# - concept coverage
# - channel diversity
# - content-style diversity
#
# NOT selected using:
# - LDA
# - LSA
# - OCR
# - model predictions
# - F1
# =========================================================

PROMOTION_MAP = {
    "c027": "v23",
    "c029": "v24",
    "c030": "v25",
    "c031": "v26",
    "c034": "v27",
    "c035": "v28",
    "c036": "v29",
    "c038": "v30",
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
    # Schema validation
    # =====================================================

    missing_columns = [
        col
        for col in EXPECTED_VIDEO_COLUMNS
        if col not in videos.columns
    ]

    if missing_columns:
        raise ValueError(
            f"videos.csv missing columns: "
            f"{missing_columns}"
        )

    selected_ids = list(
        PROMOTION_MAP.keys()
    )

    selected = candidates[
        candidates["candidate_id"].isin(
            selected_ids
        )
    ].copy()

    # =====================================================
    # Candidate validation
    # =====================================================

    if len(selected) != len(selected_ids):
        found = set(
            selected["candidate_id"]
        )

        missing = (
            set(selected_ids)
            - found
        )

        raise ValueError(
            f"Missing Java candidates: "
            f"{sorted(missing)}"
        )

    if selected[
        "candidate_id"
    ].duplicated().any():
        raise ValueError(
            "Duplicate selected candidate_id."
        )

    # -----------------------------------------------------
    # Subject
    # -----------------------------------------------------

    bad_subject = selected[
        selected["subject"]
        .astype(str)
        .str.strip()
        .str.lower()
        != "java"
    ]

    if not bad_subject.empty:
        raise ValueError(
            "Non-Java candidate selected."
        )

    # -----------------------------------------------------
    # Transcript
    # -----------------------------------------------------

    bad_transcript = selected[
        selected[
            "transcript_available"
        ]
        .astype(str)
        .str.strip()
        .str.lower()
        != "yes"
    ]

    if not bad_transcript.empty:
        raise ValueError(
            "Selected Java candidate "
            "without confirmed transcript."
        )

    # -----------------------------------------------------
    # Eligibility
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
            "Selected Java candidate "
            "is not decision=include."
        )

    # -----------------------------------------------------
    # URL uniqueness inside final Java subset
    # -----------------------------------------------------

    if selected["url"].duplicated().any():
        raise ValueError(
            "Duplicate URL inside "
            "selected Java candidates."
        )

    # =====================================================
    # Build promotion rows
    # =====================================================

    promotion_rows = []

    for candidate_id, video_id in (
        PROMOTION_MAP.items()
    ):
        row = selected[
            selected["candidate_id"]
            == candidate_id
        ].iloc[0]

        promotion_rows.append({
            "video_id": video_id,

            "subject": "Java",

            "title":
                clean_text(
                    row["title"]
                ),

            "url":
                clean_text(
                    row["url"]
                ),

            "status": "selected",

            "split": "dev",

            "cv_fold": "",

            "annotation_status":
                "pending",

            "content_style":
                clean_text(
                    row[
                        "content_style"
                    ]
                ),

            "duration_min":
                clean_text(
                    row[
                        "duration_min"
                    ]
                ),

            "channel":
                clean_text(
                    row["channel"]
                ),
        })

    promotion_df = pd.DataFrame(
        promotion_rows,
        columns=EXPECTED_VIDEO_COLUMNS,
    )

    # =====================================================
    # Protect against accidental re-promotion
    # =====================================================

    target_video_ids = set(
        PROMOTION_MAP.values()
    )

    existing_targets = videos[
        videos["video_id"].isin(
            target_video_ids
        )
    ].copy()

    if not existing_targets.empty:

        if len(
            existing_targets
        ) != len(
            target_video_ids
        ):
            raise ValueError(
                "Only some of v23-v30 "
                "already exist. "
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
                existing_targets[
                    "video_id"
                ],
                existing_targets[
                    "url"
                ],
            )
        )

        if expected_map != actual_map:
            raise ValueError(
                "Existing v23-v30 URLs "
                "do not match frozen "
                "Java selection."
            )

        print(
            "Java v23-v30 already "
            "promoted with correct URLs."
        )

    else:

        # -------------------------------------------------
        # Prevent overlap with existing official videos
        # -------------------------------------------------

        overlap = (
            set(
                promotion_df["url"]
            )
            & set(
                videos["url"]
                .astype(str)
            )
        )

        if overlap:
            raise ValueError(
                "Java URL already exists "
                "in videos.csv: "
                f"{sorted(overlap)}"
            )

        videos = pd.concat(
            [
                videos,
                promotion_df,
            ],
            ignore_index=True,
        )

        videos.to_csv(
            VIDEOS_FILE,
            index=False,
            encoding="utf-8-sig",
        )

        print(
            "Promoted Java candidates "
            "successfully."
        )

    # =====================================================
    # Final audit
    # =====================================================

    final_videos = pd.read_csv(
        VIDEOS_FILE
    )

    java_new = final_videos[
        final_videos["video_id"].isin(
            target_video_ids
        )
    ].copy()

    java_new["_num"] = (
        java_new["video_id"]
        .str.replace(
            "v",
            "",
            regex=False,
        )
        .astype(int)
    )

    java_new = (
        java_new
        .sort_values("_num")
    )

    print()
    print("=" * 72)
    print("JAVA PROMOTION MAP")
    print("=" * 72)

    print(
        java_new[
            [
                "video_id",
                "title",
                "channel",
                "content_style",
                "url",
                "annotation_status",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print("=" * 72)
    print("DEV DATASET AUDIT")
    print("=" * 72)

    dev = final_videos[
        (
            final_videos["status"]
            == "selected"
        )
        &
        (
            final_videos["split"]
            == "dev"
        )
    ]

    print(
        "TOTAL DEV:",
        len(dev),
    )

    print()

    print(
        dev["subject"]
        .value_counts()
    )

    print()

    print(
        "DUP VIDEO IDs:",
        int(
            dev[
                "video_id"
            ]
            .duplicated()
            .sum()
        ),
    )

    print(
        "DUP URLs:",
        int(
            dev[
                "url"
            ]
            .duplicated()
            .sum()
        ),
    )


if __name__ == "__main__":
    main()