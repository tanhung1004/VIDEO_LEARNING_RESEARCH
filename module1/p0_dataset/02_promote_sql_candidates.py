from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CANDIDATE_FILE = (
    PROJECT_ROOT / "data" / "raw" / "video_candidates.csv"
)

VIDEOS_FILE = (
    PROJECT_ROOT / "data" / "raw" / "videos.csv"
)


EXPECTED_SQL_CANDIDATES = [
    "c001",
    "c002",
    "c003",
    "c004",
    "c005",
    "c006",
    "c007",
    "c008",
]

NEW_VIDEO_IDS = [
    "v6",
    "v7",
    "v8",
    "v9",
    "v10",
    "v11",
    "v12",
    "v13",
]


def main():
    candidates = pd.read_csv(CANDIDATE_FILE)
    videos = pd.read_csv(VIDEOS_FILE)

    selected = candidates[
        candidates["candidate_id"].isin(
            EXPECTED_SQL_CANDIDATES
        )
    ].copy()

    # ---------------------------------------------
    # Scientific / data-integrity checks
    # ---------------------------------------------

    if len(selected) != 8:
        raise ValueError(
            f"Expected 8 SQL candidates, found {len(selected)}"
        )

    if set(selected["candidate_id"]) != set(
        EXPECTED_SQL_CANDIDATES
    ):
        raise ValueError(
            "SQL candidate IDs do not match expected set."
        )

    if not (selected["subject"] == "SQL").all():
        raise ValueError(
            "Non-SQL candidate detected."
        )

    if not (selected["decision"] == "include").all():
        raise ValueError(
            "At least one candidate is not marked include."
        )

    if not (
        selected["transcript_available"] == "yes"
    ).all():
        raise ValueError(
            "At least one candidate lacks a verified transcript."
        )

    if selected["url"].duplicated().any():
        raise ValueError(
            "Duplicate URL inside SQL candidate batch."
        )

    if selected["url"].isin(videos["url"]).any():
        bad = selected[
            selected["url"].isin(videos["url"])
        ]

        raise ValueError(
            "Candidate URL already exists in videos.csv:\n"
            + bad[
                ["candidate_id", "url"]
            ].to_string(index=False)
        )

    if any(
        video_id in set(videos["video_id"])
        for video_id in NEW_VIDEO_IDS
    ):
        raise ValueError(
            "One or more IDs v6-v13 already exist."
        )

    # Keep deterministic ordering.
    selected["candidate_order"] = (
        selected["candidate_id"]
        .str.extract(r"(\d+)")
        .astype(int)
    )

    selected = selected.sort_values(
        "candidate_order"
    ).reset_index(drop=True)

    # ---------------------------------------------
    # Promote candidates to official DEV videos
    # ---------------------------------------------

    new_rows = pd.DataFrame({
        "video_id": NEW_VIDEO_IDS,
        "subject": selected["subject"],
        "title": selected["title"],
        "url": selected["url"],
        "status": "selected",
        "split": "dev",
        "cv_fold": "",
        "annotation_status": "pending",
        "content_style": selected["content_style"],
        "duration_min": selected["duration_min"],
        "channel": selected["channel"],
    })

    updated = pd.concat(
        [videos, new_rows],
        ignore_index=True
    )

    # Final integrity checks.
    if updated["video_id"].duplicated().any():
        raise ValueError(
            "Duplicate video_id after promotion."
        )

    if updated["url"].duplicated().any():
        raise ValueError(
            "Duplicate URL after promotion."
        )

    updated.to_csv(
        VIDEOS_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print("=" * 72)
    print("SQL CANDIDATES PROMOTED")
    print("=" * 72)

    print(
        new_rows[
            [
                "video_id",
                "subject",
                "title",
                "channel",
                "duration_min",
                "content_style",
                "annotation_status",
            ]
        ].to_string(index=False)
    )

    print()
    print(
        "Total videos:",
        len(updated)
    )

    print()
    print("Subject counts:")

    print(
        updated[
            updated["split"] == "dev"
        ]["subject"].value_counts()
    )


if __name__ == "__main__":
    main()