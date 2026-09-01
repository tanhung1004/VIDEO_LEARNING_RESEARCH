from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

VIDEOS_FILE = (
    PROJECT_ROOT / "data" / "raw" / "videos.csv"
)

CATALOG_FILE = (
    PROJECT_ROOT / "data" / "concepts" / "concept_catalog.csv"
)

TRANSCRIPT_DIR = (
    PROJECT_ROOT / "data" / "raw" / "transcripts"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "ground_truth"
    / "annotation_working"
)

SHEET_FILE = OUTPUT_DIR / "annotation_sheet.csv"

MANIFEST_FILE = OUTPUT_DIR / "annotation_manifest.csv"

REVIEW_DIR = OUTPUT_DIR / "video_review"


def video_number(video_id):
    return int(
        str(video_id)
        .strip()
        .replace("v", "")
    )


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    REVIEW_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    videos = pd.read_csv(VIDEOS_FILE)
    catalog = pd.read_csv(CATALOG_FILE)

    # =====================================================
    # 1. ONLY pending DEV videos
    # =====================================================

    pending = videos[
        (
            videos["status"]
            .astype(str)
            .str.strip()
            .str.lower()
            == "selected"
        )
        &
        (
            videos["split"]
            .astype(str)
            .str.strip()
            .str.lower()
            == "dev"
        )
        &
        (
            videos["annotation_status"]
            .astype(str)
            .str.strip()
            .str.lower()
            == "pending"
        )
    ].copy()

    pending["_num"] = (
        pending["video_id"]
        .apply(video_number)
    )

    pending = pending.sort_values("_num")

    print("=" * 72)
    print("P0.6 - BUILD ANNOTATION SHEET")
    print("=" * 72)

    print("Pending videos:", len(pending))

    print()
    print(pending["subject"].value_counts())

    if len(pending) != 35:
        raise ValueError(
            f"Expected 35 pending videos, got {len(pending)}"
        )

    # =====================================================
    # 2. Check transcripts
    # =====================================================

    missing_transcripts = []

    for video_id in pending["video_id"]:

        transcript_file = (
            TRANSCRIPT_DIR
            / f"{video_id}_transcript.csv"
        )

        if not transcript_file.exists():
            missing_transcripts.append(video_id)

    if missing_transcripts:
        raise ValueError(
            "Missing transcript files: "
            f"{missing_transcripts}"
        )

    # =====================================================
    # 3. Build readable review TXT for each video
    # =====================================================

    manifest_rows = []

    for _, video in pending.iterrows():

        video_id = video["video_id"]
        subject = video["subject"]

        transcript_file = (
            TRANSCRIPT_DIR
            / f"{video_id}_transcript.csv"
        )

        transcript = pd.read_csv(
            transcript_file
        )

        review_file = (
            REVIEW_DIR
            / f"{video_id}_review.txt"
        )

        with open(
            review_file,
            "w",
            encoding="utf-8"
        ) as f:

            f.write(
                f"VIDEO ID: {video_id}\n"
            )

            f.write(
                f"SUBJECT: {subject}\n"
            )

            f.write(
                f"TITLE: {video['title']}\n"
            )

            f.write(
                f"URL: {video['url']}\n"
            )

            f.write("\n")
            f.write("=" * 72)
            f.write("\nTRANSCRIPT\n")
            f.write("=" * 72)
            f.write("\n\n")

            for _, row in transcript.iterrows():

                start = row.get(
                    "start_time",
                    ""
                )

                end = row.get(
                    "end_time",
                    ""
                )

                text = str(
                    row.get(
                        "text",
                        ""
                    )
                )

                f.write(
                    f"[{start} - {end}] "
                    f"{text}\n"
                )

        manifest_rows.append({
            "video_id": video_id,
            "subject": subject,
            "title": video["title"],
            "url": video["url"],
            "transcript_file":
                str(transcript_file),
            "review_file":
                str(review_file),
        })

    manifest_df = pd.DataFrame(
        manifest_rows
    )

    manifest_df.to_csv(
        MANIFEST_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    # =====================================================
    # 4. Build video × subject-concept annotation grid
    # =====================================================

    annotation_rows = []

    for _, video in pending.iterrows():

        video_id = video["video_id"]
        subject = str(
            video["subject"]
        ).strip()

        subject_catalog = catalog[
            catalog["subject"]
            .astype(str)
            .str.strip()
            .str.lower()
            == subject.lower()
        ].copy()

        if subject_catalog.empty:
            raise ValueError(
                f"No concepts found for subject {subject}"
            )

        for _, concept_row in (
            subject_catalog.iterrows()
        ):

            annotation_rows.append({
                "video_id":
                    video_id,

                "subject":
                    subject,

                "title":
                    video["title"],

                "url":
                    video["url"],

                "concept":
                    concept_row["concept"],

                "concept_description":
                    concept_row["description"],

                # Human annotator fills this:
                # 1 = meaningfully taught/explained/demo
                # 0 = not positive
                "positive":
                    "",

                # Strongly recommended:
                "evidence_start_sec":
                    "",

                "evidence_end_sec":
                    "",

                "evidence_note":
                    "",

                "annotator":
                    "",

                "review_status":
                    "pending",
            })

    annotation_df = pd.DataFrame(
        annotation_rows
    )

    annotation_df.to_csv(
        SHEET_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    # =====================================================
    # 5. Audit expected row count
    # =====================================================

    expected_rows = 0

    for subject, count in (
        pending["subject"]
        .value_counts()
        .items()
    ):

        concept_count = len(
            catalog[
                catalog["subject"]
                .astype(str)
                .str.strip()
                .str.lower()
                == str(subject)
                .strip()
                .lower()
            ]
        )

        expected_rows += (
            count * concept_count
        )

    print()
    print("=" * 72)
    print("ANNOTATION SHEET SUMMARY")
    print("=" * 72)

    print(
        "Videos:",
        annotation_df["video_id"].nunique()
    )

    print(
        "Annotation rows:",
        len(annotation_df)
    )

    print(
        "Expected rows:",
        expected_rows
    )

    print()
    print(
        annotation_df
        .groupby("subject")
        .size()
    )

    print()
    print("Sheet:", SHEET_FILE)
    print("Manifest:", MANIFEST_FILE)
    print("Review dir:", REVIEW_DIR)

    if len(annotation_df) != expected_rows:
        raise ValueError(
            "Annotation row count mismatch."
        )

    print()
    print("=" * 72)
    print("P0.6 ANNOTATION SHEET BUILD: PASS")
    print("=" * 72)


if __name__ == "__main__":
    main()
    