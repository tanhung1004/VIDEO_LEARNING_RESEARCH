from pathlib import Path

import pandas as pd
from youtube_transcript_api import YouTubeTranscriptApi


PROJECT_ROOT = Path(__file__).resolve().parents[2]

VIDEOS_FILE = (
    PROJECT_ROOT / "data" / "raw" / "videos.csv"
)

TRANSCRIPT_DIR = (
    PROJECT_ROOT / "data" / "raw" / "transcripts"
)

RESULT_DIR = (
    PROJECT_ROOT / "module1" / "results" / "p0_dataset"
)

TRANSCRIPT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


def get_youtube_id(url):
    url = str(url).strip()

    if "watch?v=" in url:
        return (
            url.split("watch?v=")[1]
            .split("&")[0]
        )

    if "youtu.be/" in url:
        return (
            url.split("youtu.be/")[1]
            .split("?")[0]
        )

    return url


def seconds_to_time(seconds):
    seconds = int(seconds)

    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60

    return (
        f"{hours:02d}:"
        f"{minutes:02d}:"
        f"{secs:02d}"
    )


def fetch_transcript(api, video_row):
    video_id = str(
        video_row["video_id"]
    ).strip()

    subject = str(
        video_row["subject"]
    ).strip()

    title = str(
        video_row["title"]
    ).strip()

    url = str(
        video_row["url"]
    ).strip()

    youtube_id = get_youtube_id(url)

    output_file = (
        TRANSCRIPT_DIR
        / f"{video_id}_transcript.csv"
    )

    print()
    print("=" * 72)
    print("Video:", video_id)
    print("Subject:", subject)
    print("Title:", title)
    print("YouTube ID:", youtube_id)
    print("URL:", url)

    # =====================================================
    # IMPORTANT:
    # Preserve existing pilot/raw transcript files.
    # =====================================================

    if output_file.exists():
        try:
            existing = pd.read_csv(
                output_file
            )

            rows = len(existing)

        except Exception:
            rows = -1

        print(
            "SKIP - transcript already exists"
        )

        print(
            "File:",
            output_file,
        )

        return {
            "video_id": video_id,
            "subject": subject,
            "youtube_id": youtube_id,
            "status": "skipped_existing",
            "transcript_rows": rows,
            "output_file": str(output_file),
            "error": "",
        }

    try:
        # English only:
        # matches P0 video-selection protocol.
        transcript = api.fetch(
            youtube_id,
            languages=["en"],
        )

        raw_data = transcript.to_raw_data()

        if not raw_data:
            raise ValueError(
                "Empty English transcript"
            )

        rows = []

        for item in raw_data:
            start = float(
                item["start"]
            )

            duration = float(
                item.get(
                    "duration",
                    0,
                )
            )

            end = start + duration

            text = str(
                item["text"]
            )

            text = (
                text
                .replace("\n", " ")
                .strip()
            )

            rows.append({
                "video_id":
                    video_id,

                "subject":
                    subject,

                "youtube_id":
                    youtube_id,

                "start_sec":
                    round(start, 2),

                "end_sec":
                    round(end, 2),

                "start_time":
                    seconds_to_time(
                        start
                    ),

                "end_time":
                    seconds_to_time(
                        end
                    ),

                "text":
                    text,
            })

        transcript_df = pd.DataFrame(
            rows
        )

        transcript_df.to_csv(
            output_file,
            index=False,
            encoding="utf-8-sig",
        )

        print("SUCCESS")
        print(
            "Transcript rows:",
            len(transcript_df),
        )
        print(
            "File:",
            output_file,
        )

        return {
            "video_id": video_id,
            "subject": subject,
            "youtube_id": youtube_id,
            "status": "success",
            "transcript_rows":
                len(transcript_df),
            "output_file":
                str(output_file),
            "error": "",
        }

    except Exception as exc:
        error_name = type(
            exc
        ).__name__

        print("FAILED")
        print(
            "Error:",
            error_name,
            str(exc),
        )

        return {
            "video_id": video_id,
            "subject": subject,
            "youtube_id": youtube_id,
            "status": "failed",
            "transcript_rows": 0,
            "output_file": "",
            "error":
                f"{error_name}: {exc}",
        }


def main():
    print("=" * 72)
    print("P0.5 - DEV TRANSCRIPT ACQUISITION")
    print("=" * 72)

    videos = pd.read_csv(
        VIDEOS_FILE
    )

    required_columns = {
        "video_id",
        "subject",
        "title",
        "url",
        "status",
        "split",
    }

    missing_columns = (
        required_columns
        - set(videos.columns)
    )

    if missing_columns:
        raise ValueError(
            "videos.csv missing columns: "
            f"{sorted(missing_columns)}"
        )

    # =====================================================
    # Only official development videos.
    # =====================================================

    dev = videos[
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
    ].copy()

    dev["_num"] = (
        dev["video_id"]
        .astype(str)
        .str.replace(
            "v",
            "",
            regex=False,
        )
        .astype(int)
    )

    dev = dev.sort_values(
        "_num"
    )

    print(
        "Selected DEV videos:",
        len(dev),
    )

    print(
        dev["subject"]
        .value_counts()
    )

    # =====================================================
    # Hard dataset checks
    # =====================================================

    if len(dev) != 40:
        raise ValueError(
            f"Expected 40 DEV videos, "
            f"got {len(dev)}."
        )

    if dev["video_id"].duplicated().any():
        raise ValueError(
            "Duplicate video_id."
        )

    if dev["url"].duplicated().any():
        raise ValueError(
            "Duplicate URL."
        )

    expected_ids = {
        f"v{i}"
        for i in range(1, 41)
    }

    actual_ids = set(
        dev["video_id"]
        .astype(str)
    )

    if actual_ids != expected_ids:
        missing = sorted(
            expected_ids - actual_ids,
            key=lambda x: int(x[1:]),
        )

        extra = sorted(
            actual_ids - expected_ids
        )

        raise ValueError(
            f"DEV IDs incorrect. "
            f"Missing={missing}, "
            f"Extra={extra}"
        )

    # =====================================================
    # Fetch
    # =====================================================

    api = YouTubeTranscriptApi()

    summary = []

    for _, video_row in (
        dev.iterrows()
    ):
        result = fetch_transcript(
            api,
            video_row,
        )

        summary.append(
            result
        )

    summary_df = pd.DataFrame(
        summary
    )

    summary_file = (
        RESULT_DIR
        / "transcript_acquisition_summary.csv"
    )

    summary_df.to_csv(
        summary_file,
        index=False,
        encoding="utf-8-sig",
    )

    # =====================================================
    # Final audit
    # =====================================================

    print()
    print("=" * 72)
    print("P0.5 SUMMARY")
    print("=" * 72)

    print(
        summary_df["status"]
        .value_counts()
    )

    transcript_files = list(
        TRANSCRIPT_DIR.glob(
            "v*_transcript.csv"
        )
    )

    found_ids = {
        path.stem.replace(
            "_transcript",
            "",
        )
        for path in transcript_files
    }

    missing_files = sorted(
        expected_ids - found_ids,
        key=lambda x: int(x[1:]),
    )

    print()
    print(
        "Transcript files found:",
        len(
            expected_ids
            & found_ids
        ),
    )

    print(
        "Missing transcript files:",
        missing_files,
    )

    print(
        "Summary:",
        summary_file,
    )

    failed = summary_df[
        summary_df["status"]
        == "failed"
    ]

    if not failed.empty:
        print()
        print("FAILED VIDEOS:")

        print(
            failed[
                [
                    "video_id",
                    "subject",
                    "error",
                ]
            ].to_string(
                index=False
            )
        )

        print()
        print(
            "P0.5 NOT COMPLETE - "
            "retry failed videos later."
        )

    elif missing_files:
        print(
            "P0.5 NOT COMPLETE - "
            "some transcript files missing."
        )

    else:
        print()
        print("=" * 72)
        print(
            "P0.5 TRANSCRIPT ACQUISITION: PASS"
        )
        print("=" * 72)


if __name__ == "__main__":
    main()