from pathlib import Path
from datetime import date

import pandas as pd
from youtube_transcript_api import YouTubeTranscriptApi


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CANDIDATE_FILE = (
    PROJECT_ROOT / "data" / "raw" / "video_candidates.csv"
)


CANDIDATES = [
    {
        "candidate_id": "c009",
        "youtube_id": "jk6_L0k8VPg",
        "subject": "SQL",
        "title": "Group By And Having Clause In SQL",
        "channel": "Simplilearn",
        "duration_min": 31.30,
        "content_style": "ide_demo",
        "search_query": "SQL GROUP BY HAVING tutorial",
    },
    {
        "candidate_id": "c010",
        "youtube_id": "2HVMiPPuPIM",
        "subject": "SQL",
        "title": "SQL Joins Tutorial for Beginners",
        "channel": "Joey Blue",
        "duration_min": 15.5,
        "content_style": "code_tutorial",
        "search_query": "SQL JOIN INNER LEFT tutorial",
    },
    {
        "candidate_id": "c011",
        "youtube_id": "i_IwrISHu7A",
        "subject": "SQL",
        "title": "SQL Aggregate Functions Explained",
        "channel": "Rita Onuoha",
        "duration_min": "",
        "content_style": "concept_explanation",
        "search_query": "SQL aggregate SUM COUNT AVG tutorial",
    },
    {
        "candidate_id": "c012",
        "youtube_id": "iq52vhD45A4",
        "subject": "SQL",
        "title": "SQL Subqueries With Examples",
        "channel": "Extern Code",
        "duration_min": "",
        "content_style": "code_tutorial",
        "search_query": "SQL subquery tutorial",
    },
]


COLUMNS = [
    "candidate_id",
    "subject",
    "title",
    "url",
    "channel",
    "duration_min",
    "content_style",
    "transcript_available",
    "search_query",
    "retrieval_date",
    "decision",
    "exclusion_reason",
]


def check_transcript(api, youtube_id):
    try:
        transcript = api.fetch(
            youtube_id,
            languages=["en"]
        )

        rows = transcript.to_raw_data()

        if not rows:
            return "no", "exclude", "empty_transcript", 0

        return "yes", "include", "", len(rows)

    except Exception as exc:
        error_name = type(exc).__name__

        hard_failures = {
            "TranscriptsDisabled",
            "NoTranscriptFound",
            "VideoUnavailable",
        }

        if error_name in hard_failures:
            return "no", "exclude", error_name, 0

        # Network / API / IP failure is not evidence
        # that the video itself is unsuitable.
        return "api_error", "pending", error_name, 0


def main():
    api = YouTubeTranscriptApi()

    rows = []

    for item in CANDIDATES:
        transcript_status, decision, reason, transcript_rows = (
            check_transcript(api, item["youtube_id"])
        )

        print("=" * 70)
        print(item["candidate_id"], item["title"])
        print("Transcript:", transcript_status)
        print("Transcript rows:", transcript_rows)
        print("Decision:", decision)

        if reason:
            print("Reason:", reason)

        rows.append({
            "candidate_id": item["candidate_id"],
            "subject": item["subject"],
            "title": item["title"],
            "url": (
                "https://www.youtube.com/watch?v="
                + item["youtube_id"]
            ),
            "channel": item["channel"],
            "duration_min": item["duration_min"],
            "content_style": item["content_style"],
            "transcript_available": transcript_status,
            "search_query": item["search_query"],
            "retrieval_date": date.today().isoformat(),
            "decision": decision,
            "exclusion_reason": reason,
        })

    reserve_df = pd.DataFrame(
        rows,
        columns=COLUMNS
    )

    if CANDIDATE_FILE.exists():
        existing = pd.read_csv(CANDIDATE_FILE)

        existing = existing[
            ~existing["candidate_id"].isin(
                reserve_df["candidate_id"]
            )
        ]

        output_df = pd.concat(
            [existing, reserve_df],
            ignore_index=True
        )

    else:
        output_df = reserve_df

    output_df.to_csv(
        CANDIDATE_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print("\n" + "=" * 70)
    print("SQL RESERVE SUMMARY")
    print("=" * 70)

    print(
        reserve_df[
            [
                "candidate_id",
                "transcript_available",
                "decision",
                "exclusion_reason",
            ]
        ].to_string(index=False)
    )

    print()
    print(
        "Include:",
        int((reserve_df["decision"] == "include").sum())
    )

    print(
        "Exclude:",
        int((reserve_df["decision"] == "exclude").sum())
    )

    print(
        "Pending:",
        int((reserve_df["decision"] == "pending").sum())
    )


if __name__ == "__main__":
    main()