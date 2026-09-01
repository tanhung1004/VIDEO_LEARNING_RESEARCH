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
        "candidate_id": "c001",
        "youtube_id": "OlT3FispsMU",
        "subject": "SQL",
        "title": "Learn Basic SQL Commands",
        "channel": "1Keydata",
        "duration_min": 7.08,
        "content_style": "concept_explanation",
        "search_query": "SQL SELECT WHERE GROUP BY HAVING tutorial",
    },
    {
        "candidate_id": "c002",
        "youtube_id": "YufocuHbYZo",
        "subject": "SQL",
        "title": "SQL SELECT Tutorial",
        "channel": "Socratica",
        "duration_min": 8.93,
        "content_style": "concept_explanation",
        "search_query": "SQL SELECT WHERE ORDER BY tutorial",
    },
    {
        "candidate_id": "c003",
        "youtube_id": "kbKty5ZVKMY",
        "subject": "SQL",
        "title": "Learn Basic SQL in 15 Minutes",
        "channel": "Adam Finer - Learn BI",
        "duration_min": 17.67,
        "content_style": "ide_demo",
        "search_query": "SQL beginner SELECT WHERE GROUP BY tutorial",
    },
    {
        "candidate_id": "c004",
        "youtube_id": "RGIVS8RGBaI",
        "subject": "SQL",
        "title": "Aggregate Functions in SQL",
        "channel": "Neso Academy",
        "duration_min": 12.0,
        "content_style": "slide_lecture",
        "search_query": "SQL aggregate functions COUNT SUM AVG tutorial",
    },
    {
        "candidate_id": "c005",
        "youtube_id": "wsbp1J-AFhY",
        "subject": "SQL",
        "title": "GROUP BY and HAVING Clause in SELECT Command",
        "channel": "Sundeep Saradhi Kanthety",
        "duration_min": "",
        "content_style": "code_tutorial",
        "search_query": "SQL GROUP BY HAVING tutorial",
    },
    {
        "candidate_id": "c006",
        "youtube_id": "Vj6RqA_X-IE",
        "subject": "SQL",
        "title": "Subqueries in MySQL",
        "channel": "Alex The Analyst",
        "duration_min": 11.0,
        "content_style": "code_tutorial",
        "search_query": "SQL subquery tutorial",
    },
    {
        "candidate_id": "c007",
        "youtube_id": "lXQzD09BOH0",
        "subject": "SQL",
        "title": "Joins in MySQL",
        "channel": "Alex The Analyst",
        "duration_min": 17.0,
        "content_style": "code_tutorial",
        "search_query": "SQL JOIN INNER LEFT tutorial",
    },
    {
        "candidate_id": "c008",
        "youtube_id": "_vxobA36UN4",
        "subject": "SQL",
        "title": "Beginner SQL - Analyzing Student Records",
        "channel": "Maven Analytics",
        "duration_min": 15.0,
        "content_style": "ide_demo",
        "search_query": "SQL beginner SELECT WHERE GROUP BY JOIN tutorial",
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

        # Network/IP/API problems are NOT evidence
        # that the candidate is unsuitable.
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
            "url":
                f"https://www.youtube.com/watch?v={item['youtube_id']}",
            "channel": item["channel"],
            "duration_min": item["duration_min"],
            "content_style": item["content_style"],
            "transcript_available": transcript_status,
            "search_query": item["search_query"],
            "retrieval_date": date.today().isoformat(),
            "decision": decision,
            "exclusion_reason": reason,
        })

    new_df = pd.DataFrame(rows, columns=COLUMNS)

    if CANDIDATE_FILE.exists():
        old_df = pd.read_csv(CANDIDATE_FILE)

        if not old_df.empty:
            old_df = old_df[
                ~old_df["candidate_id"].isin(
                    new_df["candidate_id"]
                )
            ]

            new_df = pd.concat(
                [old_df, new_df],
                ignore_index=True
            )

    new_df.to_csv(
        CANDIDATE_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    current = new_df[
        new_df["candidate_id"].isin(
            [x["candidate_id"] for x in CANDIDATES]
        )
    ]

    print("\n" + "=" * 70)
    print("SQL SCREENING SUMMARY")
    print("=" * 70)

    print(
        current[
            [
                "candidate_id",
                "transcript_available",
                "decision",
                "exclusion_reason",
            ]
        ].to_string(index=False)
    )

    print("\nInclude:",
          (current["decision"] == "include").sum())

    print("Exclude:",
          (current["decision"] == "exclude").sum())

    print("Pending:",
          (current["decision"] == "pending").sum())

    print("\nSaved:", CANDIDATE_FILE)


if __name__ == "__main__":
    main()