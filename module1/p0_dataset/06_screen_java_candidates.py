from pathlib import Path
from datetime import date

import pandas as pd
from youtube_transcript_api import YouTubeTranscriptApi


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CANDIDATE_FILE = (
    PROJECT_ROOT / "data" / "raw" / "video_candidates.csv"
)


# =========================================================
# JAVA CANDIDATE POOL
#
# Frozen candidate URLs.
# Do NOT replace/search again by title.
#
# Goal:
# 12 candidates > 8 new Java videos required.
# =========================================================

CANDIDATES = [
    {
        "candidate_id": "c027",
        "youtube_id": "Pt-gPG9LOV8",
        "subject": "Java",
        "title": "Java Classes and Objects (Complete Tutorial)",
        "channel": "Bill Barnum",
        "duration_min": 31.2,
        "content_style": "code_tutorial",
        "search_query": "Java class object method tutorial",
    },
    {
        "candidate_id": "c028",
        "youtube_id": "RlHXptWV_PI",
        "subject": "Java",
        "title": (
            "Java Classes and Objects Tutorial | "
            "How to Create and correctly use Classes"
        ),
        "channel": "Java Coding Community",
        "duration_min": "",
        "content_style": "concept_explanation",
        "search_query": "Java class object tutorial",
    },
    {
        "candidate_id": "c029",
        "youtube_id": "v5p_SUfi710",
        "subject": "Java",
        "title": "Java methods explained in 10+ minutes!",
        "channel": "Bro Code",
        "duration_min": "",
        "content_style": "code_tutorial",
        "search_query": "Java method tutorial",
    },
    {
        "candidate_id": "c030",
        "youtube_id": "kTpp5n_CppQ",
        "subject": "Java",
        "title": "Java Interface Tutorial #78",
        "channel": "Alex Lee",
        "duration_min": "",
        "content_style": "concept_explanation",
        "search_query": "Java interface implements tutorial",
    },
    {
        "candidate_id": "c031",
        "youtube_id": "KjEkdELJua4",
        "subject": "Java",
        "title": (
            "Java Interface & Implements Keyword | "
            "Java Tutorial for Beginners"
        ),
        "channel": "The Caffeinated Programmer",
        "duration_min": 5.2,
        "content_style": "code_tutorial",
        "search_query": "Java interface implements tutorial",
    },
    {
        "candidate_id": "c032",
        "youtube_id": "GhslBwrRsnw",
        "subject": "Java",
        "title": "Java interface",
        "channel": "Bro Code",
        "duration_min": "",
        "content_style": "concept_explanation",
        "search_query": "Java interface tutorial",
    },
    {
        "candidate_id": "c033",
        "youtube_id": "GTP5lVEKXaU",
        "subject": "Java",
        "title": "Learn Java inheritance in 9 minutes!",
        "channel": "Bro Code",
        "duration_min": 9.0,
        "content_style": "code_tutorial",
        "search_query": "Java inheritance extends tutorial",
    },
    {
        "candidate_id": "c034",
        "youtube_id": "_9R_in_7cT4",
        "subject": "Java",
        "title": "#6.1 Java Tutorial | Inheritance",
        "channel": "Telusko",
        "duration_min": "",
        "content_style": "concept_explanation",
        "search_query": "Java inheritance extends tutorial",
    },
    {
        "candidate_id": "c035",
        "youtube_id": "lz1Cx6GzOuc",
        "subject": "Java",
        "title": (
            "Java Tutorial For Beginners 25 - "
            "Inheritance in Java"
        ),
        "channel": "ProgrammingKnowledge",
        "duration_min": "",
        "content_style": "code_tutorial",
        "search_query": "Java inheritance extends tutorial",
    },
    {
        "candidate_id": "c036",
        "youtube_id": "52frlN8webg",
        "subject": "Java",
        "title": "Abstract Class In Java Tutorial #79",
        "channel": "Alex Lee",
        "duration_min": "",
        "content_style": "concept_explanation",
        "search_query": (
            "Java abstract class abstract method tutorial"
        ),
    },
    {
        "candidate_id": "c037",
        "youtube_id": "4B8XKEORJss",
        "subject": "Java",
        "title": "Learn Java abstraction in 9 minutes!",
        "channel": "Bro Code",
        "duration_min": 9.0,
        "content_style": "code_tutorial",
        "search_query": (
            "Java abstract class abstract method tutorial"
        ),
    },
    {
        "candidate_id": "c038",
        "youtube_id": "ZRdgrLs-rRo",
        "subject": "Java",
        "title": (
            "When and How to Utilize Java Abstract "
            "Classes and Abstract Methods?"
        ),
        "channel": "Ram N Java",
        "duration_min": "",
        "content_style": "concept_explanation",
        "search_query": (
            "Java abstract class abstract method tutorial"
        ),
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
            languages=["en"],
        )

        raw_data = transcript.to_raw_data()

        if not raw_data:
            return {
                "available": "no",
                "decision": "exclude",
                "reason": "empty_english_transcript",
                "rows": 0,
            }

        return {
            "available": "yes",
            "decision": "include",
            "reason": "",
            "rows": len(raw_data),
        }

    except Exception as exc:
        error_name = type(exc).__name__

        hard_failures = {
            "TranscriptsDisabled",
            "NoTranscriptFound",
            "VideoUnavailable",
        }

        if error_name in hard_failures:
            return {
                "available": "no",
                "decision": "exclude",
                "reason": error_name,
                "rows": 0,
            }

        # Network/IP/API errors are not evidence
        # that the educational video is unsuitable.
        return {
            "available": "api_error",
            "decision": "pending",
            "reason": error_name,
            "rows": 0,
        }


def main():
    api = YouTubeTranscriptApi()

    checked_rows = []

    print("=" * 72)
    print("JAVA CANDIDATE SCREENING")
    print("=" * 72)

    for candidate in CANDIDATES:
        result = check_transcript(
            api,
            candidate["youtube_id"],
        )

        url = (
            "https://www.youtube.com/watch?v="
            + candidate["youtube_id"]
        )

        print()
        print(
            candidate["candidate_id"],
            candidate["title"],
        )

        print(
            "Channel:",
            candidate["channel"],
        )

        print(
            "URL:",
            url,
        )

        print(
            "Transcript:",
            result["available"],
        )

        print(
            "Transcript rows:",
            result["rows"],
        )

        print(
            "Decision:",
            result["decision"],
        )

        if result["reason"]:
            print(
                "Reason:",
                result["reason"],
            )

        checked_rows.append({
            "candidate_id":
                candidate["candidate_id"],

            "subject":
                candidate["subject"],

            "title":
                candidate["title"],

            "url":
                url,

            "channel":
                candidate["channel"],

            "duration_min":
                candidate["duration_min"],

            "content_style":
                candidate["content_style"],

            "transcript_available":
                result["available"],

            "search_query":
                candidate["search_query"],

            "retrieval_date":
                date.today().isoformat(),

            "decision":
                result["decision"],

            "exclusion_reason":
                result["reason"],
        })

    java_df = pd.DataFrame(
        checked_rows,
        columns=COLUMNS,
    )

    # =====================================================
    # Preserve all previous SQL/Python candidate records.
    #
    # Re-running this script only replaces c027-c038.
    # =====================================================

    if CANDIDATE_FILE.exists():
        existing = pd.read_csv(
            CANDIDATE_FILE
        )

        current_ids = set(
            java_df["candidate_id"]
        )

        existing = existing[
            ~existing["candidate_id"].isin(
                current_ids
            )
        ].copy()

        output_df = pd.concat(
            [existing, java_df],
            ignore_index=True,
        )

    else:
        output_df = java_df

    # -----------------------------------------------------
    # Global duplicate checks
    # -----------------------------------------------------

    if output_df["candidate_id"].duplicated().any():
        raise ValueError(
            "Duplicate candidate_id after Java screening."
        )

    duplicate_urls = output_df[
        output_df["url"].duplicated(
            keep=False
        )
    ]

    if not duplicate_urls.empty:
        print()
        print("WARNING: duplicate candidate URLs:")
        print(
            duplicate_urls[
                [
                    "candidate_id",
                    "subject",
                    "url",
                ]
            ].to_string(index=False)
        )

        raise ValueError(
            "Duplicate URL found in candidate registry."
        )

    output_df.to_csv(
        CANDIDATE_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    # =====================================================
    # Summary
    # =====================================================

    print()
    print("=" * 72)
    print("JAVA SCREENING SUMMARY")
    print("=" * 72)

    print(
        java_df[
            [
                "candidate_id",
                "channel",
                "transcript_available",
                "decision",
                "exclusion_reason",
            ]
        ].to_string(index=False)
    )

    print()
    print(
        "Candidates:",
        len(java_df),
    )

    print(
        "Include:",
        int(
            (
                java_df["decision"]
                == "include"
            ).sum()
        ),
    )

    print(
        "Exclude:",
        int(
            (
                java_df["decision"]
                == "exclude"
            ).sum()
        ),
    )

    print(
        "Pending:",
        int(
            (
                java_df["decision"]
                == "pending"
            ).sum()
        ),
    )

    eligible = java_df[
        java_df["decision"] == "include"
    ]

    print()
    print("Eligible channels:")

    print(
        eligible["channel"]
        .value_counts()
        .to_string()
    )

    print()
    print("Eligible styles:")

    print(
        eligible["content_style"]
        .value_counts()
        .to_string()
    )

    print()
    print(
        "Saved:",
        CANDIDATE_FILE,
    )


if __name__ == "__main__":
    main()