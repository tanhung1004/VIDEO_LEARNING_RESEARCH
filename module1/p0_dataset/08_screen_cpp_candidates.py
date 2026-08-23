from pathlib import Path
from datetime import date

import pandas as pd
from youtube_transcript_api import YouTubeTranscriptApi


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CANDIDATE_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "video_candidates.csv"
)


# =========================================================
# C++ CANDIDATE POOL
#
# IMPORTANT:
# - URLs are frozen at candidate-screening time.
# - Do NOT replace/search again by title.
# - Do NOT use LDA / LSA / OCR / F1 for selection.
#
# 16 candidates > 10 videos required.
# =========================================================

CANDIDATES = [
    {
        "candidate_id": "c039",
        "youtube_id": "zB9RI8_wExo",
        "subject": "C++",
        "title": "Variables in C++",
        "channel": "The Cherno",
        "duration_min": 13.77,
        "content_style": "concept_explanation",
        "search_query": "C++ variables data types tutorial",
    },
    {
        "candidate_id": "c040",
        "youtube_id": "solufpKPDwY",
        "subject": "C++",
        "title": (
            "C++ FOR BEGINNERS - Variables, "
            "Data Types, Overflow, Sizeof"
        ),
        "channel": "CodeBeauty",
        "duration_min": 32.73,
        "content_style": "code_tutorial",
        "search_query": "C++ variables data types tutorial",
    },
    {
        "candidate_id": "c041",
        "youtube_id": "nrNdn6lSQwE",
        "subject": "C++",
        "title": (
            "C++ Programming: Variables and "
            "Data Types Explained"
        ),
        "channel": "Code Trainers",
        "duration_min": "",
        "content_style": "concept_explanation",
        "search_query": "C++ variables data types beginner",
    },
    {
        "candidate_id": "c042",
        "youtube_id": "eNofmKYzje4",
        "subject": "C++",
        "title": (
            "C++ POINTERS - Introduction to "
            "C++ pointers for beginners"
        ),
        "channel": "CodeBeauty",
        "duration_min": "",
        "content_style": "code_tutorial",
        "search_query": "C++ pointer memory address tutorial",
    },
    {
        "candidate_id": "c043",
        "youtube_id": "IzoFn3dfsPA",
        "subject": "C++",
        "title": "REFERENCES in C++",
        "channel": "The Cherno",
        "duration_min": 10.22,
        "content_style": "concept_explanation",
        "search_query": "C++ reference tutorial",
    },
    {
        "candidate_id": "c044",
        "youtube_id": "LTXFr_RJE6M",
        "subject": "C++",
        "title": (
            "C++ Reference Variable | "
            "C++ Pointers vs References Explained"
        ),
        "channel": "Programming Horizons",
        "duration_min": "",
        "content_style": "code_tutorial",
        "search_query": "C++ pointer reference tutorial",
    },
    {
        "candidate_id": "c045",
        "youtube_id": "iVLQeWbgbXs",
        "subject": "C++",
        "title": (
            "C++ OOP - Introduction to "
            "classes and objects for beginners"
        ),
        "channel": "CodeBeauty",
        "duration_min": "",
        "content_style": "concept_explanation",
        "search_query": "C++ class object tutorial",
    },
    {
        "candidate_id": "c046",
        "youtube_id": "Ks97R1knQDY",
        "subject": "C++",
        "title": (
            "How to CREATE/INSTANTIATE "
            "OBJECTS in C++"
        ),
        "channel": "The Cherno",
        "duration_min": 13.05,
        "content_style": "code_tutorial",
        "search_query": "C++ class object instantiate tutorial",
    },
    {
        "candidate_id": "c047",
        "youtube_id": "5z3KEX9AZEQ",
        "subject": "C++",
        "title": "C++ CONSTRUCTORS explained easy",
        "channel": "Bro Code",
        "duration_min": 8.92,
        "content_style": "code_tutorial",
        "search_query": "C++ constructor tutorial",
    },
    {
        "candidate_id": "c048",
        "youtube_id": "X8nYM8wdNRE",
        "subject": "C++",
        "title": "Inheritance in C++",
        "channel": "The Cherno",
        "duration_min": 8.0,
        "content_style": "concept_explanation",
        "search_query": "C++ inheritance tutorial",
    },
    {
        "candidate_id": "c049",
        "youtube_id": "zsqhrxlp-Fo",
        "subject": "C++",
        "title": (
            "Inheritance, Poly Morphism | "
            "Introduction | CPP OOPS"
        ),
        "channel": "LearningLad",
        "duration_min": "",
        "content_style": "code_tutorial",
        "search_query": "C++ inheritance polymorphism tutorial",
    },
    {
        "candidate_id": "c050",
        "youtube_id": "oIV2KchSyGQ",
        "subject": "C++",
        "title": "Virtual Functions in C++",
        "channel": "The Cherno",
        "duration_min": "",
        "content_style": "concept_explanation",
        "search_query": "C++ polymorphism virtual functions tutorial",
    },
    {
        "candidate_id": "c051",
        "youtube_id": "I-hZkUa9mIs",
        "subject": "C++",
        "title": "Templates in C++",
        "channel": "The Cherno",
        "duration_min": 17.98,
        "content_style": "concept_explanation",
        "search_query": "C++ function class template tutorial",
    },
    {
        "candidate_id": "c052",
        "youtube_id": "GV0tmh_HYaE",
        "subject": "C++",
        "title": (
            "Mastering C++ Templates | "
            "Full Tutorial with Examples"
        ),
        "channel": "Jeevan Pant",
        "duration_min": 43.0,
        "content_style": "code_tutorial",
        "search_query": "C++ templates tutorial examples",
    },
    {
        "candidate_id": "c053",
        "youtube_id": "iNlmsLrzGD4",
        "subject": "C++",
        "title": "C++ Pointers and References FULL Guide",
        "channel": "Velcode",
        "duration_min": 49.1,
        "content_style": "mixed",
        "search_query": "C++ pointers references guide",
    },
    {
        "candidate_id": "c054",
        "youtube_id": "wN0x9eZLix4",
        "subject": "C++",
        "title": (
            "Object Oriented Programming "
            "(OOP) in C++ Course"
        ),
        "channel": "freeCodeCamp.org / CodeBeauty",
        "duration_min": 89.0,
        "content_style": "mixed",
        "search_query": (
            "C++ classes objects constructors "
            "inheritance polymorphism"
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

        # Network / IP / API errors are NOT evidence
        # that the video itself is unsuitable.
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
    print("C++ CANDIDATE SCREENING")
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

    cpp_df = pd.DataFrame(
        checked_rows,
        columns=COLUMNS,
    )

    # =====================================================
    # Preserve SQL / Python / Java records.
    # Re-running this script replaces only c039-c054.
    # =====================================================

    if CANDIDATE_FILE.exists():
        existing = pd.read_csv(
            CANDIDATE_FILE
        )

        current_ids = set(
            cpp_df["candidate_id"]
        )

        existing = existing[
            ~existing["candidate_id"]
            .isin(current_ids)
        ].copy()

        output_df = pd.concat(
            [existing, cpp_df],
            ignore_index=True,
        )

    else:
        output_df = cpp_df

    # =====================================================
    # Global duplicate checks
    # =====================================================

    if output_df[
        "candidate_id"
    ].duplicated().any():
        raise ValueError(
            "Duplicate candidate_id after "
            "C++ screening."
        )

    duplicate_urls = output_df[
        output_df["url"].duplicated(
            keep=False
        )
    ]

    if not duplicate_urls.empty:
        print()
        print("DUPLICATE URLS FOUND:")

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
            "Duplicate URL in candidate registry."
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
    print("C++ SCREENING SUMMARY")
    print("=" * 72)

    print(
        cpp_df[
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
        len(cpp_df),
    )

    print(
        "Include:",
        int(
            (
                cpp_df["decision"]
                == "include"
            ).sum()
        ),
    )

    print(
        "Exclude:",
        int(
            (
                cpp_df["decision"]
                == "exclude"
            ).sum()
        ),
    )

    print(
        "Pending:",
        int(
            (
                cpp_df["decision"]
                == "pending"
            ).sum()
        ),
    )

    eligible = cpp_df[
        cpp_df["decision"] == "include"
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