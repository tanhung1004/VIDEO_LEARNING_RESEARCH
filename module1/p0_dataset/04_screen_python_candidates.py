from pathlib import Path
from datetime import date

import pandas as pd
from youtube_transcript_api import YouTubeTranscriptApi


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CANDIDATE_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "video_candidates.csv"
)

CATALOG_FILE = (
    PROJECT_ROOT
    / "data"
    / "concepts"
    / "concept_catalog.csv"
)


# =========================================================
# PYTHON CANDIDATE POOL
#
# IMPORTANT:
# - v2 (Programming with Mosh) is NOT included here.
# - Candidates were chosen before model evaluation.
# - Model predictions / F1 / OCR were NOT used.
#
# Goal:
# screen a pool larger than the required 9 new videos.
# =========================================================

CANDIDATES = [
    {
        "candidate_id": "c013",
        "youtube_id": "6iF8Xb7Z3wQ",
        "subject": "Python",
        "title": (
            "Python Tutorial for Beginners 7: "
            "Loops and Iterations - For/While Loops"
        ),
        "channel": "Corey Schafer",
        "duration_min": 10.23,
        "content_style": "code_tutorial",
        "search_query": (
            "Python loops iteration break continue tutorial"
        ),
    },
    {
        "candidate_id": "c014",
        "youtube_id": "xtXexPSfcZg",
        "subject": "Python",
        "title": "Python 3 Programming Tutorial - For loop",
        "channel": "sentdex",
        "duration_min": 9.18,
        "content_style": "code_tutorial",
        "search_query": "Python for loop iteration tutorial",
    },
    {
        "candidate_id": "c015",
        "youtube_id": "ohCDWZgNIU0",
        "subject": "Python",
        "title": "Python Lists",
        "channel": "Socratica",
        "duration_min": 5.73,
        "content_style": "concept_explanation",
        "search_query": "Python list tutorial",
    },
    {
        "candidate_id": "c016",
        "youtube_id": "dHANJ4l6fwA",
        "subject": "Python",
        "title": (
            "Python For Loops - Programming for Beginners"
        ),
        "channel": "Python Simplified",
        "duration_min": "",
        "content_style": "mixed",
        "search_query": (
            "Python for loop iteration range tutorial"
        ),
    },
    {
        "candidate_id": "c017",
        "youtube_id": "U_nugSKtbSk",
        "subject": "Python",
        "title": (
            "Python Programming Tutorial #7 - While Loops"
        ),
        "channel": "Tech With Tim",
        "duration_min": "",
        "content_style": "code_tutorial",
        "search_query": "Python while loop tutorial",
    },
    {
        "candidate_id": "c018",
        "youtube_id": "rRTjPnVooxE",
        "subject": "Python",
        "title": "While loops in Python are easy!",
        "channel": "Bro Code",
        "duration_min": 8.07,
        "content_style": "code_tutorial",
        "search_query": "Python while loop tutorial",
    },
    {
        "candidate_id": "c019",
        "youtube_id": "KWgYha0clzw",
        "subject": "Python",
        "title": "Learn Python for loops in 5 minutes!",
        "channel": "Bro Code",
        "duration_min": 5.78,
        "content_style": "code_tutorial",
        "search_query": (
            "Python for range break continue tutorial"
        ),
    },
    {
        "candidate_id": "c020",
        "youtube_id": "APWy6Pc83gE",
        "subject": "Python",
        "title": "Nested loops in Python are easy",
        "channel": "Bro Code",
        "duration_min": 6.20,
        "content_style": "code_tutorial",
        "search_query": "Python nested loops tutorial",
    },
    {
        "candidate_id": "c021",
        "youtube_id": "gOMW_n2-2Mw",
        "subject": "Python",
        "title": (
            "Python lists, sets, and tuples explained"
        ),
        "channel": "Bro Code",
        "duration_min": 13.07,
        "content_style": "code_tutorial",
        "search_query": "Python list iteration tutorial",
    },
    {
        "candidate_id": "c022",
        "youtube_id": "VL_g3LjsFqs",
        "subject": "Python",
        "title": "Learn Python iterables in 6 minutes!",
        "channel": "Bro Code",
        "duration_min": 6.80,
        "content_style": "concept_explanation",
        "search_query": "Python iteration iterable tutorial",
    },
    {
        "candidate_id": "c023",
        "youtube_id": "yCZBnjF4_tU",
        "subject": "Python",
        "title": (
            "Python Tutorial for Beginners - "
            "Break Continue Pass"
        ),
        "channel": "Telusko",
        "duration_min": "",
        "content_style": "concept_explanation",
        "search_query": (
            "Python break continue pass tutorial"
        ),
    },
    {
        "candidate_id": "c024",
        "youtube_id": "3ykIpmAxdoY",
        "subject": "Python",
        "title": (
            "For loop - Python 3 Programming Tutorials"
        ),
        "channel": "codebasics",
        "duration_min": 20.68,
        "content_style": "ide_demo",
        "search_query": (
            "Python for loop range while tutorial"
        ),
    },
    {
        "candidate_id": "c025",
        "youtube_id": "HVqcW9k9RPU",
        "subject": "Python",
        "title": "Python Tutorial: Nested Loops",
        "channel": "Professor Hank Stalica",
        "duration_min": "",
        "content_style": "mixed",
        "search_query": "Python nested loops tutorial",
    },
    {
        "candidate_id": "c026",
        "youtube_id": "Evycb99_XIc",
        "subject": "Python",
        "title": (
            "Range Function in Python Step-by-Step Tutorial"
        ),
        "channel": "Learning Vibes",
        "duration_min": "",
        "content_style": "concept_explanation",
        "search_query": "Python range function tutorial",
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


# =========================================================
# TRANSCRIPT CHECK
# =========================================================

def check_transcript(api, youtube_id):
    try:
        transcript = api.fetch(
            youtube_id,
            languages=["en"]
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

        # RequestBlocked / IPBlocked / network errors
        # are NOT evidence that the video is unsuitable.
        return {
            "available": "api_error",
            "decision": "pending",
            "reason": error_name,
            "rows": 0,
        }


# =========================================================
# CATALOG CHECK
# =========================================================

def print_python_catalog():
    catalog = pd.read_csv(CATALOG_FILE)

    python_catalog = catalog[
        catalog["subject"]
        .astype(str)
        .str.strip()
        .str.lower()
        == "python"
    ].copy()

    print("=" * 72)
    print("CURRENT PYTHON CONCEPT CATALOG")
    print("=" * 72)

    print(
        python_catalog[
            ["concept", "description"]
        ].to_string(index=False)
    )

    print()
    print(
        "Python concepts:",
        len(python_catalog)
    )

    if len(python_catalog) != 9:
        print(
            "WARNING: expected 9 Python concepts "
            "from the current pilot."
        )

    print()


# =========================================================
# MAIN
# =========================================================

def main():
    print_python_catalog()

    api = YouTubeTranscriptApi()

    checked_rows = []

    youtube_prefix = (
        "https:"
        + "//www.youtube.com/watch?v="
    )

    for candidate in CANDIDATES:
        result = check_transcript(
            api,
            candidate["youtube_id"]
        )

        print("=" * 72)
        print(
            candidate["candidate_id"],
            candidate["title"]
        )
        print(
            "Channel:",
            candidate["channel"]
        )
        print(
            "Transcript:",
            result["available"]
        )
        print(
            "Transcript rows:",
            result["rows"]
        )
        print(
            "Decision:",
            result["decision"]
        )

        if result["reason"]:
            print(
                "Reason:",
                result["reason"]
            )

        checked_rows.append({
            "candidate_id":
                candidate["candidate_id"],

            "subject":
                candidate["subject"],

            "title":
                candidate["title"],

            "url":
                youtube_prefix
                + candidate["youtube_id"],

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

    python_df = pd.DataFrame(
        checked_rows,
        columns=COLUMNS
    )

    # =====================================================
    # PRESERVE PREVIOUS SQL CANDIDATES
    # =====================================================

    if CANDIDATE_FILE.exists():
        existing = pd.read_csv(
            CANDIDATE_FILE
        )

        if not existing.empty:
            current_ids = set(
                python_df["candidate_id"]
            )

            existing = existing[
                ~existing["candidate_id"]
                .isin(current_ids)
            ].copy()

            output_df = pd.concat(
                [existing, python_df],
                ignore_index=True
            )

        else:
            output_df = python_df

    else:
        output_df = python_df

    output_df.to_csv(
        CANDIDATE_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    # =====================================================
    # SUMMARY
    # =====================================================

    print()
    print("=" * 72)
    print("PYTHON SCREENING SUMMARY")
    print("=" * 72)

    print(
        python_df[
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
        len(python_df)
    )

    print(
        "Include:",
        int(
            (
                python_df["decision"]
                == "include"
            ).sum()
        )
    )

    print(
        "Exclude:",
        int(
            (
                python_df["decision"]
                == "exclude"
            ).sum()
        )
    )

    print(
        "Pending:",
        int(
            (
                python_df["decision"]
                == "pending"
            ).sum()
        )
    )

    print()
    print("Eligible channels:")

    eligible = python_df[
        python_df["decision"] == "include"
    ]

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
        CANDIDATE_FILE
    )


if __name__ == "__main__":
    main()