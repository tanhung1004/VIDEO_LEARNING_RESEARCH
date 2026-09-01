from pathlib import Path
import re

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

VIDEOS_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "videos.csv"
)

CATALOG_FILE = (
    PROJECT_ROOT
    / "data"
    / "concepts"
    / "concept_catalog.csv"
)

TRANSCRIPT_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "transcripts"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "ground_truth"
    / "annotation_working"
    / "review_assisted"
)

EVIDENCE_DIR = (
    OUTPUT_DIR
    / "evidence"
)

EVIDENCE_INDEX_FILE = (
    OUTPUT_DIR
    / "evidence_index.csv"
)


# =========================================================
# Search aliases
#
# IMPORTANT:
# These aliases ONLY locate possible evidence.
# They NEVER decide ground truth.
# =========================================================

ALIASES = {
    "SQL": {
        "select": [
            "select",
            "select statement",
            "select query",
        ],
        "where": [
            "where",
            "where clause",
        ],
        "group by": [
            "group by",
        ],
        "having": [
            "having",
            "having clause",
        ],
        "count": [
            "count",
            "count function",
        ],
        "aggregate function": [
            "aggregate function",
            "aggregate functions",
            "aggregation",
            "sum",
            "average",
            "avg",
            "minimum",
            "maximum",
        ],
        "join": [
            "join",
            "joins",
            "joining",
        ],
        "inner join": [
            "inner join",
        ],
        "left join": [
            "left join",
            "left outer join",
        ],
        "subquery": [
            "subquery",
            "subqueries",
            "nested query",
        ],
        "order by": [
            "order by",
        ],
    },

    "Python": {
        "for loop": [
            "for loop",
            "for loops",
        ],
        "for": [
            "for statement",
            "for loop",
            "for in",
        ],
        "iteration": [
            "iteration",
            "iterations",
            "iterate",
            "iterating",
        ],
        "range": [
            "range",
            "range function",
        ],
        "list": [
            "list",
            "lists",
        ],
        "nested loop": [
            "nested loop",
            "nested loops",
        ],
        "index": [
            "index",
            "indexing",
            "indices",
        ],
        "break": [
            "break",
            "break statement",
        ],
        "continue": [
            "continue",
            "continue statement",
        ],
    },

    "Java": {
        "class": [
            "class",
            "classes",
        ],
        "object": [
            "object",
            "objects",
            "instance",
        ],
        "method": [
            "method",
            "methods",
        ],
        "interface": [
            "interface",
            "interfaces",
        ],
        "implements": [
            "implements",
            "implement interface",
            "implementing interface",
        ],
        "extends": [
            "extends",
            "extend class",
        ],
        "inheritance": [
            "inheritance",
            "inherit",
            "inherited",
        ],
        "abstract class": [
            "abstract class",
            "abstract classes",
        ],
        "abstract method": [
            "abstract method",
            "abstract methods",
        ],
    },

    "C++": {
        "variable": [
            "variable",
            "variables",
        ],
        "data type": [
            "data type",
            "data types",
            "datatype",
        ],
        "pointer": [
            "pointer",
            "pointers",
            "memory address",
        ],
        "reference": [
            "reference",
            "references",
        ],
        "class": [
            "class",
            "classes",
        ],
        "object": [
            "object",
            "objects",
            "instance",
            "instantiate",
        ],
        "constructor": [
            "constructor",
            "constructors",
        ],
        "inheritance": [
            "inheritance",
            "inherit",
            "base class",
            "derived class",
        ],
        "polymorphism": [
            "polymorphism",
            "polymorphic",
            "virtual function",
            "virtual functions",
        ],
        "template": [
            "template",
            "templates",
            "generic",
        ],
    },
}


SUBJECT_FILE_NAMES = {
    "SQL": "sql",
    "Python": "python",
    "Java": "java",
    "C++": "cpp",
}


def video_number(video_id):
    return int(
        str(video_id)
        .strip()
        .replace("v", "")
    )


def normalize_text(value):
    text = str(value).lower()

    text = text.replace(
        "\n",
        " "
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def alias_matches(text, alias):
    text = normalize_text(text)
    alias = normalize_text(alias)

    pattern = re.escape(alias)

    pattern = pattern.replace(
        r"\ ",
        r"\s+",
    )

    # Word-like boundary to reduce accidental substring hits.
    pattern = (
        r"(?<!\w)"
        + pattern
        + r"(?!\w)"
    )

    return (
        re.search(
            pattern,
            text,
            flags=re.IGNORECASE,
        )
        is not None
    )


def find_evidence(
    transcript,
    aliases,
    max_hits=3,
):
    hits = []

    for idx, row in transcript.iterrows():

        text = str(
            row.get(
                "text",
                "",
            )
        )

        matched_aliases = [
            alias
            for alias in aliases
            if alias_matches(
                text,
                alias,
            )
        ]

        if not matched_aliases:
            continue

        # Add one transcript row before + after
        # so the human sees context.
        start_idx = max(
            0,
            idx - 1,
        )

        end_idx = min(
            len(transcript) - 1,
            idx + 1,
        )

        context = transcript.iloc[
            start_idx:
            end_idx + 1
        ]

        snippet = " ".join(
            str(x)
            .replace("\n", " ")
            .strip()
            for x in context["text"]
        )

        start_sec = float(
            context.iloc[0]
            .get(
                "start_sec",
                0,
            )
        )

        end_sec = float(
            context.iloc[-1]
            .get(
                "end_sec",
                start_sec,
            )
        )

        start_time = str(
            context.iloc[0]
            .get(
                "start_time",
                "",
            )
        )

        end_time = str(
            context.iloc[-1]
            .get(
                "end_time",
                "",
            )
        )

        # Avoid duplicate overlapping hits.
        duplicate = any(
            abs(
                previous[
                    "start_sec"
                ]
                - start_sec
            )
            < 2
            for previous in hits
        )

        if duplicate:
            continue

        hits.append({
            "start_sec":
                round(
                    start_sec,
                    2,
                ),

            "end_sec":
                round(
                    end_sec,
                    2,
                ),

            "start_time":
                start_time,

            "end_time":
                end_time,

            "matched_aliases":
                ", ".join(
                    matched_aliases
                ),

            "snippet":
                snippet,
        })

        if len(hits) >= max_hits:
            break

    return hits


def build_matrix(
    subject_videos,
    subject_catalog,
    output_file,
):
    concepts = list(
        subject_catalog["concept"]
    )

    rows = []

    for _, video in (
        subject_videos.iterrows()
    ):
        row = {
            "video_id":
                video["video_id"],

            "title":
                video["title"],

            "url":
                video["url"],
        }

        # Human fills 1 or 0.
        for concept in concepts:
            row[concept] = ""

        row["annotator"] = ""
        row["review_status"] = "pending"

        rows.append(row)

    matrix = pd.DataFrame(rows)

    # Protect human work if script is re-run.
    if output_file.exists():

        print(
            "SKIP existing matrix:",
            output_file,
        )

    else:
        matrix.to_csv(
            output_file,
            index=False,
            encoding="utf-8-sig",
        )

        print(
            "Created matrix:",
            output_file,
        )


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    EVIDENCE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    videos = pd.read_csv(
        VIDEOS_FILE
    )

    catalog = pd.read_csv(
        CATALOG_FILE
    )

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
            videos[
                "annotation_status"
            ]
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

    pending = pending.sort_values(
        "_num"
    )

    print("=" * 72)
    print(
        "P0.6 - PREPARE ASSISTED HUMAN REVIEW"
    )
    print("=" * 72)

    print(
        "Pending videos:",
        len(pending),
    )

    if len(pending) != 35:
        raise ValueError(
            f"Expected 35 pending videos, "
            f"got {len(pending)}."
        )

    evidence_index = []

    total_pairs = 0

    for subject in [
        "SQL",
        "Python",
        "Java",
        "C++",
    ]:

        subject_videos = pending[
            pending["subject"]
            == subject
        ].copy()

        subject_catalog = catalog[
            catalog["subject"]
            .astype(str)
            .str.strip()
            == subject
        ].copy()

        slug = (
            SUBJECT_FILE_NAMES[
                subject
            ]
        )

        matrix_file = (
            OUTPUT_DIR
            / f"{slug}_annotation_matrix.csv"
        )

        build_matrix(
            subject_videos,
            subject_catalog,
            matrix_file,
        )

        total_pairs += (
            len(subject_videos)
            * len(subject_catalog)
        )

        for _, video in (
            subject_videos.iterrows()
        ):

            video_id = (
                video["video_id"]
            )

            transcript_file = (
                TRANSCRIPT_DIR
                / f"{video_id}_transcript.csv"
            )

            if not transcript_file.exists():
                raise FileNotFoundError(
                    transcript_file
                )

            transcript = pd.read_csv(
                transcript_file
            )

            evidence_file = (
                EVIDENCE_DIR
                / f"{video_id}_evidence.txt"
            )

            with open(
                evidence_file,
                "w",
                encoding="utf-8",
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

                f.write(
                    "\nIMPORTANT:\n"
                )

                f.write(
                    "Keyword hits below are ONLY navigation aids.\n"
                )

                f.write(
                    "They are NOT automatic labels.\n"
                )

                f.write(
                    "Human reviewer must decide 1/0.\n"
                )

                f.write(
                    "\n"
                    + "=" * 72
                    + "\n"
                )

                for _, concept_row in (
                    subject_catalog
                    .iterrows()
                ):

                    concept = str(
                        concept_row[
                            "concept"
                        ]
                    ).strip()

                    description = str(
                        concept_row[
                            "description"
                        ]
                    ).strip()

                    aliases = (
                        ALIASES
                        .get(
                            subject,
                            {},
                        )
                        .get(
                            concept,
                            [concept],
                        )
                    )

                    hits = find_evidence(
                        transcript,
                        aliases,
                    )

                    f.write(
                        f"\nCONCEPT: {concept}\n"
                    )

                    f.write(
                        f"DESCRIPTION: "
                        f"{description}\n"
                    )

                    f.write(
                        "SEARCH ALIASES: "
                        + ", ".join(
                            aliases
                        )
                        + "\n"
                    )

                    f.write(
                        f"DIRECT HIT COUNT "
                        f"(top max 3 shown): "
                        f"{len(hits)}\n"
                    )

                    if hits:

                        for number, hit in (
                            enumerate(
                                hits,
                                start=1,
                            )
                        ):

                            f.write(
                                f"\n  HIT {number}\n"
                            )

                            f.write(
                                "  TIME: "
                                f"{hit['start_time']} "
                                f"- "
                                f"{hit['end_time']}\n"
                            )

                            f.write(
                                "  SECONDS: "
                                f"{hit['start_sec']} "
                                f"- "
                                f"{hit['end_sec']}\n"
                            )

                            f.write(
                                "  MATCH: "
                                f"{hit['matched_aliases']}\n"
                            )

                            f.write(
                                "  TEXT: "
                                f"{hit['snippet']}\n"
                            )

                    else:
                        f.write(
                            "\n  NO DIRECT KEYWORD HIT.\n"
                        )

                        f.write(
                            "  This does NOT mean negative.\n"
                        )

                        f.write(
                            "  Review transcript/video "
                            "if concept may be expressed "
                            "without exact terminology.\n"
                        )

                    first_hit = (
                        hits[0]
                        if hits
                        else {}
                    )

                    evidence_index.append({
                        "video_id":
                            video_id,

                        "subject":
                            subject,

                        "concept":
                            concept,

                        "hit_count_shown":
                            len(hits),

                        "first_hit_start_sec":
                            first_hit.get(
                                "start_sec",
                                "",
                            ),

                        "first_hit_end_sec":
                            first_hit.get(
                                "end_sec",
                                "",
                            ),

                        "first_hit_snippet":
                            first_hit.get(
                                "snippet",
                                "",
                            ),

                        "evidence_file":
                            str(
                                evidence_file
                            ),
                    })

                    f.write(
                        "\n"
                        + "-" * 72
                        + "\n"
                    )

    evidence_df = pd.DataFrame(
        evidence_index
    )

    evidence_df.to_csv(
        EVIDENCE_INDEX_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print("=" * 72)
    print("ASSISTED REVIEW SUMMARY")
    print("=" * 72)

    print(
        "Videos:",
        evidence_df[
            "video_id"
        ].nunique(),
    )

    print(
        "Concept pairs:",
        len(evidence_df),
    )

    print(
        "Expected pairs:",
        total_pairs,
    )

    print()
    print(
        evidence_df
        .groupby("subject")
        .size()
    )

    print()
    print(
        "Evidence directory:",
        EVIDENCE_DIR,
    )

    print(
        "Evidence index:",
        EVIDENCE_INDEX_FILE,
    )

    if len(evidence_df) != 341:
        raise ValueError(
            "Expected 341 concept pairs."
        )

    print()
    print("=" * 72)
    print(
        "P0.6 ASSISTED REVIEW PREPARATION: PASS"
    )
    print("=" * 72)


if __name__ == "__main__":
    main()