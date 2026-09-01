from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# STEP 12 - BUILD OLD vs NEW COMBINED SCORES
#
# FIX QUAN TRỌNG:
# KHÔNG rebuild transcript chunks từ transcript raw.
#
# Dùng chính xác frozen Scaffold40:
#   OLD = 494 chunks / 60s
#   NEW = 172 chunks / 180s
#
# OLD:
#   frozen OLD transcript + OCR D
#   -> LDA .40 + LSA .60
#
# NEW:
#   frozen NEW transcript + OCR D -> LSA-only score
#   frozen NEW transcript + OCR E -> LSA-only score
#   -> F = mean(score D, score E)
#
# STEP 12 KHÔNG dùng GT.
# STEP 12 KHÔNG tính F1.
# STEP 12 KHÔNG tune threshold.
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SRC_DIR = PROJECT_ROOT / "module1" / "src"
sys.path.insert(0, str(SRC_DIR))

from preprocessing import preprocess_text
from lda_model import train_lda
from lsa_model import train_lsa, calculate_similarity


# ============================================================
# INPUT
# ============================================================

D_OCR_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ocr_best_integration"
    / "ocr_fullframe_cleaned.csv"
)

E_OCR_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ocr_best_integration"
    / "ocr_roi_cleaned.csv"
)

CONCEPT_FILE = (
    PROJECT_ROOT
    / "data"
    / "concepts"
    / "concept_catalog.csv"
)


# ============================================================
# OUTPUT
# ============================================================

RESULT_ROOT = (
    PROJECT_ROOT
    / "module1"
    / "results"
    / "ocr_best_integration"
    / "combined_system_comparison"
)

OLD_DIR = (
    RESULT_ROOT
    / "OLD_transcript_plus_D"
)

NEW_D_DIR = (
    RESULT_ROOT
    / "NEW_transcript_plus_D"
)

NEW_E_DIR = (
    RESULT_ROOT
    / "NEW_transcript_plus_E"
)

NEW_F_DIR = (
    RESULT_ROOT
    / "NEW_transcript_plus_F"
)

AUDIT_FILE = (
    RESULT_ROOT
    / "combined_score_build_audit.csv"
)

SOURCE_FILE = (
    RESULT_ROOT
    / "frozen_chunk_sources.csv"
)

CONFIG_FILE = (
    RESULT_ROOT
    / "COMBINED_BUILD_CONFIG.txt"
)


# ============================================================
# LOCKED CONFIG
# ============================================================

DEV_IDS = [
    f"v{i}"
    for i in range(1, 41)
]

EXPECTED_OCR_FRAMES = 2398
EXPECTED_CONCEPTS = 39

OLD_EXPECTED_CHUNKS = 494
OLD_CHUNK_SECONDS = 60

OLD_LDA_WEIGHT = 0.40
OLD_LSA_WEIGHT = 0.60

NEW_EXPECTED_CHUNKS = 172
NEW_CHUNK_SECONDS = 180

NEW_LDA_WEIGHT = 0.00
NEW_LSA_WEIGHT = 1.00

BEST_OCR_FUSION = "mean"


TEXT_COLUMNS = [
    "raw_text",
    "chunk_text",
    "transcript_text",
    "text",
    "processed_text",
]

BAD_DISCOVERY_WORDS = [
    "score",
    "evidence",
    "prediction",
    "metric",
    "audit",
    "ground_truth",
    "concept",
    "bootstrap",
]


# ============================================================
# HELPERS
# ============================================================

def video_number(video_id):

    try:

        return int(
            str(video_id)
            .strip()
            .lower()
            .replace(
                "v",
                "",
            )
        )

    except Exception:

        return 999999


def normalize_subject(value):

    value = (
        str(value)
        .strip()
        .lower()
    )

    aliases = {
        "cpp": "c++",
        "cxx": "c++",
        "c plus plus": "c++",
    }

    return aliases.get(
        value,
        value,
    )


def clean_string_series(series):

    return (
        series
        .fillna("")
        .astype(str)
        .str.strip()
    )


def choose_text_column(columns):

    for column in TEXT_COLUMNS:

        if column in columns:

            return column

    return None


def normalize_series(series):

    values = (
        pd.to_numeric(
            series,
            errors="coerce",
        )
        .fillna(0.0)
        .astype(float)
    )

    minimum = float(
        values.min()
    )

    maximum = float(
        values.max()
    )

    if maximum == minimum:

        return pd.Series(
            np.zeros(
                len(values),
                dtype=float,
            ),
            index=values.index,
        )

    return (
        values - minimum
    ) / (
        maximum - minimum
    )


# ============================================================
# DISCOVER EXACT FROZEN SCAFFOLD40 CHUNKS
# ============================================================

def inspect_chunk_candidate(
    path,
    expected_rows,
    chunk_seconds,
    branch,
):

    try:

        header = pd.read_csv(
            path,
            nrows=2,
        )

    except Exception:

        return None


    required = {
        "video_id",
        "subject",
        "chunk_id",
    }

    if not required.issubset(
        header.columns
    ):

        return None


    text_column = choose_text_column(
        header.columns
    )

    if text_column is None:

        return None


    try:

        df = pd.read_csv(
            path
        )

    except Exception:

        return None


    if len(df) != expected_rows:

        return None


    df["video_id"] = (
        clean_string_series(
            df["video_id"]
        )
    )


    if df["video_id"].nunique() != 40:

        return None


    if set(
        df["video_id"]
    ) != set(
        DEV_IDS
    ):

        return None


    duplicates = int(
        df.duplicated(
            [
                "video_id",
                "chunk_id",
            ]
        ).sum()
    )

    if duplicates != 0:

        return None


    path_text = str(
        path
    ).lower()

    score = 0


    if "chunk" in path.name.lower():

        score += 10


    if branch.lower() in path_text:

        score += 8


    if str(
        chunk_seconds
    ) in path_text:

        score += 2


    if (
        "start_sec" in df.columns
        and
        "end_sec" in df.columns
    ):

        score += 6


    if "raw_text" in df.columns:

        score += 4


    if "processed_text" in df.columns:

        score += 3


    if any(
        word in path_text
        for word in BAD_DISCOVERY_WORDS
    ):

        score -= 20


    median_span = np.nan


    if (
        "start_sec" in df.columns
        and
        "end_sec" in df.columns
    ):

        start = pd.to_numeric(
            df["start_sec"],
            errors="coerce",
        )

        end = pd.to_numeric(
            df["end_sec"],
            errors="coerce",
        )

        spans = (
            end - start
        ).dropna()

        spans = spans[
            spans > 0
        ]


        if not spans.empty:

            median_span = float(
                spans.median()
            )


            if abs(
                median_span
                - chunk_seconds
            ) <= 1.0:

                score += 8

            else:

                score -= 8


    return {
        "path":
            path,

        "score":
            score,

        "text_column":
            text_column,

        "median_span":
            median_span,
    }


def discover_frozen_chunk_file(
    expected_rows,
    chunk_seconds,
    branch,
):

    candidates = []


    # --------------------------------------------------------
    # ONLY files belonging to frozen Scaffold40 are eligible.
    # --------------------------------------------------------

    for path in PROJECT_ROOT.rglob(
        "*.csv"
    ):

        path_text = str(
            path
        ).lower()


        if "scaffold40" not in path_text:

            continue


        # Never read active OCR experiment output as source.
        if "ocr_best_integration" in path_text:

            continue


        candidate = (
            inspect_chunk_candidate(
                path=path,

                expected_rows=(
                    expected_rows
                ),

                chunk_seconds=(
                    chunk_seconds
                ),

                branch=branch,
            )
        )


        if candidate is not None:

            candidates.append(
                candidate
            )


    if not candidates:

        raise FileNotFoundError(
            "\n"
            f"Could not find frozen {branch} "
            f"Scaffold40 chunk CSV with "
            f"exactly {expected_rows} rows.\n"
            "\n"
            "DO NOT regenerate transcript chunks.\n"
            f"Expected frozen config: "
            f"{chunk_seconds}s / "
            f"{expected_rows} chunks."
        )


    candidates = sorted(
        candidates,

        key=lambda item: (
            item["score"],
            -len(
                str(
                    item["path"]
                )
            ),
        ),

        reverse=True,
    )


    print()
    print(
        f"Frozen {branch} chunk candidates:"
    )


    for item in candidates[:10]:

        print(
            f"  score={item['score']:>3} "
            f"rows={expected_rows} "
            f"span={item['median_span']} "
            f"path={item['path']}"
        )


    selected = candidates[0]


    print()
    print(
        f"SELECTED {branch} "
        f"FROZEN CHUNKS:"
    )

    print(
        selected[
            "path"
        ]
    )

    print(
        "Text column:",
        selected[
            "text_column"
        ],
    )


    return selected[
        "path"
    ]


# ============================================================
# LOAD FROZEN TRANSCRIPT CHUNKS
# ============================================================

def load_frozen_chunks(
    path,
    expected_rows,
    chunk_seconds,
    label,
):

    df = pd.read_csv(
        path
    )


    required = {
        "video_id",
        "subject",
        "chunk_id",
    }


    missing = (
        required
        - set(
            df.columns
        )
    )


    if missing:

        raise RuntimeError(
            f"{label}: missing columns "
            f"{sorted(missing)}"
        )


    text_column = choose_text_column(
        df.columns
    )


    if text_column is None:

        raise RuntimeError(
            f"{label}: no transcript text "
            f"column found.\n"
            f"Columns: {list(df.columns)}"
        )


    df["video_id"] = (
        clean_string_series(
            df["video_id"]
        )
    )

    df["subject"] = (
        clean_string_series(
            df["subject"]
        )
    )


    df["chunk_id"] = pd.to_numeric(
        df["chunk_id"],
        errors="coerce",
    )


    if df["chunk_id"].isna().any():

        raise RuntimeError(
            f"{label}: invalid chunk_id."
        )


    df["chunk_id"] = (
        df["chunk_id"]
        .astype(int)
    )


    # --------------------------------------------------------
    # HARD LOCK
    # --------------------------------------------------------

    if len(df) != expected_rows:

        raise RuntimeError(
            f"{label} frozen chunk mismatch: "
            f"{len(df)} != {expected_rows}"
        )


    if df["video_id"].nunique() != 40:

        raise RuntimeError(
            f"{label}: expected 40 videos, "
            f"found "
            f"{df['video_id'].nunique()}"
        )


    if set(
        df["video_id"]
    ) != set(
        DEV_IDS
    ):

        raise RuntimeError(
            f"{label}: DEV video IDs mismatch."
        )


    duplicates = int(
        df.duplicated(
            [
                "video_id",
                "chunk_id",
            ]
        ).sum()
    )


    if duplicates != 0:

        raise RuntimeError(
            f"{label}: duplicate chunks = "
            f"{duplicates}"
        )


    # --------------------------------------------------------
    # Keep EXACT frozen transcript representation.
    # Prefer raw text if it exists.
    # --------------------------------------------------------

    df[
        "transcript_base_text"
    ] = clean_string_series(
        df[
            text_column
        ]
    )


    # --------------------------------------------------------
    # Prefer exact stored temporal boundaries.
    # --------------------------------------------------------

    if (
        "start_sec" in df.columns
        and
        "end_sec" in df.columns
    ):

        df["start_sec"] = pd.to_numeric(
            df["start_sec"],
            errors="coerce",
        )

        df["end_sec"] = pd.to_numeric(
            df["end_sec"],
            errors="coerce",
        )


        if (
            df["start_sec"].isna().any()
            or
            df["end_sec"].isna().any()
        ):

            raise RuntimeError(
                f"{label}: invalid "
                "start/end seconds."
            )


    else:

        print(
            f"{label}: no saved start/end. "
            "Reconstructing by exact frozen "
            "chunk order."
        )


        df["_video_order"] = (
            df["video_id"]
            .map(
                video_number
            )
        )


        df = (
            df
            .sort_values(
                [
                    "_video_order",
                    "chunk_id",
                ]
            )
            .drop(
                columns=[
                    "_video_order"
                ]
            )
            .reset_index(
                drop=True
            )
        )


        df["_position"] = (
            df
            .groupby(
                "video_id"
            )
            .cumcount()
        )


        df["start_sec"] = (
            df["_position"]
            * chunk_seconds
        ).astype(float)


        df["end_sec"] = (
            df["start_sec"]
            + chunk_seconds
        ).astype(float)


        df = df.drop(
            columns=[
                "_position"
            ]
        )


    if (
        df["end_sec"]
        <= df["start_sec"]
    ).any():

        raise RuntimeError(
            f"{label}: invalid chunk interval."
        )


    df["_video_order"] = (
        df["video_id"]
        .map(
            video_number
        )
    )


    df = (
        df
        .sort_values(
            [
                "_video_order",
                "start_sec",
                "chunk_id",
            ]
        )
        .drop(
            columns=[
                "_video_order"
            ]
        )
        .reset_index(
            drop=True
        )
    )


    print()
    print(
        f"{label} FROZEN CHUNK AUDIT PASS"
    )

    print(
        "Rows:",
        len(df),
    )

    print(
        "Videos:",
        df[
            "video_id"
        ].nunique(),
    )

    print(
        "Text source column:",
        text_column,
    )


    return df


# ============================================================
# LOAD CLEANED OCR
# ============================================================

def detect_ocr_text_column(df):

    candidates = [
        "ocr_text_clean",
        "clean_text",
        "cleaned_text",
        "ocr_cleaned",
    ]


    for column in candidates:

        if column in df.columns:

            return column


    return None


def load_cleaned_ocr(
    path,
    label,
):

    if not path.exists():

        raise FileNotFoundError(
            f"{label} missing:\n"
            f"{path}"
        )


    df = pd.read_csv(
        path
    )


    required = {
        "video_id",
        "subject",
        "timestamp_sec",
    }


    missing = (
        required
        - set(
            df.columns
        )
    )


    if missing:

        raise RuntimeError(
            f"{label}: missing "
            f"{sorted(missing)}"
        )


    text_column = (
        detect_ocr_text_column(
            df
        )
    )


    if text_column is None:

        raise RuntimeError(
            f"{label}: cleaned OCR text "
            "column not found.\n"
            f"Columns={list(df.columns)}"
        )


    df["video_id"] = (
        clean_string_series(
            df["video_id"]
        )
    )

    df["subject"] = (
        clean_string_series(
            df["subject"]
        )
    )


    df["timestamp_sec"] = pd.to_numeric(
        df["timestamp_sec"],
        errors="coerce",
    )


    if df[
        "timestamp_sec"
    ].isna().any():

        raise RuntimeError(
            f"{label}: bad timestamp."
        )


    df["timestamp_sec"] = (
        df["timestamp_sec"]
        .astype(float)
    )


    df["ocr_text_clean"] = (
        clean_string_series(
            df[
                text_column
            ]
        )
    )


    duplicates = int(
        df.duplicated(
            [
                "video_id",
                "timestamp_sec",
            ]
        ).sum()
    )


    missing_ids = (
        set(DEV_IDS)
        - set(
            df["video_id"]
        )
    )


    extra_ids = (
        set(
            df["video_id"]
        )
        - set(DEV_IDS)
    )


    print()
    print(
        f"{label} INPUT AUDIT"
    )

    print(
        "Rows:",
        len(df),
    )

    print(
        "Videos:",
        df[
            "video_id"
        ].nunique(),
    )

    print(
        "Duplicates:",
        duplicates,
    )

    print(
        "Missing DEV:",
        sorted(
            missing_ids,
            key=video_number,
        ),
    )

    print(
        "Extra videos:",
        sorted(
            extra_ids,
            key=video_number,
        ),
    )


    if (
        len(df)
        != EXPECTED_OCR_FRAMES
        or
        df["video_id"].nunique()
        != 40
        or
        duplicates != 0
        or
        missing_ids
        or
        extra_ids
    ):

        raise RuntimeError(
            f"{label}: INPUT AUDIT FAILED."
        )


    print(
        f"{label} INPUT PASS"
    )


    return df[
        [
            "video_id",
            "subject",
            "timestamp_sec",
            "ocr_text_clean",
        ]
    ].copy()


# ============================================================
# BUILD COMBINED CHUNKS ON FROZEN GRID
# ============================================================

def build_combined_chunks(
    frozen_chunks,
    ocr_frames,
    label,
):

    rows = []

    assigned_keys = set()


    ocr_by_video = {
        video_id:
            group
            .sort_values(
                "timestamp_sec"
            )
            .reset_index(
                drop=True
            )

        for video_id, group
        in ocr_frames.groupby(
            "video_id"
        )
    }


    for chunk in frozen_chunks.itertuples(
        index=False
    ):

        video_id = str(
            chunk.video_id
        )

        start_sec = float(
            chunk.start_sec
        )

        end_sec = float(
            chunk.end_sec
        )


        video_ocr = ocr_by_video[
            video_id
        ]


        selected = video_ocr[
            (
                video_ocr[
                    "timestamp_sec"
                ]
                >= start_sec
            )
            &
            (
                video_ocr[
                    "timestamp_sec"
                ]
                < end_sec
            )
        ]


        text_values = (
            selected[
                "ocr_text_clean"
            ]
            .fillna("")
            .astype(str)
            .str.strip()
            .tolist()
        )


        text_values = [
            text
            for text in text_values
            if text
        ]


        for frame in selected.itertuples(
            index=False
        ):

            assigned_keys.add(
                (
                    str(
                        frame.video_id
                    ),
                    float(
                        frame.timestamp_sec
                    ),
                )
            )


        ocr_text = " ".join(
            text_values
        ).strip()


        transcript_text = str(
            chunk.transcript_base_text
        ).strip()


        combined_raw_text = " ".join(
            part
            for part in [
                transcript_text,
                ocr_text,
            ]
            if part
        ).strip()


        processed_text = (
            preprocess_text(
                combined_raw_text
            )
            if combined_raw_text
            else ""
        )


        rows.append(
            {
                "video_id":
                    video_id,

                "subject":
                    str(
                        chunk.subject
                    ).strip(),

                "method":
                    label,

                "chunk_id":
                    int(
                        chunk.chunk_id
                    ),

                "start_sec":
                    start_sec,

                "end_sec":
                    end_sec,

                "transcript_text":
                    transcript_text,

                "ocr_text":
                    ocr_text,

                "ocr_frames_in_interval":
                    int(
                        len(
                            selected
                        )
                    ),

                "ocr_nonempty_frames":
                    int(
                        len(
                            text_values
                        )
                    ),

                "combined_raw_text":
                    combined_raw_text,

                "processed_text":
                    processed_text,
            }
        )


    result = pd.DataFrame(
        rows
    )


    duplicates = int(
        result.duplicated(
            [
                "video_id",
                "chunk_id",
            ]
        ).sum()
    )


    if duplicates != 0:

        raise RuntimeError(
            f"{label}: duplicate "
            f"combined chunks = "
            f"{duplicates}"
        )


    source_keys = {
        (
            str(
                row.video_id
            ),
            float(
                row.timestamp_sec
            ),
        )

        for row in ocr_frames.itertuples(
            index=False
        )
    }


    unassigned = (
        source_keys
        - assigned_keys
    )


    print()
    print(
        f"{label} COMBINED CHUNK AUDIT"
    )

    print(
        "Chunks:",
        len(result),
    )

    print(
        "Videos:",
        result[
            "video_id"
        ].nunique(),
    )

    print(
        "Chunks with OCR text:",
        int(
            result[
                "ocr_text"
            ].ne("").sum()
        ),
    )

    print(
        "Empty processed chunks:",
        int(
            result[
                "processed_text"
            ].eq("").sum()
        ),
    )

    print(
        "OCR frames assigned:",
        len(
            assigned_keys
        ),
    )

    print(
        "OCR frames outside frozen grid:",
        len(
            unassigned
        ),
    )


    return (
        result,
        len(
            assigned_keys
        ),
        len(
            unassigned
        ),
    )


# ============================================================
# CONCEPTS
# ============================================================

def load_concepts():

    if not CONCEPT_FILE.exists():

        raise FileNotFoundError(
            CONCEPT_FILE
        )


    concepts = pd.read_csv(
        CONCEPT_FILE
    )


    required = {
        "subject",
        "concept",
        "description",
    }


    missing = (
        required
        - set(
            concepts.columns
        )
    )


    if missing:

        raise RuntimeError(
            "Concept catalog missing: "
            f"{sorted(missing)}"
        )


    concepts["subject"] = (
        clean_string_series(
            concepts[
                "subject"
            ]
        )
    )

    concepts["concept"] = (
        clean_string_series(
            concepts[
                "concept"
            ]
        )
    )

    concepts["description"] = (
        clean_string_series(
            concepts[
                "description"
            ]
        )
    )


    concepts["subject_norm"] = (
        concepts[
            "subject"
        ]
        .map(
            normalize_subject
        )
    )


    concepts["concept_text"] = (
        concepts["concept"]
        + " "
        + concepts["description"]
    ).str.strip()


    concepts[
        "processed_concept_text"
    ] = (
        concepts[
            "concept_text"
        ]
        .apply(
            preprocess_text
        )
    )


    duplicates = int(
        concepts.duplicated(
            [
                "subject_norm",
                "concept",
            ]
        ).sum()
    )


    if (
        len(concepts)
        != EXPECTED_CONCEPTS
        or
        duplicates != 0
    ):

        raise RuntimeError(
            "Concept catalog audit FAILED."
        )


    print()
    print(
        "CONCEPT AUDIT PASS"
    )

    print(
        "Concepts:",
        len(
            concepts
        ),
    )

    print(
        concepts
        .groupby(
            "subject"
        )
        .size()
        .to_string()
    )


    return concepts


# ============================================================
# EXPECTED SCORE COUNT
# ============================================================

def expected_score_rows(
    chunks,
    concepts,
):

    counts = (
        concepts
        .groupby(
            "subject_norm"
        )
        .size()
        .to_dict()
    )


    total = 0


    for subject in chunks[
        "subject"
    ]:

        subject_norm = normalize_subject(
            subject
        )


        if subject_norm not in counts:

            raise RuntimeError(
                "No concepts for subject: "
                f"{subject}"
            )


        total += int(
            counts[
                subject_norm
            ]
        )


    return int(
        total
    )


# ============================================================
# BUILD SAME-SUBJECT SCORE DATAFRAME
# ============================================================

def build_same_subject_scores(
    chunks,
    concepts,
    matrix,
    score_column,
):

    rows = []


    for chunk_index, chunk in (
        chunks.iterrows()
    ):

        chunk_subject = (
            normalize_subject(
                chunk[
                    "subject"
                ]
            )
        )


        for concept_index, concept in (
            concepts.iterrows()
        ):

            if (
                chunk_subject
                != concept[
                    "subject_norm"
                ]
            ):

                continue


            rows.append(
                {
                    "video_id":
                        chunk[
                            "video_id"
                        ],

                    "subject":
                        chunk[
                            "subject"
                        ],

                    "method":
                        chunk[
                            "method"
                        ],

                    "chunk_id":
                        int(
                            chunk[
                                "chunk_id"
                            ]
                        ),

                    "start_sec":
                        float(
                            chunk[
                                "start_sec"
                            ]
                        ),

                    "end_sec":
                        float(
                            chunk[
                                "end_sec"
                            ]
                        ),

                    "concept":
                        concept[
                            "concept"
                        ],

                    score_column:
                        round(
                            float(
                                matrix[
                                    chunk_index,
                                    concept_index,
                                ]
                            ),
                            4,
                        ),
                }
            )


    return pd.DataFrame(
        rows
    )


# ============================================================
# LSA
# ============================================================

def run_lsa_scores(
    chunks,
    concepts,
):

    documents = (
        chunks[
            "processed_text"
        ]
        .fillna("")
        .astype(str)
        .tolist()
    )


    concept_texts = (
        concepts[
            "processed_concept_text"
        ]
        .fillna("")
        .astype(str)
        .tolist()
    )


    (
        _vectorizer,
        _svd,
        document_vectors,
        concept_vectors,
    ) = train_lsa(
        documents,
        concept_texts,
    )


    matrix = (
        calculate_similarity(
            document_vectors,
            concept_vectors,
        )
    )


    return build_same_subject_scores(
        chunks=chunks,

        concepts=concepts,

        matrix=matrix,

        score_column="lsa_score",
    )


# ============================================================
# LDA
# ============================================================

def run_lda_scores(
    chunks,
    concepts,
):

    documents = (
        chunks[
            "processed_text"
        ]
        .fillna("")
        .astype(str)
        .tolist()
    )


    (
        vectorizer,
        lda_model,
        document_topic_matrix,
    ) = train_lda(
        documents
    )


    topic_word = (
        lda_model
        .components_
        .astype(float)
    )


    topic_sums = (
        topic_word
        .sum(
            axis=1,
            keepdims=True,
        )
    )


    topic_sums[
        topic_sums
        == 0
    ] = 1.0


    topic_word_distribution = (
        topic_word
        / topic_sums
    )


    concept_vectors = (
        vectorizer
        .transform(
            concepts[
                "processed_concept_text"
            ]
        )
    )


    topic_concept_affinity = (
        cosine_similarity(
            topic_word_distribution,
            concept_vectors,
        )
    )


    matrix = np.matmul(
        document_topic_matrix,
        topic_concept_affinity,
    )


    return build_same_subject_scores(
        chunks=chunks,

        concepts=concepts,

        matrix=matrix,

        score_column="lda_score",
    )


# ============================================================
# NORMALIZE
# ============================================================

def add_normalized_score(
    df,
    source_column,
    output_column,
):

    df[
        output_column
    ] = (
        df
        .groupby(
            [
                "video_id",
                "chunk_id",
            ]
        )[
            source_column
        ]
        .transform(
            normalize_series
        )
    )


    return df


# ============================================================
# OLD = TRANSCRIPT OLD + OCR D
# ============================================================

def score_old_combined(
    chunks,
    concepts,
):

    print()
    print(
        "=" * 72
    )

    print(
        "SCORE OLD COMBINED"
    )

    print(
        "OLD transcript + OCR D"
    )

    print(
        "60s / LDA .40 + LSA .60"
    )

    print(
        "=" * 72
    )


    OLD_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    chunks.to_csv(
        OLD_DIR
        / "combined_chunks.csv",

        index=False,

        encoding="utf-8-sig",
    )


    print(
        "Running OLD LDA..."
    )

    lda = run_lda_scores(
        chunks,
        concepts,
    )


    print(
        "Running OLD LSA..."
    )

    lsa = run_lsa_scores(
        chunks,
        concepts,
    )


    keys = [
        "video_id",
        "subject",
        "method",
        "chunk_id",
        "start_sec",
        "end_sec",
        "concept",
    ]


    scores = pd.merge(
        lda,
        lsa,

        on=keys,

        how="inner",

        validate="one_to_one",
    )


    if (
        len(scores) != len(lda)
        or
        len(scores) != len(lsa)
    ):

        raise RuntimeError(
            "OLD LDA/LSA merge "
            "lost rows."
        )


    scores = add_normalized_score(
        scores,
        "lda_score",
        "lda_norm",
    )


    scores = add_normalized_score(
        scores,
        "lsa_score",
        "lsa_norm",
    )


    scores["final_score"] = (
        OLD_LDA_WEIGHT
        * scores[
            "lda_norm"
        ]

        +

        OLD_LSA_WEIGHT
        * scores[
            "lsa_norm"
        ]
    ).round(
        4
    )


    expected = (
        expected_score_rows(
            chunks,
            concepts,
        )
    )


    duplicates = int(
        scores.duplicated(
            [
                "video_id",
                "chunk_id",
                "concept",
            ]
        ).sum()
    )


    if (
        len(scores) != expected
        or
        duplicates != 0
        or
        scores[
            "final_score"
        ].isna().any()
    ):

        raise RuntimeError(
            "OLD combined score "
            "audit FAILED."
        )


    scores.to_csv(
        OLD_DIR
        / "fusion_scores.csv",

        index=False,

        encoding="utf-8-sig",
    )


    print(
        "OLD score rows:",
        len(
            scores
        ),
    )

    print(
        "OLD SCORE PASS"
    )


    return scores


# ============================================================
# NEW LSA-ONLY BRANCH
# ============================================================

def score_new_lsa_branch(
    chunks,
    concepts,
    output_dir,
    label,
):

    print()
    print(
        "=" * 72
    )

    print(
        label
    )

    print(
        "180s / LDA 0 + LSA 1"
    )

    print(
        "=" * 72
    )


    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    chunks.to_csv(
        output_dir
        / "combined_chunks.csv",

        index=False,

        encoding="utf-8-sig",
    )


    lsa = run_lsa_scores(
        chunks,
        concepts,
    )


    lsa = add_normalized_score(
        lsa,
        "lsa_score",
        "lsa_norm",
    )


    lsa["lda_norm"] = 0.0


    lsa["final_score"] = (
        lsa[
            "lsa_norm"
        ]
        .astype(float)
        .round(4)
    )


    expected = (
        expected_score_rows(
            chunks,
            concepts,
        )
    )


    duplicates = int(
        lsa.duplicated(
            [
                "video_id",
                "chunk_id",
                "concept",
            ]
        ).sum()
    )


    if (
        len(lsa) != expected
        or
        duplicates != 0
        or
        lsa[
            "final_score"
        ].isna().any()
    ):

        raise RuntimeError(
            f"{label}: score audit FAILED."
        )


    lsa.to_csv(
        output_dir
        / "fusion_scores.csv",

        index=False,

        encoding="utf-8-sig",
    )


    print(
        label,
        "score rows:",
        len(
            lsa
        ),
    )

    print(
        label,
        "PASS",
    )


    return lsa


# ============================================================
# NEW F = MEAN OF NEW+D AND NEW+E SCORE
# ============================================================

def build_new_f_scores(
    d_scores,
    e_scores,
):

    NEW_F_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    keys = [
        "video_id",
        "subject",
        "chunk_id",
        "start_sec",
        "end_sec",
        "concept",
    ]


    d = (
        d_scores[
            keys
            + [
                "final_score"
            ]
        ]
        .rename(
            columns={
                "final_score":
                    "final_score_d"
            }
        )
    )


    e = (
        e_scores[
            keys
            + [
                "final_score"
            ]
        ]
        .rename(
            columns={
                "final_score":
                    "final_score_e"
            }
        )
    )


    merged = pd.merge(
        d,
        e,

        on=keys,

        how="outer",

        validate="one_to_one",

        indicator=True,
    )


    mismatch = int(
        merged[
            "_merge"
        ]
        .ne(
            "both"
        )
        .sum()
    )


    if mismatch != 0:

        print(
            merged[
                merged[
                    "_merge"
                ]
                .ne(
                    "both"
                )
            ]
            .head(20)
            .to_string(
                index=False
            )
        )


        raise RuntimeError(
            "NEW D/E score-key mismatch: "
            f"{mismatch}"
        )


    merged = merged.drop(
        columns=[
            "_merge"
        ]
    )


    merged["method"] = (
        "NEW_transcript_plus_F"
    )


    merged["final_score"] = (
        (
            merged[
                "final_score_d"
            ]
            +
            merged[
                "final_score_e"
            ]
        )
        / 2.0
    ).round(
        4
    )


    merged[
        "fusion_strategy"
    ] = BEST_OCR_FUSION


    expected_mean = (
        (
            merged[
                "final_score_d"
            ]
            +
            merged[
                "final_score_e"
            ]
        )
        / 2.0
    ).round(
        4
    )


    formula_error = float(
        (
            merged[
                "final_score"
            ]
            - expected_mean
        )
        .abs()
        .max()
    )


    if (
        merged[
            "final_score"
        ].isna().any()
        or
        formula_error > 1e-12
    ):

        raise RuntimeError(
            "NEW F mean formula "
            "audit FAILED."
        )


    merged.to_csv(
        NEW_F_DIR
        / "fusion_scores.csv",

        index=False,

        encoding="utf-8-sig",
    )


    print()
    print(
        "NEW F FUSION PASS"
    )

    print(
        "Rows:",
        len(
            merged
        ),
    )

    print(
        "Strategy:",
        BEST_OCR_FUSION,
    )

    print(
        "Formula max error:",
        formula_error,
    )


    return (
        merged,
        formula_error,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 72
    )

    print(
        "STEP 12 - BUILD "
        "OLD vs NEW COMBINED SCORES"
    )

    print(
        "=" * 72
    )

    print()
    print(
        "THIS VERSION DOES NOT "
        "REBUILD TRANSCRIPT CHUNKS."
    )

    print(
        "OLD must be exactly "
        "494 frozen chunks."
    )

    print(
        "NEW must be exactly "
        "172 frozen chunks."
    )

    print(
        "No GT / no F1 / "
        "no threshold used."
    )


    RESULT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )


    # ========================================================
    # 1. DISCOVER FROZEN CHUNKS
    # ========================================================

    old_chunk_file = (
        discover_frozen_chunk_file(
            expected_rows=(
                OLD_EXPECTED_CHUNKS
            ),

            chunk_seconds=(
                OLD_CHUNK_SECONDS
            ),

            branch="OLD",
        )
    )


    new_chunk_file = (
        discover_frozen_chunk_file(
            expected_rows=(
                NEW_EXPECTED_CHUNKS
            ),

            chunk_seconds=(
                NEW_CHUNK_SECONDS
            ),

            branch="NEW",
        )
    )


    if (
        old_chunk_file.resolve()
        ==
        new_chunk_file.resolve()
    ):

        raise RuntimeError(
            "OLD and NEW selected "
            "the same chunk file."
        )


    pd.DataFrame(
        [
            {
                "branch":
                    "OLD",

                "chunk_seconds":
                    OLD_CHUNK_SECONDS,

                "expected_chunks":
                    OLD_EXPECTED_CHUNKS,

                "source_file":
                    str(
                        old_chunk_file.resolve()
                    ),
            },

            {
                "branch":
                    "NEW",

                "chunk_seconds":
                    NEW_CHUNK_SECONDS,

                "expected_chunks":
                    NEW_EXPECTED_CHUNKS,

                "source_file":
                    str(
                        new_chunk_file.resolve()
                    ),
            },
        ]
    ).to_csv(
        SOURCE_FILE,

        index=False,

        encoding="utf-8-sig",
    )


    # ========================================================
    # 2. LOAD INPUT
    # ========================================================

    old_base = load_frozen_chunks(
        path=old_chunk_file,

        expected_rows=(
            OLD_EXPECTED_CHUNKS
        ),

        chunk_seconds=(
            OLD_CHUNK_SECONDS
        ),

        label="OLD",
    )


    new_base = load_frozen_chunks(
        path=new_chunk_file,

        expected_rows=(
            NEW_EXPECTED_CHUNKS
        ),

        chunk_seconds=(
            NEW_CHUNK_SECONDS
        ),

        label="NEW",
    )


    d_ocr = load_cleaned_ocr(
        D_OCR_FILE,

        "OCR D FULL-FRAME CLEANED",
    )


    e_ocr = load_cleaned_ocr(
        E_OCR_FILE,

        "OCR E ROI CLEANED",
    )


    concepts = load_concepts()


    # ========================================================
    # 3. COMBINE OCR ON EXACT FROZEN GRID
    # ========================================================

    (
        old_combined,
        old_assigned,
        old_unassigned,
    ) = build_combined_chunks(
        frozen_chunks=old_base,

        ocr_frames=d_ocr,

        label="OLD_transcript_plus_D",
    )


    (
        new_d_combined,
        new_d_assigned,
        new_d_unassigned,
    ) = build_combined_chunks(
        frozen_chunks=new_base,

        ocr_frames=d_ocr,

        label="NEW_transcript_plus_D",
    )


    (
        new_e_combined,
        new_e_assigned,
        new_e_unassigned,
    ) = build_combined_chunks(
        frozen_chunks=new_base,

        ocr_frames=e_ocr,

        label="NEW_transcript_plus_E",
    )


    # --------------------------------------------------------
    # HARD LOCK COUNTS
    # --------------------------------------------------------

    if (
        len(
            old_combined
        )
        !=
        OLD_EXPECTED_CHUNKS
    ):

        raise RuntimeError(
            "OLD combined changed "
            "frozen chunk count."
        )


    if (
        len(
            new_d_combined
        )
        !=
        NEW_EXPECTED_CHUNKS
    ):

        raise RuntimeError(
            "NEW+D changed "
            "frozen chunk count."
        )


    if (
        len(
            new_e_combined
        )
        !=
        NEW_EXPECTED_CHUNKS
    ):

        raise RuntimeError(
            "NEW+E changed "
            "frozen chunk count."
        )


    # --------------------------------------------------------
    # NEW D/E MUST SHARE EXACT SAME CHUNK KEYS
    # --------------------------------------------------------

    key_columns = [
        "video_id",
        "chunk_id",
        "start_sec",
        "end_sec",
    ]


    new_d_keys = set(
        map(
            tuple,

            new_d_combined[
                key_columns
            ].itertuples(
                index=False,
                name=None,
            ),
        )
    )


    new_e_keys = set(
        map(
            tuple,

            new_e_combined[
                key_columns
            ].itertuples(
                index=False,
                name=None,
            ),
        )
    )


    chunk_mismatch = len(
        new_d_keys.symmetric_difference(
            new_e_keys
        )
    )


    if chunk_mismatch != 0:

        raise RuntimeError(
            "NEW D/E chunk grids mismatch."
        )


    # ========================================================
    # 4. SCORE SYSTEMS
    # ========================================================

    old_scores = (
        score_old_combined(
            old_combined,
            concepts,
        )
    )


    new_d_scores = (
        score_new_lsa_branch(
            chunks=(
                new_d_combined
            ),

            concepts=concepts,

            output_dir=(
                NEW_D_DIR
            ),

            label=(
                "NEW TRANSCRIPT + OCR D"
            ),
        )
    )


    new_e_scores = (
        score_new_lsa_branch(
            chunks=(
                new_e_combined
            ),

            concepts=concepts,

            output_dir=(
                NEW_E_DIR
            ),

            label=(
                "NEW TRANSCRIPT + OCR E"
            ),
        )
    )


    # ========================================================
    # 5. BUILD NEW OCR F
    # ========================================================

    (
        new_f_scores,
        formula_error,
    ) = build_new_f_scores(
        new_d_scores,
        new_e_scores,
    )


    # ========================================================
    # 6. FINAL AUDIT
    # ========================================================

    old_expected_scores = (
        expected_score_rows(
            old_combined,
            concepts,
        )
    )


    new_expected_scores = (
        expected_score_rows(
            new_d_combined,
            concepts,
        )
    )


    audit = pd.DataFrame(
        [
            {
                "check":
                    "OLD_frozen_chunks",

                "value":
                    len(
                        old_base
                    ),

                "expected":
                    OLD_EXPECTED_CHUNKS,

                "pass":
                    len(
                        old_base
                    )
                    ==
                    OLD_EXPECTED_CHUNKS,
            },

            {
                "check":
                    "NEW_frozen_chunks",

                "value":
                    len(
                        new_base
                    ),

                "expected":
                    NEW_EXPECTED_CHUNKS,

                "pass":
                    len(
                        new_base
                    )
                    ==
                    NEW_EXPECTED_CHUNKS,
            },

            {
                "check":
                    "OLD_combined_chunks",

                "value":
                    len(
                        old_combined
                    ),

                "expected":
                    OLD_EXPECTED_CHUNKS,

                "pass":
                    len(
                        old_combined
                    )
                    ==
                    OLD_EXPECTED_CHUNKS,
            },

            {
                "check":
                    "NEW_D_combined_chunks",

                "value":
                    len(
                        new_d_combined
                    ),

                "expected":
                    NEW_EXPECTED_CHUNKS,

                "pass":
                    len(
                        new_d_combined
                    )
                    ==
                    NEW_EXPECTED_CHUNKS,
            },

            {
                "check":
                    "NEW_E_combined_chunks",

                "value":
                    len(
                        new_e_combined
                    ),

                "expected":
                    NEW_EXPECTED_CHUNKS,

                "pass":
                    len(
                        new_e_combined
                    )
                    ==
                    NEW_EXPECTED_CHUNKS,
            },

            {
                "check":
                    "NEW_D_E_chunk_mismatch",

                "value":
                    chunk_mismatch,

                "expected":
                    0,

                "pass":
                    chunk_mismatch
                    == 0,
            },

            {
                "check":
                    "OLD_score_rows",

                "value":
                    len(
                        old_scores
                    ),

                "expected":
                    old_expected_scores,

                "pass":
                    len(
                        old_scores
                    )
                    ==
                    old_expected_scores,
            },

            {
                "check":
                    "NEW_D_score_rows",

                "value":
                    len(
                        new_d_scores
                    ),

                "expected":
                    new_expected_scores,

                "pass":
                    len(
                        new_d_scores
                    )
                    ==
                    new_expected_scores,
            },

            {
                "check":
                    "NEW_E_score_rows",

                "value":
                    len(
                        new_e_scores
                    ),

                "expected":
                    new_expected_scores,

                "pass":
                    len(
                        new_e_scores
                    )
                    ==
                    new_expected_scores,
            },

            {
                "check":
                    "NEW_F_score_rows",

                "value":
                    len(
                        new_f_scores
                    ),

                "expected":
                    new_expected_scores,

                "pass":
                    len(
                        new_f_scores
                    )
                    ==
                    new_expected_scores,
            },

            {
                "check":
                    "NEW_F_formula_error",

                "value":
                    formula_error,

                "expected":
                    0.0,

                "pass":
                    formula_error
                    <= 1e-12,
            },
        ]
    )


    audit.to_csv(
        AUDIT_FILE,

        index=False,

        encoding="utf-8-sig",
    )


    all_pass = bool(
        audit[
            "pass"
        ].all()
    )


    CONFIG_FILE.write_text(
        "\n".join(
            [
                "STEP 12 - COMBINED SCORE BUILD",
                "",
                (
                    "OLD chunk file = "
                    + str(
                        old_chunk_file
                    )
                ),
                (
                    "OLD chunks = "
                    + str(
                        OLD_EXPECTED_CHUNKS
                    )
                ),
                (
                    "OLD chunk seconds = "
                    + str(
                        OLD_CHUNK_SECONDS
                    )
                ),
                (
                    "OLD LDA weight = "
                    + str(
                        OLD_LDA_WEIGHT
                    )
                ),
                (
                    "OLD LSA weight = "
                    + str(
                        OLD_LSA_WEIGHT
                    )
                ),
                "",
                (
                    "NEW chunk file = "
                    + str(
                        new_chunk_file
                    )
                ),
                (
                    "NEW chunks = "
                    + str(
                        NEW_EXPECTED_CHUNKS
                    )
                ),
                (
                    "NEW chunk seconds = "
                    + str(
                        NEW_CHUNK_SECONDS
                    )
                ),
                (
                    "NEW LDA weight = "
                    + str(
                        NEW_LDA_WEIGHT
                    )
                ),
                (
                    "NEW LSA weight = "
                    + str(
                        NEW_LSA_WEIGHT
                    )
                ),
                (
                    "NEW OCR F fusion = "
                    + BEST_OCR_FUSION
                ),
                "",
                "GT used = NO",
                "F1 calculated = NO",
                "Threshold used = NO",
                "DEV retuning = NO",
            ]
        ),

        encoding="utf-8",
    )


    # ========================================================
    # PRINT FINAL
    # ========================================================

    print()
    print(
        "=" * 72
    )

    print(
        "STEP 12 FINAL AUDIT"
    )

    print(
        "=" * 72
    )


    print(
        "Frozen OLD chunks:",
        len(
            old_base
        ),
    )

    print(
        "Frozen NEW chunks:",
        len(
            new_base
        ),
    )

    print()


    print(
        "OLD combined chunks:",
        len(
            old_combined
        ),
    )

    print(
        "NEW+D combined chunks:",
        len(
            new_d_combined
        ),
    )

    print(
        "NEW+E combined chunks:",
        len(
            new_e_combined
        ),
    )

    print(
        "NEW D/E chunk mismatch:",
        chunk_mismatch,
    )

    print()


    print(
        "OLD score rows:",
        len(
            old_scores
        ),
    )

    print(
        "NEW+D score rows:",
        len(
            new_d_scores
        ),
    )

    print(
        "NEW+E score rows:",
        len(
            new_e_scores
        ),
    )

    print(
        "NEW+F score rows:",
        len(
            new_f_scores
        ),
    )

    print()


    print(
        "OLD OCR assigned:",
        old_assigned,
    )

    print(
        "OLD OCR outside frozen grid:",
        old_unassigned,
    )

    print(
        "NEW D OCR assigned:",
        new_d_assigned,
    )

    print(
        "NEW D OCR outside frozen grid:",
        new_d_unassigned,
    )

    print(
        "NEW E OCR assigned:",
        new_e_assigned,
    )

    print(
        "NEW E OCR outside frozen grid:",
        new_e_unassigned,
    )

    print()


    print(
        "NEW F fusion:",
        BEST_OCR_FUSION,
    )

    print(
        "NEW F formula max error:",
        formula_error,
    )

    print()


    print(
        audit.to_string(
            index=False
        )
    )


    if not all_pass:

        raise RuntimeError(
            "STEP 12 FAILED FINAL AUDIT."
        )


    print()
    print(
        "STEP 12 PASS - "
        "OLD AND NEW COMBINED "
        "SCORES COMPLETE"
    )

    print()
    print(
        "NEXT STEP ONLY: "
        "evaluate OLD combined "
        "vs NEW combined."
    )


if __name__ == "__main__":

    main()