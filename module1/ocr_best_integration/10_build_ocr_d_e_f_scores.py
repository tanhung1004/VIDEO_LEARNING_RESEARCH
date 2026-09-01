from pathlib import Path
import sys
import re

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# CONFIG
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SRC_DIR = (
    PROJECT_ROOT
    / "module1"
    / "src"
)

sys.path.insert(
    0,
    str(SRC_DIR),
)

from preprocessing import preprocess_text
from lda_model import train_lda
from lsa_model import (
    train_lsa,
    calculate_similarity,
)


D_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ocr_best_integration"
    / "ocr_fullframe_cleaned.csv"
)

E_FILE = (
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

RESULT_ROOT = (
    PROJECT_ROOT
    / "module1"
    / "results"
    / "ocr_best_integration"
    / "ocr_only_comparison"
)

D_DIR = (
    RESULT_ROOT
    / "D_fullframe"
)

E_DIR = (
    RESULT_ROOT
    / "E_roi"
)

F_DIR = (
    RESULT_ROOT
    / "F_mean"
)

AUDIT_FILE = (
    RESULT_ROOT
    / "ocr_d_e_f_score_audit.csv"
)

EXPECTED_FRAMES = 2398

DEV_IDS = [
    f"v{i}"
    for i in range(1, 41)
]

# ---------------------------------------------------------
# CONTROLLED OCR-ONLY COMPARISON CONFIG
#
# Same downstream configuration for D and E.
# This stage isolates OCR representation.
# ---------------------------------------------------------

CHUNK_SECONDS = 60

OCR_DUPLICATE_THRESHOLD = 0.85

LDA_WEIGHT = 0.40
LSA_WEIGHT = 0.60

FUSION_STRATEGY = "mean"


# =========================================================
# HELPERS
# =========================================================

def video_sort_key(video_id):
    try:
        return int(
            str(video_id)
            .strip()
            .lower()
            .replace("v", "")
        )
    except Exception:
        return 999999


def get_token_set(text):
    return set(
        re.findall(
            r"[a-z0-9_]+",
            str(text).lower(),
        )
    )


def jaccard_similarity(
    text_a,
    text_b,
):
    set_a = get_token_set(
        text_a
    )

    set_b = get_token_set(
        text_b
    )

    if (
        not set_a
        or not set_b
    ):
        return 0.0

    union = (
        set_a
        | set_b
    )

    if not union:
        return 0.0

    return (
        len(
            set_a
            & set_b
        )
        / len(union)
    )


def remove_duplicate_ocr_texts(
    texts,
):
    kept = []

    for text in texts:
        text = str(
            text or ""
        ).strip()

        if not text:
            continue

        duplicate = False

        for existing in kept:
            similarity = (
                jaccard_similarity(
                    text,
                    existing,
                )
            )

            if (
                similarity
                >= OCR_DUPLICATE_THRESHOLD
            ):
                duplicate = True
                break

        if not duplicate:
            kept.append(
                text
            )

    return kept


def minmax_normalize(series):
    series = (
        pd.to_numeric(
            series,
            errors="coerce",
        )
        .fillna(0.0)
        .astype(float)
    )

    minimum = float(
        series.min()
    )

    maximum = float(
        series.max()
    )

    if maximum == minimum:
        return pd.Series(
            [0.0] * len(series),
            index=series.index,
        )

    return (
        (series - minimum)
        /
        (maximum - minimum)
    )


# =========================================================
# LOAD OCR
# =========================================================

def load_ocr(
    path,
    label,
):
    if not path.exists():
        raise FileNotFoundError(
            f"{label} missing:\n{path}"
        )

    df = pd.read_csv(
        path
    )

    required = {
        "video_id",
        "subject",
        "timestamp_sec",
        "ocr_text_clean",
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            f"{label} missing columns: "
            f"{sorted(missing)}"
        )

    df["video_id"] = (
        df["video_id"]
        .astype(str)
        .str.strip()
    )

    df["subject"] = (
        df["subject"]
        .astype(str)
        .str.strip()
    )

    df["timestamp_sec"] = (
        pd.to_numeric(
            df["timestamp_sec"],
            errors="coerce",
        )
    )

    df = df.dropna(
        subset=[
            "timestamp_sec"
        ]
    ).copy()

    df["timestamp_sec"] = (
        df["timestamp_sec"]
        .astype(int)
    )

    df["ocr_text_clean"] = (
        df["ocr_text_clean"]
        .fillna("")
        .astype(str)
    )

    duplicates = int(
        df.duplicated(
            [
                "video_id",
                "timestamp_sec",
            ]
        ).sum()
    )

    missing_videos = sorted(
        set(DEV_IDS)
        - set(
            df["video_id"]
        )
    )

    extra_videos = sorted(
        set(
            df["video_id"]
        )
        - set(DEV_IDS)
    )

    print()
    print(
        f"========== {label} INPUT =========="
    )

    print(
        "Rows:",
        len(df),
    )

    print(
        "Videos:",
        df["video_id"].nunique(),
    )

    print(
        "Duplicates:",
        duplicates,
    )

    print(
        "Missing DEV:",
        missing_videos,
    )

    print(
        "Extra videos:",
        extra_videos,
    )

    passed = (
        len(df)
        == EXPECTED_FRAMES

        and df[
            "video_id"
        ].nunique()
        == 40

        and duplicates
        == 0

        and len(
            missing_videos
        )
        == 0

        and len(
            extra_videos
        )
        == 0
    )

    if not passed:
        raise RuntimeError(
            f"{label} input audit FAILED."
        )

    print(
        f"{label} INPUT PASS"
    )

    return df


# =========================================================
# D / E FRAME-KEY ALIGNMENT
# =========================================================

def audit_frame_alignment(
    d,
    e,
):
    columns = [
        "video_id",
        "subject",
        "timestamp_sec",
    ]

    d_keys = set(
        map(
            tuple,
            d[
                columns
            ].itertuples(
                index=False,
                name=None,
            ),
        )
    )

    e_keys = set(
        map(
            tuple,
            e[
                columns
            ].itertuples(
                index=False,
                name=None,
            ),
        )
    )

    d_only = (
        d_keys
        - e_keys
    )

    e_only = (
        e_keys
        - d_keys
    )

    print()
    print(
        "========== D/E FRAME ALIGNMENT =========="
    )

    print(
        "D keys:",
        len(d_keys),
    )

    print(
        "E keys:",
        len(e_keys),
    )

    print(
        "D-only:",
        len(d_only),
    )

    print(
        "E-only:",
        len(e_only),
    )

    if (
        d_only
        or e_only
    ):
        raise RuntimeError(
            "D/E frame alignment FAILED."
        )

    print(
        "D/E FRAME ALIGNMENT PASS"
    )


# =========================================================
# BUILD OCR-ONLY 60s CHUNKS
# =========================================================

def build_ocr_chunks(
    df,
    tag,
):
    rows = []

    data = df.copy()

    data["chunk_id"] = (
        data[
            "timestamp_sec"
        ]
        // CHUNK_SECONDS
        + 1
    )

    grouped = data.groupby(
        [
            "video_id",
            "subject",
            "chunk_id",
        ],
        sort=False,
    )

    for (
        video_id,
        subject,
        chunk_id,
    ), group in grouped:

        start_sec = (
            (
                int(
                    chunk_id
                )
                - 1
            )
            * CHUNK_SECONDS
        )

        end_sec = (
            start_sec
            + CHUNK_SECONDS
        )

        group = (
            group
            .sort_values(
                "timestamp_sec"
            )
        )

        texts = (
            group[
                "ocr_text_clean"
            ]
            .fillna("")
            .astype(str)
            .tolist()
        )

        kept_texts = (
            remove_duplicate_ocr_texts(
                texts
            )
        )

        raw_text = " ".join(
            kept_texts
        ).strip()

        processed_text = (
            preprocess_text(
                raw_text
            )
            if raw_text
            else ""
        )

        rows.append(
            {
                "video_id":
                    video_id,

                "subject":
                    subject,

                "method":
                    "ocr_only",

                "chunk_id":
                    int(
                        chunk_id
                    ),

                "start_sec":
                    float(
                        start_sec
                    ),

                "end_sec":
                    float(
                        end_sec
                    ),

                "ocr_frame_count":
                    int(
                        len(group)
                    ),

                "ocr_nonempty_frames":
                    int(
                        sum(
                            bool(
                                str(text)
                                .strip()
                            )
                            for text
                            in texts
                        )
                    ),

                "ocr_kept_frames":
                    int(
                        len(
                            kept_texts
                        )
                    ),

                "raw_text":
                    raw_text,

                "processed_text":
                    processed_text,

                "branch":
                    tag,
            }
        )

    result = pd.DataFrame(
        rows
    )

    result["_video_order"] = (
        result[
            "video_id"
        ]
        .map(
            video_sort_key
        )
    )

    result = (
        result
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
            f"{tag}: duplicate chunks = "
            f"{duplicates}"
        )

    return result


# =========================================================
# CONCEPTS
# =========================================================

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
        raise ValueError(
            "Concept catalog missing: "
            f"{sorted(missing)}"
        )

    concepts["subject"] = (
        concepts[
            "subject"
        ]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    concepts["concept"] = (
        concepts[
            "concept"
        ]
        .fillna("")
        .astype(str)
    )

    concepts["description"] = (
        concepts[
            "description"
        ]
        .fillna("")
        .astype(str)
    )

    concepts["concept_text"] = (
        concepts[
            "concept"
        ]
        + " "
        + concepts[
            "description"
        ]
    )

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

    return concepts


# =========================================================
# LDA SCORE
# =========================================================

def build_lda_scores(
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

    row_sum = (
        topic_word
        .sum(
            axis=1,
            keepdims=True,
        )
    )

    row_sum[
        row_sum
        == 0
    ] = 1.0

    topic_word_distribution = (
        topic_word
        / row_sum
    )

    concept_vectors = (
        vectorizer
        .transform(
            concepts[
                "processed_concept_text"
            ]
        )
    )

    affinity = (
        cosine_similarity(
            topic_word_distribution,
            concept_vectors,
        )
    )

    document_concept_scores = (
        np.matmul(
            document_topic_matrix,
            affinity,
        )
    )

    rows = []

    for chunk_index, chunk in (
        chunks.iterrows()
    ):

        subject = str(
            chunk[
                "subject"
            ]
        ).strip()

        for concept_index, concept in (
            concepts.iterrows()
        ):

            if (
                str(
                    concept[
                        "subject"
                    ]
                ).strip()
                != subject
            ):
                continue

            rows.append(
                {
                    "video_id":
                        chunk[
                            "video_id"
                        ],

                    "subject":
                        subject,

                    "method":
                        "ocr_only",

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

                    "lda_score":
                        float(
                            document_concept_scores[
                                chunk_index,
                                concept_index,
                            ]
                        ),
                }
            )

    return pd.DataFrame(
        rows
    )


# =========================================================
# LSA SCORE
# =========================================================

def build_lsa_scores(
    chunks,
    concepts,
):
    document_texts = (
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
        vectorizer,
        svd,
        document_vectors,
        concept_vectors,
    ) = train_lsa(
        document_texts,
        concept_texts,
    )

    similarity_matrix = (
        calculate_similarity(
            document_vectors,
            concept_vectors,
        )
    )

    rows = []

    for chunk_index, chunk in (
        chunks.iterrows()
    ):

        subject = str(
            chunk[
                "subject"
            ]
        ).strip()

        for concept_index, concept in (
            concepts.iterrows()
        ):

            if (
                str(
                    concept[
                        "subject"
                    ]
                ).strip()
                != subject
            ):
                continue

            rows.append(
                {
                    "video_id":
                        chunk[
                            "video_id"
                        ],

                    "subject":
                        subject,

                    "method":
                        "ocr_only",

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

                    "lsa_score":
                        float(
                            similarity_matrix[
                                chunk_index,
                                concept_index,
                            ]
                        ),
                }
            )

    return pd.DataFrame(
        rows
    )


# =========================================================
# OLD LDA/LSA FUSION
# =========================================================

def fuse_lda_lsa(
    lda,
    lsa,
):
    keys = [
        "video_id",
        "subject",
        "method",
        "chunk_id",
        "start_sec",
        "end_sec",
        "concept",
    ]

    merged = pd.merge(
        lda,
        lsa,
        on=keys,
        how="inner",
        validate="one_to_one",
    )

    if (
        len(merged)
        != len(lda)
        or len(merged)
        != len(lsa)
    ):
        raise RuntimeError(
            "LDA/LSA merge lost rows."
        )

    group_columns = [
        "video_id",
        "chunk_id",
    ]

    merged[
        "lda_norm"
    ] = (
        merged
        .groupby(
            group_columns
        )[
            "lda_score"
        ]
        .transform(
            minmax_normalize
        )
    )

    merged[
        "lsa_norm"
    ] = (
        merged
        .groupby(
            group_columns
        )[
            "lsa_score"
        ]
        .transform(
            minmax_normalize
        )
    )

    merged[
        "final_score"
    ] = (
        LDA_WEIGHT
        * merged[
            "lda_norm"
        ]

        +

        LSA_WEIGHT
        * merged[
            "lsa_norm"
        ]
    )

    merged["rank"] = (
        merged
        .groupby(
            group_columns
        )[
            "final_score"
        ]
        .rank(
            method="first",
            ascending=False,
        )
        .astype(int)
    )

    merged = (
        merged
        .sort_values(
            [
                "video_id",
                "chunk_id",
                "rank",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    for column in [
        "lda_score",
        "lsa_score",
        "lda_norm",
        "lsa_norm",
        "final_score",
    ]:
        merged[
            column
        ] = (
            pd.to_numeric(
                merged[
                    column
                ],
                errors="coerce",
            )
            .astype(float)
        )

    return merged


# =========================================================
# SCORE ONE OCR BRANCH
# =========================================================

def score_branch(
    tag,
    frames,
    concepts,
    result_dir,
):
    print()
    print("=" * 72)
    print(
        f"SCORE BRANCH {tag}"
    )
    print("=" * 72)

    result_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    chunks = (
        build_ocr_chunks(
            frames,
            tag,
        )
    )

    print(
        "OCR-only chunks:",
        len(chunks),
    )

    print(
        "Empty processed chunks:",
        int(
            (
                chunks[
                    "processed_text"
                ]
                .astype(str)
                .str.strip()
                == ""
            ).sum()
        ),
    )

    chunks.to_csv(
        result_dir
        / "ocr_only_chunks.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "Running LDA..."
    )

    lda = build_lda_scores(
        chunks,
        concepts,
    )

    lda.to_csv(
        result_dir
        / "lda_concept_scores.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "LDA rows:",
        len(lda),
    )

    print(
        "Running LSA..."
    )

    lsa = build_lsa_scores(
        chunks,
        concepts,
    )

    lsa.to_csv(
        result_dir
        / "lsa_concept_scores.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "LSA rows:",
        len(lsa),
    )

    fusion = fuse_lda_lsa(
        lda,
        lsa,
    )

    fusion.to_csv(
        result_dir
        / "fusion_scores.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "Fusion rows:",
        len(fusion),
    )

    print(
        f"{tag} SCORE PASS"
    )

    return (
        chunks,
        fusion,
    )


# =========================================================
# BUILD F = MEAN(D_SCORE, E_SCORE)
# =========================================================

def build_f_scores(
    d_fusion,
    e_fusion,
):
    keys = [
        "video_id",
        "subject",
        "method",
        "chunk_id",
        "start_sec",
        "end_sec",
        "concept",
    ]

    merged = pd.merge(
        d_fusion,
        e_fusion,
        on=keys,
        how="outer",
        suffixes=(
            "_d",
            "_e",
        ),
        indicator=True,
        validate="one_to_one",
    )

    mismatch = int(
        (
            merged["_merge"]
            != "both"
        ).sum()
    )

    if mismatch != 0:
        bad = merged[
            merged[
                "_merge"
            ]
            != "both"
        ]

        print(
            bad.head(
                20
            ).to_string(
                index=False
            )
        )

        raise RuntimeError(
            f"D/E score-key mismatch: "
            f"{mismatch}"
        )

    merged = merged.drop(
        columns=[
            "_merge"
        ]
    )

    merged[
        "final_score"
    ] = (
        merged[
            "final_score_d"
        ]
        +

        merged[
            "final_score_e"
        ]
    ) / 2.0

    merged[
        "fusion_strategy"
    ] = FUSION_STRATEGY

    merged["rank"] = (
        merged
        .groupby(
            [
                "video_id",
                "chunk_id",
            ]
        )[
            "final_score"
        ]
        .rank(
            method="first",
            ascending=False,
        )
        .astype(int)
    )

    merged = (
        merged
        .sort_values(
            [
                "video_id",
                "chunk_id",
                "rank",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    return merged


# =========================================================
# MAIN
# =========================================================

def main():
    print("=" * 72)
    print(
        "STEP 10 - BUILD OCR-ONLY D / E / F SCORES"
    )
    print("=" * 72)

    print(
        "Comparison mode: CONTROLLED OCR-ONLY"
    )

    print(
        f"Chunk size: {CHUNK_SECONDS}s"
    )

    print(
        "D/E downstream scoring identical:"
    )

    print(
        "LDA 0.40 + LSA 0.60"
    )

    print(
        "F late fusion:"
    )

    print(
        "F_score = "
        "(D_final_score + E_final_score) / 2"
    )

    print()

    print(
        "NO transcript used."
    )

    print(
        "NO GT used."
    )

    RESULT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    D_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    E_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    F_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # =====================================================
    # INPUT
    # =====================================================

    d_frames = load_ocr(
        D_FILE,
        "D FULL-FRAME CLEANED",
    )

    e_frames = load_ocr(
        E_FILE,
        "E ROI CLEANED",
    )

    audit_frame_alignment(
        d_frames,
        e_frames,
    )

    concepts = (
        load_concepts()
    )

    print()
    print(
        "Concepts:",
        len(concepts),
    )

    print(
        "Concepts by subject:"
    )

    print(
        concepts
        .groupby(
            "subject"
        )
        .size()
        .to_string()
    )

    # =====================================================
    # SCORE D
    # =====================================================

    (
        d_chunks,
        d_fusion,
    ) = score_branch(
        "D_fullframe",
        d_frames,
        concepts,
        D_DIR,
    )

    # =====================================================
    # SCORE E
    # =====================================================

    (
        e_chunks,
        e_fusion,
    ) = score_branch(
        "E_roi",
        e_frames,
        concepts,
        E_DIR,
    )

    # =====================================================
    # CHUNK ALIGNMENT
    # =====================================================

    chunk_keys = [
        "video_id",
        "subject",
        "chunk_id",
        "start_sec",
        "end_sec",
    ]

    d_chunk_keys = set(
        map(
            tuple,
            d_chunks[
                chunk_keys
            ].itertuples(
                index=False,
                name=None,
            ),
        )
    )

    e_chunk_keys = set(
        map(
            tuple,
            e_chunks[
                chunk_keys
            ].itertuples(
                index=False,
                name=None,
            ),
        )
    )

    chunk_mismatch = (
        len(
            d_chunk_keys
            - e_chunk_keys
        )
        +
        len(
            e_chunk_keys
            - d_chunk_keys
        )
    )

    if chunk_mismatch != 0:
        raise RuntimeError(
            "D/E chunk alignment FAILED."
        )

    # =====================================================
    # BUILD F
    # =====================================================

    print()
    print("=" * 72)
    print(
        "BUILD NEW OCR F"
    )
    print("=" * 72)

    f_scores = (
        build_f_scores(
            d_fusion,
            e_fusion,
        )
    )

    f_scores.to_csv(
        F_DIR
        / "fusion_scores.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # =====================================================
    # F FORMULA AUDIT
    # =====================================================

    expected_mean = (
        (
            f_scores[
                "final_score_d"
            ]
            +

            f_scores[
                "final_score_e"
            ]
        )
        / 2.0
    )

    formula_diff = (
        f_scores[
            "final_score"
        ]
        - expected_mean
    ).abs()

    max_formula_error = float(
        formula_diff.max()
    )

    nan_scores = int(
        f_scores[
            [
                "final_score_d",
                "final_score_e",
                "final_score",
            ]
        ]
        .isna()
        .any(
            axis=1
        )
        .sum()
    )

    out_of_range = int(
        (
            (
                f_scores[
                    "final_score"
                ]
                < -1e-9
            )
            |
            (
                f_scores[
                    "final_score"
                ]
                > 1.0
                + 1e-9
            )
        ).sum()
    )

    # =====================================================
    # FINAL AUDIT
    # =====================================================

    checks = [
        {
            "check":
                "D_input_frames",

            "value":
                len(
                    d_frames
                ),

            "expected":
                EXPECTED_FRAMES,

            "pass":
                len(
                    d_frames
                )
                == EXPECTED_FRAMES,
        },

        {
            "check":
                "E_input_frames",

            "value":
                len(
                    e_frames
                ),

            "expected":
                EXPECTED_FRAMES,

            "pass":
                len(
                    e_frames
                )
                == EXPECTED_FRAMES,
        },

        {
            "check":
                "D_E_chunk_key_mismatch",

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
                "D_E_score_rows_equal",

            "value":
                len(
                    d_fusion
                )
                - len(
                    e_fusion
                ),

            "expected":
                0,

            "pass":
                len(
                    d_fusion
                )
                == len(
                    e_fusion
                ),
        },

        {
            "check":
                "F_rows",

            "value":
                len(
                    f_scores
                ),

            "expected":
                len(
                    d_fusion
                ),

            "pass":
                len(
                    f_scores
                )
                == len(
                    d_fusion
                ),
        },

        {
            "check":
                "F_formula_max_abs_error",

            "value":
                max_formula_error,

            "expected":
                0.0,

            "pass":
                max_formula_error
                <= 1e-12,
        },

        {
            "check":
                "F_nan_scores",

            "value":
                nan_scores,

            "expected":
                0,

            "pass":
                nan_scores
                == 0,
        },

        {
            "check":
                "F_out_of_range",

            "value":
                out_of_range,

            "expected":
                0,

            "pass":
                out_of_range
                == 0,
        },
    ]

    audit = pd.DataFrame(
        checks
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

    # =====================================================
    # PRINT
    # =====================================================

    print()
    print("=" * 72)
    print(
        "OCR D / E / F SCORE FINAL AUDIT"
    )
    print("=" * 72)

    print(
        "D frames:",
        len(
            d_frames
        ),
    )

    print(
        "E frames:",
        len(
            e_frames
        ),
    )

    print()

    print(
        "D chunks:",
        len(
            d_chunks
        ),
    )

    print(
        "E chunks:",
        len(
            e_chunks
        ),
    )

    print(
        "Chunk-key mismatch:",
        chunk_mismatch,
    )

    print()

    print(
        "D score rows:",
        len(
            d_fusion
        ),
    )

    print(
        "E score rows:",
        len(
            e_fusion
        ),
    )

    print(
        "F score rows:",
        len(
            f_scores
        ),
    )

    print()

    print(
        "F strategy:",
        FUSION_STRATEGY,
    )

    print(
        "F formula max error:",
        max_formula_error,
    )

    print(
        "F NaN rows:",
        nan_scores,
    )

    print(
        "F out-of-range rows:",
        out_of_range,
    )

    print()
    print(
        "D output:",
        D_DIR,
    )

    print(
        "E output:",
        E_DIR,
    )

    print(
        "F output:",
        F_DIR,
    )

    print(
        "Audit:",
        AUDIT_FILE,
    )

    if all_pass:
        print()
        print(
            "STEP 10 PASS - "
            "OCR D / E / F SCORES COMPLETE"
        )

        print()
        print(
            "NEXT: evaluate OLD OCR D "
            "vs NEW OCR F."
        )

    else:
        print()
        print(
            "STEP 10 FAILED AUDIT"
        )

        print(
            audit.to_string(
                index=False
            )
        )

        raise RuntimeError(
            "Step 10 audit failed."
        )


if __name__ == "__main__":
    main()