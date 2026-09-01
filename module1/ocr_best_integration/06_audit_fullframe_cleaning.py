from pathlib import Path
import re

import pandas as pd


# =========================================================
# CONFIG
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ocr_best_integration"
    / "ocr_fullframe_cleaned.csv"
)

RESULT_DIR = (
    PROJECT_ROOT
    / "module1"
    / "results"
    / "ocr_best_integration"
)

SUMMARY_FILE = (
    RESULT_DIR
    / "ocr_fullframe_cleaning_qa_summary.csv"
)

TERM_LOSS_FILE = (
    RESULT_DIR
    / "ocr_fullframe_cleaning_term_loss.csv"
)

HIGH_REDUCTION_FILE = (
    RESULT_DIR
    / "ocr_fullframe_cleaning_high_reduction.csv"
)

EMPTY_AFTER_CLEAN_FILE = (
    RESULT_DIR
    / "ocr_fullframe_cleaning_empty_after_clean.csv"
)

DEV_IDS = [f"v{i}" for i in range(1, 41)]

EXPECTED_ROWS = 2398


# =========================================================
# CORE TECHNICAL TERMS
#
# Chỉ dùng các term tương đối đặc trưng kỹ thuật.
# Không dùng generic words như:
# file, new, delete, query, class...
# để tránh false alarm do UI.
# =========================================================

CORE_TERMS_BY_SUBJECT = {
    "sql": [
        "select",
        "where",
        "having",
        "join",
        "group by",
        "order by",
        "primary key",
        "foreign key",
        "constraint",
    ],

    "python": [
        "def",
        "return",
        "import",
        "range",
        "list",
        "dict",
        "tuple",
    ],

    "java": [
        "public",
        "private",
        "protected",
        "static",
        "void",
        "interface",
        "extends",
        "implements",
        "override",
    ],

    "c++": [
        "pointer",
        "reference",
        "template",
        "constructor",
        "destructor",
        "inheritance",
        "virtual",
        "namespace",
        "include",
    ],
}


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


def normalize_text(value):
    if pd.isna(value):
        return ""

    return re.sub(
        r"\s+",
        " ",
        str(value).lower(),
    ).strip()


def term_present(text, term):
    """
    Conservative token/phrase detection.

    Examples:
      select
      group by
      primary key
    """

    escaped = re.escape(
        term.lower()
    )

    pattern = (
        r"(?<![a-z0-9_])"
        + escaped
        + r"(?![a-z0-9_])"
    )

    return bool(
        re.search(
            pattern,
            text.lower(),
        )
    )


# =========================================================
# LOAD + STRUCTURAL AUDIT
# =========================================================

def load_input():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input not found:\n{INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE
    )

    required = {
        "video_id",
        "subject",
        "timestamp_sec",
        "ocr_text_raw",
        "ocr_text_clean",
        "raw_word_count",
        "clean_word_count",
        "word_reduction_pct",
        "clean_status",
    }

    missing_columns = (
        required
        - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing columns: "
            f"{sorted(missing_columns)}"
        )

    df["video_id"] = (
        df["video_id"]
        .astype(str)
        .str.strip()
    )

    df["subject_norm"] = (
        df["subject"]
        .apply(
            normalize_subject
        )
    )

    df["timestamp_sec"] = (
        pd.to_numeric(
            df["timestamp_sec"],
            errors="coerce",
        )
    )

    df["raw_word_count"] = (
        pd.to_numeric(
            df["raw_word_count"],
            errors="coerce",
        )
        .fillna(0)
        .astype(int)
    )

    df["clean_word_count"] = (
        pd.to_numeric(
            df["clean_word_count"],
            errors="coerce",
        )
        .fillna(0)
        .astype(int)
    )

    df["word_reduction_pct"] = (
        pd.to_numeric(
            df["word_reduction_pct"],
            errors="coerce",
        )
        .fillna(0.0)
    )

    return df


# =========================================================
# TECHNICAL TERM RETENTION AUDIT
# =========================================================

def build_term_loss_audit(df):
    rows = []

    for _, row in df.iterrows():
        subject = (
            row["subject_norm"]
        )

        terms = (
            CORE_TERMS_BY_SUBJECT
            .get(
                subject,
                [],
            )
        )

        raw = normalize_text(
            row["ocr_text_raw"]
        )

        clean = normalize_text(
            row["ocr_text_clean"]
        )

        for term in terms:
            raw_has = term_present(
                raw,
                term,
            )

            clean_has = term_present(
                clean,
                term,
            )

            # Critical case:
            # term clearly existed before cleaning
            # but disappeared afterwards.
            if raw_has and not clean_has:
                rows.append(
                    {
                        "video_id":
                            row["video_id"],

                        "subject":
                            row["subject"],

                        "timestamp_sec":
                            int(
                                row[
                                    "timestamp_sec"
                                ]
                            ),

                        "term":
                            term,

                        "raw_word_count":
                            row[
                                "raw_word_count"
                            ],

                        "clean_word_count":
                            row[
                                "clean_word_count"
                            ],

                        "word_reduction_pct":
                            row[
                                "word_reduction_pct"
                            ],

                        "ocr_text_raw":
                            row[
                                "ocr_text_raw"
                            ],

                        "ocr_text_clean":
                            row[
                                "ocr_text_clean"
                            ],
                    }
                )

    return pd.DataFrame(
        rows
    )


# =========================================================
# HIGH REDUCTION DIAGNOSTICS
# =========================================================

def build_high_reduction(df):
    """
    Diagnostic only.

    Focus on frames where substantial text existed before
    cleaning and >=60% of words disappeared.
    """

    out = df[
        (
            df["raw_word_count"]
            >= 10
        )
        &
        (
            df["word_reduction_pct"]
            >= 60.0
        )
    ].copy()

    if out.empty:
        return out

    out["_video_order"] = (
        out["video_id"]
        .map(video_sort_key)
    )

    out = (
        out
        .sort_values(
            [
                "word_reduction_pct",
                "raw_word_count",
                "_video_order",
                "timestamp_sec",
            ],
            ascending=[
                False,
                False,
                True,
                True,
            ],
        )
        .drop(
            columns=[
                "_video_order",
                "subject_norm",
            ],
            errors="ignore",
        )
        .reset_index(
            drop=True
        )
    )

    return out


# =========================================================
# RAW NONEMPTY -> CLEAN EMPTY
# =========================================================

def build_empty_after_clean(df):
    out = df[
        (
            df["raw_word_count"]
            > 0
        )
        &
        (
            df["clean_word_count"]
            == 0
        )
    ].copy()

    if out.empty:
        return out

    out["_video_order"] = (
        out["video_id"]
        .map(video_sort_key)
    )

    out = (
        out
        .sort_values(
            [
                "_video_order",
                "timestamp_sec",
            ]
        )
        .drop(
            columns=[
                "_video_order",
                "subject_norm",
            ],
            errors="ignore",
        )
        .reset_index(
            drop=True
        )
    )

    return out


# =========================================================
# SUBJECT SUMMARY
# =========================================================

def build_subject_summary(df):
    summary = (
        df.groupby(
            "subject",
            dropna=False,
        )
        .agg(
            frames=(
                "video_id",
                "count",
            ),

            raw_words=(
                "raw_word_count",
                "sum",
            ),

            clean_words=(
                "clean_word_count",
                "sum",
            ),

            mean_frame_reduction_pct=(
                "word_reduction_pct",
                "mean",
            ),

            median_frame_reduction_pct=(
                "word_reduction_pct",
                "median",
            ),
        )
        .reset_index()
    )

    summary[
        "overall_word_reduction_pct"
    ] = (
        (
            1
            - summary[
                "clean_words"
            ]
            / summary[
                "raw_words"
            ].replace(
                0,
                1,
            )
        )
        * 100
    ).round(
        2
    )

    summary[
        "mean_frame_reduction_pct"
    ] = (
        summary[
            "mean_frame_reduction_pct"
        ]
        .round(2)
    )

    summary[
        "median_frame_reduction_pct"
    ] = (
        summary[
            "median_frame_reduction_pct"
        ]
        .round(2)
    )

    return summary


# =========================================================
# MAIN
# =========================================================

def main():
    print("=" * 72)
    print(
        "STEP 06 - AUDIT FULL-FRAME OCR CLEANING"
    )
    print("=" * 72)

    print(
        "Purpose: QA only."
    )

    print(
        "No GT / no predictions / no F1 used."
    )

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = load_input()

    # =====================================================
    # STRUCTURAL CHECKS
    # =====================================================

    duplicates = int(
        df.duplicated(
            [
                "video_id",
                "timestamp_sec",
            ]
        ).sum()
    )

    video_count = int(
        df["video_id"]
        .nunique()
    )

    missing_videos = sorted(
        set(DEV_IDS)
        - set(df["video_id"])
    )

    unknown_subjects = sorted(
        set(df["subject_norm"])
        - set(
            CORE_TERMS_BY_SUBJECT
        )
    )

    print()
    print(
        "========== STRUCTURAL AUDIT =========="
    )

    print(
        "Rows:",
        len(df),
    )

    print(
        "Videos:",
        video_count,
    )

    print(
        "Duplicates:",
        duplicates,
    )

    print(
        "Missing videos:",
        missing_videos,
    )

    print(
        "Unknown subjects:",
        unknown_subjects,
    )

    structural_pass = (
        len(df)
        == EXPECTED_ROWS
        and video_count
        == 40
        and duplicates
        == 0
        and len(
            missing_videos
        )
        == 0
        and len(
            unknown_subjects
        )
        == 0
    )

    # =====================================================
    # TERM RETENTION
    # =====================================================

    term_loss = (
        build_term_loss_audit(
            df
        )
    )

    term_loss.to_csv(
        TERM_LOSS_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    critical_term_losses = len(
        term_loss
    )

    # =====================================================
    # HIGH REDUCTION
    # =====================================================

    high_reduction = (
        build_high_reduction(
            df
        )
    )

    high_reduction.to_csv(
        HIGH_REDUCTION_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    # =====================================================
    # EMPTY AFTER CLEAN
    # =====================================================

    empty_after_clean = (
        build_empty_after_clean(
            df
        )
    )

    empty_after_clean.to_csv(
        EMPTY_AFTER_CLEAN_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    # =====================================================
    # SUBJECT SUMMARY
    # =====================================================

    subject_summary = (
        build_subject_summary(
            df
        )
    )

    subject_summary[
        "critical_term_losses"
    ] = 0

    if not term_loss.empty:
        counts = (
            term_loss
            .groupby(
                "subject"
            )
            .size()
            .to_dict()
        )

        subject_summary[
            "critical_term_losses"
        ] = (
            subject_summary[
                "subject"
            ]
            .map(counts)
            .fillna(0)
            .astype(int)
        )

    subject_summary.to_csv(
        SUMMARY_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    # =====================================================
    # FINAL QA DECISION
    # =====================================================

    # Structural integrity MUST pass.
    #
    # Any critical technical term loss triggers REVIEW,
    # because cleaning should not remove core technical terms.
    #
    # High reduction / empty-after-clean are diagnostic only.
    qa_pass = (
        structural_pass
        and critical_term_losses
        == 0
    )

    print()
    print("=" * 72)
    print(
        "FULL-FRAME CLEANING QA RESULTS"
    )
    print("=" * 72)

    print(
        "Structural PASS:",
        structural_pass,
    )

    print(
        "Critical technical term losses:",
        critical_term_losses,
    )

    print(
        "High-reduction frames "
        "(>=10 raw words, >=60% removed):",
        len(
            high_reduction
        ),
    )

    print(
        "Raw-nonempty -> clean-empty frames:",
        len(
            empty_after_clean
        ),
    )

    print()

    print(
        "Subject summary:"
    )

    print(
        subject_summary.to_string(
            index=False
        )
    )

    print()
    print(
        "Term-loss report:",
        TERM_LOSS_FILE,
    )

    print(
        "High-reduction report:",
        HIGH_REDUCTION_FILE,
    )

    print(
        "Empty-after-clean report:",
        EMPTY_AFTER_CLEAN_FILE,
    )

    print(
        "Summary:",
        SUMMARY_FILE,
    )

    if qa_pass:
        print()
        print(
            "STEP 06 PASS - "
            "FULL-FRAME CLEANING QA COMPLETE"
        )

        print(
            "No core technical terms were lost."
        )

    else:
        print()
        print(
            "STEP 06 REVIEW REQUIRED"
        )

        if not structural_pass:
            print(
                "Reason: structural audit failed."
            )

        if critical_term_losses > 0:
            print(
                "Reason: core technical terms "
                "were present in RAW but absent in CLEAN."
            )

            print()

            print(
                term_loss[
                    [
                        "video_id",
                        "subject",
                        "timestamp_sec",
                        "term",
                        "word_reduction_pct",
                    ]
                ]
                .head(30)
                .to_string(
                    index=False
                )
            )

        print()
        print(
            "Do NOT proceed to ROI until reviewed."
        )


if __name__ == "__main__":
    main()