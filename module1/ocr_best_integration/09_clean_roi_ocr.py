from pathlib import Path
import importlib.util
import re

import pandas as pd


# =========================================================
# CONFIG
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "module1"
    / "results"
    / "ocr_best_integration"
    / "ocr_roi_raw_v2.csv"
)

FULLFRAME_CLEANER_FILE = (
    PROJECT_ROOT
    / "module1"
    / "ocr_best_integration"
    / "05_clean_fullframe_ocr.py"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ocr_best_integration"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "ocr_roi_cleaned.csv"
)

RESULT_DIR = (
    PROJECT_ROOT
    / "module1"
    / "results"
    / "ocr_best_integration"
)

SUMMARY_FILE = (
    RESULT_DIR
    / "ocr_roi_clean_summary.csv"
)

AUDIT_FILE = (
    RESULT_DIR
    / "ocr_roi_clean_audit.csv"
)

TERM_LOSS_FILE = (
    RESULT_DIR
    / "ocr_roi_cleaning_term_loss.csv"
)

EXPECTED_ROWS = 2398

DEV_IDS = [f"v{i}" for i in range(1, 41)]


# =========================================================
# TECHNICAL TERM QA
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
# LOAD EXACT CLEANER FROM STEP 05
# =========================================================

def load_fullframe_cleaner():
    if not FULLFRAME_CLEANER_FILE.exists():
        raise FileNotFoundError(
            f"Step 05 cleaner missing:\n"
            f"{FULLFRAME_CLEANER_FILE}"
        )

    spec = importlib.util.spec_from_file_location(
        "fullframe_cleaner",
        FULLFRAME_CLEANER_FILE,
    )

    module = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(
        module
    )

    required = [
        "clean_ocr_text",
        "classify_text",
    ]

    for name in required:
        if not hasattr(
            module,
            name,
        ):
            raise AttributeError(
                f"Step 05 cleaner missing function: {name}"
            )

    return module


# =========================================================
# HELPERS
# =========================================================

def video_sort_key(video_id):
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


def normalize_text(value):
    if pd.isna(value):
        return ""

    return re.sub(
        r"\s+",
        " ",
        str(value).lower(),
    ).strip()


def term_present(text, term):
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
# INPUT AUDIT
# =========================================================

def load_and_audit_input():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input not found:\n"
            f"{INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE
    )

    required = {
        "video_id",
        "subject",
        "frame_id",
        "timestamp_sec",
        "image_file",
        "roi_source",
        "ocr_text",
        "ocr_status",
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

    duplicates = int(
        df.duplicated(
            [
                "video_id",
                "timestamp_sec",
            ]
        ).sum()
    )

    failed = int(
        (
            df["ocr_status"]
            .astype(str)
            .str.strip()
            .str.lower()
            == "failed"
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
        "========== ROI RAW INPUT AUDIT =========="
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
        "OCR failed rows:",
        failed,
    )

    print(
        "Missing DEV videos:",
        missing_videos,
    )

    print(
        "Extra videos:",
        extra_videos,
    )

    passed = (
        len(df)
        == EXPECTED_ROWS

        and df[
            "video_id"
        ].nunique()
        == 40

        and duplicates
        == 0

        and failed
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
            "ROI raw input audit FAILED."
        )

    print(
        "ROI RAW INPUT AUDIT PASS"
    )

    return df


# =========================================================
# TECHNICAL TERM RETENTION
# =========================================================

def audit_term_retention(df):
    rows = []

    for _, row in df.iterrows():

        subject = normalize_subject(
            row["subject"]
        )

        terms = CORE_TERMS_BY_SUBJECT.get(
            subject,
            [],
        )

        raw = normalize_text(
            row["ocr_text_raw"]
        )

        clean = normalize_text(
            row["ocr_text_clean"]
        )

        for term in terms:

            if (
                term_present(
                    raw,
                    term,
                )
                and not term_present(
                    clean,
                    term,
                )
            ):

                rows.append(
                    {
                        "video_id":
                            row["video_id"],

                        "subject":
                            row["subject"],

                        "timestamp_sec":
                            row[
                                "timestamp_sec"
                            ],

                        "term":
                            term,

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
# MAIN
# =========================================================

def main():
    print("=" * 72)

    print(
        "STEP 09 - CLEAN ROI OCR"
    )

    print("=" * 72)

    print(
        "Purpose: build cleaned ROI OCR branch E."
    )

    print(
        "Cleaner: EXACT SAME cleaning function as Step 05."
    )

    print(
        "No GT / no concept prediction / no F1 used."
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    cleaner = load_fullframe_cleaner()

    print()
    print(
        "Step 05 cleaner loaded successfully."
    )

    df = load_and_audit_input()

    # =====================================================
    # COPY RAW OCR
    # =====================================================

    df["ocr_text_raw"] = (
        df["ocr_text"]
        .fillna("")
        .astype(str)
    )

    # =====================================================
    # CLEAN WITH EXACT STEP 05 FUNCTION
    # =====================================================

    cleaned_texts = []

    audit_rows = []

    for text in df[
        "ocr_text_raw"
    ]:

        cleaned, audit = (
            cleaner.clean_ocr_text(
                text
            )
        )

        cleaned_texts.append(
            cleaned
        )

        audit_rows.append(
            audit
        )

    df["ocr_text_clean"] = (
        cleaned_texts
    )

    clean_audit = pd.DataFrame(
        audit_rows
    )

    for column in clean_audit.columns:
        df[column] = (
            clean_audit[
                column
            ]
            .fillna(0)
            .astype(int)
        )

    # =====================================================
    # COUNTS
    # =====================================================

    df["raw_word_count"] = (
        df["ocr_text_raw"]
        .apply(
            lambda text:
                len(
                    str(text)
                    .split()
                )
        )
    )

    df["clean_word_count"] = (
        df["ocr_text_clean"]
        .apply(
            lambda text:
                len(
                    str(text)
                    .split()
                )
        )
    )

    df["word_reduction_pct"] = (
        (
            1
            - df[
                "clean_word_count"
            ]
            / df[
                "raw_word_count"
            ].replace(
                0,
                1,
            )
        )
        * 100
    ).round(
        2
    )

    df["clean_status"] = (
        df[
            "ocr_text_clean"
        ]
        .apply(
            cleaner.classify_text
        )
    )

    # =====================================================
    # SORT
    # =====================================================

    df["_video_order"] = (
        df["video_id"]
        .map(video_sort_key)
    )

    df = (
        df
        .sort_values(
            [
                "_video_order",
                "timestamp_sec",
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

    # =====================================================
    # TERM RETENTION QA
    # =====================================================

    term_loss = audit_term_retention(
        df
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
    # OUTPUT
    # =====================================================

    output_columns = [
        "video_id",
        "subject",
        "frame_id",
        "timestamp_sec",
        "image_file",

        "roi_source",
        "context_label",
        "roi_confidence",

        "ocr_status",
        "ocr_char_count",
        "ocr_word_count",

        "ocr_text_raw",
        "ocr_text_clean",

        "raw_word_count",
        "clean_word_count",
        "word_reduction_pct",

        "ui_phrase_removals",
        "ui_cluster_tokens_removed",
        "ocr_corrections",
        "long_numbers_removed",

        "clean_status",
    ]

    output_columns = [
        column
        for column
        in output_columns

        if column
        in df.columns
    ]

    result = df[
        output_columns
    ].copy()

    result.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    # =====================================================
    # SUMMARY
    # =====================================================

    summary = (
        result
        .groupby(
            [
                "video_id",
                "subject",
            ],
            dropna=False,
        )
        .agg(
            total_frames=(
                "frame_id",
                "count",
            ),

            useful_frames=(
                "clean_status",
                lambda x:
                    int(
                        (
                            x
                            == "useful"
                        ).sum()
                    ),
            ),

            empty_frames=(
                "clean_status",
                lambda x:
                    int(
                        (
                            x
                            == "empty"
                        ).sum()
                    ),
            ),

            noise_frames=(
                "clean_status",
                lambda x:
                    int(
                        (
                            x
                            == "noise"
                        ).sum()
                    ),
            ),

            raw_words=(
                "raw_word_count",
                "sum",
            ),

            clean_words=(
                "clean_word_count",
                "sum",
            ),
        )
        .reset_index()
    )

    summary[
        "word_reduction_pct"
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

    summary["_video_order"] = (
        summary["video_id"]
        .map(video_sort_key)
    )

    summary = (
        summary
        .sort_values(
            "_video_order"
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

    summary.to_csv(
        SUMMARY_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    # =====================================================
    # FINAL AUDIT
    # =====================================================

    duplicates = int(
        result.duplicated(
            [
                "video_id",
                "timestamp_sec",
            ]
        ).sum()
    )

    useful_count = int(
        (
            result[
                "clean_status"
            ]
            == "useful"
        ).sum()
    )

    empty_count = int(
        (
            result[
                "clean_status"
            ]
            == "empty"
        ).sum()
    )

    noise_count = int(
        (
            result[
                "clean_status"
            ]
            == "noise"
        ).sum()
    )

    raw_words = int(
        result[
            "raw_word_count"
        ].sum()
    )

    clean_words = int(
        result[
            "clean_word_count"
        ].sum()
    )

    removed_words = (
        raw_words
        - clean_words
    )

    overall_reduction = (
        0.0

        if raw_words == 0

        else (
            1
            - clean_words
            / raw_words
        )
        * 100
    )

    accounted = (
        useful_count
        + empty_count
        + noise_count
    )

    audit = pd.DataFrame(
        [
            {
                "check":
                    "rows",

                "value":
                    len(result),

                "expected":
                    EXPECTED_ROWS,

                "pass":
                    len(result)
                    == EXPECTED_ROWS,
            },

            {
                "check":
                    "videos",

                "value":
                    result[
                        "video_id"
                    ].nunique(),

                "expected":
                    40,

                "pass":
                    result[
                        "video_id"
                    ].nunique()
                    == 40,
            },

            {
                "check":
                    "duplicates",

                "value":
                    duplicates,

                "expected":
                    0,

                "pass":
                    duplicates
                    == 0,
            },

            {
                "check":
                    "status_accounted",

                "value":
                    accounted,

                "expected":
                    EXPECTED_ROWS,

                "pass":
                    accounted
                    == EXPECTED_ROWS,
            },

            {
                "check":
                    "critical_term_losses",

                "value":
                    critical_term_losses,

                "expected":
                    0,

                "pass":
                    critical_term_losses
                    == 0,
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

    # =====================================================
    # PRINT
    # =====================================================

    print()
    print("=" * 72)

    print(
        "CLEAN ROI OCR FINAL AUDIT"
    )

    print("=" * 72)

    print(
        "Rows:",
        len(result),
    )

    print(
        "Videos:",
        result[
            "video_id"
        ].nunique(),
    )

    print(
        "Duplicates:",
        duplicates,
    )

    print()

    print(
        "Useful:",
        useful_count,
    )

    print(
        "Empty:",
        empty_count,
    )

    print(
        "Noise:",
        noise_count,
    )

    print(
        "Rows accounted for:",
        accounted,
    )

    print()

    print(
        "Raw words:",
        raw_words,
    )

    print(
        "Clean words:",
        clean_words,
    )

    print(
        "Words removed:",
        removed_words,
    )

    print(
        "Overall word reduction:",
        f"{overall_reduction:.2f}%",
    )

    print()

    print(
        "Critical technical term losses:",
        critical_term_losses,
    )

    print()

    print(
        "Output:",
        OUTPUT_FILE,
    )

    print(
        "Summary:",
        SUMMARY_FILE,
    )

    print(
        "Audit:",
        AUDIT_FILE,
    )

    print(
        "Term-loss report:",
        TERM_LOSS_FILE,
    )

    if all_pass:

        print()

        print(
            "STEP 09 PASS - "
            "CLEAN ROI OCR E COMPLETE"
        )

    else:

        print()

        print(
            "STEP 09 REVIEW REQUIRED"
        )

        print(
            audit.to_string(
                index=False
            )
        )

        if critical_term_losses > 0:

            print()

            print(
                term_loss[
                    [
                        "video_id",
                        "subject",
                        "timestamp_sec",
                        "term",
                    ]
                ]
                .head(30)
                .to_string(
                    index=False
                )
            )


if __name__ == "__main__":
    main()