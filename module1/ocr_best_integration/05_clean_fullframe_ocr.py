from pathlib import Path
import re
from difflib import SequenceMatcher

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
    / "ocr_fullframe_raw.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ocr_best_integration"
)

OUTPUT_FILE = (
    OUTPUT_DIR
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
    / "ocr_fullframe_clean_summary.csv"
)

AUDIT_FILE = (
    RESULT_DIR
    / "ocr_fullframe_clean_audit.csv"
)

DEV_IDS = [f"v{i}" for i in range(1, 41)]

EXPECTED_TOTAL_FRAMES = 2398


# =========================================================
# CONSERVATIVE CLEANING CONFIG
# =========================================================

# Chỉ xóa các UI phrase có độ tin cậy cao.
HIGH_CONFIDENCE_UI_PATTERNS = [
    # Windows / desktop
    r"\bwindows\s+\d+(?:\.\d+)?\s+x64\b",
    r"\btype\s+here\s+to\s+search\b",
    r"\bquick\s+launch\b",

    # SQL Server / SSMS
    r"\bmicrosoft\s+sql\s+server\s+management\s+studio\b",
    r"\bmicrosoft\s+sol\s+server\s+management\s+studio\b",
    r"\bobject\s+explor\w*\b",
    r"\bobj?ect\s+splor\w*\b",
    r"\blocalhost\\?\w*\b",

    # IntelliJ / IDE UI
    r"\bexternal\s+libraries\b",
    r"\bexternat\s+libraries\b",
    r"\bscratches?\s+and\s+consoles\b",
    r"\bscratch\w*\s+and\s+consol\w*\b",
    r"\bversion\s+control\b",
    r"\bcurrent\s+file\b",
    r"\bcurrentfie\w*\b",
    r"\bproject\s+structure\b",
    r"\binvalidate\s+caches\b",
    r"\bpower\s+save\s+mode\b",

    # VS Code
    r"\bpython:\s*current\s+file\s*\(integrated\s+terminal\)\b",
    r"\bln\s*\d+\s*,?\s*col\s*\d+\b",
    r"\bspaces\s*:\s*\d+\b",
    r"\b\d+\s*spaces\s+(?:of|for)\b",
    r"\bcrlf\b",
    r"\butf[- ]?8\b",

    # Video / website UI
    r"\bsomething\s+went\s+wrong\b",
    r"\brefresh\s+or\s+try\s+again\s+later\b",
    r"\bvideo\s+unavailable\b",
    r"\bplayback\s+error\b",
    r"\blearn\s+more\b",
    r"\bnesoacademy\s*\.\s*org\b",
    r"\bcodewithmosh\s*\.\s*com\b",
    r"\bmosh\s+hamedani\b",
]


# Không xóa từng từ generic riêng lẻ.
# Chỉ dùng để phát hiện một cụm UI/menu có mật độ cao.
GENERIC_UI_TOKENS = {
    "file",
    "edit",
    "view",
    "project",
    "tools",
    "tool",
    "window",
    "windows",
    "help",
    "run",
    "search",
    "source",
    "refactor",
    "navigate",
    "query",
    "debug",
    "terminal",
    "go",
    "format",
    "build",
    "new",
    "open",
    "close",
    "save",
    "find",
    "replace",
    "connect",
}


# Technical vocabulary phải được bảo vệ.
PROTECTED_TECHNICAL_TERMS = {
    # SQL
    "select",
    "from",
    "where",
    "group",
    "by",
    "group by",
    "having",
    "join",
    "left join",
    "right join",
    "inner join",
    "outer join",
    "count",
    "sum",
    "avg",
    "min",
    "max",
    "order",
    "order by",
    "insert",
    "update",
    "delete",
    "create",
    "alter",
    "drop",
    "table",
    "database",
    "query",
    "constraint",
    "primary",
    "foreign",
    "key",

    # Python
    "def",
    "return",
    "import",
    "for",
    "while",
    "range",
    "list",
    "dict",
    "tuple",
    "set",
    "class",

    # Java
    "public",
    "private",
    "protected",
    "static",
    "void",
    "interface",
    "extends",
    "implements",
    "override",
    "method",
    "class",

    # C++
    "pointer",
    "reference",
    "template",
    "constructor",
    "destructor",
    "object",
    "inheritance",
    "variable",
    "int",
    "float",
    "double",
    "char",
    "bool",
    "struct",
    "virtual",
    "namespace",
    "include",
    "new",
    "delete",
}


# Chỉ sửa các OCR corruption có độ tin cậy cao.
OCR_CORRECTIONS = [
    (r"\bgroup\s+5v\b", "group by"),
    (r"\bgroup\s+sv\b", "group by"),
    (r"\bgroup\s+5y\b", "group by"),
    (r"\bcqunt\b", "count"),
    (r"\bcqunt\s*\(\s*id\s*\)", "count(id)"),
    (r"\bseiect\b", "select"),
    (r"\bwbere\b", "where"),
    (r"\bhavlng\b", "having"),
]


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


def normalize_input_text(text):
    if pd.isna(text):
        return ""

    return re.sub(
        r"\s+",
        " ",
        str(text),
    ).strip().lower()


def apply_ocr_corrections(text):
    changed = 0

    for pattern, replacement in OCR_CORRECTIONS:
        text, n = re.subn(
            pattern,
            replacement,
            text,
            flags=re.IGNORECASE,
        )

        changed += n

    return text, changed


def remove_high_confidence_ui(text):
    removed = 0

    for pattern in HIGH_CONFIDENCE_UI_PATTERNS:
        text, n = re.subn(
            pattern,
            " ",
            text,
            flags=re.IGNORECASE,
        )

        removed += n

    return text, removed


def ui_token_match(token):
    if not token:
        return False

    if token in GENERIC_UI_TOKENS:
        return True

    if len(token) < 4:
        return False

    for candidate in GENERIC_UI_TOKENS:
        if len(candidate) < 4:
            continue

        similarity = SequenceMatcher(
            None,
            token,
            candidate,
        ).ratio()

        if similarity >= 0.82:
            return True

    return False


def remove_menu_clusters(text):
    """
    Chỉ xóa một cluster menu khi:
      - có >= 4 UI-like tokens
      - trong window 8 tokens
      - có strong UI anchor
      - KHÔNG chứa protected technical term
    """

    tokens = text.split()

    if not tokens:
        return text, 0

    normalized = [
        re.sub(
            r"[^a-z0-9_]+",
            "",
            token.lower(),
        )
        for token in tokens
    ]

    ui_flags = [
        ui_token_match(token)
        for token in normalized
    ]

    strong_anchors = {
        "file",
        "edit",
        "view",
        "project",
        "window",
        "help",
    }

    remove = [
        False
        for _ in tokens
    ]

    window_size = 8
    min_ui_tokens = 4

    for start in range(
        len(tokens)
    ):
        end = min(
            len(tokens),
            start + window_size,
        )

        window_tokens = (
            normalized[start:end]
        )

        ui_positions = [
            i
            for i, flag
            in enumerate(
                ui_flags[start:end]
            )
            if flag
        ]

        has_anchor = any(
            token in strong_anchors
            for token
            in window_tokens
        )

        protected_in_window = any(
            token in PROTECTED_TECHNICAL_TERMS
            for token
            in window_tokens
        )

        if (
            len(ui_positions)
            >= min_ui_tokens
            and has_anchor
            and not protected_in_window
        ):
            for position in ui_positions:
                remove[
                    start + position
                ] = True

    cleaned_tokens = [
        token
        for i, token
        in enumerate(tokens)
        if not remove[i]
    ]

    return (
        " ".join(cleaned_tokens),
        sum(remove),
    )


def count_meaningful_tokens(text):
    tokens = re.findall(
        r"[a-z0-9_]+",
        text.lower(),
    )

    return sum(
        len(token) >= 2
        for token in tokens
    )


def classify_text(text):
    if not str(text).strip():
        return "empty"

    if count_meaningful_tokens(
        text
    ) > 0:
        return "useful"

    return "noise"


# =========================================================
# MAIN CLEANING FUNCTION
# =========================================================

def clean_ocr_text(text):
    text = normalize_input_text(
        text
    )

    audit = {
        "ui_phrase_removals": 0,
        "ui_cluster_tokens_removed": 0,
        "ocr_corrections": 0,
        "long_numbers_removed": 0,
    }

    if not text:
        return "", audit

    # -----------------------------------------------------
    # 1. Conservative known OCR corrections
    # -----------------------------------------------------

    text, n = apply_ocr_corrections(
        text
    )

    audit[
        "ocr_corrections"
    ] += n

    # -----------------------------------------------------
    # 2. URLs / obvious metadata domains
    # -----------------------------------------------------

    text = re.sub(
        r"https?://\S+",
        " ",
        text,
    )

    text = re.sub(
        r"www\.\S+",
        " ",
        text,
    )

    text = re.sub(
        r"\b[\w.-]+\.(?:com|org|net|io)\b",
        " ",
        text,
    )

    # -----------------------------------------------------
    # 3. High-confidence UI phrases
    # -----------------------------------------------------

    text, n = remove_high_confidence_ui(
        text
    )

    audit[
        "ui_phrase_removals"
    ] += n

    # -----------------------------------------------------
    # 4. Conservative UI menu clusters
    # -----------------------------------------------------

    text, n = remove_menu_clusters(
        text
    )

    audit[
        "ui_cluster_tokens_removed"
    ] += n

    # -----------------------------------------------------
    # 5. Symbol normalization
    #
    # Preserve common programming syntax.
    # -----------------------------------------------------

    text = re.sub(
        r"""[^a-z0-9_+\-*/=<>(){}\[\].,:;'\" ]+""",
        " ",
        text,
    )

    # -----------------------------------------------------
    # 6. Remove very long numeric IDs
    # -----------------------------------------------------

    text, n = re.subn(
        r"\b\d{5,}\b",
        " ",
        text,
    )

    audit[
        "long_numbers_removed"
    ] += n

    # -----------------------------------------------------
    # 7. Repeated symbol garbage
    # -----------------------------------------------------

    text = re.sub(
        r"([=_\-*])\1{2,}",
        " ",
        text,
    )

    # -----------------------------------------------------
    # 8. Standalone symbols
    # -----------------------------------------------------

    text = re.sub(
        r"(?<!\w)[^\w\s](?!\w)",
        " ",
        text,
    )

    # -----------------------------------------------------
    # 9. Whitespace normalization
    # -----------------------------------------------------

    text = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    return text, audit


# =========================================================
# INPUT AUDIT
# =========================================================

def audit_input(df):
    print()
    print(
        "========== INPUT AUDIT =========="
    )

    required_columns = {
        "video_id",
        "subject",
        "frame_id",
        "timestamp_sec",
        "image_file",
        "ocr_text",
        "ocr_status",
    }

    missing_columns = (
        required_columns
        - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing input columns: "
            f"{sorted(missing_columns)}"
        )

    df["video_id"] = (
        df["video_id"]
        .astype(str)
        .str.strip()
    )

    duplicate_count = int(
        df.duplicated(
            [
                "video_id",
                "timestamp_sec",
            ]
        ).sum()
    )

    videos = int(
        df["video_id"]
        .nunique()
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

    missing_dev = sorted(
        set(DEV_IDS)
        - set(df["video_id"])
    )

    extra_videos = sorted(
        set(df["video_id"])
        - set(DEV_IDS)
    )

    print(
        "Rows:",
        len(df),
    )

    print(
        "Videos:",
        videos,
    )

    print(
        "Duplicate video/timestamp:",
        duplicate_count,
    )

    print(
        "OCR failed rows:",
        failed,
    )

    print(
        "Missing DEV videos:",
        missing_dev,
    )

    print(
        "Extra videos:",
        extra_videos,
    )

    checks = {
        "rows_2398":
            len(df)
            == EXPECTED_TOTAL_FRAMES,

        "videos_40":
            videos
            == 40,

        "duplicates_0":
            duplicate_count
            == 0,

        "ocr_failed_0":
            failed
            == 0,

        "missing_dev_0":
            len(missing_dev)
            == 0,

        "extra_video_0":
            len(extra_videos)
            == 0,
    }

    failed_checks = [
        name
        for name, passed
        in checks.items()
        if not passed
    ]

    if failed_checks:
        raise RuntimeError(
            "INPUT AUDIT FAILED: "
            + ", ".join(
                failed_checks
            )
        )

    print(
        "INPUT AUDIT PASS"
    )


# =========================================================
# MAIN
# =========================================================

def main():
    print("=" * 72)
    print(
        "STEP 05 - CLEAN FULL-FRAME OCR"
    )
    print("=" * 72)

    print(
        "Cleaning mode: CONSERVATIVE"
    )

    print(
        "Source: full-frame raw OCR"
    )

    print(
        "No model scoring / no GT used."
    )

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input not found:\n{INPUT_FILE}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.read_csv(
        INPUT_FILE
    )

    audit_input(
        df
    )

    # =====================================================
    # CLEAN
    # =====================================================

    df["ocr_text_raw"] = (
        df["ocr_text"]
        .fillna("")
        .astype(str)
    )

    cleaned_rows = []
    audit_rows = []

    for text in df[
        "ocr_text_raw"
    ]:
        cleaned, audit = clean_ocr_text(
            text
        )

        cleaned_rows.append(
            cleaned
        )

        audit_rows.append(
            audit
        )

    df["ocr_text_clean"] = (
        cleaned_rows
    )

    audit_df = pd.DataFrame(
        audit_rows
    )

    for column in audit_df.columns:
        df[column] = (
            audit_df[column]
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
                    str(text).split()
                )
        )
    )

    df["clean_word_count"] = (
        df["ocr_text_clean"]
        .apply(
            lambda text:
                len(
                    str(text).split()
                )
        )
    )

    df["word_reduction_pct"] = (
        (
            1
            - df["clean_word_count"]
            / df["raw_word_count"]
            .replace(
                0,
                1,
            )
        )
        * 100
    ).round(
        2
    )

    df["clean_status"] = (
        df["ocr_text_clean"]
        .apply(
            classify_text
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
    # OUTPUT
    # =====================================================

    output_columns = [
        "video_id",
        "subject",
        "frame_id",
        "timestamp_sec",
        "image_file",

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
        for column in output_columns
        if column in df.columns
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
    # VIDEO SUMMARY
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
                            x == "useful"
                        ).sum()
                    ),
            ),

            empty_frames=(
                "clean_status",
                lambda x:
                    int(
                        (
                            x == "empty"
                        ).sum()
                    ),
            ),

            noise_frames=(
                "clean_status",
                lambda x:
                    int(
                        (
                            x == "noise"
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

            ui_phrase_removals=(
                "ui_phrase_removals",
                "sum",
            ),

            ui_cluster_tokens_removed=(
                "ui_cluster_tokens_removed",
                "sum",
            ),

            ocr_corrections=(
                "ocr_corrections",
                "sum",
            ),

            long_numbers_removed=(
                "long_numbers_removed",
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
            - summary["clean_words"]
            / summary["raw_words"]
            .replace(
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

    audit_rows_output = [
        {
            "check":
                "input_rows",

            "value":
                len(df),

            "expected":
                EXPECTED_TOTAL_FRAMES,

            "pass":
                len(df)
                == EXPECTED_TOTAL_FRAMES,
        },

        {
            "check":
                "output_rows",

            "value":
                len(result),

            "expected":
                EXPECTED_TOTAL_FRAMES,

            "pass":
                len(result)
                == EXPECTED_TOTAL_FRAMES,
        },

        {
            "check":
                "video_count",

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
                "duplicate_video_timestamp",

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
                (
                    useful_count
                    + empty_count
                    + noise_count
                ),

            "expected":
                EXPECTED_TOTAL_FRAMES,

            "pass":
                (
                    useful_count
                    + empty_count
                    + noise_count
                )
                == EXPECTED_TOTAL_FRAMES,
        },
    ]

    audit_result = pd.DataFrame(
        audit_rows_output
    )

    audit_result.to_csv(
        AUDIT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    all_pass = bool(
        audit_result[
            "pass"
        ].all()
    )

    # =====================================================
    # PRINT
    # =====================================================

    print()
    print("=" * 72)
    print(
        "CLEAN FULL-FRAME OCR FINAL AUDIT"
    )
    print("=" * 72)

    print(
        "Input rows:",
        len(df),
    )

    print(
        "Output rows:",
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
        (
            useful_count
            + empty_count
            + noise_count
        ),
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
        "UI phrase removals:",
        int(
            result[
                "ui_phrase_removals"
            ].sum()
        ),
    )

    print(
        "UI cluster tokens removed:",
        int(
            result[
                "ui_cluster_tokens_removed"
            ].sum()
        ),
    )

    print(
        "OCR corrections:",
        int(
            result[
                "ocr_corrections"
            ].sum()
        ),
    )

    print(
        "Long numbers removed:",
        int(
            result[
                "long_numbers_removed"
            ].sum()
        ),
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

    if all_pass:
        print()
        print(
            "STEP 05 PASS - "
            "FULL-FRAME OCR CLEANING COMPLETE"
        )

    else:
        print()
        print(
            "STEP 05 FAILED AUDIT"
        )

        print(
            audit_result.to_string(
                index=False
            )
        )

        raise RuntimeError(
            "Step 05 audit failed."
        )


if __name__ == "__main__":
    main()
    