from pathlib import Path
import re
import pandas as pd


# =========================================================
# 1. PATH
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "module1"
    / "results"
    / "proposed"
    / "ocr_text.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ocr"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "ocr_cleaned.csv"
)

SUMMARY_FILE = (
    PROJECT_ROOT
    / "module1"
    / "results"
    / "proposed"
    / "ocr_clean_summary.csv"
)


# =========================================================
# 2. OCR-SPECIFIC UI NOISE
# =========================================================
#
# CHỈ xóa những cụm giao diện khá chắc chắn là noise.
#
# Không xóa:
# group by
# having
# class
# interface
# range
# loop
# join
# select
# ...
#
# để không làm mất learning concepts.
# =========================================================

NOISE_PATTERNS = [

    # Windows / desktop
    r"\bwindows\s+10\s+x64\b",
    r"\btype here to search\b",
    r"\bdesktop[-\w]*\b",

    # SQL Server Management Studio UI
    r"\bmicrosoft\s+(?:sql|sol)\s+server\s+management\s+studio\b",
    r"\bobject\s+explorer\b",
    r"\bsolution\s*\d*\b",
    r"\bconnected\b",
    r"\blocalhost\\?\w*\b",

    # IDE menus
    r"\bfile\s+edit\s+view\s+project\s+tools?\s+window\s+help\b",
    r"\bfile\s+edit\s+source\s+refactor\s+navigate\s+search\s+project\s+run\s+window\s+help\b",

    # VS Code / Python UI
    r"\bpython:\s*current\s+file\s*\(integrated\s+terminal\)\b",
    r"\binstalling\s+packages\b",
    r"\bln\s*\d+\s*,?\s*col\s*\d+\b",
    r"\bspaces\s*:\s*\d+\b",
    r"\butf[- ]?8\b",

    # IntelliJ UI
    r"\bexternal\s+libraries\b",
    r"\bscratches\s+and\s+consoles\b",
    r"\bversion\s+control\b",
    r"\bcurrent\s+file\b",

    # Website / video branding
    r"\bnesoacademy\.org\b",
    r"\bcodewithmosh\.com\b",
    r"\bmosh\s+hamedani\b",

    # YouTube error/UI nếu còn lọt frame lỗi
    r"\bsomething\s+went\s+wrong\b",
    r"\brefresh\s+or\s+try\s+again\s+later\b",
    r"\blearn\s+more\b",
    r"\bvideo\s+unavailable\b",
    r"\bplayback\s+error\b",
]


# =========================================================
# 3. CLEAN OCR TEXT
# =========================================================

def clean_ocr_text(text):

    if pd.isna(text):
        return ""

    text = str(text)

    # -----------------------------------------------------
    # LOWERCASE
    # -----------------------------------------------------

    text = text.lower()


    # -----------------------------------------------------
    # REMOVE URL
    # -----------------------------------------------------

    text = re.sub(
        r"https?://\S+",
        " ",
        text
    )

    text = re.sub(
        r"www\.\S+",
        " ",
        text
    )


    # -----------------------------------------------------
    # REMOVE DOMAIN
    # -----------------------------------------------------

    text = re.sub(
        r"\b[\w.-]+\.(?:com|org|net|io)\b",
        " ",
        text
    )


    # -----------------------------------------------------
    # REMOVE KNOWN UI NOISE
    # -----------------------------------------------------

    for pattern in NOISE_PATTERNS:

        text = re.sub(
            pattern,
            " ",
            text,
            flags=re.IGNORECASE
        )


    # -----------------------------------------------------
    # REMOVE COMMON SYMBOL NOISE
    #
    # Vẫn giữ:
    # _
    # +
    # -
    # *
    # /
    # =
    # <
    # >
    # ()
    # {}
    # []
    #
    # vì đây có thể là code.
    # -----------------------------------------------------

    text = re.sub(
        r"[^a-z0-9_+\-*/=<>(){}\[\].,:;'\" ]+",
        " ",
        text
    )


    # -----------------------------------------------------
    # REMOVE VERY LONG RANDOM NUMBERS
    #
    # ví dụ OCR đọc nhầm ID/machine number.
    #
    # Không xóa số nhỏ vì code có thể có:
    # range(5)
    # count(10)
    # -----------------------------------------------------

    text = re.sub(
        r"\b\d{5,}\b",
        " ",
        text
    )


    # -----------------------------------------------------
    # REMOVE REPEATED SYMBOL GARBAGE
    # -----------------------------------------------------

    text = re.sub(
        r"([=_\-*])\1{2,}",
        " ",
        text
    )


    # -----------------------------------------------------
    # REMOVE STANDALONE OCR SYMBOL/TOKEN GARBAGE
    #
    # Chỉ xóa token 1 ký tự không phải số/chữ hữu ích.
    # -----------------------------------------------------

    text = re.sub(
        r"(?<!\w)[^\w\s](?!\w)",
        " ",
        text
    )


    # -----------------------------------------------------
    # NORMALIZE WHITESPACE
    # -----------------------------------------------------

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# =========================================================
# 4. CHECK WHETHER CLEANED TEXT IS USEFUL
# =========================================================

def classify_text(text):

    if not text:
        return "empty"

    words = text.split()

    # Nếu OCR chỉ còn 1–2 token rất ngắn
    # thì xem là noise.
    meaningful_words = [
        word
        for word in words
        if len(
            re.sub(
                r"[^a-z0-9]",
                "",
                word
            )
        ) >= 2
    ]

    if len(meaningful_words) == 0:
        return "noise"

    return "useful"


# =========================================================
# 5. MAIN
# =========================================================

def main():

    print(
        "======================================"
    )
    print(
        "MODULE 1 - PROPOSED"
    )
    print(
        "STEP 04 - CLEAN OCR TEXT"
    )
    print(
        "======================================"
    )


    # =====================================================
    # CHECK INPUT
    # =====================================================

    if not INPUT_FILE.exists():

        print(
            "Không tìm thấy:"
        )
        print(
            INPUT_FILE
        )

        return


    # =====================================================
    # READ OCR
    # =====================================================

    df = pd.read_csv(
        INPUT_FILE
    )

    print(
        "OCR rows:",
        len(df)
    )


    # =====================================================
    # KEEP RAW TEXT
    # =====================================================

    df["ocr_text_raw"] = (
        df["ocr_text"]
        .fillna("")
        .astype(str)
    )


    # =====================================================
    # CLEAN
    # =====================================================

    df["ocr_text_clean"] = (
        df["ocr_text_raw"]
        .apply(
            clean_ocr_text
        )
    )


    # =====================================================
    # COUNT BEFORE / AFTER
    # =====================================================

    df["raw_word_count"] = (
        df["ocr_text_raw"]
        .apply(
            lambda x:
                len(
                    str(x).split()
                )
        )
    )

    df["clean_word_count"] = (
        df["ocr_text_clean"]
        .apply(
            lambda x:
                len(
                    str(x).split()
                )
        )
    )


    # =====================================================
    # STATUS
    # =====================================================

    df["clean_status"] = (
        df["ocr_text_clean"]
        .apply(
            classify_text
        )
    )


    # =====================================================
    # CHOOSE OUTPUT COLUMNS
    # =====================================================

    output_columns = [

        "video_id",
        "subject",
        "frame_id",
        "timestamp_sec",
        "image_file",

        "ocr_text_raw",
        "ocr_text_clean",

        "raw_word_count",
        "clean_word_count",

        "clean_status"
    ]


    result = df[
        output_columns
    ].copy()


    # =====================================================
    # SAVE CLEAN OCR
    # =====================================================

    result.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )


    # =====================================================
    # SUMMARY PER VIDEO
    # =====================================================

    summary = (
        result
        .groupby(
            [
                "video_id",
                "subject"
            ]
        )
        .agg(

            total_frames=(
                "frame_id",
                "count"
            ),

            useful_frames=(
                "clean_status",
                lambda x:
                    int(
                        (
                            x == "useful"
                        ).sum()
                    )
            ),

            raw_words=(
                "raw_word_count",
                "sum"
            ),

            clean_words=(
                "clean_word_count",
                "sum"
            )
        )
        .reset_index()
    )


    summary.to_csv(
        SUMMARY_FILE,
        index=False,
        encoding="utf-8-sig"
    )


    # =====================================================
    # PREVIEW
    # =====================================================

    print(
        "\n========== CLEAN OCR PREVIEW =========="
    )

    useful = result[
        result["clean_status"]
        == "useful"
    ]

    print(
        useful[
            [
                "video_id",
                "timestamp_sec",
                "ocr_text_clean"
            ]
        ]
        .head(15)
        .to_string(
            index=False
        )
    )


    print(
        "\n========== SUMMARY =========="
    )

    print(
        summary.to_string(
            index=False
        )
    )


    print(
        "\n======================================"
    )
    print(
        "HOÀN THÀNH STEP 04"
    )

    print(
        "Clean OCR:"
    )
    print(
        OUTPUT_FILE
    )

    print(
        "\nSummary:"
    )
    print(
        SUMMARY_FILE
    )

    print(
        "======================================"
    )


if __name__ == "__main__":
    main()