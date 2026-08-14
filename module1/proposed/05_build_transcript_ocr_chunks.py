from pathlib import Path
import sys
import re

import pandas as pd


# =========================================================
# 1. PROJECT PATH
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SRC_DIR = (
    PROJECT_ROOT
    / "module1"
    / "src"
)

sys.path.insert(
    0,
    str(SRC_DIR)
)

from preprocessing import preprocess_text


# =========================================================
# 2. INPUT
# =========================================================

TRANSCRIPT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "transcript"
    / "transcript_chunks.csv"
)

OCR_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ocr"
    / "ocr_cleaned.csv"
)


# =========================================================
# 3. OUTPUT
# =========================================================

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "transcript_ocr"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "transcript_ocr_chunks.csv"
)

RESULT_DIR = (
    PROJECT_ROOT
    / "module1"
    / "results"
    / "proposed"
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

SUMMARY_FILE = (
    RESULT_DIR
    / "transcript_ocr_combine_summary.csv"
)


# =========================================================
# 4. CONFIG
# =========================================================

# Hai OCR frame có >= 85% token giống nhau
# thì xem là gần-trùng.
#
# Đây chỉ là rule chống lặp OCR.
# KHÔNG liên quan ground truth/concept.
OCR_DUPLICATE_THRESHOLD = 0.85


# =========================================================
# 5. NORMALIZE TOKENS
# =========================================================

def get_token_set(text):

    text = str(text).lower()

    tokens = re.findall(
        r"[a-z0-9_]+",
        text
    )

    return set(tokens)


# =========================================================
# 6. JACCARD SIMILARITY
# =========================================================

def jaccard_similarity(
    text_a,
    text_b
):

    set_a = get_token_set(
        text_a
    )

    set_b = get_token_set(
        text_b
    )

    if not set_a or not set_b:
        return 0.0

    intersection = len(
        set_a & set_b
    )

    union = len(
        set_a | set_b
    )

    if union == 0:
        return 0.0

    return (
        intersection
        / union
    )


# =========================================================
# 7. REMOVE NEAR-DUPLICATE OCR FRAMES
# =========================================================

def remove_duplicate_ocr(
    ocr_rows
):

    kept_rows = []

    for _, row in (
        ocr_rows.iterrows()
    ):

        text = str(
            row["ocr_text_clean"]
        ).strip()

        if not text:
            continue

        duplicate = False

        for kept in kept_rows:

            similarity = (
                jaccard_similarity(
                    text,
                    kept["ocr_text_clean"]
                )
            )

            if (
                similarity
                >= OCR_DUPLICATE_THRESHOLD
            ):

                duplicate = True
                break

        if not duplicate:

            kept_rows.append(
                row.to_dict()
            )

    return kept_rows


# =========================================================
# 8. MAIN
# =========================================================

def main():

    print(
        "======================================"
    )
    print(
        "MODULE 1 - PROPOSED"
    )
    print(
        "STEP 05 - BUILD TRANSCRIPT + OCR CHUNKS"
    )
    print(
        "======================================"
    )


    # =====================================================
    # CHECK INPUT
    # =====================================================

    if not TRANSCRIPT_FILE.exists():

        print(
            "Không tìm thấy transcript:"
        )

        print(
            TRANSCRIPT_FILE
        )

        return


    if not OCR_FILE.exists():

        print(
            "Không tìm thấy OCR:"
        )

        print(
            OCR_FILE
        )

        return


    # =====================================================
    # READ
    # =====================================================

    transcript = pd.read_csv(
        TRANSCRIPT_FILE
    )

    ocr = pd.read_csv(
        OCR_FILE
    )


    # =====================================================
    # CLEAN TYPES
    # =====================================================

    transcript["video_id"] = (
        transcript["video_id"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    ocr["video_id"] = (
        ocr["video_id"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    ocr["timestamp_sec"] = (
        pd.to_numeric(
            ocr["timestamp_sec"],
            errors="coerce"
        )
    )


    # Chỉ dùng OCR Step 04 đánh dấu useful
    ocr = ocr[
        ocr["clean_status"]
        .astype(str)
        .str.lower()
        == "useful"
    ].copy()


    print(
        "Transcript chunks:",
        len(transcript)
    )

    print(
        "Useful OCR frames:",
        len(ocr)
    )


    # =====================================================
    # BUILD EACH CHUNK
    # =====================================================

    output_rows = []

    for index, chunk in (
        transcript.iterrows()
    ):

        video_id = str(
            chunk["video_id"]
        )

        subject = str(
            chunk["subject"]
        )

        chunk_id = int(
            chunk["chunk_id"]
        )

        start_sec = float(
            chunk["start_sec"]
        )

        end_sec = float(
            chunk["end_sec"]
        )


        # =================================================
        # OCR NẰM TRONG CHUNK NÀY
        #
        # start <= timestamp < end
        # =================================================

        chunk_ocr = ocr[
            (
                ocr["video_id"]
                == video_id
            )
            &
            (
                ocr["timestamp_sec"]
                >= start_sec
            )
            &
            (
                ocr["timestamp_sec"]
                < end_sec
            )
        ].copy()


        chunk_ocr = (
            chunk_ocr
            .sort_values(
                "timestamp_sec"
            )
        )


        total_ocr_frames = len(
            chunk_ocr
        )


        # =================================================
        # REMOVE NEAR DUPLICATES
        # =================================================

        kept_ocr = (
            remove_duplicate_ocr(
                chunk_ocr
            )
        )


        kept_ocr_count = len(
            kept_ocr
        )


        # =================================================
        # BUILD OCR TEXT
        # =================================================

        ocr_texts = [

            str(
                row[
                    "ocr_text_clean"
                ]
            ).strip()

            for row in kept_ocr

            if str(
                row[
                    "ocr_text_clean"
                ]
            ).strip()
        ]


        combined_ocr_text = (
            " ".join(
                ocr_texts
            )
        )


        # =================================================
        # TRANSCRIPT TEXT
        # =================================================

        transcript_text = str(
            chunk["raw_text"]
        )


        # =================================================
        # TRANSCRIPT + OCR
        # =================================================

        if combined_ocr_text:

            combined_raw_text = (
                transcript_text
                + " "
                + combined_ocr_text
            )

        else:

            combined_raw_text = (
                transcript_text
            )


        # =================================================
        # SAME PREPROCESSING AS BASELINE
        # =================================================

        processed_text = (
            preprocess_text(
                combined_raw_text
            )
        )


        # =================================================
        # OCR TIMESTAMPS USED
        # =================================================

        timestamps_used = [

            str(
                int(
                    float(
                        row[
                            "timestamp_sec"
                        ]
                    )
                )
            )

            for row in kept_ocr
        ]


        # =================================================
        # SAVE
        # =================================================

        output_rows.append({

            "video_id":
                video_id,

            "subject":
                subject,

            "method":
                "transcript_ocr",

            "chunk_id":
                chunk_id,

            "start_sec":
                start_sec,

            "end_sec":
                end_sec,

            "transcript_text":
                transcript_text,

            "ocr_text":
                combined_ocr_text,

            "ocr_frames_total":
                total_ocr_frames,

            "ocr_frames_kept":
                kept_ocr_count,

            "ocr_timestamps":
                ";".join(
                    timestamps_used
                ),

            # giữ tên giống baseline
            # để LDA / LSA dùng dễ
            "raw_text":
                combined_raw_text,

            "processed_text":
                processed_text
        })


        print(
            f"[{index + 1}/{len(transcript)}] "
            f"{video_id} chunk {chunk_id} | "
            f"OCR {total_ocr_frames}"
            f" -> {kept_ocr_count}"
        )


    # =====================================================
    # DATAFRAME
    # =====================================================

    result = pd.DataFrame(
        output_rows
    )


    # =====================================================
    # SAVE CHUNKS
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

            num_chunks=(
                "chunk_id",
                "count"
            ),

            ocr_frames_total=(
                "ocr_frames_total",
                "sum"
            ),

            ocr_frames_kept=(
                "ocr_frames_kept",
                "sum"
            ),

            chunks_with_ocr=(
                "ocr_text",
                lambda x:
                    int(
                        (
                            x.astype(str)
                            .str.len()
                            > 0
                        ).sum()
                    )
            )
        )
        .reset_index()
    )


    summary["ocr_frames_removed"] = (

        summary[
            "ocr_frames_total"
        ]

        -

        summary[
            "ocr_frames_kept"
        ]
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
        "\n========== TRANSCRIPT + OCR PREVIEW =========="
    )

    print(

        result[
            [
                "video_id",
                "chunk_id",
                "start_sec",
                "end_sec",
                "ocr_frames_total",
                "ocr_frames_kept"
            ]
        ]
        .head(20)
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
        "HOÀN THÀNH STEP 05"
    )

    print(
        "Output:"
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