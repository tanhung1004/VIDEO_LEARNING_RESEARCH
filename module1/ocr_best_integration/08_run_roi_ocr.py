from pathlib import Path
import argparse
import os
import shutil
import time

import pandas as pd
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
import pytesseract


# =========================================================
# CONFIG
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ROI_SUMMARY_FILE = (
    PROJECT_ROOT
    / "module1"
    / "results"
    / "ocr_best_integration"
    / "roi_keyframe_summary.csv"
)

RESULT_DIR = (
    PROJECT_ROOT
    / "module1"
    / "results"
    / "ocr_best_integration"
)

# ---------------------------------------------------------
# LEGACY FILE
#
# File cũ có thể đang bị Windows lock.
# Ta CHỈ ĐỌC file này để resume nếu cần.
# Không ghi đè vào nó nữa.
# ---------------------------------------------------------

LEGACY_OUTPUT_FILE = (
    RESULT_DIR
    / "ocr_roi_raw.csv"
)

# ---------------------------------------------------------
# NEW SAFE OUTPUT
# ---------------------------------------------------------

OUTPUT_FILE = (
    RESULT_DIR
    / "ocr_roi_raw_v2.csv"
)

SUMMARY_FILE = (
    RESULT_DIR
    / "ocr_roi_raw_summary_v2.csv"
)

DEV_IDS = [f"v{i}" for i in range(1, 41)]

EXPECTED_FRAMES = 2398

CHECKPOINT_EVERY = 1


# =========================================================
# CLI
# =========================================================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Optional single video, e.g. --video v10",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run target frames instead of resume.",
    )

    return parser.parse_args()


# =========================================================
# BASIC HELPERS
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


def configure_tesseract():
    system_tesseract = shutil.which(
        "tesseract"
    )

    if system_tesseract:
        pytesseract.pytesseract.tesseract_cmd = (
            system_tesseract
        )

        print(
            "Tesseract:",
            system_tesseract,
        )

        return

    possible_paths = [
        Path(
            r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        ),

        Path(
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"
        ),

        Path(
            r"C:\Users\HUNG\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"
        ),
    ]

    for path in possible_paths:
        if path.exists():
            pytesseract.pytesseract.tesseract_cmd = str(
                path
            )

            print(
                "Tesseract:",
                path,
            )

            return

    raise FileNotFoundError(
        "Không tìm thấy tesseract.exe."
    )


def valid_image(path):
    path = Path(path)

    return (
        path.exists()
        and path.is_file()
        and path.stat().st_size > 0
    )


# =========================================================
# LOAD + AUDIT ROI MANIFEST
# =========================================================

def load_roi_frames():
    if not ROI_SUMMARY_FILE.exists():
        raise FileNotFoundError(
            f"ROI summary not found:\n"
            f"{ROI_SUMMARY_FILE}"
        )

    frames = pd.read_csv(
        ROI_SUMMARY_FILE
    )

    required = {
        "video_id",
        "subject",
        "frame_id",
        "timestamp_sec",
        "image_file",
        "roi_source",
        "status",
    }

    missing_columns = (
        required
        - set(frames.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing ROI columns: "
            f"{sorted(missing_columns)}"
        )

    frames["video_id"] = (
        frames["video_id"]
        .astype(str)
        .str.strip()
    )

    frames["timestamp_sec"] = (
        pd.to_numeric(
            frames["timestamp_sec"],
            errors="coerce",
        )
    )

    frames = frames.dropna(
        subset=[
            "timestamp_sec"
        ]
    ).copy()

    frames["timestamp_sec"] = (
        frames["timestamp_sec"]
        .astype(int)
    )

    duplicates = int(
        frames.duplicated(
            [
                "video_id",
                "timestamp_sec",
            ]
        ).sum()
    )

    non_success = frames[
        frames["status"]
        .astype(str)
        .str.strip()
        .str.lower()
        != "success"
    ]

    missing_videos = sorted(
        set(DEV_IDS)
        - set(
            frames[
                "video_id"
            ]
        )
    )

    extra_videos = sorted(
        set(
            frames[
                "video_id"
            ]
        )
        - set(DEV_IDS)
    )

    invalid_images = []

    resolved_paths = []

    for _, row in frames.iterrows():

        raw_path = str(
            row.get(
                "image_file",
                "",
            )
        ).strip()

        path = Path(
            raw_path
        )

        resolved_paths.append(
            str(
                path.resolve()
            )
            if path.exists()
            else str(path)
        )

        if not valid_image(
            path
        ):
            invalid_images.append(
                {
                    "video_id":
                        row["video_id"],

                    "timestamp_sec":
                        row["timestamp_sec"],

                    "image_file":
                        str(path),
                }
            )

    frames[
        "resolved_image_file"
    ] = resolved_paths

    print()
    print(
        "========== ROI INPUT AUDIT =========="
    )

    print(
        "Rows:",
        len(frames),
    )

    print(
        "Videos:",
        frames[
            "video_id"
        ].nunique(),
    )

    print(
        "Duplicates:",
        duplicates,
    )

    print(
        "Non-success ROI rows:",
        len(
            non_success
        ),
    )

    print(
        "Invalid ROI images:",
        len(
            invalid_images
        ),
    )

    print(
        "Missing DEV videos:",
        missing_videos,
    )

    print(
        "Extra videos:",
        extra_videos,
    )

    checks = {
        "rows":
            len(frames)
            == EXPECTED_FRAMES,

        "videos":
            frames[
                "video_id"
            ].nunique()
            == 40,

        "duplicates":
            duplicates
            == 0,

        "status":
            len(
                non_success
            )
            == 0,

        "images":
            len(
                invalid_images
            )
            == 0,

        "missing":
            len(
                missing_videos
            )
            == 0,

        "extra":
            len(
                extra_videos
            )
            == 0,
    }

    failed_checks = [
        name
        for name, passed
        in checks.items()
        if not passed
    ]

    if failed_checks:

        if invalid_images:
            print()

            print(
                pd.DataFrame(
                    invalid_images
                )
                .head(20)
                .to_string(
                    index=False
                )
            )

        raise RuntimeError(
            "ROI INPUT AUDIT FAILED: "
            + ", ".join(
                failed_checks
            )
        )

    print(
        "ROI INPUT AUDIT PASS"
    )

    frames["_video_order"] = (
        frames["video_id"]
        .map(video_sort_key)
    )

    frames = (
        frames
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

    return frames


# =========================================================
# OCR PREPROCESSING
#
# SAME AS FULL-FRAME OCR STEP 04
# =========================================================

def preprocess_image(image_path):
    """
    Same settings as Step 04:

    1. grayscale
    2. autocontrast
    3. contrast x1.8
    4. upscale 2x
    5. sharpen
    """

    with Image.open(
        image_path
    ) as source:

        image = ImageOps.grayscale(
            source
        )

        image = ImageOps.autocontrast(
            image
        )

        image = ImageEnhance.Contrast(
            image
        ).enhance(
            1.8
        )

        image = image.resize(
            (
                image.width * 2,
                image.height * 2,
            )
        )

        image = image.filter(
            ImageFilter.SHARPEN
        )

        return image


def normalize_ocr_output(text):
    text = str(
        text
    )

    text = text.replace(
        "\n",
        " "
    )

    text = text.replace(
        "\r",
        " "
    )

    text = " ".join(
        text.split()
    )

    return text.strip()


def run_ocr(image_path):
    image = preprocess_image(
        image_path
    )

    text = pytesseract.image_to_string(
        image,
        lang="eng",
        config=(
            "--oem 3 "
            "--psm 6"
        ),
    )

    return normalize_ocr_output(
        text
    )


# =========================================================
# OUTPUT COLUMNS
# =========================================================

OUTPUT_COLUMNS = [
    "video_id",
    "subject",
    "frame_id",
    "timestamp_sec",
    "image_file",
    "roi_source",
    "context_label",
    "roi_confidence",
    "ocr_text",
    "ocr_char_count",
    "ocr_word_count",
    "ocr_status",
]


# =========================================================
# LOAD CHECKPOINT
#
# Priority:
# 1. NEW v2 checkpoint
# 2. OLD legacy checkpoint
# =========================================================

def read_checkpoint_file(path):
    if not path.exists():
        return []

    try:
        df = pd.read_csv(
            path
        )

    except Exception as error:
        print(
            "WARNING: cannot read checkpoint:",
            path.name,
            type(error).__name__,
        )

        return []

    if df.empty:
        return []

    return df.to_dict(
        orient="records"
    )


def load_existing():
    # -----------------------------------------------------
    # Prefer V2 if it already exists
    # -----------------------------------------------------

    if OUTPUT_FILE.exists():

        rows = read_checkpoint_file(
            OUTPUT_FILE
        )

        print(
            "Resume source:",
            OUTPUT_FILE.name,
        )

        print(
            "Rows loaded:",
            len(rows),
        )

        return rows

    # -----------------------------------------------------
    # Otherwise recover from OLD checkpoint
    # -----------------------------------------------------

    if LEGACY_OUTPUT_FILE.exists():

        rows = read_checkpoint_file(
            LEGACY_OUTPUT_FILE
        )

        print(
            "Resume source:",
            LEGACY_OUTPUT_FILE.name,
            "(legacy read-only)"
        )

        print(
            "Rows recovered:",
            len(rows),
        )

        return rows

    print(
        "Resume source: none"
    )

    return []


# =========================================================
# SAFE SAVE CHECKPOINT
# =========================================================

def save_checkpoint(rows):
    if not rows:
        return

    df = pd.DataFrame(
        rows
    )

    for column in OUTPUT_COLUMNS:
        if column not in df.columns:
            df[column] = ""

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
        .drop_duplicates(
            [
                "video_id",
                "timestamp_sec",
            ],
            keep="last",
        )
        .reset_index(
            drop=True
        )
    )

    temp_file = (
        RESULT_DIR
        / "ocr_roi_raw_v2.tmp.csv"
    )

    df[
        OUTPUT_COLUMNS
    ].to_csv(
        temp_file,
        index=False,
        encoding="utf-8-sig",
    )

    try:
        os.replace(
            temp_file,
            OUTPUT_FILE,
        )

    except PermissionError:

        emergency_file = (
            RESULT_DIR
            / "ocr_roi_raw_v2_emergency.csv"
        )

        df[
            OUTPUT_COLUMNS
        ].to_csv(
            emergency_file,
            index=False,
            encoding="utf-8-sig",
        )

        print()
        print(
            "ERROR: ocr_roi_raw_v2.csv is locked."
        )

        print(
            "Emergency checkpoint saved:"
        )

        print(
            emergency_file
        )

        print(
            "Close the CSV in Excel/VS Code "
            "and rerun Step 08."
        )

        raise


# =========================================================
# PRE-FLIGHT WRITE TEST
#
# Catch lock BEFORE spending time OCR-ing.
# =========================================================

def check_output_writable():
    probe = (
        RESULT_DIR
        / "_ocr_roi_write_test.tmp"
    )

    try:
        probe.write_text(
            "write_test",
            encoding="utf-8",
        )

        probe.unlink(
            missing_ok=True
        )

    except Exception as error:
        raise RuntimeError(
            "Cannot write to OCR results directory: "
            f"{error}"
        )

    # If V2 exists, test whether Windows lets us replace it.
    if OUTPUT_FILE.exists():

        test_temp = (
            RESULT_DIR
            / "_ocr_roi_replace_test.tmp"
        )

        backup_bytes = (
            OUTPUT_FILE.read_bytes()
        )

        try:
            test_temp.write_bytes(
                backup_bytes
            )

            os.replace(
                test_temp,
                OUTPUT_FILE,
            )

        except PermissionError:
            test_temp.unlink(
                missing_ok=True
            )

            raise PermissionError(
                "\nocr_roi_raw_v2.csv is currently locked.\n"
                "Close it in Excel / VS Code CSV viewer, "
                "then run Step 08 again."
            )

    print(
        "Output write-lock check: PASS"
    )


# =========================================================
# SUMMARY
# =========================================================

def write_summary(result):
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
                "timestamp_sec",
                "count",
            ),

            ocr_success=(
                "ocr_status",
                lambda x:
                    int(
                        (
                            x
                            == "success"
                        ).sum()
                    ),
            ),

            ocr_empty=(
                "ocr_status",
                lambda x:
                    int(
                        (
                            x
                            == "empty"
                        ).sum()
                    ),
            ),

            ocr_failed=(
                "ocr_status",
                lambda x:
                    int(
                        (
                            x
                            == "failed"
                        ).sum()
                    ),
            ),

            total_chars=(
                "ocr_char_count",
                "sum",
            ),

            total_words=(
                "ocr_word_count",
                "sum",
            ),
        )
        .reset_index()
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


# =========================================================
# MAIN
# =========================================================

def main():
    args = parse_args()

    print("=" * 72)

    print(
        "STEP 08 - ROI OCR"
    )

    print("=" * 72)

    print(
        "OCR preprocessing:"
    )

    print(
        "grayscale -> autocontrast -> "
        "contrast 1.8 -> upscale 2x -> sharpen"
    )

    print(
        "Tesseract: lang=eng, OEM=3, PSM=6"
    )

    print(
        "Same OCR engine/settings as Step 04."
    )

    print()

    print(
        "Legacy checkpoint:"
    )

    print(
        LEGACY_OUTPUT_FILE
    )

    print(
        "New output:"
    )

    print(
        OUTPUT_FILE
    )

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    configure_tesseract()

    # =====================================================
    # INPUT AUDIT
    # =====================================================

    frames = load_roi_frames()

    # =====================================================
    # SINGLE VIDEO MODE
    # =====================================================

    if args.video is not None:

        target_video = (
            str(
                args.video
            )
            .strip()
            .lower()
        )

        if target_video not in DEV_IDS:
            raise ValueError(
                f"Invalid --video: "
                f"{target_video}"
            )

        frames = frames[
            frames["video_id"]
            .eq(
                target_video
            )
        ].copy()

        print()
        print(
            "Single-video mode:",
            target_video,
        )

    # =====================================================
    # CHECK OUTPUT LOCK BEFORE OCR
    # =====================================================

    check_output_writable()

    # =====================================================
    # LOAD CHECKPOINT
    # =====================================================

    existing_rows = (
        []
        if args.force
        else load_existing()
    )

    result_map = {}

    for row in existing_rows:

        try:
            video_id = str(
                row.get(
                    "video_id",
                    "",
                )
            ).strip()

            timestamp = int(
                float(
                    row.get(
                        "timestamp_sec",
                        -1,
                    )
                )
            )

            if (
                video_id
                not in DEV_IDS
                or timestamp
                < 0
            ):
                continue

            key = (
                video_id,
                timestamp,
            )

            result_map[
                key
            ] = row

        except Exception:
            continue

    completed_keys = {
        key
        for key, row
        in result_map.items()

        if str(
            row.get(
                "ocr_status",
                "",
            )
        )
        .strip()
        .lower()
        in {
            "success",
            "empty",
        }
    }

    target_keys = {
        (
            str(
                row["video_id"]
            ).strip(),

            int(
                row[
                    "timestamp_sec"
                ]
            ),
        )

        for _, row
        in frames.iterrows()
    }

    already_done = len(
        target_keys
        & completed_keys
    )

    remaining = (
        len(
            target_keys
        )
        - already_done
    )

    print()
    print(
        "Target ROI frames:",
        len(
            target_keys
        ),
    )

    print(
        "Already completed:",
        already_done,
    )

    print(
        "Remaining:",
        remaining,
    )

    if args.force:
        print(
            "FORCE mode enabled."
        )

    # =====================================================
    # IMPORTANT
    #
    # Immediately copy legacy checkpoint into NEW V2
    # before doing more OCR.
    # =====================================================

    if (
        existing_rows
        and not OUTPUT_FILE.exists()
        and not args.force
    ):

        print()
        print(
            "Migrating legacy checkpoint -> V2..."
        )

        save_checkpoint(
            list(
                result_map.values()
            )
        )

        print(
            "Legacy checkpoint migrated successfully."
        )

    start_time = (
        time.perf_counter()
    )

    processed_this_run = 0

    # =====================================================
    # OCR LOOP
    # =====================================================

    try:

        for _, row in (
            frames.iterrows()
        ):

            video_id = str(
                row["video_id"]
            ).strip()

            timestamp = int(
                row[
                    "timestamp_sec"
                ]
            )

            key = (
                video_id,
                timestamp,
            )

            if (
                not args.force
                and key
                in completed_keys
            ):
                continue

            processed_this_run += 1

            current_number = (
                already_done
                + processed_this_run
            )

            image_path = Path(
                row[
                    "resolved_image_file"
                ]
            )

            print(
                f"[{current_number}/"
                f"{len(target_keys)}] "
                f"{video_id} - "
                f"{timestamp}s",
                end="",
                flush=True,
            )

            try:

                ocr_text = run_ocr(
                    image_path
                )

                char_count = len(
                    ocr_text
                )

                word_count = len(
                    ocr_text.split()
                )

                if char_count > 0:

                    status = (
                        "success"
                    )

                    print(
                        f" OK "
                        f"({char_count} chars)"
                    )

                else:

                    status = (
                        "empty"
                    )

                    print(
                        " EMPTY"
                    )

            except Exception as error:

                ocr_text = ""
                char_count = 0
                word_count = 0

                status = (
                    "failed"
                )

                print(
                    " FAILED:",
                    type(
                        error
                    ).__name__,
                    str(
                        error
                    ),
                )

            result_map[
                key
            ] = {
                "video_id":
                    video_id,

                "subject":
                    str(
                        row[
                            "subject"
                        ]
                    ).strip(),

                "frame_id":
                    row[
                        "frame_id"
                    ],

                "timestamp_sec":
                    timestamp,

                "image_file":
                    str(
                        image_path.resolve()
                    ),

                "roi_source":
                    str(
                        row.get(
                            "roi_source",
                            "",
                        )
                    ),

                "context_label":
                    str(
                        row.get(
                            "context_label",
                            "",
                        )
                    ),

                "roi_confidence":
                    str(
                        row.get(
                            "roi_confidence",
                            "",
                        )
                    ),

                "ocr_text":
                    ocr_text,

                "ocr_char_count":
                    char_count,

                "ocr_word_count":
                    word_count,

                "ocr_status":
                    status,
            }

            if (
                processed_this_run
                % CHECKPOINT_EVERY
                == 0
            ):

                save_checkpoint(
                    list(
                        result_map.values()
                    )
                )

    except KeyboardInterrupt:

        print()

        print(
            "Interrupted by user."
        )

        print(
            "Saving checkpoint..."
        )

        save_checkpoint(
            list(
                result_map.values()
            )
        )

        print(
            "Checkpoint saved."
        )

        print(
            "Safe to rerun Step 08."
        )

        return

    # =====================================================
    # FINAL SAVE
    # =====================================================

    save_checkpoint(
        list(
            result_map.values()
        )
    )

    result = pd.read_csv(
        OUTPUT_FILE
    )

    # =====================================================
    # ENSURE NUMERIC
    # =====================================================

    result[
        "timestamp_sec"
    ] = pd.to_numeric(
        result[
            "timestamp_sec"
        ],
        errors="coerce",
    )

    result = result.dropna(
        subset=[
            "timestamp_sec"
        ]
    ).copy()

    result[
        "timestamp_sec"
    ] = (
        result[
            "timestamp_sec"
        ]
        .astype(int)
    )

    result[
        "ocr_char_count"
    ] = pd.to_numeric(
        result[
            "ocr_char_count"
        ],
        errors="coerce",
    ).fillna(0)

    result[
        "ocr_word_count"
    ] = pd.to_numeric(
        result[
            "ocr_word_count"
        ],
        errors="coerce",
    ).fillna(0)

    write_summary(
        result
    )

    # =====================================================
    # TARGET RESULT
    # =====================================================

    mask = result.apply(
        lambda row:
            (
                str(
                    row[
                        "video_id"
                    ]
                ).strip(),

                int(
                    row[
                        "timestamp_sec"
                    ]
                ),
            )
            in target_keys,

        axis=1,
    )

    target_result = (
        result[
            mask
        ]
        .copy()
    )

    success_count = int(
        (
            target_result[
                "ocr_status"
            ]
            .astype(str)
            .str.lower()
            == "success"
        ).sum()
    )

    empty_count = int(
        (
            target_result[
                "ocr_status"
            ]
            .astype(str)
            .str.lower()
            == "empty"
        ).sum()
    )

    failed_count = int(
        (
            target_result[
                "ocr_status"
            ]
            .astype(str)
            .str.lower()
            == "failed"
        ).sum()
    )

    accounted = (
        success_count
        + empty_count
        + failed_count
    )

    duplicates = int(
        target_result.duplicated(
            [
                "video_id",
                "timestamp_sec",
            ]
        ).sum()
    )

    elapsed_minutes = (
        time.perf_counter()
        - start_time
    ) / 60.0

    # =====================================================
    # FINAL AUDIT
    # =====================================================

    print()
    print("=" * 72)

    print(
        "ROI OCR FINAL AUDIT"
    )

    print("=" * 72)

    print(
        "Target frames:",
        len(
            target_keys
        ),
    )

    print(
        "OCR Success:",
        success_count,
    )

    print(
        "OCR Empty:",
        empty_count,
    )

    print(
        "OCR Failed:",
        failed_count,
    )

    print(
        "Rows accounted for:",
        accounted,
    )

    print(
        "Duplicates:",
        duplicates,
    )

    print(
        "Runtime this run:",
        f"{elapsed_minutes:.2f} min",
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

    pass_check = (
        len(
            target_result
        )
        == len(
            target_keys
        )

        and accounted
        == len(
            target_keys
        )

        and failed_count
        == 0

        and duplicates
        == 0
    )

    if pass_check:

        print()

        print(
            "STEP 08 PASS - "
            "ROI OCR COMPLETE"
        )

    else:

        print()

        print(
            "STEP 08 INCOMPLETE"
        )

        print(
            "Safe to rerun: "
            "success/empty rows are skipped; "
            "failed/missing rows are retried."
        )


if __name__ == "__main__":
    main()