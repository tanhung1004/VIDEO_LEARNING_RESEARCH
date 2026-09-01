from pathlib import Path
import argparse
import shutil
import time

import pandas as pd
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
import pytesseract


# =========================================================
# CONFIG
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SUMMARY_DIR = (
    PROJECT_ROOT
    / "module1"
    / "results"
    / "ocr_best_integration"
    / "keyframe_summaries"
)

RESULT_DIR = (
    PROJECT_ROOT
    / "module1"
    / "results"
    / "ocr_best_integration"
)

OUTPUT_FILE = (
    RESULT_DIR
    / "ocr_fullframe_raw.csv"
)

SUMMARY_FILE = (
    RESULT_DIR
    / "ocr_fullframe_raw_summary.csv"
)

DEV_IDS = [f"v{i}" for i in range(1, 41)]

MIN_IMAGE_BYTES = 5000

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
        help="Re-run OCR instead of resuming existing successful/empty rows.",
    )

    return parser.parse_args()


# =========================================================
# HELPERS
# =========================================================

def video_sort_key(video_id):
    try:
        return int(
            str(video_id).strip().lower().replace("v", "")
        )
    except Exception:
        return 999999


def configure_tesseract():
    system_tesseract = shutil.which("tesseract")

    if system_tesseract:
        pytesseract.pytesseract.tesseract_cmd = (
            system_tesseract
        )

        print(
            "Tesseract:",
            system_tesseract
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
                path
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
        and path.stat().st_size > MIN_IMAGE_BYTES
    )


def resolve_image_path(raw_path, video_id, timestamp):
    raw_text = str(
        raw_path or ""
    ).strip()

    if raw_text and raw_text.lower() != "nan":
        path = Path(raw_text)

        if valid_image(path):
            return path

    # Authoritative local keyframe path.
    fallback = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "ocr_best_integration"
        / "keyframes_full"
        / video_id
        / f"{video_id}_{int(timestamp):05d}s.png"
    )

    return fallback


# =========================================================
# LOAD AUTHORITATIVE KEYFRAME MANIFESTS
# =========================================================

def load_all_keyframes():
    all_rows = []

    for video_id in DEV_IDS:
        file = (
            SUMMARY_DIR
            / f"{video_id}.csv"
        )

        if not file.exists():
            raise FileNotFoundError(
                f"Missing keyframe summary: {file}"
            )

        df = pd.read_csv(
            file
        )

        if df.empty:
            raise RuntimeError(
                f"Empty keyframe summary: {file}"
            )

        required = {
            "video_id",
            "subject",
            "frame_id",
            "timestamp_sec",
            "image_file",
            "status",
        }

        missing_columns = (
            required
            - set(df.columns)
        )

        if missing_columns:
            raise ValueError(
                f"{file.name} missing columns: "
                f"{sorted(missing_columns)}"
            )

        df["video_id"] = (
            df["video_id"]
            .astype(str)
            .str.strip()
        )

        df["timestamp_sec"] = pd.to_numeric(
            df["timestamp_sec"],
            errors="coerce",
        )

        df = df.dropna(
            subset=["timestamp_sec"]
        ).copy()

        df["timestamp_sec"] = (
            df["timestamp_sec"]
            .astype(int)
        )

        all_rows.append(
            df
        )

    frames = pd.concat(
        all_rows,
        ignore_index=True,
    )

    frames = frames[
        frames["video_id"].isin(
            DEV_IDS
        )
    ].copy()

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
            columns=["_video_order"]
        )
        .reset_index(drop=True)
    )

    # =====================================================
    # AUTHORITATIVE PRE-OCR AUDIT
    # =====================================================

    duplicates = int(
        frames.duplicated(
            [
                "video_id",
                "timestamp_sec",
            ]
        ).sum()
    )

    if duplicates != 0:
        raise RuntimeError(
            f"Duplicate keyframes: {duplicates}"
        )

    non_success = frames[
        frames["status"]
        .astype(str)
        .str.strip()
        .str.lower()
        != "success"
    ]

    if len(non_success) != 0:
        preview = non_success[
            [
                "video_id",
                "timestamp_sec",
                "status",
            ]
        ].head(20)

        raise RuntimeError(
            "Keyframe stage is not complete.\n"
            + preview.to_string(index=False)
        )

    resolved_paths = []

    invalid_rows = []

    for _, row in frames.iterrows():
        video_id = str(
            row["video_id"]
        ).strip()

        timestamp = int(
            row["timestamp_sec"]
        )

        path = resolve_image_path(
            row["image_file"],
            video_id,
            timestamp,
        )

        resolved_paths.append(
            str(path.resolve())
            if path.exists()
            else str(path)
        )

        if not valid_image(path):
            invalid_rows.append(
                {
                    "video_id":
                        video_id,

                    "timestamp_sec":
                        timestamp,

                    "image_file":
                        str(path),
                }
            )

    frames["resolved_image_file"] = (
        resolved_paths
    )

    if invalid_rows:
        bad = pd.DataFrame(
            invalid_rows
        )

        raise RuntimeError(
            "Invalid/missing PNG before OCR:\n"
            + bad.head(30).to_string(
                index=False
            )
        )

    return frames


# =========================================================
# EXACT FULL-FRAME OCR PREPROCESSING
# =========================================================

def preprocess_image(image_path):
    """
    Same full-frame OCR preprocessing used by collaborator:

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
# CHECKPOINT
# =========================================================

OUTPUT_COLUMNS = [
    "video_id",
    "subject",
    "frame_id",
    "timestamp_sec",
    "image_file",
    "ocr_text",
    "ocr_char_count",
    "ocr_word_count",
    "ocr_status",
]


def save_checkpoint(rows):
    if not rows:
        return

    df = pd.DataFrame(
        rows
    )

    for column in OUTPUT_COLUMNS:
        if column not in df.columns:
            df[column] = ""

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
            columns=["_video_order"]
        )
        .drop_duplicates(
            [
                "video_id",
                "timestamp_sec",
            ],
            keep="last",
        )
        .reset_index(drop=True)
    )

    temp_file = OUTPUT_FILE.with_suffix(
        ".tmp.csv"
    )

    df[
        OUTPUT_COLUMNS
    ].to_csv(
        temp_file,
        index=False,
        encoding="utf-8-sig",
    )

    temp_file.replace(
        OUTPUT_FILE
    )


def load_existing_results():
    if not OUTPUT_FILE.exists():
        return []

    df = pd.read_csv(
        OUTPUT_FILE
    )

    if df.empty:
        return []

    return df.to_dict(
        orient="records"
    )


# =========================================================
# SUMMARY
# =========================================================

def write_summary(result):
    if result.empty:
        return

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
                            x == "success"
                        ).sum()
                    ),
            ),
            ocr_empty=(
                "ocr_status",
                lambda x:
                    int(
                        (
                            x == "empty"
                        ).sum()
                    ),
            ),
            ocr_failed=(
                "ocr_status",
                lambda x:
                    int(
                        (
                            x == "failed"
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
            columns=["_video_order"]
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
        "STEP 04 - FULL-FRAME OCR"
    )
    print("=" * 72)

    print(
        "Method:"
    )

    print(
        "grayscale -> autocontrast -> "
        "contrast 1.8 -> upscale 2x -> sharpen"
    )

    print(
        "Tesseract: lang=eng, OEM=3, PSM=6"
    )

    print()

    configure_tesseract()

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    frames = load_all_keyframes()

    print()
    print(
        "Keyframe audit PASS"
    )

    print(
        "Total valid keyframes:",
        len(frames),
    )

    print(
        "Videos:",
        frames["video_id"].nunique(),
    )

    if args.video is not None:
        target_video = (
            str(args.video)
            .strip()
            .lower()
        )

        if target_video not in DEV_IDS:
            raise ValueError(
                f"Invalid --video {target_video}"
            )

        frames = frames[
            frames["video_id"]
            .eq(target_video)
        ].copy()

        print(
            "Single-video mode:",
            target_video,
        )

        print(
            "Frames to process:",
            len(frames),
        )

    # =====================================================
    # LOAD RESUME CHECKPOINT
    # =====================================================

    existing_rows = (
        []
        if args.force
        else load_existing_results()
    )

    result_map = {}

    for row in existing_rows:
        key = (
            str(
                row.get(
                    "video_id",
                    "",
                )
            ).strip(),
            int(
                float(
                    row.get(
                        "timestamp_sec",
                        -1,
                    )
                )
            ),
        )

        result_map[key] = row

    completed_keys = {
        key
        for key, row
        in result_map.items()
        if str(
            row.get(
                "ocr_status",
                "",
            )
        ).strip().lower()
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
                row["timestamp_sec"]
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
        len(target_keys)
        - already_done
    )

    print()
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
            "FORCE mode: target frames will be reprocessed."
        )

    start_time = time.perf_counter()

    processed_this_run = 0

    # =====================================================
    # OCR LOOP
    # =====================================================

    try:
        for _, row in frames.iterrows():

            video_id = str(
                row["video_id"]
            ).strip()

            timestamp = int(
                row["timestamp_sec"]
            )

            key = (
                video_id,
                timestamp,
            )

            if (
                not args.force
                and key in completed_keys
            ):
                continue

            image_path = Path(
                row[
                    "resolved_image_file"
                ]
            )

            processed_this_run += 1

            current_number = (
                already_done
                + processed_this_run
            )

            print(
                f"[{current_number}/{len(target_keys)}] "
                f"{video_id} - {timestamp}s",
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
                    status = "success"

                    print(
                        f" OK ({char_count} chars)"
                    )

                else:
                    status = "empty"

                    print(
                        " EMPTY"
                    )

            except Exception as error:
                ocr_text = ""
                char_count = 0
                word_count = 0
                status = "failed"

                print(
                    " FAILED:",
                    type(error).__name__,
                    str(error),
                )

            result_map[key] = {
                "video_id":
                    video_id,

                "subject":
                    str(
                        row["subject"]
                    ).strip(),

                "frame_id":
                    row["frame_id"],

                "timestamp_sec":
                    timestamp,

                "image_file":
                    str(
                        image_path.resolve()
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
            "Safe to rerun this same command."
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

    write_summary(
        result
    )

    # Only evaluate rows targeted by this run mode.
    target_result = result[
        result.apply(
            lambda r:
                (
                    str(
                        r["video_id"]
                    ).strip(),
                    int(
                        float(
                            r["timestamp_sec"]
                        )
                    ),
                )
                in target_keys,
            axis=1,
        )
    ].copy()

    success_count = int(
        (
            target_result[
                "ocr_status"
            ]
            == "success"
        ).sum()
    )

    empty_count = int(
        (
            target_result[
                "ocr_status"
            ]
            == "empty"
        ).sum()
    )

    failed_count = int(
        (
            target_result[
                "ocr_status"
            ]
            == "failed"
        ).sum()
    )

    elapsed_min = (
        time.perf_counter()
        - start_time
    ) / 60.0

    print()
    print("=" * 72)
    print(
        "FULL-FRAME OCR FINAL AUDIT"
    )
    print("=" * 72)

    print(
        "Target frames:",
        len(target_keys),
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
        (
            success_count
            + empty_count
            + failed_count
        ),
    )

    print(
        "Runtime this run:",
        f"{elapsed_min:.2f} min",
    )

    print(
        "Output:",
        OUTPUT_FILE,
    )

    print(
        "Summary:",
        SUMMARY_FILE,
    )

    if (
        len(target_result)
        == len(target_keys)
        and failed_count == 0
    ):
        print()
        print(
            "STEP 04 PASS - FULL-FRAME OCR COMPLETE"
        )

    else:
        print()
        print(
            "STEP 04 INCOMPLETE"
        )

        print(
            "Safe to rerun: success/empty frames are skipped; "
            "failed/missing frames are retried."
        )


if __name__ == "__main__":
    main()