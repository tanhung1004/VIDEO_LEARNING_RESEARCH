from pathlib import Path
import shutil

import pandas as pd
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
import pytesseract


# =========================================================
# 1. PATH
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

KEYFRAME_SUMMARY = (
    PROJECT_ROOT
    / "module1"
    / "results"
    / "proposed"
    / "keyframe_summary.csv"
)

RESULT_DIR = (
    PROJECT_ROOT
    / "module1"
    / "results"
    / "proposed"
)

OUTPUT_FILE = (
    RESULT_DIR
    / "ocr_text.csv"
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# 2. FIND TESSERACT AUTOMATICALLY
# =========================================================

def configure_tesseract():
    """
    Tự tìm tesseract.exe.
    Không cần sửa PATH thủ công.
    """

    # Nếu Windows đã nhận PATH
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


    # Các vị trí Windows phổ biến
    possible_paths = [

        Path(
            r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        ),

        Path(
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"
        ),

        Path(
            r"C:\Users\HUNG\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"
        )
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


# =========================================================
# 3. IMAGE PREPROCESSING
# =========================================================

def preprocess_image(image_path):
    """
    Chuẩn hóa keyframe trước OCR:

    1. RGB -> grayscale
    2. autocontrast
    3. tăng contrast
    4. upscale 2x
    5. sharpen nhẹ
    """

    image = Image.open(
        image_path
    )

    # -----------------------------------------------------
    # GRAYSCALE
    # -----------------------------------------------------

    image = ImageOps.grayscale(
        image
    )

    # -----------------------------------------------------
    # AUTO CONTRAST
    # -----------------------------------------------------

    image = ImageOps.autocontrast(
        image
    )

    # -----------------------------------------------------
    # CONTRAST
    # -----------------------------------------------------

    image = ImageEnhance.Contrast(
        image
    ).enhance(
        1.8
    )

    # -----------------------------------------------------
    # UPSCALE
    # -----------------------------------------------------

    image = image.resize(
        (
            image.width * 2,
            image.height * 2
        )
    )

    # -----------------------------------------------------
    # SHARPEN
    # -----------------------------------------------------

    image = image.filter(
        ImageFilter.SHARPEN
    )

    return image


# =========================================================
# 4. CLEAN OCR TEXT
# =========================================================

def clean_ocr_text(text):

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

    # Gom nhiều space thành 1 space
    text = " ".join(
        text.split()
    )

    return text.strip()


# =========================================================
# 5. OCR ONE FRAME
# =========================================================

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
        )
    )

    return clean_ocr_text(
        text
    )


# =========================================================
# 6. MAIN
# =========================================================

def main():

    print(
        "======================================"
    )

    print(
        "MODULE 1 - PROPOSED"
    )

    print(
        "OCR STEP 03 - OCR KEYFRAMES"
    )

    print(
        "======================================"
    )


    # =====================================================
    # CONFIG TESSERACT
    # =====================================================

    configure_tesseract()


    # =====================================================
    # CHECK INPUT
    # =====================================================

    if not KEYFRAME_SUMMARY.exists():

        print(
            "Không tìm thấy:"
        )

        print(
            KEYFRAME_SUMMARY
        )

        return


    # =====================================================
    # READ KEYFRAMES
    # =====================================================

    frames = pd.read_csv(
        KEYFRAME_SUMMARY
    )


    # Chỉ OCR keyframe Step 02 báo success
    frames = frames[
        frames["status"]
        .astype(str)
        .str.lower()
        == "success"
    ].copy()


    frames = frames.reset_index(
        drop=True
    )


    total_frames = len(
        frames
    )


    print(
        "Số keyframe cần OCR:",
        total_frames
    )


    # =====================================================
    # OCR
    # =====================================================

    result_rows = []


    for index, row in (
        frames.iterrows()
    ):

        video_id = str(
            row["video_id"]
        ).strip()


        timestamp = float(
            row["timestamp_sec"]
        )


        image_path = Path(
            str(
                row["image_file"]
            )
        )


        print(
            f"[{index + 1}/{total_frames}] "
            f"{video_id} - "
            f"{timestamp:g}s",
            end=""
        )


        # -------------------------------------------------
        # FILE CHECK
        # -------------------------------------------------

        if not image_path.exists():

            print(
                " FILE NOT FOUND"
            )

            result_rows.append({

                "video_id":
                    video_id,

                "subject":
                    row["subject"],

                "frame_id":
                    row["frame_id"],

                "timestamp_sec":
                    timestamp,

                "image_file":
                    str(
                        image_path
                    ),

                "ocr_text":
                    "",

                "ocr_char_count":
                    0,

                "ocr_word_count":
                    0,

                "ocr_status":
                    "failed"
            })

            continue


        # -------------------------------------------------
        # OCR
        # -------------------------------------------------

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

            print(
                " FAILED:",
                error
            )

            ocr_text = ""

            char_count = 0

            word_count = 0

            status = "failed"


        # -------------------------------------------------
        # SAVE ROW
        # -------------------------------------------------

        result_rows.append({

            "video_id":
                video_id,

            "subject":
                row["subject"],

            "frame_id":
                row["frame_id"],

            "timestamp_sec":
                timestamp,

            "image_file":
                str(
                    image_path
                ),

            "ocr_text":
                ocr_text,

            "ocr_char_count":
                char_count,

            "ocr_word_count":
                word_count,

            "ocr_status":
                status
        })


    # =====================================================
    # CREATE DATAFRAME
    # =====================================================

    result = pd.DataFrame(
        result_rows
    )


    # =====================================================
    # SAVE
    # =====================================================

    result.to_csv(

        OUTPUT_FILE,

        index=False,

        encoding="utf-8-sig"
    )


    # =====================================================
    # SUMMARY
    # =====================================================

    success_count = int(
        (
            result["ocr_status"]
            == "success"
        ).sum()
    )


    empty_count = int(
        (
            result["ocr_status"]
            == "empty"
        ).sum()
    )


    failed_count = int(
        (
            result["ocr_status"]
            == "failed"
        ).sum()
    )


    # =====================================================
    # PREVIEW
    # =====================================================

    print(
        "\n========== OCR PREVIEW =========="
    )


    preview = (
        result[
            result["ocr_status"]
            == "success"
        ]
        .head(15)
    )


    if preview.empty:

        print(
            "Không có OCR text."
        )

    else:

        print(

            preview[
                [
                    "video_id",
                    "timestamp_sec",
                    "ocr_text"
                ]
            ]

            .to_string(
                index=False
            )
        )


    # =====================================================
    # DONE
    # =====================================================

    print(
        "\n======================================"
    )

    print(
        "HOÀN THÀNH OCR STEP 03"
    )

    print(
        "Total:",
        len(result)
    )

    print(
        "OCR Success:",
        success_count
    )

    print(
        "OCR Empty:",
        empty_count
    )

    print(
        "OCR Failed:",
        failed_count
    )

    print(
        "Output:"
    )

    print(
        OUTPUT_FILE
    )

    print(
        "======================================"
    )


if __name__ == "__main__":
    main()