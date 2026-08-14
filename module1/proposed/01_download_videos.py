from pathlib import Path
import pandas as pd
from yt_dlp import YoutubeDL


# =========================================================
# 1. PATH
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

VIDEOS_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "videos.csv"
)

VIDEO_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "videos"
)

RESULT_DIR = (
    PROJECT_ROOT
    / "module1"
    / "results"
    / "proposed"
)

VIDEO_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# 2. DOWNLOAD ONE VIDEO
# =========================================================

def download_video(row):

    video_id = str(
        row["video_id"]
    ).strip()

    subject = str(
        row["subject"]
    ).strip()

    url = str(
        row["url"]
    ).strip()

    print("\n======================================")
    print("Video:", video_id)
    print("Subject:", subject)
    print("URL:", url)

    # File output:
    #
    # v1.mp4
    # v2.mp4
    # ...

    output_template = str(
        VIDEO_DIR
        / f"{video_id}.%(ext)s"
    )

    ydl_opts = {

        # Ưu tiên video MP4 dễ dùng với OpenCV
        "format":
            "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",

        "outtmpl":
            output_template,

        "quiet":
            False,

        "no_warnings":
            False,

        # Nếu file đã tồn tại thì không tải lại
        "overwrites":
            False
    }

    try:

        with YoutubeDL(
            ydl_opts
        ) as ydl:

            ydl.download(
                [url]
            )

        print(
            "OK - Download thành công"
        )

        return {
            "video_id":
                video_id,

            "subject":
                subject,

            "status":
                "success"
        }

    except Exception as error:

        print(
            "FAILED:",
            error
        )

        return {
            "video_id":
                video_id,

            "subject":
                subject,

            "status":
                "failed"
        }


# =========================================================
# 3. MAIN
# =========================================================

def main():

    print("======================================")
    print("MODULE 1 - PROPOSED")
    print("OCR STEP 01 - DOWNLOAD VIDEOS")
    print("======================================")

    if not VIDEOS_FILE.exists():

        print(
            "Không tìm thấy:"
        )

        print(
            VIDEOS_FILE
        )

        return

    videos = pd.read_csv(
        VIDEOS_FILE
    )

    selected = videos[
        videos["status"]
        .astype(str)
        .str.lower()
        == "selected"
    ]

    print(
        "Số video selected:",
        len(selected)
    )

    results = []

    for _, row in selected.iterrows():

        result = download_video(
            row
        )

        results.append(
            result
        )

    # =====================================================
    # 4. SAVE SUMMARY
    # =====================================================

    summary = pd.DataFrame(
        results
    )

    summary_file = (
        RESULT_DIR
        / "download_summary.csv"
    )

    summary.to_csv(
        summary_file,
        index=False,
        encoding="utf-8-sig"
    )

    print("\n======================================")
    print("HOÀN THÀNH OCR STEP 01")
    print("Summary:")
    print(summary_file)
    print("======================================")


if __name__ == "__main__":
    main()