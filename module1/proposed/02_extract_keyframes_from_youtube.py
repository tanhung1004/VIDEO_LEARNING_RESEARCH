from pathlib import Path
import time
import shutil
import pandas as pd

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait


# =========================================================
# CONFIG
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

VIDEOS_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "videos.csv"
)

KEYFRAME_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "keyframes"
)

RESULT_DIR = (
    PROJECT_ROOT
    / "module1"
    / "results"
    / "proposed"
)

OUTPUT_FILE = (
    RESULT_DIR
    / "keyframe_summary.csv"
)

KEYFRAME_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# MỖI 10 GIÂY LẤY 1 FRAME
# =========================================================

FRAME_INTERVAL = 10

# mỗi timestamp thử tối đa 2 lần
MAX_RETRIES = 2

# video load tối đa
WAIT_SECONDS = 20


# =========================================================
# CHROME CHẠY ẨN
# =========================================================

def create_driver():

    options = Options()

    options.add_argument("--headless=new")
    options.add_argument("--window-size=1600,1000")
    options.add_argument("--mute-audio")
    options.add_argument("--disable-notifications")
    options.add_argument("--disable-popup-blocking")
    options.add_argument("--autoplay-policy=no-user-gesture-required")

    driver = webdriver.Chrome(
        options=options
    )

    driver.set_page_load_timeout(35)

    return driver


# =========================================================
# TẠO URL TẠI TIMESTAMP
# =========================================================

def build_timestamp_url(url, timestamp):

    separator = "&" if "?" in url else "?"

    return (
        f"{url}"
        f"{separator}"
        f"t={int(timestamp)}s"
        f"&autoplay=1"
    )


# =========================================================
# WAIT VIDEO READY
# =========================================================

def wait_video_ready(driver):

    def check(d):

        try:

            video = d.find_element(
                By.TAG_NAME,
                "video"
            )

            info = d.execute_script(
                """
                const v = arguments[0];

                return {
                    duration: v.duration,
                    readyState: v.readyState,
                    width: v.videoWidth,
                    height: v.videoHeight,
                    currentTime: v.currentTime
                };
                """,
                video
            )

            if (
                info["duration"]
                and info["duration"] > 0
                and info["readyState"] >= 2
                and info["width"] > 0
                and info["height"] > 0
            ):
                return video

        except Exception:
            pass

        return False

    return WebDriverWait(
        driver,
        WAIT_SECONDS
    ).until(check)


# =========================================================
# CHECK PLAYER ERROR
# =========================================================

def player_has_error(driver):

    try:

        text = (
            driver.find_element(
                By.TAG_NAME,
                "body"
            )
            .text
            .lower()
        )

        bad_text = [
            "something went wrong",
            "video unavailable",
            "playback error",
            "an error occurred",
            "refresh or try again"
        ]

        return any(
            phrase in text
            for phrase in bad_text
        )

    except Exception:

        return False


# =========================================================
# HIDE YOUTUBE UI
# =========================================================

def hide_ui(driver):

    try:

        driver.execute_script(
            """
            [
                '.ytp-chrome-bottom',
                '.ytp-chrome-top',
                '.ytp-gradient-bottom',
                '.ytp-gradient-top',
                '.ytp-pause-overlay',
                '.ytp-ce-element',
                '.ytp-tooltip',
                '.ytp-spinner',
                '.ytp-bezel'
            ].forEach(selector => {

                document
                    .querySelectorAll(selector)
                    .forEach(element => {
                        element.style.display = 'none';
                    });

            });
            """
        )

    except Exception:
        pass


# =========================================================
# LẤY DURATION
# =========================================================

def get_duration(driver, url):

    print("Đang lấy duration...")

    driver.get(url)

    video = wait_video_ready(
        driver
    )

    duration = driver.execute_script(
        """
        return arguments[0].duration;
        """,
        video
    )

    return float(duration)


# =========================================================
# CHỤP 1 TIMESTAMP
# =========================================================

def capture_timestamp(
    driver,
    url,
    timestamp,
    output_file
):

    timestamp_url = build_timestamp_url(
        url,
        timestamp
    )

    for attempt in range(
        1,
        MAX_RETRIES + 1
    ):

        try:

            driver.get(
                timestamp_url
            )

            video = wait_video_ready(
                driver
            )

            if player_has_error(
                driver
            ):
                continue

            # ép về đúng timestamp
            driver.execute_script(
                """
                const v = arguments[0];
                const t = arguments[1];

                v.muted = true;
                v.currentTime = t;
                """,
                video,
                float(timestamp)
            )

            # chờ frame render
            time.sleep(1.5)

            driver.execute_script(
                """
                arguments[0].pause();
                """,
                video
            )

            time.sleep(0.4)

            if player_has_error(
                driver
            ):
                continue

            hide_ui(
                driver
            )

            time.sleep(0.2)

            # lấy lại element sau khi DOM ổn định
            video = driver.find_element(
                By.TAG_NAME,
                "video"
            )

            video.screenshot(
                str(output_file)
            )

            # kiểm tra file ảnh thật
            if (
                output_file.exists()
                and output_file.stat().st_size > 5000
            ):
                return True

        except Exception:

            pass

        print(
            f" retry{attempt}",
            end=""
        )

        time.sleep(1)

    return False


# =========================================================
# XỬ LÝ 1 VIDEO
# =========================================================

def process_video(
    driver,
    video_id,
    subject,
    url
):

    print("\n======================================")
    print("VIDEO:", video_id)
    print("SUBJECT:", subject)
    print("======================================")

    output_dir = (
        KEYFRAME_DIR
        / video_id
    )

    # xóa keyframe cũ
    if output_dir.exists():

        shutil.rmtree(
            output_dir
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # -----------------------------------------------------
    # DURATION
    # -----------------------------------------------------

    duration = get_duration(
        driver,
        url
    )

    print(
        f"Duration: {duration:.1f}s"
    )

    # -----------------------------------------------------
    # TIMESTAMPS
    # -----------------------------------------------------

    timestamps = list(
        range(
            0,
            int(duration),
            FRAME_INTERVAL
        )
    )

    print(
        "Keyframes dự kiến:",
        len(timestamps)
    )

    rows = []

    # -----------------------------------------------------
    # CAPTURE
    # -----------------------------------------------------

    for frame_id, timestamp in enumerate(
        timestamps
    ):

        print(
            f"[{frame_id + 1}/{len(timestamps)}] "
            f"{video_id} - {timestamp}s",
            end=""
        )

        filename = (
            f"{video_id}_"
            f"{timestamp:05d}s.png"
        )

        output_file = (
            output_dir
            / filename
        )

        success = capture_timestamp(
            driver,
            url,
            timestamp,
            output_file
        )

        if success:

            print(" OK")

            status = "success"
            image_file = str(
                output_file
            )

        else:

            print(" FAILED")

            status = "failed"
            image_file = ""

        rows.append({

            "video_id":
                video_id,

            "subject":
                subject,

            "frame_id":
                frame_id,

            "timestamp_sec":
                timestamp,

            "image_file":
                image_file,

            "status":
                status
        })

    return rows


# =========================================================
# MAIN
# =========================================================

def main():

    print("======================================")
    print("MODULE 1 - PROPOSED")
    print("OCR STEP 02")
    print("KEYFRAME EVERY 10 SECONDS")
    print("======================================")

    videos = pd.read_csv(
        VIDEOS_FILE
    )

    videos = videos[
        videos["status"]
        .astype(str)
        .str.lower()
        == "selected"
    ]

    driver = create_driver()

    all_rows = []

    try:

        for _, row in (
            videos.iterrows()
        ):

            video_id = str(
                row["video_id"]
            ).strip()

            subject = str(
                row["subject"]
            ).strip()

            url = str(
                row["url"]
            ).strip()

            try:

                rows = process_video(
                    driver,
                    video_id,
                    subject,
                    url
                )

                all_rows.extend(
                    rows
                )

            except Exception as error:

                print(
                    "\nFAILED VIDEO:",
                    video_id
                )

                print(error)

    finally:

        driver.quit()

    # -----------------------------------------------------
    # SAVE SUMMARY
    # -----------------------------------------------------

    result = pd.DataFrame(
        all_rows
    )

    result.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    # -----------------------------------------------------
    # SUMMARY
    # -----------------------------------------------------

    if not result.empty:

        success = int(
            (
                result["status"]
                == "success"
            ).sum()
        )

        failed = int(
            (
                result["status"]
                == "failed"
            ).sum()
        )

    else:

        success = 0
        failed = 0

    print("\n======================================")
    print("HOÀN THÀNH OCR STEP 02")
    print("Interval:", FRAME_INTERVAL, "seconds")
    print("Success:", success)
    print("Failed:", failed)
    print("Output:")
    print(OUTPUT_FILE)
    print("======================================")


if __name__ == "__main__":
    main()