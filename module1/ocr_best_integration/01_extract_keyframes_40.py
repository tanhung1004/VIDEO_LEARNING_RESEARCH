from pathlib import Path
import argparse
import time

import pandas as pd

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

VIDEOS_FILE = PROJECT_ROOT / "data" / "raw" / "videos.csv"

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ocr_best_integration"
)

KEYFRAME_DIR = DATA_DIR / "keyframes_full"

RESULT_DIR = (
    PROJECT_ROOT
    / "module1"
    / "results"
    / "ocr_best_integration"
)

PER_VIDEO_SUMMARY_DIR = RESULT_DIR / "keyframe_summaries"
OUTPUT_FILE = RESULT_DIR / "keyframe_summary.csv"


# =========================================================
# FROZEN SCIENTIFIC CONFIG
# =========================================================

DEV_IDS = [f"v{i}" for i in range(1, 41)]

# Giữ nguyên đúng 1 frame / 10 giây.
FRAME_INTERVAL = 10

MIN_VALID_IMAGE_BYTES = 5000

PAGE_LOAD_TIMEOUT = 35
VIDEO_READY_TIMEOUT = 20

# Seek nhanh trước, chỉ reload khi bị kẹt.
FAST_SEEK_TIMEOUT = 3.0
RECOVERY_SEEK_TIMEOUT = 8.0

FRAME_SETTLE_SECONDS = 0.15


# =========================================================
# CREATE OUTPUT FOLDERS
# =========================================================

for folder in [
    KEYFRAME_DIR,
    RESULT_DIR,
    PER_VIDEO_SUMMARY_DIR,
]:
    folder.mkdir(parents=True, exist_ok=True)


# =========================================================
# HELPERS
# =========================================================

def video_sort_key(video_id):
    video_id = str(video_id).strip()

    try:
        return int(video_id[1:])
    except Exception:
        return 999999


def create_driver():
    options = Options()

    options.add_argument("--headless=new")
    options.add_argument("--window-size=1600,1000")
    options.add_argument("--mute-audio")
    options.add_argument("--disable-notifications")
    options.add_argument("--disable-popup-blocking")
    options.add_argument("--disable-extensions")
    options.add_argument("--disable-sync")
    options.add_argument("--disable-default-apps")
    options.add_argument("--no-first-run")
    options.add_argument("--autoplay-policy=no-user-gesture-required")

    driver = webdriver.Chrome(options=options)
    driver.set_page_load_timeout(PAGE_LOAD_TIMEOUT)

    return driver


def build_timestamp_url(url, timestamp):
    base = str(url).strip()

    if "&t=" in base:
        base = base.split("&t=")[0]

    if "?t=" in base:
        base = base.split("?t=")[0]

    separator = "&" if "?" in base else "?"

    return (
        f"{base}"
        f"{separator}"
        f"t={int(timestamp)}s"
        f"&autoplay=1"
    )


# =========================================================
# YOUTUBE PLAYER
# =========================================================

def wait_video_ready(driver):
    def check(d):
        try:
            video = d.find_element(By.TAG_NAME, "video")

            info = d.execute_script(
                """
                const v = arguments[0];
                return {
                    duration: v.duration,
                    readyState: v.readyState,
                    width: v.videoWidth,
                    height: v.videoHeight
                };
                """,
                video,
            )

            if (
                info["duration"]
                and float(info["duration"]) > 0
                and int(info["readyState"]) >= 2
                and int(info["width"]) > 0
                and int(info["height"]) > 0
            ):
                return video

        except Exception:
            pass

        return False

    return WebDriverWait(
        driver,
        VIDEO_READY_TIMEOUT,
    ).until(check)


def player_has_error(driver):
    try:
        body_text = (
            driver.find_element(By.TAG_NAME, "body")
            .text
            .lower()
        )

        bad_phrases = [
            "video unavailable",
            "playback error",
            "something went wrong",
            "an error occurred",
            "refresh or try again",
        ]

        return any(
            phrase in body_text
            for phrase in bad_phrases
        )

    except Exception:
        return False


def hide_youtube_ui(driver):
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
                '.ytp-bezel',
                '.ytp-cards-teaser',
                '.ytp-paid-content-overlay'
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


def open_video_once(driver, url):
    print("Opening YouTube once...")

    driver.get(url)

    video = wait_video_ready(driver)

    if player_has_error(driver):
        raise RuntimeError("YouTube player error")

    driver.execute_script(
        """
        const v = arguments[0];
        v.muted = true;

        const p = v.play();
        if (p !== undefined) {
            p.catch(() => {});
        }
        """,
        video,
    )

    time.sleep(0.4)

    driver.execute_script(
        "arguments[0].pause();",
        video,
    )

    hide_youtube_ui(driver)

    duration = driver.execute_script(
        "return arguments[0].duration;",
        video,
    )

    return float(duration)


# =========================================================
# SEEK / SCREENSHOT
# =========================================================

def seek_and_wait(
    driver,
    timestamp,
    timeout_seconds,
):
    video = driver.find_element(
        By.TAG_NAME,
        "video",
    )

    driver.execute_script(
        """
        const v = arguments[0];
        const t = arguments[1];

        v.muted = true;
        v.currentTime = t;

        const p = v.play();
        if (p !== undefined) {
            p.catch(() => {});
        }
        """,
        video,
        float(timestamp),
    )

    deadline = (
        time.perf_counter()
        + timeout_seconds
    )

    while time.perf_counter() < deadline:
        try:
            video = driver.find_element(
                By.TAG_NAME,
                "video",
            )

            state = driver.execute_script(
                """
                const v = arguments[0];
                return {
                    currentTime: v.currentTime,
                    readyState: v.readyState,
                    seeking: v.seeking
                };
                """,
                video,
            )

            current_time = float(
                state["currentTime"]
            )

            delta = abs(
                current_time
                - float(timestamp)
            )

            if (
                delta <= 0.75
                and int(state["readyState"]) >= 2
                and not bool(state["seeking"])
            ):
                driver.execute_script(
                    "arguments[0].pause();",
                    video,
                )

                time.sleep(
                    FRAME_SETTLE_SECONDS
                )

                return driver.find_element(
                    By.TAG_NAME,
                    "video",
                )

        except Exception:
            pass

        time.sleep(0.03)

    raise TimeoutError(
        f"Seek timeout at {timestamp}s"
    )


def save_video_screenshot(
    driver,
    output_file,
):
    hide_youtube_ui(driver)

    video = driver.find_element(
        By.TAG_NAME,
        "video",
    )

    video.screenshot(
        str(output_file)
    )

    return (
        output_file.exists()
        and output_file.stat().st_size
        > MIN_VALID_IMAGE_BYTES
    )


def capture_timestamp(
    driver,
    url,
    timestamp,
    output_file,
):
    # FAST PATH
    try:
        seek_and_wait(
            driver=driver,
            timestamp=timestamp,
            timeout_seconds=FAST_SEEK_TIMEOUT,
        )

        if save_video_screenshot(
            driver,
            output_file,
        ):
            return True, "fast"

    except Exception:
        pass

    # RECOVERY PATH
    print(
        " recover",
        end="",
    )

    try:
        timestamp_url = build_timestamp_url(
            url,
            timestamp,
        )

        driver.get(timestamp_url)

        wait_video_ready(driver)

        if player_has_error(driver):
            raise RuntimeError(
                "YouTube player error"
            )

        seek_and_wait(
            driver=driver,
            timestamp=timestamp,
            timeout_seconds=RECOVERY_SEEK_TIMEOUT,
        )

        if save_video_screenshot(
            driver,
            output_file,
        ):
            return True, "recovery"

    except Exception as error:
        print(
            f" recovery_error="
            f"{type(error).__name__}",
            end="",
        )

    return False, "failed"


# =========================================================
# RESUME SUPPORT
# =========================================================

def load_existing_rows(summary_file):
    if not summary_file.exists():
        return {}

    try:
        df = pd.read_csv(summary_file)

    except Exception:
        return {}

    rows = {}

    for _, row in df.iterrows():
        try:
            timestamp = int(
                row["timestamp_sec"]
            )
        except Exception:
            continue

        rows[timestamp] = row.to_dict()

    return rows


def existing_frame_valid(row):
    if row is None:
        return False

    if str(
        row.get(
            "status",
            "",
        )
    ) != "success":
        return False

    image_file = str(
        row.get(
            "image_file",
            "",
        )
    ).strip()

    if not image_file:
        return False

    path = Path(image_file)

    return (
        path.exists()
        and path.stat().st_size
        > MIN_VALID_IMAGE_BYTES
    )


def summary_is_complete(summary_file):
    if not summary_file.exists():
        return False

    try:
        df = pd.read_csv(summary_file)

    except Exception:
        return False

    if df.empty:
        return False

    required = {
        "timestamp_sec",
        "duration_sec",
        "image_file",
        "status",
    }

    if not required.issubset(df.columns):
        return False

    try:
        duration = float(
            df["duration_sec"].iloc[0]
        )
    except Exception:
        return False

    expected_timestamps = list(
        range(
            0,
            int(duration),
            FRAME_INTERVAL,
        )
    )

    rows = {}

    for _, row in df.iterrows():
        try:
            timestamp = int(
                row["timestamp_sec"]
            )
        except Exception:
            continue

        rows[timestamp] = row.to_dict()

    if set(rows) != set(expected_timestamps):
        return False

    return all(
        existing_frame_valid(
            rows[timestamp]
        )
        for timestamp in expected_timestamps
    )


def save_video_summary(
    rows_by_timestamp,
    summary_file,
):
    rows = [
        rows_by_timestamp[t]
        for t in sorted(
            rows_by_timestamp
        )
    ]

    pd.DataFrame(rows).to_csv(
        summary_file,
        index=False,
        encoding="utf-8-sig",
    )


# =========================================================
# PROCESS ONE VIDEO
# =========================================================

def process_video(
    driver,
    video_id,
    subject,
    url,
    force=False,
):
    print()
    print("=" * 72)
    print("VIDEO:", video_id)
    print("SUBJECT:", subject)
    print("=" * 72)

    summary_file = (
        PER_VIDEO_SUMMARY_DIR
        / f"{video_id}.csv"
    )

    output_dir = (
        KEYFRAME_DIR
        / video_id
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if (
        not force
        and summary_is_complete(
            summary_file
        )
    ):
        print(
            "SKIP: complete video already exists"
        )

        return pd.read_csv(
            summary_file
        )

    duration = open_video_once(
        driver,
        url,
    )

    print(
        f"Duration: {duration:.1f}s"
    )

    timestamps = list(
        range(
            0,
            int(duration),
            FRAME_INTERVAL,
        )
    )

    print(
        "Expected keyframes:",
        len(timestamps),
    )

    existing = (
        {}
        if force
        else load_existing_rows(
            summary_file
        )
    )

    rows_by_timestamp = dict(existing)

    start_time = time.perf_counter()

    skipped = 0
    recoveries = 0

    for frame_id, timestamp in enumerate(
        timestamps
    ):
        existing_row = rows_by_timestamp.get(
            timestamp
        )

        if (
            not force
            and existing_frame_valid(
                existing_row
            )
        ):
            skipped += 1

            print(
                f"[{frame_id + 1}/{len(timestamps)}] "
                f"{video_id} - {timestamp}s SKIP"
            )

            continue

        print(
            f"[{frame_id + 1}/{len(timestamps)}] "
            f"{video_id} - {timestamp}s",
            end="",
        )

        filename = (
            f"{video_id}_"
            f"{timestamp:05d}s.png"
        )

        output_file = (
            output_dir
            / filename
        )

        success, method = capture_timestamp(
            driver=driver,
            url=url,
            timestamp=timestamp,
            output_file=output_file,
        )

        if success:
            if method == "recovery":
                recoveries += 1

            print(
                f" OK [{method}]"
            )

            status = "success"

            image_file = str(
                output_file.resolve()
            )

        else:
            print(" FAILED")

            status = "failed"
            image_file = ""

        rows_by_timestamp[
            timestamp
        ] = {
            "video_id": video_id,
            "subject": subject,
            "frame_id": frame_id,
            "timestamp_sec": timestamp,
            "duration_sec": duration,
            "frame_interval_sec": FRAME_INTERVAL,
            "image_file": image_file,
            "status": status,
            "capture_method": method,
        }

        # Save progress after every frame.
        save_video_summary(
            rows_by_timestamp,
            summary_file,
        )

    final_rows = []
    final_failed = 0

    for frame_id, timestamp in enumerate(
        timestamps
    ):
        row = rows_by_timestamp.get(
            timestamp
        )

        if row is None:
            row = {
                "video_id": video_id,
                "subject": subject,
                "frame_id": frame_id,
                "timestamp_sec": timestamp,
                "duration_sec": duration,
                "frame_interval_sec": FRAME_INTERVAL,
                "image_file": "",
                "status": "missing",
                "capture_method": "missing",
            }

        if not existing_frame_valid(row):
            final_failed += 1

        final_rows.append(row)

    result = pd.DataFrame(final_rows)

    result.to_csv(
        summary_file,
        index=False,
        encoding="utf-8-sig",
    )

    elapsed = (
        time.perf_counter()
        - start_time
    )

    print()
    print(
        f"{video_id} successful frames:",
        len(timestamps) - final_failed,
    )

    print(
        f"{video_id} failed frames:",
        final_failed,
    )

    print(
        "Skipped existing frames:",
        skipped,
    )

    print(
        "Recovery reloads:",
        recoveries,
    )

    print(
        "Video processing time:",
        f"{elapsed:.1f}s",
    )

    return result


# =========================================================
# GLOBAL SUMMARY
# =========================================================

def rebuild_global_summary():
    files = sorted(
        PER_VIDEO_SUMMARY_DIR.glob(
            "v*.csv"
        ),
        key=lambda p:
            video_sort_key(p.stem),
    )

    frames = []

    for file in files:
        try:
            df = pd.read_csv(file)

            if not df.empty:
                frames.append(df)

        except Exception as error:
            print(
                "WARNING:",
                file,
                repr(error),
            )

    if not frames:
        result = pd.DataFrame()

    else:
        result = pd.concat(
            frames,
            ignore_index=True,
        )

        result["_video_order"] = (
            result["video_id"]
            .map(video_sort_key)
        )

        result = (
            result
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

    result.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    return result


# =========================================================
# MAIN
# =========================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Run one video only, e.g. --video v10",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore saved progress and recapture selected videos",
    )

    args = parser.parse_args()

    print("=" * 72)
    print("OCR BEST INTEGRATION")
    print(
        "STEP 01 - FAST KEYFRAME EXTRACTION "
        "+ AUTO RECOVERY + RESUME"
    )
    print("=" * 72)

    print(
        "Scientific sampling interval:",
        FRAME_INTERVAL,
        "seconds"
    )

    print(
        "Fast mode: one YouTube page load per video"
    )

    print(
        "Recovery: reload only when seek fails"
    )

    print(
        "Resume: frame-level checkpoints enabled"
    )

    videos = pd.read_csv(VIDEOS_FILE)

    videos["video_id"] = (
        videos["video_id"]
        .astype(str)
        .str.strip()
    )

    videos["subject"] = (
        videos["subject"]
        .astype(str)
        .str.strip()
    )

    videos["status"] = (
        videos["status"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    videos["url"] = (
        videos["url"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    videos = videos[
        videos["status"].eq(
            "selected"
        )
        & videos["video_id"].isin(
            DEV_IDS
        )
    ].copy()

    videos["_order"] = (
        videos["video_id"]
        .map(video_sort_key)
    )

    videos = (
        videos
        .sort_values("_order")
        .drop(columns=["_order"])
        .reset_index(drop=True)
    )

    assert len(videos) == 40

    assert set(
        videos["video_id"]
    ) == set(DEV_IDS)

    assert (
        videos["url"]
        .ne("")
        .all()
    )

    if args.video:
        target = (
            str(args.video)
            .strip()
        )

        if target not in DEV_IDS:
            raise ValueError(
                "--video must be v1-v40"
            )

        videos = videos[
            videos["video_id"]
            .eq(target)
        ].copy()

        print(
            "Run scope:",
            target,
        )

    else:
        print(
            "Run scope: ALL 40 DEV VIDEOS"
        )

    driver = create_driver()

    failed_videos = []

    run_start = time.perf_counter()

    try:
        for _, row in videos.iterrows():
            video_id = row["video_id"]

            try:
                process_video(
                    driver=driver,
                    video_id=video_id,
                    subject=row["subject"],
                    url=row["url"],
                    force=args.force,
                )

            except Exception as error:
                failed_videos.append(
                    video_id
                )

                print()
                print(
                    "FAILED VIDEO:",
                    video_id,
                )

                print(
                    repr(error)
                )

                try:
                    driver.quit()
                except Exception:
                    pass

                driver = create_driver()

    finally:
        try:
            driver.quit()
        except Exception:
            pass

    global_result = (
        rebuild_global_summary()
    )

    elapsed = (
        time.perf_counter()
        - run_start
    )

    print()
    print("=" * 72)
    print("STEP 01 RUN SUMMARY")
    print("=" * 72)

    if global_result.empty:
        videos_present = 0
        successful_frames = 0
        failed_frames = 0

    else:
        videos_present = int(
            global_result[
                "video_id"
            ].nunique()
        )

        successful_frames = int(
            global_result[
                "status"
            ]
            .eq("success")
            .sum()
        )

        failed_frames = int(
            global_result[
                "status"
            ]
            .ne("success")
            .sum()
        )

    print(
        "Summary videos present:",
        videos_present,
    )

    print(
        "Successful frames:",
        successful_frames,
    )

    print(
        "Failed frames:",
        failed_frames,
    )

    print(
        "Failed videos this run:",
        failed_videos,
    )

    print(
        "Runtime:",
        f"{elapsed / 60:.2f} min",
    )

    print(
        "Global summary:",
        OUTPUT_FILE,
    )

    if (
        failed_frames == 0
        and not failed_videos
        and (
            args.video is not None
            or videos_present == 40
        )
    ):
        print()
        print("STEP 01 PASS")

    else:
        print()
        print(
            "STEP 01 NOT COMPLETE YET."
        )

        print(
            "Safe to rerun: "
            "completed frames/videos will be skipped."
        )


if __name__ == "__main__" :
    main()