from pathlib import Path
import os
import subprocess
import sys
import tempfile

import pandas as pd


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
    / "ocr_best_integration"
    / "keyframes_full"
)

SUMMARY_DIR = (
    PROJECT_ROOT
    / "module1"
    / "results"
    / "ocr_best_integration"
    / "keyframe_summaries"
)

REPORT_FILE = (
    PROJECT_ROOT
    / "module1"
    / "results"
    / "ocr_best_integration"
    / "targeted_keyframe_repair_report.csv"
)

DEV_IDS = [f"v{i}" for i in range(1, 41)]

FRAME_INTERVAL = 10

# Giữ cùng điều kiện với audit Step 02
MIN_IMAGE_BYTES = 5000

# Tải một đoạn ngắn quanh timestamp lỗi
SECONDS_BEFORE = 20
SECONDS_AFTER = 10


# =========================================================
# BASIC HELPERS
# =========================================================

def output_path(video_id, timestamp):
    folder = KEYFRAME_DIR / video_id

    folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    return (
        folder
        / f"{video_id}_{timestamp:05d}s.png"
    )


def valid_image(path_value):
    value = str(path_value or "").strip()

    if not value or value.lower() == "nan":
        return False

    path = Path(value)

    return (
        path.exists()
        and path.is_file()
        and path.stat().st_size > MIN_IMAGE_BYTES
    )


def run_command(command):
    """
    Run subprocess while disabling yt-dlp plugins.

    This avoids the bgutil / Deno timeout
    that occurred on this machine.
    """

    env = os.environ.copy()

    env["YTDLP_NO_PLUGINS"] = "1"

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )

    return (
        result.returncode,
        result.stdout,
    )


# =========================================================
# AUTHORITATIVE MISSING-FRAME CHECK
# =========================================================

def find_missing(video_id):
    summary_file = (
        SUMMARY_DIR
        / f"{video_id}.csv"
    )

    if not summary_file.exists():
        raise FileNotFoundError(
            f"Missing summary file: {summary_file}"
        )

    df = pd.read_csv(
        summary_file
    )

    if df.empty:
        raise RuntimeError(
            f"Empty summary file: {summary_file}"
        )

    df["timestamp_sec"] = pd.to_numeric(
        df["timestamp_sec"],
        errors="coerce",
    )

    duration_values = pd.to_numeric(
        df["duration_sec"],
        errors="coerce",
    ).dropna()

    if duration_values.empty:
        raise RuntimeError(
            f"Missing duration for {video_id}"
        )

    duration = float(
        duration_values.iloc[0]
    )

    expected_timestamps = list(
        range(
            0,
            int(duration),
            FRAME_INTERVAL,
        )
    )

    rows = {}

    for _, row in df.dropna(
        subset=["timestamp_sec"]
    ).iterrows():

        timestamp = int(
            row["timestamp_sec"]
        )

        rows[timestamp] = row

    missing = []

    for timestamp in expected_timestamps:
        row = rows.get(timestamp)

        if row is None:
            missing.append(timestamp)
            continue

        status = str(
            row.get(
                "status",
                "",
            )
        ).strip().lower()

        if status != "success":
            missing.append(timestamp)
            continue

        image_file = row.get(
            "image_file",
            "",
        )

        if not valid_image(
            image_file
        ):
            missing.append(timestamp)

    return missing


# =========================================================
# UPDATE SUMMARY
# =========================================================

def update_summary(
    video_id,
    timestamp,
    success,
    method,
):
    summary_file = (
        SUMMARY_DIR
        / f"{video_id}.csv"
    )

    df = pd.read_csv(
        summary_file
    )

    df["timestamp_sec"] = pd.to_numeric(
        df["timestamp_sec"],
        errors="coerce",
    )

    mask = df[
        "timestamp_sec"
    ].eq(timestamp)

    if not mask.any():
        raise RuntimeError(
            f"{video_id}: "
            f"{timestamp}s not found "
            f"in per-video summary"
        )

    if "capture_method" not in df.columns:
        df["capture_method"] = ""

    frame_file = output_path(
        video_id,
        timestamp,
    )

    if success:
        df.loc[
            mask,
            "status",
        ] = "success"

        df.loc[
            mask,
            "image_file",
        ] = str(
            frame_file.resolve()
        )

        df.loc[
            mask,
            "capture_method",
        ] = method

    else:
        df.loc[
            mask,
            "status",
        ] = "failed"

        df.loc[
            mask,
            "image_file",
        ] = ""

        df.loc[
            mask,
            "capture_method",
        ] = method

    df.to_csv(
        summary_file,
        index=False,
        encoding="utf-8-sig",
    )


# =========================================================
# DOWNLOAD SHORT SECTION + EXTRACT PNG
# =========================================================

def repair_frame(
    url,
    video_id,
    timestamp,
):
    section_start = max(
        0,
        timestamp - SECONDS_BEFORE,
    )

    section_end = (
        timestamp + SECONDS_AFTER
    )

    relative_timestamp = (
        timestamp - section_start
    )

    print(
        f"  section: "
        f"{section_start}s -> {section_end}s"
    )

    print(
        f"  target inside section: "
        f"+{relative_timestamp}s"
    )

    with tempfile.TemporaryDirectory(
        prefix=f"{video_id}_repair_"
    ) as temp_dir_string:

        temp_dir = Path(
            temp_dir_string
        )

        output_template = (
            temp_dir
            / "repair_section.%(ext)s"
        )

        # =================================================
        # A. DOWNLOAD SHORT SECTION
        # =================================================

        yt_command = [
            sys.executable,
            "-m",
            "yt_dlp",

            "--no-playlist",
            "--force-overwrites",
            "--no-part",

            "--download-sections",
            (
                f"*{section_start}"
                f"-{section_end}"
            ),

            "--force-keyframes-at-cuts",

            "-f",
            (
                "bv*[height<=720]+ba/"
                "b[height<=720]/best"
            ),

            "--merge-output-format",
            "mp4",

            "-o",
            str(output_template),

            url,
        ]

        print(
            "  yt-dlp: downloading short section..."
        )

        return_code, yt_output = (
            run_command(
                yt_command
            )
        )

        if return_code != 0:
            print()
            print(
                "YT-DLP ERROR"
            )
            print(
                yt_output
            )

            return (
                False,
                "yt_dlp_failed",
            )

        # =================================================
        # B. LOCATE DOWNLOADED MEDIA
        # =================================================

        media_extensions = {
            ".mp4",
            ".mkv",
            ".webm",
            ".mov",
        }

        candidates = [
            path
            for path in temp_dir.iterdir()
            if (
                path.is_file()
                and path.suffix.lower()
                in media_extensions
            )
        ]

        if not candidates:
            print(
                "  ERROR: no downloaded media found."
            )

            return (
                False,
                "downloaded_media_missing",
            )

        media_file = max(
            candidates,
            key=lambda path:
                path.stat().st_size,
        )

        print(
            "  downloaded:",
            media_file.name,
        )

        print(
            "  size:",
            round(
                media_file.stat().st_size
                / 1024
                / 1024,
                2,
            ),
            "MB",
        )

        # =================================================
        # C. EXTRACT TARGET FRAME
        # =================================================

        final_frame = output_path(
            video_id,
            timestamp,
        )

        # Remove any previous invalid image first.
        if final_frame.exists():
            final_frame.unlink()

        print(
            f"  ffmpeg: extracting "
            f"{video_id} @ {timestamp}s..."
        )

        ffmpeg_command = [
            "ffmpeg",

            "-hide_banner",
            "-loglevel",
            "error",

            "-y",

            # Accurate seek inside the short local clip.
            "-i",
            str(media_file),

            "-ss",
            f"{relative_timestamp:.3f}",

            "-frames:v",
            "1",

            # Explicit PNG codec.
            "-c:v",
            "png",

            # Important:
            # no PNG compression so a simple/static
            # slide does not accidentally fall below
            # the existing 5000-byte audit threshold.
            "-compression_level",
            "0",

            "-pix_fmt",
            "rgb24",

            str(final_frame),
        ]

        ffmpeg_code, ffmpeg_output = (
            run_command(
                ffmpeg_command
            )
        )

        if ffmpeg_code != 0:
            print()
            print(
                "FFMPEG ERROR"
            )
            print(
                ffmpeg_output
            )

            return (
                False,
                "ffmpeg_failed",
            )

        # =================================================
        # D. VALIDATE OUTPUT
        # =================================================

        if not final_frame.exists():
            print(
                "  ERROR: PNG was not created."
            )

            return (
                False,
                "png_missing",
            )

        image_size = (
            final_frame.stat().st_size
        )

        print(
            "  PNG size:",
            image_size,
            "bytes",
        )

        if image_size <= MIN_IMAGE_BYTES:
            print(
                "  ERROR: PNG <= 5000 bytes."
            )

            return (
                False,
                "png_too_small",
            )

        print(
            "  PNG valid."
        )

        return (
            True,
            "yt_dlp_ffmpeg_png",
        )


# =========================================================
# MAIN
# =========================================================

def main():
    print("=" * 72)

    print(
        "STEP 03 - TARGETED KEYFRAME REPAIR "
        "WITH YT-DLP + FFMPEG"
    )

    print("=" * 72)

    # =====================================================
    # LOAD VIDEO URLS
    # =====================================================

    videos = pd.read_csv(
        VIDEOS_FILE
    )

    videos["video_id"] = (
        videos["video_id"]
        .astype(str)
        .str.strip()
    )

    videos["url"] = (
        videos["url"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    videos = videos[
        videos["video_id"].isin(
            DEV_IDS
        )
    ].copy()

    url_map = dict(
        zip(
            videos["video_id"],
            videos["url"],
        )
    )

    # =====================================================
    # AUDIT BEFORE REPAIR
    # =====================================================

    missing_map = {}

    for video_id in DEV_IDS:
        missing = find_missing(
            video_id
        )

        if missing:
            missing_map[
                video_id
            ] = missing

    total_before = sum(
        len(timestamps)
        for timestamps
        in missing_map.values()
    )

    print()

    print(
        "Missing before repair:",
        total_before,
    )

    print(
        "Affected videos:",
        len(missing_map),
    )

    for video_id, timestamps in (
        missing_map.items()
    ):
        print(
            f"{video_id}: "
            + ", ".join(
                f"{timestamp}s"
                for timestamp
                in timestamps
            )
        )

    if total_before == 0:
        print()

        print(
            "STEP 03 PASS - "
            "NOTHING TO REPAIR"
        )

        return

    # =====================================================
    # REPAIR ONLY ACTUAL MISSING FRAMES
    # =====================================================

    report_rows = []

    for video_id, timestamps in (
        missing_map.items()
    ):

        url = url_map.get(
            video_id,
            "",
        )

        if not url:
            raise RuntimeError(
                f"URL missing for {video_id}"
            )

        print()
        print("=" * 72)

        print(
            f"REPAIR VIDEO: {video_id}"
        )

        print("=" * 72)

        for timestamp in timestamps:
            print()

            print(
                f"{video_id} @ {timestamp}s"
            )

            success, method = repair_frame(
                url=url,
                video_id=video_id,
                timestamp=timestamp,
            )

            update_summary(
                video_id=video_id,
                timestamp=timestamp,
                success=success,
                method=method,
            )

            print(
                "  RESULT:",
                "OK"
                if success
                else "FAILED",
            )

            report_rows.append(
                {
                    "video_id":
                        video_id,

                    "timestamp_sec":
                        timestamp,

                    "success":
                        success,

                    "method":
                        method,
                }
            )

    pd.DataFrame(
        report_rows
    ).to_csv(
        REPORT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    # =====================================================
    # FINAL LIVE AUDIT
    # =====================================================

    remaining_map = {}

    for video_id in DEV_IDS:
        missing = find_missing(
            video_id
        )

        if missing:
            remaining_map[
                video_id
            ] = missing

    total_after = sum(
        len(timestamps)
        for timestamps
        in remaining_map.values()
    )

    print()
    print("=" * 72)

    print(
        "TARGETED REPAIR FINAL AUDIT"
    )

    print("=" * 72)

    print(
        "Missing before:",
        total_before,
    )

    print(
        "Missing after:",
        total_after,
    )

    if total_after == 0:
        print(
            "Affected videos: 0"
        )

        print()

        print(
            "STEP 03 PASS - "
            "ALL KEYFRAMES PRESENT"
        )

    else:
        print(
            "Affected videos:",
            len(remaining_map),
        )

        for video_id, timestamps in (
            remaining_map.items()
        ):
            print(
                f"{video_id}: "
                + ", ".join(
                    f"{timestamp}s"
                    for timestamp
                    in timestamps
                )
            )

        print()

        print(
            "STEP 03 INCOMPLETE"
        )

        print(
            "Do NOT proceed to OCR."
        )


if __name__ == "__main__":
    main()