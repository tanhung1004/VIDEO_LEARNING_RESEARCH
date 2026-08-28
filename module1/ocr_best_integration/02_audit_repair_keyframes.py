from pathlib import Path
import argparse
import subprocess
import sys
import time

import pandas as pd


# =========================================================
# PATHS / FROZEN SETTINGS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

EXTRACTOR = (
    PROJECT_ROOT
    / "module1"
    / "ocr_best_integration"
    / "01_extract_keyframes_40.py"
)

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

AUDIT_BEFORE = (
    RESULT_DIR
    / "keyframe_missing_before_repair.csv"
)

AUDIT_AFTER = (
    RESULT_DIR
    / "keyframe_missing_after_repair.csv"
)

REPAIR_LOG = (
    RESULT_DIR
    / "keyframe_repair_passes.csv"
)

DEV_IDS = [f"v{i}" for i in range(1, 41)]

FRAME_INTERVAL = 10
MIN_VALID_IMAGE_BYTES = 5000


# =========================================================
# HELPERS
# =========================================================

def video_sort_key(video_id):

    try:
        return int(
            str(video_id).strip()[1:]
        )

    except Exception:
        return 999999


def valid_image(path_value):

    path_text = str(
        path_value or ""
    ).strip()

    if (
        not path_text
        or path_text.lower() == "nan"
    ):
        return False

    path = Path(path_text)

    return (
        path.exists()
        and path.is_file()
        and path.stat().st_size
        > MIN_VALID_IMAGE_BYTES
    )


# =========================================================
# AUDIT ONE VIDEO
# =========================================================

def audit_video(video_id):

    summary_file = (
        SUMMARY_DIR
        / f"{video_id}.csv"
    )

    if not summary_file.exists():

        return [{
            "video_id":
                video_id,

            "timestamp_sec":
                pd.NA,

            "reason":
                "summary_missing",

            "summary_file":
                str(summary_file),
        }]

    try:

        df = pd.read_csv(
            summary_file
        )

    except Exception as error:

        return [{
            "video_id":
                video_id,

            "timestamp_sec":
                pd.NA,

            "reason":
                (
                    "summary_unreadable:"
                    + type(error).__name__
                ),

            "summary_file":
                str(summary_file),
        }]

    if df.empty:

        return [{
            "video_id":
                video_id,

            "timestamp_sec":
                pd.NA,

            "reason":
                "summary_empty",

            "summary_file":
                str(summary_file),
        }]

    required = {
        "timestamp_sec",
        "duration_sec",
        "image_file",
        "status",
    }

    missing_columns = sorted(
        required
        - set(df.columns)
    )

    if missing_columns:

        return [{
            "video_id":
                video_id,

            "timestamp_sec":
                pd.NA,

            "reason":
                (
                    "summary_missing_columns:"
                    + ",".join(
                        missing_columns
                    )
                ),

            "summary_file":
                str(summary_file),
        }]

    duration_values = pd.to_numeric(
        df["duration_sec"],
        errors="coerce",
    ).dropna()

    if duration_values.empty:

        return [{
            "video_id":
                video_id,

            "timestamp_sec":
                pd.NA,

            "reason":
                "duration_missing",

            "summary_file":
                str(summary_file),
        }]

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

    work = df.copy()

    work["timestamp_sec"] = (
        pd.to_numeric(
            work["timestamp_sec"],
            errors="coerce",
        )
    )

    work = work.dropna(
        subset=["timestamp_sec"]
    ).copy()

    work["timestamp_sec"] = (
        work["timestamp_sec"]
        .astype(int)
    )

    work = (
        work
        .drop_duplicates(
            subset=[
                "timestamp_sec"
            ],
            keep="last",
        )
    )

    by_timestamp = {
        int(row["timestamp_sec"]):
            row

        for _, row
        in work.iterrows()
    }

    problems = []

    for timestamp in expected_timestamps:

        row = by_timestamp.get(
            timestamp
        )

        if row is None:

            reason = (
                "row_missing"
            )

        elif (
            str(
                row.get(
                    "status",
                    "",
                )
            )
            .strip()
            .lower()
            != "success"
        ):

            reason = (
                "status_"
                + str(
                    row.get(
                        "status",
                        "missing",
                    )
                )
                .strip()
                .lower()
            )

        elif not valid_image(
            row.get(
                "image_file",
                "",
            )
        ):

            reason = (
                "image_missing_or_invalid"
            )

        else:

            continue

        problems.append({
            "video_id":
                video_id,

            "timestamp_sec":
                timestamp,

            "reason":
                reason,

            "summary_file":
                str(summary_file),
        })

    return problems


# =========================================================
# AUDIT ALL 40
# =========================================================

def audit_all():

    rows = []

    for video_id in DEV_IDS:

        rows.extend(
            audit_video(
                video_id
            )
        )

    if not rows:

        return pd.DataFrame(
            columns=[
                "video_id",
                "timestamp_sec",
                "reason",
                "summary_file",
            ]
        )

    result = pd.DataFrame(
        rows
    )

    result["_order"] = (
        result["video_id"]
        .map(
            video_sort_key
        )
    )

    result["_ts"] = (
        pd.to_numeric(
            result[
                "timestamp_sec"
            ],
            errors="coerce",
        )
        .fillna(-1)
    )

    result = (
        result
        .sort_values(
            [
                "_order",
                "_ts",
            ]
        )
        .drop(
            columns=[
                "_order",
                "_ts",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    return result


# =========================================================
# PRINT AUDIT
# =========================================================

def print_audit(
    title,
    audit_df,
):

    print()

    print(
        "=" * 72
    )

    print(title)

    print(
        "=" * 72
    )

    if audit_df.empty:

        print(
            "Missing/failed frames: 0"
        )

        print(
            "Affected videos: 0"
        )

        print(
            "KEYFRAME COVERAGE PASS"
        )

        return

    print(
        "Missing/failed frames:",
        len(audit_df),
    )

    print(
        "Affected videos:",
        audit_df[
            "video_id"
        ].nunique(),
    )

    for (
        video_id,
        group,
    ) in audit_df.groupby(
        "video_id",
        sort=False,
    ):

        timestamps = []

        for value in group[
            "timestamp_sec"
        ]:

            if pd.isna(value):

                timestamps.append(
                    "<summary problem>"
                )

            else:

                timestamps.append(
                    f"{int(value)}s"
                )

        print(
            f"{video_id}: "
            + ", ".join(
                timestamps
            )
        )


# =========================================================
# CALL EXISTING EXTRACTOR
# =========================================================

def run_extractor_for_video(
    video_id,
):

    command = [
        sys.executable,
        str(EXTRACTOR),
        "--video",
        video_id,
    ]

    print()

    print(
        "REPAIR:",
        video_id,
    )

    completed = subprocess.run(
        command,
        cwd=str(
            PROJECT_ROOT
        ),
        check=False,
    )

    return int(
        completed.returncode
    )


# =========================================================
# MAIN
# =========================================================

def main():

    parser = (
        argparse.ArgumentParser(
            description=(
                "Audit missing OCR keyframes "
                "and retry only affected videos."
            )
        )
    )

    parser.add_argument(
        "--repair",
        action="store_true",
        help=(
            "Retry only affected videos. "
            "Existing successful frames "
            "will be skipped."
        ),
    )

    parser.add_argument(
        "--passes",
        type=int,
        default=4,
        help=(
            "Maximum repair passes. "
            "Default: 4."
        ),
    )

    args = parser.parse_args()

    if not EXTRACTOR.exists():

        raise FileNotFoundError(
            f"Missing extractor: "
            f"{EXTRACTOR}"
        )

    SUMMARY_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # =====================================================
    # AUDIT BEFORE
    # =====================================================

    before = audit_all()

    before.to_csv(
        AUDIT_BEFORE,
        index=False,
        encoding="utf-8-sig",
    )

    print_audit(
        "KEYFRAME AUDIT — BEFORE REPAIR",
        before,
    )

    print()

    print(
        "Audit file:",
        AUDIT_BEFORE,
    )

    if before.empty:

        AUDIT_AFTER.write_text(
            (
                "video_id,"
                "timestamp_sec,"
                "reason,"
                "summary_file\n"
            ),
            encoding="utf-8-sig",
        )

        print()

        print(
            "STEP 02 PASS — "
            "NOTHING TO REPAIR"
        )

        return

    # =====================================================
    # AUDIT ONLY
    # =====================================================

    if not args.repair:

        print()

        print(
            "AUDIT ONLY."
        )

        print(
            "Run again with "
            "--repair to download "
            "missing frames."
        )

        return

    # =====================================================
    # REPAIR
    # =====================================================

    repair_rows = []

    previous_missing = len(
        before
    )

    no_improvement_rounds = 0

    current = before

    max_passes = max(
        1,
        args.passes,
    )

    for pass_number in range(
        1,
        max_passes + 1,
    ):

        if current.empty:
            break

        affected_videos = sorted(
            current[
                "video_id"
            ].unique(),
            key=video_sort_key,
        )

        print()

        print(
            "=" * 72
        )

        print(
            f"REPAIR PASS "
            f"{pass_number}"
        )

        print(
            "=" * 72
        )

        print(
            "Affected videos:",
            affected_videos,
        )

        for video_id in (
            affected_videos
        ):

            before_video = (
                current[
                    current[
                        "video_id"
                    ].eq(
                        video_id
                    )
                ]
            )

            missing_before_video = (
                len(
                    before_video
                )
            )

            started = (
                time.perf_counter()
            )

            return_code = (
                run_extractor_for_video(
                    video_id
                )
            )

            elapsed = (
                time.perf_counter()
                - started
            )

            after_video = (
                pd.DataFrame(
                    audit_video(
                        video_id
                    )
                )
            )

            missing_after_video = (
                len(
                    after_video
                )
            )

            repair_rows.append({
                "pass":
                    pass_number,

                "video_id":
                    video_id,

                "missing_before":
                    missing_before_video,

                "missing_after":
                    missing_after_video,

                "fixed":
                    (
                        missing_before_video
                        - missing_after_video
                    ),

                "extractor_return_code":
                    return_code,

                "elapsed_sec":
                    round(
                        elapsed,
                        2,
                    ),
            })

            print(
                f"{video_id}: "
                f"{missing_before_video}"
                f" -> "
                f"{missing_after_video} "
                f"missing"
            )

        # =================================================
        # AUDIT AFTER EACH PASS
        # =================================================

        current = (
            audit_all()
        )

        current_missing = len(
            current
        )

        print_audit(
            (
                "KEYFRAME AUDIT — "
                f"AFTER PASS "
                f"{pass_number}"
            ),
            current,
        )

        if current_missing == 0:
            break

        if (
            current_missing
            < previous_missing
        ):

            no_improvement_rounds = 0

        else:

            no_improvement_rounds += 1

        previous_missing = (
            current_missing
        )

        # Do not loop forever if
        # exact same frames keep failing.
        if (
            no_improvement_rounds
            >= 2
        ):

            print()

            print(
                "STOP AUTO-REPAIR: "
                "no improvement for "
                "2 consecutive passes."
            )

            break

    # =====================================================
    # FINAL AUDIT
    # =====================================================

    after = audit_all()

    after.to_csv(
        AUDIT_AFTER,
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(
        repair_rows
    ).to_csv(
        REPAIR_LOG,
        index=False,
        encoding="utf-8-sig",
    )

    print_audit(
        "KEYFRAME AUDIT — FINAL",
        after,
    )

    print()

    print(
        "Before audit:",
        AUDIT_BEFORE,
    )

    print(
        "After audit:",
        AUDIT_AFTER,
    )

    print(
        "Repair log:",
        REPAIR_LOG,
    )

    if after.empty:

        print()

        print(
            "=" * 72
        )

        print(
            "STEP 02 PASS — "
            "ALL EXPECTED 10s "
            "KEYFRAMES PRESENT"
        )

        print(
            "=" * 72
        )

    else:

        print()

        print(
            "=" * 72
        )

        print(
            "STEP 02 INCOMPLETE — "
            "SOME FRAMES STILL "
            "NEED REPAIR"
        )

        print(
            "=" * 72
        )

        print(
            "Do NOT proceed "
            "to OCR yet."
        )


if __name__ == "__main__":
    main()