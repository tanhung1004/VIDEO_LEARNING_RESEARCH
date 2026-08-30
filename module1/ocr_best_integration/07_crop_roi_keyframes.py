from pathlib import Path
import re

import pandas as pd
from PIL import Image, ImageDraw


# =========================================================
# CONFIG
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

KEYFRAME_SUMMARY_DIR = (
    PROJECT_ROOT
    / "module1"
    / "results"
    / "ocr_best_integration"
    / "keyframe_summaries"
)

FULL_FRAME_OCR_FILE = (
    PROJECT_ROOT
    / "module1"
    / "results"
    / "ocr_best_integration"
    / "ocr_fullframe_raw.csv"
)

ROI_KEYFRAME_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ocr_best_integration"
    / "keyframes_roi"
)

RESULT_DIR = (
    PROJECT_ROOT
    / "module1"
    / "results"
    / "ocr_best_integration"
)

OUTPUT_FILE = (
    RESULT_DIR
    / "roi_keyframe_summary.csv"
)

ROI_SOURCE_SUMMARY_FILE = (
    RESULT_DIR
    / "roi_source_summary.csv"
)

PREVIEW_DIR = (
    RESULT_DIR
    / "roi_preview"
)

DEV_IDS = [f"v{i}" for i in range(1, 41)]

EXPECTED_FRAMES = 2398
MIN_IMAGE_BYTES = 5000


# =========================================================
# EXACT ROI CONFIG FROM COLLABORATOR PACKAGE
# =========================================================

ROI_CONFIG = {
    "v1": {
        "top": 0.18,
        "left": 0.235,
        "right": 0.0,
        "bottom": 0.06,
        "note": "SSMS video-specific",
    },

    "v2": {
        "top": 0.05,
        "left": 0.04,
        "right": 0.0,
        "bottom": 0.12,
        "note": "VS Code video-specific",
    },

    "v3": {
        "top": 0.0,
        "left": 0.0,
        "right": 0.0,
        "bottom": 0.15,
        "note": "Webcam; crop bottom watermark",
    },

    "v4": {
        "top": 0.0,
        "left": 0.0,
        "right": 0.0,
        "bottom": 0.08,
        "note": "Full-bleed slide",
    },

    "v5": {
        "top": 0.07,
        "left": 0.14,
        "right": 0.0,
        "bottom": 0.12,
        "note": "IntelliJ video-specific",
    },
}


SUBJECT_DEFAULT = {
    "sql": {
        "top": 0.15,
        "left": 0.20,
        "right": 0.0,
        "bottom": 0.06,
    },

    "python": {
        "top": 0.05,
        "left": 0.04,
        "right": 0.0,
        "bottom": 0.10,
    },

    "java": {
        "top": 0.07,
        "left": 0.12,
        "right": 0.0,
        "bottom": 0.10,
    },
}


GLOBAL_DEFAULT = {
    "top": 0.05,
    "left": 0.03,
    "right": 0.0,
    "bottom": 0.08,
}


CONTEXT_MIN_MATCHES_HIGH_CONFIDENCE = 2


CONTEXT_ROI_PROFILES = {
    "ssms": {
        "top": 0.18,
        "left": 0.235,
        "right": 0.0,
        "bottom": 0.06,
    },

    "vscode": {
        "top": 0.05,
        "left": 0.04,
        "right": 0.0,
        "bottom": 0.12,
    },

    "intellij": {
        "top": 0.07,
        "left": 0.14,
        "right": 0.0,
        "bottom": 0.12,
    },

    "slide_fullbleed": {
        "top": 0.0,
        "left": 0.0,
        "right": 0.0,
        "bottom": 0.08,
    },
}


CONTEXT_PATTERNS = {
    "ssms": [
        (
            "ssms_title",
            r"sql\W*server\W*management\W*(studio|saudio|stuoio)",
        ),
        (
            "object_explorer",
            r"ob\w*ect\W*(explorer|splorer|explor\w*)",
        ),
        (
            "ssms_menu_bar",
            r"file\W+edit\W+view\W+project\W+tools?\W+window\W+help",
        ),
        (
            "connect_security",
            r"connect\W+security\W+replication\W+polybase",
        ),
        (
            "localhost_instance",
            r"localhost\W*\(",
        ),
    ],

    "vscode": [
        (
            "ln_col_status",
            r"\bln\W?\d+\W?,?\W?col\W?\d+\b",
        ),
        (
            "spaces_indent",
            r"\bspaces?:\W?\d+\b",
        ),
        (
            "encoding_status",
            r"\butf-?8\b",
        ),
        (
            "eol_status",
            r"\b(crlf|lf)\b",
        ),
        (
            "integrated_terminal",
            r"integrated\W+terminal",
        ),
        (
            "python_current_file",
            r"python\W*:\W*current\W*file",
        ),
        (
            "run_and_debug",
            r"run\W+and\W+debug",
        ),
    ],

    "intellij": [
        (
            "external_libraries",
            r"externat?\W+libraries",
        ),
        (
            "scratches_consoles",
            r"scratches\W+and\W+consoles",
        ),
        (
            "version_control_panel",
            r"version\W+control",
        ),
        (
            "project_structure",
            r"project\W+structure",
        ),
        (
            "invalidate_caches",
            r"invalidate\W+caches",
        ),
        (
            "power_save_mode",
            r"power\W+save\W+mode",
        ),
    ],

    "slide_fullbleed": [
        (
            "neso_academy",
            r"nesoacademy\W*org",
        ),
        (
            "codewithmosh",
            r"codewithmosh\W*com",
        ),
        (
            "mosh_hamedani",
            r"mosh\W+hamedani",
        ),
        (
            "video_unavailable",
            r"video\W+unavailable",
        ),
        (
            "playback_error",
            r"playback\W+error",
        ),
    ],
}


# =========================================================
# HELPERS
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


def normalize_subject(subject):
    value = str(subject).strip().lower()

    aliases = {
        "cpp": "c++",
        "cxx": "c++",
        "c plus plus": "c++",
    }

    return aliases.get(
        value,
        value,
    )


def normalize_frame_id(value):
    try:
        return str(
            int(
                float(value)
            )
        )
    except Exception:
        return str(
            value
        ).strip()


def valid_image(path):
    path = Path(path)

    return (
        path.exists()
        and path.is_file()
        and path.stat().st_size > MIN_IMAGE_BYTES
    )


# =========================================================
# LOAD KEYFRAMES
# =========================================================

def load_keyframes():
    rows = []

    for video_id in DEV_IDS:
        file = (
            KEYFRAME_SUMMARY_DIR
            / f"{video_id}.csv"
        )

        if not file.exists():
            raise FileNotFoundError(
                f"Missing summary: {file}"
            )

        df = pd.read_csv(file)

        if df.empty:
            raise RuntimeError(
                f"Empty summary: {file}"
            )

        rows.append(df)

    frames = pd.concat(
        rows,
        ignore_index=True,
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
        subset=["timestamp_sec"]
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

    failed = int(
        (
            frames["status"]
            .astype(str)
            .str.strip()
            .str.lower()
            != "success"
        ).sum()
    )

    if len(frames) != EXPECTED_FRAMES:
        raise RuntimeError(
            f"Expected {EXPECTED_FRAMES} frames, "
            f"found {len(frames)}."
        )

    if duplicates != 0:
        raise RuntimeError(
            f"Duplicate frames: {duplicates}"
        )

    if failed != 0:
        raise RuntimeError(
            f"Non-success keyframes: {failed}"
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
            columns=["_video_order"]
        )
        .reset_index(drop=True)
    )

    return frames


# =========================================================
# FULL-FRAME OCR LOOKUP
# =========================================================

def load_ocr_lookup():
    if not FULL_FRAME_OCR_FILE.exists():
        raise FileNotFoundError(
            f"Missing full-frame OCR:\n"
            f"{FULL_FRAME_OCR_FILE}"
        )

    df = pd.read_csv(
        FULL_FRAME_OCR_FILE
    )

    lookup = {}

    for _, row in df.iterrows():
        video_id = str(
            row.get(
                "video_id",
                "",
            )
        ).strip()

        frame_id = normalize_frame_id(
            row.get(
                "frame_id",
                "",
            )
        )

        text = str(
            row.get(
                "ocr_text",
                "",
            )
            or ""
        ).lower()

        lookup[
            (
                video_id,
                frame_id,
            )
        ] = text

    return lookup


# =========================================================
# CONTEXT DETECTION
# =========================================================

def detect_context(text):
    if not text:
        return (
            "",
            "low",
            [],
            {},
        )

    scores = {}
    matched = {}

    for context, patterns in (
        CONTEXT_PATTERNS.items()
    ):
        hits = [
            name
            for name, pattern
            in patterns
            if re.search(
                pattern,
                text,
            )
        ]

        if hits:
            scores[
                context
            ] = len(hits)

            matched[
                context
            ] = hits

    if not scores:
        return (
            "",
            "low",
            [],
            {},
        )

    ranked = sorted(
        scores.items(),
        key=lambda item:
            item[1],
        reverse=True,
    )

    best_context = (
        ranked[0][0]
    )

    best_score = (
        ranked[0][1]
    )

    second_score = (
        ranked[1][1]
        if len(ranked) > 1
        else 0
    )

    if (
        best_score
        >= CONTEXT_MIN_MATCHES_HIGH_CONFIDENCE
        and best_score
        > second_score
    ):
        return (
            best_context,
            "high",
            matched[
                best_context
            ],
            scores,
        )

    return (
        best_context,
        "low",
        matched[
            best_context
        ],
        scores,
    )


# =========================================================
# ROI HIERARCHY
# =========================================================

def resolve_roi(
    video_id,
    subject,
    frame_id,
    ocr_lookup,
):
    # 1. v1-v5 fixed manually tuned ROI
    if video_id in ROI_CONFIG:
        return (
            ROI_CONFIG[
                video_id
            ],
            "video_specific",
            "",
            "",
            "",
        )

    # 2. Context-aware
    text = ocr_lookup.get(
        (
            video_id,
            normalize_frame_id(
                frame_id
            ),
        ),
        "",
    )

    best_context = ""
    confidence = ""
    matched_names = []

    if text:
        (
            best_context,
            confidence,
            matched_names,
            _,
        ) = detect_context(
            text
        )

        if (
            best_context
            and confidence
            == "high"
        ):
            return (
                CONTEXT_ROI_PROFILES[
                    best_context
                ],
                "context",
                best_context,
                confidence,
                ";".join(
                    matched_names
                ),
            )

    # 3. Subject fallback
    subject_key = (
        normalize_subject(
            subject
        )
    )

    if subject_key in SUBJECT_DEFAULT:
        return (
            SUBJECT_DEFAULT[
                subject_key
            ],
            "subject_default",
            best_context,
            confidence,
            ";".join(
                matched_names
            ),
        )

    # 4. Global conservative fallback
    return (
        GLOBAL_DEFAULT,
        "global_default",
        best_context,
        confidence,
        ";".join(
            matched_names
        ),
    )


# =========================================================
# CROP
# =========================================================

def crop_roi(
    image,
    roi,
):
    width, height = image.size

    left = int(
        width
        * roi["left"]
    )

    top = int(
        height
        * roi["top"]
    )

    right = int(
        width
        * (
            1
            - roi["right"]
        )
    )

    bottom = int(
        height
        * (
            1
            - roi["bottom"]
        )
    )

    left = max(
        0,
        min(
            left,
            width - 1,
        ),
    )

    top = max(
        0,
        min(
            top,
            height - 1,
        ),
    )

    right = max(
        left + 1,
        min(
            right,
            width,
        ),
    )

    bottom = max(
        top + 1,
        min(
            bottom,
            height,
        ),
    )

    box = (
        left,
        top,
        right,
        bottom,
    )

    return (
        image.crop(
            box
        ),
        box,
    )


# =========================================================
# PREVIEW
# =========================================================

def make_box_preview(
    image,
    box,
):
    preview = (
        image
        .copy()
        .convert("RGB")
    )

    draw = ImageDraw.Draw(
        preview
    )

    draw.rectangle(
        box,
        outline="red",
        width=5,
    )

    return preview


def build_subject_preview(
    subject,
    selected_rows,
):
    """
    One contact sheet / subject.

    10 videos x 3 representative frames:
    first / middle / last.
    """

    if not selected_rows:
        return

    thumb_w = 320
    thumb_h = 180
    label_h = 24

    cols = 3

    rows_count = len(
        selected_rows
    )

    canvas = Image.new(
        "RGB",
        (
            cols * thumb_w,
            rows_count
            * (
                thumb_h
                + label_h
            ),
        ),
        "white",
    )

    draw = ImageDraw.Draw(
        canvas
    )

    for row_index, item in enumerate(
        selected_rows
    ):
        for column, frame in enumerate(
            item["frames"]
        ):
            path = Path(
                frame["image_path"]
            )

            with Image.open(path) as image:
                preview = (
                    make_box_preview(
                        image,
                        frame["box"],
                    )
                )

                preview.thumbnail(
                    (
                        thumb_w,
                        thumb_h,
                    )
                )

                x = (
                    column
                    * thumb_w
                )

                y = (
                    row_index
                    * (
                        thumb_h
                        + label_h
                    )
                    + label_h
                )

                canvas.paste(
                    preview,
                    (
                        x,
                        y,
                    ),
                )

                label = (
                    f"{item['video_id']} "
                    f"{frame['timestamp']}s "
                    f"[{frame['source']}]"
                )

                draw.text(
                    (
                        x + 4,
                        y - label_h + 4,
                    ),
                    label,
                    fill="black",
                )

    safe_subject = (
        str(subject)
        .replace(
            "+",
            "plus",
        )
        .replace(
            " ",
            "_",
        )
    )

    output = (
        PREVIEW_DIR
        / f"roi_preview_{safe_subject}.jpg"
    )

    canvas.save(
        output,
        quality=90,
    )


# =========================================================
# MAIN
# =========================================================

def main():
    print("=" * 72)
    print(
        "STEP 07 - CROP ROI KEYFRAMES"
    )
    print("=" * 72)

    print(
        "ROI hierarchy:"
    )

    print(
        "video_specific -> context -> "
        "subject_default -> global_default"
    )

    print(
        "No GT / no model score used."
    )

    ROI_KEYFRAME_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    PREVIEW_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    frames = load_keyframes()

    ocr_lookup = (
        load_ocr_lookup()
    )

    print()
    print(
        "Input keyframes:",
        len(frames),
    )

    print(
        "Videos:",
        frames[
            "video_id"
        ].nunique(),
    )

    print(
        "Keyframe audit PASS"
    )

    result_rows = []

    # Store representative previews
    preview_candidates = {}

    for index, row in (
        frames.iterrows()
    ):
        video_id = str(
            row["video_id"]
        ).strip()

        subject = str(
            row["subject"]
        ).strip()

        timestamp = int(
            row["timestamp_sec"]
        )

        frame_id = row[
            "frame_id"
        ]

        image_path = Path(
            str(
                row[
                    "image_file"
                ]
            )
        )

        if not valid_image(
            image_path
        ):
            fallback = (
                PROJECT_ROOT
                / "data"
                / "processed"
                / "ocr_best_integration"
                / "keyframes_full"
                / video_id
                / f"{video_id}_{timestamp:05d}s.png"
            )

            image_path = (
                fallback
            )

        print(
            f"[{index + 1}/{len(frames)}] "
            f"{video_id} - {timestamp}s",
            end="",
        )

        if not valid_image(
            image_path
        ):
            print(
                " FAILED: source missing"
            )

            result_rows.append(
                {
                    "video_id":
                        video_id,
                    "subject":
                        subject,
                    "frame_id":
                        frame_id,
                    "timestamp_sec":
                        timestamp,
                    "image_file":
                        "",
                    "roi_source":
                        "",
                    "context_label":
                        "",
                    "roi_confidence":
                        "",
                    "matched_patterns":
                        "",
                    "roi_top":
                        None,
                    "roi_left":
                        None,
                    "roi_right":
                        None,
                    "roi_bottom":
                        None,
                    "roi_coordinates":
                        "",
                    "status":
                        "failed",
                }
            )

            continue

        try:
            (
                roi,
                roi_source,
                context_label,
                confidence,
                matched_patterns,
            ) = resolve_roi(
                video_id,
                subject,
                frame_id,
                ocr_lookup,
            )

            with Image.open(
                image_path
            ) as image:
                cropped, box = (
                    crop_roi(
                        image,
                        roi,
                    )
                )

                output_dir = (
                    ROI_KEYFRAME_DIR
                    / video_id
                )

                output_dir.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                output_file = (
                    output_dir
                    / image_path.name
                )

                cropped.save(
                    output_file
                )

            if (
                not output_file.exists()
                or output_file.stat().st_size
                <= 0
            ):
                raise RuntimeError(
                    "ROI output invalid"
                )

            print(
                f" OK ({roi_source})"
            )

            result_rows.append(
                {
                    "video_id":
                        video_id,
                    "subject":
                        subject,
                    "frame_id":
                        frame_id,
                    "timestamp_sec":
                        timestamp,
                    "image_file":
                        str(
                            output_file.resolve()
                        ),
                    "roi_source":
                        roi_source,
                    "context_label":
                        context_label,
                    "roi_confidence":
                        confidence,
                    "matched_patterns":
                        matched_patterns,
                    "roi_top":
                        roi["top"],
                    "roi_left":
                        roi["left"],
                    "roi_right":
                        roi["right"],
                    "roi_bottom":
                        roi["bottom"],
                    "roi_coordinates":
                        (
                            f"{box[0]},{box[1]},"
                            f"{box[2]},{box[3]}"
                        ),
                    "status":
                        "success",
                }
            )

            preview_candidates.setdefault(
                video_id,
                [],
            ).append(
                {
                    "subject":
                        subject,
                    "timestamp":
                        timestamp,
                    "image_path":
                        str(
                            image_path
                        ),
                    "box":
                        box,
                    "source":
                        roi_source,
                }
            )

        except Exception as error:
            print(
                " FAILED:",
                type(error).__name__,
                str(error),
            )

            result_rows.append(
                {
                    "video_id":
                        video_id,
                    "subject":
                        subject,
                    "frame_id":
                        frame_id,
                    "timestamp_sec":
                        timestamp,
                    "image_file":
                        "",
                    "roi_source":
                        "",
                    "context_label":
                        "",
                    "roi_confidence":
                        "",
                    "matched_patterns":
                        "",
                    "roi_top":
                        None,
                    "roi_left":
                        None,
                    "roi_right":
                        None,
                    "roi_bottom":
                        None,
                    "roi_coordinates":
                        "",
                    "status":
                        "failed",
                }
            )

    # =====================================================
    # SAVE SUMMARY
    # =====================================================

    result = pd.DataFrame(
        result_rows
    )

    result.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    # =====================================================
    # ROI SOURCE SUMMARY
    # =====================================================

    success = result[
        result["status"]
        == "success"
    ].copy()

    source_summary = (
        success
        .groupby(
            [
                "subject",
                "roi_source",
            ],
            dropna=False,
        )
        .size()
        .reset_index(
            name="frames"
        )
    )

    source_summary.to_csv(
        ROI_SOURCE_SUMMARY_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    # =====================================================
    # BUILD 4 SUBJECT CONTACT SHEETS
    # =====================================================

    selected_by_subject = {}

    for video_id in DEV_IDS:
        candidates = (
            preview_candidates
            .get(
                video_id,
                [],
            )
        )

        if not candidates:
            continue

        candidate_count = len(
            candidates
        )

        indices = sorted(
            set(
                [
                    0,
                    candidate_count // 2,
                    candidate_count - 1,
                ]
            )
        )

        selected = [
            candidates[i]
            for i in indices
        ]

        # Very short videos may have <3 unique frames.
        while len(selected) < 3:
            selected.append(
                selected[-1]
            )

        subject = (
            selected[0][
                "subject"
            ]
        )

        selected_by_subject.setdefault(
            subject,
            [],
        ).append(
            {
                "video_id":
                    video_id,
                "frames":
                    selected[:3],
            }
        )

    for subject, rows in (
        selected_by_subject.items()
    ):
        rows = sorted(
            rows,
            key=lambda item:
                video_sort_key(
                    item[
                        "video_id"
                    ]
                ),
        )

        build_subject_preview(
            subject,
            rows,
        )

    # =====================================================
    # FINAL AUDIT
    # =====================================================

    success_count = int(
        (
            result["status"]
            == "success"
        ).sum()
    )

    failed_count = int(
        (
            result["status"]
            == "failed"
        ).sum()
    )

    duplicates = int(
        result.duplicated(
            [
                "video_id",
                "timestamp_sec",
            ]
        ).sum()
    )

    global_default_rows = (
        success[
            success[
                "roi_source"
            ]
            == "global_default"
        ]
    )

    global_default_videos = (
        sorted(
            global_default_rows[
                "video_id"
            ]
            .unique()
            .tolist(),
            key=video_sort_key,
        )
    )

    print()
    print("=" * 72)
    print(
        "ROI CROP FINAL AUDIT"
    )
    print("=" * 72)

    print(
        "Total:",
        len(result),
    )

    print(
        "Success:",
        success_count,
    )

    print(
        "Failed:",
        failed_count,
    )

    print(
        "Duplicates:",
        duplicates,
    )

    print()

    print(
        "ROI source totals:"
    )

    print(
        success[
            "roi_source"
        ]
        .value_counts()
        .to_string()
    )

    print()
    print(
        "ROI source by subject:"
    )

    print(
        success
        .groupby(
            [
                "subject",
                "roi_source",
            ]
        )
        .size()
        .unstack(
            fill_value=0
        )
        .to_string()
    )

    print()
    print(
        "Global-default frames:",
        len(
            global_default_rows
        ),
    )

    print(
        "Global-default videos:",
        global_default_videos,
    )

    print()
    print(
        "Summary:",
        OUTPUT_FILE,
    )

    print(
        "Source audit:",
        ROI_SOURCE_SUMMARY_FILE,
    )

    print(
        "Visual previews:",
        PREVIEW_DIR,
    )

    structural_pass = (
        len(result)
        == EXPECTED_FRAMES
        and success_count
        == EXPECTED_FRAMES
        and failed_count
        == 0
        and duplicates
        == 0
    )

    if structural_pass:
        print()
        print(
            "STEP 07 CROP COMPLETE"
        )

        print(
            "STRUCTURAL PASS - "
            "VISUAL ROI QA STILL REQUIRED"
        )
    else:
        print()
        print(
            "STEP 07 FAILED"
        )

        raise RuntimeError(
            "ROI crop structural audit failed."
        )


if __name__ == "__main__":
    main()