from pathlib import Path
import sys
import pandas as pd


# =========================================================
# STEP 04 - BUILD TRANSCRIPT CHUNKS
# OLD vs NEW, SAME 40 DEV VIDEOS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

sys.path.append(
    str(PROJECT_ROOT / "module1" / "src")
)

from preprocessing import preprocess_text

from step00_compare_config import (
    VARIANT,
    CHUNK_SECONDS,
    CHUNK_FILE,
)


# =========================================================
# 1. INPUT / OUTPUT
# =========================================================

VIDEOS_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "videos.csv"
)

TRANSCRIPT_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "transcripts"
)

OUTPUT_FILE = CHUNK_FILE


# =========================================================
# 2. BUILD CHUNKS FOR ONE VIDEO
# =========================================================

def build_chunks(video_id, subject):

    transcript_file = (
        TRANSCRIPT_DIR
        / f"{video_id}_transcript.csv"
    )

    if not transcript_file.exists():

        raise FileNotFoundError(
            f"Missing transcript for selected DEV video: "
            f"{video_id} -> {transcript_file}"
        )

    df = pd.read_csv(transcript_file)

    required_columns = {
        "start_sec",
        "end_sec",
        "text",
    }

    missing_columns = (
        required_columns
        - set(df.columns)
    )

    if missing_columns:

        raise RuntimeError(
            f"{video_id} transcript missing columns: "
            f"{sorted(missing_columns)}"
        )

    df["start_sec"] = pd.to_numeric(
        df["start_sec"],
        errors="coerce"
    )

    df["end_sec"] = pd.to_numeric(
        df["end_sec"],
        errors="coerce"
    )

    df["text"] = (
        df["text"]
        .fillna("")
        .astype(str)
    )

    df = df.dropna(
        subset=[
            "start_sec",
            "end_sec",
        ]
    ).copy()

    if df.empty:

        raise RuntimeError(
            f"Transcript has no valid timed rows: {video_id}"
        )

    max_time = df["end_sec"].max()

    chunks = []

    chunk_id = 0
    start_time = 0

    # Giữ nguyên logic chunk của scaffold cũ
    while start_time <= max_time:

        end_time = (
            start_time
            + CHUNK_SECONDS
        )

        chunk_df = df[
            (df["start_sec"] < end_time)
            &
            (df["end_sec"] >= start_time)
        ]

        raw_text = " ".join(
            chunk_df["text"].tolist()
        )

        processed_text = (
            preprocess_text(raw_text)
        )

        # Giữ nguyên rule cũ:
        # chỉ lưu chunk thực sự có text
        if processed_text.strip():

            chunks.append({
                "video_id": video_id,
                "subject": subject,
                "method": "transcript_only",
                "chunk_id": chunk_id,
                "start_sec": start_time,
                "end_sec": end_time,
                "raw_text": raw_text,
                "processed_text": processed_text,
            })

        chunk_id += 1
        start_time += CHUNK_SECONDS

    return chunks


# =========================================================
# 3. LOAD + AUDIT EXACT 40 DEV VIDEOS
# =========================================================

def load_dev_videos():

    videos = pd.read_csv(
        VIDEOS_FILE
    )

    required_columns = {
        "video_id",
        "subject",
        "status",
        "split",
    }

    missing_columns = (
        required_columns
        - set(videos.columns)
    )

    if missing_columns:

        raise RuntimeError(
            "videos.csv missing columns: "
            f"{sorted(missing_columns)}"
        )

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

    status_norm = (
        videos["status"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    split_norm = (
        videos["split"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    videos = videos[
        (status_norm == "selected")
        &
        (split_norm == "dev")
    ].copy()


    # -----------------------------------------------------
    # AUDIT 1: exactly 40 videos
    # -----------------------------------------------------

    if len(videos) != 40:

        raise RuntimeError(
            "Expected exactly 40 selected DEV videos. "
            f"Found: {len(videos)}"
        )


    # -----------------------------------------------------
    # AUDIT 2: no duplicate IDs
    # -----------------------------------------------------

    if videos["video_id"].duplicated().any():

        duplicates = videos[
            videos["video_id"].duplicated(
                keep=False
            )
        ]["video_id"].tolist()

        raise RuntimeError(
            f"Duplicate DEV video IDs: {duplicates}"
        )


    # -----------------------------------------------------
    # AUDIT 3: exact v1-v40
    # -----------------------------------------------------

    expected_ids = {
        f"v{i}"
        for i in range(1, 41)
    }

    actual_ids = set(
        videos["video_id"]
    )

    missing_ids = sorted(
        expected_ids - actual_ids
    )

    unexpected_ids = sorted(
        actual_ids - expected_ids
    )

    if missing_ids or unexpected_ids:

        raise RuntimeError(
            "DEV video ID mismatch. "
            f"Missing={missing_ids}, "
            f"Unexpected={unexpected_ids}"
        )


    # -----------------------------------------------------
    # AUDIT 4: 10 videos per subject
    # -----------------------------------------------------

    subject_counts = (
        videos["subject"]
        .value_counts()
        .to_dict()
    )

    expected_subject_counts = {
        "SQL": 10,
        "Python": 10,
        "Java": 10,
        "C++": 10,
    }

    if subject_counts != expected_subject_counts:

        raise RuntimeError(
            "Unexpected subject distribution. "
            f"Found: {subject_counts}"
        )


    # -----------------------------------------------------
    # AUDIT 5: all 40 transcript files exist
    # -----------------------------------------------------

    missing_transcripts = []

    for video_id in videos["video_id"]:

        transcript_file = (
            TRANSCRIPT_DIR
            / f"{video_id}_transcript.csv"
        )

        if not transcript_file.exists():

            missing_transcripts.append(
                video_id
            )

    if missing_transcripts:

        raise RuntimeError(
            "Missing transcript files for: "
            f"{missing_transcripts}"
        )


    # -----------------------------------------------------
    # deterministic v1 -> v40 order
    # -----------------------------------------------------

    videos["_video_number"] = (
        videos["video_id"]
        .str.extract(r"(\d+)")
        .astype(int)
    )

    videos = (
        videos
        .sort_values("_video_number")
        .drop(columns="_video_number")
        .reset_index(drop=True)
    )

    return videos


# =========================================================
# 4. MAIN
# =========================================================

def main():

    print(
        "======================================"
    )

    print(
        "SCAFFOLD40 OLD vs NEW COMPARISON"
    )

    print(
        "STEP 04 - BUILD TRANSCRIPT CHUNKS"
    )

    print(
        "======================================"
    )

    videos = load_dev_videos()


    # -----------------------------------------------------
    # INPUT AUDIT SUMMARY
    # -----------------------------------------------------

    print(
        "\nVariant:",
        VARIANT
    )

    print(
        "Chunk seconds:",
        CHUNK_SECONDS
    )

    print(
        "Selected DEV videos:",
        len(videos)
    )

    print(
        "Subject counts:"
    )

    print(
        videos["subject"]
        .value_counts()
        .to_dict()
    )

    print(
        "Video IDs:"
    )

    print(
        videos["video_id"]
        .tolist()
    )


    # -----------------------------------------------------
    # BUILD ALL CHUNKS
    # -----------------------------------------------------

    all_chunks = []

    for _, row in videos.iterrows():

        video_id = row["video_id"]
        subject = row["subject"]

        print(
            f"\nBuilding chunks: "
            f"{video_id} - {subject}"
        )

        chunks = build_chunks(
            video_id,
            subject
        )

        all_chunks.extend(
            chunks
        )

        print(
            "Chunks:",
            len(chunks)
        )


    chunks_df = pd.DataFrame(
        all_chunks
    )


    # -----------------------------------------------------
    # OUTPUT AUDIT
    # -----------------------------------------------------

    output_video_ids = set(
        chunks_df["video_id"]
        .astype(str)
    )

    expected_ids = {
        f"v{i}"
        for i in range(1, 41)
    }

    if output_video_ids != expected_ids:

        raise RuntimeError(
            "Chunk output does not contain exactly "
            "v1-v40. "
            f"Missing: "
            f"{sorted(expected_ids - output_video_ids)}"
        )


    duplicate_chunks = (
        chunks_df
        .duplicated(
            [
                "video_id",
                "chunk_id",
            ]
        )
        .sum()
    )

    if duplicate_chunks != 0:

        raise RuntimeError(
            "Duplicate video_id/chunk_id rows found: "
            f"{duplicate_chunks}"
        )


    # -----------------------------------------------------
    # SAVE
    # -----------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    chunks_df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )


    # -----------------------------------------------------
    # FINAL SUMMARY
    # -----------------------------------------------------

    print(
        "\n======================================"
    )

    print(
        "STEP 04 PASS"
    )

    print(
        "======================================"
    )

    print(
        "Variant:",
        VARIANT
    )

    print(
        "DEV videos:",
        chunks_df["video_id"].nunique()
    )

    print(
        "Total chunks:",
        len(chunks_df)
    )

    print(
        "Duplicate chunks:",
        duplicate_chunks
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