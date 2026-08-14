from pathlib import Path
import sys
import pandas as pd


# =========================================================
# 1. ĐƯỜNG DẪN PROJECT
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

sys.path.append(
    str(PROJECT_ROOT / "module1" / "src")
)

from preprocessing import preprocess_text


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

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "transcript"
    / "transcript_chunks.csv"
)


# Mỗi chunk dài 60 giây
CHUNK_SECONDS = 60


# =========================================================
# 2. CHIA TRANSCRIPT THÀNH CHUNK
# =========================================================

def build_chunks(video_id, subject):

    transcript_file = (
        TRANSCRIPT_DIR
        / f"{video_id}_transcript.csv"
    )

    if not transcript_file.exists():

        print(f"Không tìm thấy transcript: {video_id}")
        return []

    df = pd.read_csv(transcript_file)

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

    max_time = df["end_sec"].max()

    chunks = []

    chunk_id = 0
    start_time = 0

    while start_time <= max_time:

        end_time = start_time + CHUNK_SECONDS

        # Lấy transcript nằm trong khoảng thời gian chunk
        chunk_df = df[
            (df["start_sec"] < end_time)
            &
            (df["end_sec"] >= start_time)
        ]

        raw_text = " ".join(
            chunk_df["text"].tolist()
        )

        processed_text = preprocess_text(raw_text)

        # Chỉ lưu chunk thực sự có text
        if processed_text.strip():

            chunks.append({
                "video_id": video_id,
                "subject": subject,
                "method": "transcript_only",
                "chunk_id": chunk_id,
                "start_sec": start_time,
                "end_sec": end_time,
                "raw_text": raw_text,
                "processed_text": processed_text
            })

        chunk_id += 1
        start_time += CHUNK_SECONDS

    return chunks


# =========================================================
# 3. MAIN
# =========================================================

def main():

    print("======================================")
    print("MODULE 1 - BASELINE")
    print("STEP 04 - BUILD TRANSCRIPT CHUNKS")
    print("======================================")

    videos = pd.read_csv(VIDEOS_FILE)

    videos = videos[
        videos["status"]
        .astype(str)
        .str.lower()
        == "selected"
    ]

    all_chunks = []

    for _, row in videos.iterrows():

        video_id = row["video_id"]
        subject = row["subject"]

        print(
            f"\nĐang chia chunk: {video_id} - {subject}"
        )

        chunks = build_chunks(
            video_id,
            subject
        )

        all_chunks.extend(chunks)

        print(
            "Số chunk:",
            len(chunks)
        )

    chunks_df = pd.DataFrame(all_chunks)

    chunks_df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print("\n======================================")
    print("HOÀN THÀNH STEP 04")
    print("Tổng số chunk:", len(chunks_df))
    print("Output:")
    print(OUTPUT_FILE)
    print("======================================")


if __name__ == "__main__":
    main()