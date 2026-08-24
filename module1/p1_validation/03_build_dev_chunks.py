from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

sys.path.append(
    str(ROOT / "module1" / "src")
)

from preprocessing import preprocess_text


VIDEOS_PATH = ROOT / "data" / "raw" / "videos.csv"
TRANSCRIPT_DIR = ROOT / "data" / "raw" / "transcripts"

OUTPUT_DIR = (
    ROOT
    / "module1"
    / "results"
    / "p1_validation"
)

OUTPUT_PATH = OUTPUT_DIR / "dev_chunks_60s.csv"

CHUNK_SECONDS = 60

EXPECTED_IDS = {
    f"v{i}"
    for i in range(1, 41)
}


def build_chunks(video_id, subject):
    transcript_path = (
        TRANSCRIPT_DIR
        / f"{video_id}_transcript.csv"
    )

    if not transcript_path.exists():
        raise FileNotFoundError(
            f"Missing transcript: {transcript_path}"
        )

    df = pd.read_csv(
        transcript_path,
        dtype={
            "video_id": str,
            "subject": str,
            "text": str,
        },
    )

    required = {
        "start_sec",
        "end_sec",
        "text",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"{video_id}: transcript missing columns "
            f"{sorted(missing)}"
        )

    df["start_sec"] = pd.to_numeric(
        df["start_sec"],
        errors="coerce",
    )

    df["end_sec"] = pd.to_numeric(
        df["end_sec"],
        errors="coerce",
    )

    df["text"] = (
        df["text"]
        .fillna("")
        .astype(str)
    )

    df = df.dropna(
        subset=["start_sec", "end_sec"]
    )

    if df.empty:
        raise ValueError(
            f"{video_id}: transcript has no valid timing rows"
        )

    max_time = float(df["end_sec"].max())

    chunks = []

    chunk_id = 0
    start_time = 0.0

    while start_time <= max_time:
        end_time = (
            start_time + CHUNK_SECONDS
        )

        chunk_df = df[
            (df["start_sec"] < end_time)
            &
            (df["end_sec"] >= start_time)
        ]

        raw_text = " ".join(
            chunk_df["text"].tolist()
        )

        processed_text = preprocess_text(
            raw_text
        )

        if processed_text.strip():
            chunks.append(
                {
                    "video_id": video_id,
                    "subject": subject,
                    "method": "transcript_only",
                    "chunk_id": chunk_id,
                    "start_sec": start_time,
                    "end_sec": end_time,
                    "raw_text": raw_text,
                    "processed_text": processed_text,
                }
            )

        chunk_id += 1
        start_time += CHUNK_SECONDS

    return chunks


def main():
    print("=" * 72)
    print("P1.3 BUILD 40-VIDEO DEV CHUNKS")
    print("=" * 72)

    videos = pd.read_csv(
        VIDEOS_PATH,
        dtype=str,
        keep_default_na=False,
    )

    videos["video_id"] = (
        videos["video_id"]
        .str.strip()
        .str.lower()
    )

    videos["subject"] = (
        videos["subject"]
        .str.strip()
    )

    videos["split"] = (
        videos["split"]
        .str.strip()
        .str.lower()
    )

    dev = videos[
        videos["video_id"].isin(
            EXPECTED_IDS
        )
        &
        videos["split"].eq("dev")
    ].copy()

    actual_ids = set(
        dev["video_id"]
    )

    if actual_ids != EXPECTED_IDS:
        print("P1.3: FAIL")
        print(
            "Missing:",
            sorted(
                EXPECTED_IDS
                - actual_ids
            ),
        )
        print(
            "Extra:",
            sorted(
                actual_ids
                - EXPECTED_IDS
            ),
        )
        sys.exit(1)

    dev["_video_num"] = (
        dev["video_id"]
        .str.extract(
            r"^v(\d+)$",
            expand=False,
        )
        .astype(int)
    )

    dev = (
        dev
        .sort_values("_video_num")
        .reset_index(drop=True)
    )

    all_chunks = []

    per_video = []

    for _, row in dev.iterrows():
        video_id = row["video_id"]
        subject = row["subject"]

        chunks = build_chunks(
            video_id,
            subject,
        )

        all_chunks.extend(chunks)

        per_video.append(
            {
                "video_id": video_id,
                "subject": subject,
                "num_chunks": len(chunks),
            }
        )

        print(
            f"{video_id:<4} "
            f"{subject:<7} "
            f"chunks={len(chunks)}"
        )

    chunks_df = pd.DataFrame(
        all_chunks
    )

    summary_df = pd.DataFrame(
        per_video
    )

    # ------------------------------------------------------
    # AUDIT
    # ------------------------------------------------------

    errors = []

    found_ids = set(
        chunks_df["video_id"]
    )

    missing_chunk_videos = sorted(
        EXPECTED_IDS - found_ids
    )

    if missing_chunk_videos:
        errors.append(
            "Videos with no chunks: "
            f"{missing_chunk_videos}"
        )

    if chunks_df[
        "processed_text"
    ].fillna("").str.strip().eq("").any():
        errors.append(
            "Blank processed_text found"
        )

    duplicated = chunks_df.duplicated(
        subset=[
            "video_id",
            "chunk_id",
        ]
    ).sum()

    if duplicated:
        errors.append(
            f"Duplicate video/chunk IDs: "
            f"{duplicated}"
        )

    subject_video_counts = (
        summary_df
        .groupby("subject")["video_id"]
        .nunique()
        .sort_index()
    )

    print("\nVideos per subject:")
    print(
        subject_video_counts
        .to_string()
    )

    print(
        "\nTotal videos:",
        chunks_df["video_id"].nunique(),
    )

    print(
        "Total chunks:",
        len(chunks_df),
    )

    print(
        "Duplicate video/chunk pairs:",
        int(duplicated),
    )

    if errors:
        print("\nP1.3 BUILD DEV CHUNKS: FAIL")

        for i, error in enumerate(
            errors,
            start=1,
        ):
            print(f"{i}. {error}")

        print("NO OUTPUT WRITTEN.")
        sys.exit(1)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    chunks_df.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n" + "=" * 72)
    print("P1.3 BUILD DEV CHUNKS: PASS")
    print("40/40 DEV videos represented.")
    print(
        f"Chunk size: {CHUNK_SECONDS} seconds"
    )
    print(
        "Saved:",
        OUTPUT_PATH.relative_to(ROOT),
    )
    print(
        "Frozen baseline outputs were NOT modified."
    )


if __name__ == "__main__":
    main()