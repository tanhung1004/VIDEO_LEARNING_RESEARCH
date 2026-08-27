from pathlib import Path
import pandas as pd


# =========================================================
# 1. ĐƯỜNG DẪN PROJECT
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

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

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "transcript"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# 2. HÀM GOM TRANSCRIPT CỦA 1 VIDEO
# =========================================================

def build_document(video_id, subject):

    transcript_file = (
        TRANSCRIPT_DIR
        / f"{video_id}_transcript.csv"
    )

    if not transcript_file.exists():

        print(
            f"Không tìm thấy transcript của {video_id}"
        )

        return None

    transcript_df = pd.read_csv(transcript_file)

    # Bỏ các dòng text rỗng
    transcript_df["text"] = (
        transcript_df["text"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    transcript_df = transcript_df[
        transcript_df["text"] != ""
    ]

    # Ghép toàn bộ các đoạn transcript
    # thành 1 document cho video
    full_text = " ".join(
        transcript_df["text"].tolist()
    )

    # Xóa khoảng trắng dư
    full_text = " ".join(
        full_text.split()
    )

    return {
        "video_id": video_id,
        "subject": subject,
        "method": "transcript_only",
        "num_segments": len(transcript_df),
        "text": full_text
    }


# =========================================================
# 3. MAIN
# =========================================================

def main():

    print("======================================")
    print("MODULE 1 - BASELINE")
    print("STEP 02 - BUILD DOCUMENTS")
    print("======================================")

    if not VIDEOS_FILE.exists():

        print("Không tìm thấy videos.csv")
        return

    videos = pd.read_csv(VIDEOS_FILE)

    selected_videos = videos[
        videos["status"]
        .astype(str)
        .str.lower()
        == "selected"
    ]

    documents = []

    for _, row in selected_videos.iterrows():

        video_id = row["video_id"]
        subject = row["subject"]

        print(f"\nĐang xử lý {video_id} - {subject}")

        document = build_document(
            video_id,
            subject
        )

        if document is not None:

            documents.append(document)

            print(
                "Số transcript segment:",
                document["num_segments"]
            )

            print(
                "Độ dài document:",
                len(document["text"]),
                "ký tự"
            )

    # =====================================================
    # 4. LƯU CORPUS
    # =====================================================

    documents_df = pd.DataFrame(documents)

    output_file = (
        OUTPUT_DIR
        / "transcript_documents.csv"
    )

    documents_df.to_csv(
        output_file,
        index=False,
        encoding="utf-8-sig"
    )

    print("\n======================================")
    print("HOÀN THÀNH STEP 02")
    print("Số document:", len(documents_df))
    print("Output:", output_file)
    print("======================================")


if __name__ == "__main__":
    main()
