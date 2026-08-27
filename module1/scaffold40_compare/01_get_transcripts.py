from pathlib import Path
import pandas as pd
from youtube_transcript_api import YouTubeTranscriptApi


# =========================================================
# 1. ĐƯỜNG DẪN PROJECT
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

VIDEOS_FILE = PROJECT_ROOT / "data" / "raw" / "videos.csv"

TRANSCRIPT_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "transcripts"
)

RESULT_DIR = (
    PROJECT_ROOT
    / "module1"
    / "results"
    / "baseline"
)


# Tạo folder nếu chưa tồn tại
TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# 2. HÀM LẤY YOUTUBE ID
# =========================================================

def get_youtube_id(url):
    """
    Chuyển URL YouTube thành YouTube video ID.
    """

    url = str(url).strip()

    if "watch?v=" in url:
        return url.split("watch?v=")[1].split("&")[0]

    if "youtu.be/" in url:
        return url.split("youtu.be/")[1].split("?")[0]

    return url


# =========================================================
# 3. ĐỔI GIÂY THÀNH HH:MM:SS
# =========================================================

def seconds_to_time(seconds):

    seconds = int(seconds)

    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60

    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


# =========================================================
# 4. LẤY TRANSCRIPT CỦA MỘT VIDEO
# =========================================================

def download_transcript(video_row):

    video_id = video_row["video_id"]
    subject = video_row["subject"]
    title = video_row["title"]
    url = video_row["url"]

    youtube_id = get_youtube_id(url)

    print("\n======================================")
    print("Video:", video_id)
    print("Subject:", subject)
    print("Title:", title)
    print("YouTube ID:", youtube_id)

    try:

        # Khởi tạo YouTube Transcript API
        api = YouTubeTranscriptApi()

        # Ưu tiên transcript tiếng Anh,
        # nếu không có thì thử tiếng Việt
        transcript = api.fetch(
            youtube_id,
            languages=["en", "vi"]
        )

        raw_data = transcript.to_raw_data()

        rows = []

        for item in raw_data:

            start = float(item["start"])
            duration = float(item.get("duration", 0))
            end = start + duration

            text = str(item["text"])
            text = text.replace("\n", " ").strip()

            rows.append({
                "video_id": video_id,
                "subject": subject,
                "youtube_id": youtube_id,
                "start_sec": round(start, 2),
                "end_sec": round(end, 2),
                "start_time": seconds_to_time(start),
                "end_time": seconds_to_time(end),
                "text": text
            })

        transcript_df = pd.DataFrame(rows)

        output_file = (
            TRANSCRIPT_DIR
            / f"{video_id}_transcript.csv"
        )

        transcript_df.to_csv(
            output_file,
            index=False,
            encoding="utf-8-sig"
        )

        print("OK - Đã lấy transcript")
        print("Số đoạn:", len(transcript_df))
        print("File:", output_file)

        return {
            "video_id": video_id,
            "subject": subject,
            "status": "success",
            "transcript_rows": len(transcript_df),
            "output_file": str(output_file)
        }

    except Exception as error:

        print("FAILED - Không lấy được transcript")
        print("Lỗi:", error)

        return {
            "video_id": video_id,
            "subject": subject,
            "status": "failed",
            "transcript_rows": 0,
            "output_file": ""
        }


# =========================================================
# 5. MAIN
# =========================================================

def main():

    print("======================================")
    print("MODULE 1 - BASELINE")
    print("STEP 01 - GET TRANSCRIPTS")
    print("======================================")

    if not VIDEOS_FILE.exists():
        print("Không tìm thấy:")
        print(VIDEOS_FILE)
        return

    videos = pd.read_csv(VIDEOS_FILE)

    # Chỉ chạy các video có status = selected
    selected_videos = videos[
        videos["status"].str.lower() == "selected"
    ].copy()

    print("Số video selected:", len(selected_videos))

    summary = []

    for _, video_row in selected_videos.iterrows():

        result = download_transcript(video_row)

        summary.append(result)

    # =====================================================
    # 6. LƯU FILE TỔNG KẾT
    # =====================================================

    summary_df = pd.DataFrame(summary)

    summary_file = (
        RESULT_DIR
        / "transcript_summary.csv"
    )

    summary_df.to_csv(
        summary_file,
        index=False,
        encoding="utf-8-sig"
    )

    print("\n======================================")
    print("HOÀN THÀNH STEP 01")
    print("Summary:", summary_file)
    print("======================================")


if __name__ == "__main__":
    main()