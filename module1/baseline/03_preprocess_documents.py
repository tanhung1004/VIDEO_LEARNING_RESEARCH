from pathlib import Path
import sys
import pandas as pd


# =========================================================
# 1. ĐƯỜNG DẪN PROJECT
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Cho phép file baseline gọi code dùng chung trong module1/src
sys.path.append(
    str(PROJECT_ROOT / "module1" / "src")
)

from preprocessing import preprocess_text


# Input từ Step 02
INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "transcript"
    / "transcript_documents.csv"
)

# Output sau preprocessing
OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "transcript"
    / "transcript_preprocessed.csv"
)


# =========================================================
# 2. MAIN
# =========================================================

def main():

    print("======================================")
    print("MODULE 1 - BASELINE")
    print("STEP 03 - PREPROCESS DOCUMENTS")
    print("======================================")

    # Kiểm tra file Step 02
    if not INPUT_FILE.exists():

        print("Không tìm thấy input:")
        print(INPUT_FILE)

        return

    # Đọc 5 document transcript
    df = pd.read_csv(INPUT_FILE)

    # Đảm bảo cột text không bị null
    df["text"] = (
        df["text"]
        .fillna("")
        .astype(str)
    )

    # =====================================================
    # 3. PREPROCESS
    # =====================================================

    df["processed_text"] = (
        df["text"]
        .apply(preprocess_text)
    )

    # Đếm số từ trước khi clean
    df["words_before"] = (
        df["text"]
        .apply(lambda text: len(text.split()))
    )

    # Đếm số từ sau khi clean
    df["words_after"] = (
        df["processed_text"]
        .apply(lambda text: len(text.split()))
    )

    # =====================================================
    # 4. XEM NHANH KẾT QUẢ
    # =====================================================

    for _, row in df.iterrows():

        print("\n--------------------------------------")

        print(
            "Video:",
            row["video_id"],
            "-",
            row["subject"]
        )

        print(
            "Words before:",
            row["words_before"]
        )

        print(
            "Words after:",
            row["words_after"]
        )

        print(
            "Preview:",
            row["processed_text"][:200]
        )

    # =====================================================
    # 5. LƯU OUTPUT
    # =====================================================

    df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print("\n======================================")
    print("HOÀN THÀNH STEP 03")
    print("Output:")
    print(OUTPUT_FILE)
    print("======================================")


if __name__ == "__main__":
    main()