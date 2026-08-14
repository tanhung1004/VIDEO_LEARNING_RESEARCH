import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


# =========================================================
# 1. STOPWORDS
# =========================================================

# Các từ bình thường là stopword,
# nhưng trong lập trình lại mang ý nghĩa kỹ thuật
PROTECTED_WORDS = {
    "for",
    "by",
    "where"
}

STOP_WORDS = (
    set(ENGLISH_STOP_WORDS)
    - PROTECTED_WORDS
)

# Một số từ nói thường gặp trong video nhưng không hữu ích
CUSTOM_STOP_WORDS = {
    "uh",
    "um",
    "yeah",
    "okay",
    "ok",
    "actually",
    "basically",
    "really",
    "like",
    "gonna",
    "wanna"
}

STOP_WORDS = STOP_WORDS.union(CUSTOM_STOP_WORDS)


# =========================================================
# 2. CLEAN TEXT
# =========================================================

def clean_text(text):
    """
    Làm sạch transcript trước khi đưa vào LDA / LSA.
    """

    # Đảm bảo luôn là string
    text = str(text)

    # Chuyển thành chữ thường
    text = text.lower()

    # Xóa URL
    text = re.sub(
        r"http\S+|www\S+",
        " ",
        text
    )

    # Chỉ giữ chữ và số
    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text
    )

    # Xóa khoảng trắng dư
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# =========================================================
# 3. REMOVE STOPWORDS
# =========================================================

def remove_stopwords(text):
    """
    Loại bỏ các từ ít mang ý nghĩa nội dung.
    """

    words = text.split()

    filtered_words = [
        word
        for word in words
        if word not in STOP_WORDS
        and len(word) > 1
    ]

    return " ".join(filtered_words)


# =========================================================
# 4. PREPROCESS HOÀN CHỈNH
# =========================================================

def preprocess_text(text):
    """
    Pipeline preprocessing chung cho Module 1.
    """

    text = clean_text(text)

    text = remove_stopwords(text)

    return text