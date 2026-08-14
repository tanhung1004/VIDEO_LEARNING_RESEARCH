import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# CẤU HÌNH
# =========================================================

MAX_COMPONENTS = 100


# =========================================================
# TRAIN LSA
# =========================================================

def train_lsa(
    document_texts,
    concept_texts
):
    """
    Xây dựng không gian LSA chung cho:
    - transcript chunks
    - concept descriptions
    """

    # Gộp hai loại text để chúng nằm
    # trong cùng một semantic space
    all_texts = (
        list(document_texts)
        + list(concept_texts)
    )

    # -----------------------------------------------------
    # 1. TF-IDF
    # -----------------------------------------------------

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=1,
        max_features=5000
    )

    tfidf_matrix = (
        vectorizer.fit_transform(all_texts)
    )

    # -----------------------------------------------------
    # 2. CHỌN SỐ CHIỀU SVD AN TOÀN
    # -----------------------------------------------------

    max_allowed = min(
        tfidf_matrix.shape[0] - 1,
        tfidf_matrix.shape[1] - 1,
        MAX_COMPONENTS
    )

    if max_allowed < 2:
        raise ValueError(
            "Dữ liệu quá nhỏ để chạy LSA."
        )

    n_components = max_allowed

    # -----------------------------------------------------
    # 3. SVD = PHẦN CHÍNH CỦA LSA
    # -----------------------------------------------------

    svd = TruncatedSVD(
        n_components=n_components,
        random_state=42
    )

    latent_matrix = (
        svd.fit_transform(tfidf_matrix)
    )

    # -----------------------------------------------------
    # 4. TÁCH DOCUMENT VÀ CONCEPT
    # -----------------------------------------------------

    num_documents = len(document_texts)

    document_vectors = (
        latent_matrix[:num_documents]
    )

    concept_vectors = (
        latent_matrix[num_documents:]
    )

    return (
        vectorizer,
        svd,
        document_vectors,
        concept_vectors
    )


# =========================================================
# COSINE SIMILARITY
# =========================================================

def calculate_similarity(
    document_vectors,
    concept_vectors
):
    """
    Trả về ma trận:

    rows    = transcript chunks
    columns = concepts
    """

    return cosine_similarity(
        document_vectors,
        concept_vectors
    )