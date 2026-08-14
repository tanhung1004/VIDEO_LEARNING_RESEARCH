import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# CẤU HÌNH LDA
# =========================================================

NUM_TOPICS = 6
TOP_WORDS = 12


# =========================================================
# 1. TRAIN LDA
# =========================================================

def train_lda(documents):
    """
    Train LDA trên danh sách transcript chunks.

    Input
    -----
    documents:
        List processed transcript chunks.

    Output
    ------
    vectorizer:
        CountVectorizer chứa vocabulary.

    lda_model:
        LDA model đã train.

    document_topic_matrix:
        P(topic | document)
        shape = [num_documents, num_topics]
    """

    # -----------------------------------------------------
    # TEXT -> DOCUMENT TERM MATRIX
    # -----------------------------------------------------

    vectorizer = CountVectorizer(
        # unigram + bigram
        # ví dụ:
        # group
        # group by
        ngram_range=(1, 2),

        min_df=1,

        # bỏ những term xuất hiện gần như toàn corpus
        max_df=0.95,

        max_features=3000
    )

    document_term_matrix = (
        vectorizer.fit_transform(documents)
    )

    # -----------------------------------------------------
    # TRAIN LDA
    # -----------------------------------------------------

    lda_model = LatentDirichletAllocation(
        n_components=NUM_TOPICS,

        max_iter=50,

        learning_method="batch",

        # để chạy lại cho kết quả ổn định
        random_state=42
    )

    # P(topic | document)
    document_topic_matrix = (
        lda_model.fit_transform(
            document_term_matrix
        )
    )

    return (
        vectorizer,
        lda_model,
        document_topic_matrix
    )


# =========================================================
# 2. LẤY TOP WORDS CỦA TỪNG TOPIC
# =========================================================

def get_topic_words(
    lda_model,
    vectorizer,
    top_n=TOP_WORDS
):
    """
    Lấy các term quan trọng nhất của từng topic.

    Dùng cho:
        lda_topics.csv
    """

    feature_names = (
        vectorizer.get_feature_names_out()
    )

    topics = []

    for topic_id, weights in enumerate(
        lda_model.components_
    ):

        top_indices = (
            weights
            .argsort()[::-1][:top_n]
        )

        top_words = [
            feature_names[index]
            for index in top_indices
        ]

        topics.append({
            "topic_id": topic_id,
            "top_words": "; ".join(top_words)
        })

    return topics


# =========================================================
# 3. TOPIC -> WORD DISTRIBUTION
# =========================================================

def get_topic_word_distribution(
    lda_model
):
    """
    Chuyển trọng số word của từng LDA topic
    thành phân bố chuẩn hóa.

    Có thể hiểu gần giống:

        P(word | topic)

    Input shape:
        [num_topics, vocabulary_size]

    Output shape:
        [num_topics, vocabulary_size]
    """

    components = (
        lda_model
        .components_
        .astype(float)
    )

    # Tổng trọng số của từng topic
    row_sums = components.sum(
        axis=1,
        keepdims=True
    )

    # tránh chia cho 0
    row_sums[
        row_sums == 0
    ] = 1.0

    topic_word_distribution = (
        components / row_sums
    )

    return topic_word_distribution


# =========================================================
# 4. TOPIC <-> CONCEPT AFFINITY
# =========================================================

def calculate_topic_concept_affinity(
    lda_model,
    vectorizer,
    concept_texts
):
    """
    Đo độ liên quan giữa:

        từng latent topic
            và
        từng concept description

    Concept được vector hóa bằng chính vocabulary
    mà LDA đã học từ transcript.

    Output shape:

        [num_topics, num_concepts]
    """

    # -----------------------------------------------------
    # Topic representation
    # -----------------------------------------------------

    topic_word_distribution = (
        get_topic_word_distribution(
            lda_model
        )
    )

    # -----------------------------------------------------
    # Concept representation
    # -----------------------------------------------------

    concept_term_matrix = (
        vectorizer.transform(
            concept_texts
        )
    )

    # -----------------------------------------------------
    # Topic <-> Concept similarity
    # -----------------------------------------------------

    topic_concept_affinity = (
        cosine_similarity(
            topic_word_distribution,
            concept_term_matrix
        )
    )

    return topic_concept_affinity


# =========================================================
# 5. DOCUMENT -> CONCEPT SCORE
# =========================================================

def calculate_document_concept_scores(
    document_topic_matrix,
    topic_concept_affinity
):
    """
    Tính LDA Concept Score.

    Ý tưởng:

        score(document, concept)

        =
        SUM over topics:

        P(topic | document)
        *
        affinity(topic, concept)

    Matrix:

        [documents x topics]
                  @
        [topics x concepts]

                  =

        [documents x concepts]
    """

    document_concept_scores = (
        document_topic_matrix
        @ topic_concept_affinity
    )

    # đảm bảo nằm trong khoảng 0 -> 1
    document_concept_scores = (
        np.clip(
            document_concept_scores,
            0.0,
            1.0
        )
    )

    return document_concept_scores