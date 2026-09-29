@'

# V2 Metric Implementation Audit

## 1. Purpose

This document records the metric definitions, score aggregation behavior, threshold application, evaluation units, and evaluation universe implemented in the project.

This audit documents the existing implementation. It does not introduce a new metric formula, retune the model, or modify frozen historical results.

## 2. Audited Implementation

The following files were audited:

- module1/p1_validation/07_nested_inner_tuning.py
- module1/p1_validation/08_evaluate_outer_folds.py
- module1/ocr_best_integration/11_compare_old_ocr_d_vs_new_ocr_f.py
- module1/ocr_best_integration/13_compare_old_vs_new_combined.py

## 3. Chunk Score Fusion

The P1 implementation constructs a chunk-level final score from normalized LDA and LSA scores:

    final_score = alpha * lda_norm + (1 - alpha) * lsa_norm

The fused score is rounded before video-level aggregation.

The frozen current P1 configuration uses:

- LDA weight = 0
- LSA weight = 1

Therefore, the locked P1 configuration is LSA-only.

## 4. Video-Level Aggregation

Chunk scores are grouped at the video-concept level using:

- video_id
- subject
- concept

The audited aggregation methods include:

- max
- mean
- top2_mean
- top3_mean

For top2_mean, the implementation uses the two largest available chunk scores and calculates their mean.

The frozen current P1 configuration uses:

    aggregation = max

The historical OLD combined system uses:

    aggregation = top2_mean

## 5. Threshold Application

Thresholding is applied after video-level aggregation.

The binary rule is:

    prediction = 1 if video_score >= threshold
    prediction = 0 otherwise

Historical Step 13 settings are:

OLD combined system:

- aggregation = top2_mean
- threshold = 0.40

NEW combined system:

- aggregation = max
- threshold = 0.50

V2 statistical analysis does not retune these settings.

## 6. Ground Truth

A video-concept pair has y_true = 1 when it exists in the frozen positive ground-truth annotations.

Otherwise, y_true = 0 within the defined evaluation universe.

Ground truth is not modified after model predictions are observed.

## 7. Macro Video Metrics

Metrics are first calculated separately for each video.

Macro Video metrics are then calculated by averaging the corresponding per-video values.

For example:

    Macro Video F1 = mean of the 40 individual video F1 values

The same interpretation applies to:

- Macro Video Precision
- Macro Video Recall
- Macro Video F1

The V2 primary endpoint is Macro Video F1.

## 8. Micro Metrics

Micro metrics are calculated once using the pooled evaluation rows.

These include:

- Micro Precision
- Micro Recall
- Micro F1

Therefore, Macro Video metrics and Micro metrics operate at different aggregation levels.

## 9. Step 13 Evaluation Universe

Step 13 does not evaluate all 40 x 39 possible video-concept combinations.

Each video is evaluated only against concepts belonging to the same subject.

The frozen Step 13 evaluation universe is:

- 40 DEV videos
- 390 same-subject video-concept pairs
- 153 positive pairs
- 237 negative pairs

Subject pair counts are:

- C++: 100
- Java: 90
- Python: 90
- SQL: 110

Each subject contains 10 DEV videos.

Therefore, the correct Step 13 evaluation universe is 390 same-subject video-concept pairs, not 1560 pairs.

## 10. Frozen Step 13 Results

OLD combined system:

- Macro Video Precision = 0.48056186868686873
- Macro Video Recall = 0.914811507936508
- Macro Video F1 = 0.5734297034664682
- Micro F1 = 0.6218097447795824

NEW combined system:

- Macro Video Precision = 0.656228354978355
- Macro Video Recall = 0.8515476190476191
- Macro Video F1 = 0.6681238206238206
- Micro F1 = 0.6666666666666666

Observed DEV Macro Video F1 difference:

    NEW - OLD = +0.09469411715735243

These values are DEV results, not independent final-test estimates.

## 11. Statistical Unit

The 390 rows are video-concept pairs, but multiple concept-pairs belong to the same video.

Therefore, new V2 dependence-aware analyses preserve the video as the statistical cluster.

The historical exact McNemar analysis is retained as a historical pair-level result only.

It is not treated as cluster-aware V2 inference.

## 12. Audit Conclusion

The audited implementation establishes:

- Primary endpoint: Macro Video F1
- Macro analysis unit: video
- Evaluation universe: 390 same-subject video-concept pairs
- DEV videos: 40
- Positive pairs: 153
- Negative pairs: 237
- Threshold applied after aggregation
- Ground truth frozen before V2 statistical analysis

No metric definition was changed as part of this audit.
'@ | Set-Content -Path docs/V2_METRIC_AUDIT.md -Encoding UTF8
