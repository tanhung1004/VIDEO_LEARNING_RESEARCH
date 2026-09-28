# V2 Metric Implementation Audit

## 1. Purpose

This document records the metric and aggregation behavior currently implemented in the project.

The purpose of this audit is to document the existing implementation before V2 statistical analysis and methodology revisions.

This document does not redefine the metrics and does not introduce a new evaluation formula.

---

## 2. Audited Implementation Files

The following implementation files were inspected:

- `module1/p1_validation/07_nested_inner_tuning.py`
- `module1/p1_validation/08_evaluate_outer_folds.py`
- `module1/ocr_best_integration/11_compare_old_ocr_d_vs_new_ocr_f.py`
- `module1/ocr_best_integration/13_compare_old_vs_new_combined.py`

The audit focuses on:

- score fusion
- video-level aggregation
- threshold application
- Precision
- Recall
- F1
- Macro Video metrics
- Micro metrics
- evaluation universe

---

# 3. Score Fusion and Video-Level Aggregation

## 3.1 Chunk-level fused score

### Implementation

`module1/p1_validation/07_nested_inner_tuning.py`

Function:

`aggregate_scores(...)`

The implementation constructs a chunk-level `final_score` from normalized LDA and LSA scores:

```python
work["final_score"] = (
    alpha * work["lda_norm"]
    +
    (1.0 - alpha) * work["lsa_norm"]
)
```
