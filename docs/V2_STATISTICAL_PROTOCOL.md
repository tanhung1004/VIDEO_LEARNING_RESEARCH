@'

# V2 Statistical Protocol

## 1. Purpose

This document defines the statistical role of the current 40-video DEV dataset and the V2 methodology and statistical reporting protocol.

It distinguishes development evidence from future independent confirmatory evaluation.

The frozen P1 configuration and historical V1 artifacts are not modified.

## 2. DEV Dataset Role

The current dataset contains 40 DEV videos:

- 10 SQL
- 10 Python
- 10 Java
- 10 C++

These videos have already been used for model development, configuration selection, nested cross-validation, OCR analysis, and combined-system comparison.

Therefore, the 40 videos are a development dataset and not an untouched independent final test set.

Results from these videos are development-stage evidence.

## 3. Final Configuration

The locked P1 configuration is:

- chunk duration = 180 seconds
- LDA weight = 0
- LSA weight = 1
- aggregation = max
- threshold = 0.50

The configuration was selected using the DEV workflow.

V2 performs no further DEV tuning.

The statistical analyses do not change the threshold, aggregation, chunk duration, LDA/LSA weights, or ground truth after observing V2 results.

## 4. Primary Endpoint

The V2 primary endpoint is:

    Macro Video F1

Machine-readable identifier:

    macro_video_f1

The primary comparison is:

    NEW combined system minus OLD combined system

Frozen Step 13 values are:

- OLD combined system = 0.5734297034664682
- NEW combined system = 0.6681238206238206
- DEV difference = +0.09469411715735243

Macro Video F1 remains the single primary endpoint.

## 5. Secondary and Descriptive Metrics

Secondary or descriptive outcomes include:

- Macro Video Precision
- Macro Video Recall
- Macro Video F1 standard deviation
- Micro Precision
- Micro Recall
- Micro F1
- MCC
- Balanced Accuracy
- Specificity
- False Positive Rate
- PR-AUC
- ROC-AUC

These metrics do not replace Macro Video F1 as the primary endpoint.

## 6. Multiple Comparisons Policy

The primary statistical focus is Macro Video F1.

V2 does not add a formal hypothesis test for every secondary metric.

Secondary metrics are mainly reported using:

- point estimates
- NEW-minus-OLD differences
- uncertainty intervals
- DEV-stage interpretation

This multiple comparisons policy avoids unplanned mass hypothesis testing across many related secondary outcomes.

If future confirmatory work formally tests multiple secondary outcomes, the testing family and correction method must be specified before examining those results.

## 7. Recall

Recall is an important secondary metric because the Step 13 comparison contains a precision-recall trade-off.

Macro Video Recall:

- OLD combined system = 0.914812
- NEW combined system = 0.851548
- difference = -0.063264
- 95% DEV interval = [-0.105694, -0.027083]

Micro Recall:

- OLD combined system = 0.875817
- NEW combined system = 0.758170
- difference = -0.117647
- 95% DEV interval = [-0.188679, -0.051613]

These remain secondary DEV results.

## 8. Statistical Dependence

The frozen Step 13 evaluation universe contains:

- 40 DEV videos
- 390 same-subject video-concept pairs
- 153 positive pairs
- 237 negative pairs

Multiple concept-pairs belong to the same video.

For new V2 dependence-aware analyses, the video is preserved as the cluster.

All concept-pairs belonging to a sampled video remain together during video-cluster resampling.

## 9. Historical McNemar Analysis

The historical exact McNemar analysis uses individual video-concept pair correctness.

It is retained for reproducibility.

It is classified as a historical pair-level result only and is not treated as cluster-aware V2 inference.

## 10. Primary Cluster-Aware Analysis

Macro Video F1 is already a video-level quantity.

The V2 primary uncertainty analysis uses:

- subject-stratified video bootstrap
- resampling unit = video
- bootstrap iterations = 10000
- random state = 42

Observed DEV difference:

    +0.094694

95% DEV interval:

    [0.038772, 0.153846]

This is development-stage evidence, not independent confirmation.

## 11. Secondary Metric Uncertainty

V2 reports bootstrap uncertainty for the pre-specified secondary metrics without introducing mass new hypothesis tests.

For example:

Macro Video Precision:

- difference = +0.175666
- 95% CI = [0.119034, 0.236322]

Macro Video Recall:

- difference = -0.063264
- 95% CI = [-0.105694, -0.027083]

Micro F1:

- difference = +0.044857
- 95% CI = [-0.009056, 0.098567]

## 12. Subject-Level Analysis

Each subject contains only 10 DEV videos.

Subject-level results are therefore descriptive and have limited precision.

No subject-level hypothesis-testing family is introduced.

Subject Macro Video F1 differences are:

- C++: +0.183081
- Java: +0.109414
- Python: +0.085415
- SQL: +0.000867

The corresponding bootstrap uncertainty is reported in V2_SUBJECT_UNCERTAINTY.csv.

## 13. MDE / Sensitivity Analysis

V2 uses a Minimum Detectable Effect sensitivity analysis instead of automatic post-hoc power.

Assumptions:

- primary endpoint = Macro Video F1
- paired videos = 40
- two-sided alpha = 0.05
- paired-difference SD = 0.1977056446558819

At 80% target power:

- MDE = 0.089809
- standardized dz = 0.454257

This analysis is planning/sensitivity description only.

It is not post-hoc power and is not independent confirmatory evidence.

## 14. Naming Convention

The following canonical V2 names are used in documentation.

### P1 frozen transcript comparator

This refers specifically to the frozen transcript-only comparator.

It must not be confused with the OLD combined system.

### OLD combined system

Canonical human-readable name:

    OLD combined system

Historical artifacts may retain:

- OLD_COMBINED
- OLD_combined

Historical frozen files are not renamed.

### NEW combined system

Canonical human-readable name:

    NEW combined system

Historical artifacts may retain:

- NEW_COMBINED_F
- NEW_combined_F

Historical frozen files are not renamed.

## 15. Historical Artifact Preservation

V2 does not rename frozen V1 files, directories, CSV columns, Git tags, or historical result artifacts.

The explicit V2 naming convention is used only in new V2 documentation and outputs where it reduces ambiguity.

## 16. Independent Confirmation

The current 40-video DEV set cannot provide an independent confirmatory test of a system selected using the same development data.

A future independent dataset should be evaluated using the frozen configuration without retuning.

Independent confirmatory claims require such future data.

## 17. Final Protocol Summary

Dataset:

- 40 DEV videos

Evaluation universe:

- 390 same-subject video-concept pairs

Primary endpoint:

- Macro Video F1

Primary comparison:

- NEW combined system minus OLD combined system

Cluster-aware unit:

- video

Secondary metrics:

- uncertainty reported without mass hypothesis testing

Subject analysis:

- descriptive, 10 DEV videos per subject

MDE:

- planning/sensitivity analysis only

Further DEV tuning:

- no further DEV tuning

Independent confirmatory claim:

- not supported by the current DEV dataset
  '@ | Set-Content -Path docs/V2_STATISTICAL_PROTOCOL.md -Encoding UTF8
