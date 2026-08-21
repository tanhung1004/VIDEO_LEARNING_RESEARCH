# Video Selection Protocol

## Dataset scope

The development dataset contains 40 educational videos:

- SQL: 10
- Python: 10
- Java: 10
- C++: 10

Existing pilot videos v1-v5 are retained.

New development videos:

- SQL: 8
- Python: 9
- Java: 8
- C++: 10

## Inclusion criteria

A candidate video is included only when:

1. It is an educational/tutorial video.
2. Its primary subject is SQL, Python, Java, or C++.
3. It contains meaningful instructional coverage of at least one concept from the subject-specific concept catalog.
4. It has usable English spoken content/transcript.
5. The video is not a duplicate, re-upload, or excerpt of another selected video.
6. The instructional content is sufficiently substantial for concept detection.
7. The video can be processed by the current transcript-based pipeline.

## Preferred duration

Videos approximately 5-30 minutes long are preferred.

Videos outside this range may be retained only when they remain suitable for the research task.

## Exclusion criteria

A candidate is excluded when:

- no usable transcript is available;
- the content is primarily non-English;
- it is a duplicate/re-upload;
- it is extremely short and contains insufficient instructional content;
- it is mainly promotional, entertainment, news, or non-instructional content;
- it does not meaningfully cover the current concept catalog;
- transcript/audio quality is unusable.

## Channel diversity

To reduce instructor/channel-specific bias, avoid selecting more than 2-3 videos from the same channel within one subject when alternatives are available.

## Content-style diversity

Where possible, each subject should include multiple presentation styles:

- slide_lecture
- code_tutorial
- ide_demo
- concept_explanation
- problem_solving
- mixed

No exact quota is imposed, but the selected set should not consist almost entirely of one presentation style.

## Anti-cherry-picking rule

Candidate inclusion or exclusion must not depend on baseline, OCR, fusion, or other model performance.

Model predictions must not be inspected before the candidate selection decision is finalized.

## Candidate screening

All screened candidates, including excluded videos, are recorded in:

data/raw/video_candidates.csv

with a decision and, when excluded, an exclusion reason.

## Final development IDs

Selected new videos will be assigned IDs only after screening:

- SQL: v6-v13
- Python: v14-v22
- Java: v23-v30
- C++: v31-v40

## Final test set

Videos v41-v60 will later be collected using the same eligibility criteria but kept completely separate from model tuning and cross-validation.
