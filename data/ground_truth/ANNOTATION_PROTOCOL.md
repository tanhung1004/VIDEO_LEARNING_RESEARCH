# Ground Truth Annotation Protocol

## Research task

The current task is learning-concept detection in educational videos.

## Current subject scope

The expanded dataset contains four educational subjects:

- SQL
- Python
- Java
- C++

Each video is evaluated only against the concept catalog of its assigned subject.

## Candidate concept space

Only concepts explicitly defined in:

data/concepts/concept_catalog.csv

are considered labels in the current experiment.

## Annotation evidence

Ground-truth annotation must be based on the instructional content of the complete video.

Annotators may use evidence from:

- spoken explanation;
- transcript;
- slides;
- source code;
- SQL queries;
- diagrams;
- demonstrations;
- other meaningful instructional visual content.

## Positive concept rule

A concept is labeled positive when it is meaningfully taught, explained, demonstrated, or used as an instructional focus in the video.

The mere appearance of a word is not sufficient to label a concept as positive.

## Negative concept rule

For a given video, all concepts belonging to that video's subject that are not annotated as positive are treated as negative during evaluation.

## Anti-leakage rule

Ground truth for a new video must be completed before inspecting model predictions for that video.

Baseline, OCR, fusion, or other model outputs must not be used to decide the original ground-truth labels.

If a later independent annotation review identifies a genuine annotation error, the correction must be documented separately.

## Existing pilot

The existing annotations for videos v1-v5 are retained as the original pilot ground truth.

## Development set

Videos v1-v40 form the development set.

Development videos may be used for training and video-level cross-validation.

## Final test set

Videos v41-v60 form the final held-out test set.

The final test videos must not be used to select:

- fusion weights;
- prediction thresholds;
- aggregation rules;
- chunk sizes;
- OCR preprocessing rules;
- other model hyperparameters.

## Annotation timing

For every newly added video:

1. assign subject;
2. inspect the educational video;
3. annotate applicable concepts;
4. mark annotation as completed;
5. only then allow the video to enter model experiments.
