from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT / "module1" / "baseline"
OUT_DIR = ROOT / "module1" / "results" / "p1_validation"
OUT_PATH = OUT_DIR / "baseline_config_context.txt"

TARGETS = {
    "04_build_chunks.py": [
        r"chunk",
        r"60",
    ],
    "05_run_lda.py": [
        r"CountVectorizer",
        r"LatentDirichletAllocation",
        r"fit_transform",
        r"\.fit\(",
        r"\.transform\(",
        r"n_components",
        r"n_topics",
        r"random_state",
        r"max_iter",
        r"max_features",
        r"ngram_range",
        r"max_df",
    ],
    "06_run_lsa.py": [
        r"TfidfVectorizer",
        r"TruncatedSVD",
        r"fit_transform",
        r"\.fit\(",
        r"\.transform\(",
        r"n_components",
        r"max_features",
        r"ngram_range",
        r"max_df",
    ],
    "07_run_lda_concept_matching.py": [
        r"normalize",
        r"score",
        r"concept",
    ],
    "08_fuse_lda_lsa.py": [
        r"0\.4",
        r"0\.6",
        r"alpha",
        r"normalize",
        r"min",
        r"max",
        r"lda",
        r"lsa",
    ],
    "09_build_video_concept_evidence.py": [
        r"top2",
        r"top3",
        r"mean",
        r"groupby",
        r"aggregation",
        r"nlargest",
    ],
    "10_predict_evaluate_baseline.py": [
        r"threshold",
        r"0\.4",
        r"0\.40",
        r"f1",
        r"precision",
        r"recall",
    ],
}

CV_PATTERNS = [
    r"StratifiedKFold",
    r"GroupKFold",
    r"KFold",
    r"train_test_split",
    r"cv_fold",
    r"dev_cv_folds",
    r"train_idx",
    r"test_idx",
    r"train_index",
    r"test_index",
]


def contexts(path, patterns, radius=2):
    lines = path.read_text(
        encoding="utf-8",
        errors="replace",
    ).splitlines()

    hit_lines = set()

    for i, line in enumerate(lines):
        for pattern in patterns:
            if re.search(pattern, line, flags=re.IGNORECASE):
                for j in range(
                    max(0, i - radius),
                    min(len(lines), i + radius + 1),
                ):
                    hit_lines.add(j)
                break

    if not hit_lines:
        return ["  [NO MATCHES]"]

    output = []
    previous = None

    for i in sorted(hit_lines):
        if previous is not None and i > previous + 1:
            output.append("  ...")

        output.append(
            f"{i + 1:4d} | {lines[i]}"
        )
        previous = i

    return output


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    report = []

    report.append("=" * 78)
    report.append("P1.2 FROZEN BASELINE CONFIGURATION CONTEXT")
    report.append("=" * 78)

    for filename, patterns in TARGETS.items():
        path = BASELINE / filename

        report.append("")
        report.append("=" * 78)
        report.append(filename)
        report.append("=" * 78)

        if not path.exists():
            report.append("[FILE MISSING]")
            continue

        report.extend(
            contexts(path, patterns, radius=2)
        )

    report.append("")
    report.append("=" * 78)
    report.append("EXPLICIT CROSS-VALIDATION / TRAIN-TEST CONTROL SEARCH")
    report.append("=" * 78)

    cv_found = False

    for filename in sorted(TARGETS.keys()):
        path = BASELINE / filename

        if not path.exists():
            continue

        result = contexts(
            path,
            CV_PATTERNS,
            radius=2,
        )

        if result != ["  [NO MATCHES]"]:
            cv_found = True
            report.append("")
            report.append(f"[{filename}]")
            report.extend(result)

    if not cv_found:
        report.append("")
        report.append(
            "NO EXPLICIT CV / TRAIN-TEST CONTROL FOUND "
            "IN BASELINE FILES 04-10."
        )

    text = "\n".join(report)

    OUT_PATH.write_text(
        text,
        encoding="utf-8",
    )

    print(text)

    print("\n" + "=" * 78)
    print("P1.2 CONFIG EXTRACTION: PASS")
    print(
        "Saved:",
        OUT_PATH.relative_to(ROOT),
    )
    print("=" * 78)


if __name__ == "__main__":
    main()