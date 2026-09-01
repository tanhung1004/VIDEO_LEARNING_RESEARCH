from pathlib import Path
import hashlib
import re
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

BASELINE_DIR = ROOT / "module1" / "baseline"
RESULT_DIR = ROOT / "module1" / "results" / "p1_validation"

HASH_PATH = RESULT_DIR / "baseline_file_hashes.csv"
AUDIT_PATH = RESULT_DIR / "baseline_static_audit.csv"
SUMMARY_PATH = RESULT_DIR / "baseline_audit_summary.txt"


EXPECTED_FILES = [
    "01_get_transcripts.py",
    "02_build_documents.py",
    "03_preprocess_documents.py",
    "04_build_chunks.py",
    "05_run_lda.py",
    "06_run_lsa.py",
    "07_run_lda_concept_matching.py",
    "08_fuse_lda_lsa.py",
    "09_build_video_concept_evidence.py",
    "10_predict_evaluate_baseline.py",
]


PATTERNS = {
    "vectorizer": [
        r"CountVectorizer",
        r"TfidfVectorizer",
    ],
    "lda": [
        r"LatentDirichletAllocation",
        r"\bLDA\b",
        r"n_topics",
        r"num_topics",
    ],
    "lsa": [
        r"TruncatedSVD",
        r"\bLSA\b",
        r"n_components",
    ],
    "fit_operation": [
        r"\.fit\s*\(",
        r"\.fit_transform\s*\(",
    ],
    "transform_operation": [
        r"\.transform\s*\(",
    ],
    "chunk_config": [
        r"chunk",
        r"CHUNK",
    ],
    "random_state": [
        r"random_state",
    ],
    "max_iter": [
        r"max_iter",
    ],
    "max_features": [
        r"max_features",
    ],
    "ngram": [
        r"ngram_range",
    ],
    "fusion": [
        r"alpha",
        r"lda.*lsa",
        r"lsa.*lda",
        r"0\.4",
        r"0\.6",
    ],
    "aggregation": [
        r"top2",
        r"top3",
        r"mean",
        r"aggregation",
    ],
    "threshold": [
        r"threshold",
        r"0\.40",
        r"0\.4",
    ],
    "normalization": [
        r"normalize",
        r"normalization",
        r"minmax",
        r"MinMax",
    ],
    "cv_control": [
        r"cv_fold",
        r"dev_cv_folds",
        r"StratifiedKFold",
        r"GroupKFold",
        r"\bKFold\b",
        r"train_idx",
        r"test_idx",
        r"train_index",
        r"test_index",
    ],
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)

    return h.hexdigest()


def normalize_line(line: str) -> str:
    return line.strip().replace("\t", " ")


def main():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 76)
    print("P1.1 FROZEN BASELINE STATIC AUDIT")
    print("=" * 76)

    # --------------------------------------------------------------
    # 1. Verify frozen baseline files exist
    # --------------------------------------------------------------

    missing = []

    for name in EXPECTED_FILES:
        path = BASELINE_DIR / name

        if not path.exists():
            missing.append(name)

    if missing:
        print("\nP1.1 BASELINE AUDIT: FAIL")
        print("Missing baseline files:")

        for name in missing:
            print(f"  - {name}")

        sys.exit(1)

    print(f"\nBaseline files found : {len(EXPECTED_FILES)}/10")

    # --------------------------------------------------------------
    # 2. Hash frozen baseline files
    # --------------------------------------------------------------

    hash_rows = []

    for name in EXPECTED_FILES:
        path = BASELINE_DIR / name

        hash_rows.append(
            {
                "file": name,
                "sha256": sha256_file(path),
            }
        )

    hash_df = pd.DataFrame(hash_rows)

    hash_df.to_csv(
        HASH_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------------
    # 3. Static source-code audit
    # --------------------------------------------------------------

    audit_rows = []

    for name in EXPECTED_FILES:
        path = BASELINE_DIR / name

        text = path.read_text(
            encoding="utf-8",
            errors="replace",
        )

        lines = text.splitlines()

        for line_no, raw_line in enumerate(lines, start=1):
            line = normalize_line(raw_line)

            if not line:
                continue

            for category, patterns in PATTERNS.items():
                for pattern in patterns:
                    if re.search(
                        pattern,
                        line,
                        flags=re.IGNORECASE,
                    ):
                        audit_rows.append(
                            {
                                "file": name,
                                "line_no": line_no,
                                "category": category,
                                "matched_pattern": pattern,
                                "code": line,
                            }
                        )

                        break

    audit_df = pd.DataFrame(
        audit_rows,
        columns=[
            "file",
            "line_no",
            "category",
            "matched_pattern",
            "code",
        ],
    )

    audit_df.to_csv(
        AUDIT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------------
    # 4. Important scientific indicators
    # --------------------------------------------------------------

    categories = set(audit_df["category"]) if len(audit_df) else set()

    has_fit = "fit_operation" in categories
    has_cv = "cv_control" in categories

    fit_hits = audit_df[
        audit_df["category"] == "fit_operation"
    ].copy()

    cv_hits = audit_df[
        audit_df["category"] == "cv_control"
    ].copy()

    print(f"SHA256 hashes written : {len(hash_df)}")
    print(f"Relevant code hits    : {len(audit_df)}")
    print(
        "Model fit operations : "
        + ("FOUND" if has_fit else "NOT FOUND")
    )
    print(
        "CV controls detected : "
        + ("FOUND" if has_cv else "NOT FOUND")
    )

    # --------------------------------------------------------------
    # 5. Print important lines only
    # --------------------------------------------------------------

    important_categories = [
        "vectorizer",
        "lda",
        "lsa",
        "fit_operation",
        "chunk_config",
        "random_state",
        "max_iter",
        "max_features",
        "ngram",
        "fusion",
        "aggregation",
        "threshold",
        "normalization",
        "cv_control",
    ]

    print("\n" + "=" * 76)
    print("IMPORTANT BASELINE CONFIGURATION HITS")
    print("=" * 76)

    for category in important_categories:
        rows = audit_df[
            audit_df["category"] == category
        ]

        if rows.empty:
            continue

        print(f"\n[{category}]")

        # Avoid flooding terminal with repetitive hits
        for _, row in rows.head(15).iterrows():
            print(
                f"{row['file']}:{row['line_no']} | "
                f"{row['code']}"
            )

        if len(rows) > 15:
            print(
                f"... {len(rows) - 15} additional hits "
                f"saved in {AUDIT_PATH.name}"
            )

    # --------------------------------------------------------------
    # 6. Build human-readable summary
    # --------------------------------------------------------------

    summary_lines = [
        "P1.1 FROZEN BASELINE STATIC AUDIT",
        "=" * 60,
        "",
        f"Baseline directory: {BASELINE_DIR.relative_to(ROOT)}",
        f"Baseline files: {len(EXPECTED_FILES)}/10",
        f"Relevant source-code hits: {len(audit_df)}",
        "",
        f"Model fit operations detected: {'YES' if has_fit else 'NO'}",
        f"Cross-validation controls detected: {'YES' if has_cv else 'NO'}",
        "",
    ]

    if has_fit and not has_cv:
        summary_lines.extend(
            [
                "SCIENTIFIC INTERPRETATION:",
                (
                    "The frozen pilot pipeline contains model/vectorizer "
                    "fitting operations but no explicit CV/fold controls "
                    "were detected by this static audit."
                ),
                (
                    "Therefore the frozen pilot scripts must not be used "
                    "directly as the P1 cross-validation evaluator."
                ),
                (
                    "P1 must implement train-only fitting and held-out "
                    "transformation in a separate validation pipeline."
                ),
            ]
        )
    else:
        summary_lines.extend(
            [
                "SCIENTIFIC INTERPRETATION:",
                (
                    "Static audit alone is insufficient to conclude "
                    "whether train/test leakage exists."
                ),
                (
                    "Relevant fit and fold handling must be reviewed "
                    "before implementing P1."
                ),
            ]
        )

    SUMMARY_PATH.write_text(
        "\n".join(summary_lines),
        encoding="utf-8",
    )

    print("\n" + "=" * 76)
    print("AUDIT ARTIFACTS")
    print("=" * 76)

    print(
        f"Hashes : {HASH_PATH.relative_to(ROOT)}"
    )
    print(
        f"Hits   : {AUDIT_PATH.relative_to(ROOT)}"
    )
    print(
        f"Summary: {SUMMARY_PATH.relative_to(ROOT)}"
    )

    print("\n" + "=" * 76)
    print("P1.1 BASELINE STATIC AUDIT: PASS")
    print("=" * 76)

    if has_fit and not has_cv:
        print(
            "Potential CV leakage risk confirmed for follow-up review:"
        )
        print(
            "fitting operations exist, but no fold-aware controls "
            "were detected."
        )
        print(
            "Frozen baseline remains untouched."
        )
    else:
        print(
            "Audit completed. Review detected configuration "
            "before implementing P1 CV."
        )


if __name__ == "__main__":
    main()