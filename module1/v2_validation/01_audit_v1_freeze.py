from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pandas as pd

# ============================================================
# V1 FREEZE AUDIT
# Place this file at:
#   module1/v2_validation/01_audit_v1_freeze.py
#
# This script ONLY READS V1 artifacts and writes a V2 audit CSV.
# It does not rerun or modify P0/P1/OCR/Step 12/13.
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "module1" / "results" / "v2_validation"
OUTPUT_FILE = OUTPUT_DIR / "v1_freeze_audit.csv"

EXPECTED = {
    "dev_videos": 40,
    "concepts": 39,
    "gt_positives": 153,
    "folds": 5,
    "scaffold_old_chunks": 494,
    "scaffold_new_chunks": 172,
    "ocr_frames": 2398,
}

PATHS = {
    "videos": PROJECT_ROOT / "data" / "raw" / "videos.csv",
    "concepts": PROJECT_ROOT / "data" / "concepts" / "concept_catalog.csv",
    "gt": PROJECT_ROOT / "data" / "ground_truth" / "ground_truth_concepts.csv",
    "folds": PROJECT_ROOT / "data" / "splits" / "dev_cv_folds.csv",
    "p1_dir": PROJECT_ROOT / "module1" / "results" / "p1_validation",
    "scaffold_old": PROJECT_ROOT / "module1" / "results" / "scaffold40_compare" / "old" / "transcript_chunks.csv",
    "scaffold_new": PROJECT_ROOT / "module1" / "results" / "scaffold40_compare" / "new" / "transcript_chunks.csv",
    "ocr_fullframe": PROJECT_ROOT / "data" / "processed" / "ocr_best_integration" / "ocr_fullframe_cleaned.csv",
    "ocr_roi": PROJECT_ROOT / "data" / "processed" / "ocr_best_integration" / "ocr_roi_cleaned.csv",
    "combined_dir": PROJECT_ROOT / "module1" / "results" / "ocr_best_integration" / "combined_system_comparison",
    "step13_dir": PROJECT_ROOT / "module1" / "results" / "ocr_best_integration" / "combined_system_comparison" / "final_evaluation",
}

V1_DIRS = [
    PROJECT_ROOT / "module1" / "baseline",
    PROJECT_ROOT / "module1" / "proposed",
    PROJECT_ROOT / "module1" / "p0_dataset",
    PROJECT_ROOT / "module1" / "p1_validation",
    PROJECT_ROOT / "module1" / "scaffold40_compare",
    PROJECT_ROOT / "module1" / "ocr_best_integration",
]

def add(rows, check, status, expected, actual, path="", details=""):
    rows.append(
        {
            "check": check,
            "status": status,
            "expected": expected,
            "actual": actual,
            "path": str(path),
            "details": details,
        }
    )

def check_exists(rows, check, path, required=True):
    ok = path.exists()
    status = "PASS" if ok else ("FAIL" if required else "WARN")
    add(rows, check, status, "exists", "exists" if ok else "missing", path)
    return ok

def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)

def audit_v1_dirs(rows):
    for path in V1_DIRS:
        check_exists(rows, f"V1 directory: {path.name}", path)

def audit_videos(rows):
    if not check_exists(rows, "DEV videos source", PATHS["videos"]):
        return
    try:
        df = read_csv(PATHS["videos"])
        if "video_id" not in df.columns:
            add(rows, "DEV videos", "FAIL", "video_id column", "missing video_id", PATHS["videos"])
            return
        ids = df["video_id"].astype(str).str.strip()
        expected_ids = {f"v{i}" for i in range(1, EXPECTED["dev_videos"] + 1)}
        actual_ids = set(ids)
        dev_ids = actual_ids & expected_ids
        missing = sorted(expected_ids - actual_ids, key=lambda x: int(x[1:]))
        duplicates = int(ids.duplicated().sum())
        status = "PASS" if len(dev_ids) == EXPECTED["dev_videos"] and duplicates == 0 and not missing else "FAIL"
        add(
            rows,
            "DEV videos",
            status,
            f"{EXPECTED['dev_videos']} unique IDs v1-v40; no duplicate IDs",
            f"matching={len(dev_ids)}; duplicates={duplicates}; missing={missing}",
            PATHS["videos"],
        )
    except Exception as exc:
        add(rows, "DEV videos", "FAIL", "readable videos.csv", f"error: {exc}", PATHS["videos"])

def audit_concepts(rows):
    if not check_exists(rows, "Concept catalog source", PATHS["concepts"]):
        return
    try:
        df = read_csv(PATHS["concepts"])
        required = {"subject", "concept"}
        missing = sorted(required - set(df.columns))
        if missing:
            add(rows, "Concept catalog", "FAIL", "subject + concept columns", f"missing={missing}", PATHS["concepts"])
            return
        keys = (
            df["subject"].astype(str).str.strip().str.lower()
            + "::"
            + df["concept"].astype(str).str.strip().str.lower()
        )
        duplicates = int(keys.duplicated().sum())
        count = len(df)
        status = "PASS" if count == EXPECTED["concepts"] and duplicates == 0 else "FAIL"
        add(rows, "Concept catalog", status, f"{EXPECTED['concepts']} unique concepts", f"count={count}; duplicates={duplicates}", PATHS["concepts"])
    except Exception as exc:
        add(rows, "Concept catalog", "FAIL", f"{EXPECTED['concepts']} concepts", f"error: {exc}", PATHS["concepts"])

def audit_gt(rows):
    if not check_exists(rows, "Ground truth source", PATHS["gt"]):
        return
    try:
        df = read_csv(PATHS["gt"])
        required = {"video_id", "subject", "concept"}
        missing = sorted(required - set(df.columns))
        if missing:
            add(rows, "Ground truth", "FAIL", "video_id + subject + concept columns", f"missing={missing}", PATHS["gt"])
            return
        dev_ids = {f"v{i}" for i in range(1, EXPECTED["dev_videos"] + 1)}
        gt = df[df["video_id"].astype(str).str.strip().isin(dev_ids)].copy()
        keys = (
            gt["video_id"].astype(str).str.strip()
            + "::"
            + gt["subject"].astype(str).str.strip().str.lower()
            + "::"
            + gt["concept"].astype(str).str.strip().str.lower()
        )
        duplicates = int(keys.duplicated().sum())
        count = len(gt)
        status = "PASS" if count == EXPECTED["gt_positives"] and duplicates == 0 else "FAIL"
        add(rows, "Ground truth positives", status, f"{EXPECTED['gt_positives']} DEV positive rows; no duplicates", f"count={count}; duplicates={duplicates}", PATHS["gt"])
    except Exception as exc:
        add(rows, "Ground truth positives", "FAIL", f"{EXPECTED['gt_positives']} rows", f"error: {exc}", PATHS["gt"])

def audit_folds(rows):
    if not check_exists(rows, "DEV folds source", PATHS["folds"]):
        return
    try:
        df = read_csv(PATHS["folds"])
        fold_cols = [c for c in df.columns if "fold" in c.lower()]
        if not fold_cols:
            add(rows, "DEV folds", "FAIL", f"{EXPECTED['folds']} fold labels", f"no fold column; columns={df.columns.tolist()}", PATHS["folds"])
            return
        fold_col = fold_cols[0]
        unique = sorted(df[fold_col].dropna().astype(str).str.strip().unique().tolist())
        status = "PASS" if len(unique) == EXPECTED["folds"] else "FAIL"
        add(rows, "DEV folds", status, f"{EXPECTED['folds']} unique folds", f"column={fold_col}; folds={unique}", PATHS["folds"])
    except Exception as exc:
        add(rows, "DEV folds", "FAIL", f"{EXPECTED['folds']} folds", f"error: {exc}", PATHS["folds"])

def audit_p1(rows):
    p1 = PATHS["p1_dir"]
    # BUG FIX: the old version used `"PASS" if existing else "FAIL"`, which
    # is truthy on a NON-EMPTY list - it PASSED as long as ANY ONE of the
    # 3 required files existed, even with the other 2 missing. Split into
    # one check per required artifact (same style as the rest of this
    # file) so each file's presence is verified independently and a
    # partial/incomplete P1 freeze can no longer report as PASS.
    required = [
        p1 / "FINAL_MODEL_CONFIG.json",
        p1 / "P1_FINAL_LOCK_SUMMARY.txt",
        p1 / "p1_final_lock_audit.json",
    ]
    for path in required:
        check_exists(rows, f"P1 artifact: {path.name}", path)

    config_path = p1 / "FINAL_MODEL_CONFIG.json"
    if config_path.is_file():
        try:
            cfg = json.loads(config_path.read_text(encoding="utf-8"))
            add(
                rows,
                "P1 final configuration (content)",
                "WARN",
                "manually confirm NEW protocol values (chunk/lda/lsa/aggregation/threshold)",
                "see details",
                config_path,
                "FINAL_MODEL_CONFIG=" + json.dumps(cfg, ensure_ascii=False, sort_keys=True),
            )
        except Exception as exc:
            add(rows, "P1 final configuration (content)", "FAIL", "readable JSON", f"error: {exc}", config_path)

def audit_chunk_count(rows, name, path, expected):
    if not check_exists(rows, f"{name} source", path):
        return
    try:
        df = read_csv(path)
        count = len(df)
        status = "PASS" if count == expected else "FAIL"
        add(rows, name, status, f"{expected} rows", f"{count} rows", path)
    except Exception as exc:
        add(rows, name, "FAIL", f"{expected} rows", f"error: {exc}", path)

def audit_ocr(rows):
    for name, path in [
        ("OCR full-frame cleaned", PATHS["ocr_fullframe"]),
        ("OCR ROI cleaned", PATHS["ocr_roi"]),
    ]:
        if not check_exists(rows, f"{name} source", path):
            continue
        try:
            df = read_csv(path)
            count = len(df)
            status = "PASS" if count == EXPECTED["ocr_frames"] else "FAIL"
            add(rows, f"{name} frame count", status, f"{EXPECTED['ocr_frames']} rows", f"{count} rows", path)
        except Exception as exc:
            add(rows, f"{name} frame count", "FAIL", f"{EXPECTED['ocr_frames']} rows", f"error: {exc}", path)

def audit_step12(rows):
    cd = PATHS["combined_dir"]
    required = [
        cd / "COMBINED_BUILD_CONFIG.txt",
        cd / "combined_score_build_audit.csv",
        cd / "frozen_chunk_sources.csv",
        cd / "transcript_input_audit.csv",
        cd / "OLD_transcript_plus_D" / "fusion_scores.csv",
        cd / "NEW_transcript_plus_D" / "fusion_scores.csv",
        cd / "NEW_transcript_plus_E" / "fusion_scores.csv",
        cd / "NEW_transcript_plus_F" / "fusion_scores.csv",
    ]
    for path in required:
        check_exists(rows, f"Step 12 artifact: {path.name}", path)

def audit_step13(rows):
    sd = PATHS["step13_dir"]
    required = [
        "old_vs_new_combined_audit.csv",
        "old_vs_new_combined_bootstrap_deltas.csv",
        "old_vs_new_combined_overall_metrics.csv",
        "old_vs_new_combined_pair_scores.csv",
        "old_vs_new_combined_statistics.csv",
        "old_vs_new_combined_subject_metrics.csv",
        "old_vs_new_combined_video_metrics.csv",
        "OLD_VS_NEW_COMBINED_SUMMARY.txt",
    ]
    for filename in required:
        check_exists(rows, f"Step 13 artifact: {filename}", sd / filename)

def audit_git_tags(rows):
    try:
        completed = subprocess.run(
            ["git", "tag", "--list"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        tags = [x.strip() for x in completed.stdout.splitlines() if x.strip()]
        status = "PASS" if tags else "WARN"
        add(rows, "Git tags", status, "tags inspectable if used by project", "; ".join(tags) if tags else "no tags found", PROJECT_ROOT)
    except Exception as exc:
        add(rows, "Git tags", "WARN", "git tag command works", f"error: {exc}", PROJECT_ROOT)

def main():
    rows = []
    print("=" * 78)
    print("V1 FREEZE AUDIT")
    print("=" * 78)
    print(f"Project root: {PROJECT_ROOT}")

    audit_v1_dirs(rows)
    audit_videos(rows)
    audit_concepts(rows)
    audit_gt(rows)
    audit_folds(rows)
    audit_p1(rows)
    audit_chunk_count(rows, "Scaffold OLD chunks", PATHS["scaffold_old"], EXPECTED["scaffold_old_chunks"])
    audit_chunk_count(rows, "Scaffold NEW chunks", PATHS["scaffold_new"], EXPECTED["scaffold_new_chunks"])
    audit_ocr(rows)
    audit_step12(rows)
    audit_step13(rows)
    audit_git_tags(rows)

    df = pd.DataFrame(rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    pass_count = int((df["status"] == "PASS").sum())
    fail_count = int((df["status"] == "FAIL").sum())
    warn_count = int((df["status"] == "WARN").sum())

    print("\nSUMMARY")
    print(f"PASS: {pass_count}")
    print(f"FAIL: {fail_count}")
    print(f"WARN: {warn_count}")
    print(f"\nAudit file: {OUTPUT_FILE}")
    print("\nRESULTS")
    print("-" * 78)
    print(
        df[["check", "status", "expected", "actual"]]
        .to_string(index=False, justify="left")
    )
    print("-" * 78)

    if fail_count:
        raise SystemExit("V1 FREEZE AUDIT FAILED")

    print("\nV1 FREEZE AUDIT PASS")

if __name__ == "__main__":
    main()
