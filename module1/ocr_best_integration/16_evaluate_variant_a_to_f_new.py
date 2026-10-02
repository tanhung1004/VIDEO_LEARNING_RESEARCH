from pathlib import Path
import importlib.util

import pandas as pd

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 30)


# ============================================================
# STEP 16 - COMPARE A-F (NEW PROTOCOL) AGAINST GROUND TRUTH
#
# NEZ yeu cau (thay vi chi so B vs C): so sanh ca 6 variant A-F,
# tat ca cung chay duoi NEW protocol (180s / LDA=0 / LSA=1 / max / th=0.50).
#
# Nguon fusion_scores.csv cho tung variant - TAT CA deu tai su dung
# file co san, KHONG script nao trong day tu build lai score:
#
#   A = transcript-only            -> co san:
#       results/scaffold40_compare/new/fusion_scores.csv
#   B = OCR-only, full-frame clean -> can chay step 14 truoc
#       results/.../combined_system_comparison/NEW_B_ocr_only_fullframe/fusion_scores.csv
#   C = transcript + OCR raw       -> can chay step 15 truoc
#       results/.../combined_system_comparison/NEW_C_transcript_plus_raw_fullframe/fusion_scores.csv
#   D = transcript + OCR full-frame cleaned -> co san (step 12 da chay):
#       results/.../combined_system_comparison/NEW_transcript_plus_D/fusion_scores.csv
#   E = transcript + OCR ROI cleaned        -> co san:
#       results/.../combined_system_comparison/NEW_transcript_plus_E/fusion_scores.csv
#   F = mean(D, E)                          -> co san:
#       results/.../combined_system_comparison/NEW_transcript_plus_F/fusion_scores.csv
#
# LUU Y QUAN TRONG (tu data_structure.txt NEZ gui):
#   data/processed/ocr_best_integration/ chi co ocr_fullframe_cleaned.csv
#   va ocr_roi_cleaned.csv - KHONG co ocr_fullframe_raw.csv. Nghia la
#   step 15 (build C) hien KHONG chay duoc, va file C se bi thieu.
#   Script nay KHONG dung lai vi thieu C - no se SKIP C, in canh bao
#   ro rang, va van ra ket qua day du cho A/B/D/E/F. Khi nao NEZ tai
#   tao xong ocr_fullframe_raw.csv (chay lai step 04) va chay step 15,
#   chi can chay lai step 16 nay, C se tu duoc gop vao bang so sanh.
#
# Dung DUNG cac ham cua step 13 (load_videos/load_concepts/load_gt/
# build_pair_space/load_scores/build_video_evidence/compute_metrics/
# calculate_map) - khong viet lai cong thuc metric, khong sua step 13.
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parents[2]

OCR_BEST_DIR = PROJECT_ROOT / "module1" / "ocr_best_integration"

COMBINED_ROOT = (
    PROJECT_ROOT / "module1" / "results"
    / "ocr_best_integration" / "combined_system_comparison"
)

SCAFFOLD40_NEW_DIR = PROJECT_ROOT / "module1" / "results" / "scaffold40_compare" / "new"

OUTPUT_DIR = COMBINED_ROOT / "variant_a_to_f_evaluation"

PAIR_FILE = OUTPUT_DIR / "a_to_f_pair_scores.csv"
VIDEO_FILE = OUTPUT_DIR / "a_to_f_video_metrics.csv"
METRICS_FILE = OUTPUT_DIR / "a_to_f_overall_metrics.csv"
AUDIT_FILE = OUTPUT_DIR / "a_to_f_evaluation_audit.csv"


def load_python_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


STEP13 = load_python_module(
    "step13_old_vs_new_combined_eval",
    OCR_BEST_DIR / "13_compare_old_vs_new_combined.py",
)

# Which variants are REQUIRED (script fails if missing) vs OPTIONAL
# (script skips with a warning if the file isn't there yet).
SYSTEMS = {
    "A": {
        "name": "A_transcript_only_NEW",
        "path": SCAFFOLD40_NEW_DIR / "fusion_scores.csv",
        "required": True,
    },
    "B": {
        "name": "B_ocr_only_fullframe_NEW",
        "path": COMBINED_ROOT / "NEW_B_ocr_only_fullframe" / "fusion_scores.csv",
        "required": True,
    },
    "C": {
        "name": "C_transcript_plus_raw_ocr_NEW",
        "path": COMBINED_ROOT / "NEW_C_transcript_plus_raw_fullframe" / "fusion_scores.csv",
        "required": False,
    },
    "D": {
        "name": "D_transcript_plus_ocr_fullframe_cleaned_NEW",
        "path": COMBINED_ROOT / "NEW_transcript_plus_D" / "fusion_scores.csv",
        "required": True,
    },
    "E": {
        "name": "E_transcript_plus_ocr_roi_cleaned_NEW",
        "path": COMBINED_ROOT / "NEW_transcript_plus_E" / "fusion_scores.csv",
        "required": True,
    },
    "F": {
        "name": "F_mean_D_E_NEW",
        "path": COMBINED_ROOT / "NEW_transcript_plus_F" / "fusion_scores.csv",
        "required": True,
    },
}


def main():
    print("=" * 72)
    print("STEP 16 - COMPARE VARIANT A-F (NEW PROTOCOL) VS GROUND TRUTH")
    print("=" * 72)
    print()
    print(f"Aggregation = {STEP13.NEW_AGGREGATION}")
    print(f"Threshold   = {STEP13.NEW_THRESHOLD:.2f}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # 1. SHARED GT / CONCEPT / PAIR SPACE (identical for every variant)
    # --------------------------------------------------------

    videos = STEP13.load_videos()
    concepts = STEP13.load_concepts()
    gt = STEP13.load_gt()
    pairs = STEP13.build_pair_space(videos, concepts, gt)

    # --------------------------------------------------------
    # 2. LOAD EACH AVAILABLE VARIANT + AGGREGATE + THRESHOLD (max / 0.50)
    # --------------------------------------------------------

    audit_rows = []
    available_systems = []

    for system, info in SYSTEMS.items():
        path = info["path"]

        if not path.exists():
            message = f"score file not found: {path}"

            if info["required"]:
                raise FileNotFoundError(
                    f"Variant {system} ({info['name']}) is REQUIRED but "
                    f"{message}\nChay step tuong ung truoc khi chay step 16."
                )

            print()
            print(f"[SKIP] Variant {system} ({info['name']}): {message}")
            print(
                "       -> B qua trong bang so sanh lan nay. "
                "Chay step 15 sau khi co ocr_fullframe_raw.csv de bo sung C."
            )

            audit_rows.append(
                {
                    "check": f"{system}_score_rows",
                    "value": "SKIPPED (file missing)",
                    "expected": STEP13.EXPECTED_NEW_SCORE_ROWS,
                    "pass": None,
                }
            )
            continue

        scores = STEP13.load_scores(
            path,
            f"Variant {system} ({info['name']})",
            expected_rows=STEP13.EXPECTED_NEW_SCORE_ROWS,
        )

        evidence = STEP13.build_video_evidence(
            scores,
            score_name=f"score_{system}",
            aggregation=STEP13.NEW_AGGREGATION,
        )

        pairs = pairs.merge(
            evidence,
            on=["video_id", "subject_norm", "concept_norm"],
            how="left",
        )

        pairs[f"score_{system}"] = pairs[f"score_{system}"].fillna(0.0)
        pairs[f"score_{system}_chunks_used"] = (
            pairs[f"score_{system}_chunks_used"].fillna(0).astype(int)
        )

        pairs[f"pred_{system}"] = (
            pairs[f"score_{system}"] >= STEP13.NEW_THRESHOLD
        ).astype(int)

        available_systems.append(system)

        audit_rows.append(
            {
                "check": f"{system}_score_rows",
                "value": len(scores),
                "expected": STEP13.EXPECTED_NEW_SCORE_ROWS,
                "pass": len(scores) == STEP13.EXPECTED_NEW_SCORE_ROWS,
            }
        )

    if not available_systems:
        raise RuntimeError("Khong co variant nao co san de danh gia.")

    print()
    print("Variants co trong lan so sanh nay:", ", ".join(available_systems))

    # --------------------------------------------------------
    # 3. PER-VIDEO MACRO METRICS (compute_metrics per video, per system)
    # build_video_metrics() in step 13 is hardcoded to OLD/NEW, so this
    # loop reuses the lower-level compute_metrics() directly instead.
    # --------------------------------------------------------

    video_rows = []
    for (video_id, subject), group in pairs.groupby(["video_id", "subject"], sort=False):
        row = {
            "video_id": video_id,
            "subject": subject,
            "concepts": len(group),
            "gt_positive": int(group["y_true"].sum()),
        }
        for system in available_systems:
            metrics = STEP13.compute_metrics(
                group["y_true"],
                group[f"pred_{system}"],
                group[f"score_{system}"],
            )
            for key, value in metrics.items():
                row[f"{system}_{key}"] = value
        video_rows.append(row)

    video_metrics = pd.DataFrame(video_rows)
    video_metrics["_order"] = video_metrics["video_id"].map(STEP13.video_sort_key)
    video_metrics = (
        video_metrics.sort_values("_order").drop(columns=["_order"]).reset_index(drop=True)
    )

    if len(video_metrics) != 40:
        raise RuntimeError("Per-video output is not 40 rows.")

    # --------------------------------------------------------
    # 4. OVERALL (macro video F1 + micro pair-level metrics + mAP)
    # --------------------------------------------------------

    overall_rows = []
    for system in available_systems:
        info = SYSTEMS[system]

        pair_metrics = STEP13.compute_metrics(
            pairs["y_true"], pairs[f"pred_{system}"], pairs[f"score_{system}"]
        )
        overall_rows.append(
            {
                "variant": system,
                "system": info["name"],
                "aggregation": STEP13.NEW_AGGREGATION,
                "threshold": STEP13.NEW_THRESHOLD,
                "macro_video_precision": float(video_metrics[f"{system}_precision"].mean()),
                "macro_video_recall": float(video_metrics[f"{system}_recall"].mean()),
                "macro_video_f1": float(video_metrics[f"{system}_f1"].mean()),
                "macro_video_f1_std": float(video_metrics[f"{system}_f1"].std(ddof=1)),
                "micro_precision": pair_metrics["precision"],
                "micro_recall": pair_metrics["recall"],
                "micro_f1": pair_metrics["f1"],
                "mcc": pair_metrics["mcc"],
                "balanced_accuracy": pair_metrics["balanced_accuracy"],
                "specificity": pair_metrics["specificity"],
                "fpr": pair_metrics["fpr"],
                "pr_auc": pair_metrics["pr_auc"],
                "roc_auc": pair_metrics["roc_auc"],
                "mAP_video": STEP13.calculate_map(pairs, system),
                "TP": pair_metrics["TP"],
                "FP": pair_metrics["FP"],
                "FN": pair_metrics["FN"],
                "TN": pair_metrics["TN"],
                "predicted_positive": pair_metrics["predicted_positive"],
                "actual_positive": pair_metrics["actual_positive"],
            }
        )

    overall_metrics = pd.DataFrame(overall_rows).sort_values("variant").reset_index(drop=True)

    # --------------------------------------------------------
    # 5. SAVE + AUDIT
    # --------------------------------------------------------

    pairs.to_csv(PAIR_FILE, index=False, encoding="utf-8-sig")
    video_metrics.to_csv(VIDEO_FILE, index=False, encoding="utf-8-sig")
    overall_metrics.to_csv(METRICS_FILE, index=False, encoding="utf-8-sig")

    audit = pd.DataFrame(audit_rows)
    audit.to_csv(AUDIT_FILE, index=False, encoding="utf-8-sig")

    print()
    print("=" * 72)
    print("A-F EVALUATION AUDIT (comparability check)")
    print("=" * 72)
    print(audit.to_string(index=False))
    print()
    key_columns = [
        "variant",
        "macro_video_f1",
        "macro_video_precision",
        "macro_video_recall",
        "micro_f1",
        "pr_auc",
        "roc_auc",
        "mAP_video",
        "TP",
        "FP",
        "FN",
        "TN",
    ]

    summary_view = overall_metrics[key_columns].copy()
    float_cols = summary_view.select_dtypes(include="float").columns
    summary_view[float_cols] = summary_view[float_cols].round(4)

    print()
    print("BANG TOM TAT (day du 22 cot nam trong file CSV, xem duong dan ben duoi):")
    print(summary_view.to_string(index=False))
    print()
    print("Pair scores:", PAIR_FILE)
    print("Video metrics:", VIDEO_FILE)
    print("Overall metrics:", METRICS_FILE)

    required_pass = audit[audit["pass"].notna()]["pass"].all()

    if not bool(required_pass):
        raise RuntimeError(
            "Step 16 audit FAILED: mot variant co san khong ra dung "
            f"{STEP13.EXPECTED_NEW_SCORE_ROWS} score rows tren NEW protocol - "
            "khong comparable. Kiem tra lai file build cua variant do."
        )

    print()
    print("STEP 16 PASS -", ", ".join(available_systems), "comparable voi nhau:")
    print("cung 172-chunk NEW grid, cung 39 concepts, cung GT,")
    print("cung aggregation=max / threshold=0.50.")

    if "C" not in available_systems:
        print()
        print("C chua co trong bang (thieu ocr_fullframe_raw.csv).")
        print("Tai tao file raw (chay lai step 04), roi chay step 15, sau do")
        print("chay lai step 16 nay de bo sung C vao bang so sanh.")


if __name__ == "__main__":
    main()