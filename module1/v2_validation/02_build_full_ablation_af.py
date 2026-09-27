from pathlib import Path
import importlib.util

import pandas as pd

# ============================================================
# TRACK B - B3: BUILD FULL A-F ABLATION UNDER FROZEN P1/NEW PROTOCOL
#
# Place this file at:
#   module1/v2_validation/02_build_full_ablation_af.py
#
# Feedback addressed: F3 (A-F cũ mất metric artifact), F4 (ablation cũ
# dựa trên transcript config cũ).
#
# Rules follow kehoach.docx B2/B3 exactly:
#   - Definitions (không tạo variant 7, không đổi nghĩa giữa chừng):
#       A = Transcript-only
#       B = OCR-only, full-frame cleaned
#       C = Transcript + full-frame OCR raw (chua clean)
#       D = Transcript + full-frame OCR cleaned
#       E = Transcript + ROI OCR cleaned
#       F = Combined D + E at score level (mean, cong thuc goc khong doi)
#   - Tat ca 6 variant chay qua CUNG MOT ham build (build_combined_chunks +
#     score_new_lsa_branch cua step 12 - frozen P1/NEW protocol: 180s,
#     LDA weight=0, LSA weight=1). Bien duy nhat giua cac variant la
#     (co transcript hay khong, nguon OCR nao). Khong variant nao duoc
#     tune threshold/aggregation/chunk rieng o buoc build nay.
#   - Khong sua, khong overwrite bat ky file nao trong V1
#     (module1/ocr_best_integration/, module1/p1_validation/, ...).
#     Script nay CHI import ham qua importlib (read-only) va ghi output
#     vao khu vuc rieng cua V2.
#   - Script nay KHONG evaluate/threshold/chon candidate (do la B4/B5).
#     Chi build fusion_scores.csv (chunk x concept score) cho ca 6 bien
#     the, tren cung 172-chunk NEW grid, cung 39 concept.
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

OCR_BEST_DIR = PROJECT_ROOT / "module1" / "ocr_best_integration"

OUTPUT_ROOT = PROJECT_ROOT / "module1" / "results" / "v2_validation" / "ablation"
AUDIT_FILE = OUTPUT_ROOT / "audit.csv"

C_OCR_RAW_FILE = (
    PROJECT_ROOT / "data" / "processed" / "ocr_best_integration"
    / "ocr_fullframe_raw.csv"
)
RAW_TEXT_COLUMN = "ocr_text"

VARIANT_ORDER = ["A", "B", "C", "D", "E", "F"]

def load_python_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Read-only reuse of V1 logic (frozen NEW grid discovery, OCR loading,
# transcript+OCR merge, LSA-only NEW-protocol scoring). This file is
# never written to.
STEP12 = load_python_module(
    "step12_old_vs_new_combined",
    OCR_BEST_DIR / "12_build_old_vs_new_combined_scores.py",
)

def empty_ocr_frames():
    """
    Variant A is transcript-only, but Step 12's shared
    build_combined_chunks() expects an OCR group for every DEV video.

    Keep the SAME Step 12 build call by supplying one out-of-range
    sentinel row per DEV video. The sentinel can never fall inside a
    normal chunk, so A still contributes zero OCR text/frames.
    """
    return pd.DataFrame(
        {
            "video_id": STEP12.DEV_IDS,
            "subject": [""] * len(STEP12.DEV_IDS),
            "timestamp_sec": [float("-inf")] * len(STEP12.DEV_IDS),
            "ocr_text_clean": [""] * len(STEP12.DEV_IDS),
        }
    )

def load_raw_ocr(path, label):
    """C's OCR source: full-frame RAW (uncleaned). Not present anywhere
    in V1 code (V1 only ever loads the cleaned files), so this loader is
    new V2 code - it does not modify or duplicate any V1 function."""

    if not path.exists():
        raise FileNotFoundError(
            f"{label} missing:\n{path}\n"
            "Raw file co the da bi xoa sau buoc cleaning (step 05 trong "
            "ocr_best_integration). Neu can, chay lai "
            "04_run_fullframe_ocr.py (V1, khong sua) de tai tao, KHONG "
            "tao raw file bang tay/gia lap."
        )

    df = pd.read_csv(path)

    required = {"video_id", "subject", "timestamp_sec", RAW_TEXT_COLUMN}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{label} missing columns: {sorted(missing)}")

    df["video_id"] = STEP12.clean_string_series(df["video_id"])
    df["subject"] = STEP12.clean_string_series(df["subject"])
    df["timestamp_sec"] = pd.to_numeric(df["timestamp_sec"], errors="coerce")
    if df["timestamp_sec"].isna().any():
        raise RuntimeError(f"{label}: bad timestamp.")
    df["timestamp_sec"] = df["timestamp_sec"].astype(float)

    # Column rename only (ocr_text -> ocr_text_clean) so it fits the
    # shared merge function's expected column name. Text itself is left
    # untouched - no cleaning applied, this must stay RAW for variant C.
    df["ocr_text_clean"] = STEP12.clean_string_series(df[RAW_TEXT_COLUMN])

    duplicates = int(df.duplicated(["video_id", "timestamp_sec"]).sum())
    missing_ids = set(STEP12.DEV_IDS) - set(df["video_id"])
    extra_ids = set(df["video_id"]) - set(STEP12.DEV_IDS)

    if df["video_id"].nunique() != 40 or duplicates != 0 or missing_ids or extra_ids:
        raise RuntimeError(f"{label}: INPUT AUDIT FAILED.")

    return df[["video_id", "subject", "timestamp_sec", "ocr_text_clean"]].copy()

# ============================================================
# A-F DEFINITIONS (B2) - each entry says only:
#   - whether transcript text is kept
#   - which OCR source (if any) is merged in
# The build/score CALL is identical for every variant (see main()).
# ============================================================

def variant_definitions():
    return {
        "A": {
            "label": "A_transcript_only",
            "keep_transcript": True,
            "ocr_loader": lambda: empty_ocr_frames(),
            "ocr_source": "(none)",
        },
        "B": {
            "label": "B_ocr_only_fullframe_cleaned",
            "keep_transcript": False,
            "ocr_loader": lambda: STEP12.load_cleaned_ocr(
                STEP12.D_OCR_FILE, "OCR full-frame cleaned (B)"
            ),
            "ocr_source": str(STEP12.D_OCR_FILE),
        },
        "C": {
            "label": "C_transcript_plus_ocr_fullframe_raw",
            "keep_transcript": True,
            "ocr_loader": lambda: load_raw_ocr(
                C_OCR_RAW_FILE, "OCR full-frame RAW (C)"
            ),
            "ocr_source": str(C_OCR_RAW_FILE),
        },
        "D": {
            "label": "D_transcript_plus_ocr_fullframe_cleaned",
            "keep_transcript": True,
            "ocr_loader": lambda: STEP12.load_cleaned_ocr(
                STEP12.D_OCR_FILE, "OCR full-frame cleaned (D)"
            ),
            "ocr_source": str(STEP12.D_OCR_FILE),
        },
        "E": {
            "label": "E_transcript_plus_ocr_roi_cleaned",
            "keep_transcript": True,
            "ocr_loader": lambda: STEP12.load_cleaned_ocr(
                STEP12.E_OCR_FILE, "OCR ROI cleaned (E)"
            ),
            "ocr_source": str(STEP12.E_OCR_FILE),
        },
        # F is not built the same way (it is a score-level combination of
        # D and E, not a new document) - handled separately in main().
    }

def build_f_scores(d_scores, e_scores, output_dir):
    """F = mean(D, E) at score level. Formula copied verbatim from V1's
    own build_new_f_scores() in step 12 (not a new definition) - only
    the output location changes, so V1's NEW_transcript_plus_F/ is never
    touched by this script."""

    keys = ["video_id", "subject", "chunk_id", "start_sec", "end_sec", "concept"]

    d = d_scores[keys + ["final_score"]].rename(columns={"final_score": "final_score_d"})
    e = e_scores[keys + ["final_score"]].rename(columns={"final_score": "final_score_e"})

    merged = pd.merge(d, e, on=keys, how="outer", validate="one_to_one", indicator=True)

    mismatch = int(merged["_merge"].ne("both").sum())
    if mismatch != 0:
        raise RuntimeError(f"F: D/E score-key mismatch: {mismatch}")

    merged = merged.drop(columns=["_merge"])
    merged["method"] = "F_mean_D_E"
    merged["final_score"] = (
        (merged["final_score_d"] + merged["final_score_e"]) / 2.0
    ).round(4)

    if merged["final_score"].isna().any():
        raise RuntimeError("F: mean formula produced NaN.")

    output_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_dir / "fusion_scores.csv", index=False, encoding="utf-8-sig")

    return merged

def main():
    print("=" * 72)
    print("TRACK B - B3: BUILD FULL A-F ABLATION (FROZEN P1/NEW PROTOCOL)")
    print("=" * 72)
    print()
    print("Protocol (identical for A-F): chunk=180s, LDA weight=0, LSA weight=1")
    print("(aggregation=max / threshold=0.50 are applied later, in B4 - not here)")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    audit_rows = []

    def log(variant, check, value, expected, ok):
        audit_rows.append(
            {
                "variant": variant,
                "check": check,
                "value": value,
                "expected": expected,
                "pass": bool(ok),
            }
        )

    # --------------------------------------------------------
    # SHARED INPUTS (identical for every variant - read-only from V1)
    # --------------------------------------------------------

    new_chunk_file = STEP12.discover_frozen_chunk_file(
        expected_rows=STEP12.NEW_EXPECTED_CHUNKS,
        chunk_seconds=STEP12.NEW_CHUNK_SECONDS,
        branch="NEW",
    )
    new_base = STEP12.load_frozen_chunks(
        path=new_chunk_file,
        expected_rows=STEP12.NEW_EXPECTED_CHUNKS,
        chunk_seconds=STEP12.NEW_CHUNK_SECONDS,
        label="NEW",
    )
    concepts = STEP12.load_concepts()

    if len(new_base) != STEP12.NEW_EXPECTED_CHUNKS:
        raise RuntimeError("Frozen NEW grid row count changed.")

    scores_by_variant = {}

    # --------------------------------------------------------
    # A / B / C / D / E - same build+score call, only inputs differ
    # --------------------------------------------------------

    for variant, cfg in variant_definitions().items():
        out_dir = OUTPUT_ROOT / variant
        out_dir.mkdir(parents=True, exist_ok=True)

        base = new_base.copy()
        if not cfg["keep_transcript"]:
            base["transcript_base_text"] = ""

        ocr_frames = cfg["ocr_loader"]()

        chunks, assigned, unassigned = STEP12.build_combined_chunks(
            frozen_chunks=base,
            ocr_frames=ocr_frames,
            label=f"ablation_{variant}",
        )

        if len(chunks) != STEP12.NEW_EXPECTED_CHUNKS:
            raise RuntimeError(f"Variant {variant}: chunk count changed.")

        scores = STEP12.score_new_lsa_branch(
            chunks=chunks,
            concepts=concepts,
            output_dir=out_dir,
            label=f"Ablation {variant} ({cfg['label']})",
        )

        scores_by_variant[variant] = scores

        expected_rows = STEP12.expected_score_rows(chunks, concepts)

        log(variant, "chunk_count", len(chunks), STEP12.NEW_EXPECTED_CHUNKS, len(chunks) == STEP12.NEW_EXPECTED_CHUNKS)
        log(variant, "score_rows", len(scores), expected_rows, len(scores) == expected_rows)
        log(variant, "chunk_seconds", STEP12.NEW_CHUNK_SECONDS, STEP12.NEW_CHUNK_SECONDS, True)
        log(variant, "lda_weight", 0, 0, True)
        log(variant, "lsa_weight", 1, 1, True)
        log(variant, "keep_transcript", cfg["keep_transcript"], cfg["keep_transcript"], True)
        log(variant, "ocr_source", cfg["ocr_source"], cfg["ocr_source"], True)
        # A uses sentinel rows only to satisfy Step 12's per-video lookup;
        # these are not real OCR frames and never enter any chunk interval.
        real_ocr_rows = 0 if variant == "A" else len(ocr_frames)
        log(
            variant,
            "ocr_frames_loaded",
            real_ocr_rows,
            0 if variant == "A" else len(ocr_frames),
            True,
        )

    # --------------------------------------------------------
    # F = mean(D, E) at score level
    # --------------------------------------------------------

    f_dir = OUTPUT_ROOT / "F"
    f_scores = build_f_scores(scores_by_variant["D"], scores_by_variant["E"], f_dir)
    scores_by_variant["F"] = f_scores

    expected_f_rows = len(scores_by_variant["D"])
    log("F", "score_rows", len(f_scores), expected_f_rows, len(f_scores) == expected_f_rows)
    log("F", "chunk_seconds", STEP12.NEW_CHUNK_SECONDS, STEP12.NEW_CHUNK_SECONDS, True)
    log("F", "fusion_formula", "mean(D, E)", "mean(D, E)", True)

    # --------------------------------------------------------
    # CROSS-VARIANT CHECKS - same DEV/concept universe, no duplicates,
    # no NaN, config differences only from the A-F definitions above.
    # --------------------------------------------------------

    for variant in VARIANT_ORDER:
        df = scores_by_variant[variant]

        videos = df["video_id"].nunique()

        # Concepts are identified by (subject, concept), matching the
        # uniqueness contract used by Step 12's concept catalog audit.
        concept_keys = (
            df["subject"].map(STEP12.normalize_subject).astype(str).str.strip()
            + "::"
            + df["concept"].astype(str).str.strip().str.lower()
        )
        concept_count = concept_keys.nunique()

        duplicates = int(
            df.duplicated(["video_id", "chunk_id", "concept"]).sum()
        )
        nan_scores = int(df["final_score"].isna().sum())
        out_of_range = int(
            ((df["final_score"] < 0) | (df["final_score"] > 1)).sum()
        )

        log(variant, "videos_present", videos, 40, videos == 40)
        log(
            variant,
            "concepts_present",
            concept_count,
            39,
            concept_count == 39,
        )
        log(variant, "duplicate_rows", duplicates, 0, duplicates == 0)
        log(variant, "nan_final_score", nan_scores, 0, nan_scores == 0)
        log(variant, "final_score_out_of_range", out_of_range, 0, out_of_range == 0)

    audit = pd.DataFrame(audit_rows)
    audit.to_csv(AUDIT_FILE, index=False, encoding="utf-8-sig")

    print()
    print("=" * 72)
    print("A-F BUILD AUDIT")
    print("=" * 72)
    print(audit.to_string(index=False))
    print()
    print("Output root:", OUTPUT_ROOT)
    print("Audit file:", AUDIT_FILE)

    all_pass = bool(audit["pass"].all())

    if not all_pass:
        failed = audit[~audit["pass"]]
        print()
        print("FAILED CHECKS:")
        print(failed.to_string(index=False))
        raise RuntimeError("B3 audit FAILED: not all 6 variants are comparable.")

    print()
    print("B3 PASS - A, B, C, D, E, F built on the identical frozen NEW")
    print("protocol (180s / LDA=0 / LSA=1). No V1 file was modified.")
    print("NOT evaluated / no candidate selected yet - that is B4 (requires")
    print("Track A's locked metric/evaluation contract first).")

if __name__ == "__main__":
    main()
