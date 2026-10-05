from pathlib import Path
import importlib.util

import pandas as pd


# ============================================================
# STEP 15 - BUILD VARIANT C (TRANSCRIPT + FULL-FRAME OCR RAW) ON NEW PROTOCOL
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parents[2]

OCR_BEST_DIR = PROJECT_ROOT / "module1" / "ocr_best_integration"

RESULT_ROOT = (
    PROJECT_ROOT / "module1" / "results"
    / "ocr_best_integration" / "combined_system_comparison"
)

C_DIR = RESULT_ROOT / "NEW_C_transcript_plus_raw_fullframe"

AUDIT_FILE = C_DIR / "variant_c_build_audit.csv"

# Same raw file that step 04/05/06 in ocr_best_integration produce and
# audit (2,398 frames, columns include "ocr_text" - BEFORE cleaning).
C_OCR_RAW_FILE = (
    PROJECT_ROOT / "data" / "processed" / "ocr_best_integration"
    / "ocr_fullframe_raw.csv"
)

RAW_TEXT_COLUMN = "ocr_text"


def load_python_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


STEP12 = load_python_module(
    "step12_old_vs_new_combined",
    OCR_BEST_DIR / "12_build_old_vs_new_combined_scores.py",
)


# ============================================================
# RAW OCR LOADER
# Cung audit-style voi STEP12.load_cleaned_ocr(), chi khac cot text
# nguon ("ocr_text" thay vi "ocr_text_clean"/"clean_text"/...).
# Tai su dung STEP12.clean_string_series / STEP12.DEV_IDS /
# STEP12.EXPECTED_OCR_FRAMES / STEP12.video_number de audit dong nhat.
# ============================================================

def load_raw_ocr(path, label):
    if not path.exists():
        raise FileNotFoundError(
            f"{label} missing:\n{path}\n"
            "File raw co the da bi xoa sau buoc cleaning (step 05). "
            "Neu con giu keyframe/anh goc, chay lai step 04 "
            "(04_run_fullframe_ocr.py) de tao lai file nay truoc."
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

    # Rename raw column -> "ocr_text_clean" ONLY so it satisfies the
    # column name build_combined_chunks() already expects. The text
    # itself stays untouched/uncleaned (raw), no cleaning is applied.
    df["ocr_text_clean"] = STEP12.clean_string_series(df[RAW_TEXT_COLUMN])

    duplicates = int(df.duplicated(["video_id", "timestamp_sec"]).sum())
    missing_ids = set(STEP12.DEV_IDS) - set(df["video_id"])
    extra_ids = set(df["video_id"]) - set(STEP12.DEV_IDS)

    print()
    print(f"{label} INPUT AUDIT")
    print("Rows:", len(df))
    print("Videos:", df["video_id"].nunique())
    print("Duplicates:", duplicates)
    print("Missing DEV:", sorted(missing_ids, key=STEP12.video_number))
    print("Extra videos:", sorted(extra_ids, key=STEP12.video_number))

    # NOTE: raw frame count is NOT compared against EXPECTED_OCR_FRAMES
    # (2398) - that constant is the CLEANED frame count. Cleaning drops
    # empty/near-empty OCR frames, so raw legitimately has MORE rows
    # (observed: 2873). Row count is informational only here.
    if (
        df["video_id"].nunique() != 40
        or duplicates != 0
        or missing_ids
        or extra_ids
    ):
        raise RuntimeError(f"{label}: INPUT AUDIT FAILED.")

    print(f"{label} INPUT PASS")

    return df[["video_id", "subject", "timestamp_sec", "ocr_text_clean"]].copy()


def main():
    print("=" * 72)
    print("STEP 15 - BUILD VARIANT C (TRANSCRIPT + FULL-FRAME OCR RAW, NEW PROTOCOL)")
    print("=" * 72)
    print()
    print("Reusing step 12: frozen NEW grid (172 chunks / 180s),")
    print("build_combined_chunks(), score_new_lsa_branch() (LDA=0 / LSA=1).")
    print("OCR source = RAW full-frame text (uncleaned), NOT ocr_fullframe_cleaned.csv.")

    C_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # 1. FROZEN NEW TRANSCRIPT GRID (SAME GRID / SAME TEXT AS D / E / F)
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

    if len(new_base) != STEP12.NEW_EXPECTED_CHUNKS:
        raise RuntimeError(
            "Frozen NEW grid row count changed: "
            f"{len(new_base)} != {STEP12.NEW_EXPECTED_CHUNKS}"
        )

    # --------------------------------------------------------
    # 2. RAW OCR SOURCE (FULL-FRAME, UNCLEANED)
    # --------------------------------------------------------

    c_ocr = load_raw_ocr(
        C_OCR_RAW_FILE,
        "OCR C FULL-FRAME RAW (uncleaned)",
    )

    concepts = STEP12.load_concepts()

    # --------------------------------------------------------
    # 3. MERGE TRANSCRIPT + RAW OCR ON THE FROZEN NEW GRID
    # (identical merge function used for D, just a different OCR source)
    # --------------------------------------------------------

    c_chunks, assigned, unassigned = STEP12.build_combined_chunks(
        frozen_chunks=new_base,
        ocr_frames=c_ocr,
        label="NEW_C_transcript_plus_raw_fullframe",
    )

    if len(c_chunks) != STEP12.NEW_EXPECTED_CHUNKS:
        raise RuntimeError(
            "Variant C combined chunk count changed: "
            f"{len(c_chunks)} != {STEP12.NEW_EXPECTED_CHUNKS}"
        )

    # Sanity: transcript must actually be present (this is C, not B).
    transcript_present = int(
        (c_chunks["transcript_text"].astype(str).str.strip() != "").sum()
    )

    # --------------------------------------------------------
    # 4. SCORE - REUSE THE EXACT SAME NEW-PROTOCOL SCORER AS D / E
    # --------------------------------------------------------

    c_scores = STEP12.score_new_lsa_branch(
        chunks=c_chunks,
        concepts=concepts,
        output_dir=C_DIR,
        label="NEW C (transcript + full-frame OCR raw)",
    )

    # --------------------------------------------------------
    # 5. AUDIT
    # --------------------------------------------------------

    expected_score_rows = STEP12.expected_score_rows(c_chunks, concepts)

    checks = [
        {
            "check": "C_chunk_count",
            "value": len(c_chunks),
            "expected": STEP12.NEW_EXPECTED_CHUNKS,
            "pass": len(c_chunks) == STEP12.NEW_EXPECTED_CHUNKS,
        },
        {
            "check": "C_transcript_present",
            "value": transcript_present,
            "expected": "> 0",
            "pass": transcript_present > 0,
        },
        {
            # Informational only (matches step 12's own treatment of D/E) -
            # see the comment in step 14 for why this is not a hard check.
            "check": "C_ocr_frames_assigned",
            "value": assigned,
            "expected": f"informational (total frames={len(c_ocr)})",
            "pass": True,
        },
        {
            "check": "C_ocr_frames_unassigned",
            "value": unassigned,
            "expected": "informational (compare with D's own count)",
            "pass": True,
        },
        {
            "check": "C_score_rows",
            "value": len(c_scores),
            "expected": expected_score_rows,
            "pass": len(c_scores) == expected_score_rows,
        },
    ]

    audit = pd.DataFrame(checks)
    audit.to_csv(AUDIT_FILE, index=False, encoding="utf-8-sig")

    all_pass = bool(
        audit.loc[audit["check"] != "C_transcript_present", "pass"].all()
        and transcript_present > 0
    )

    print()
    print("=" * 72)
    print("VARIANT C BUILD AUDIT")
    print("=" * 72)
    print(audit.to_string(index=False))
    print()
    print("Output dir:", C_DIR)
    print("Audit file:", AUDIT_FILE)

    if not all_pass:
        raise RuntimeError("Step 15 audit FAILED.")

    print()
    print("STEP 15 PASS - VARIANT C (TRANSCRIPT + RAW OCR, NEW PROTOCOL) COMPLETE")
    print("NEXT: run step 16 to evaluate A-F against ground truth.")


if __name__ == "__main__":
    main()