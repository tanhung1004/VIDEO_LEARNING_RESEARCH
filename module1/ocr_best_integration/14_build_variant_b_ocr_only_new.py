from pathlib import Path
import importlib.util

import pandas as pd


# ============================================================
# STEP 14 - BUILD VARIANT B (OCR-ONLY, FULL-FRAME CLEANED) ON NEW PROTOCOL
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parents[2]

OCR_BEST_DIR = PROJECT_ROOT / "module1" / "ocr_best_integration"

RESULT_ROOT = (
    PROJECT_ROOT / "module1" / "results"
    / "ocr_best_integration" / "combined_system_comparison"
)

B_DIR = RESULT_ROOT / "NEW_B_ocr_only_fullframe"

AUDIT_FILE = B_DIR / "variant_b_build_audit.csv"


# ============================================================
# LOAD STEP 12 AS A MODULE (KHONG CHINH SUA FILE GOC)
# Cung cach importlib.util ma p1_validation/12_select_final_dev_configuration.py
# da dung de tai su dung code giua cac step trong repo nay.
# ============================================================

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


def main():
    print("=" * 72)
    print("STEP 14 - BUILD VARIANT B (OCR-ONLY, FULL-FRAME CLEANED, NEW PROTOCOL)")
    print("=" * 72)
    print()
    print("Reusing step 12: frozen NEW grid (172 chunks / 180s),")
    print("build_combined_chunks(), score_new_lsa_branch() (LDA=0 / LSA=1).")
    print("transcript_base_text is blanked out -> OCR-only document.")

    B_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # 1. FROZEN NEW TRANSCRIPT GRID (SAME GRID AS D / E / F)
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

    # Blank out transcript text -> OCR-only document.
    # Chunk grid (video_id/subject/chunk_id/start_sec/end_sec) is
    # kept identical to D/E/F so B stays on the exact same chunk axis.
    new_base_ocr_only = new_base.copy()
    new_base_ocr_only["transcript_base_text"] = ""

    # --------------------------------------------------------
    # 2. OCR D SOURCE (FULL-FRAME CLEANED) - SAME FILE AS VARIANT D
    # --------------------------------------------------------

    d_ocr = STEP12.load_cleaned_ocr(
        STEP12.D_OCR_FILE,
        "OCR D FULL-FRAME CLEANED (reused for B)",
    )

    concepts = STEP12.load_concepts()

    # --------------------------------------------------------
    # 3. BUILD OCR-ONLY CHUNKS ON THE FROZEN NEW GRID
    # --------------------------------------------------------

    b_chunks, assigned, unassigned = STEP12.build_combined_chunks(
        frozen_chunks=new_base_ocr_only,
        ocr_frames=d_ocr,
        label="NEW_B_ocr_only_fullframe",
    )

    if len(b_chunks) != STEP12.NEW_EXPECTED_CHUNKS:
        raise RuntimeError(
            "Variant B combined chunk count changed: "
            f"{len(b_chunks)} != {STEP12.NEW_EXPECTED_CHUNKS}"
        )

    # Sanity: this must be a true OCR-only document (no transcript leak).
    transcript_leak = int((b_chunks["transcript_text"].astype(str).str.strip() != "").sum())
    if transcript_leak != 0:
        raise RuntimeError(
            f"Variant B leaked transcript text into {transcript_leak} chunks."
        )

    # --------------------------------------------------------
    # 4. SCORE - REUSE THE EXACT SAME NEW-PROTOCOL SCORER AS D / E
    # (180s / LDA=0 / LSA=1). This is the same function call step 12
    # uses to build D and E; nothing here is re-implemented.
    # --------------------------------------------------------

    b_scores = STEP12.score_new_lsa_branch(
        chunks=b_chunks,
        concepts=concepts,
        output_dir=B_DIR,
        label="NEW B (OCR-only, full-frame cleaned)",
    )

    # --------------------------------------------------------
    # 5. AUDIT
    # --------------------------------------------------------

    expected_score_rows = STEP12.expected_score_rows(b_chunks, concepts)

    checks = [
        {
            "check": "B_chunk_count",
            "value": len(b_chunks),
            "expected": STEP12.NEW_EXPECTED_CHUNKS,
            "pass": len(b_chunks) == STEP12.NEW_EXPECTED_CHUNKS,
        },
        {
            "check": "B_no_transcript_leak",
            "value": transcript_leak,
            "expected": 0,
            "pass": transcript_leak == 0,
        },
        {
            # Informational only (matches step 12's own treatment of D/E):
            # a small number of OCR frames can fall outside the 172-chunk
            # frozen grid (video duration not an exact multiple of 180s).
            # step 12 does not fail on this for D/E, so B does not either -
            # this must equal D's own "outside frozen grid" count exactly,
            # since B reuses the identical OCR file + identical grid.
            "check": "B_ocr_frames_assigned",
            "value": assigned,
            "expected": f"informational (total frames={len(d_ocr)})",
            "pass": True,
        },
        {
            "check": "B_ocr_frames_unassigned",
            "value": unassigned,
            "expected": "informational (compare with D's own count)",
            "pass": True,
        },
        {
            "check": "B_score_rows",
            "value": len(b_scores),
            "expected": expected_score_rows,
            "pass": len(b_scores) == expected_score_rows,
        },
    ]

    audit = pd.DataFrame(checks)
    audit.to_csv(AUDIT_FILE, index=False, encoding="utf-8-sig")

    all_pass = bool(audit["pass"].all())

    print()
    print("=" * 72)
    print("VARIANT B BUILD AUDIT")
    print("=" * 72)
    print(audit.to_string(index=False))
    print()
    print("Output dir:", B_DIR)
    print("Audit file:", AUDIT_FILE)

    if not all_pass:
        raise RuntimeError("Step 14 audit FAILED.")

    print()
    print("STEP 14 PASS - VARIANT B (OCR-ONLY, NEW PROTOCOL) COMPLETE")
    print("NEXT: run step 15 (variant C), then step 16 (evaluate B/C vs GT).")


if __name__ == "__main__":
    main()