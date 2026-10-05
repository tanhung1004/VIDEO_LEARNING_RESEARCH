from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

import numpy as np
import pandas as pd


# ============================================================
# TRACK B - B5: LOCK V2 CANDIDATE
#
# Place this file at:
#   module1/v2_validation/04_lock_v2_candidate.py
#
# Runs AFTER B4 (03_evaluate_ablation_af_dev.py) has produced
# A_F_DEV_COMPARISON.csv. This script does not recompute metrics from
# scratch - it reads B4's output and B3's build audit, and INDEPENDENTLY
# re-applies the same locked selection rule (V2_STATISTICAL_PROTOCOL.md
# section 18) as a cross-check before freezing. If the re-check disagrees
# with what B4 marked as selected, this script FAILS instead of locking
# a possibly-wrong candidate.
#
# further_dev_tuning_allowed is set to false in the output. From this
# point on, no variant-specific threshold/aggregation/representation
# retuning is permitted using this DEV set.
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parents[2]

V2_DIR = PROJECT_ROOT / "module1" / "results" / "v2_validation"
COMPARISON_FILE = V2_DIR / "A_F_DEV_COMPARISON.csv"
ABLATION_AUDIT_FILE = V2_DIR / "ablation" / "audit.csv"

CONFIG_FILE = V2_DIR / "V2_FINAL_CONFIG.json"
SUMMARY_FILE = V2_DIR / "V2_FINAL_LOCK_SUMMARY.txt"

VARIANT_ORDER = ["A", "B", "C", "D", "E", "F"]

VARIANT_DEFINITION = {
    "A": "Transcript-only",
    "B": "OCR-only, full-frame cleaned",
    "C": "Transcript + full-frame OCR raw",
    "D": "Transcript + full-frame OCR cleaned",
    "E": "Transcript + ROI OCR cleaned",
    "F": "Combined D + E at score level (mean)",
}

# Locked P1/NEW protocol values (V2_STATISTICAL_PROTOCOL.md section 3),
# reused here only to record them in the frozen config - not re-derived.
TRANSCRIPT_CONFIG = {
    "chunk_seconds": 180,
    "lda_weight": 0,
    "lsa_weight": 1,
}
AGGREGATION = "max"
THRESHOLD = 0.50

SELECTION_RULE_TEXT = (
    "V2_STATISTICAL_PROTOCOL.md section 18: "
    "1) highest macro_video_f1, "
    "2) highest micro_f1, "
    "3) lowest macro_video_f1_std, "
    "4) deterministic order A -> B -> C -> D -> E -> F"
)


def apply_selection_rule(comparison):
    """Identical rule to the one used in B4 - reproduced here, not
    reinvented, purely as an independent cross-check before locking."""

    variant_rank = {v: i for i, v in enumerate(VARIANT_ORDER)}

    ranked = comparison.copy()
    ranked["_rank_key"] = list(
        zip(
            -ranked["macro_video_f1"],
            -ranked["micro_f1"],
            ranked["macro_video_f1_std"],
            ranked["variant"].map(variant_rank),
        )
    )
    ranked = ranked.sort_values("_rank_key").reset_index(drop=True)
    return ranked.iloc[0]["variant"]


def get_build_config(ablation_audit, variant):
    def value_of(check):
        rows = ablation_audit[
            (ablation_audit["variant"] == variant) & (ablation_audit["check"] == check)
        ]
        if rows.empty:
            return None
        return rows.iloc[0]["value"]

    keep_transcript_raw = value_of("keep_transcript")
    ocr_source_raw = value_of("ocr_source")

    if keep_transcript_raw is None or ocr_source_raw is None:
        raise RuntimeError(
            f"Variant {variant}: could not find 'keep_transcript' / 'ocr_source' "
            f"rows in {ABLATION_AUDIT_FILE}. Re-run B3 (02_build_full_ablation_af.py) "
            "or check that its audit.csv schema has not changed."
        )

    keep_transcript = str(keep_transcript_raw).strip().lower() == "true"

    return {
        "keep_transcript": keep_transcript,
        "ocr_source": None if str(ocr_source_raw).strip() == "(none)" else str(ocr_source_raw),
    }


def main():
    print("=" * 72)
    print("TRACK B - B5: LOCK V2 CANDIDATE")
    print("=" * 72)

    if not COMPARISON_FILE.exists():
        raise FileNotFoundError(
            f"{COMPARISON_FILE} missing.\n"
            "Chay B4 (03_evaluate_ablation_af_dev.py) truoc."
        )
    if not ABLATION_AUDIT_FILE.exists():
        raise FileNotFoundError(
            f"{ABLATION_AUDIT_FILE} missing.\n"
            "Chay B3 (02_build_full_ablation_af.py) truoc."
        )

    comparison = pd.read_csv(COMPARISON_FILE)
    ablation_audit = pd.read_csv(ABLATION_AUDIT_FILE)

    required_columns = {
        "variant", "macro_video_f1", "macro_video_f1_std", "micro_f1",
        "macro_video_precision", "macro_video_recall", "n_videos", "n_pairs",
        "selected", "selection_rank",
    }
    missing = required_columns - set(comparison.columns)
    if missing:
        raise RuntimeError(f"A_F_DEV_COMPARISON.csv missing columns: {sorted(missing)}")

    if set(comparison["variant"]) != set(VARIANT_ORDER) or len(comparison) != 6:
        raise RuntimeError(
            "A_F_DEV_COMPARISON.csv does not contain exactly the 6 locked "
            f"A-F variants ({VARIANT_ORDER})."
        )

    marked_rows = comparison[comparison["selected"] == True]  # noqa: E712
    if len(marked_rows) != 1:
        raise RuntimeError(
            f"Expected exactly 1 variant marked selected=True in "
            f"A_F_DEV_COMPARISON.csv, found {len(marked_rows)}."
        )
    marked_winner = marked_rows.iloc[0]["variant"]

    # --------------------------------------------------------
    # INDEPENDENT RE-CHECK before locking anything
    # --------------------------------------------------------

    recomputed_winner = apply_selection_rule(comparison)

    if marked_winner != recomputed_winner:
        raise RuntimeError(
            f"SELECTION MISMATCH: B4 marked variant '{marked_winner}' as selected, "
            f"but independently re-applying the locked rule from "
            f"V2_STATISTICAL_PROTOCOL.md section 18 selects '{recomputed_winner}'. "
            "Refusing to lock. Re-run B4 or inspect A_F_DEV_COMPARISON.csv."
        )

    winner = marked_winner
    print(f"Independent re-check PASS: selection rule also selects '{winner}'.")

    winner_row = comparison[comparison["variant"] == winner].iloc[0]
    build_config = get_build_config(ablation_audit, winner)

    # --------------------------------------------------------
    # BUILD FROZEN CONFIG
    # --------------------------------------------------------

    config = {
        "v2_final_config_version": 1,
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "selected_variant": winner,
        "selected_variant_definition": VARIANT_DEFINITION[winner],
        "transcript_config": TRANSCRIPT_CONFIG,
        "ocr_config": build_config,
        "aggregation": AGGREGATION,
        "threshold": THRESHOLD,
        "dev_metrics": {
            "macro_video_f1": float(winner_row["macro_video_f1"]),
            "macro_video_f1_std": float(winner_row["macro_video_f1_std"]),
            "macro_video_precision": float(winner_row["macro_video_precision"]),
            "macro_video_recall": float(winner_row["macro_video_recall"]),
            "micro_f1": float(winner_row["micro_f1"]),
            "n_videos": int(winner_row["n_videos"]),
            "n_pairs": int(winner_row["n_pairs"]),
        },
        "selection_rule": SELECTION_RULE_TEXT,
        "all_variant_macro_video_f1": {
            row.variant: float(row.macro_video_f1) for row in comparison.itertuples()
        },
        "source_files": {
            "a_f_dev_comparison": str(COMPARISON_FILE),
            "ablation_build_audit": str(ABLATION_AUDIT_FILE),
        },
        "evaluation_status": "DEV selection only; not an independent confirmatory result.",
        "further_dev_tuning_allowed": False,
    }

    fingerprint_source = json.dumps(
        {k: v for k, v in config.items() if k != "locked_at_utc"},
        sort_keys=True,
    )
    config["config_fingerprint_sha256"] = hashlib.sha256(
        fingerprint_source.encode("utf-8")
    ).hexdigest()

    V2_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

    # --------------------------------------------------------
    # SUMMARY.txt
    # --------------------------------------------------------

    lines = [
        "V2 FINAL CONFIG - LOCK SUMMARY",
        "=" * 72,
        "",
        f"Selected variant: {winner} ({VARIANT_DEFINITION[winner]})",
        f"Selection rule:   {SELECTION_RULE_TEXT}",
        "",
        "Transcript config:",
        f"  chunk_seconds = {TRANSCRIPT_CONFIG['chunk_seconds']}",
        f"  lda_weight    = {TRANSCRIPT_CONFIG['lda_weight']}",
        f"  lsa_weight    = {TRANSCRIPT_CONFIG['lsa_weight']}",
        "",
        "OCR config:",
        f"  keep_transcript = {build_config['keep_transcript']}",
        f"  ocr_source      = {build_config['ocr_source']}",
        "",
        f"Aggregation: {AGGREGATION}",
        f"Threshold:   {THRESHOLD:.2f}",
        "",
        "DEV metrics (selected variant):",
        f"  macro_video_f1     = {winner_row['macro_video_f1']:.6f}",
        f"  macro_video_f1_std = {winner_row['macro_video_f1_std']:.6f}",
        f"  macro_video_precision = {winner_row['macro_video_precision']:.6f}",
        f"  macro_video_recall    = {winner_row['macro_video_recall']:.6f}",
        f"  micro_f1            = {winner_row['micro_f1']:.6f}",
        f"  n_videos / n_pairs  = {int(winner_row['n_videos'])} / {int(winner_row['n_pairs'])}",
        "",
        f"Config fingerprint (SHA-256): {config['config_fingerprint_sha256']}",
        "",
        f"Variant {winner} was selected on DEV under the frozen development "
        "protocol. This is a development-stage selection, not an independent "
        "confirmatory result.",
        "",
        "further_dev_tuning_allowed = false",
        "No further threshold, aggregation, chunk, or OCR-representation "
        "tuning is permitted against this DEV set from this point forward. "
        "An independent confirmatory evaluation (if pursued) must use a "
        "separate dataset and this exact frozen configuration, unmodified.",
        "",
        f"Saved: {CONFIG_FILE}",
    ]

    SUMMARY_FILE.write_text("\n".join(lines), encoding="utf-8")

    print()
    print("\n".join(lines))
    print()
    print("Saved:", CONFIG_FILE)
    print("Saved:", SUMMARY_FILE)
    print()
    print("B5 DONE. V2 candidate locked.")


if __name__ == "__main__":
    main()
