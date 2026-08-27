from pathlib import Path
import os


# =========================================================
# STEP 00 - SCAFFOLD40 OLD vs NEW CONFIG
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]


VARIANT = (
    os.environ
    .get("SCAFFOLD40_VARIANT", "old")
    .strip()
    .lower()
)


CONFIGS = {

    "old": {
        "chunk_seconds": 60,
        "lda_weight": 0.40,
        "lsa_weight": 0.60,
        "aggregation_column": "top2_mean_score",
        "threshold": 0.40,
    },

    "new": {
        "chunk_seconds": 180,
        "lda_weight": 0.00,
        "lsa_weight": 1.00,
        "aggregation_column": "max_final_score",
        "threshold": 0.50,
    },

}


if VARIANT not in CONFIGS:

    raise ValueError(
        f"Invalid SCAFFOLD40_VARIANT='{VARIANT}'. "
        "Use old or new."
    )


CONFIG = CONFIGS[VARIANT]

CHUNK_SECONDS = CONFIG["chunk_seconds"]
LDA_WEIGHT = CONFIG["lda_weight"]
LSA_WEIGHT = CONFIG["lsa_weight"]

AGGREGATION_COLUMN = (
    CONFIG["aggregation_column"]
)

PREDICTION_THRESHOLD = (
    CONFIG["threshold"]
)


# =========================================================
# OUTPUT ROOT
# =========================================================

RUN_ROOT = (
    PROJECT_ROOT
    / "module1"
    / "results"
    / "scaffold40_compare"
    / VARIANT
)

RUN_ROOT.mkdir(
    parents=True,
    exist_ok=True
)


CHUNK_FILE = (
    RUN_ROOT
    / "transcript_chunks.csv"
)


def print_config():

    print("======================================")
    print("SCAFFOLD40 CONFIG")
    print("======================================")

    print("Variant:", VARIANT)
    print("Chunk seconds:", CHUNK_SECONDS)
    print("LDA weight:", LDA_WEIGHT)
    print("LSA weight:", LSA_WEIGHT)
    print("Aggregation:", AGGREGATION_COLUMN)
    print("Threshold:", PREDICTION_THRESHOLD)
    print("Run root:", RUN_ROOT)


if __name__ == "__main__":

    print_config()