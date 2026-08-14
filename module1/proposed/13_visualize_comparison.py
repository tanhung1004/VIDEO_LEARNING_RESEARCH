from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# 1. PROJECT PATH
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# =========================================================
# 2. INPUT
# =========================================================

INPUT_FILE = (
    PROJECT_ROOT
    / "module1"
    / "results"
    / "comparison"
    / "baseline_vs_ocr.csv"
)


# =========================================================
# 3. OUTPUT
# =========================================================

FIGURE_DIR = (
    PROJECT_ROOT
    / "module1"
    / "results"
    / "comparison"
    / "figures"
)

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# 4. ADD VALUE LABEL
# =========================================================

def add_value_labels(
    ax,
    decimals=4
):

    for container in ax.containers:

        labels = []

        for bar in container:

            height = bar.get_height()

            if decimals == 0:
                labels.append(
                    f"{height:.0f}"
                )

            else:
                labels.append(
                    f"{height:.{decimals}f}"
                )

        ax.bar_label(
            container,
            labels=labels,
            padding=3,
            fontsize=9
        )


# =========================================================
# 5. CHART 1:
# PRECISION / RECALL / F1
# =========================================================

def plot_main_metrics(
    baseline,
    proposed
):

    metrics = [
        "Precision",
        "Recall",
        "F1"
    ]

    baseline_values = [
        baseline[m]
        for m in metrics
    ]

    proposed_values = [
        proposed[m]
        for m in metrics
    ]


    chart_df = pd.DataFrame({

        "Metric":
            metrics,

        "Transcript-only":
            baseline_values,

        "Transcript+OCR":
            proposed_values
    })


    ax = (
        chart_df
        .set_index("Metric")
        .plot(
            kind="bar",
            figsize=(9, 6)
        )
    )


    ax.set_title(
        "Baseline vs Transcript + OCR"
    )

    ax.set_xlabel(
        "Evaluation Metric"
    )

    ax.set_ylabel(
        "Score"
    )

    ax.set_ylim(
        0,
        1.05
    )

    ax.tick_params(
        axis="x",
        rotation=0
    )

    ax.legend(
        title="Method"
    )


    add_value_labels(
        ax,
        decimals=4
    )


    plt.tight_layout()


    output_file = (
        FIGURE_DIR
        / "precision_recall_f1.png"
    )


    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


    print(
        "Created:",
        output_file
    )


# =========================================================
# 6. CHART 2:
# TP / FP / FN
# =========================================================

def plot_prediction_counts(
    baseline,
    proposed
):

    metrics = [
        "TP",
        "FP",
        "FN"
    ]

    baseline_values = [
        baseline[m]
        for m in metrics
    ]

    proposed_values = [
        proposed[m]
        for m in metrics
    ]


    chart_df = pd.DataFrame({

        "Result":
            metrics,

        "Transcript-only":
            baseline_values,

        "Transcript+OCR":
            proposed_values
    })


    ax = (
        chart_df
        .set_index("Result")
        .plot(
            kind="bar",
            figsize=(9, 6)
        )
    )


    ax.set_title(
        "Prediction Error Analysis"
    )

    ax.set_xlabel(
        "Prediction Result"
    )

    ax.set_ylabel(
        "Number of Concepts"
    )

    ax.tick_params(
        axis="x",
        rotation=0
    )

    ax.legend(
        title="Method"
    )


    add_value_labels(
        ax,
        decimals=0
    )


    plt.tight_layout()


    output_file = (
        FIGURE_DIR
        / "tp_fp_fn.png"
    )


    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


    print(
        "Created:",
        output_file
    )


# =========================================================
# 7. CHART 3:
# DIFFERENCE OCR - BASELINE
# =========================================================

def plot_metric_difference(
    baseline,
    proposed
):

    metrics = [
        "Precision",
        "Recall",
        "F1"
    ]


    differences = [

        float(
            proposed[m]
            -
            baseline[m]
        )

        for m in metrics
    ]


    fig, ax = plt.subplots(
        figsize=(9, 6)
    )


    bars = ax.bar(
        metrics,
        differences
    )


    ax.axhline(
        y=0,
        linewidth=1
    )


    ax.set_title(
        "Metric Difference: Transcript+OCR - Baseline"
    )

    ax.set_xlabel(
        "Evaluation Metric"
    )

    ax.set_ylabel(
        "Difference"
    )


    for bar, value in zip(
        bars,
        differences
    ):

        if value >= 0:

            y = value + 0.001

            va = "bottom"

        else:

            y = value - 0.001

            va = "top"


        ax.text(
            bar.get_x()
            + bar.get_width() / 2,

            y,

            f"{value:+.4f}",

            ha="center",

            va=va,

            fontsize=10
        )


    plt.tight_layout()


    output_file = (
        FIGURE_DIR
        / "metrics_difference.png"
    )


    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


    print(
        "Created:",
        output_file
    )


# =========================================================
# 8. MAIN
# =========================================================

def main():

    print(
        "======================================"
    )

    print(
        "MODULE 1"
    )

    print(
        "STEP 13 - VISUALIZE COMPARISON"
    )

    print(
        "======================================"
    )


    # -----------------------------------------------------
    # CHECK INPUT
    # -----------------------------------------------------

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Không tìm thấy:\n{INPUT_FILE}"
        )


    # -----------------------------------------------------
    # READ DATA
    # -----------------------------------------------------

    df = pd.read_csv(
        INPUT_FILE
    )


    print(
        "\nInput:"
    )

    print(
        INPUT_FILE
    )


    print(
        "\nComparison table:"
    )

    print(
        df.to_string(
            index=False
        )
    )


    # -----------------------------------------------------
    # GET BASELINE
    # -----------------------------------------------------

    baseline_rows = df[
        df["Method"]
        .astype(str)
        .str.strip()
        .str.lower()
        == "transcript-only"
    ]


    # -----------------------------------------------------
    # GET PROPOSED
    # -----------------------------------------------------

    proposed_rows = df[
        df["Method"]
        .astype(str)
        .str.strip()
        .str.lower()
        == "transcript+ocr"
    ]


    if baseline_rows.empty:

        raise ValueError(
            "Không tìm thấy row Transcript-only."
        )


    if proposed_rows.empty:

        raise ValueError(
            "Không tìm thấy row Transcript+OCR."
        )


    baseline = (
        baseline_rows.iloc[0]
    )

    proposed = (
        proposed_rows.iloc[0]
    )


    # -----------------------------------------------------
    # NUMERIC COLUMNS
    # -----------------------------------------------------

    numeric_columns = [
        "TP",
        "FP",
        "FN",
        "Precision",
        "Recall",
        "F1"
    ]


    for column in numeric_columns:

        baseline[column] = float(
            baseline[column]
        )

        proposed[column] = float(
            proposed[column]
        )


    # -----------------------------------------------------
    # CHART 1
    # -----------------------------------------------------

    plot_main_metrics(
        baseline,
        proposed
    )


    # -----------------------------------------------------
    # CHART 2
    # -----------------------------------------------------

    plot_prediction_counts(
        baseline,
        proposed
    )


    # -----------------------------------------------------
    # CHART 3
    # -----------------------------------------------------

    plot_metric_difference(
        baseline,
        proposed
    )


    # -----------------------------------------------------
    # DONE
    # -----------------------------------------------------

    print(
        "\n======================================"
    )

    print(
        "HOÀN THÀNH STEP 13"
    )

    print(
        "\nFigures:"
    )

    print(
        FIGURE_DIR
    )

    print(
        "======================================"
    )


if __name__ == "__main__":
    main()