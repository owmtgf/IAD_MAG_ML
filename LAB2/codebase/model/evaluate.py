from pathlib import Path
from argparse import ArgumentParser

import pandas as pd
from seqeval.metrics import precision_score, recall_score, f1_score, accuracy_score

from ..globals import ID2LABEL


def compute_metrics(eval_pred: tuple) -> dict:

    predictions, labels = eval_pred

    predictions = predictions.argmax(axis=2)

    true_predictions = []
    true_labels = []

    for pred, lab in zip(predictions, labels):

        current_preds = []
        current_labels = []

        for p, l in zip(pred, lab):

            if l != -100:
                current_preds.append(ID2LABEL[p])
                current_labels.append(ID2LABEL[l])

        true_predictions.append(current_preds)
        true_labels.append(current_labels)

    return {
        "precision": precision_score(true_labels, true_predictions),
        "recall": recall_score(true_labels, true_predictions),
        "f1": f1_score(true_labels, true_predictions),
        "accuracy": accuracy_score(true_labels, true_predictions),
    }


def run_evaluation(input_csv: Path, output_csv: Path, metrics_csv: Path | None = None):

    if not input_csv.exists():
        raise FileNotFoundError(f"Ground truth file not found: {input_csv}")

    if not output_csv.exists():
        raise FileNotFoundError(f"Prediction file not found: {output_csv}")

    print("Loading ground truth...")
    gt_df = pd.read_csv(input_csv)

    print("Loading predictions...")
    pred_df = pd.read_csv(output_csv)

    if len(gt_df) != len(pred_df):
        raise ValueError(
            f"Row mismatch: ground truth {len(gt_df)} vs predictions {len(pred_df)}"
        )

    true_tags = gt_df["Tag"].tolist()
    pred_tags = pred_df["Tag"].tolist()

    true_labels = [[ID2LABEL[t]] for t in true_tags]
    pred_labels = [[ID2LABEL[p]] for p in pred_tags]

    metrics = {
        "precision": precision_score(true_labels, pred_labels),
        "recall": recall_score(true_labels, pred_labels),
        "f1": f1_score(true_labels, pred_labels),
        "accuracy": accuracy_score(true_labels, pred_labels),
    }

    print("\nEvaluation Results")
    print("-------------------")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    if metrics_csv:
        metrics_df = pd.DataFrame([metrics])

        metrics_csv.parent.mkdir(parents=True, exist_ok=True)
        metrics_df.to_csv(metrics_csv, index=False)

        print(f"\nMetrics saved to: {metrics_csv}")

    return metrics


if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument(
        "-i", "--input_csv",
        type=Path,
        default=Path("./dataset/test.csv"),
        help="Ground truth dataset"
    )

    parser.add_argument(
        "-o", "--output_csv",
        type=Path,
        default=Path("./outputs/submission.csv"),
        help="Model predictions"
    )

    parser.add_argument(
        "-m", "--metrics_csv",
        type=Path,
        default=Path("./outputs/metrics.csv"),
        help="Where to save evaluation metrics"
    )

    args = parser.parse_args()

    print(f"Start running {__file__}. Running arguments:\n{args}")

    run_evaluation(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        metrics_csv=args.metrics_csv
    )
